"""MCP connector (nx/_mcp.py): config validation, the stdio JSON-RPC client
against a fake in-repo MCP server subprocess, the Streamable-HTTP client
against a fake localhost HTTP server, gateway behavior, and gates."""
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from nx import _mcp
import nx._state as _state


# --- fakes --------------------------------------------------------------

FAKE_SERVER = r'''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg["method"]
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18",
                  "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": "echo", "description": "Echo the text back",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}},
                             "required": ["text"]}},
            {"name": "boom", "description": "Always fails",
             "inputSchema": {"type": "object", "properties": {}}}]}
    elif method == "tools/call":
        name = msg["params"]["name"]
        if name == "boom":
            result = {"content": [{"type": "text", "text": "kaboom"}],
                      "isError": True}
        else:
            text = msg["params"]["arguments"]["text"]
            result = {"content": [{"type": "text", "text": f"echo: {text}"}],
                      "isError": False}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                 "result": result}) + "\n")
    sys.stdout.flush()
'''

HANG_SERVER = r'''
import json, sys, time
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "initialize":
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                     "result": {"protocolVersion": "2025-06-18"}}) + "\n")
        sys.stdout.flush()
    # tools/list never answers -> client should time out
'''


@pytest.fixture()
def fake_server_path(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER)
    return path


@pytest.fixture()
def mcp_config(tmp_path, monkeypatch, fake_server_path):
    """NX_MCP=1 + a project .nx/mcp.json pointing at the fake server."""
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setenv("NX_MCP_TIMEOUT", "10")
    monkeypatch.setenv("NX_MCP_TOOL_TIMEOUT", "10")
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
    (tmp_path / ".nx").mkdir()
    (tmp_path / ".nx" / "mcp.json").write_text(json.dumps({
        "mcpServers": {"fake": {"command": sys.executable,
                                "args": [str(fake_server_path)]}}}))
    yield tmp_path
    _mcp.shutdown_sessions()


# --- config validation (no subprocess launched) -------------------------

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NX_MCP", raising=False)
    servers, problem = _mcp._servers()
    assert servers == {} and problem and "NX_MCP=1" in problem


def test_missing_config_reports_howto(monkeypatch, tmp_path):
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))  # empty user store
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
    monkeypatch.chdir(tmp_path)  # the cwd fallback must not see the repo config
    servers, problem = _mcp._servers()
    assert servers == {} and "mcp.json" in problem and "mcpServers" in problem


def test_invalid_config_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
    (tmp_path / ".nx").mkdir()
    cases = [
        ("[]", "root must be an object"),
        ('{"mcpServers": {"bad name!": {"command": "x"}}}', "invalid server name"),
        ('{"mcpServers": {"s": "x"}}', "must be an object"),
        ('{"mcpServers": {"s": {"args": [1]}}}', "command must be a non-empty string"),
        ('{"mcpServers": {"s": {"command": "x", "args": "y"}}}', "list of strings"),
        ('{"mcpServers": {"s": {"command": "x", "env": {"A": 1}}}}', "object of strings"),
    ]
    for raw, needle in cases:
        (tmp_path / ".nx" / "mcp.json").write_text(raw)
        servers, problem = _mcp._servers()
        assert problem and needle in problem, raw


def test_top_level_server_map_accepted(tmp_path, monkeypatch, fake_server_path):
    """Claude-Code-style bare {name: spec} (no mcpServers wrapper) also works."""
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
    (tmp_path / ".nx").mkdir()
    (tmp_path / ".nx" / "mcp.json").write_text(json.dumps(
        {"fake": {"command": sys.executable, "args": [str(fake_server_path)]}}))
    servers, problem = _mcp._servers()
    assert problem is None and "fake" in servers


# --- live client against the fake stdio server --------------------------

def test_search_and_call_roundtrip(mcp_config):
    listing = _mcp.search_mcp_tools("")
    assert "echo" in listing and "inputSchema" in listing
    filtered = _mcp.search_mcp_tools("boom")
    assert "boom" in filtered and "echo" not in filtered
    result = _mcp.call_mcp_tool("fake", "echo", {"text": "hello"})
    assert result == "echo: hello"


def test_tool_error_is_error_string(mcp_config):
    result = _mcp.call_mcp_tool("fake", "boom", {})
    assert result.startswith("ERROR: mcp: fake.boom:") and "kaboom" in result


def test_unknown_server_and_bad_names(mcp_config):
    assert "unknown server" in _mcp.call_mcp_tool("nope", "echo", {})
    assert "invalid server name" in _mcp.call_mcp_tool("a b", "echo", {})
    assert "invalid tool name" in _mcp.call_mcp_tool("fake", "a b", {})
    assert "must be an object" in _mcp.call_mcp_tool("fake", "echo", "x")


def test_initialize_negotiation(mcp_config):
    session = _mcp._get_session("fake", _mcp._servers()[0]["fake"])
    assert session.negotiated_version == "2025-06-18"
    assert session.server_info.get("name") == "fake"


def test_timeout_fails_soft(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setenv("NX_MCP_TIMEOUT", "10")
    monkeypatch.setenv("NX_MCP_TOOL_TIMEOUT", "1")
    hang = tmp_path / "hang_mcp_server.py"
    hang.write_text(HANG_SERVER)
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
    (tmp_path / ".nx").mkdir()
    (tmp_path / ".nx" / "mcp.json").write_text(json.dumps(
        {"mcpServers": {"hang": {"command": sys.executable, "args": [str(hang)]}}}))
    try:
        assert "TimeoutError" in _mcp.search_mcp_tools("")
    finally:
        _mcp.shutdown_sessions()


def test_env_is_minimal(tmp_path, monkeypatch, fake_server_path):
    """The subprocess env must be minimal + declared entries, never the
    parent environment (MCP security guidance)."""
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setenv("NX_SECRET_PASSTHROUGH_PROBE", "leak")
    seen = {}
    probe = tmp_path / "env_probe_server.py"
    probe.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        "    msg=json.loads(line)\n"
        "    if 'id' not in msg: continue\n"
        "    r={'tools':[]}\n"
        "    if msg['method']=='initialize':\n"
        "        r={'protocolVersion':'2025-06-18'}\n"
        "    elif msg['method']=='tools/list':\n"
        "        import os as _os\n"
        "        r={'tools':[{'name':'probe','description':json.dumps(dict(_os.environ)),'inputSchema':{'type':'object'}}]}\n"
        "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':r})+'\\n')\n"
        "    sys.stdout.flush()\n")
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
    (tmp_path / ".nx").mkdir()
    (tmp_path / ".nx" / "mcp.json").write_text(json.dumps(
        {"mcpServers": {"probe": {"command": sys.executable, "args": [str(probe)],
                                  "env": {"MY_VAR": "declared"}}}}))
    try:
        out = _mcp.search_mcp_tools("probe")
        record = json.loads(out.split("\n", 1)[1])[0]
        env = json.loads(record["description"])
        assert env.get("MY_VAR") == "declared"
        assert "NX_SECRET_PASSTHROUGH_PROBE" not in env
        assert "PATH" in env
    finally:
        _mcp.shutdown_sessions()


# --- Streamable-HTTP transport (fake localhost server) ------------------

class _FakeMCPHandler(BaseHTTPRequestHandler):
    state = {"sessions": set(), "deleted": []}

    def log_message(self, *a):
        pass

    def do_DELETE(self):
        self.state["deleted"].append(self.headers.get("Mcp-Session-Id", ""))
        self.state["sessions"].discard(self.headers.get("Mcp-Session-Id", ""))
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(length))
        accept = self.headers.get("Accept", "")
        assert "text/event-stream" in accept, accept
        reply_id = msg.get("id")
        if reply_id is None:
            # notification: 202 Accepted, empty body
            self.send_response(202)
            self.end_headers()
            return
        method = msg["method"]
        if method == "initialize":
            sid = "fake-sid-1"
            self.state["sessions"].add(sid)
            result = {"protocolVersion": "2025-06-18",
                      "serverInfo": {"name": "httpfake", "version": "1"}}
            extra = {"mcp-session-id": sid}
        else:
            sid_header = self.headers.get("Mcp-Session-Id")
            assert sid_header in self.state["sessions"], (
                f"request without a live session: {sid_header!r}")
            extra = {"mcp-session-id": sid_header}
            if method == "tools/list":
                result = {"tools": [{"name": "ping",
                                     "description": "reply pong",
                                     "inputSchema": {"type": "object"}}]}
            elif method == "tools/call":
                result = {"content": [{"type": "text",
                                       "text": "pong: " + msg["params"]["arguments"]["x"]}],
                          "isError": False}
            else:
                result = {}
        sse = ("event: message\ndata: "
               + json.dumps({"jsonrpc": "2.0", "id": reply_id, "result": result})
               + "\n\n")
        body = sse.encode()
        self.send_response(200)
        for key, value in extra.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeMCPHandler)
    _FakeMCPHandler.state = {"sessions": set(), "deleted": []}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    server.shutdown()


def test_http_transport_roundtrip(http_server):
    session = _mcp._HttpSession("httpfake", {"url": http_server, "headers": {}})
    session.start()
    init = session.request("initialize", {}, 10)
    assert init["serverInfo"]["name"] == "httpfake"
    session.notify("notifications/initialized")
    result = session.request("tools/list", {}, 10)
    assert result["tools"][0]["name"] == "ping"
    called = session.request("tools/call",
                             {"name": "ping", "arguments": {"x": "hi"}}, 10)
    assert _mcp._content_text(called) == "pong: hi"
    session.stop()
    assert _FakeMCPHandler.state["deleted"], "DELETE (session teardown) not sent"


def test_http_remote_host_refused(monkeypatch):
    monkeypatch.setenv("NX_MCP", "1")
    with tempfile.TemporaryDirectory() as td:
        nx_dir = Path(td) / ".nx"
        nx_dir.mkdir()
        (nx_dir / "mcp.json").write_text(json.dumps(
            {"mcpServers": {"remote": {"url": "http://10.0.0.5/mcp"}}}))
        _state.PROJECT_ROOT = td
        try:
            servers, problem = _mcp._servers()
            assert problem and "localhost" in problem and servers == {}
        finally:
            _state.PROJECT_ROOT = None


def test_http_localhost_url_accepted(http_server, monkeypatch):
    monkeypatch.setenv("NX_MCP", "1")
    monkeypatch.setenv("NX_HOME", "/tmp/nx-mcp-test-home")
    with tempfile.TemporaryDirectory() as td:
        nx_dir = Path(td) / ".nx"
        nx_dir.mkdir()
        (nx_dir / "mcp.json").write_text(json.dumps(
            {"mcpServers": {"kb": {"url": http_server,
                                   "headers": {"Authorization": "Bearer t"}}}}))
        _state.PROJECT_ROOT = td
        try:
            servers, problem = _mcp._servers()
            assert problem is None and "kb" in servers
            assert _mcp.call_mcp_tool("kb", "ping", {"x": "deep"}) == "pong: deep"
        finally:
            _mcp.shutdown_sessions()
            _state.PROJECT_ROOT = None


def test_gate_closed_without_env(monkeypatch):
    from nx import _tool_gates
    monkeypatch.delenv("NX_MCP", raising=False)
    assert not _tool_gates._TOOL_GATES["search_mcp_tools"]()
    assert not _tool_gates._TOOL_GATES["call_mcp_tool"]()
    monkeypatch.setenv("NX_MCP", "1")
    assert _tool_gates._TOOL_GATES["search_mcp_tools"]()


def test_dispatch_entries_registered():
    from nx import _dispatch
    assert "search_mcp_tools" in _dispatch.DISPATCH
    assert "call_mcp_tool" in _dispatch.DISPATCH
