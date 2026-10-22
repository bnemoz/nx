"""MCP connector (nx/_mcp.py): config validation, the stdio JSON-RPC client
against a fake in-repo MCP server subprocess, gateway behavior, and gates."""
import json
import sys

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
    monkeypatch.setattr(_state, "PROJECT_ROOT", str(tmp_path), raising=False)
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
