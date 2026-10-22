"""MCP (Model Context Protocol) connector — nx as a stdio MCP client.

Opt-in via ``NX_MCP=1``. Servers are declared in ``.nx/mcp.json`` under the
project root (falling back to ``~/.nx/mcp.json`` for user-wide servers), using
the de-facto standard ``mcpServers`` shape shared with Claude Code / Codex:

    {"mcpServers": {"fetch": {"command": "uvx",
                              "args": ["mcp-server-fetch"],
                              "env": {"KEY": "value"}}}}

Following the project-tools gateway pattern (nx/_project_tools.py): two stable
schemas — ``search_mcp_tools`` and ``call_mcp_tool`` — so the request's tool
prefix stays cacheable no matter how many servers or tools are configured.
Discovery and calls speak line-delimited JSON-RPC 2.0 over the server's
stdio (initialize / notifications/initialized / tools/list / tools/call),
stdlib-only: no SDK dependency for the host.

Security posture (per the MCP spec's security guidance): a server is code you
chose to run — enable this only for servers you trust. The subprocess env is
NOT the parent environment; only config-declared ``env`` entries plus a
minimal safe default (PATH, HOME, LANG, TERM) are forwarded. Server-provided
tool descriptions are untrusted data; they are returned verbatim in tool
results, never injected into prompts. Unknown server names, oversized
responses, and protocol deadlocks fail soft into actionable ERROR strings —
they never raise into the turn loop.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

_NAME = r"[A-Za-z0-9_-]{1,64}"
_CONNECT_TIMEOUT = 30.0   # initialize handshake ceiling, seconds
_CALL_TIMEOUT = 120.0     # tools/call ceiling, seconds
_MAX_RESPONSE_CHARS = 200_000  # a tool result larger than this is truncated
_MIN_ENV_KEYS = ("PATH", "HOME", "LANG", "TERM")


def _enabled() -> bool:
    return os.environ.get("NX_MCP", "0") not in ("", "0", "false", "False")


def _config_candidates() -> list[Path]:
    from . import _state as st
    roots = [Path(getattr(st, "PROJECT_ROOT", None) or Path.cwd()),
             Path.cwd(), Path.home() / ".nx"]
    out, seen = [], set()
    for root in roots:
        path = root / ".nx" / "mcp.json" if root.name != ".nx" else root / "mcp.json"
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _config_path() -> Path | None:
    for path in _config_candidates():
        if path.exists():
            return path
    return None


def _error(message: str) -> str:
    return f"ERROR: mcp: {message}"


def _servers() -> tuple[dict[str, dict[str, Any]], str | None]:
    """Read and validate server declarations without launching anything."""
    if not _enabled():
        return {}, "disabled (set NX_MCP=1 to enable MCP servers)"
    path = _config_path()
    if path is None:
        return {}, ("no server config (create .nx/mcp.json — "
                    '{"mcpServers": {"<name>": {"command": "...", "args": [...], "env": {}}}})')
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"cannot read {path}: {exc}"
    if not isinstance(raw, dict):
        return {}, f"{path}: root must be an object"
    raw = raw.get("mcpServers", raw)
    if not isinstance(raw, dict):
        return {}, f"{path}: mcpServers must be an object"
    out: dict[str, dict[str, Any]] = {}
    import re as _re
    for name, spec in raw.items():
        if not isinstance(name, str) or not _re.fullmatch(_NAME, name):
            return {}, f"{path}: invalid server name {name!r}"
        if not isinstance(spec, dict):
            return {}, f"{path}: {name}: server definition must be an object"
        command = spec.get("command")
        if not isinstance(command, str) or not command.strip():
            return {}, f"{path}: {name}: command must be a non-empty string"
        args = spec.get("args", [])
        env = spec.get("env", {})
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return {}, f"{path}: {name}: args must be a list of strings"
        if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            return {}, f"{path}: {name}: env must be an object of strings"
        out[name] = {"command": command, "args": args, "env": env}
    return out, None


class _ServerSession:
    """One stdio JSON-RPC session against one MCP server subprocess.

    A reader thread consumes stdout so a server that interleaves
    notifications with responses cannot deadlock the pipe; requests block on
    per-id events with an overall timeout.
    """

    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self._proc: subprocess.Popen | None = None
        self._responses: dict[Any, dict[str, Any]] = {}
        self._events: dict[Any, threading.Event] = {}
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._spec = spec
        self.negotiated_version: str | None = None
        self.server_info: dict[str, Any] = {}

    # --- lifecycle ---

    def start(self) -> None:
        env = {k: v for k, v in os.environ.items() if k in _MIN_ENV_KEYS}
        env.update(self._spec["env"])
        self._proc = subprocess.Popen(
            [self._spec["command"], *self._spec["args"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "id" in msg and msg["id"] is not None:
                with self._lock:
                    self._responses[msg["id"]] = msg
                    event = self._events.pop(msg["id"], None)
                if event is not None:
                    event.set()
        # EOF: wake every waiter so calls fail instead of hanging.
        with self._lock:
            events = list(self._events.values())
            self._events.clear()
        for event in events:
            event.set()

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise ConnectionError(f"server {self.name!r} is not running")
        proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any],
                timeout: float) -> dict[str, Any]:
        """One JSON-RPC round-trip; raises ConnectionError/TimeoutError on
        transport failure — callers convert to fail-soft strings."""
        request_id = f"nx-{method}-{next(_REQ_COUNTER)}-{threading.get_ident()}"
        event = threading.Event()
        with self._lock:
            self._events[request_id] = event
        try:
            self._send({"jsonrpc": "2.0", "id": request_id,
                        "method": method, "params": params})
            if not event.wait(timeout):
                raise TimeoutError(
                    f"{method} exceeded {timeout:g}s on server {self.name!r}")
            with self._lock:
                response = self._responses.pop(request_id, None)
            if response is None:
                raise ConnectionError(
                    f"server {self.name!r} closed stdout before answering {method}")
            if "error" in response:
                err = response["error"]
                detail = err.get("message", json.dumps(err)) if isinstance(err, dict) else str(err)
                raise ConnectionError(f"server error on {method}: {detail}")
            return response.get("result") or {}
        finally:
            with self._lock:
                self._events.pop(request_id, None)
                self._responses.pop(request_id, None)

    def notify(self, method: str) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": method})
        except (ConnectionError, BrokenPipeError, OSError):
            pass

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass


_REQ_COUNTER = iter(range(1, 1 << 62))


# --- one session per server per process, created lazily, torn down at exit ---

_SESSIONS: dict[str, _ServerSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_session(name: str, spec: dict[str, Any]) -> _ServerSession:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(name)
        if session is None:
            session = _ServerSession(name, spec)
            session.start()
            try:
                result = session.request("initialize", {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "nx", "version": "mcp-connector"},
                }, _CONNECT_TIMEOUT)
                # Echo the negotiated version back; the notification carries no id.
                negotiated = result.get("protocolVersion", "2025-06-18")
                session.notify("notifications/initialized")
                session.negotiated_version = negotiated
                session.server_info = result.get("serverInfo") or {}
            except Exception:
                session.stop()
                raise
            _SESSIONS[name] = session
        return session


def _timeout() -> float:
    from ._env import _env_float
    return max(1.0, _env_float("NX_MCP_TIMEOUT", _CONNECT_TIMEOUT))


def _call_timeout() -> float:
    from ._env import _env_float
    return max(1.0, _env_float("NX_MCP_TOOL_TIMEOUT", _CALL_TIMEOUT))


def search_mcp_tools(pattern: str = "") -> str:
    """List every MCP server's tools with their JSON Schemas (search only —
    nothing is executed). Servers launch lazily on first use and are cached."""
    servers, problem = _servers()
    if problem:
        return _error(problem) if problem.startswith("cannot") else f"MCP: {problem}."
    try:
        rx = __import__("re").compile(pattern, __import__("re").IGNORECASE) if pattern else None
    except __import__("re").error as exc:
        return _error(f"invalid pattern {pattern!r}: {exc}")
    found, failures = [], []
    for name, spec in servers.items():
        try:
            session = _get_session(name, spec)
            result = session.request("tools/list", {}, _timeout())
        except Exception as exc:  # noqa: BLE001 — fail soft per server
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        tools = result.get("tools")
        if not isinstance(tools, list):
            failures.append(f"{name}: tools/list returned no tool list")
            continue
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                continue
            record = {"server": name, "name": tool["name"],
                      "description": tool.get("description", ""),
                      "inputSchema": tool.get("inputSchema", {})}
            if rx and not rx.search(
                    f"{record['name']} {record['description']} {json.dumps(record['inputSchema'])}"):
                continue
            found.append(record)
    if failures:
        head = f"MCP server problems: {'; '.join(failures)}"
        return head if not found else head + "\n" + json.dumps(found, ensure_ascii=False)
    if not found:
        return ("No MCP tools matched." if servers
                else "MCP: no servers declared in .nx/mcp.json.")
    return "MCP tool schemas (call with call_mcp_tool):\n" + json.dumps(found, ensure_ascii=False)


def call_mcp_tool(server: str, tool: str, arguments: dict[str, Any]) -> str:
    """Run one tool on one MCP server; returns the tool's text content (or an
    ERROR string — server failures never raise into the turn loop)."""
    servers, problem = _servers()
    if problem:
        return _error(problem) if problem.startswith("cannot") else f"MCP: {problem}."
    import re as _re
    if not isinstance(server, str) or not _re.fullmatch(_NAME, server):
        return _error(f"invalid server name {server!r}")
    if not isinstance(tool, str) or not _re.fullmatch(_NAME, tool):
        return _error(f"invalid tool name {tool!r}")
    if not isinstance(arguments, dict):
        return _error("arguments must be an object")
    spec = servers.get(server)
    if spec is None:
        return _error(f"unknown server {server!r}; declared: {', '.join(sorted(servers)) or 'none'}")
    try:
        session = _get_session(server, spec)
        result = session.request("tools/call",
                                 {"name": tool, "arguments": arguments},
                                 _call_timeout())
    except Exception as exc:  # noqa: BLE001 — fail soft
        return _error(f"{server}.{tool}: {type(exc).__name__}: {exc}")
    if result.get("isError"):
        return _error(f"{server}.{tool}: {_content_text(result)}")
    text = _content_text(result)
    if len(text) > _MAX_RESPONSE_CHARS:
        text = (text[:_MAX_RESPONSE_CHARS]
                + f"\n… [truncated at {_MAX_RESPONSE_CHARS} chars; total {len(text)}]")
    return text or "(empty tool result)"


def _content_text(result: dict[str, Any]) -> str:
    blocks = result.get("content")
    parts: list[str] = []
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(parts)


def shutdown_sessions() -> None:
    """Terminate every live server subprocess (called at interpreter exit)."""
    with _SESSIONS_LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for session in sessions:
        session.stop()


atexit.register(shutdown_sessions)
