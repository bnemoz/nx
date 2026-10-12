"""F3: /ctx token report + pinned turns surviving compress."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    import nx
    # The old `importlib.reload(nx)` here rebuilt nx's import-time state from
    # this fixture's env — most importantly the model source: conftest imports
    # nx once under a scrubbed env, so st.client/MODEL are bound to whatever the
    # baseline discovered. Instead of a reload, rebuild just that wiring:
    # re-discover sources from the env we just set and switch to the first one
    # (binds st.client + st.MODEL). resolve_model() reads NX_MODEL
    # ("test-model"), so no network round-trip happens; the dead 127.0.0.1:9
    # endpoint is never contacted (and the compress tests monkeypatch create).
    # See tests/test_subagent.py's nx_fresh for the canonical form.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # st._CTX_PINNED used to be wiped for free by conftest's module-dict rewind
    # (Tier B). Tier A files skip that walk, and /ctx pin reads the existing
    # set, so a left-over pin from an earlier test would change results: pin it
    # per-test (monkeypatch auto-restores the pre-fixture value at teardown).
    monkeypatch.setattr(nx._state, "_CTX_PINNED", [], raising=False)
    yield nx
    # env rewind is universal (conftest) and the pin is monkeypatch-owned, so
    # no manual pop/restore here.


def _msgs(n=12, big_every=3):
    msgs = [{"role": "system", "content": "sys prompt"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        size = 4000 if (i % big_every == 0) else 40
        msgs.append({"role": role, "content": f"msg {i}: " + "x" * size})
    return msgs


def test_token_report_shape_and_role_subtotals(nx_fresh):
    from nx._ctxreport import _messages_token_report, _estimate_tokens
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a" * 400) == 100
    msgs = _msgs()
    rep = _messages_token_report(msgs)
    assert rep["total_exact"] is None or isinstance(rep["total_exact"], int)
    assert len(rep["rows"]) == len(msgs)
    assert rep["by_role"]["system"] == rep["rows"][0][2]
    # rows carry (idx, role, tokens, preview, pinned)
    idx, role, tok, preview, pinned = rep["rows"][1]
    assert idx == 1 and role == "user" and tok > 0 and preview.startswith("msg 0")
    # the big turns dominate
    bigs = [r for r in rep["rows"] if r[2] > 500]
    assert len(bigs) >= 3


def test_total_includes_per_turn_tools_overhead(nx_fresh):
    """The `tools=` payload is sent on every model call but lives outside the
    message list — /ctx's total must count it (as its own overhead_est), or
    the report undercounts the real per-turn floor (the bug where a fresh
    session showed ~1331 est while --tokens said ~3x that)."""
    import json as _json
    from nx._ctxreport import _messages_token_report, _overhead_lines
    from nx._dispatch import _active_tools
    msgs = _msgs(4)
    rep = _messages_token_report(msgs)
    tools = _active_tools()
    assert tools, "no active tool schemas — overhead would be vacuously 0"
    expected = max(1, len(_json.dumps(tools)) // 4)
    assert rep["overhead_est"] == expected
    assert rep["total_est"] == expected + sum(r[2] for r in rep["rows"])
    # by_role stays messages-only: the overhead has no message row
    assert sum(rep["by_role"].values()) == sum(r[2] for r in rep["rows"]) < rep["total_est"]
    # the footer names the overhead so "by role" summing low isn't confusing
    lines = _overhead_lines(rep["total_est"], rep["overhead_est"])
    assert len(lines) == 1 and "tools schema" in lines[0]
    assert _overhead_lines(rep["total_est"], 0) == []


def test_ctx_report_renders_overhead_line(nx_fresh, capsys):
    from nx._repl_settings_cmds import _cmd_ctx
    from nx._repl_state import _ReplState
    state = _ReplState(_msgs(4), "t", False, [])
    _cmd_ctx("/ctx", state)
    out = capsys.readouterr().out
    assert "tools schema" in out
    assert "sent every turn" in out


def test_pin_spec_parsing(nx_fresh):
    from nx._ctxreport import _parse_pin_spec
    assert _parse_pin_spec("3", 20) == [2]
    assert _parse_pin_spec("3-5", 20) == [2, 3, 4]
    assert _parse_pin_spec("1,3,10-12", 20) == [0, 2, 9, 10, 11]
    with pytest.raises(ValueError):
        _parse_pin_spec("99", 20)
    with pytest.raises(ValueError):
        _parse_pin_spec("0", 20)
    with pytest.raises(ValueError):
        _parse_pin_spec("abc", 20)


def test_pin_unpin_roundtrip(nx_fresh):
    from nx._ctxreport import _ctx_pin, _ctx_unpin
    msgs = _msgs()
    added, pinned = _ctx_pin("3-4", msgs)
    assert added == [2, 3] and pinned == [2, 3]
    # re-pinning is idempotent
    added, pinned = _ctx_pin("4", msgs)
    assert added == []
    removed, remaining = _ctx_unpin("3", msgs)
    assert removed == [2] and remaining == [3]
    removed, remaining = _ctx_unpin(None, msgs)
    assert removed == [3] and remaining == []


def test_compress_keeps_pinned_turns_verbatim(nx_fresh, monkeypatch):
    """A pinned early turn must survive a fold that would otherwise summarize it."""
    from nx._compress import compress
    msgs = _msgs(14)
    pin_target = 1  # the first big user turn (early — always in the fold range)
    monkeypatch.setattr(nx_fresh._state, "_CTX_PINNED", [pin_target])
    original = msgs[pin_target]["content"]

    def fake_create(model, messages, stream=False, **kw):
        class R:  # minimal response stand-in
            class choices:
                pass
        resp = type("Resp", (), {})()
        resp.choices = [type("C", (), {"message": type("M", (), {"content": "summary of old turns"})()})()]
        resp.usage = None
        return resp

    monkeypatch.setattr(nx_fresh._state.client.chat.completions, "create", fake_create)
    result = compress(msgs, keep=2)
    assert result is not None
    kept_n, summarized_n, _ = result
    # the pinned message content survives verbatim somewhere in messages
    assert any(m.get("content") == original for m in msgs)
    # and it's in the kept tail region, not the summarized header
    assert kept_n >= 3  # 2 verbatim tail + at least the pinned pair


def test_compress_pin_on_tool_result_pulls_its_call(nx_fresh, monkeypatch):
    """Pinning a tool result keeps the assistant tool_call turn with it —
    the tail must stay renderable (no dangling tool result)."""
    from nx._compress import compress
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}",
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "read_file", "arguments": '{"path": "x"}'}}]})
        msgs.append({"role": "tool", "content": f"t{i} " + "y" * 200})
    # pin ONLY the first tool result (index 3); its call turn (index 2) must ride along
    monkeypatch.setattr(nx_fresh._state, "_CTX_PINNED", [3])

    def fake_create(model, messages, stream=False, **kw):
        resp = type("Resp", (), {})()
        resp.choices = [type("C", (), {"message": type("M", (), {"content": "summary"})()})()]
        resp.usage = None
        return resp

    monkeypatch.setattr(nx_fresh._state.client.chat.completions, "create", fake_create)
    result = compress(msgs, keep=2)
    assert result is not None
    body = msgs[1:]
    # the pinned tool result survives verbatim
    assert any(m.get("role") == "tool" and m.get("content", "").startswith("t0") for m in body)
    # and its assistant tool_call turn came with it
    assert any(m.get("role") == "assistant" and m.get("content") == "a0" and m.get("tool_calls") for m in body)
    # no dangling tool result: every tool turn has a preceding assistant with tool_calls
    for i, m in enumerate(body):
        if m.get("role") == "tool":
            assert any(mm.get("tool_calls") for mm in body[:i]), f"dangling tool at body[{i}]"


def test_ctx_command_renders_and_pins(nx_fresh, capsys):
    from nx._repl_settings_cmds import _cmd_ctx
    from nx._repl_state import _ReplState
    state = _ReplState(_msgs(8), "t", False, [])
    _cmd_ctx("/ctx", state)
    out = capsys.readouterr().out
    assert "context by message" in out
    assert "by role:" in out
    assert "pin" in out
    _cmd_ctx("/ctx pin 2", state)
    out = capsys.readouterr().out
    assert "pinned 1 turn(s)" in out
    assert nx_fresh._state._CTX_PINNED == [1]
    _cmd_ctx("/ctx unpin 2", state)
    capsys.readouterr()
    assert nx_fresh._state._CTX_PINNED == []
    _cmd_ctx("/ctx pin 99", state)
    out = capsys.readouterr().out
    assert "outside 1..9" in out
    # pin + clear
    _cmd_ctx("/ctx pin 1,3", state)
    capsys.readouterr()
    _cmd_ctx("/ctx clear", state)
    capsys.readouterr()
    assert nx_fresh._state._CTX_PINNED == []
