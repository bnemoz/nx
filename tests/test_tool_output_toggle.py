"""Tests for the `/tooloutput` toggle and its effect on run_tool() rendering.

Mirrors the `/reasoning` on|off|last design: `/tooloutput` flips
`TOOL_OUTPUT_VISIBLE` (persisted to ~/.env as NX_TOOL_OUTPUT_VISIBLE), and
run_tool() hides the tool result body when collapsed while still printing the
box header/args lines. `/tools` is kept as a misleading-name legacy alias that
routes to the same handler. `todo_write` always renders its colored list
regardless of the flag.

Run offline — no model endpoint is contacted. run_tool() is exercised directly
with a stubbed dispatch entry so we assert on the captured stdout.
"""
import builtins
import contextlib
import importlib
import io
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
# Start from a clean env so the default (`1`) is deterministic below.
for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reload_nx(env=None):
    """Reload nx (and its _state submodule so the env-var defaults re-resolve)."""
    for k in list(os.environ):
        if k.startswith("NX_"):
            del os.environ[k]
    os.environ["NX_ENV_FILE"] = "/dev/null"
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    if env:
        os.environ.update(env)
    import nx
    importlib.reload(nx._state)
    return importlib.reload(nx)


@contextlib.contextmanager
def capture_stdout():
    out = io.StringIO()
    real = builtins.print

    def fake(*a, **kw):
        out.write(" ".join(str(x) for x in a) + (kw.get("end", "\n")))

    builtins.print = fake
    try:
        yield out
    finally:
        builtins.print = real


def test_default_visible():
    m = reload_nx()
    assert m.st.TOOL_OUTPUT_VISIBLE is True


def test_env_zero_collapses():
    m = reload_nx(env={"NX_TOOL_OUTPUT_VISIBLE": "0"})
    assert m.st.TOOL_OUTPUT_VISIBLE is False


def test_env_false_collapses():
    m = reload_nx(env={"NX_TOOL_OUTPUT_VISIBLE": "false"})
    assert m.st.TOOL_OUTPUT_VISIBLE is False


def test_env_off_collapses():
    m = reload_nx(env={"NX_TOOL_OUTPUT_VISIBLE": "off"})
    assert m.st.TOOL_OUTPUT_VISIBLE is False


def test_cmd_tooloutput_on_off_last(tmp_path, monkeypatch):
    m = reload_nx()
    env_path = tmp_path / "nx.env"
    monkeypatch.setenv("NX_ENV_FILE", str(env_path))

    from nx._repl_state import _ReplState
    from nx._repl_settings_cmds import _cmd_tooloutput
    state = _ReplState([], "s", False, [])

    # /tooloutput off → collapsed, persisted to the env file
    with capture_stdout() as out:
        _cmd_tooloutput("/tooloutput off", state)
    assert m.st.TOOL_OUTPUT_VISIBLE is False
    assert os.environ["NX_TOOL_OUTPUT_VISIBLE"] == "0"
    assert "NX_TOOL_OUTPUT_VISIBLE=0" in env_path.read_text()
    assert "collapsed" in out.getvalue()
    assert m.st._LAST_TOOL_OUTPUT == ""  # nothing captured yet

    # /tooloutput on → visible again, persisted
    with capture_stdout() as out:
        _cmd_tooloutput("/tooloutput on", state)
    assert m.st.TOOL_OUTPUT_VISIBLE is True
    assert os.environ["NX_TOOL_OUTPUT_VISIBLE"] == "1"
    assert "visible" in out.getvalue()

    # bare /tooloutput → reports current state without changing it
    with capture_stdout() as out:
        _cmd_tooloutput("/tooloutput", state)
    assert "visible" in out.getvalue()

    # /tooloutput last with no captured output → friendly empty message
    with capture_stdout() as out:
        _cmd_tooloutput("/tooloutput last", state)
    assert "no tool output captured yet" in out.getvalue()

    # unknown arg → error message
    with capture_stdout() as out:
        _cmd_tooloutput("/tooloutput bogus", state)
    assert "unknown /tooloutput arg" in out.getvalue()


def test_tools_legacy_alias_still_dispatches():
    """`/tools` is a misleading name, kept only as a backward-compat alias —
    it must route to the same handler as `/tooloutput`."""
    reload_nx()
    from nx._repl import _REPL_COMMANDS
    assert "/tooloutput" in _REPL_COMMANDS
    assert "/tools" in _REPL_COMMANDS
    assert _REPL_COMMANDS["/tools"] is _REPL_COMMANDS["/tooloutput"]


def test_tools_legacy_alias_changing_state(tmp_path, monkeypatch):
    """The alias must actually flip the flag, not just be registered."""
    m = reload_nx()
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "nx.env"))
    from nx._repl_state import _ReplState
    from nx._repl_settings_cmds import _cmd_tooloutput
    state = _ReplState([], "s", False, [])
    with capture_stdout():
        _cmd_tooloutput("/tools off", state)  # alias used to invoke the handler
    assert m.st.TOOL_OUTPUT_VISIBLE is False


def test_run_tool_renders_body_when_visible(monkeypatch):
    m = reload_nx()
    m.st.TOOL_OUTPUT_VISIBLE = True
    from nx import _dispatch as d
    monkeypatch.setitem(d.DISPATCH, "read_file", lambda **a: "line one\nline two")

    with capture_stdout() as out:
        result = d.run_tool("read_file", {"path": "x"})

    assert result == "line one\nline two"
    assert m.st._LAST_TOOL_NAME == "read_file"
    assert m.st._LAST_TOOL_OUTPUT == "line one\nline two"
    text = out.getvalue()
    assert "read_file" in text          # box header always renders
    assert "line one" in text           # result body renders when visible
    assert "collapsed" not in text


def test_run_tool_hides_body_when_collapsed(monkeypatch):
    m = reload_nx()
    m.st.TOOL_OUTPUT_VISIBLE = False
    from nx import _dispatch as d
    monkeypatch.setitem(d.DISPATCH, "read_file", lambda **a: "line one\nline two")

    with capture_stdout() as out:
        result = d.run_tool("read_file", {"path": "x"})

    # The sanitized result still goes back to the model and is captured…
    assert result == "line one\nline two"
    assert m.st._LAST_TOOL_OUTPUT == "line one\nline two"
    # …but the body is hidden on screen, replaced by the collapse hint.
    text = out.getvalue()
    assert "read_file" in text      # header still shows
    assert "collapsed" in text
    assert "line one" not in text   # body hidden


def test_todo_write_always_renders(monkeypatch):
    m = reload_nx()
    m.st.TOOL_OUTPUT_VISIBLE = False
    m.st._TODOS = [{"text": "write tests", "status": "in_progress"}]
    from nx import _dispatch as d
    monkeypatch.setitem(d.DISPATCH, "todo_write", lambda **a: "ok")

    with capture_stdout() as out:
        d.run_tool("todo_write", {"todos": [{"text": "write tests", "status": "in_progress"}]})

    text = out.getvalue()
    assert "todo_write" in text
    assert "write tests" in text    # todo list renders even when collapsed
    assert "collapsed" not in text  # no collapse hint for todo_write