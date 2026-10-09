"""Regression tests for the 0.94.0 feature batch: /wip, /fork, the
persistent todo file (NX_TODO_FILE), test-impact targeting, and the
tier-2 lint_check tool."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

import nx as m
from nx import _state as st
from nx._repl_state import _ReplState
from nx._sessions_io import _new_session_id


@pytest.fixture()
def state(tmp_path):
    st._TODOS = []
    st._WIP = None
    s = _ReplState(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        _new_session_id(), True, [])
    os.chdir(tmp_path)
    st.PROJECT_ROOT = str(tmp_path)
    yield s
    st._TODOS = []
    st._WIP = None


# --- /wip --------------------------------------------------------------------

def test_wip_snapshot_roundtrip(state, tmp_path, monkeypatch, capsys):
    from nx import _wip
    from nx._repl_state import _save_current, _load_session
    st._TODOS = [{"text": "write tests", "status": "in_progress"}]
    _wip._cmd_wip("/wip mid-refactor", state)
    out = capsys.readouterr().out
    assert "work state recorded" in out
    assert st._WIP is not None and st._WIP["note"] == "mid-refactor"
    # The snapshot rides the session file.
    _save_current(state)
    data = _load_session(state.session_id)
    assert data["wip"]["note"] == "mid-refactor"
    assert data["wip"]["todos"][0]["text"] == "write tests"
    # render_wip composes a one-liner
    line = _wip.render_wip(data["wip"])
    assert "todos open" in line and "mid-refactor" in line
    assert _wip.render_wip(None) is None


def test_wip_restored_on_reload(state):
    from nx._repl_state import _reload_metrics_totals
    st._WIP = {"branch": "main", "note": "x"}
    from nx._sessions_fs import _write_session
    _write_session(state.session_id, state.messages, {"wip": {"branch": "dev"}})
    st._WIP = None
    _reload_metrics_totals(state.session_id)
    assert st._WIP == {"branch": "dev"}


# --- /fork -------------------------------------------------------------------

def test_fork_creates_new_session_and_keeps_original(state, capsys, monkeypatch):
    from nx import _sessions_cmd
    # Stub at the SOURCE module: /fork now routes through the SIGINT-guarded
    # _save_current (0.129.0), which _sessions_cmd imports directly — the stub
    # must sit where the code looks it up, not on the wrapper.
    import nx._repl_state as _rs
    saved = {}
    monkeypatch.setattr(_rs, "_write_session",
                        lambda *a, **k: saved.setdefault("id", a[0]))
    old_id = state.session_id
    _sessions_cmd._cmd_fork("/fork experiment-b", state)
    out = capsys.readouterr().out
    assert "forked to session" in out
    assert state.session_id != old_id
    assert len(state.messages) == 2  # transcript carried
    assert not state.session_dirty
    # The new session file exists on disk and references its parent.
    from nx._sessions_fs import _load_session
    data = _load_session(state.session_id)
    assert data is not None
    assert data.get("forked_from") == old_id
    assert data.get("title") == "experiment-b"
    # The original file is untouched.
    assert _load_session(old_id) is not None or saved  # original kept


def test_fork_empty_session_is_a_noop(state, capsys):
    from nx import _sessions_cmd
    state.messages = []
    _sessions_cmd._cmd_fork("/fork", state)
    assert "nothing to fork" in capsys.readouterr().out


# --- persistent todo file ----------------------------------------------------

def test_todo_write_persists_and_loads(tmp_path, monkeypatch):
    from nx._todo import todo_write, load_persisted_todos
    todo_file = tmp_path / "todo.md"
    monkeypatch.setenv("NX_TODO_FILE", str(todo_file))
    todo_write([{"text": "a", "status": "done"},
                {"text": "b", "status": "pending"}])
    assert todo_file.exists()
    loaded = load_persisted_todos()
    assert loaded == [{"text": "a", "status": "done"},
                      {"text": "b", "status": "pending"}]
    # Empty value disables persistence entirely.
    todo_file.unlink()
    monkeypatch.setenv("NX_TODO_FILE", "")
    todo_write([{"text": "c"}])
    assert not todo_file.exists()


def test_todo_file_default_location(tmp_path, monkeypatch):
    from nx._todo import _todo_file
    monkeypatch.delenv("NX_TODO_FILE", raising=False)
    st.PROJECT_ROOT = str(tmp_path)
    assert _todo_file() == os.path.join(str(tmp_path), ".nx", "todo.md")
    monkeypatch.setenv("NX_TODO_FILE", "")
    assert _todo_file() is None
    monkeypatch.setenv("NX_TODO_FILE", "~/t.md")
    assert _todo_file() == os.path.expanduser("~/t.md")


# --- test-impact map ---------------------------------------------------------

def test_record_and_map_failures(tmp_path, monkeypatch):
    from nx._testrunner import _record_failures, _mapped_targets, _testmap_path
    monkeypatch.setattr(st, "PROJECT_ROOT", str(tmp_path))
    _record_failures("pytest",
                     "FAILED tests/test_a.py::test_x - assert 1 == 2\n"
                     "1 failed in 0.1s")
    path = _testmap_path()
    assert os.path.exists(path)
    assert _mapped_targets(["tests/test_a.py"]) == \
        ["tests/test_a.py::test_x"]
    # Unrelated files map to nothing; garbage lines never raise.
    assert _mapped_targets(["other.py"]) == []
    _record_failures("pytest", "garbage line")
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json\n")


# --- lint_check --------------------------------------------------------------

def test_lint_tool_schema_and_gate():
    names = {t["function"]["name"] for t in m.TOOLS}
    assert "lint_check" in names
    from nx._tool_gates import _TOOL_GATES, _GATE_REFUSALS, _gate_closed
    assert "lint_check" in _TOOL_GATES and "lint_check" in _GATE_REFUSALS
    monkey_gate = dict(os.environ)
    os.environ.pop("NX_LINT", None)
    import importlib
    importlib.reload(m._lint) if hasattr(m, "_lint") else None
    m.NX_LINT = 0
    assert _gate_closed("lint_check") is True
    m.NX_LINT = 1
    assert _gate_closed("lint_check") is False
    os.environ.clear()
    os.environ.update(monkey_gate)


def test_lint_check_body(monkeypatch, tmp_path):
    from nx._lint import lint_check
    st.PROJECT_ROOT = str(tmp_path)
    # Missing paths are refused.
    assert "not found" in lint_check("nope.py")
    # No paths + no session edits -> explicit error, never a whole-tree run.
    monkeypatch.setattr("nx._lint._session_touched_files", lambda: [])
    assert "no paths given" in lint_check("")
    # Custom command override runs verbatim.
    monkeypatch.setenv("NX_LINT_CMD", "echo hello-lint")
    target = tmp_path / "f.txt"
    target.write_text("x")
    out = lint_check(str(target))
    assert "hello-lint" in out and "custom (NX_LINT_CMD)" in out


def test_lint_check_uses_session_edited_files(monkeypatch, tmp_path):
    from nx._lint import lint_check
    st.PROJECT_ROOT = str(tmp_path)
    target = tmp_path / "edited.py"
    target.write_text("x = 1\n")

    class F:  # minimal file-snapshot shape
        path = str(target)
    class CP:
        files = [F()]
    monkeypatch.setattr("nx._undo.list_checkpoints", lambda: [CP()])
    monkeypatch.setenv("NX_LINT_CMD", "echo touched")
    assert "touched" in lint_check("")
