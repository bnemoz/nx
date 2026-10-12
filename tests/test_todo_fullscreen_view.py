"""The /todo fullscreen view (0.148.5): render, empty state, keys, dispatch.

/todo keeps its pre-view inline print as the fallback (what runs off a tty /
with NX_FULLSCREEN=0); on a tty it opens `TodoView` through the Phase-D
fullscreen shell, exactly like the /tabs view. These tests cover the view
(data from st._TODOS, colored via render_todos_display, counts in the panel
title, an explicit empty state) and the command dispatch (open_view plus the
fallback), without needing a real terminal.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-todo-fs-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _state as st  # noqa: E402


def test_todo_view_renders_counts_and_items(monkeypatch):
    from nx import _render, _todo
    monkeypatch.setattr(st, "_TODOS", [
        {"text": "inspect repo", "status": "done"},
        {"text": "add the view", "status": "in_progress"},
        {"text": "write tests", "status": "pending"}])
    body = "\n".join(_todo.todo_view().render(80, 24))
    assert _render.BOX_TL in body and _render.BOX_BL in body
    assert "inspect repo" in body and "[x]" in body
    # counts per status ride in the panel title
    assert "3 items" in body and "1 done" in body
    assert "1 active" in body and "1 pending" in body


def test_todo_view_empty_state(monkeypatch):
    from nx import _render, _todo
    monkeypatch.setattr(st, "_TODOS", [])
    body = "\n".join(_todo.todo_view().render(80, 24))
    assert _render.BOX_TL in body, "empty state must still be a panel"
    assert "todo list empty" in body


def test_todo_view_keys_are_advertised_by_the_registry(monkeypatch):
    from nx import _keys, _todo
    monkeypatch.setattr(st, "_TODOS", [{"text": "a", "status": "pending"},
                                       {"text": "b", "status": "pending"}])
    view = _todo.todo_view()
    view.render(80, 24)  # first paint refreshes the rows (TabsView pattern)
    for key, _meaning in _keys.keys_for("todos"):
        assert view.on_key("j" if key == "j/k" else key, None) in (
            "repaint", "quit", "run"), key
    # j/k actually move the selection
    view.sel = 0
    view.on_key("j", None)
    assert view.sel == 1
    view.on_key("k", None)
    assert view.sel == 0


def test_todo_command_opens_the_view_with_the_inline_fallback(capsys, monkeypatch):
    """Dispatch: `_cmd_todo` opens TodoView through `_fullscreen.open_view`,
    and the fallback is the inline print (what the REPL ran before the view
    existed, and still runs off a tty)."""
    from nx import _fullscreen
    import nx._repl_settings_cmds as rs
    monkeypatch.setattr(st, "_TODOS", [{"text": "seeded", "status": "done"}])
    opened = []

    def _fake_open(view, fallback=None):
        opened.append(view)
        fallback()
        return True

    monkeypatch.setattr(_fullscreen, "open_view", _fake_open)
    rs._cmd_todo("/todo", None)
    assert len(opened) == 1 and opened[0].surface == "todos"
    out = capsys.readouterr().out
    assert "seeded" in out and "[x]" in out
