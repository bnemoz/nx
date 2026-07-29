"""todo_write tool: whole-list task tracking for multi-step turns.

No separate read tool and no per-item id — each call rewrites the full list
(same convention as Claude Code's own TodoWrite), and the rendered list is
handed straight back as the tool result so the model never needs a round
trip just to see its own state. State lives in `st._TODOS` and is persisted
into the session JSON (see _repl_state.py) so /resume restores it.
"""
from __future__ import annotations

from typing import Any

from . import _state as st

_STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}


def render_todos(todos: list[dict[str, str]] | None = None) -> str:
    todos = st._TODOS if todos is None else todos
    if not todos:
        return "(todo list empty)"
    lines = []
    for item in todos:
        mark = _STATUS_MARK.get(item.get("status", "pending"), "[ ]")
        lines.append(f"{mark} {item.get('text', '')}")
    return "\n".join(lines)


def todo_write(todos: list[dict[str, Any]]) -> str:
    cleaned = []
    for item in todos:
        status = item.get("status", "pending")
        if status not in _STATUS_MARK:
            status = "pending"
        cleaned.append({"text": str(item.get("text", "")), "status": status})
    st._TODOS = cleaned
    return render_todos(cleaned)
