"""todo_write tool: whole-list task tracking for multi-step turns.

No separate read tool and no per-item id — each call rewrites the full list
(same convention as Claude Code's own TodoWrite), and the rendered list is
handed straight back as the tool result so the model never needs a round
trip just to see its own state. State lives in `st._TODOS` and is persisted
into the session JSON (see _repl_state.py) so /resume restores it.

There are two renderers: `render_todos` returns plain text for the model
(the tool result body that enters message history), while
`render_todos_display` returns the same list colored per-status for the
terminal (done=green, in_progress=yellow, pending=orange), used by the
on-screen tool-output box and the `/todo` slash command.
"""
from __future__ import annotations

from typing import Any

from ._ansi import C, RESET, strip_ansi_for_display
from . import _state as st

_STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
# Status -> semantic color ROLE whose hue is resolved dynamically at render
# time (C.<ROLE>), so a live /theme switch is reflected without re-importing.
# done=SUCCESS (green), in_progress=WARNING (yellow), pending=TODO (orange) —
# each a distinct hue (see _ansi.py), so the three states are visually separable.
_STATUS_COLOR_ROLE = {"pending": "TODO", "in_progress": "WARNING", "done": "SUCCESS"}


def render_todos(todos: list[dict[str, str]] | None = None) -> str:
    """Plain-text rendering for the model (the todo_write tool result body).

    No color codes — this is the message-history copy. The on-screen colored
    form is render_todos_display().
    """
    todos = st._TODOS if todos is None else todos
    if not todos:
        return "(todo list empty)"
    lines = []
    for item in todos:
        mark = _STATUS_MARK.get(item.get("status", "pending"), "[ ]")
        lines.append(f"{mark} {item.get('text', '')}")
    return "\n".join(lines)


def render_todos_display(todos: list[dict[str, str]] | None = None) -> str:
    """Colored rendering for the terminal: each item's status sets its hue.

    done=green (SUCCESS), in_progress=yellow (WARNING), pending=orange (TODO),
    so the three states are visually distinct at a glance. Colors are read off
    the live `C` palette at call time (not captured at import), so a /theme
    switch updates them without a re-import. Item text is sanitized of any
    ANSI/control sequences before nx's own color codes are applied, so
    model-supplied text cannot smuggle a terminal escape through.
    """
    todos = st._TODOS if todos is None else todos
    if not todos:
        return "(todo list empty)"
    lines = []
    for item in todos:
        mark = _STATUS_MARK.get(item.get("status", "pending"), "[ ]")
        role = _STATUS_COLOR_ROLE.get(item.get("status", "pending"), "TODO")
        color = getattr(C, role)
        text = strip_ansi_for_display(str(item.get("text", "")))
        lines.append(f"{color}{mark} {text}{RESET}")
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


def _latest_todo_turn(body: list[dict[str, Any]]) -> int | None:
    """Index of the latest todo_write call+result pair in `body` (a foldable
    head slice, no system message), or None.

    Returns the index of the assistant turn that called todo_write. compress()
    expands pins to complete tool_call/tool-result pairs, so pinning this one
    index carries the rendered list result along too — re-injecting the most
    recent todo_write turn verbatim into the kept tail after every fold, so a
    mid-flight multi-step plan survives auto-compression.
    """
    for i in range(len(body) - 1, -1, -1):
        for tc in body[i].get("tool_calls") or []:
            if isinstance(tc, dict) and (tc.get("function") or {}).get("name") == "todo_write":
                return i
    return None
