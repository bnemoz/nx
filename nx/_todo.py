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

import hashlib
import os

from typing import Any

from ._ansi import C, RESET, strip_ansi_for_display
from . import _fullscreen
from . import _state as st


def _todo_file() -> str | None:
    """The persistent todo file (0.94.0): NX_TODO_FILE when set (an empty
    value disables persistence); default .nx/todo.md under the project root.
    None = persistence off."""
    v = os.environ.get("NX_TODO_FILE")
    if v is not None:
        return os.path.expanduser(v) if v.strip() else None
    root = st.PROJECT_ROOT
    if not root:
        return None
    return os.path.join(root, ".nx", "todo.md")

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


def _plan_revision(todos: list[dict[str, Any]] | None = None) -> int:
    """A stable revision id for the plan: a hash of the ordered (text, status)
    pairs (0.161.0, AUTOWAKE_SPEC Phase C).

    The plan-stall wake fires AT MOST ONCE per revision, so this value must be
    stable (same plan -> same int, across processes and runs: hence blake2b,
    never Python's per-process-salted `hash()`), and it must move whenever the
    plan changes in any way the model could act on. `todo_write` is the ONLY
    writer of st._TODOS (it replaces the list wholesale), so the value is
    computed there and stashed on `st._PLAN_REVISION`: the wake poller then
    compares an int every 2s instead of re-hashing the list.

    An empty/absent plan hashes to a fixed value too (not -1), so the initial
    `st._WAKE_STALL_REVISION = -1` never spuriously matches a real plan, and a
    plan appearing from nothing still counts as a change."""
    todos = st._TODOS if todos is None else todos
    h = hashlib.blake2b(digest_size=8)
    for item in todos or []:
        h.update(str(item.get("text", "")).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(item.get("status", "pending")).encode("utf-8"))
        h.update(b"\x01")
    return int.from_bytes(h.digest(), "big")


def todo_write(todos: list[dict[str, Any]]) -> str:
    cleaned = []
    for item in todos:
        status = item.get("status", "pending")
        if status not in _STATUS_MARK:
            status = "pending"
        cleaned.append({"text": str(item.get("text", "")), "status": status})
    st._TODOS = cleaned
    # Phase C (0.161.0): stash the plan revision where the wake poller can read
    # an int instead of hashing the list every tick. Computed here because this
    # function is the plan's only writer.
    st._PLAN_REVISION = _plan_revision(cleaned)  # pyrefly: ignore[bad-assignment]  # dynamic st-module attribute
    _persist(cleaned)
    # Online Context Compact (0.67.0): a todo_write that newly completes >= 1
    # item is a plan-step boundary — the OCC module decides at the next
    # settled turn whether the economics justify a native fold. 0.159.3 (A′):
    # the boundary result is actually recorded now — inspect_todo_write's
    # return feeds record_boundary, which is the ONLY writer of
    # _OCC_PENDING_BOUNDARY; without this the whole NX_OCC turn-end seam was
    # wired at both ends and dead in the middle.
    try:
        from ._occompact import inspect_todo_write, record_boundary
        boundary = inspect_todo_write(cleaned)
        if boundary is not None:
            record_boundary(boundary[0], boundary[1])
    except Exception:  # noqa: BLE001 — never break the todo tool
        pass
    return render_todos(cleaned)


def _persist(todos: list[dict[str, str]]) -> None:
    """Write-through the todo list to the persistent todo file, if enabled.
    Fire-and-forget: persistence is an enhancement, never a failure path."""
    try:
        path = _todo_file()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        body = render_todos(todos)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# nx todos (auto-managed; safe to delete)\n\n{body}\n")
    except OSError:
        pass


def load_persisted_todos() -> list[dict[str, str]]:
    """Read the persistent todo file back into the st._TODOS shape. Returns
    [] when there is no file or it parses to nothing. Used at session start
    so a fresh session picks up the plan a previous session recorded."""
    try:
        path = _todo_file()
        if not path or not os.path.isfile(path):
            return []
        out = []
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            for status, mark in _STATUS_MARK.items():
                if ln.startswith(mark + " "):
                    out.append({"text": ln[len(mark) + 1:].strip(),
                                "status": status})
                    break
        return out
    except OSError:
        return []


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


# ---------------------------------------------------------------------------
# /todo — alternate-screen view (same Phase-D pattern as /tabs, /diff,
# /sessions: the view lives in the module that owns the data)
# ---------------------------------------------------------------------------
# Deliberately READ-ONLY, like every other view (`_fullscreen`'s contract: a
# view never mutates state): the plan is owned by `todo_write`, so cycling a
# status from here would duplicate that tool's whole rewrite-the-list logic.
# The inline one-shot print stays as the command's fallback, byte-identical to
# the pre-view `/todo` output.

class TodoView(_fullscreen.View):
    """The `/todo` view: the model's current todo_write list, full screen.

    Renders through `_render.py` (`panel`, `role`) and `_keys.py` (`hint_line`);
    the item lines are the SAME `render_todos_display` colors the tool output
    and the inline command use (done=SUCCESS, in_progress=WARNING,
    pending=TODO). j/k scroll the selection, counts per status ride in the
    panel title, and an empty list gets an explicit empty state instead of a
    blank box.
    """

    surface = "todos"
    title = "todos"

    def __init__(self) -> None:
        super().__init__()
        self.sel = 0
        self.items: list[dict] = []

    def refresh(self) -> None:
        """Re-read the live list (the poll-time repaint calls this, so a
        todo_write from a background turn lands without reopening the view)."""
        self.items = list(st._TODOS or [])
        self.sel = min(self.sel, max(0, len(self.items) - 1))

    def counts(self) -> dict:
        """Items per status; an unknown status counts as pending (the same
        fallback `_STATUS_MARK` applies)."""
        out = {"done": 0, "in_progress": 0, "pending": 0}
        for item in self.items:
            status = item.get("status", "pending")
            out[status if status in out else "pending"] += 1
        return out

    def _heading(self) -> str:
        n = len(self.items)
        if not n:
            return "todos · empty"
        c = self.counts()
        return (f"todos · {n} item{'' if n == 1 else 's'} · "
                f"{c['done']} done · {c['in_progress']} active · "
                f"{c['pending']} pending")

    def render(self, width: int, height: int) -> list[str]:
        from ._ansi import clip_display
        from ._keys import hint_line
        from ._render import RESET, panel, role
        self.refresh()
        lines = [f"{role('ACCENT')}todos{RESET}{role('MUTED')} — "
                 f"{hint_line('todos')}{RESET}"]
        chrome = 1 + 2  # title line + panel top/bottom
        budget = max(1, height - chrome)
        body: list[str] = []
        if not self.items:
            body.append(f"{role('MUTED')}  (todo list empty — the model "
                        f"records its plan with todo_write; /todo prints the "
                        f"same list inline){RESET}")
        else:
            # One body line per item, windowed on the selection so a long plan
            # scrolls instead of clipping the row the cursor is on. The
            # overflow note gets its own line only when something is hidden.
            shown = budget if len(self.items) <= budget else max(1, budget - 1)
            start = max(0, min(self.sel - shown + 1, len(self.items) - shown))
            stop = min(len(self.items), start + shown)
            for i in range(start, stop):
                marker = "›" if i == self.sel else " "
                body.append(clip_display(
                    f"{role('ACCENT')}{marker}{RESET} "
                    f"{render_todos_display([self.items[i]])}", width - 4))
            hidden = len(self.items) - (stop - start)
            if hidden > 0:
                body.append(f"{role('MUTED')}  … {hidden} more item"
                            f"{'s' if hidden != 1 else ''} (j/k to "
                            f"scroll){RESET}")
        lines.extend(panel(self._heading(), body, footer="q/esc back to chat"))
        return lines[:height]

    def on_key(self, key: str, shell) -> str:
        self.notice = ""
        if key in ("j", "down"):
            self.sel = min(max(0, len(self.items) - 1), self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            self.sel = max(0, self.sel - 1)
            return "repaint"
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def todo_view() -> "TodoView":
    """The /todo view object the fullscreen shell runs (`_fullscreen.todos()`
    is the thin alias the command path uses)."""
    return TodoView()
