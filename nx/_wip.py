"""Git-aware work-state snapshot (0.94.0): the /wip command.

/sessions and /save are conversation-centric; neither captures TASK state.
/wip records, in one shot, where the work stands so a later resume can pick
it up: branch, dirty files, open todos, and an optional free-text note —
composed locally (no model call, instant and free).

Storage: `st._WIP` rides the session JSON via `_save_current`'s meta
(`wip` key), restored on resume by `_reload_metrics_totals`.
`/sessions --wip` marks sessions that carry one.
"""
from __future__ import annotations

import subprocess

from ._ansi import DIM, RESET
from . import _state as st


def _git(args: list[str], cwd: str | None = None) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           timeout=5, cwd=cwd or st.PROJECT_ROOT or None)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _cmd_wip(user: str, state) -> None:
    """/wip [note] — snapshot work state into the session (branch, dirty
    files, todos, optional free-text note). The next session save carries
    it; /resume restores it and prints it in the banner line."""
    note = user.split(None, 1)[1].strip() if " " in user else ""
    snap = {
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": (_git(["status", "--short"]) or "").splitlines()[:40],
        "todos": [dict(t) for t in st._TODOS],
        "note": note,
        "last_user": next(
            (m.get("content", "")[:200] for m in reversed(state.messages)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            ""),
    }
    st._WIP = snap
    parts = []
    if snap["branch"]:
        parts.append(snap["branch"])
    if snap["dirty"]:
        parts.append(f"{len(snap['dirty'])} dirty file(s)")
    if st._TODOS:
        open_n = sum(1 for t in st._TODOS if t.get("status") != "done")
        parts.append(f"{open_n}/{len(st._TODOS)} todos open")
    if note:
        parts.append(f"note: {note}")
    summary = " · ".join(parts) if parts else \
        "nothing tracked (not a git repo, no todos)"
    print(f"{DIM}  ✓ work state recorded: {summary}{RESET}")
    from ._repl_state import _save_current
    _save_current(state, {"wip": snap})


def render_wip(snap: dict | None) -> str | None:
    """One-line summary of a stored wip snapshot (for /resume's banner and
    /sessions --wip listings). None when absent/empty."""
    if not isinstance(snap, dict):
        return None
    parts = []
    if snap.get("branch"):
        parts.append(str(snap["branch"]))
    dirty = snap.get("dirty") or []
    if dirty:
        parts.append(f"{len(dirty)} dirty file(s)")
    todos = snap.get("todos") or []
    if todos:
        open_n = sum(1 for t in todos if t.get("status") != "done")
        parts.append(f"{open_n}/{len(todos)} todos open")
    note = snap.get("note")
    if note:
        parts.append(f"note: {note}")
    return " · ".join(parts) if parts else None
