"""Self-retro (nx #1): turn this session's usage telemetry into durable
project memories.

`collect_friction()` reads the session's usage.jsonl (the same stream
`nx/_logging.py` writes) with plain line-by-line `json.loads` — the file is
small and a subprocess per retro would be pure overhead, so no duckdb here.
`run_retro()` distills each qualifying friction pattern (count >= 2) into one
`gotcha` memory note via `_write_memory_file`, deduped against existing
project notes by a tool+event substring scan.

Two entry points share `run_retro`:
  * `/retro` (interactive): prints the friction table and asks before writing.
  * the session-end hook, which is ON by default so the retro fires without
    the user remembering to ask; it writes silently through `retro_on_exit`,
    once-per-session, and never raises. `NX_RETRO_ON_EXIT=0` opts out.
"""
from __future__ import annotations

import json
import os
import time

from . import _state as st

# Events that mean "something in this session fought us": harness-side
# repairs, breaker trips, withheld tools, retries, surrenders — plus
# tool_call events whose outcome wasn't "ok" (folded under their tool name).
_FRICTION_EVENTS = ("repair", "breaker", "gate", "retry", "give_up")

_MIN_COUNT = 2    # a pattern needs at least this many hits to be a pattern
_MAX_PATTERNS = 3  # cap: three notes per retro is already a loud one


def _usage_path() -> str:
    """Same path nx/_logging.py's USAGE constant is built from, resolved at
    call time so a monkeypatched SESSION_HOME (tests, /resume) is honored."""
    return os.path.join(st.SESSION_HOME, "logs", "usage.jsonl")


def _iter_usage_lines(path: str):
    """Yield parsed JSON records from a usage.jsonl. A missing or unreadable
    file yields nothing (the caller reports "no usage data"); a malformed
    line is skipped, not fatal — the log is append-only best-effort data."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    yield rec
    except OSError:
        return


def _friction_key(rec: dict) -> tuple[str, str] | None:
    """(event, subject) for a friction record, or None when it isn't one.
    `tool_call` events count only when the outcome wasn't ok and the args
    are well-formed; their subject is the tool. The other events key on
    their tool when present, else error_class, else the repair kind."""
    event = rec.get("event")
    if event not in _FRICTION_EVENTS and event != "tool_call":
        return None
    data = rec.get("data")
    if not isinstance(data, dict):
        return None
    if event == "tool_call":
        if data.get("outcome") == "ok":
            return None
        return event, str(data.get("tool") or "?")
    subject = (data.get("tool") or data.get("error_class")
               or data.get("kind") or "?")
    return event, str(subject)


def _sample_detail(data: dict) -> str:
    """One-line detail for a note body: the ctx_tail's last non-empty line
    (the model's words before the failure) or a compact error field."""
    tail = data.get("ctx_tail")
    if isinstance(tail, str) and tail.strip():
        last = [ln.strip() for ln in tail.strip().splitlines() if ln.strip()]
        if last:
            return last[-1][:200]
    for k in ("error", "reason", "what", "kind", "error_kind"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:200]
    return ""


def collect_friction(session_id: str | None = None,
                     path: str | None = None) -> list[dict]:
    """Group this session's friction events by (event, tool-or-error_class)
    and return the qualifying patterns: count >= _MIN_COUNT, sorted by count
    descending, capped at _MAX_PATTERNS. Each pattern is
    {event, subject, count, detail} where detail is a sample ctx_tail /
    error line from the grouped events. `session_id` filters records by the
    `session` field when given (records without one still count — older
    emitters); `path` overrides the usage.jsonl location (tests)."""
    # Drain the async usage writer first so events logged this very turn
    # (the session-end hook's whole point) are on disk before we read.
    try:
        from ._logging import _ulog
        _ulog.flush()
    except Exception:  # noqa: BLE001 — best-effort; the read below is the truth
        pass
    usage_path = path or _usage_path()
    groups: dict[tuple[str, str], dict] = {}
    for rec in _iter_usage_lines(usage_path):
        data = rec.get("data")
        if not isinstance(data, dict):
            data = {}
        sid = rec.get("session") or data.get("session")
        if session_id and sid and sid != session_id:
            continue
        key = _friction_key(rec)
        if key is None:
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = {"event": key[0], "subject": key[1], "count": 1,
                           "detail": _sample_detail(data)}
        else:
            g["count"] += 1
            if not g["detail"]:
                g["detail"] = _sample_detail(data)
    patterns = [g for g in groups.values() if g["count"] >= _MIN_COUNT]
    patterns.sort(key=lambda g: (-g["count"], g["event"], g["subject"]))
    return patterns[:_MAX_PATTERNS]


def _already_noted(pattern: dict, notes: list) -> bool:
    """True when an existing note mentions this pattern's tool AND event
    (substring check over title+content — coarse on purpose: a false-skip
    is cheap, a duplicate note cluster is not)."""
    tool = str(pattern["subject"]).lower()
    event = str(pattern["event"]).lower()
    for _name, _path, _mtime, title, content in notes:
        blob = f"{title}\n{content}".lower()
        if tool in blob and event in blob:
            return True
    return False


def _note_title(pattern: dict) -> str:
    return (f"Retro {time.strftime('%Y-%m-%d')}: repeated {pattern['event']} "
            f"on {pattern['subject']}")


def _note_body(pattern: dict, session_id: str | None) -> str:
    lines = [
        f"Repeated friction pattern this session: `{pattern['event']}` on "
        f"`{pattern['subject']}` occurred {pattern['count']} times.",
    ]
    if pattern.get("detail"):
        lines.append(f"Sample detail: {pattern['detail']}")
    if session_id:
        lines.append(f"Session: {session_id}")
    lines.append("Recorded by self-retro from usage.jsonl — check whether "
                 "the harness or the workflow around this tool needs a fix.")
    return "\n\n".join(lines)


def _existing_project_notes() -> list:
    from ._memory_store import _iter_notes, _project_memories_dir
    pdir = _project_memories_dir(getattr(st, "PROJECT_ROOT", None))
    if not pdir:
        return []
    return _iter_notes(pdir)


def _confirm_retro(title: str, body: str) -> bool:
    """The y/n gate for a manual /retro note — same pattern as the memory
    writer's own confirm (through the top-level `nx._confirm`, which tests
    monkeypatch)."""
    import nx as _m
    return bool(_m._confirm(f"write memory (retro): {title}", preview=body))


def run_retro(session_id: str | None = None, interactive: bool = True) -> str:
    """Summarize the session's friction and persist one gotcha note per
    qualifying pattern (deduped against existing project notes). Returns a
    one-line human summary. `interactive=True` (manual /retro) prints the
    table and requires confirmation before writing; the session-end hook
    passes False and writes silently."""
    from ._memory_store import _write_memory_file

    usage_path = _usage_path()
    try:
        has_data = os.path.getsize(usage_path) > 0
    except OSError:
        has_data = False
    if not has_data:
        return "retro: no usage data"

    patterns = collect_friction(session_id)
    if not patterns:
        return "retro: no significant friction"

    if interactive:
        from ._ansi import DIM, RESET
        print(f"{DIM}  friction patterns this session:{RESET}")
        for p in patterns:
            detail = f" — {p['detail']}" if p["detail"] else ""
            print(f"  {p['count']:>3}x {p['event']} on {p['subject']}"
                  f"{DIM}{detail}{RESET}")

    notes = _existing_project_notes()
    titles: list[str] = []
    skipped = 0
    for p in patterns:
        if _already_noted(p, notes):
            skipped += 1
            continue
        title = _note_title(p)
        body = _note_body(p, session_id)
        if interactive and not _confirm_retro(title, body):
            continue
        written = _write_memory_file(title, body, scope="project",
                                     kind="gotcha", gate=False)
        if written is not None:
            titles.append(title)

    if not titles:
        if skipped:
            return (f"retro: {len(patterns)} friction patterns, "
                    f"{skipped} already noted — nothing new written")
        return f"retro: {len(patterns)} friction patterns, 0 notes written"
    return (f"retro: {len(patterns)} friction patterns, "
            f"{len(titles)} notes written: " + "; ".join(titles))


def retro_on_exit(session_id: str | None) -> None:
    """Session-end hook entry: runs the non-interactive retro at most once
    per session id (module-level `_done`), never raises.

    ON by default: the retro must fire without the user remembering to ask,
    so only an explicit `NX_RETRO_ON_EXIT=0` silences it (opt-out). The
    module-level `_done` set makes the once-per-session guard survive both
    the /quit path and the atexit path in one process."""
    try:
        if os.environ.get("NX_RETRO_ON_EXIT", "1") == "0":
            return
        key = session_id or ""
        if key in _done:
            return
        _done.add(key)
        run_retro(session_id, interactive=False)
    except Exception:  # noqa: BLE001 — a retro must never block an exit
        pass


_done: set[str] = set()


def _reset_retro_done() -> None:
    """Test hook: forget which session ids already ran the exit retro."""
    _done.clear()
