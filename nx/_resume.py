"""One resume path, shared by `nx --resume` and `/resume` (0.40, ROADMAP 1.4).

Both entry points used to load a session their own way, and the split was user-
visible: `/resume` handed the API store-form `image_ref` parts (the CLI path
expanded them), never re-applied provider routing, and neither path ever
restored the `cwd` the session was recorded with — while `_ReplState` had been
saving it since the beginning. Resume state now lives in exactly one place:

    resume_session(session_id, state)   # the only supported way to enter a session

Anything a resumed session needs in order to be request-ready AND to keep the
prompt cache intact belongs here. The presentation layers (`_repl._repl()` for
the CLI, `_sessions_cmd._cmd_resume_handler()` for the slash command) keep only
their own pre/post concerns (persisting the outgoing session, minting a fresh
id when there is nothing to resume).
"""
from __future__ import annotations

import os
from typing import Any

from ._ansi import BOLD, DIM, RESET, C
from . import _state as st
from ._env import _env_int


def resumed_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """A stored session's `messages`, in the shape the next request needs.

    Three fixups, all of them things one of the two old resume paths forgot:

    • store → live image form: `_write_session` persists `image_ref` sidecar
      references, and the API rejects that part type outright (400), so a
      resumed transcript is not request-ready until they are expanded. A
      payload missing from the image store degrades to a text note.
    • the saved system prompt is dropped and the live `SYSTEM` prepended, so a
      resumed session picks up SYSTEM/tool edits.
    • the memory-awareness and delegation notes are added/refreshed. Both are
      no-ops when the transcript already carries an up-to-date note, which is
      what keeps a normal resume byte-identical (prompt cache intact).
    """
    from ._sessions_fs import _deserialize_image_parts
    from ._memory_cmd import _refresh_memory_note
    from ._subagent import _add_delegation_note_if_absent
    from ._prefix import is_prefix_note, pin_prefix, prefix_notes
    import nx as _m

    msgs = [dict(m) if isinstance(m, dict) else m
            for m in (data.get("messages") or []) if isinstance(m, dict)]
    _deserialize_image_parts(msgs)
    msgs = [m for m in msgs if m.get("role") != "system"]
    # The canonical prefix block (PERF.md PF-15): one producer, shared with a
    # fresh session. A resumed session used to rebuild only the memory and
    # delegation notes, so it could lack the AGENTS.md and orientation notes a
    # fresh session carries — different leading bytes, i.e. a prompt-prefix
    # cache miss for the whole prompt. Any saved copy of a block note is dropped
    # (wherever it sits) and the live block re-inserted right after SYSTEM; the
    # pin then records it as the bytes later turns must preserve.
    msgs = [{"role": "system", "content": _m.SYSTEM}] \
        + prefix_notes() + [m for m in msgs if not is_prefix_note(m)]
    # Memory / delegation notes drift: the transcript's snapshot may predate the
    # current store. Both refresh append-only, so a resume over an untouched
    # store stays byte-identical (prompt cache intact) — see PF-12. (No-ops when
    # the block above already carries the up-to-date note.)
    _refresh_memory_note(msgs)
    _add_delegation_note_if_absent(msgs)
    pin_prefix(msgs)
    return msgs


def restore_endpoint(data: dict[str, Any]) -> str | None:
    """Re-select the source (+ model override + provider routing) the session
    was recorded on. Returns the source name on success, else None.

    Provider routing is part of this: `/source <name> <model>` and `/provider`
    both mutate the active Source, so a resume that only restored the source
    name would land on the same endpoint with different upstream routing than
    the transcript's calls actually used."""
    from ._sources import restore_source, _set_provider_order

    name = data.get("source")
    if not name or not restore_source(name, model=data.get("model")):
        return None
    saved_prov = data.get("provider")
    if isinstance(saved_prov, list) and st.ACTIVE and name in st.SOURCES:
        _set_provider_order(st.SOURCES[name], saved_prov)
    return name


def restore_cwd(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the session's `cwd`, changing into it when it still exists.

    Returns `(restored_cwd, warning)`. Opt out with NX_RESUME_CWD=0 (the
    session's relative paths, allow-rules and PROJECT_ROOT were all recorded
    against that directory, so restoring is the correct default; a user who
    resumes from a worktree on purpose can turn it off).

    `st.PROJECT_ROOT` is recomputed after the chdir: it is detected once at
    import (nx/_risk.py) and every path fence, allow-rule scope and memory
    scope keys off it, so a resume that changed directory without refreshing it
    would judge the new tree's files against the old root."""
    want = data.get("cwd")
    if not isinstance(want, str) or not want:
        return None, None
    if _env_int("NX_RESUME_CWD", 1) == 0:
        return None, None
    if not os.path.isdir(want):
        return None, f"session cwd {want!r} no longer exists — staying in {os.getcwd()!r}"
    if os.path.realpath(want) == os.path.realpath(os.getcwd()):
        return None, None
    try:
        os.chdir(want)
    except OSError as e:
        return None, f"couldn't change to {want!r}: {e}"
    from ._risk import _detect_project_root
    st.PROJECT_ROOT = _detect_project_root()
    return want, None


def _apply_resumed_nudge_flags(messages: list[dict[str, Any]]) -> None:
    """Suppress the session-level nudges a resumed transcript has already had.
    Without this, resuming a session that backgrounded a job or delegated
    re-sends guidance the transcript already contains."""
    import nx as _m

    st._SESSION_HAS_BACKGROUNDED = _m._messages_show_prior_backgrounding(messages)
    st._MULTI_EDIT_NUDGED = _m._messages_show_prior_multi_edit_nudge(messages)
    st._DELEGATION_NUDGED = _m._messages_show_prior_delegation_nudge(messages)


def resume_session(session_id: str, state: Any, *, show_transcript: bool = True,
                   header: str = "resumed session",
                   announce_missing: bool = True) -> bool:
    """Load `session_id` into `state` — the single resume implementation.

    On success: `state.messages` holds the live-form transcript, `state.session_id`
    is the resumed id, `state.session_dirty` is False (just loaded, in sync with
    disk), the endpoint + cwd match what the session recorded, and the
    cumulative token recorder has been reloaded from its saved totals so the
    footer keeps climbing instead of resetting to zero. Returns False when the
    session can't be read; `state` is left untouched. `announce_missing=False`
    keeps that failure silent — `--session <id>` legitimately names a session
    that does not exist yet (it creates one), so warning there would be noise.
    """
    from ._repl_state import _fresh_metrics, _reload_metrics_totals
    import nx as _m

    data = _m._load_session(session_id)
    if not data or not isinstance(data.get("messages"), list):
        if announce_missing:
            print(f"{C.WARNING}  ✗ no saved session found for {session_id!r}{RESET}")
        return False

    messages = resumed_messages(data)
    n_msgs = len([m for m in messages if m.get("role") != "system"])
    src = restore_endpoint(data)
    if src:
        print(f"{BOLD}{C.WARNING}  → source {src}{RESET} {DIM}({st.MODEL}){RESET}")
    cwd, cwd_warning = restore_cwd(data)
    if cwd:
        print(f"{DIM}  → cwd {cwd}{RESET}")
    if cwd_warning:
        print(f"{C.WARNING}  ⚠ {cwd_warning}{RESET}")

    _apply_resumed_nudge_flags(messages)

    state.messages[:] = messages
    state.session_id = session_id
    st.SESSION_ID = session_id  # mirror: the undo store keys per-session on it
    state.session_dirty = False

    title = data.get("title") or "(untitled)"
    print(f"{DIM}  ↻ {header} {session_id} ({title!r}, {n_msgs} messages){RESET}")
    # Work-state snapshot (0.94.0): the stored /wip summary, if any.
    try:
        from ._wip import render_wip
        wip_line = render_wip(data.get("wip"))
    except Exception:  # noqa: BLE001 — presentation only
        wip_line = None
    if wip_line:
        print(f"{DIM}  ⚑ wip: {wip_line}{RESET}")
    if show_transcript and n_msgs:
        from ._sessions_cmd import _print_transcript
        print()
        print(f"{DIM}  ── transcript ──{RESET}")
        _print_transcript(messages)
        print(f"{DIM}  ──────────────{RESET}")
    print()

    # Cumulative tokens + todos/plan ride in the session JSON: rebuild the
    # recorder for THIS id and fold the saved totals back in.
    st._METRICS = _fresh_metrics(session_id)
    _reload_metrics_totals(session_id)
    return True
