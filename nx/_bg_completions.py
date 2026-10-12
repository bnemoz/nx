"""Background-job completion scanning: the one-time [Runtime note] for tracked
bg jobs that finished since the model last looked.

Split out of `_dispatch.py` (which keeps run_tool — the first-tool-of-turn
hook that calls the sweep). Shared by the wake path (`_autowake.wake_due`),
whose text-only turns otherwise never run it."""
from __future__ import annotations

import os
import threading

from . import _state as st
from . import _notices

# The scan runs from both the model turn (run_tool's first-tool hook) and the
# wake poller thread; the seen-set is read-modify-rewritten, so both sides
# serialize on this (also held by _bash_jobs._mark_bg_notice_seen).
_BG_SCAN_LOCK = threading.Lock()


def _scan_background_completions(jobs=None) -> str | None:
    """One-time notice for tracked background jobs that finished since the
    model last looked (checked on the first tool call of each turn).

    The model must otherwise REMEMBER to wait_background/read_file a job it
    backgrounded turns ago; a build that failed 20 minutes ago doesn't
    announce itself. Returns the note text (or None): lists finished jobs with
    exit codes once, then marks them seen — no repeated noise.

    *jobs* is an already-computed list_bg_jobs() result a caller can share
    (wake_due scans the map once and threads it through the whole sweep);
    None scans it here."""
    from ._bash_jobs import list_bg_jobs, _read_bg_exit_status
    from ._verify import _is_verify_job  # verify jobs report via _stage_verify_outcome instead
    with _BG_SCAN_LOCK:
        try:
            if jobs is None:
                jobs = list_bg_jobs()
        except Exception:
            return None
        mine = os.getpid()
        seen = _notices.get("_BG_NOTICE_SEEN")
        if seen is None:
            seen = set(getattr(st, "_BG_NOTICE_BASELINE", set()) or set())
        notes = []
        for j in jobs:
            try:
                pid = str(j.get("pid", ""))
                if not pid:
                    continue
                # only jobs this nx process launched — never another session's strays
                if j.get("owner_pid") != mine:
                    continue
                if _is_verify_job(j):
                    continue  # NX_VERIFY_CMD outcome has its own dedicated note path
                if j.get("alive"):
                    continue  # still running — the existing [background] flow covers it
                if pid in seen:
                    continue
                # A dead job whose log is already gone has nothing left to
                # announce: its output reached the model another way (run_bash
                # finished inside its own poll window, wait_background, or a
                # read_file of the log) and _finish_bg_job deleted the log.
                # "pid N exited (code unknown)" with no tail is pure noise —
                # and each such orphan woke a pointless auto-continuation turn
                # — so mark it consumed silently (0.37.4).
                if not j.get("log_path") or not os.path.exists(j["log_path"]):
                    seen.add(pid)
                    continue
                seen.add(pid)
                code = _read_bg_exit_status(pid)[1]
                tail = (j.get("tail") or "").strip()[:80]
                code_s = "unknown" if code is None else str(code)
                notes.append(f"  pid {pid} exited (code {code_s})" + (f" — tail: {tail}" if tail else ""))
            except Exception:
                # One malformed job must not discard the whole sweep: the
                # marks made so far are committed below either way, so a probe
                # that raises can never make the same pid re-announce forever.
                continue
        _notices.set_("_BG_NOTICE_SEEN", seen)
    if not notes:
        return None
    return ("[Runtime note: background job(s) you started have finished:\n"
            + "\n".join(notes)
            + "\nCheck their output with read_file(log) or wait_background before relying on them.]")


def _stage_bg_note(note: str) -> None:
    """Stage the sweep's note with the merge+dedup discipline of the old
    monolithic _run_bg_completion_scan (a staged note can still be unflushed
    here — the poller's own sweep may have staged one before this turn's
    first tool call; assigning would drop the earlier job's exit line)."""
    prev = _notices.get("_PENDING_BG_COMPLETION_NOTICE")
    if prev:
        # Dedup (0.37.4): the poller thread and this turn can both sweep
        # the same job in a race — _BG_SCAN_LOCK marks it seen once, but
        # each sweep may have already staged its own copy of the note.
        fresh = [ln for ln in note.splitlines()
                 if ln.strip().startswith("pid ") and ln not in prev]
        if fresh:
            note = f"{prev}\n" + "\n".join(fresh)
        # else: every line already staged — keep prev untouched
        else:
            note = prev
    _notices.set_("_PENDING_BG_COMPLETION_NOTICE", note)


def _stage_verify_outcome_safe(jobs=None) -> None:
    """The verify half of the sweep, split out so run_tool()'s inline call and
    _run_bg_completion_scan share one implementation (NX_VERIFY_CMD job just
    finished? Its outcome is staged as its own pending notice)."""
    from ._verify import _stage_verify_outcome
    try:
        _stage_verify_outcome(jobs=jobs)
    except Exception:
        pass


def _stage_subagent_completions_safe() -> None:
    """The sub-agent half of the sweep: async agents finished since the last
    look get their [Runtime note] staged (idempotent per job id)."""
    from ._subagent import _stage_subagent_completions
    try:
        _stage_subagent_completions()
    except Exception:
        pass


def _run_bg_completion_scan(jobs=None) -> None:
    """The completion sweep run_tool() runs on the first tool call of a turn:
    scan finished bg jobs, NX_VERIFY_CMD outcomes, and finished async agents,
    staging each as its one-time [Runtime note]. The wake path shares it —
    see _autowake.wake_due for why a wake turn must not fire on an empty
    sweep.

    *jobs* lets a caller (wake_due) pass the list_bg_jobs() result it already
    paid for, so one keystroke scans the pid->log map once, not three times."""
    note = _scan_background_completions(jobs=jobs)
    if note:
        _stage_bg_note(note)
    _stage_verify_outcome_safe(jobs=jobs)
    _stage_subagent_completions_safe()
    _stage_orphan_agent_intents_safe()


def _stage_orphan_agent_intents_safe() -> None:
    """The crash-residue half of the sweep (0.147.2, effect sandwich): intent
    records for async agents whose owner process died mid-run are announced
    once as a [Runtime note], then deleted. A live process's own agents are
    skipped (their intents are cleared at finalize, and liveness is checked
    before announcing anything)."""
    from ._subagent import _scan_orphan_agent_intents
    try:
        lines = _scan_orphan_agent_intents()
    except Exception:  # noqa: BLE001 — the sweep must never break the turn
        return
    if lines:
        _stage_bg_note(
            "[Runtime note: sub-agent(s) from a crashed nx session were killed "
            "mid-run and never finished:\n" + "\n".join(lines) + "\n"
            "Re-spawn deliberately if the work still matters.]")
