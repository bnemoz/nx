"""Self-triggering REPL turns (0.36.0): wake the model when a background job
finishes.

Before this, the runtime notes that announce finished background work
(bg-bash completions, async sub-agent answers, NX_VERIFY_CMD outcomes) were
only *staged* while the model was idle — they flush into the transcript on
the first tool call of a turn, so they sat parked until the user happened to
type something. A sub-agent that finished while the REPL sat at the prompt
left the model (and the user) waiting on work that was already done.

Two halves:

- a poller thread, armed the first time this session backgrounds a run_bash
  job or launches an async sub-agent. It sleeps on a Condition (not a busy
  loop) and sets st._WAKE_DUE as soon as any pending-notice slot is non-empty
  or a sweep finds a finished-but-unseen job — exactly the condition
  run_tool()'s first-tool-of-turn scan would have staged a notice for.

- a wake-aware prompt: read_multiline()'s editor accepts wake=True and
  returns the WAKE_INPUT sentinel when st._WAKE_DUE flips while the user
  hovers at the prompt. The REPL translates that sentinel into a synthetic
  "background work finished — continue" user turn. Typed-but-unsent text is
  stashed in st._WAKE_PREFILL and restored after the wake turn, so the wake
  never eats half a message.

NX_WAKE_PROMPT=0 turns the whole mechanism off (the notes then wait for the
next real user turn, as before).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from ._env import _env_float
from . import _state as st

# Sentinel the chatbox editor returns when the wake poller fires while the
# user sits at an empty prompt. Identity-checked (never ==-compared) so a
# user who literally types this string is still treated as normal input.
WAKE_INPUT = object()

# st slot that carries the user's typed-but-unsent buffer across a wake turn
# (see read_multiline's wake path in _chatbox.py).
WAKE_PREFILL_SLOT = "_WAKE_PREFILL"

# Synthetic user instruction the REPL appends for a wake turn. Terse on
# purpose: it is sent every wake, and the real payload (the finished agent's
# answer head + report path, the bg job's exit line) arrives right after it
# via _flush_pending_background_notice.
WAKE_INSTRUCTION = (
    "[Auto-continuation: one of your background jobs (a detached command or an "
    "async sub-agent) finished while you were idle — the REPL woke this turn "
    "for you. The completion note is below. Act on it now: collect the "
    "result, do the work that was waiting on it, and stop when done. If "
    "nothing needs doing, say so in one line.]"
)

_POLLER: Optional["_WakePoller"] = None  # the single poller, created lazily


def _wake_enabled() -> bool:
    """NX_WAKE_PROMPT gate. Default on; only an explicit 0/false/off disables."""
    return os.environ.get("NX_WAKE_PROMPT", "1").strip().lower() not in ("0", "false", "off")


def arm_wake() -> None:
    """Call when this session starts (or could start) background work that
    only a later turn can observe: the first backgrounded run_bash, the first
    async spawn_agent. Idempotent; no-ops when NX_WAKE_PROMPT=0."""
    if not _wake_enabled():
        return
    st._WAKE_ARMED = True
    ensure_poller()


# --- what counts as "a wake is due" ------------------------------------------

def _pending_notice_slots() -> tuple[str, ...]:
    return ("_PENDING_BACKGROUND_NOTICE", "_PENDING_BG_COMPLETION_NOTICE",
            "_PENDING_SUBAGENT_NOTICE", "_PENDING_MULTI_EDIT_NOTICE",
            "_PENDING_VERIFY_NOTICE", "_PENDING_DELEGATION_NOTICE")


def _finished_unstaged_subagents() -> list[str]:
    """Async agents whose answer exists but whose note isn't staged yet."""
    try:
        from ._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
        with _ASYNC_LOCK:
            return [jid for jid, rec in _ASYNC_AGENTS.items()
                    if rec.get("status") != "running" and not rec.get("announced")]
    except Exception:
        return []


def _finished_unseen_bg_jobs() -> list[str]:
    """Finished bg-bash pids the completion sweep hasn't announced yet.

    Mirrors _dispatch._scan_background_completions' filter (this process's
    jobs only, verify jobs excluded) but read-only: it never mutates
    _BG_NOTICE_SEEN, so the real sweep still announces them later."""
    try:
        from ._bash_jobs import list_bg_jobs
        from ._verify import _is_verify_job
        mine = os.getpid()
        seen = getattr(st, "_BG_NOTICE_SEEN", None)
        if seen is None:
            seen = set(getattr(st, "_BG_NOTICE_BASELINE", set()) or set())
        out = []
        for j in list_bg_jobs():
            pid = str(j.get("pid", ""))
            if not pid or pid in seen or j.get("alive"):
                continue
            if j.get("owner_pid") != mine or _is_verify_job(j):
                continue
            out.append(pid)
        return out
    except Exception:
        return []


def wake_due() -> bool:
    """True when the REPL should start a turn by itself: a completion note is
    staged (or stageable) and no turn is currently running.

    Conservative on every axis: not while a turn/sub-agent/batch loop owns the
    machinery, not while a sub-agent is still running (its own finish will
    wake later), and never before the session has backgrounded anything."""
    if not _wake_enabled() or not getattr(st, "_WAKE_ARMED", False):
        return False
    if st._WAKE_DUE:
        return True
    if getattr(st, "_TURN_ACTIVE", False):
        return False
    try:
        from ._subagent import _subagent_ctx, _async_running
        from ._batch import _batch_active
        if _subagent_ctx() is not None or _batch_active() or _async_running():
            return False
    except Exception:
        return False
    if any(getattr(st, s, None) for s in _pending_notice_slots()):
        return True
    return bool(_finished_unstaged_subagents() or _finished_unseen_bg_jobs())


# --- the poller ---------------------------------------------------------------

class _WakePoller(threading.Thread):
    """Daemon thread: sleeps on a Condition with a short timeout and raises
    st._WAKE_DUE when a wake becomes due. It never starts the turn itself —
    the REPL does that when it next looks — because the REPL is busy whenever
    the state could race."""

    def __init__(self, interval: float) -> None:
        super().__init__(name="nx-wake-poller", daemon=True)
        self._interval = interval
        self._cond = threading.Condition()

    def nudge(self) -> None:
        """Hint the poller to look soon (called from arm_wake)."""
        with self._cond:
            self._cond.notify_all()

    def run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait(self._interval)
            if st._WAKE_DUE:
                continue  # already raised; the REPL clears it
            try:
                if wake_due():
                    st._WAKE_DUE = True
            except Exception:
                pass  # a wake poll must never take the REPL down


def ensure_poller() -> None:
    """Start the poller once per process. The interval is NX_WAKE_POLL_SECONDS
    (default 2.0, floor 0.2) — how long a finished job can sit unnoticed while
    the user idles at an empty prompt; tests shrink it."""
    global _POLLER
    if _POLLER is not None:
        _POLLER.nudge()
        return
    interval = max(0.2, _env_float("NX_WAKE_POLL_SECONDS", 2.0))
    _POLLER = _WakePoller(interval)
    _POLLER.start()
