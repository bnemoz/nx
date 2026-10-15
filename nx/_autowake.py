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

Since 0.36.3 the wake path also runs the completion sweep itself (the one
run_tool() runs on a turn's first tool call): a turn that backgrounded a job
and then answered in plain text never runs that scan, so its finished job
used to wake the REPL again and again while every wake turn — also text-only
— left it unseen, ad infinitum. The sweep marks what it reports seen, so the
first wake delivers the exit line and the next wake_due() is False.

NX_WAKE_PROMPT=0 turns the whole mechanism off (the notes then wait for the
next real user turn, as before).

0.37.4 hardening: short-lived run_bash commands never go through the
[background] flow at all (they finish inside the 3s poll window, output
inline), but their pid->log map outlives the log _finish_bg_job deletes —
an orphan that scanned as "pid N exited (code unknown)" once per turn and
woke a "Nothing to do" auto-continuation every time. Fix is three-sided:
run_bash/wait_background mark a job seen the moment its full output lands
in a tool result, read_file on a finished job's log marks it seen too, and
the wake sweep skips dead jobs with no log left to read.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from ._env import _env_float, _env_int
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

# The wake-turn instruction for an unattended campaign (0.121.0): pending
# todos AND this session has delegated implementation to write-tier agents.
# The plain WAKE_INSTRUCTION's "stop when done" is right for a wake that
# answers a question; mid-campaign it read as "wrap up after this delivery"
# and stalled multi-phase plans until the user hand-fed the next "continue".
CAMPAIGN_WAKE_INSTRUCTION = (
    "[Auto-continuation: background work finished while the user was away — "
    "unattended wake {n} of {m}. The completion note is below. You are the "
    "coordinator of your todo plan, not its implementer: briefly review the "
    "finished agent's result (read the full report only if the head is not "
    "enough), update todo_write — mark what completed, set the next item "
    "in_progress — then spawn the next phase as a BACKGROUND write/implement-"
    "tier agent (tier='advisor' if the phase has real design forks) with the "
    "settled design baked into its instruction, and end your turn to let it "
    "run. Keep your own context light: file edits and test runs belong to the "
    "agents; you review and merge. Stop spawning — finish current work, "
    "verify, and leave a summary of what's done, what's untested, and what "
    "needs the user — when the todo list is complete, when a phase fails in "
    "a way another agent cannot fix, or when you are within 3 wakes of the "
    "limit. A typed user message resets the streak.]"
)


def _wake_streak_limit() -> int:
    """Cap on consecutive unattended wake turns (user typed nothing since the
    chain started). NX_WAKE_STREAK_LIMIT, default 25; clamped to >= 1 so the
    guard can't be configured off by accident (NX_WAKE_PROMPT=0 remains the
    off switch for wakes entirely)."""
    return max(1, _env_int("NX_WAKE_STREAK_LIMIT", 25))


def _campaign_pending() -> bool:
    """True when a todo plan is mid-flight AND this session has delegated
    implementation to write-tier agents (st._SPAWNED_WRITE_TIER, set at
    spawn time). Both are required: todos alone are the model's own working
    list in an interactive session (the plain wake instruction stays), and
    the write-tier flag alone is a one-off delegated edit, not a campaign."""
    todos = getattr(st, "_TODOS", None)
    if not todos or not any(t.get("status") != "done" for t in todos):
        return False
    return bool(getattr(st, "_SPAWNED_WRITE_TIER", False))


def wake_instruction() -> str:
    """The synthetic user message for THIS wake turn: the campaign variant
    when a plan is mid-flight, the plain one otherwise. The streak is
    incremented by the REPL before this is called, so n is this wake's
    1-based index; stamping both n and m lets the model see its own runway
    without counting turns itself."""
    if _campaign_pending():
        return CAMPAIGN_WAKE_INSTRUCTION.format(
            n=max(1, getattr(st, "_WAKE_STREAK", 1)),
            m=_wake_streak_limit())
    return WAKE_INSTRUCTION


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

# Slots staged as a [Runtime note] but which must never themselves cause a
# wake. Each entry needs a one-line reason: these notes ride the next REAL
# prompt; waking for them would either loop (the note re-stages itself) or
# nag with nothing actionable.
_NON_WAKE_NOTICE_SLOTS: frozenset[str] = frozenset({
    # placeholder: Phase B adds _PENDING_WAKE_HOLD_NOTICE here — a hold notice
    # must not satisfy _any_pending_notice() (that would wake forever).
})


def _pending_notice_slots() -> tuple[str, ...]:
    """Every runtime-note slot that can wake the REPL, derived not hand-listed
    (0.159.3): _NOTICE_DEFAULTS keys plus every _PENDING_* attr on st (the
    same dir() oracle nx_state_reset_body uses) plus _PENDING_BACKGROUND_NOTICE,
    which has no _state attr — it is a plain module global staged by run_tool
    (dispatch) and read below. _NON_WAKE_NOTICE_SLOTS names slots that must
    deliberately NOT wake (each with its reason); a notice type added to
    _NOTICE_DEFAULTS now wakes the REPL without a code change here."""
    slots = (set(st._NOTICE_DEFAULTS)
             | {s for s in dir(st) if s.startswith("_PENDING_")}
             | {"_PENDING_BACKGROUND_NOTICE"})
    # _BG_SCAN_DONE / _BG_NOTICE_SEEN ride _NOTICE_DEFAULTS but are not
    # notices (scan latches), so they never wake.
    return tuple(sorted(slots - _NON_WAKE_NOTICE_SLOTS
                        - {"_BG_SCAN_DONE", "_BG_NOTICE_SEEN"}))


def _any_pending_notice() -> bool:
    """True when any runtime-note slot holds staged text. The slots routed by
    nx._notices (swarm P2) are read through it; _PENDING_BACKGROUND_NOTICE
    (staged inline by run_bash on the caller's own thread) stays a plain
    module global. getattr-default, not bare st.X: a conftest rewind can drop
    names from _state's dict. _NON_WAKE_NOTICE_SLOTS are excluded in this
    predicate, not at one call site: this function also feeds the REPL
    post-turn flag raise and the Telegram mirror, so a caller-side exclusion
    would leak (Phase B's hold-notice anti-loop depends on it)."""
    from . import _notices
    for s in _pending_notice_slots():
        if s == "_PENDING_BACKGROUND_NOTICE":
            if getattr(st, s, None):
                return True
        elif _notices.get(s):
            return True
    return False


def _finished_unstaged_subagents() -> list[str]:
    """Async agents whose answer exists but whose note isn't staged yet."""
    try:
        from ._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
        with _ASYNC_LOCK:
            return [jid for jid, rec in _ASYNC_AGENTS.items()
                    if rec.get("status") != "running" and not rec.get("announced")]
    except Exception:
        return []


def _finished_unseen_bg_jobs(jobs=None) -> list[str]:
    """Finished bg-bash pids the completion sweep hasn't announced yet.

    Mirrors _dispatch._scan_background_completions' filter (this process's
    jobs only, verify jobs excluded) but read-only: it never mutates
    _BG_NOTICE_SEEN, so the real sweep still announces them later.

    *jobs* lets a caller that already ran list_bg_jobs() (wake_due) share that
    single scan instead of paying a second listdir+json+tail pass."""
    try:
        from ._bash_jobs import list_bg_jobs
        from ._verify import _is_verify_job
        if jobs is None:
            jobs = list_bg_jobs()
        mine = os.getpid()
        seen = getattr(st, "_BG_NOTICE_SEEN", None)
        if seen is None:
            seen = set(getattr(st, "_BG_NOTICE_BASELINE", set()) or set())
        out = []
        for j in jobs:
            pid = str(j.get("pid", ""))
            if not pid or pid in seen or j.get("alive"):
                continue
            if j.get("owner_pid") != mine or _is_verify_job(j):
                continue
            # A dead job whose log is already gone has nothing left to
            # announce: nx delivered its output inline (run_bash finished in
            # its own poll window) or via wait_background/read_file, and
            # _finish_bg_job deleted the log. Announcing it here is what made
            # every short bash command wake the next turn with
            # "pid N exited (code unknown)" and a "Nothing to do" reply —
            # one orphan map per bash call, one bogus wake each (0.37.4).
            if not j.get("log_path") or not os.path.exists(j["log_path"]):
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
    wake later), and never before the session has backgrounded anything.

    With nothing staged it runs the same completion sweep run_tool()'s
    first-tool-of-turn hook runs (0.36.3) and re-checks the slots. That scan
    marks what it reports seen, so a job whose containing turn died as plain
    text — no tool call, scan never ran — is announced by the wake turn
    itself instead of re-waking forever with nothing to deliver: the wake
    turn ends text-only, nothing marks the job seen, and the poller (which
    also calls this) would find it again every 2s. A sweep that comes back
    empty therefore means "nothing left to say" — no wake."""
    if not _wake_enabled() or not getattr(st, "_WAKE_ARMED", False):
        return False
    # Chain guard (0.121.0): every other guard here is a LIVENESS guard (it
    # stops wake-loops with nothing to say); a wake turn that legitimately
    # spawns new work creates new announced-once jobs, so the chain is
    # productive-by-construction and never trips them. When nobody is at the
    # keyboard that chain is unbounded — cap consecutive unattended wakes.
    # Any user-typed turn resets the streak in _repl.py.
    if getattr(st, "_WAKE_STREAK", 0) >= _wake_streak_limit():
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
    if _any_pending_notice():
        return True
    # Nothing staged: maybe a sweep would stage one now. Run the SAME scan
    # run_tool()'s first-tool-of-turn hook runs (0.36.3): it marks what it
    # reports seen, so a job whose turn died as plain text — the scan never
    # ran because there was no tool call — is announced by the wake turn
    # itself instead of re-waking forever with nothing to deliver. A sweep
    # that comes back empty means there is genuinely nothing to say: don't
    # wake on work that has already been reported.
    # Scan the pid->log map once and share it with the sweep below: the sweep
    # and _finished_unseen_bg_jobs each ran their own list_bg_jobs() (listdir +
    # json.load per map + a log tail per job) — three full scans per keystroke
    # was a measurable share of the poll cost (PERF).
    try:
        from ._bash_jobs import list_bg_jobs
        jobs = list_bg_jobs()
    except Exception:
        jobs = None
    if not jobs:
        # No bg-pid maps at all: the sweep can only ever announce a bg job, a
        # verify outcome, or a finished async agent — and the first two cannot
        # exist without a map. Skip the expensive sweep and report on the async
        # agents directly. A scan that found no map files genuinely means
        # "nothing left to say" — no wake.
        return bool(_finished_unstaged_subagents())
    try:
        from ._dispatch import _run_bg_completion_scan
        _run_bg_completion_scan(jobs=jobs)
    except Exception:
        pass
    if _any_pending_notice():
        return True
    # Anything left here is finished-but-unstaged work the sweep could not
    # announce (e.g. a sub-agent record with no answer) — still wake-worthy.
    return bool(_finished_unstaged_subagents() or _finished_unseen_bg_jobs(jobs=jobs))


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
        self._stop_flag = False

    def stop(self) -> None:
        """Ask the poller to exit (tests use this instead of dropping a live
        daemon thread on the floor, which can still flip st._WAKE_DUE)."""
        self._stop_flag = True
        with self._cond:
            self._cond.notify_all()

    def nudge(self) -> None:
        """Hint the poller to look soon (called from arm_wake)."""
        with self._cond:
            self._cond.notify_all()

    def run(self) -> None:
        while not self._stop_flag:
            with self._cond:
                self._cond.wait(self._interval)
            if self._stop_flag:
                return
            if st._WAKE_DUE:
                continue  # already raised; the REPL clears it
            try:
                if wake_due() and not st._TURN_ACTIVE:
                    # Re-check _TURN_ACTIVE after wake_due() returns: the sweep
                    # it ran can take longer than the turn that just started
                    # (it listdirs + json-loads + log-tails EVERY job), and a
                    # flag decided by a pre-turn snapshot must not fire a
                    # phantom wake at the post-turn prompt (0.109.5).
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
