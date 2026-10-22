"""Shared mutable process state.

Historically all of this lived as bare module globals in `nx/__init__.py`,
mutated via `global X` from functions defined in that same file. Once those
functions were split across several `nx/_*.py` submodules, `global X` in
one of those submodules no longer reaches back into `__init__.py`'s
namespace — each module has its own globals dict. So instead every module
that reads or writes one of these names imports this module (`from . import
_state as st`) and does `st.ACTIVE = ...` / `st.ACTIVE` — one shared home,
plain attribute access, no magic.

`nx/__init__.py` proxies attribute access for these names through to
this module (see the `_StateProxyModule` class at the bottom of
`__init__.py`), so `import nx as m; m.ACTIVE` / `m.ACTIVE = x` — which
the test suite relies on extensively — keeps working unchanged and stays in
sync with whatever any submodule does via `st.ACTIVE`.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._metrics import _Metrics
    from ._openai_client import OpenAI
    from ._sources import Source
    from ._undo import Checkpoint

# --- sessions filesystem root -------------------------------------------------
# Set (and re-set on every /reload) by __init__.py's top-level assignment from
# NX_HOME. Kept here so _bash_jobs.py's background-log paths (which live under
# SESSION_HOME/bg-logs) stay in sync with a mid-test reload without a circular
# import back into __init__.py.
SESSION_HOME: str = os.path.expanduser("~/.nx")

# --- sources / active model connection --------------------------------------
SOURCES: dict[str, "Source"] = {}
SOURCE_ORDER: list[str] = []   # preserve definition order for /source listing
ACTIVE: "Source | None" = None

# `client` and `MODEL` mirror the active source; switch_source() reassigns
# both. Every function that needs them does a call-time lookup (via this
# module), so a mid-session swap is picked up instantly.
client: "OpenAI | None" = None
MODEL: str | None = None

# Bumped by switch_source() on every swap (source and/or model override).
# A background probe (capability / thinking / context-window) records the
# generation it was armed at and aborts its backoff once that number is
# history: before 0.53.2 a probe armed for a dead endpoint kept printing
# retry notices naming the abandoned model and slept out the whole curve
# while holding STARTUP_PROBE_LOCK, delaying the source the user switched
# TO. Read-only for anything but the probes themselves.
PROBE_GENERATION: int = 0


def __getattr__(name: str):
    # Lazy re-export of the API exception types: they live in _openai_client,
    # but importing that at _state's module top would (pre-0.159.0) drag the
    # ~45 ms httpx chain into every `import nx`. _state is imported by nearly
    # every submodule, so `from ._state import APIError` must stay cheap.
    if name in ("APIError", "APIConnectionError"):
        from ._openai_client import APIError, APIConnectionError  # noqa: F401
        return globals().setdefault(name, APIError if name == "APIError"
                                    else APIConnectionError)
    raise AttributeError(name)

# Updated by model_turn() on every turn that reports a prompt-token count.
# Read by _maybe_autocompress() to decide whether to silently auto-compress.
_LAST_PROMPT_TOKENS: int = 0

# The installed-loop stack for nx/_recovery.py's per-loop recovery contexts
# (1.2). Held HERE so a module reload of _recovery — or conftest's post-test
# eviction of it — can't strand an installed context or fork the resolution
# the st.* descriptors captured at class-install time.
import contextvars as _contextvars  # noqa: E402
_RECOVERY_CURRENT: _contextvars.ContextVar = _contextvars.ContextVar(
    "nx_recovery_frame", default=None)
# Fallback store for the routed recovery fields when no loop is installed (see
# nx/_recovery.py's get/set_). Held HERE — not in _recovery — for the same
# reason as _RECOVERY_CURRENT just above: conftest's between-test module
# rewind can delete nx._recovery from sys.modules and re-import it, and a
# second module object's globals would silently fork the store the
# (first-import) descriptors on this module's class still read. One home here
# keeps every generation of descriptors, install(), and get()/set_() agreed
# on where the no-loop defaults live.
_RECOVERY_DEFAULTS: dict = {}

# Terminal API-failure marker for the turn path. open_stream() latches
# "<ExceptionType>: <msg>" here when a request hard-fails (endpoint
# unreachable, HTTP transport error, API error after all fallbacks) instead of
# returning a stream; model_turn() reads it to distinguish "the request never
# happened" from a legitimately completed (possibly empty) turn, and
# _run_model_turn_loop() turns that into a {"status": "error"} summary so
# headless batch/one-shot runs report exit_code=4 instead of a clean pass.
# Cleared by model_turn() before every open and consumed on read.
_LAST_API_ERROR: str | None = None

# Context-window-overrun marker for the turn path. open_stream() latches the
# server's verbatim rejection text here when a request is refused with a
# context-length 400 (instead of latching _LAST_API_ERROR); model_turn() maps
# that to TURN_CONTEXT_OVERRUN so _run_model_turn_loop() can auto-compress and
# retry once instead of reporting a terminal error — and, worse, instead of the
# pre-latch behavior where the failed open looked like a legit empty turn and
# the long task was reported "done" with nothing done.
# Cleared by model_turn() before every open and consumed on read.
_LAST_CONTEXT_OVERRUN: str | None = None

# --- last-prompt tap (EXPLAINABLE_HARNESS_SPEC Phase 1) -----------------------
# Per-message DIGESTS of the message list the last model request was built
# from, written by _turn.py at request-build and read by the /ctx transcript
# view ("show as-sent"). Digests, never text: the as-sent view only has to say
# whether a row is `identical to live` or `elided/summarized since`, and a
# full-text snapshot of every message would be an unbounded second copy of the
# transcript. Session-lifetime like the other st.* state; drained by
# nx_state_reset. Entry shape (see record_last_prompt):
#   {"role": str, "index": int, "sha256": str, "changed": bool, "compressed": bool}
# (`compressed` is the spec's name for the same marker — one value, two keys.)
# None = no settled request recorded yet (fresh or resumed session) — the view
# renders "— no settled call recorded —" and stays on live.
_LAST_PROMPT: list[dict] | None = None


def record_last_prompt(messages: list) -> None:
    """Snapshot per-message digests of the message list a request is about to
    be built from, into `_LAST_PROMPT` (EXPLAINABLE_HARNESS_SPEC Phase 1).

    One pass over `messages`; nothing is retained but role, position, and a
    sha256 over `role` + the message's canonical text (the SAME
    `message_content_text` projection /ctx tokenizes, so the digest and the
    view's rendered text agree on what a message IS — native tool-call args
    included only insofar as that projection carries them).

    `changed` (aliased `compressed`, the spec's wording) is a cheap DIFF MARKER,
    not a compression verdict: True when the PREVIOUS snapshot had an entry at
    this index whose (role, digest) differs — i.e. this row is not what the last
    settled call sent. False on the first recorded call (no baseline to diff)
    and for appended rows with no previous entry at that index, so the view can
    point at which rows moved/elided between two settled calls without holding
    two snapshots."""
    global _LAST_PROMPT
    from hashlib import sha256
    # Lazy like __getattr__ above: _state is imported by nearly every module,
    # and a top-level `from ._tokens import ...` would ride that import chain.
    from ._tokens import message_content_text
    prev = _LAST_PROMPT
    snap: list[dict] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role") or "")
        text = message_content_text(msg) if isinstance(msg, dict) else str(msg)
        digest = sha256(role.encode("utf-8") + b"\x00" + text.encode("utf-8")).hexdigest()
        was = prev[i] if (prev is not None and i < len(prev)) else None
        changed = was is not None and (was.get("role"), was.get("sha256")) != (role, digest)
        snap.append({"role": role, "index": i, "sha256": digest,
                     "changed": changed, "compressed": changed})
    _LAST_PROMPT = snap


# --- last-compression ledger (EXPLAINABLE_HARNESS_SPEC Phase 2) ----------------
# What the last completed compress() did to the transcript, written on the
# real compression-completed path only (a failed/empty fold records nothing —
# the context is unchanged) and read by the /ctx before/after view. Counts and
# 0-based PRE-compression indexes ONLY, never message text: the ledger must not
# become an unbounded second copy of the transcript (same rule as _LAST_PROMPT
# above). Session-lifetime like the other st.* state; drained by nx_state_reset.
# None = no fold completed yet this session. Entry shape (see
# record_last_compression):
#   {"kept", "dropped", "summarized", "summary_tokens_est", "pinned",
#    "before_messages", "after_messages", "ts",
#    "ranges": [{"start", "end", "action": "dropped"|"summarized"|"pinned-kept"}]}
# with start/end half-open indexes into the pre-compression message list.
_LAST_COMPRESSION: dict | None = None


def record_last_compression(kept: int, summarized: int, summary_tokens_est: int,
                            pinned: int, before_messages: int,
                            after_messages: int, ts: float,
                            ranges: list[dict] | None = None) -> None:
    """Record what the last completed compress() did into `_LAST_COMPRESSION`
    (EXPLAINABLE_HARNESS_SPEC Phase 2).

    `kept` = messages kept verbatim in the tail, `summarized` = messages folded
    into the summary, `pinned` = foldable messages that survived verbatim
    because they were pinned, `before/after_messages` = total list lengths.
    `ranges` carries half-open [start, end) spans over the PRE-compression
    message list with an action per span; the setter derives `dropped` as
    whatever the kept/summarized/pinned partition leaves unexplained. No
    message text is retained — counts and indexes only."""
    global _LAST_COMPRESSION
    kept = int(kept)
    summarized = int(summarized)
    pinned = int(pinned)
    _LAST_COMPRESSION = {
        "kept": kept,
        "summarized": summarized,
        "dropped": max(0, before_messages - kept - summarized - pinned),
        "summary_tokens_est": int(summary_tokens_est),
        "pinned": pinned,
        "before_messages": int(before_messages),
        "after_messages": int(after_messages),
        "ts": float(ts),
        "ranges": list(ranges or []),
    }


# Last turn's prompt-cache hit fraction (0.0–1.0, or None when the server
# didn't report cache numbers). Written by _capture_last_prompt_tokens();
# read by the stats footer and /ctx so the user can see whether the KV/prompt
# cache is actually hitting. A drop to ~0 after an edit to early history is
# the tell-tale of a silently busted prompt prefix.
_LAST_CACHE_HIT: float | None = None

# --- approval gating ---------------------------------------------------------
YOLO: bool = False
APPROVE_LEVEL: str | None = None
DEFAULT_APPROVE_LEVEL: str | None = None


def toggle_yolo() -> bool:
    """Flip YOLO mode, mirroring /yolo. Shared by the Shift+Tab handlers in
    _chatbox.py's editor and _approval.py's approval prompt so both stay in
    sync instead of each re-implementing the toggle independently."""
    global YOLO, APPROVE_LEVEL
    YOLO = not YOLO
    APPROVE_LEVEL = None if YOLO else DEFAULT_APPROVE_LEVEL
    return YOLO


# --- per-agent approval posture (0.155.0) -------------------------------------
# A sub-agent's posture belongs to the agent, ON ITS THREAD — never to the
# process globals. Until 0.155.0 a read-only agent forced `st.YOLO=True /
# APPROVE_LEVEL="high"` process-wide and restored them in a `finally`; the swap
# sat BEFORE the try, so anything raising in between (spinner stop, model-target
# install) leaked YOLO=True onto the thread forever (the confirmed
# test_preview_visibility flake: a later test's _confirm() hit its YOLO fast path
# and the approval preview never printed), and the async worker's swap was
# visible to the PARENT thread for the agent's whole run.
#
# Same shape as `_METRICS_TLS` above (which replaced the identical swap-and-
# restore race for the batch recorder): threading.local + install-at-body-start
# + finally-restore + a resolver that falls back to the process globals.
# Decision reads (the approval gate, the sub-agent gates, the parallel-work
# checks) go through `_posture()`; parent-facing reads/writes (the UI, /yolo,
# /approval, Shift+Tab, CLI init, `nx eval`) stay on the globals, which remain
# the single source of truth for the PARENT's posture.
_POSTURE_TLS: threading.local = threading.local()


def _posture() -> tuple[bool, str | None]:
    """Live approval posture for THIS thread: the installed agent override when
    a sub-agent body is running, else the process globals.

    The read is deliberately LIVE (no snapshot): Shift+Tab at an approval prompt
    and /yolo //approval mutate the globals mid-turn and the parent thread must
    keep seeing the new value on its very next decision."""
    return (getattr(_POSTURE_TLS, "yolo", YOLO),
            getattr(_POSTURE_TLS, "level", APPROVE_LEVEL))


@contextmanager
def _posture_scope(force: "tuple[bool, str | None] | None"):
    """Install an agent-scoped posture override on THIS thread for the duration
    of the `with` body; restore (i.e. delete the override) in `finally`.

    `force=None` means INHERIT: nothing is installed, so `_posture()` keeps
    reading the globals live — exactly the semantics a write-tier agent needs.

    The body of a sub-agent loop is this scope's content: there is no code
    between install and the loop's own `try`, so no window exists in which an
    exception (a watchdog kill included) can skip the restore. TLS values are
    DELETED, not assigned back, so "no override installed" stays observable
    (and `_POSTURE_TLS.__dict__.clear()` in the test drain restores the same
    fresh-thread state the finally does)."""
    if force is None:
        yield
        return
    had_yolo = hasattr(_POSTURE_TLS, "yolo")
    had_level = hasattr(_POSTURE_TLS, "level")
    prev_yolo = getattr(_POSTURE_TLS, "yolo", None)
    prev_level = getattr(_POSTURE_TLS, "level", None)
    _POSTURE_TLS.yolo, _POSTURE_TLS.level = force
    try:
        yield
    finally:
        if had_yolo:
            _POSTURE_TLS.yolo = prev_yolo
        elif hasattr(_POSTURE_TLS, "yolo"):
            del _POSTURE_TLS.yolo
        if had_level:
            _POSTURE_TLS.level = prev_level
        elif hasattr(_POSTURE_TLS, "level"):
            del _POSTURE_TLS.level


# The id of the session this process is currently serving. Mirrored from the
# REPL's state.session_id at mint/resume/fork time so non-REPL code (the
# undo store, staleness bookkeeping) can key per-session state without
# threading _ReplState through. None in embedded uses/tests that never ran
# the REPL.
SESSION_ID: str | None = None

# This session's own git worktree (0.103.0, NX_SESSION_WORKTREE / `nx
# --isolate`): the path this REPL treats as the project root. None in the
# normal shared-tree mode. Set together with PROJECT_ROOT at REPL startup.
SESSION_WORKTREE: str | None = None

# Project root used by the risk classifier's path-scoping. Set once at
# startup by _detect_project_root() in _risk.py / _approval.py.
PROJECT_ROOT: str | None = None

# True only when PROJECT_ROOT came from an actual `git rev-parse
# --show-toplevel` hit, not the cwd fallback. Gates the "in_project ⇒
# medium" fast path in _local_risk_hint(): outside a git repo, PROJECT_ROOT
# is just cwd (often $HOME), so that shortcut must not apply.
PROJECT_ROOT_IS_GIT: bool = False

# Set by run_tool() while a tool body is executing; read by _confirm() to
# pause/resume the spinner around approval prompts / risk-classifier calls.
_ACTIVE_SPINNER: Any | None = None
# Set by model_turn() while the pre-first-token "thinking" spinner animates;
# read by the REPL's ^C handler to force-wipe the line a ^C racing the render
# thread would otherwise leave painted above the chatbox.
_TURN_SPINNER: Any | None = None
# Progress sink for headless frontends (0.95.0): when set, note_progress()
# forwards its text here INSTEAD of only onto the spinner. The Telegram
# gateway installs one so a sub-agent's `agent turn n/m` notes reach the
# phone mid-call. A GIL-atomic attribute write; a swallowed note is harmless
# (display-only).
_PROGRESS_SINK: Any | None = None

RISK_CONNECTION_RETRIES: int = 3
RISK_CONNECTION_RETRY_SECONDS: float = 1.0

# --- interrupt watcher --------------------------------------------------------
# (the Event objects themselves are created once in _interrupt.py; this
# module only holds the thread handle, which does get reassigned.)
_INTERRUPT_THREAD: threading.Thread | None = None

# Text the user typed *after* pressing Esc, while the watcher was still armed
# and draining stdin during the disarm handshake. Written only by the watcher
# thread (in its inner `finally`, after _INTERRUPT_RESTORED is set — so by the
# time main reads these the write has already happened-before). If the user
# hit Enter before disarm completed, _INTERRUPT_REDIRECT_SUBMITTED is True and
# model_turn() injects the text directly as the next user turn instead of
# dropping to a blank prompt. Otherwise it's partial text, consumed as a
# prefill for the next read_multiline() call in the REPL.
_INTERRUPT_REDIRECT_TEXT: str = ""
_INTERRUPT_REDIRECT_SUBMITTED: bool = False

# Set by model_turn() when an Esc-interrupt captured a partial (no Enter yet)
# redirect line. The REPL consumes-and-clears this as the `initial` prefill
# for its next read_multiline() call so the user's half-typed line reappears
# in the prompt instead of vanishing.
_INTERRUPT_REDIRECT_PREFILL: str = ""

# --- auto-continuation ("wake") state (nx/_autowake.py, 0.36.0) --------------
# armed = this session has backgrounded a job or spawned an async agent, so a
# self-triggered turn can ever be warranted. due = the poller saw a completion
# note staged (or stageable) while no turn was running; the prompt returns the
# WAKE_INPUT sentinel and the REPL starts the turn. prefill carries the user's
# typed-but-unsent buffer across that wake turn so it is never eaten.
_WAKE_ARMED: bool = False
_WAKE_DUE: bool = False
_WAKE_PREFILL: str = ""

# Consecutive unattended auto-continuation turns (0.121.0): incremented on
# each wake turn, reset to 0 by any user-typed turn. Guards the wake chain
# when a campaign advances itself while nobody is at the keyboard.
_WAKE_STREAK = 0
# True once this session has successfully spawned a write/implement/advisor
# tier sub-agent (set in _subagent.spawn_agent) — gates the campaign wake
# instruction.
_SPAWNED_WRITE_TIER: bool = False

# --- plan-stall wake (0.161.0, AUTOWAKE_SPEC Phase C) -------------------------
# A campaign can stall with nothing in flight (an agent FAILED and its note was
# already delivered, the model marked a todo in_progress and answered in text):
# every existing wake guard then correctly concludes "nothing to say", forever.
# The stall trigger fires once per plan revision after a grace period, and
# self-terminates by construction — the revision terminator plus the hold below.
# Clock = monotonic time the last done turn left a campaign with no work in
# flight (0.0 = not armed). _WAKE_STALL_REVISION is the plan revision stamped
# when a stall wake fires: at most ONE stall wake per revision. _WAKE_STALL_HOLD
# parks the trigger when such a wake changed nothing (deliberate stop).
_WAKE_STALL_AT: float = 0.0
_WAKE_STALL_REVISION: int = -1
_WAKE_STALL_HOLD: bool = False
# One-shot: set by wake_due() when the branch that fired was the stall branch,
# consumed by wake_instruction() to pick STALL_WAKE_INSTRUCTION.
_WAKE_STALL_PENDING: bool = False

# --- chain cost bound (0.162.0, AUTOWAKE_SPEC Phase D) ------------------------
# The streak cap counts WAKES; cost per wake varies ~100x. _WAKE_CHAIN_SPEND
# accumulates the context-token growth of every wake turn in the current
# unattended chain (deltas of st._LAST_PROMPT_TOKENS, captured per step by
# _turn.py's _capture_last_prompt_tokens); _WAKE_CHAIN_BASE is the reading
# snapshotted at wake start and re-snapshotted after each turn-end delta, so
# every turn in the chain adds only its own growth. wake_due() trips
# NX_WAKE_CHAIN_TOKEN_BUDGET (default 0 = off) against the spend; both reset
# with the streak on typed input.
_WAKE_CHAIN_SPEND: int = 0
_WAKE_CHAIN_BASE: int = 0

# True for the whole duration of one _run_model_turn_loop() call in the
# interactive parent. wake_due() refuses to fire mid-turn: a notice that
# stages while a turn runs is flushed by that turn's own tool-loop machinery.
_TURN_ACTIVE: bool = False

# --- background run_bash notice ----------------------------------------------
_SESSION_HAS_BACKGROUNDED: bool = False
_PENDING_BACKGROUND_NOTICE: str | None = None

# --- multi_edit batching nudge ------------------------------------------------
_CONSECUTIVE_EDITS: int = 0
_MULTI_EDIT_NUDGED: bool = False
_PENDING_MULTI_EDIT_NOTICE: str | None = None
# One-shot yolo async:false nudge (see _subagent._maybe_nudge_yolo_async):
# declared here so the st.X assignments in _dispatch.py resolve for ty.
_YOLO_ASYNC_NUDGED: bool = False
_PENDING_YOLO_ASYNC_NOTICE: str | None = None

# --- delegation (spawn_agent) nudge -------------------------------------------
# How many consecutive self-performed exploration calls (read_file/list_dir/
# search_files/duckdb) since the last spawn_agent. Crossing the threshold stages
# the DELEGATION_STREAK_NOTICE AND resets the streak, so the notice RE-ARMS every
# 2x threshold additional consecutive reads (5, then 15, then 25 cumulative).
# The observed failure mode is a model that has the tool and never uses it,
# reading the repo one file per turn forever, and a once-per-session nudge it can
# simply ignore was not enough. _DELEGATION_NUDGE_COUNT is how many times it has
# fired this session (0 = never; the first fire needs `threshold` reads and every
# later one needs `2*threshold`, which is exactly what the reset streak buys — see
# run_tool()). The resume path seeds it from a loaded transcript so a session that
# already saw a nudge still counts as having fired (see
# _resume._apply_resumed_nudge_flags).
_READ_STREAK: int = 0
_DELEGATION_NUDGE_COUNT: int = 0
# Same-failing-path loop breaker state (assigned in _dispatch.py; see the
# comment there). tuple[str, str] | None — the (tool, path) key of the last
# existence-type failure — and its consecutive run.
_PATH_FAIL_LAST: tuple[str, str] | None = None
_PATH_FAIL_RUN: int = 0
# OCC reminder slots (assigned in _sessionlock.py / _compress.py).
_PENDING_OCC_REMINDER: str | None = None
_OCC_NATIVE_COMPACTIONS: int = 0
_PENDING_DELEGATION_NOTICE: str | None = None

# --- native tool-call malformed-args circuit breaker --------------------------
# An endpoint that reliably truncates one tool's JSON args (observed: a local
# server dying right before multi_edit's `[` array literal, 7+ retries in a
# row) keeps poisoning every retry while the tool stays advertised. After
# MALFORMED_TOOL_BREAKER_LIMIT malformed-args failures naming the SAME tool,
# it is dropped from _active_tools() for the rest of the session and the
# retry nudge says to use the plain alternative instead. 0 disables.
#
# The LIMIT has NO default here on purpose (0.40): it used to be `2` while
# nx/__init__.py's env resolution assigns 3, and a module that ever read the
# value before __init__ ran would silently arm the breaker one failure early.
# `__init__.py` (NX_MALFORMED_TOOL_BREAKER) is the only owner; reading it
# before that assignment is now an AttributeError instead of a wrong number.
MALFORMED_TOOL_BREAKER_LIMIT: int
_MALFORMED_TOOL_FAILURES: dict[str, int] = {}
_MALFORMED_TOOL_BROKEN: set[str] = set()
# (source name, model) the two containers above describe. The breaker is
# EVIDENCE ABOUT AN ENDPOINT, not about the tool — a local server that truncates
# multi_edit says nothing about what a hosted endpoint does — so it is dropped
# when the (source, model) identity changes (0.40). The sentinel marks "no
# truncation observed yet on this process", and is what /toolcalls reset and
# /reset re-arm to; it is a string so it can never collide with a real key tuple.
BREAKER_UNOBSERVED: Any = "<unobserved>"
_BREAKER_ENDPOINT_KEY: Any = BREAKER_UNOBSERVED

# --- 0.30.0 recovery tally -----------------------------------------------------
# Self-healing is invisible in aggregate: nothing recorded THAT a repair
# happened, so NX_ARG_REPAIR / NX_MALFORMED_TOOL_BREAKER were untunable in
# practice. Two counters, incremented by the repair sites, shown as one footer
# field ("2 args repaired · 1 retry"), reset per turn by model_turn():
_TALLY_ARG_REPAIRS: int = 0   # args/name repairs applied this turn
_TALLY_RETRIES: int = 0       # corrective nudges / recovery cuts this turn

# --- PTC Phase 0 baseline (docs/ptc-migration-spec.md) ------------------------
# Per-turn tool-call / repair-loop counters, incremented in
# _handle_native_tool_calls, shown as the `ptc-baseline` footer field, reset
# per turn alongside the _TALLY_* counters above. Fan-out size is the N a
# _should_parallelize batch would have covered (0 = sequential turn).
_PTC_TURN_TOOL_CALLS: int = 0        # tool calls in this turn's native batch
_PTC_FANOUT_SIZE: int = 0            # batch size when the turn parallelized
_PTC_ENVELOPE_CHARS: int = 0         # sum of the raw JSON args strings
_PTC_REPAIRED_BY_CANDIDATES: int = 0 # malformed args repaired by candidates
_PTC_BRACKET_SALVAGED: int = 0       # read-tool bracket-closure salvages
_PTC_NORMALIZED: int = 0             # double-encoded / pair-list folds
_PTC_HARD_FAILED: int = 0            # unrecoverable parse failures
_PTC_BREAKER_TRIPPED: int = 0        # breaker opened for a tool this turn
_PTC_FANOUT_SCRIPTS: int = 0         # run_tools scripts with >1 stub call
                                     # (Phase 3 deprecation baseline)
# PTC Phase 2 (0.162.0): flips True once the one-time run_tools fan-out nudge
# has fired this session (see _dispatch.run_tool / _session_preamble's
# PTC_FANOUT_GUIDANCE); _PTC_SOURCE_SHIPPED flips True once the stub-module
# source rode a runtime note; _ACTIVE_PTC_PROC is the live run_tools child
# (None between calls) so Esc/teardown paths can see the script's group.
_PTC_NUDGED: bool = False
_PTC_SOURCE_SHIPPED: bool = False
_ACTIVE_PTC_PROC = None
# PTC Phase 4 (0.164.0, docs/ptc-migration-spec.md §Phase 4): the session
# fallback ladder. _PTC_BREAKER_TRIPS counts the times the malformed-args
# breaker OPENED for `run_tools` this session; on the 2nd trip the harness
# drops run_tools from the gate vector (_PTC_DISABLED) and nudges back to
# JSON. NX_PTC=1 is the hard override that forces it back on (the resolver in
# nx/_ptc_rollout.py checks the override before the ladder).
_PTC_BREAKER_TRIPS: int = 0
_PTC_DISABLED: bool = False

# --- text-only-after-stall guard ---------------------------------------------
# Set when a real tool call runs after this reasoning-stall episode's first
# nudge (see _run_model_turn_loop's TURN_TOOL branch). While it is True, a
# later text-only turn is accepted as the final summary of completed work
# rather than re-nudged — and the stall nudge itself asks for a wrap-up
# instead of "act now" (see STALL_WRAPUP_NUDGE). Reset when a fresh episode
# begins (cut_count == 0 in _handle_loop_cut).
_STALL_EPISODE_ACTED: bool = False

# --- stall/empty retry → native escalation ------------------------------------
# Set when a reasoning-only stall or an empty turn happens while the request
# went out with tool_choice="none" — which since 0.54.0 only happens when
# native tool calls were explicitly disabled (NX_NATIVE_TOOLS=0). On servers
# whose models only ever act through the native tool_calls channel (glm/qwen
# behind vLLM), that posture suppresses the model's ONLY action path: it
# thinks, emits nothing, and every "act now" nudge is unsatisfiable, so the
# ladder climbs to a forced final whose salvage prints the model's reasoning
# as the answer. A retry with tool_choice="auto" re-opens the native channel;
# a model that prefers the text protocol still works (auto permits, it doesn't
# force). Cleared when a native tool call confirms, or when a turn completes
# normally.
_STALL_FORCE_NATIVE: bool = False

# --- recovery / turn-loop tuning (env-configurable, test-monkeypatchable) ----
# Real defaults (env-resolved) are assigned onto these names at import time
# by the module that owns each setting (_turn.py, _compress.py) — this is
# just the shared home so a cross-module `st.NAME = x` mutation (tests, or
# /autocompress etc.) is visible everywhere.
REASONING_ONLY_RETRY_LIMIT: int = 3
MALFORMED_STREAM_RETRY_LIMIT: int = 2
FORCED_FINAL_MAX_TOKENS: int = 2048
# Reasoning token budget (None = no cap). Shared home for env resolution
# (nx/__init__.py) and the /budget command (nx/_repl_settings_cmds.py).
REASONING_BUDGET_TOKENS: int | None = None
EMPTY_TURN_RETRY_LIMIT: int = 3
# Auto-continue budget when a turn ends with finish_reason="length" — the model
# hit the per-request token cap (NX_MAX_TOKENS, or the server's own limit) and
# the answer was cut mid-stream. Each continuation appends the partial text as
# a real assistant turn plus a short "continue exactly where you stopped"
# user note and re-opens the stream, so a long answer arrives in seamless
# pieces instead of ending mid-sentence. 0 disables (turn just ends, as before).
MAX_CONTINUATIONS: int = 2
# Partial answer text of the most recent TURN_LENGTH_CUT, handed from
# model_turn() to _run_model_turn_loop() (which appends it as an assistant
# turn when continuing, or closes it out with the truncation marker).
_LAST_PARTIAL_TEXT: str = ""
REASONING_ONLY_CHAR_LIMIT: int = 36000
RECOVERY_TEMPERATURE: float = 1.0
RECOVERY_TOP_P: float = 0.95
MAX_MODEL_TURNS: int = 500

# --- reasoning display ---------------------------------------------------
# Whether live reasoning_content is streamed to the terminal as it arrives.
# Toggled by /reasoning (persisted to ~/.env as NX_REASONING_VISIBLE, same
# mechanism as /theme); the full text of the most recent turn's reasoning
# is always captured into _LAST_REASONING regardless, so /reasoning show
# works even right after switching into collapsed mode.
REASONING_VISIBLE: bool = os.environ.get("NX_REASONING_VISIBLE", "0") not in ("0", "false", "off")
_LAST_REASONING: str = ""

# --- thinking-separation auto-config ---------------------------------------
# Some servers (glm-5.3-flash via vLLM proxies) run a hybrid reasoning model
# with thinking ON but never surfaced as `reasoning_content` — the private
# reasoning streams as plain `content`, running straight into the visible
# answer with no separator. nx/_thinking.py probes the server at startup /
# on a source switch for a `chat_template_kwargs`-style body field that makes
# it separate them, then attaches that field to every request. "off" disables
# the probe and the attach entirely (server's stock behavior, merged or not).
_mode = os.environ.get("NX_THINKING", "auto").strip().lower()
THINKING_MODE: str = "off" if _mode in ("off", "0", "false", "no") else "auto"

# --- tool-call output display --------------------------------------------
# Whether each tool call's result preview is rendered to the terminal (the
# box header + args lines always show, so you still see *what* was called;
# only the result body is hidden). Toggled by /tooloutput (the /tools command is
# kept as a legacy alias) — persisted to ~/.env as NX_TOOL_OUTPUT_VISIBLE, same
# renders its colored list regardless of this flag. Approval previews ride the
# same flag (0.53.1): a git_commit staged diff or an edit_file before/after is
# tool output on screen too, so collapsed hides them like any other result
# body. The last tool's sanitized
# result is captured into _LAST_TOOL_OUTPUT so /tooloutput last can replay it even
# while collapsed.
TOOL_OUTPUT_VISIBLE: bool = os.environ.get("NX_TOOL_OUTPUT_VISIBLE", "0") not in ("0", "false", "off")
_LAST_TOOL_NAME: str = ""
_LAST_TOOL_OUTPUT: str = ""

# --- auto context-compression -------------------------------------------------
AUTOCOMPRESS_PERCENT: float = 85
AUTOCOMPRESS_DEFAULT: float = 85
COMPRESS_KEEP: int = 2

# --- usage metrics -------------------------------------------------------------
# `_METRICS` is the REPL's recorder (set in main() / /reset / /resume; see
# _metrics.py). Batch workers must not swap it in place: with --parallel > 1
# each worker's assignment and `finally` restore race every sibling's
# _record_usage, silently misattributing token totals. Instead a worker
# installs ITS OWN recorder on `_METRICS_TLS` (below) — per thread, like the
# quiet slot and the batch task slot — and `_active_metrics()` resolves that
# per-thread recorder first, falling back to the process-global for the REPL.
# Everything outside _metrics.py/_batch.py keeps reading/writing `_METRICS`
# exactly as before.
_METRICS: "_Metrics | None" = None  # set to a _Metrics() instance in _metrics.py
_METRICS_TLS: threading.local = threading.local()


def _active_metrics() -> "_Metrics | None":
    """The recorder THIS thread's usage should accumulate into: the batch
    worker's per-task metrics when one is installed on this thread, else the
    process-global REPL recorder."""
    m = getattr(_METRICS_TLS, "metrics", None)
    return m if m is not None else _METRICS


def _set_thread_metrics(m: "_Metrics | None") -> "_Metrics | None":
    """Install (and return the previously installed) per-thread recorder.
    Batch workers bracket their turn loop with this so parallel tasks each
    own their totals; the previous value is restored in `finally`."""
    prev = getattr(_METRICS_TLS, "metrics", None)
    _METRICS_TLS.metrics = m
    return prev

# --- native tool-calling confirmation ------------------------------------------
# Set True by _handle_native_tool_calls() the first time a turn actually comes
# back with structured `tool_calls` deltas (native mode). Read by
# _compress.py's _grammar_extra_body() to skip attaching our own <tool_call>
# text-tag GBNF grammar once native mode is confirmed: llama.cpp already
# derives its own argument-constraining grammar internally from `tools=`
# for templates it recognizes as native-tool-call-capable, and there is no
# client-observable trigger string to hook a *second*, argument-level grammar
# off of in that mode — sending our own `grammar` field would at best be
# redundant and at worst fight/override llama.cpp's own tool-calling grammar.
# Reset per source switch (see switch_source()) since a different endpoint
# may behave differently.
_NATIVE_TOOL_CALLS_CONFIRMED: bool = False

# --- sub-agent dispatch --------------------------------------------------------
# spawn_agent() runs its sub-loop with this table installed here (plus a
# per-thread copy in _subagent._SUBAGENT_TLS): read-only entries ONLY. See
# _dispatch.py (which assigns the initial values) and _subagent.py.
_SUBAGENT_DISPATCH: dict[str, Any] | None = None
_SUBAGENT_ACTIVE: bool = False

# --- bg-job / sub-agent completion sweep ---------------------------------------
# Per-turn scan flags and staged runtime-note text, owned by _dispatch.py's
# first-tool-of-turn scan (initial values assigned there).
#
# Swarm (0.98): the pending-notice slots used to be one process-wide str each.
# A parallel batch (`nx run --parallel > 1`) or concurrent in-process agents
# stage their finished answers into the SAME slot, and whichever transcript
# called _flush_pending_background_notice next consumed them — an async
# agent's answer landed in a sibling task's turn (documented in _batch.py).
# They are now thread-isolated: default values live in _NOTICE_DEFAULTS and
# every running turn loop installs its own dict of slots (see
# _notices.py). Outside a loop (tests, import time) the module-global default
# store keeps every existing test's `st._PENDING_* = x` write working.
_BG_SCAN_DONE: bool = False
_PENDING_BG_COMPLETION_NOTICE: str | None = None
_PENDING_SUBAGENT_NOTICE: str | None = None
_BG_NOTICE_SEEN: set[str] | None = None
# One turn loop's private notice slots; installed per-thread by _notices.py.
_NOTICE_DEFAULTS: dict[str, Any] = {
    "_BG_SCAN_DONE": False,
    "_PENDING_BG_COMPLETION_NOTICE": None,
    "_PENDING_SUBAGENT_NOTICE": None,
    "_BG_NOTICE_SEEN": None,
    "_PENDING_VERIFY_NOTICE": None,
    "_PENDING_YOLO_ASYNC_NOTICE": None,
    # 0.160.0 (AUTOWAKE_SPEC Phase B): staged by _autowake._stage_hold_notice
    # when a chain guard parks auto-continuation. Rides the normal flush to the
    # next prompt but must never itself wake the REPL (it is in
    # _autowake._NON_WAKE_NOTICE_SLOTS).
    "_PENDING_WAKE_HOLD_NOTICE": None,
    # 0.162.0 (PTC Phase 2): the run_tools fan-out nudge and the one-time
    # stub-module source note, staged by run_tool()/run_tools() (nx/_ptc_exec,
    # nx/_dispatch) and flushed with the other pending notes.
    "_PENDING_PTC_NOTICE": None,
}

# --- post-edit verification (NX_VERIFY_CMD) ------------------------------------
# Owned by _verify.py (initial values assigned there): timestamps of recent
# .py edits, an in-flight latch, the staged verify notice, and the pids
# already announced.
_VERIFY_EDIT_LOG: list[float] = []
_VERIFY_INFLIGHT: bool = False
_PENDING_VERIFY_NOTICE: str | None = None
_VERIFY_SEEN: set[str] = set()

# --- /ctx pinned messages ------------------------------------------------------
# 1-based message indexes pinned by /ctx pin (see _ctxreport.py).
_CTX_PINNED: list[int] = []

# --- write/edit checkpoint stack (for /undo) -----------------------------------
# Populated by run_tool() after a successful write_file/edit_file/multi_edit;
# see _undo.py for the entry shape and undo_last()'s restore logic.
_UNDO_STACK: list["Checkpoint"] = []

# --- read-only tool result cache ----------------------------------------------
# Keyed by (tool name, json.dumps(sorted args)) -> _tool_cache._CacheEntry
# (result, abspath, st_mtime_ns of the backing file at read time). Populated
# and consulted only by run_tool() for idempotent reads (read_file/list_dir);
# a hit re-stats and is valid while the mtime is unchanged. Invalidated
# per-path on write_file/edit_file/multi_edit; after run_bash only entries
# whose backing file changed are dropped (was: whole-cache clear). See
# _tool_cache.py and run_tool().
_TOOL_RESULT_CACHE: dict = {}

# --- todo list -----------------------------------------------------------------
# Whole-list state for the todo_write tool (see _todo.py). Rewritten wholesale
# on each call (no per-item id/merge logic — same convention as Claude Code's
# own TodoWrite), persisted into the session JSON's "todos" key so /resume
# restores it (see _repl_state.py's _save_current/_reload_metrics_totals).
_TODOS: list[dict[str, str]] = []

# Work-state snapshot (0.94.0, nx/_wip.py): the /wip record — branch, dirty
# files, todos, note — carried in the session JSON's "wip" key. None = no
# snapshot this session; /resume restores it.
_WIP: dict | None = None

# --- plan mode -----------------------------------------------------------------
# Plan-mode state machine (see _plan.py): while _PLAN_ACTIVE is True every
# mutating tool call is refused by run_tool()'s gate check with a gate message
# ("edit the plan or get it confirmed") instead of running. _PLAN_SLUG names the
# plan file .nx/<slug>.md (under PROJECT_ROOT, or cwd when unset). All four are
# persisted into the session JSON's "plan" key so /resume restores an active
# plan (see _repl_state.py's _save_current/_reload_metrics_totals).
_PLAN_ACTIVE: bool = False
_PLAN_GOAL: str | None = None
_PLAN_SLUG: str | None = None
_PLAN_STARTED: float | None = None

# --- env-resolved request/overflow/compression knobs --------------------------
# Assigned once at startup by nx/__init__.py (the NX_* env resolution block);
# declared here so static checkers (ty, pyrefly) see every `st.X` read site.
# Defaults mirror __init__.py's `_env_*` fallbacks — __init__.py's assignment
# is the authoritative initialization, never a second resolution path.
TEMPERATURE: float = -1.0
TOP_P: float = -1.0
TOKENS_MARGIN: int = 2048
MIN_COMPLETION_TOKENS: int = 256
TOKENIZE_PREFLIGHT: int = 1
OVERFLOW_FOLD_TIERS: int = 3
OVERFLOW_FOLD_KEEP_FLOOR: int = 1
OVERFLOW_EVICT_MIN_CHARS: int = 4000
OVERFLOW_EVICT_MAX: int = 8
COMPRESS_KEEP_TOKENS: int = 0
COMPRESS_SPILL_MIN_CHARS: int = 8000
COMPRESS_SPILL_MAX: int = 6
COMPRESS_SOURCE: str = ""
COMPRESS_MODEL: str = ""
COMPRESS_SUMMARY_TOKENS: int = 8192
_COMPRESS_SOURCE_WARNED: bool = False
_SUBAGENT_SOURCE_WARNED: bool = False

# --- SCALAR_SLOTS: the declared scalar homes of the per-test reset -------------
# The reset contract (tests/_nxtest.nx_state_reset_body) drains containers and
# TLS homes structurally, but a module-global SCALAR (bool/int/float/str/None)
# can only be swept by name — which is why the wake/plan-stall/chain/bg-scan
# residue below used to be hand-listed in the reset body with nothing
# connecting the hand list back to this module. SCALAR_SLOTS is the declared
# bridge: every module-global mutable scalar that genuinely needs a per-test
# reset (session/turn-loop residue a leaked value would carry into an
# unrelated later test) is named here, and tests/test_state_reset_contract.py
# derives its sweep from the tuple — planting a new undeclared scalar in this
# module fails the contract test (the P2 acceptance criterion in
# docs/DURABILITY_AUDIT.md). Public config knobs (env-resolved at startup,
# saved/restored by the tests that touch them) are NOT slots; the pre-existing
# private scalars that are not yet reset-body-swept are inventoried in
# _SCALAR_SLOTS_EXEMPT below — add a NEW scalar to SCALAR_SLOTS (plus the
# reset body), never to the exempt list.
SCALAR_SLOTS: tuple[str, ...] = (
    # auto-continuation / wake state (see the sections above): a leftover
    # _WAKE_DUE/_WAKE_ARMED/_WAKE_STREAK makes wake_due() fire in an unrelated
    # later test; the stall/chain slots change which instruction the next wake
    # picks and whether the chain budget trips.
    "_WAKE_ARMED",
    "_WAKE_DUE",
    "_WAKE_STREAK",
    "_WAKE_STALL_AT",
    "_WAKE_STALL_REVISION",
    "_WAKE_STALL_HOLD",
    "_WAKE_STALL_PENDING",
    "_WAKE_CHAIN_SPEND",
    "_WAKE_CHAIN_BASE",
    # background-completion scan state: _BG_NOTICE_SEEN is the "pids already
    # announced" set (None = re-seed from the baseline) and _BG_SCAN_DONE gates
    # the sweep — either left set makes the sweep announce a real bg job as a
    # [Runtime note] mid-test.
    "_BG_SCAN_DONE",
    "_BG_NOTICE_SEEN",
    # the last-prompt digest tap and the last-compression ledger are drained
    # to None by the reset body (a stale row makes the /ctx as-sent /
    # before-after views describe a request/fold a later test never ran).
    "_LAST_PROMPT",
    "_LAST_COMPRESSION",
)

# Private module-global scalars that predate the SCALAR_SLOTS contract and are
# NOT drained by the reset body (per-turn latches consumed on read, one-shot
# nudge flags, per-turn tally counters). Inventoried here so the contract
# test's AST scan can distinguish "declared, deliberately exempt" from "new
# undeclared hidden home" — a new scalar must NOT be added here.
_SCALAR_SLOTS_EXEMPT: tuple[str, ...] = (
    # turn-path latches (cleared/consumed by model_turn within the turn)
    "_LAST_PROMPT_TOKENS",
    "_LAST_API_ERROR",
    "_LAST_CONTEXT_OVERRUN",
    "_LAST_CACHE_HIT",
    "_LAST_PARTIAL_TEXT",
    "_LAST_REASONING",
    "_LAST_TOOL_NAME",
    "_LAST_TOOL_OUTPUT",
    "_STALL_EPISODE_ACTED",
    "_STALL_FORCE_NATIVE",
    "_NATIVE_TOOL_CALLS_CONFIRMED",
    # per-turn tally / PTC baseline counters (reset per turn by model_turn)
    "_TALLY_ARG_REPAIRS",
    "_TALLY_RETRIES",
    "_PTC_TURN_TOOL_CALLS",
    "_PTC_FANOUT_SIZE",
    "_PTC_ENVELOPE_CHARS",
    "_PTC_REPAIRED_BY_CANDIDATES",
    "_PTC_BRACKET_SALVAGED",
    "_PTC_NORMALIZED",
    "_PTC_HARD_FAILED",
    "_PTC_BREAKER_TRIPPED",
    "_PTC_FANOUT_SCRIPTS",
    # session-lifetime one-shot nudge flags / streaks (deliberate session scope)
    "_PTC_NUDGED",
    "_PTC_SOURCE_SHIPPED",
    "_PTC_BREAKER_TRIPS",
    "_PTC_DISABLED",
    "_SESSION_HAS_BACKGROUNDED",
    "_CONSECUTIVE_EDITS",
    "_MULTI_EDIT_NUDGED",
    "_YOLO_ASYNC_NUDGED",
    "_READ_STREAK",
    "_DELEGATION_NUDGE_COUNT",
    "_PATH_FAIL_LAST",
    "_PATH_FAIL_RUN",
    "_OCC_NATIVE_COMPACTIONS",
    "_SPAWNED_WRITE_TIER",
    "_TURN_ACTIVE",
    # dispatch/verify transient latches
    "_ACTIVE_PTC_PROC",
    "_VERIFY_INFLIGHT",
    # None-initialized object references (spinner/console/metrics homes,
    # recovery ContextVar holder, dispatch table) — cleared or restored by the
    # structural drains, the module rewind, or the production code that owns
    # them; not scalars in the reset-body sense.
    "_ACTIVE_SPINNER",
    "_TURN_SPINNER",
    "_PROGRESS_SINK",
    "_INTERRUPT_THREAD",
    "_METRICS",
    "_RECOVERY_CURRENT",
    "_SUBAGENT_DISPATCH",
    "_SUBAGENT_ACTIVE",
    # /ctx pin is False once the turn completes; the plan slots and the /wip
    # snapshot are persisted into the session JSON ("plan"/"wip" keys) and
    # restored by /resume, not by the per-test drain.
    "_PLAN_ACTIVE",
    "_PLAN_GOAL",
    "_PLAN_SLUG",
    "_PLAN_STARTED",
    "_WIP",
    # wake-prefill buffer consumed by the prompt on the wake turn itself
    "_WAKE_PREFILL",
    # per-path verify bookkeeping (timestamps set + announced pids)
    "_VERIFY_SEEN",
    # malformed-args breaker identity (dropped when the endpoint changes; the
    # sentinel key is re-armed by /toolcalls //reset, not by the test drain)
    "_BREAKER_ENDPOINT_KEY",
    # declaration-only config knob (no default here on purpose — __init__.py's
    # NX_MALFORMED_TOOL_BREAKER resolution is the only owner)
    "MALFORMED_TOOL_BREAKER_LIMIT",
    # interrupt redirect (Esc handshake, consumed by the REPL prompt)
    "_INTERRUPT_REDIRECT_TEXT",
    "_INTERRUPT_REDIRECT_SUBMITTED",
    "_INTERRUPT_REDIRECT_PREFILL",
    # one-shot warned flags
    "_COMPRESS_SOURCE_WARNED",
    "_SUBAGENT_SOURCE_WARNED",
)
