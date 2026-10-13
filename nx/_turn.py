"""The turn loop: model_turn() streams one exchange with the model and
returns one of the TURN_* status constants; _run_model_turn_loop() wraps it
to handle retries/recovery (stream cuts, reasoning-only stalls, empty turns,
malformed tool-call JSON, the model-turn-count safety valve). This is where
streaming parse, tool-call extraction (native or text-fallback via
parse_text_calls()), reasoning-content handling, and all the stall/recovery
guards live.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from ._openai_client import APIConnectionError, APIError
import httpx
from ._ansi import (
    DIM, RESET, LifeSpinner, C,
)
from ._interrupt import (
    _HARD_CANCEL, _INTERRUPT_ARMED, _INTERRUPT_RESTORED, _USER_INTERRUPTED,
    _disarm_interrupt_armed, _ensure_interrupt_watcher,
)
from ._metrics import _abbr, _normalize_usage, _record_usage
from ._special_tokens import (
    LEAKED_TOKEN_RUN_LIMIT, REPEAT_CHUNK_LIMIT, _strip_leaked_tokens,
)
from ._stream_parse import StreamParser
# EVENTSURFACE_SPEC §2 Phase 4: raw console writes below take the shared write
# lock so a 4 fps /tabs repaint cannot tear a turn diagnostic (_surfaces is
# already loaded by nx/__init__ and imports no turn-layer module — zero new
# modules in the T-9 conftest snapshot).
from ._surfaces import write_lock
from ._text_fallback import (
    parse_text_calls_with_errors,
)
from ._compress import (
    _nudge_current_user_turn,
    _last_is_dangling_tool,
)
from ._env import _env_float, _env_int
from ._verify import _launch_verify_if_settled
from . import _prefix
from ._images import _strip_consumed_images
from . import _state as st
from ._rich import cprint
from . import (  # noqa: F401  (run_tool re-exported; tests patch _turn.run_tool)
    FINAL_ANSWER_TOOL, FINAL_ANSWER_TOOL_CHOICE, run_tool, _active_tools,
    REASONING_STALL_NUDGE, FORCED_FINAL_NUDGE, EMPTY_TURN_NUDGE,
    STALL_WRAPUP_NUDGE,
    REASONING_ONLY_TIME_LIMIT, REASONING_CAPTURE_CHARS,
)
from ._subagent import (_subagent_active, _subagent_quiet, _subagent_scoped_retry,
                        _subagent_stream_cut_limit)
from ._batch import _batch_active
from ._plan import plan_active

# --- re-exports from the split-out layers (single import surface preserved) ---
from ._turn_footer import (  # noqa: F401 (re-exports)
    _say_console, _ctx_field, _cache_hit_field, _tps_field, _window_tps,
    _capture_last_prompt_tokens, _print_stats_footer, _ensure_ctx_probe,
)
from ._tool_prefetch import (  # noqa: F401 (re-exports)
    _all_read_only, _write_capable_agent, _prefetch_cacheable_tools,
    _run_read_only_parallel, _should_parallelize,
)
from ._argsalvage import (  # noqa: F401 (re-exports)
    FALLBACK_FOR, _malformed_args_nudge, _repair_log, _salvage_truncated_args,
)
from ._turn_retry import (  # noqa: F401 (re-exports)
    _TASK_DEADLINE_TLS, _plural_ies, _retry_log, _step_backoff_delay, _step_retry_limit,
    _task_deadline, _task_deadline_at, _task_deadline_exceeded,
    _task_timeout_summary,
)
from ._turn_status import (  # noqa: F401 (re-exports)
    TURN_DONE, TURN_TOOL, TURN_STREAM_CUT, TURN_FORCE_FINAL, TURN_STALL,
    TURN_EMPTY, TURN_ESC, TURN_REDIRECT, TURN_MALFORMED_CALL, TURN_API_ERROR,
    TURN_CONTEXT_OVERRUN, TURN_LENGTH_CUT, TURN_TIMED_OUT, TURN_TIMED_OUT_RETRY,
    TRUNCATION_MARKER,
)
from ._turn_toolcalls import (  # noqa: F401 (re-exports)
    _close_out_partial, CONTINUATION_NUDGE, _handle_forced_final_tool_call,
    _handle_forced_final_text_calls, _handle_native_tool_calls,
    _handle_text_tool_calls, _strip_unterminated_tool_call,
)

Message = dict[str, Any]  # one entry in the OpenAI-style `messages` array


class _LoggingStream:
    """Iterator wrapper that tees each SSE chunk to llamacpp.log before yielding.
    Uses model_dump so we capture the chunk's full structure (incl. reasoning_content)."""
    def __init__(self, inner: Any, log_file: Any) -> None:
        self._inner = inner
        self._log = log_file

    @property
    def dropped_lines(self) -> int:
        # Proxied, not re-declared as an attribute: _turn.py reads
        # getattr(stream, "dropped_lines", 0) off THIS object (open_stream
        # returns the wrapper), so without the proxy the malformed-frame
        # warning silently reported 0 forever -- exactly the trace that was
        # missing while truncated tool arguments went undiagnosed.
        return getattr(self._inner, "dropped_lines", 0)

    def __iter__(self):
        for chunk in self._inner:
            try:
                self._log.write(json.dumps({"ts": time.time(), "dir": "resp",
                                             "data": chunk.model_dump()}) + "\n")
            except Exception:
                pass  # never let logging break the stream
            yield chunk

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


# --- re-exports from the split-out layers (single import surface preserved) ---
# (the _turn_status / _turn_toolcalls re-exports sit with the other split-layer
# imports at the top of the file)


def _maybe_arm_native_escalation() -> None:
    """After a stall/empty nudge, make the retry open with tool_choice="auto"
    when the just-failed request was sent with "none".

    Since 0.54.0 "auto" is the default posture, so this only fires when the
    session explicitly disabled native tool calls — /toolcalls text, or
    NX_NATIVE_TOOLS=0 — and the stall is evidence that disabling it took away
    the model's only action channel (glm/qwen behind vLLM never fall back to
    the text-protocol tags when native calls are suppressed: they think, emit
    nothing, and every "act now" nudge is unanswerable). A user-forced text
    protocol is left alone (they chose it); only the env-gated posture is
    escalated, once per episode, with a visible line saying so.
    """
    from ._compress import NX_NATIVE_TOOLS, toolcalls_mode
    if st._STALL_FORCE_NATIVE or not NX_NATIVE_TOOLS:
        return  # already armed / posture is already "auto"
    if toolcalls_mode() == "text":
        return  # user-forced text protocol — respect it, don't override
    st._STALL_FORCE_NATIVE = True
    if not _subagent_quiet():
        with write_lock():
            cprint(f"{DIM}  ↳ retrying with native tool-calling enabled "
                  f"(tool_choice=auto){RESET}")


def _handle_loop_cut(messages: list[Message], forced_final: bool, reasoning_loop_cut_count: int,
                     leaked_token_run: int, repeat_chunk_run: int, reasoning_only_start: Optional[float],
                     reasoning_only_chars: int, reasoning_parts: list[str],
                     malformed_stream_cut_count: int = 0) -> str:
    """Recover from a streamed turn that was cut mid-flight.
    Handles two cases that share the "cut" path:
      • special-token / repetition collapse — degenerated output, nudged fresh.
      • reasoning-only stall — the model thought but never acted; two-tier
        recovery nudges it to act (TURN_STALL), or on the last attempt forces a
        visible final answer (TURN_FORCE_FINAL).

    Returns a TURN_* status to bubble up from model_turn.
    """
    # A leaked-token or repetition collapse is a degeneration, not a
    # reasoning stall — the model was emitting real content (or reasoning)
    # but it degenerated into gibberish.  Treat it like a malformed stream:
    # discard the degenerated output and nudge the model to produce clean
    # output from the current conversation state.  This runs BEFORE the
    # reasoning-stall logic so we don't mis-classify a collapse as a stall.
    is_leak_collapse = (leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT
                       or repeat_chunk_run >= REPEAT_CHUNK_LIMIT)
    quiet = _subagent_quiet()
    if is_leak_collapse and not forced_final:
        collapse_type = ("special-token collapse" if leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT
                         else "repetition collapse")
        retry_limit = _subagent_stream_cut_limit(max(0, st.MALFORMED_STREAM_RETRY_LIMIT))
        if malformed_stream_cut_count >= retry_limit:
            # 0.121.1: background agents run quiet, so this give-up printed
            # nothing anywhere — the agent surrendered silently. Record it in
            # the event log (screen behavior unchanged).
            from ._logging import _log_event, usage_give_up
            _log_event("turn", {"kind": "give_up",
                                "failure": collapse_type,
                                "recoveries": malformed_stream_cut_count,
                                "retry_limit": retry_limit,
                                "model": st.MODEL})
            usage_give_up({"failure": collapse_type,
                           "recoveries": malformed_stream_cut_count,
                           "retry_limit": retry_limit})
            if not quiet:
                with write_lock():
                    cprint(f"{C.ERROR}  ✗ {collapse_type} after "
                          f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
            return TURN_DONE
        if not quiet:
            with write_lock():
                cprint(f"{C.WARNING}  ✂ {collapse_type.upper()} — discarding degenerated "
                      f"output and nudging clean recovery "
                      f"({malformed_stream_cut_count + 1}/{retry_limit}){RESET}")
        # Do NOT feed the degenerated reasoning/content back — it's spam
        # that would reinforce the collapse on the next turn.  Nudge fresh.
        # Once we've collapsed more than once on the same turn, the repeat is
        # itself evidence the task-as-attempted is too big for the model to
        # commit to in one shot — push it toward a smaller, incremental step
        # instead of repeating the same "rewrite the whole thing" nudge.
        if malformed_stream_cut_count > 0:
            nudge = (
                "Your previous output degenerated into a repetition collapse "
                "again. Repeated collapses on the same task usually mean the "
                "planned action is too large to commit to in one shot (e.g. a "
                "full-file rewrite with many interleaved pieces to track). "
                "Break it down: pick the SMALLEST next concrete edit (a single "
                "edit_file/multi_edit hunk, not a full rewrite), and emit only "
                "that tool call now — no restated plan, no prose.")
        else:
            nudge = (
                "Your previous output degenerated into a repetition collapse "
                "(repeated special tokens or identical chunks). This is a known "
                "failure mode — discard that output entirely. Re-orient: state "
                "what you were trying to do in one sentence, then emit the next "
                "concrete tool call (write_file, run_bash, or edit_file). Do not "
                "continue the previous stream. Start fresh.")
        _nudge_current_user_turn(messages, nudge)
        return TURN_STREAM_CUT
    retry_limit = _subagent_scoped_retry(max(0, st.REASONING_ONLY_RETRY_LIMIT))
    cut_count = reasoning_loop_cut_count
    # Set when a real tool call runs mid-episode (see the TURN_TOOL branch in
    # _run_model_turn_loop): the "has the model acted?" marker that demotes
    # this nudge from "ACT NOW" to a wrap-up request. Reset when a fresh
    # episode begins so a stale marker from an earlier episode can't suppress
    # this episode's text-only guard or rewrite the nudge text.
    if cut_count == 0:
        st._STALL_EPISODE_ACTED = False
    reasoning_secs = (time.time() - reasoning_only_start) if reasoning_only_start else 0
    if reasoning_only_start and REASONING_ONLY_TIME_LIMIT > 0 and \
            reasoning_secs >= REASONING_ONLY_TIME_LIMIT:
        loop_detail = (f"{reasoning_secs:.0f}s of reasoning "
                       f"({_abbr(reasoning_only_chars)} chars) without content/tool calls")
    else:
        loop_detail = (
            f"{_abbr(reasoning_only_chars)} reasoning chars without content/tool calls")
    if forced_final:
        # tool_choice=required still didn't produce a tool call or content —
        # some backends (reasoning models with a "thinking" phase) can burn
        # the whole forced-final budget on reasoning and stop without ever
        # emitting the required call. Dying here with nothing shown is worse
        # than showing the model's own reasoning as a best-effort answer —
        # surface it instead of leaving the user with an empty turn.
        salvaged = _strip_unterminated_tool_call("".join(reasoning_parts)).strip()
        if quiet:
            pass  # the salvage prints below are terminal noise on an agent thread
        elif salvaged and st.REASONING_VISIBLE:
            # Reasoning already streamed dim above (REASONING_VISIBLE): re-printing
            # the identical text in normal color reads as a duplicated echo. A
            # pointer to the dim block is the whole salvage here.
            with write_lock():
                cprint(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
                      f"showing captured reasoning as a best-effort answer{RESET}")
                cprint(f"{DIM}  (the reasoning is the dim block above){RESET}")
        else:
            with write_lock():
                cprint(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
                      f"returning to chat input{RESET}")
        return TURN_DONE
    if cut_count >= retry_limit:
        if retry_limit == 0:
            # Failsafes deliberately disabled (e.g. benchmark mode where
            # we let the model think as long as it wants).  The model
            # produced only reasoning and no action — that's an honest
            # model-capability signal, not a harness rescue scenario.
            if not quiet:
                with write_lock():
                    cprint(f"{DIM}  • reasoning-only turn ended naturally "
                          f"({loop_detail}); failsafes disabled, ending turn{RESET}")
        elif not quiet:
            with write_lock():
                cprint(f"{C.ERROR}  ✂ REASONING-ONLY RESCUE FAILED — gave up after "
                      f"{cut_count} stall "
                      f"attempt{'s' if cut_count != 1 else ''} "
                      f"× {loop_detail}; waiting for user input{RESET}")
        # 0.121.1: this turn ends with nothing produced and the prints above
        # are gated on `quiet` — on a background agent thread that was a
        # completely silent surrender. Log it either way (screen behavior
        # unchanged); `failsafes_disabled` separates the deliberate
        # benchmark-mode ending from a genuine rescue exhaustion.
        from ._logging import _log_event, usage_give_up
        _log_event("turn", {"kind": "give_up",
                            "failure": "reasoning_only_stall",
                            "attempts": cut_count,
                            "retry_limit": retry_limit,
                            "failsafes_disabled": retry_limit == 0,
                            "model": st.MODEL})
        usage_give_up({"failure": "reasoning_only_stall",
                       "attempts": cut_count,
                       "retry_limit": retry_limit,
                       "failsafes_disabled": retry_limit == 0})
        return TURN_DONE
    if reasoning_only_start and REASONING_ONLY_TIME_LIMIT > 0 and \
            reasoning_secs >= REASONING_ONLY_TIME_LIMIT:
        cut_msg = (f"{reasoning_secs:.0f}s of reasoning with no "
                   f"answer/tool call (limit {REASONING_ONLY_TIME_LIMIT}s)")
    elif st.REASONING_ONLY_CHAR_LIMIT > 0 and \
            reasoning_only_chars >= st.REASONING_ONLY_CHAR_LIMIT:
        cut_msg = (f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                   f"answer/tool call (limit {_abbr(st.REASONING_ONLY_CHAR_LIMIT)})")
    else:
        cut_msg = (f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                   f"answer/tool call (stream ended empty)")

    # --- Feed captured reasoning back so the model can *continue* -----
    # The old behavior discarded reasoning_content entirely and appended a
    # nudge to the existing user message.  The model saw the same starting
    # state every time → same reasoning → same stall → literal loop.  Now
    # we store the reasoning as an assistant message (truncated to a cap so
    # a 46K-char spiral doesn't blow up context) and add the nudge as a NEW
    # user message.  The model sees: [task] → [my reasoning] → [now act]
    # and can pick up where it left off instead of starting from scratch.
    reasoning_text = _strip_unterminated_tool_call("".join(reasoning_parts))
    if REASONING_CAPTURE_CHARS > 0 and reasoning_text.strip():
        cap = REASONING_CAPTURE_CHARS
        if len(reasoning_text) > cap:
            # Keep the tail — it has the model's most recent conclusions.
            reasoning_text = ("[…earlier reasoning truncated…]\n"
                              + reasoning_text[-cap:])
        messages.append({"role": "assistant", "content": reasoning_text})
        if not quiet:
            with write_lock():
                cprint(f"{DIM}  ↳ captured {len(reasoning_text)} chars of reasoning "
                      f"for feedback{RESET}")

    # Two-tier recovery:
    #   • First (and any intermediate) stall: nudge the model to ACT —
    #     emit a tool call, write a file — with FULL tool access. This
    #     is critical for coding tasks where the model got stuck planning
    #     but never executed anything.  Returns TURN_STALL so the loop
    #     retries with normal tools (not the restricted FINAL_ANSWER_TOOL).
    #   • Last stall (cut_count == retry_limit - 1): force a final answer
    #     with restricted tools (FINAL_ANSWER_TOOL only).  This is the
    #     fallback when repeated nudges still don't break the reasoning
    #     loop, so we at least get a visible answer instead of a hang.
    is_last_retry = (cut_count == retry_limit - 1)
    if is_last_retry:
        if not quiet:
            with write_lock():
                cprint(f"{C.WARNING}  ✂ REASONING-ONLY STALL — {cut_msg}; forcing final "
                      f"answer (attempt {cut_count + 1}/{retry_limit}){RESET}")
        if reasoning_text.strip():
            messages.append({"role": "user", "content":
                "[Runtime note: Your reasoning has been preserved as your "
                "previous message. Do not continue private reasoning. Use "
                "the final_answer tool if available, or return a complete "
                "visible answer now based on what you reasoned. Keep it "
                "concise. Do not repeat or restate your reasoning — it is "
                "already visible above.]"})
        else:
            _nudge_current_user_turn(messages, FORCED_FINAL_NUDGE)
        return TURN_FORCE_FINAL
    # A stall AFTER the model has already done tool work this episode is a
    # different failure: it stalled while composing its wrap-up, not while
    # dodging the work. Ask for the visible answer instead of demanding a
    # tool call — "ACT NOW" is exactly what produced the TEXT-ONLY-AFTER-
    # STALL misfires on legitimate finished answers.
    wrapup_note = (STALL_WRAPUP_NUDGE
                   if (st._STALL_EPISODE_ACTED and not plan_active()) else None)
    if not quiet:
        with write_lock():
            cprint(f"{C.WARNING}  ✂ REASONING-ONLY STALL — {cut_msg}; "
                  f"{'nudging to wrap up' if wrapup_note else 'nudging to act'} "
                  f"(attempt {cut_count + 1}/{retry_limit}){RESET}")
    _maybe_arm_native_escalation()
    if wrapup_note is not None:
        if reasoning_text.strip():
            messages.append({"role": "user", "content": f"[Runtime note: {wrapup_note}]"})
        else:
            _nudge_current_user_turn(messages, wrapup_note)
    elif reasoning_text.strip():
        messages.append({"role": "user", "content":
            "[Runtime note: You were thinking about the task above. "
            "Your reasoning has been preserved as your previous message. "
            "Now act on it — do NOT continue reasoning. Concretely:\n"
            "  • write_file: create the deliverable file the task expects "
            "(e.g. /app/solution.py, /app/main.rs). Put your planned approach "
            "into actual code now, even as a first draft.\n"
            "  • run_bash: execute it to see what happens.\n"
            "  • edit_file / write_file: fix what the run reported and re-run.\n"
            "An imperfect file you iterate on beats a perfect plan that never "
            "gets written. Emit the write_file or run_bash tool call now. "
            "Do not repeat or restate your reasoning in your answer — it is "
            "already visible above.]"})
    else:
        _nudge_current_user_turn(
            messages,
            ("A plan is active — keep refining it with plan_edit until it "
             "covers the whole task, then ask the user to run /plan done.")
            if plan_active() else REASONING_STALL_NUDGE)
    return TURN_STALL


# --- one model turn (streamed), returns TURN_* status -----------------------
def model_turn(messages: list[Message], reasoning_loop_cut_count: int = 0, malformed_stream_cut_count: int = 0,
               empty_turn_count: int = 0, forced_final: bool = False, recovery_sampling: bool = False,
               malformed_call_count: int = 0, step_retries: int = 0) -> str:
    # Start the spinner BEFORE open_stream() so the HTTP-handshake + interrupt-
    # watcher-setup window (which can be tens of ms on a warm local server but
    # seconds on a cold/wake-from-sleep one) isn't a frozen green-text gap.
    # The spinner is still killed by the first SSE chunk in the loop below.
    # t0 must be set BEFORE open_stream() — the HTTP request is sent inside
    # open_stream() (TCP/TLS handshake, request bytes, etc.), and all of that
    # latency is part of TTFT. If we set t0 after, the server has already been
    # processing and the first token may be sitting in the buffer by the time
    # we start the clock, making TTFT look implausibly tiny.
    t0 = time.time()
    st._LAST_API_ERROR = None  # cleared before every open; set by open_stream() on a terminal failure
    st._LAST_CONTEXT_OVERRUN = None  # ditto, for the context-length-400 branch of open_stream()
    # PERF.md I-4: the request is about to be sent, so this is the one place a
    # moved prompt prefix actually costs. Verifies the head still matches the
    # bytes pinned at session start and warns once when it doesn't (a mid-history
    # rewrite shows up as a mystery `cached: 0%` otherwise). A no-op off the
    # pinned path (sub-agent turns, tests).
    _prefix.check_prefix(messages)
    # 0.30.0: the tally counts ONE turn's self-healing work (the footer prints
    # it after this stream settles); a multi-recovery turn keeps accumulating
    # across its retry cycles until the next model_turn() starts.
    st._TALLY_ARG_REPAIRS = 0
    st._TALLY_RETRIES = 0
    quiet = _subagent_quiet()
    spinner_label = (
        "forcing final answer · esc to interrupt"
        if forced_final else
        "recovering · esc to interrupt" if recovery_sampling else
        "thinking · esc to interrupt"
    )
    spinner = LifeSpinner(label=spinner_label)
    if not quiet:
        spinner.start()
        # Registered so a ^C racing the render thread's last frame can be
        # force-wiped before the chatbox paints its box (see repl.py's
        # KeyboardInterrupt handler) — otherwise the spinner line survives
        # into the scrollback above the next prompt.
        st._TURN_SPINNER = spinner
    # Go through the top-level package (not a bare same-module reference) so
    # a test that monkeypatches `nx.open_stream` (to avoid a real model
    # round-trip) takes effect.
    import nx as _m
    try:
        if forced_final:
            stream = _m.open_stream(
                messages,
                tools=[FINAL_ANSWER_TOOL],
                tool_choice=FINAL_ANSWER_TOOL_CHOICE,
                max_tokens=st.FORCED_FINAL_MAX_TOKENS,
                recovery_sampling=recovery_sampling,
            )
        else:
            stream = _m.open_stream(messages, tools=_active_tools(), recovery_sampling=recovery_sampling)
    except Exception:
        if not quiet:
            spinner.stop()
        st._TURN_SPINNER = None
        raise
    if stream is None:
        if not quiet:
            spinner.stop()
        st._TURN_SPINNER = None
        if st._LAST_CONTEXT_OVERRUN:
            # Recoverable: _run_model_turn_loop() folds the older turns into a
            # summary and re-runs this turn. Not an error line — say what is
            # happening (once) so a long task doesn't just pause silently.
            if not quiet:
                with write_lock():
                    cprint(f"{C.WARNING}  ✂ context window full — compressing and retrying"
                          f"{RESET}")
            return TURN_CONTEXT_OVERRUN
        if st._LAST_API_ERROR:
            # open_stream() already printed the friendly message; interactive
            # mode keeps the REPL alive below, but headless callers must NOT
            # score this as a completed turn — report the hard failure up.
            return TURN_API_ERROR
        return TURN_DONE  # error already reported; REPL continues

    # Interrupt watcher: the singleton daemon thread watches stdin for bare Esc
    # while ARMED. On hit it sets _USER_INTERRUPTED; the main loop breaks out of
    # the chunk loop on its next iteration. We arm it BEFORE the spinner so the
    # user always has a moment to interrupt even if the first token takes a
    # while to arrive. A quiet (async-agent) loop NEVER touches it: the watcher
    # is a process singleton — the parent's stream owns it right now, and an
    # agent thread clearing/restoring it would eat keystrokes or break Esc.
    # Idempotent and event-neutral when the thread is already alive: only the
    # non-quiet path below arms it (a quiet thread must not touch the handshake).
    # Swarm P12: a quiet loop never arms (and never STARTS the thread — see
    # _ensure_interrupt_watcher); NX_INTERRUPT_WATCHER=0 disables arming for
    # the whole process (headless runs have no tty to steal).
    from ._interrupt import _watcher_disabled
    _ensure_interrupt_watcher()
    if not quiet and not _watcher_disabled():
        _USER_INTERRUPTED.clear()
        # C1: a hard cancel lives for exactly one turn — the tool group it
        # killed plus the skip-fill of its remainder. A fresh interactive
        # turn starts clean (quiet loops never clear: they don't own stdin).
        _HARD_CANCEL.clear()
        st._INTERRUPT_REDIRECT_TEXT = ""
        st._INTERRUPT_REDIRECT_SUBMITTED = False
    # Reset the per-turn bg-completion scan flag: the first tool call of this
    # new turn re-checks for finished background jobs (see run_tool()). A
    # quiet (async-agent) loop skips that scan entirely — run_tool() gates it
    # on _subagent_ctx(), because the notices belong to the PARENT's
    # transcript; the flag here just mirrors the reset for the parent turn.
    if not quiet:
        from . import _notices
        _notices.set_("_BG_SCAN_DONE", False)
    # Clear the "termios restored" flag BEFORE arming so we can wait on it
    # after disarm. The watcher sets it in its inner finally once it has
    # restored cooked mode. Without this handshake the watcher's termios
    # restore is asynchronous (it may still be in select(0.1) when we
    # return), and the chatbox can capture the watcher's raw-mode termios
    # as its "old" state — leaving the terminal stuck in VMIN=0
    # with broken Enter and garbled typing.
    # The whole handshake — clear, arm, and the wait in the finally — belongs
    # to the thread that owns stdin. A quiet (async-agent / batch) thread must
    # not touch _INTERRUPT_RESTORED at all: it never sets or waits on the
    # event, so clearing it here strands the PARENT's post-disarm wait (the
    # parent stalls for the full NX_INTERRUPT_RESTORE_WAIT and emits a spurious
    # "did not confirm input-mode restore" warning), and in the worst
    # interleaving lets the parent proceed before cooked mode is confirmed.
    # Mirrors _watch_for_interrupts ownership and _poll_pid's check_esc gating.
    if not quiet and not _watcher_disabled():
        _INTERRUPT_RESTORED.clear()
        _INTERRUPT_ARMED.set()
    parser = StreamParser(_say_console, t0, quiet=quiet)
    stream_error = None
    interrupted = False

    try:
        for chunk in stream:
            if _USER_INTERRUPTED.is_set():
                interrupted = True
                # close() makes the next iteration raise StopIteration / a
                # connection error; we're breaking anyway, but be tidy
                close = getattr(stream, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass
                break
            # first byte in: kill the spinner, let the real output take this line
            if spinner._t is not None:
                spinner.stop()
                st._TURN_SPINNER = None
            if parser.process(chunk):
                # A collapse (leaked-token or repetition) was detected —
                # parser.loop_cut is already set; close the stream and stop.
                close = getattr(stream, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass
                break
    except (APIError, APIConnectionError, httpx.HTTPError) as e:
        stream_error = e
        close = getattr(stream, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass
    finally:
        if not quiet:
            spinner.stop()
            st._TURN_SPINNER = None  # a ^C racing this wipe re-stops safely (idempotent)
        parser.clear_tool_status()
        # Safety net: interrupted/errored/collapsed breaks may leave the Live
        # display still open (the mode-transition sites above only stop it on
        # the normal tool-call/end-of-reasoning paths).
        parser.stop_say_live()
        # Disarm the watcher: it stops reading stdin, restores termios in its
        # own inner finally, and parks on _INTERRUPT_ARMED until next turn.
        # We MUST wait for _INTERRUPT_RESTORED before returning — otherwise the
        # watcher's termios restore is asynchronous (it may still be in
        # select(0.1) when we return) and the chatbox can capture our raw-mode
        # termios as its "old", leaving the terminal stuck in VMIN=0.
        # The watcher's select has a 0.1s timeout, but we poke its wake pipe
        # (PF-06) so it notices the disarm immediately; restore typically
        # completes in single-digit ms. If a bare Esc just fired the watcher is
        # instead inside _drain_redirect_text, which has its own 2.0s deadline —
        # wait past that (not just 0.5s) so we never proceed while it still
        # holds raw termios mid-drain. NX_INTERRUPT_RESTORE_WAIT tunes the wait
        # (nx/_interrupt.py); keep the two call sites' effective values equal.
        if not quiet:
            _disarm_interrupt_armed()
            _INTERRUPT_RESTORED.wait(
                timeout=_env_float("NX_INTERRUPT_RESTORE_WAIT", 2.5))
            _USER_INTERRUPTED.clear()
    dropped_lines = getattr(stream, "dropped_lines", 0)
    if dropped_lines and not quiet:
        with write_lock():
            cprint(f"{C.WARNING}  ⚠ {dropped_lines} malformed SSE frame(s) dropped mid-stream — "
                  f"any tool call in this turn may have truncated arguments{RESET}")
    # Account before every post-stream early return (stream error, user
    # interrupt, collapse/reasoning stall). The previous footer-owned hook
    # silently skipped all of those completed API calls.
    _record_usage(
        parser.usage, parser.timings,
        fallback_chars=parser.streamed_chars + sum(len(part) for part in parser.reasoning_parts),
        fallback_prompt_chars=lambda: sum(len(str(m.get("content") or "")) for m in messages),
    )
    # v2 usage metrics: one model_call event per settled stream — the same
    # common path, so error early-returns below are counted too. Tokens ride
    # along when the server reported them (the _normalize_usage shape); TTFT
    # is the client-measured first-token delta.
    from ._logging import usage_model_call
    usage_model_call(
        (time.time() - t0) * 1000.0,
        parser.finish_reasons,
        _normalize_usage(parser.usage, parser.timings),
        streamed_chars=parser.streamed_chars,
        ttft_ms=(parser.t_first * 1000.0) if parser.t_first is not None else None,
    )
    if stream_error is not None:
        if isinstance(stream_error, httpx.TimeoutException):
            # P3-2 (0.51.0): a mid-stream timeout is a FAILURE, not a completed
            # turn — it used to return TURN_DONE here, which headless callers
            # scored as a clean pass on truncated output. First spend the
            # step-level retry ladder (transient busy-server timeouts, the same
            # ladder _send uses pre-first-byte); only exhaustion returns the
            # honest TURN_TIMED_OUT.
            if step_retries < _step_retry_limit():
                if not quiet:
                    # Whole block = one visual unit (closer + RESET + notice).
                    # close_think_block re-enters the same RLock: safe.
                    with write_lock():
                        if parser.mode in ("think", "say"):
                            cprint()
                        parser.close_think_block()
                        cprint(RESET)
                        delay = _step_backoff_delay(step_retries, None)
                        cprint(f"{C.WARNING}  ↻ STREAM TIMEOUT — server stopped delivering chunks; "
                              f"discarding partial output and retrying "
                              f"({step_retries + 1}/{_step_retry_limit()} in {delay:.1f}s){RESET}")
                # v1 usage metrics: one retry event per ladder step (never
                # per chunk), with the turn-level stream error class.
                _retry_log(step_retries + 1, type(stream_error).__name__)
                return TURN_TIMED_OUT_RETRY
            if not quiet:
                with write_lock():
                    if parser.mode in ("think", "say"):
                        cprint()
                    parser.close_think_block()
                    cprint(RESET)
                    cprint(f"{C.ERROR}  ✗ STREAM TIMEOUT after {step_retries} "
                          f"retr{_plural_ies(step_retries)} — server stopped delivering chunks; "
                          f"the task did NOT complete{RESET}")
                    cprint(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_TIMED_OUT
        retry_limit = max(0, st.MALFORMED_STREAM_RETRY_LIMIT)
        if malformed_stream_cut_count >= retry_limit:
            # 0.121.1: same silent surrender as the collapse path above — a
            # quiet/background agent used to give up here with no trace at all.
            from ._logging import _log_event, usage_give_up
            _log_event("turn", {"kind": "give_up",
                                "failure": "malformed_stream",
                                "recoveries": malformed_stream_cut_count,
                                "retry_limit": retry_limit,
                                "error": (f"{type(stream_error).__name__}: "
                                          f"{stream_error}")[:300],
                                "model": st.MODEL})
            usage_give_up({"failure": "malformed_stream",
                           "recoveries": malformed_stream_cut_count,
                           "retry_limit": retry_limit,
                           "error": (f"{type(stream_error).__name__}: "
                                     f"{stream_error}")[:300]})
            if not quiet:
                with write_lock():
                    if parser.mode in ("think", "say"):
                        cprint()
                    parser.close_think_block()
                    cprint(RESET)
                    cprint(f"{C.ERROR}  ✗ malformed stream from model/API after "
                          f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
                    cprint(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_DONE
        if not quiet:
            with write_lock():
                cprint(f"{C.WARNING}  ✂ MALFORMED STREAM — discarded partial output/tool args; "
                      f"retrying cleanly ({malformed_stream_cut_count + 1}/{retry_limit}){RESET}")
        _nudge_current_user_turn(
            messages,
            "Your previous streamed response became malformed before it completed. "
            "Discard it. Retry the same task from the current conversation state, "
            "but emit either valid tool calls or a concise final answer only.")
        return TURN_STREAM_CUT
    # reasoning-only turn (no content, no tool_calls) — close out the block so
    # the stats footer doesn't run straight into the dim reasoning text
    parser.finish()
    # Stash the full reasoning text regardless of visibility so /reasoning show
    # can display it after the fact even when streamed collapsed.
    if parser.reasoning_parts:
        st._LAST_REASONING = "".join(parser.reasoning_parts)
    text = "".join(parser.content)
    # Final defense-in-depth sanitization: strip any leaked special tokens
    # that slipped through the per-chunk stripping (e.g. a single token at the
    # very end of a stream that got cut before the run limit triggered).
    text = _strip_leaked_tokens(text)
    elapsed = time.time() - t0
    _capture_last_prompt_tokens(parser.usage, parser.timings)

    if interrupted:
        if not quiet:
            with write_lock():
                cprint(f"{C.WARNING}  ↳ interrupted by user (Esc) after {elapsed:4.1f}s, "
                      f"{len(parser.content)} chars streamed{RESET}")
        # Discard partial content — it's almost certainly a half-formed
        # sentence / tool-call args.
        redirect = st._INTERRUPT_REDIRECT_TEXT.strip()
        if redirect and st._INTERRUPT_REDIRECT_SUBMITTED:
            # The user typed and hit Enter during the disarm handshake — treat
            # it exactly like a normal chat message and feed it straight back
            # into another turn, skipping the REPL prompt entirely.
            messages.append({"role": "user", "content": redirect})
            return TURN_REDIRECT
        if redirect:
            # Partial line, no Enter yet — hand it to the REPL as a prefill so
            # the chatbox resumes with those characters already typed instead
            # of a blank prompt.
            st._INTERRUPT_REDIRECT_PREFILL = redirect
        # Append a synthetic user turn so the model has context for what just
        # happened, then return so the REPL drops to the prompt instead of
        # looping into another turn.
        messages.append({"role": "user", "content":
            "[User interrupted your previous response with Esc. "
            "Acknowledge briefly and wait for their next message.]"})
        return TURN_DONE

    loop_cut = parser.loop_cut
    if (not loop_cut and not text.strip() and not parser.tcs
            and (parser.reasoning_only_chars > 0
                 or (parser.reasoning_parts and not parser.content_had_prose))):
        # A reasoning-only stall, even if reasoning_only_chars was reset by a
        # stray whitespace-only content delta (chat-template separator).
        loop_cut = True

    if loop_cut:
        # A collapse/reasoning-only cut with buffered tool-call args is a
        # malformed-args turn, not a reasoning stall: the model was mid-call
        # when the stream degenerated, and the malformed-args recovery (same
        # family _handle_native_tool_calls uses) is the correct nudge — the
        # stall ladder's "start reasoning anew" nudge would discard the args
        # context the model was actually working in.
        if parser.tcs and not text.strip():
            # Local alias: the module-level name must stay callable above.
            _native = _handle_native_tool_calls
            return _native(messages, parser.tcs, text,
                           parser.finish_reasons,
                           malformed_stream_cut_count)
        return _handle_loop_cut(
            messages, forced_final, reasoning_loop_cut_count,
            parser.leaked_token_run, parser.repeat_chunk_run, parser.reasoning_only_start,
            parser.reasoning_only_chars, parser.reasoning_parts,
            malformed_stream_cut_count)

    # v1 usage metrics: remember this turn's assistant text as the CTX_TAIL
    # source — the seam where the full text exists BEFORE any tool-call
    # handler runs, so a tool_call/repair/breaker/gate event fired later
    # carries what the model was saying when it made the call.
    from ._usage_ctx import remember as _remember_ctx
    _remember_ctx(text)

    # stats footer — prefer llama.cpp timings if present (server-measured tok/s);
    # otherwise divide the server's completion_tokens by the client-side
    # generation window (parser.gen_window(): first→last token span, not
    # wall-clock). 0.148.2: this call now RECORDS the step's numbers into the
    # user-turn accumulator begun by _run_model_turn_loop; the loop's finally
    # prints ONE aggregated footer per user turn instead of a stacked row per
    # step. TTFT is shown when available — for local it comes from llama.cpp
    # timings, for remote we measure it client-side.
    _print_stats_footer(parser.usage, parser.timings, parser.streamed_chars, parser.t_first,
                         elapsed, text, parser.tcs, messages=messages, parser=parser,
                         quiet=_subagent_quiet())

    if forced_final and parser.tcs:
        return _handle_forced_final_tool_call(messages, parser.tcs, text,
                                              parser.finish_reasons,
                                              malformed_stream_cut_count)

    if parser.tcs:  # native tool-calling path
        return _handle_native_tool_calls(messages, parser.tcs, text, parser.finish_reasons,
                                         malformed_stream_cut_count)

    if forced_final:
        calls, _errors = parse_text_calls_with_errors(text)
        if calls:
            return _handle_forced_final_text_calls(messages, calls, text,
                                                   malformed_call_count)
    else:
        status = _handle_text_tool_calls(messages, text, malformed_call_count)  # text-fallback path
        if status is not None:
            return status

    if forced_final and text.strip() and "length" in parser.finish_reasons:
        if not quiet:
            with write_lock():
                cprint(f"{C.WARNING}  ✂ FORCED FINAL ANSWER HIT TOKEN LIMIT — saved partial answer "
                      f"with truncation marker{RESET}")
        _close_out_partial(messages, text)
        return TURN_DONE

    # finish_reason="length" on a normal text turn: the response budget
    # (NX_MAX_TOKENS, or a server-side cap on some models) ran out mid-answer.
    # That's not a finished reply and not an error — it's a continuation
    # opportunity. The loop appends the partial text as a real assistant turn
    # and asks the model to resume from its exact tail; st.MAX_CONTINUATIONS
    # (counted in continuation_cuts) budgets this so a model that never
    # converges eventually lands at the truncation-marker close-out instead.
    # Native tool_calls cuts never reach here (handled above with their own
    # malformed-args recovery); a text-protocol tag cut mid-args is handled by
    # the malformed-call correction path above for the same reason.
    if not forced_final and "length" in parser.finish_reasons and text.strip():
        # Hand the partial text to the loop without appending it here — the
        # loop decides between continuing (assistant turn + resume note) and
        # closing the partial out with the truncation marker.
        st._LAST_PARTIAL_TEXT = text
        return TURN_LENGTH_CUT

    if not text.strip() and not parser.tcs:
        # parser.tcs in the guard: a stream cut mid-args (server-side abort,
        # a collapse cut after tool-call deltas started, ...) can land here
        # with an in-progress tool call buffered.  Classifying that as an
        # "empty turn" misleads the recovery: the model was actively emitting
        # call args, not stalling.  A populated tcs falls through to the
        # native handler below, whose malformed-args recovery owns it.
        # Silent stall: the model emitted nothing. In a real agentic loop this is
        # almost never a legitimate "done" — it usually follows a short/empty
        # tool result (the model "knows" the answer but produces no tokens), and
        # returning TURN_DONE here drops the user to the chat prompt mid-task with
        # no explanation. Surface it so the loop can nudge + auto-continue instead.
        #
        # Never auto-continue when:
        #   • forced_final — an empty forced answer already means "give up" (the
        #     forced-final block above printed its own message); stopping is right.
        #   • st.EMPTY_TURN_RETRY_LIMIT <= 0 — the user opted out; empty = done.
        #   • recovery already exhausted the budget — escalate to a forced answer
        #     (mirrors the reasoning-loop path) instead of spinning forever.
        if (not forced_final and st.EMPTY_TURN_RETRY_LIMIT > 0
                and empty_turn_count < _subagent_scoped_retry(st.EMPTY_TURN_RETRY_LIMIT)):
            dangling = _last_is_dangling_tool(messages)
            if not quiet:
                with write_lock():
                    cprint(f"{C.WARNING}  ✂ EMPTY TURN — no content and no tool call"
                          f"{'; dangling tool result awaits a follow-up' if dangling else ''}"
                          f"; nudging the model to continue "
                          f"({empty_turn_count + 1}/{_subagent_scoped_retry(st.EMPTY_TURN_RETRY_LIMIT)}){RESET}")
            _maybe_arm_native_escalation()
            _nudge_current_user_turn(messages, EMPTY_TURN_NUDGE)
            return TURN_EMPTY
        if not forced_final and st.EMPTY_TURN_RETRY_LIMIT > 0:
            if not quiet:
                with write_lock():
                    cprint(f"{C.WARNING}  ✂ EMPTY TURN — no content after {empty_turn_count} nudge"
                          f"{'s' if empty_turn_count != 1 else ''}; forcing a final answer{RESET}")
            _nudge_current_user_turn(messages, FORCED_FINAL_NUDGE)
            return TURN_FORCE_FINAL
        return TURN_DONE

    # Text-only reply right after a reasoning-stall nudge explicitly asked the
    # model to emit a tool call ("write_file"/"run_bash" ... "now"). A plain
    # restatement of intent ("Now I'll implement...") with no tool call means
    # the nudge didn't take — treating this as TURN_DONE drops the user to the
    # prompt mid-task, requiring a manual "go ahead" to get the model to
    # actually act. Nudge again instead of accepting it as a final answer.
    #
    # BUT only while the model has not acted since this stall episode began.
    # The last-message check (_last_is_dangling_tool) misses the common
    # layout: the model acts, then stalls once more — the re-nudge is appended
    # as the trailing user message — and then delivers
    # its finished-work summary. st._STALL_EPISODE_ACTED (set by the
    # TURN_TOOL branch of _run_model_turn_loop, cleared when a fresh
    # episode starts) records "acted at any point this episode", so a
    # text-only turn after any mid-episode tool call is accepted as the
    # final summary instead of being re-nudged.
    if (not forced_final and reasoning_loop_cut_count > 0
            and reasoning_loop_cut_count < max(0, st.REASONING_ONLY_RETRY_LIMIT)
            and not st._STALL_EPISODE_ACTED
            and not _last_is_dangling_tool(messages)):
        if not quiet:
            with write_lock():
                cprint(f"{C.WARNING}  ✂ TEXT-ONLY AFTER STALL NUDGE — model described "
                      f"intent but made no tool call; nudging to act "
                      f"(attempt {reasoning_loop_cut_count + 1}/"
                      f"{_subagent_scoped_retry(st.REASONING_ONLY_RETRY_LIMIT)}){RESET}")
        messages.append({"role": "assistant", "content": text})
        _nudge_current_user_turn(
            messages,
            ("A plan is active — keep refining it with plan_edit until it "
             "covers the whole task, then ask the user to run /plan done.")
            if plan_active() else REASONING_STALL_NUDGE)
        return TURN_STALL

    messages.append({"role": "assistant", "content": text})
    return TURN_DONE


# Maximum number of model turns (tool calls + retries) per session before the
# loop breaks.  The real bound on wall-clock work is the batch per-task
# timeout (nx/_batch.py: --timeout / NX_TASK_TIMEOUT_SECONDS, enforced at
# this loop's step boundaries); each turn costs at minimum a few seconds
# (API round-trip + generation + tool
# exec), so even 500 turns in a 900s budget means turns averaging <1.8s — that's
# fast spinning, not productive work.  Individual pathologies (reasoning stalls,
# empty turns, malformed streams) already have their own retry limits.  This cap
# is just a total-work backstop so the loop can't run away *inside* the timeout.
# Overridable via env var.  Set very high (e.g. 99999) to effectively disable.
st.MAX_MODEL_TURNS = _env_int("NX_MAX_MODEL_TURNS", 500)

def _report_length_cut(quiet: bool, used: int, limit: int) -> None:
    """Status line after appending a continuation resume note."""
    if not quiet:
        with write_lock():
            cprint(f"{C.WARNING}  ✂ RESPONSE CUT BY TOKEN LIMIT — continuing ({used}/{limit}){RESET}")


def _report_length_exhausted(quiet: bool, used: int) -> None:
    """Status line when the continuation budget ran out and the partial was closed out."""
    if not quiet:
        suffix = "s" if used != 1 else ""
        with write_lock():
            cprint(f"{C.WARNING}  ✂ RESPONSE STILL HIT TOKEN LIMIT after {used} continuation{suffix} "
                  f"— saved partial answer with truncation marker{RESET}")


def _run_model_turn_loop(messages: list[Message], force_final: bool = False,
                         recovery_sampling: bool = False,
                         on_step: Optional[Callable[[], None]] = None,
                         max_turns_override: Optional[int] = None) -> dict:
    """Runs model_turn() to completion (or MAX_MODEL_TURNS).

    Private recovery state: a per-loop context isolates the tallies, the
    stall flag, the partial text and the malformed-args breaker from any
    concurrently running sub-agent loop (1.2), then folds its final values
    back into the module globals on exit so session-scoped evidence
    survives. Outside the loop the module globals ARE the store (tests,
    REPL, model_turn() called directly)."""
    from . import _recovery
    _recovery.install()
    # Swarm (P2): per-loop notice slots. Without this, two concurrent turn
    # loops (parallel batch, async fan-out under --yolo) stage and flush
    # runtime notes into ONE process-wide str — agent A's finished answer
    # landed in agent B's transcript. Seeded from / folded back into the
    # enclosing loop by _notices, so nesting behaves exactly as before.
    from . import _notices
    _notice_frame = _notices.install_notice_ctx()
    # 0.148.2: bracket the user-turn footer accumulation. Every step of the
    # loop records its numbers (quiet turns record nothing); the finally
    # below flushes ONE aggregated footer when the loop settles — covering
    # all exit statuses (done/esc/cap/error/timeout and the rescue
    # forced-final turns after the loop) with a single call site. Thread-local
    # state, so concurrent sub-agent loops never touch this accumulator.
    from ._turn_footer import begin_user_turn_footer, end_user_turn_footer
    begin_user_turn_footer()
    try:
        return _run_model_turn_loop_inner(messages, force_final, recovery_sampling,
                                          on_step, max_turns_override)
    finally:
        end_user_turn_footer()
        # PF-05 (0.61.0): amortized undo persistence. A turn is the natural
        # boundary — the loop is shared by the REPL, batch tasks and sub-agent
        # loops, so flushing here keeps every driver's crash-durability window
        # at "one turn" instead of re-serializing the whole stack per edit.
        from ._undo import flush_undo_stack
        flush_undo_stack()
        ctx = _recovery.current()
        if ctx is not None:
            _recovery.copy_back_and_reset(ctx)
        _notices.pop_notice_ctx(_notice_frame)


def _run_model_turn_loop_inner(messages: list[Message], force_final: bool = False,
                               recovery_sampling: bool = False,
                               on_step: Optional[Callable[[], None]] = None,
                               max_turns_override: Optional[int] = None) -> dict:
    """Runs model_turn() to completion (or MAX_MODEL_TURNS). Returns a summary
    dict {"status": "done"|"esc"|"cap"|"error", "turns": int} — interactive
    callers ignore it, but headless one-shot mode (nx/_cli.py) uses it to
    report whether the model actually finished vs. hit the turn cap vs. the
    endpoint hard-failing (status "error" → exit_code 4 in batch/one-shot).
    max_turns_override caps the loop below MAX_MODEL_TURNS for scoped loops
    (spawn_agent sub-agents); None = the global cap, as before.
    Batch workers arm a wall-clock deadline via _task_deadline_at() (see
    nx/_batch.py): when it lapses the loop stops at the next step boundary
    and returns a status-"error" summary naming the timeout."""
    max_turns = st.MAX_MODEL_TURNS if max_turns_override is None else max(1, int(max_turns_override))
    steps = 0
    reasoning_loop_cuts = 0
    malformed_stream_cuts = 0
    timed_out_retries = 0  # P3-2: mid-stream-timeout step retries spent
    empty_turn_cuts = 0
    malformed_call_cuts = 0
    continuation_cuts = 0  # finish_reason="length" auto-continues (st.MAX_CONTINUATIONS budget)
    _last_partial_text = ""  # partial answer of the most recent TURN_LENGTH_CUT
    context_retries = 0  # compress+retry after a context-window overrun (once)
    prev_prompt_tokens = -1  # last _LAST_PROMPT_TOKENS this loop has already evaluated
    # Go through the top-level package (not a bare same-module reference) so
    # a test that monkeypatches `nx.model_turn` (to avoid a real model
    # round-trip) takes effect on every call this loop makes.
    import nx as _m
    # Sub-agents: make the LAST permitted turn a forced-final one. Otherwise a
    # model that never learned to stop spends the entire budget exploring —
    # and the cap rescue below then asks it to answer from scratch (or
    # salvages nothing). With the final turn opened as final_answer + forced
    # tool_choice, the agent writes its answer INSIDE the budget.
    sub_loop = _subagent_active()
    # Rescue turns actually consumed inside a sub-loop (stalls, empty turns,
    # malformed streams, collapses). Their budget is clamped to
    # NX_SUBAGENT_RECOVERY_RETRIES, but the clamp only counts WITHIN one
    # failure family — a sub-agent that stalls once AND cuts its stream once
    # still burns 2 of its turns on rescues. If a rescue has already eaten
    # most of a small budget, pre-force the final answer EARLIER so the last
    # remaining turns produce a real answer instead of more recovery re-sampling.
    sub_rescue_turns = 0
    while steps < max_turns:  # cap runaway tool/retry loops (scoped override for sub-agents)
        if sub_loop and not force_final:
            if steps >= max_turns - 1 or (sub_rescue_turns >= max(0, _subagent_scoped_retry(1))
                                          and steps >= max_turns - 2):
                force_final = True
        # v2 usage metrics: the 1-based step number this loop's model_turn /
        # give_up events carry. Thread-local, so a --parallel batch worker's
        # telemetry never reports a sibling's step.
        from ._logging import set_turn_no
        set_turn_no(steps + 1)
        _sent_len = len(messages)
        status = _m.model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                            empty_turn_count=empty_turn_cuts,
                            forced_final=force_final,
                            recovery_sampling=recovery_sampling,
                            malformed_call_count=malformed_call_cuts,
                            step_retries=timed_out_retries)
        # PF-13: any image that was on the wire for that request has now been
        # consumed — drop its base64 so it costs nothing on later turns instead
        # of riding along until an overflow happens to evict it. Only the
        # prefix that was actually sent is stripped (an image appended during
        # the turn just now has not been seen yet).
        _strip_consumed_images(messages, _sent_len)
        if on_step is not None:
            on_step()
        if _task_deadline_exceeded():
            return _task_timeout_summary(steps + 1)
        was_forced = force_final  # TURN_TOOL after a forced-final turn is special-cased below
        force_final = False
        recovery_sampling = False
        # A forced-final turn that ends TURN_DONE with NO new assistant text is
        # the "FORCED FINAL ANSWER EMPTY" path: the model was given its last
        # chance to write the answer and produced none, so the last interim
        # assistant text would be silently promoted to the final report (seen
        # live at 0.103.8 — an async agent reported "done" carrying only an
        # interim note). That is a turn-budget death, not a settled finish:
        # report "cap" so the sub-agent wrapper stamps the answer partial.
        # 0.105.0 steering, 0.106.0 per-agent queues: lines typed mid-turn
        # (captured by the interrupt watcher for the main turn, /tabs for an
        # agent) drain at the step boundary — BEFORE the TURN_DONE return
        # below, so a note queued while the model was finishing still lands.
        # The queue is resolved per-thread: an agent loop (TLS ctx carries its
        # job_id) drains only its own queue and can never steal a parent
        # steering line; the main loop drains the process-global one.
        from ._surfaces import inject_steering
        from ._subagent_tls import _subagent_ctx as _steer_ctx
        _steer_agent = (getattr(_steer_ctx(), "job_id", None)
                        if _steer_ctx() is not None else None)
        _steered = inject_steering(messages, agent=_steer_agent)
        if _steered and _steer_agent is not None:
            # Stamp the job record so the completion note tells the parent
            # its answer reflects mid-run user adjustments.
            try:
                from ._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
                with _ASYNC_LOCK:
                    rec = _ASYNC_AGENTS.get(_steer_agent)
                    if rec is not None:
                        rec["steered"] = (rec.get("steered") or 0) + len(
                            [m for m in messages if m.get("role") == "user"
                             and str(m.get("content", "")).startswith("[Steering note")])
            except Exception:  # noqa: BLE001 — breadcrumb is best-effort
                pass
        if status == TURN_DONE and was_forced and sub_loop \
                and not any(m.get("role") == "assistant"
                            and isinstance(m.get("content"), str) and m["content"].strip()
                            for m in messages[_sent_len:]):
            return {"status": "cap", "turns": steps + 1}
        if status == TURN_DONE:
            # A settled turn: whatever posture produced it works. The next
            # stall, if any, re-derives the escalation from its own evidence.
            st._STALL_FORCE_NATIVE = False
            if _steered and steps + 1 < max_turns:
                # 0.105.0 (main) / 0.106.0 (agents): the user steered while
                # the model was finishing — the note is already in `messages`;
                # keep the turn alive so the adjustment is applied rather
                # than left unread. For an agent this extra turn still
                # respects the sub-loop's forced-final ladder (the branch
                # above decides posture on the next iteration).
                if not _subagent_quiet():
                    with write_lock():
                        cprint(f"{DIM}  ▸ steering applied — continuing turn{RESET}")
                elif _steer_agent is not None:
                    with write_lock():
                        cprint(f"{DIM}  ▸ agent {_steer_agent} steered — continuing{RESET}")
                steps += 1
                continue
            if not _subagent_active() and not _batch_active():
                # Only the parent loop fires NX_VERIFY_CMD — concurrent
                # sub-agents or batch-task threads hitting this line together
                # would race the settle log and double-launch the verify
                # command.
                # Module-attribute call so a test patching nx._launch_verify_if_settled
                # (the historical seam) lands — see
                # test_batch_run.test_verify_and_bg_scan_gated_off_for_batch.
                _launch_verify_if_settled()
            return {"status": "done", "turns": steps + 1}
        if status == TURN_ESC:
            return {"status": "esc", "turns": steps + 1}  # user pressed Esc at an approval → drop to chat input
        if status == TURN_CONTEXT_OVERRUN:
            # The server refused the prompt outright with a context-length 400:
            # nothing was appended to `messages`, so the ONLY way forward is to
            # shrink them. Escalate one tier per refusal: fold ~1/3, then
            # halve the kept tail, then halve again + spill-evict the largest
            # tool results. Exhausting the ladder means even an evicted,
            # heavily folded prompt doesn't fit — report the failure honestly
            # instead of looping.
            steps += 1
            # Escalate one tier per REFUSAL — not per successful fold: tier 1's
            # smaller keep can fold what tier 0's conservative keep couldn't
            # (compress returns None only when there is truly nothing to fold
            # at that keep, so a False tier-0 fold must still open tier 1).
            # The ladder ends at OVERFLOW_FOLD_TIERS exhausted, or when a tier
            # both folded and got refused again (handled by the continue).
            if context_retries < st.OVERFLOW_FOLD_TIERS:
                # 0.103.5: an async fold may be one second from done — wait
                # for it before spending a synchronous fold on the ladder.
                _m._collect_async_fold(messages, wait=2.0)
                _m._compress_for_overrun(messages, tier=context_retries)
                context_retries += 1
                prev_prompt_tokens = -1  # the next settled reading is post-fold; re-evaluate
                continue
            return {"status": "error",
                    "error": ("context window overflow: the prompt exceeded the "
                              f"model's context window and auto-compression "
                              f"could not free enough room ({st._LAST_CONTEXT_OVERRUN})"),
                    "turns": steps}
        if status == TURN_API_ERROR:
            # Endpoint unreachable / API rejected the request outright. Unlike
            # every recovery path above this is terminal: the next open would
            # fail identically (nothing was added to `messages`), so burning
            # turns on it would only delay the same verdict.
            return {"status": "error",
                    "error": f"API request failed: {st._LAST_API_ERROR}",
                    "turns": steps + 1}
        # --- cheap in-loop context guard -----------------------------------
        # Autocompression is not only a between-user-turns REPL mechanism:
        # this loop can run up to MAX_MODEL_TURNS tool iterations, appending
        # assistant + tool messages every time, with nothing to fold between
        # them. A fresh prompt-token reading above the AUTOCOMPRESS_PERCENT
        # threshold folds early — before the next request can be REFUSED with
        # a 400 (the TURN_CONTEXT_OVERRUN path below). Evaluated only on a
        # changed reading so a server that reports no usage can't trigger
        # fold after fold on one stale number; _LAST_PROMPT_TOKENS grows
        # monotonically here (append-only history), so equal readings mean
        # no new data arrived.
        cur_tokens = st._LAST_PROMPT_TOKENS
        if cur_tokens > 0 and cur_tokens != prev_prompt_tokens:
            # Online Context Compact (0.67.0): record the request BEFORE the
            # threshold fold so boundary economics see this turn's size.
            try:
                from ._occompact import record_provider_request
                record_provider_request(cur_tokens)
            except Exception:  # noqa: BLE001
                pass
            _m._maybe_autocompress(messages, cur_tokens)
        prev_prompt_tokens = cur_tokens
        if status == TURN_REDIRECT:
            # Esc-then-type: the user already supplied their next instruction
            # (appended to `messages` by model_turn()) — loop straight into
            # another turn instead of dropping to the REPL prompt.
            steps += 1
            continue
        steps += 1
        if status == TURN_STREAM_CUT:
            malformed_stream_cuts += 1
            sub_rescue_turns += sub_loop
            recovery_sampling = True
            continue
        if status == TURN_EMPTY:
            empty_turn_cuts += 1
            sub_rescue_turns += sub_loop
            recovery_sampling = True  # more entropy to escape the empty-output attractor
            continue
        if status == TURN_MALFORMED_CALL:
            malformed_call_cuts += 1
            sub_rescue_turns += sub_loop
            continue
        if status == TURN_TIMED_OUT_RETRY:
            # Mid-stream timeout with ladder left: burn a bounded retry, not a
            # turn of the model's budget (steps was not incremented — this
            # failure is ours, not the model's) and re-enter the loop. The
            # stream was cut before anything was appended to `messages`, so a
            # clean re-open replays the exact same request.
            timed_out_retries += 1
            delay = _step_backoff_delay(timed_out_retries - 1, None)
            if not _subagent_quiet():
                with write_lock():
                    cprint(f"{DIM}  ↻ step retry {timed_out_retries}/"
                          f"{_step_retry_limit()} in {delay:.1f}s…{RESET}")
            if delay > 0:
                time.sleep(delay)
            continue
        if status == TURN_TIMED_OUT:
            # Ladder exhausted: report the failure honestly. Headless callers
            # map this to status "error" (exit 4) — never "done".
            return {"status": "error",
                    "error": "stream timeout: the server stopped delivering chunks "
                             f"after {timed_out_retries} step retries",
                    "turns": steps + 1}
        if status == TURN_LENGTH_CUT:
            # The response budget (NX_MAX_TOKENS, or a server-side cap) ran out
            # mid-answer. model_turn() appended nothing; it handed the partial
            # text back in st._LAST_PARTIAL_TEXT, mirrored here first.
            # Budgeted by _subagent_scoped_retry(st.MAX_CONTINUATIONS) so
            # a sub-agent's continuation spend narrows with its recovery clamp;
            # once the budget runs out the partial is closed out with the same
            # truncation marker the forced-final path uses, so the transcript
            # never shows a silently-truncated answer. Message order mirrors the
            # text-protocol tool flow: partial assistant text, then a user-side
            # runtime note (see _nudge_current_user_turn).
            _last_partial_text = st._LAST_PARTIAL_TEXT
            limit = max(0, _subagent_scoped_retry(st.MAX_CONTINUATIONS))
            if continuation_cuts < limit:
                continuation_cuts += 1
                messages.append({"role": "assistant", "content": _last_partial_text})
                _nudge_current_user_turn(messages, CONTINUATION_NUDGE)
                _report_length_cut(_subagent_quiet(), continuation_cuts, limit)
                continue
            _close_out_partial(messages, _last_partial_text)
            _report_length_exhausted(_subagent_quiet(), continuation_cuts)
            return {"status": "done", "turns": steps + 1}
        if status == TURN_STALL:
            reasoning_loop_cuts += 1
            sub_rescue_turns += sub_loop
            recovery_sampling = True
            # Full tools retained — the model was nudged to ACT (emit a tool
            # call / write a file) rather than produce a premature final
            # answer.  force_final stays False so model_turn() opens the
            # stream with the complete tool set, not [FINAL_ANSWER_TOOL].
            continue
        if status == TURN_FORCE_FINAL:
            reasoning_loop_cuts += 1
            empty_turn_cuts = 0  # handed off to the forced-final path; reset
            force_final = True
            sub_rescue_turns += sub_loop
            recovery_sampling = True
            continue
        if status == TURN_TOOL:
            # The native-channel escalation served its purpose: a real native
            # tool call just ran, so _handle_native_tool_calls() set
            # _NATIVE_TOOL_CALLS_CONFIRMED (the session-sticky version of this
            # flag). Drop the per-episode one so a later text-protocol turn
            # isn't needlessly forced onto the native channel.
            st._STALL_FORCE_NATIVE = False
            # A real tool call just ran. If a stall-recovery episode is in
            # progress (reasoning_loop_cuts > 0), mark it: any later text-only
            # turn is a legitimate final summary of completed work, not a
            # stalled restatement of intent — even when a further stall has
            # re-nudged in between (making the nudge the last message, the
            # layout the last-message check in model_turn can't see).
            if reasoning_loop_cuts > 0:
                st._STALL_EPISODE_ACTED = True
            # A tool call is real progress for malformed-stream and empty-turn
            # recovery — those counters are about output pathologies, and a
            # successful tool call means the model is producing again.
            #
            # But reasoning_loop_cuts is different: a single trivial tool call
            # between two 36K-char reasoning spirals does NOT mean the model
            # has recovered from its tendency to over-think.  Resetting the
            # counter here lets the model stall → tool → stall → tool → …
            # forever, and the forced-final-answer safety net (attempt 2/2)
            # never fires.  In headless mode (benchmarks) this burns the
            # entire timeout.  Keep the counter so cumulative stalls escalate.
            if was_forced:
                # The forced-final turn didn't wrap up — the model emitted an
                # actionable tool call instead and it just ran (see
                # _handle_forced_final_tool_call). That call is the escape
                # from the stall, but don't full-reset the stall counter:
                # leave it on the last rung so a fresh reasoning-only stall
                # goes straight back to forced-final (which just demonstrably
                # worked for this model) instead of re-climbing the nudge
                # ladder from attempt 1.
                reasoning_loop_cuts = max(
                    0, _subagent_scoped_retry(st.REASONING_ONLY_RETRY_LIMIT) - 1)
            malformed_stream_cuts = 0
            empty_turn_cuts = 0
            continue
    # Loop exited because steps >= max_turns.  In interactive mode the
    # user can simply type another instruction; in headless mode (benchmarks)
    # the session would end with whatever partial work exists.  Give the model
    # one last forced-final turn so it can produce a visible answer from the
    # state so far — but only if we haven't already tried (force_final was
    # already consumed by the loop above).
    if steps >= max_turns and not force_final:
        if not _subagent_quiet():
            with write_lock():
                cprint(f"{C.WARNING}  ⚂ MODEL TURN LIMIT ({max_turns}) — "
                      f"forcing a final answer from current state{RESET}")
        if _task_deadline_exceeded():
            return _task_timeout_summary(steps)
        final_status = _m.model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                   empty_turn_count=empty_turn_cuts,
                   forced_final=True,
                   recovery_sampling=recovery_sampling)
        if on_step is not None:
            on_step()
        if final_status == TURN_CONTEXT_OVERRUN:
            # The rescue prompt itself can overflow after a long run: same
            # escalation ladder as the main loop, one tier per refusal, then
            # give up honestly rather than mislabel a refused answer as a
            # completed one.
            rescue_retries = 0
            while (rescue_retries < st.OVERFLOW_FOLD_TIERS
                   and _m._compress_for_overrun(messages, tier=rescue_retries)):
                rescue_retries += 1
                final_status = _m.model_turn(messages, reasoning_loop_cuts,
                           malformed_stream_cuts, empty_turn_count=empty_turn_cuts,
                           forced_final=True, recovery_sampling=recovery_sampling)
                if on_step is not None:
                    on_step()
                if final_status != TURN_CONTEXT_OVERRUN:
                    break
            if final_status == TURN_CONTEXT_OVERRUN:
                return {"status": "error",
                        "error": ("context window overflow: the prompt exceeded the "
                                  f"model's context window and auto-compression "
                                  f"could not free enough room ({st._LAST_CONTEXT_OVERRUN})"),
                        "turns": steps + 1}
        if final_status == TURN_API_ERROR:
            return {"status": "error",
                    "error": f"API request failed: {st._LAST_API_ERROR}",
                    "turns": steps + 1}
        if final_status in (TURN_TIMED_OUT_RETRY, TURN_TIMED_OUT):
            # A rescue turn that times out mid-stream is also a failure (the
            # ladder is not re-armed here: the main loop's retries were already
            # spent getting to this point).
            return {"status": "error",
                    "error": ("stream timeout: the server stopped delivering "
                              "chunks during the forced-final answer turn"),
                    "turns": steps + 1}
        if final_status == TURN_DONE:
            return {"status": "cap", "turns": steps + 1}
    if _task_deadline_exceeded():
        return _task_timeout_summary(steps)
    return {"status": "cap", "turns": steps}

