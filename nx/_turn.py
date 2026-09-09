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
import threading
import time
from typing import Any, Callable, Optional

from ._openai_client import APIConnectionError, APIError
import httpx
from rich.console import Console
from ._ansi import (
    DIM, RESET, LifeSpinner, C, strip_ansi_for_display,
)
from ._approval import _EscToChat
from ._interrupt import (
    _INTERRUPT_ARMED, _INTERRUPT_RESTORED, _USER_INTERRUPTED, _ensure_interrupt_watcher,
)
from ._metrics import _abbr, _record_usage
from ._special_tokens import (
    LEAKED_TOKEN_RUN_LIMIT, REPEAT_CHUNK_LIMIT, _strip_leaked_tokens,
)
from ._stream_parse import StreamParser
from ._text_fallback import (
    parse_text_calls_with_errors, repair_json_candidates, normalize_tool_args,
)
from ._compress import (
    _nudge_current_user_turn,
    _flush_pending_background_notice, _last_is_dangling_tool,
)
from ._env import _env_float, _env_int
from ._verify import _launch_verify_if_settled
from . import _state as st
from . import (
    FINAL_ANSWER_TOOL, FINAL_ANSWER_TOOL_CHOICE, run_tool, _active_tools,
    REASONING_STALL_NUDGE, FORCED_FINAL_NUDGE, EMPTY_TURN_NUDGE,
    STALL_WRAPUP_NUDGE,
    REASONING_ONLY_TIME_LIMIT, REASONING_CAPTURE_CHARS, _tool_args_parse_error,
    _CACHEABLE_READ_TOOLS, _tool_cache_key,
)
from ._subagent import (_subagent_active, _subagent_quiet, _subagent_scoped_retry,
                        _subagent_stream_cut_limit)
from ._batch import _batch_active
from ._plan import plan_active
from concurrent.futures import ThreadPoolExecutor, as_completed

_say_console = Console()  # renders the assistant's final-answer markdown live

Message = dict[str, Any]  # one entry in the OpenAI-style `messages` array


class _LoggingStream:
    """Iterator wrapper that tees each SSE chunk to llamacpp.log before yielding.
    Uses model_dump so we capture the chunk's full structure (incl. reasoning_content)."""
    def __init__(self, inner: Any, log_file: Any) -> None:
        self._inner = inner
        self._log = log_file

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


TURN_DONE = "done"
TURN_TOOL = "tool"
TURN_STREAM_CUT = "stream_cut"
TURN_FORCE_FINAL = "force_final"
TURN_STALL = "stall"  # reasoning-only stall → nudge to act (full tools retained)
TURN_EMPTY = "empty"  # assistant emitted no content and no tool calls → silent stall
TURN_ESC = "esc"  # user pressed Esc at an approval prompt → drop to chat input
TURN_REDIRECT = "redirect"  # user typed+submitted a new instruction right after Esc
TURN_MALFORMED_CALL = "malformed_call"  # text-protocol tag(s) present but unparseable → one corrective retry
TURN_API_ERROR = "api_error"  # request hard-failed before any chunk (endpoint down, 4xx/5xx) — never retried
TURN_CONTEXT_OVERRUN = "context_overrun"  # 400: prompt exceeded the context window — recoverable via one compress+retry
TURN_LENGTH_CUT = "length_cut"  # finish_reason=length: the model's response budget ran out mid-stream — auto-continue it
TURN_TIMED_OUT = "timed_out"  # mid-stream timeout after the step-retry ladder was exhausted — a real failure, scored as one
TURN_TIMED_OUT_RETRY = "timed_out_retry"  # mid-stream timeout with ladder left — the loop backs off and re-opens the same request
TRUNCATION_MARKER = "\n\n[Truncated by token limit before completion.]"


def _close_out_partial(messages: list[Message], partial: str) -> None:
    """Record a length-cut partial answer, closed out with the truncation marker.

    Used when the answer can't be continued: a forced-final turn was cut, or
    the auto-continuation budget (st.MAX_CONTINUATIONS) is exhausted. One
    constant + one helper keep the transcript's truncation spelling uniform."""
    messages.append({"role": "assistant", "content": partial.rstrip() + TRUNCATION_MARKER})

# Note appended after a partial answer when finish_reason="length" fired: the
# model exhausted its response budget (NX_MAX_TOKENS, or a server-side cap on
# qwen3.8-flash-next et al.) and stopped mid-sentence. Continuation keeps the
# partial text as a real assistant turn, so the model resumes from its own
# exact tail instead of restarting the answer. "If the reply was actually
# complete, say (done)" bounds a model that wants to emit something anyway.
CONTINUATION_NUDGE = (
    "Your previous reply was cut off by the per-response token limit — it ended "
    "mid-stream. Resume your reply from the exact point it stopped. Do not repeat "
    "or summarize what you already wrote; just continue the text seamlessly. "
    "If the previous reply was actually complete, reply with the single word "
    "(done) and nothing else.")


def _ctx_field(prompt_n: int) -> str:
    """The colored `ctx` field for the stats footer: `<cur>/<max> ctx`.
    The max context window is cyan; the current context is colored by
    utilization — green under 30%, yellow under 60%, red at or above.
    When the max isn't known yet (probe still pending or unreachable)
    the field degrades to `<cur> ctx` with no color, and a background
    probe is kicked off so a later turn can fill in the max."""
    cur = _abbr(prompt_n)
    if st.ACTIVE is None:
        return f"{cur} ctx"
    mx = st.ACTIVE._context_window
    if not isinstance(mx, int) or mx <= 0:
        _ensure_ctx_probe(st.ACTIVE)
        return f"{cur} ctx"
    mx_s = _abbr(mx)
    ratio = prompt_n / mx if mx > 0 else 0.0
    if ratio < 0.30:
        color = C.GREEN
    elif ratio < 0.60:
        color = C.WARNING
    else:
        color = C.ERROR
    # RESET after each colored run, then re-apply DIM so the rest of the
    # footer (joined around this field) stays dim.
    return f"{color}{cur}{RESET}{DIM}/{C.STATUS_BAR}{mx_s}{RESET}{DIM} ctx"


_CTX_PROBE_STARTED = set()  # id(Source) currently being probed (avoid dupe threads)


def _ensure_ctx_probe(src: Any) -> None:
    """Start a background thread to resolve a Source's max context window if
    it isn't already known, it has not already missed, and no probe is in
    flight for it. The result lands in src._context_window; the footer picks
    it up on a later turn. Idempotent via _CTX_PROBE_STARTED (cleared when the
    thread finishes).

    A local server's metadata endpoints can 503 "Loading model" for tens of
    seconds at cold start (the weights are still mmapping) even while a chat
    slot is already serving. A retry must be explicitly requested after that
    state changes: retrying automatically is unsafe because some servers
    silently clamp the oversized chat probe and may generate until their
    context limit instead of returning a 400.

    Shares _capability.STARTUP_PROBE_LOCK with the capability probe so the two
    background probes don't both queue a request against a single-slot local
    server ahead of the user's actual first prompt."""
    if getattr(src, "_context_window_probe_attempted", False):
        return
    sid = id(src)
    if sid in _CTX_PROBE_STARTED:
        return
    _CTX_PROBE_STARTED.add(sid)

    def _work() -> None:
        try:
            if isinstance(src._context_window, int) and src._context_window > 0:
                return  # resolved by another path (e.g. a /source switch)
            from ._capability import STARTUP_PROBE_LOCK
            with STARTUP_PROBE_LOCK:
                if isinstance(src._context_window, int) and src._context_window > 0:
                    return
                src.resolve_context_window()
        except Exception:
            pass  # never raise from a background probe
        finally:
            _CTX_PROBE_STARTED.discard(sid)

    t = threading.Thread(target=_work, daemon=True, name="nx-ctx-probe")
    t.start()


def _capture_last_prompt_tokens(usage: Any, timings: Optional[dict]) -> None:
    """Cache the just-seen prompt-token count in st._LAST_PROMPT_TOKENS and the
    prompt-cache hit fraction in st._LAST_CACHE_HIT.

    Split out of _print_stats_footer so it can run on EVERY settled stream —
    including the loop_cut give-up paths (collapse / exhausted reasoning-stall
    retries) that return before the footer is ever printed. Those turns still
    grow `messages` (captured reasoning, runtime-note nudges), so skipping this
    left _maybe_autocompress() checking a stale, smaller token count right when
    autocompress was most needed.
    """
    prompt_n = 0
    cache_n = 0
    reported = False  # server actually sent a cache breakdown this call?
    if timings and timings.get("predicted_n"):
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        reported = True
    elif usage and (usage.prompt_tokens or 0):
        prompt_n = int(usage.prompt_tokens or 0)
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None and getattr(ptd, "cached_tokens", None) is not None:
            cache_n = int(ptd.cached_tokens)
            reported = True
    if prompt_n:
        st._LAST_PROMPT_TOKENS = prompt_n
        # Fraction of the prompt served from the server's KV/prompt cache.
        # A steady high value here is the regression signal for append-only
        # discipline (memory_write cache keys, /undo-style message edits): a
        # silent prefix bust shows up as this collapsing to ~0.
        #
        # "Server reported 0" and "server reported nothing" are different
        # facts, so they render differently: 0.0 (red, a real signal) vs None
        # (no `cached` field at all). Several OpenAI-compatible backends
        # (vLLM, some gateways) omit prompt_tokens_details entirely whenever
        # nothing was served from cache — or always — and rendering that as
        # 0% cached turns the footer into a permanent false alarm.
        st._LAST_CACHE_HIT = (min(1.0, cache_n / prompt_n)
                              if reported else None)


def _cache_hit_field(hit: Optional[float]) -> str:
    """The colored `cached: N%` field for the stats footer. Hit-rate color:
    green at/above 75%, yellow at/above 40%, red below — a cold (first-turn)
    or busted prefix is exactly what you want to notice. None (server doesn't
    report cache numbers) renders nothing."""
    if hit is None:
        return ""
    pct = hit * 100.0
    if pct >= 75:
        color = C.GREEN
    elif pct >= 40:
        color = C.WARNING
    else:
        color = C.ERROR
    return f"{color}{pct:.0f}%{RESET}{DIM} cached"


def _tps_field(tps: float, trusted: bool) -> str:
    """The colored `tok/s` footer field. Server-reported paces (llama.cpp
    timings) render bare; client-side estimates render with a `~` whenever the
    decode window was too short to trust, because five decimals of precision
    on a 0.1s span is a lie about noise, not a measurement."""
    num = f"{C.STATUS_BAR}~{tps:5.1f}" if not trusted else f"{C.STATUS_BAR}{tps:5.1f}"
    return f"{num}{RESET}{DIM} tok/s"


def _window_tps(gen_n: float, window: float) -> float:
    """gen_n tokens over a decode window, 0.0-safe."""
    return gen_n / window if window > 0 else 0.0


def _print_stats_footer(usage: Any, timings: Optional[dict], streamed_chars: int, t_first: float | None,
                         elapsed: float, text: str, tcs: dict, messages: Optional[list[Message]] = None,
                         parser: Optional[Any] = None, quiet: bool = False) -> None:
    """Render the per-turn stats footer.

    Prefers llama.cpp `timings` (server-measured, real tok/s) when present;
    falls back to the standard streaming `usage` object (OpenAI/Z.ai/vLLM)
    divided by the client-side generation window (parser.gen_window(): the
    first→last token span, NOT wall-clock — wall-clock includes the prefill
    head and the render/teardown tail, which is what made short turns and
    just-read-a-file turns swing wildly); then to a char-based estimate; then
    to wall-clock only. Untrustworthy windows (below _stream_parse's
    _TPS_TRUST_WINDOW) render with a `~` so a burst-delivered short generation
    doesn't masquerade as a precise measurement.
    Prompt-token caching itself lives in _capture_last_prompt_tokens(), called
    separately so it also runs on turns that never reach this footer.
    """
    if quiet:
        return
    # 0.30.0: one footer field for the self-healing work this turn — args
    # repairs applied + corrective retries — so the knobs are tunable in
    # practice ("2 args repaired · 1 retry"). Folded into every footer branch.
    # (st._TALLY_* route through nx/_recovery.py's descriptors: the running
    # loop's private tally, or the module globals outside any loop.)
    tally = ""
    if st._TALLY_ARG_REPAIRS or st._TALLY_RETRIES:
        bits = []
        if st._TALLY_ARG_REPAIRS:
            bits.append(f"{st._TALLY_ARG_REPAIRS} arg{'s' if st._TALLY_ARG_REPAIRS != 1 else ''} repaired")
        if st._TALLY_RETRIES:
            bits.append(f"{st._TALLY_RETRIES} retr{'ies' if st._TALLY_RETRIES != 1 else 'y'}")
        tally = " · " + " · ".join(bits)
    if timings and timings.get("predicted_n"):
        prompt_n = timings.get("prompt_n", 0)
        gen_n = timings["predicted_n"]
        tps = timings.get("predicted_per_second", 0)
        ctx = _ctx_field(prompt_n)
        prompt_ms = timings.get("prompt_ms", 0)
        ttft = (prompt_ms / 1000.0) if prompt_ms else t_first
        parts = [f"{C.STATUS_BAR}{gen_n}{RESET}{DIM} tok", _tps_field(tps, True), ctx]
        if st.MODEL:
            parts.append(f"{C.STATUS_BAR}{st.MODEL}{RESET}")
        cached = _cache_hit_field(st._LAST_CACHE_HIT)
        if cached:
            parts.append(cached)
        if ttft:
            parts.append(f"{C.STATUS_BAR}{ttft*1000:4.0f}{RESET}{DIM}ms ttft")
        parts.append(f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall")
        print(f"{DIM}  └ {(f'{RESET}{DIM} · ').join(parts)}{RESET}" + tally)
    elif usage and (usage.completion_tokens or 0):
        gen_n = usage.completion_tokens or 0
        prompt_n = usage.prompt_tokens or 0
        window, trusted = (parser.gen_window(elapsed) if parser is not None
                           else (max(0.0, (elapsed - t_first) if t_first else elapsed), False))
        ctx = _ctx_field(prompt_n)
        parts = [f"{C.STATUS_BAR}{gen_n}{RESET}{DIM} tok",
                 _tps_field(_window_tps(gen_n, window), trusted), ctx]
        if st.MODEL:
            parts.append(f"{C.STATUS_BAR}{st.MODEL}{RESET}")
        cached = _cache_hit_field(st._LAST_CACHE_HIT)
        if cached:
            parts.append(cached)
        if t_first:
            parts.append(f"{C.STATUS_BAR}{t_first*1000:4.0f}{RESET}{DIM}ms ttft")
        parts.append(f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall")
        print(f"{DIM}  └ {(f'{RESET}{DIM} · ').join(parts)}{RESET}" + tally)
    elif streamed_chars > 0:
        gen_n = max(1, streamed_chars // 4)
        window, trusted = (parser.gen_window(elapsed) if parser is not None
                           else (max(0.0, (elapsed - t_first) if t_first else elapsed), False))
        parts = [f"≈{C.STATUS_BAR}{_abbr(gen_n)}{RESET}{DIM} tok",
                 _tps_field(_window_tps(gen_n, window), trusted),
                 f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall"]
        print(f"{DIM}  └ {(f'{RESET}{DIM} · ').join(parts)}{RESET}" + tally)
    elif text or tcs:
        print(f"{DIM}  └ {elapsed:4.1f}s wall{RESET}" + tally)


def _handle_forced_final_tool_call(messages: list[Message], tcs: dict, text: str,
                                   finish_reasons: Any,
                                   malformed_stream_cut_count: int) -> str:
    """Under forced-final mode, extract the `final_answer` tool call (if any) the
    model emitted and render its answer as visible assistant text. Any OTHER
    tool call is executed through the normal native path (its approval flow
    still gates mutations) and TURN_TOOL is returned so the loop keeps going —
    the forced-final request is a nudge, not a straitjacket, and discarding a
    real tool call here threw away the very action the stall ladder had finally
    coaxed out of the model and dropped the user to the prompt mid-task."""
    quiet = _subagent_quiet()
    ordered = [tcs[i] for i in sorted(tcs)]
    for c in ordered:
        if c["name"] != "final_answer":
            continue
        try:
            args = json.loads(c["args"] or "{}")
        except json.JSONDecodeError:
            args = {}
        answer = str(args.get("answer") or "").strip()
        if answer:
            if not quiet:
                print(f"{C.TEXT}{strip_ansi_for_display(answer)}{RESET}")
            messages.append({"role": "assistant", "content": answer})
            return TURN_DONE
    actionable = {i: c for i, c in tcs.items() if c["name"] != "final_answer"}
    if actionable:
        names = ", ".join(c["name"] or "tool" for c in actionable.values())
        if not quiet:
            print(f"{C.WARNING}  ✂ FORCED FINAL — model emitted {names} instead of "
                  f"final_answer; executing and continuing the turn{RESET}")
        return _handle_native_tool_calls(messages, actionable, text, finish_reasons,
                                         malformed_stream_cut_count)
    if not quiet:
        print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER EMPTY — returning to chat input{RESET}")
    return TURN_DONE


def _handle_forced_final_text_calls(messages: list[Message], calls: list[tuple[str, dict]],
                                    text: str, malformed_call_count: int = 0) -> str:
    """Text-fallback counterpart to _handle_forced_final_tool_call(): under forced-final
    mode a text-protocol model can still emit tool_call blocks. A final_answer block
    is rendered as visible text; any OTHER tool call runs through the normal text-protocol
    path (whose approval flow still gates arbitrary tools) instead of being discarded —
    discarding it threw away the action the stall ladder had just coaxed out of the model
    and dropped the user to the prompt mid-task."""
    quiet = _subagent_quiet()
    for n, a in calls:
        if n != "final_answer":
            continue
        answer = str(a.get("answer") or "").strip()
        if answer:
            if not quiet:
                print(f"{C.TEXT}{strip_ansi_for_display(answer)}{RESET}")
            messages.append({"role": "assistant", "content": answer})
            return TURN_DONE
        break  # empty final_answer — let any remaining actionable calls decide
    actionable = [(n, a) for n, a in calls if n != "final_answer"]
    if actionable:
        names = ", ".join(n or "tool" for n, _ in actionable)
        if not quiet:
            print(f"{C.WARNING}  ✂ FORCED FINAL — model emitted {names} instead of "
                  f"final_answer; executing and continuing the turn{RESET}")
        delegated = _handle_text_tool_calls(messages, text, malformed_call_count,
                                            exclude={"final_answer"})
        return delegated if delegated is not None else TURN_DONE
    if not quiet:
        print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER EMPTY — returning to chat input{RESET}")
    return TURN_DONE


def _run_read_only_parallel(calls: list[tuple[str, dict[str, Any]]]) -> dict[int, str]:
    """Execute 2+ read-only, approval-free tool calls concurrently.

    The common agent turn is 'read 3 files + grep' — strictly sequential
    dispatch burns wall-clock on I/O that could overlap. Read-only cacheable
    tools have no approval gate, no state mutation, and no output ordering
    requirement beyond the Observation order, so a thread pool is safe here.
    Mutating/prompting calls keep the sequential path (approval UX depends on
    it). Results print in original order after all complete; observations map
    back by index so the model sees the same transcript shape as before.

    'Print in original order' is what this function did NOT do until 0.31.0:
    each worker called run_tool() non-quiet, so N threads interleaved their own
    tool boxes while N spinner threads cleared lines one another had just
    drawn. Workers now run quiet and record a ToolBlock instead (nx/_toolblocks);
    this thread — the only one owning the terminal — draws the whole turn as ONE
    group box, in call order, with a single spinner and one elapsed time. A
    quiet caller (async agent / batch task) records but never renders."""
    results: dict[int, str] = {}
    jobs = [(i, n, a) for i, (n, a) in enumerate(calls)]
    workers = min(8, len(jobs))
    blocks_by_index: dict[int, list] = {}

    # Pool threads inherit nothing from the dispatching thread's quiet state
    # (thread-local by design), which is why a quiet loop's fan-out had to
    # re-apply quiet in every worker. Since 0.31.0 the worker is ALWAYS quiet,
    # interactive parent included: "quiet" is how a thread says "I do not own
    # the terminal", which is exactly true of a pool worker — its output
    # reaches the user through the recorder instead of straight to stdout.
    from ._progress import ProgressCounter
    from ._subagent import _subagent_quiet, _ASYNC_QUIET_SLOTS
    from ._toolblocks import (attach_recorder, detach_recorder, group_spinner,
                              render_group)
    render = not _subagent_quiet()
    started = time.time()
    # One activity line for the whole group, in place of one spinner thread per
    # call erasing the lines its siblings had just printed. Since 0.35.0 the
    # line is live (elapsed + n/m landed): a group of 3 that had finished 2 of
    # its calls looked exactly like one that had finished none.
    counter = ProgressCounter(len(jobs))
    spinner = group_spinner(f"running {len(jobs)} tool calls", counter=counter) if render else None
    if spinner is not None:
        spinner.start()

    def _one(job: tuple[int, str, dict[str, Any]]) -> tuple[int, str]:
        i, n, a = job
        _ASYNC_QUIET_SLOTS.quiet = True
        rec = attach_recorder()
        try:
            return i, run_tool(n, a)
        except _EscToChat:
            raise  # read-only tools can't prompt, but propagate defensively
        except Exception as e:  # noqa: BLE001 — mirror run_tool's error surfacing
            return i, f"ERROR: {type(e).__name__}: {e}"
        finally:
            # Distinct keys per worker; the parent only reads this after the
            # pool has joined, so no lock is needed.
            blocks_by_index[i] = rec.blocks
            detach_recorder()
            _ASYNC_QUIET_SLOTS.quiet = False

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # as_completed, not ex.map: map hands back results in SUBMISSION
            # order, so with a slow first call the live counter would sit at
            # 0/3 while two calls had already landed. as_completed reports each
            # call as it actually finishes, which is the number being shown.
            # Results are keyed by index, so consumption order is free here.
            futs = [ex.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                i, res = fut.result()
                counter.bump()
                results[i] = res
    finally:
        if spinner is not None:
            spinner.stop()
        # Render even on the exception path (an Esc mid-fan-out, a tool that
        # raised past run_tool's own folding): the calls that DID happen have
        # already happened, and dropping their lines would misreport the turn.
        if render:
            render_group(
                [b for i in range(len(jobs)) for b in blocks_by_index.get(i, [])],
                time.time() - started, st.TOOL_OUTPUT_VISIBLE)
            # `/tooloutput last` replays st._LAST_TOOL_*, which each worker set
            # from whichever thread finished last. Re-point it at the group's
            # LAST call so "last" means the last thing the user just saw, not
            # an artifact of thread scheduling.
            _last = results.get(len(jobs) - 1)
            if isinstance(_last, str):
                st._LAST_TOOL_NAME = calls[-1][0]
                st._LAST_TOOL_OUTPUT = _last
    return results


def _all_read_only(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    """True when every call in the turn is a cacheable read-only tool (the
    same set run_tool() result-caches — no approval gate, no mutation)."""
    import nx as _m
    return bool(calls) and all(n in _m._CACHEABLE_READ_TOOLS for n, _ in calls)


def _write_capable_agent(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    """True when any call in the turn is a spawn_agent that will run with the
    write tier (NX_SUBAGENT_WRITE=1). Those agents' tool bodies hit the real
    _confirm() prompt, so they must stay on the sequential path — overlapping
    them would interleave two agents' approval prompts on one terminal."""
    if not any(n == "spawn_agent" for n, _ in calls):
        return False
    from ._subagent import _write_tier_enabled
    return _write_tier_enabled()


def _should_parallelize(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    """True when this turn's 2+ calls may run concurrently on the thread pool.

    Two shapes qualify:
      • the original all-cacheable-read turn ('read 3 files + grep'), and
      • a fan-out turn with spawn_agent calls — each agent is an independent
        messages list, its toolset is restricted, and nothing it runs prompts,
        so overlapping them is the whole point of delegating. Reads may ride
        along in the same turn (all read-only, no ordering requirements).

    Single calls, mutating/prompting tools, any turn while already inside
    a sub-agent (no nested fan-out), and WRITE-TIER fan-outs stay sequential:
    a write-capable sub-agent's bodies run the parent's real _confirm(), so
    two of them in flight would interleave approval prompts.

    Since 0.31.0 this gate carries a second load-bearing meaning: a fan-out
    worker runs quiet (it must not own the terminal), and _confirm() auto-DENIES
    on a quiet thread. So anything eligible here must also be incapable of
    needing an approval prompt — true for _CACHEABLE_READ_TOOLS and read-only
    spawn_agent, and exactly why the write tier stays sequential."""
    if len(calls) < 2:
        return False
    import nx as _m
    if _subagent_active():
        return False  # a sub-agent never fans out further
    spawn = sum(1 for n, _ in calls if n == "spawn_agent")
    if spawn:
        if _write_capable_agent(calls):
            return False  # write-tier agents prompt — never overlap them
        return all(n in _m._CACHEABLE_READ_TOOLS or n == "spawn_agent" for n, _ in calls)
    return _all_read_only(calls)


def _prefetch_cacheable_tools(ordered: list[dict], parsed_args: list[dict]) -> None:
    """Warm the read_file/list_dir cache concurrently for this turn's calls.

    Kept for the MIXED-call case: when a turn combines mutating and read-only
    calls, reads still run sequentially inside run_tool(), so warming the
    cache in parallel here keeps those hits instant. When EVERY call is a
    cacheable read, _run_read_only_parallel() handles dispatch directly and
    this prefetch is skipped (it would double-execute the reads).

    Each job dispatches through run_tool() — NOT the tool body directly — so a
    mixed turn's prefetched reads cross the same gates as every other call:
    the plan-mode gate, the NX_PRE_TOOL_HOOK veto (a blocked read is never
    opened, and its vetoed args never enter st._TOOL_RESULT_CACHE), and the
    NX_POST_TOOL_HOOK audit record. run_tool() populates the cache itself for
    cacheable reads, so this helper no longer stores entries by hand; the
    sequential run_tool() later in the turn is then a cache hit."""
    jobs = []
    seen_keys = set()
    for c, args in zip(ordered, parsed_args):
        if c["name"] not in _CACHEABLE_READ_TOOLS:
            continue
        cache_key = _tool_cache_key(c["name"], args)
        if cache_key in seen_keys or cache_key in st._TOOL_RESULT_CACHE:
            continue
        seen_keys.add(cache_key)
        jobs.append((c["name"], args))
    if len(jobs) < 2:
        return

    # The sequential run_tool() pass later in this turn renders each call's tool
    # box itself (cache hits included), so the prefetch stays invisible: force
    # the quiet slot in every pool worker, exactly as _run_read_only_parallel
    # does for a quiet loop, so a pool thread never prints a box / spinner /
    # pre-hook veto line onto the shared terminal (thread-local quiet is not
    # inherited — it must be re-applied here).
    from ._subagent import _ASYNC_QUIET_SLOTS

    def _run(job: tuple[str, dict[str, Any]]) -> None:
        name, args = job
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            # Applies the plan gate, the pre-hook veto, both hooks, and stores
            # the cache entry itself for cacheable reads. A veto/error simply
            # returns the message string; the sequential run_tool() call later
            # in the turn re-evaluates from the same state and surfaces it.
            run_tool(name, args)
        except _EscToChat:
            raise  # read-only tools can't prompt, but propagate defensively
        except Exception:
            return  # let the normal run_tool() call surface/report the error
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        list(ex.map(_run, jobs))


# Where the model should go when THIS tool's args keep getting truncated and
# the breaker drops it: the nearest sibling that can do the same job. A tool
# not listed here degrades the nudge to "use one of: <remaining tools>"
# (computed live, so it never names a tool the breaker itself dropped).
FALLBACK_FOR = {
    "multi_edit": "edit_file", "apply_patch": "edit_file", "sed_edit": "edit_file",
    "read_image": "read_file", "delete_file": "edit_file",
    "plan_edit": "edit_file", "wait_background": "read_file",
    "memory_write": "memory_search", "memory_delete": "memory_search",
    "search_files": "list_dir", "list_dir": "search_files",
    "todo_write": "write_file",
}


def _repair_log(kind: str, what: str) -> None:
    """0.30.0 observability: every applied repair and every repair-rejection
    gets one JSONL record in the session log (grep-able when tuning
    NX_ARG_REPAIR / NX_MALFORMED_TOOL_BREAKER). Best-effort — never raises."""
    try:
        from ._logging import _log_event
        _log_event("repair", {"kind": kind, "what": what})
    except Exception:
        pass


def _malformed_args_nudge(tool_name: str, broken_now: bool) -> str:
    """Retry nudge for a malformed/truncated native tool-call args turn.

    broken_now=True means the circuit breaker just dropped the tool from the
    session's tool list (this endpoint keeps truncating that tool's args), so
    the model must switch tactics instead of re-emitting the doomed call."""
    head = "Your previous native tool call had malformed or truncated JSON arguments"
    if broken_now:
        # 0.29.9: the suggestion derives from the dropped tool ("use plain
        # edit_file calls" is nonsense when edit_file is the dropped one) —
        # a known sibling when one exists, otherwise whatever's still offered.
        sibling = FALLBACK_FOR.get(tool_name)
        if sibling:
            alt = f"use {sibling} for this step instead"
        else:
            import nx as _m
            remaining = [t["function"]["name"] for t in _m._active_tools()]
            alt = ("use one of: " + ", ".join(remaining)) if remaining else \
                "accomplish the step without tool calls this turn"
        return (head + f" again. {tool_name} has now been removed from the tool "
                "list for the rest of this session because this endpoint keeps "
                "truncating its arguments. Accomplish the same step a different "
                f"way: {alt}, and emit only that one complete "
                "tool call with no surrounding prose.")
    return (head + " again. Repeating the identical call has failed twice now, "
            "so the tool call itself is the problem — pick a different way to "
            "accomplish the same step: a single edit_file call instead of "
            "multi_edit, write_file, or a smaller edit, and emit only that one "
            "complete tool call with no surrounding prose.")


def _salvage_truncated_args(raw: str) -> dict | None:
    """0.30.0, CARVE-OUT — read-only tools only, and only when the stricter
    required-args check passes at dispatch. WHY NOT EVERYWHERE: closure/bracket
    repair INVENTS content. On write_file/edit_file/multi_edit/apply_patch the
    truncated tail is usually the payload itself (cutting a "content" string
    mid-line and closing the braces would WRITE a silently-truncated file —
    exactly the degenerate-output-into-history failure this stack exists to
    prevent), so those keep the discard-and-nudge-smaller path unconditionally.
    On the cacheable read tools the args are small and structural (a path, an
    offset) and a cut mid-path is at worst a not-found; the completed-JSON
    guess can only ever dispatch when the required args survived intact, which
    _argrepair.check_required() enforces in run_tool(). Never raises."""
    if not isinstance(raw, str) or not raw:
        return None
    s = raw
    if s.count('"') % 2 == 1:
        # Odd quote count: the cut landed inside a string literal. Trim to the
        # last complete "key": pair boundary by dropping the trailing fragment
        # is not possible honestly — refuse instead of inventing a value.
        return None
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if stack:
                stack.pop()
    if in_str or not stack:
        return None
    closer = {"[": "]", "{": "}"}
    try:
        candidate = s + "".join(closer[c] for c in reversed(stack))
        obj = json.loads(candidate)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _handle_native_tool_calls(messages: list[Message], tcs: dict, text: str, finish_reasons: Any,
                              malformed_stream_cut_count: int) -> str:
    """Execute the native tool-calling path for one streamed turn.

    Parses each call's JSON args (retrying cleanly on malformed args rather
    than executing `{}`; after repeated malformed args for the same tool, a
    circuit breaker drops that tool from the session's tool list), records
    the assistant(tool_calls) turn, runs each
    tool, and appends matching tool results. If the user presses Esc at an
    approval, the remaining calls are marked CANCELLED/SKIPPED so the message
    history stays valid. Returns a TURN_* status for model_turn.

    0.30.0 carve-out: a truncated args payload for a CACHEABLE READ tool gets
    ONE bracket-closure salvage attempt before the nudge — see
    _salvage_truncated_args for why this is deliberately NOT offered to the
    write tools (closure repair would invent file content)."""
    from ._dispatch import _CACHEABLE_READ_TOOLS, _breaker_views, _breaker_broken
    st._NATIVE_TOOL_CALLS_CONFIRMED = True  # server is actually using native tool_calls this session
    quiet = _subagent_quiet()
    ordered = [tcs[i] for i in sorted(tcs)]
    parsed_args = []
    parse_error = None
    for c in ordered:
        raw = c["args"] or "{}"
        obj = None
        err = None
        for candidate in repair_json_candidates(raw):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError as e:
                err = e
        if obj is None and c.get("name") in _CACHEABLE_READ_TOOLS:
            # 0.30.0 carve-out: one bracket-closure salvage for the read tools
            # (see _salvage_truncated_args for why writes never get this).
            salvaged = _salvage_truncated_args(raw)
            if salvaged is not None:
                obj = salvaged
                _repair_log("salvaged_truncated_args",
                            f"{c.get('name')}: closure repair accepted "
                            f"({len(raw)} -> {len(json.dumps(salvaged))} chars)")
        if obj is not None:
            # 0.29.5: servers occasionally double-encode the arguments object
            # (a JSON string) or emit a list of {name, value} pairs — both die
            # with "argument after ** must be a mapping" in run_tool. Fold
            # them here; anything else stays a parse failure (nudge+breaker).
            folded = normalize_tool_args(obj)
            if isinstance(folded, dict):
                obj = folded
            elif obj is not None:
                err = json.JSONDecodeError(
                    f"arguments must be a JSON object, got {type(obj).__name__}", "", 0)
                obj = None
        if obj is None:
            parse_error = (c, err)
            break
        parsed_args.append(obj)
    if parse_error is not None:
        c, err = parse_error
        if err is None:
            err = json.JSONDecodeError("no candidates", "", 0)
        detail = _tool_args_parse_error(c, err, finish_reasons)
        tool_name = c.get("name") or ""
        count = 0
        if st.MALFORMED_TOOL_BREAKER_LIMIT > 0 and tool_name:
            # _breaker_views() first (0.40): it stamps which endpoint this
            # evidence belongs to and drops any streak carried over from a
            # different one, so the count below is this endpoint's alone.
            failures, broken = _breaker_views()
            count = failures.get(tool_name, 0) + 1
            failures[tool_name] = count
            if count >= st.MALFORMED_TOOL_BREAKER_LIMIT:
                broken.add(tool_name)
        broken_now = _breaker_broken(tool_name)
        st._TALLY_RETRIES += 1  # routes through _recovery's descriptors
        _repair_log("malformed_args", f"{tool_name}: {detail}")
        if st.MALFORMED_TOOL_BREAKER_LIMIT > 0 and tool_name and count >= st.MALFORMED_TOOL_BREAKER_LIMIT:
            _repair_log("breaker_open", f"{tool_name}: {count} malformed attempts")
        retry_limit = _subagent_stream_cut_limit(max(0, st.MALFORMED_STREAM_RETRY_LIMIT))
        if malformed_stream_cut_count >= retry_limit:
            if not quiet:
                print(f"{C.ERROR}  ✗ malformed tool call after "
                      f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
                print(f"{DIM}    {detail}{RESET}")
            return TURN_DONE
        if not quiet:
            print(f"{C.WARNING}  ✂ MALFORMED TOOL CALL — {detail}; "
                  f"discarded partial tool args and retrying cleanly "
                  f"({malformed_stream_cut_count + 1}/{retry_limit}){RESET}"
                  + (f"\n  ⚡ {tool_name} dropped from the tool list for this "
                     f"session — {count} malformed attempts; the endpoint keeps "
                     f"truncating its arguments" if broken_now else ""))
        if malformed_stream_cut_count > 0 or broken_now:
            # A repeat is evidence the same tool call keeps dying the same way
            # (the original failure was truncated at the same spot again). The
            # generic "retry cleanly" nudge produced 7 identical re-emits in
            # one observed session — say instead to change the action.
            nudge = _malformed_args_nudge(tool_name, broken_now)
        else:
            nudge = (
                "Your previous native tool call had malformed or truncated JSON "
                "arguments, so it was discarded before execution. Retry the same "
                "task from the current conversation state, but emit a complete, "
                "valid tool call with all required arguments. For large file "
                "writes, emit only the tool call and skip any explanatory prose.")
        _nudge_current_user_turn(messages, nudge)
        return TURN_STREAM_CUT
    # All-read-only turns and 2+ spawn_agent fan-outs dispatch concurrently
    # (the 'read 3 files + grep' / 'investigate A and B at once' patterns);
    # other turns keep the sequential contract and only prefetch.
    as_calls = [(c["name"], args) for c, args in zip(ordered, parsed_args)]
    if _should_parallelize(as_calls):
        parallel_results = _run_read_only_parallel(as_calls)
        messages.append({"role": "assistant", "content": text or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
            for c in ordered]})
        for idx in range(len(ordered)):
            messages.append({"role": "tool", "tool_call_id": ordered[idx]["id"],
                             "content": parallel_results[idx]})
        _flush_pending_background_notice(messages)
        return TURN_TOOL
    _prefetch_cacheable_tools(ordered, parsed_args)
    messages.append({"role": "assistant", "content": text or None, "tool_calls": [
        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
        for c in ordered]})
    esc_action = None
    for idx, (c, args) in enumerate(zip(ordered, parsed_args)):
        try:
            result = run_tool(c["name"], args)
        except _EscToChat as e:
            esc_action = e.args[0] if e.args else c["name"]
            # Record this call as cancelled, and fill in results for any
            # remaining tool_calls so the message history stays valid
            # (every assistant tool_call needs a matching tool result,
            # or the chat template rejects the context on the next turn).
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": "CANCELLED by user (Esc) — returned to chat input"})
            for c2 in ordered[idx + 1:]:
                messages.append({"role": "tool", "tool_call_id": c2["id"],
                                 "content": "SKIPPED (user pressed Esc at an earlier approval)"})
            break
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    if esc_action is not None:
        if not quiet:
            print(f"{C.WARNING}  ↳ escaped approval of {esc_action!r} — back to chat input{RESET}")
        messages.append({"role": "user", "content":
            "[User pressed Esc at a tool approval prompt and returned to chat to "
            "add more input. Acknowledge briefly and wait for their next message.]"})
        return TURN_ESC
    _flush_pending_background_notice(messages)
    return TURN_TOOL


def _handle_text_tool_calls(messages: list[Message], text: str, malformed_call_count: int = 0,
                            exclude: frozenset = frozenset()) -> Optional[str]:
    """Execute the text-protocol fallback tool-calling path for one streamed turn.

    Parses <tool_call>... blocks (or legacy [nx_tool_call]...) from the model's text, records the
    assistant turn, runs each parsed tool, and appends the observations as a
    user turn (the format the text-fallback protocol expects). If the user
    presses Esc at an approval, remaining calls are cancelled. Returns a
    TURN_* status for model_turn, or None if no text-protocol calls were found.

    If a tag was present but every call in it failed to parse (bad JSON, no
    "name"), that used to fall straight through to being treated as the final
    answer — the model's intent silently dropped. On the first failure, feed
    back a short correction (mirrors _assess_risk()'s retry loop in
    nx/_risk.py) and return TURN_MALFORMED_CALL so the loop retries once. If
    that retry also fails, discard the invalid output, leave a safe runtime
    note in history, and end the turn visibly instead of replaying the broken
    tag on every future turn.
    """
    calls, errors = parse_text_calls_with_errors(text)
    if exclude:
        calls = [(n, a) for n, a in calls if n not in exclude]
    quiet = _subagent_quiet()
    if not calls:
        if errors and malformed_call_count < 1:
            if not quiet:
                print(f"{C.WARNING}  ✂ MALFORMED TOOL CALL — {errors[0]}; "
                      f"nudging correction (attempt {malformed_call_count + 1}/1){RESET}")
            st._TALLY_RETRIES += 1
            _repair_log("malformed_text_call", errors[0])
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": (
                "[Runtime note: Your last tool call did not parse — "
                f"{errors[0]}. Re-emit it as a single well-formed "
                "<tool_call> block: the current dialect is XML-style, "
                "<tool_call><name>read_file</name><arguments><path>nx/_turn.py</path></arguments></tool_call> "
                "(one XML tag per argument; value is plain text or JSON). "
                "The legacy JSON-in-tag form still works: "
                "<tool_call>{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"TODO\", \"path\": \".\"}}</tool_call> "
                "or "
                "<tool_call>{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls -la\"}}</tool_call>. "
                "Emit only that one complete block with no surrounding prose.]"
            )})
            return TURN_MALFORMED_CALL
        if errors:
            if not quiet:
                print(f"{C.ERROR}  ✗ MALFORMED TOOL CALL — correction retry also failed; "
                      f"discarded invalid tool-call output and waiting for user input{RESET}")
                print(f"{DIM}    {errors[0]}{RESET}")
            _repair_log("repair_rejected", errors[0])
            # Do not retain `text`: it may be a truncated tool_call fragment
            # and would be replayed as assistant context on every later turn.
            messages.append({"role": "assistant", "content":
                "[Runtime: Discarded an invalid tool call after a failed correction retry. "
                "The requested tool action was not run.]"})
            return TURN_DONE
        return None
    messages.append({"role": "assistant", "content": text})
    # All-read-only multi-call turns and 2+ spawn_agent fan-outs dispatch
    # concurrently (same rationale as the native path); single calls and
    # other mixed turns stay sequential.
    if _should_parallelize(calls):
        parallel_results = _run_read_only_parallel(calls)
        obs = []
        for i, (n, _a) in enumerate(calls):
            res = parallel_results[i]
            # Same payload rule as the sequential branch below: a parts result
            # (read_image) on a text-protocol turn renders as a one-line hint,
            # never megabytes of base64 in a user message.
            if isinstance(res, list):
                from ._images import _parts_preview
                res = _parts_preview(res)
            obs.append(f"Observation ({n}): {res}")
        messages.append({"role": "user", "content": "\n".join(obs)})
        _flush_pending_background_notice(messages)
        return TURN_TOOL
    obs = []
    esc_action = None
    for n, a in calls:
        try:
            res = run_tool(n, a)
            # A gated read_image that slipped through on a text-protocol turn
            # (e.g. after /toolcalls native, or a native-confirmed model that
            # fell back mid-session) must not dump megabytes of base64 into a
            # user message — render it like the terminal preview does.
            if isinstance(res, list):
                from ._images import _parts_preview
                res = _parts_preview(res)
            obs.append(f"Observation ({n}): {res}")
        except _EscToChat as e:
            esc_action = e.args[0] if e.args else n
            obs.append(f"Observation ({n}): CANCELLED by user (Esc)")
            break
    if esc_action is not None:
        obs.append("[User pressed Esc at a tool approval prompt and returned to chat "
                   "to add more input. Acknowledge briefly and wait for their next message.]")
        if not quiet:
            print(f"{C.WARNING}  ↳ escaped approval of {esc_action!r} — back to chat input{RESET}")
        messages.append({"role": "user", "content": "\n".join(obs)})
        return TURN_ESC
    messages.append({"role": "user", "content": "\n".join(obs)})
    _flush_pending_background_notice(messages)
    return TURN_TOOL


def _strip_unterminated_tool_call(text: str) -> str:
    """Drop a trailing in-progress <tool_call>/[nx_tool_call] fragment.

    Reasoning captured mid-stream can end right in the middle of a tool-call
    block the model was drafting. Left in place, that fragment gets replayed
    as assistant context on every later turn, priming the model to keep
    emitting the same truncated JSON — the reasoning-stall analogue of the
    guard already applied to the text-fallback malformed-call path.
    """
    for open_tag, close_tag in (("<tool_call>", "</tool_call>"), ("[nx_tool_call]", "[/nx_tool_call]")):
        idx = text.rfind(open_tag)
        if idx != -1 and close_tag not in text[idx:]:
            text = text[:idx]
    return text


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
            if not quiet:
                print(f"{C.ERROR}  ✗ {collapse_type} after "
                      f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
            return TURN_DONE
        if not quiet:
            print(f"{C.WARNING}  ✂ {collapse_type.upper()} — discarding degenerated "
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
        elif salvaged:
            print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
                  f"showing captured reasoning as a best-effort answer{RESET}")
            _say_console.print(salvaged)
        else:
            print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
                  f"returning to chat input{RESET}")
        return TURN_DONE
    if cut_count >= retry_limit:
        if retry_limit == 0:
            # Failsafes deliberately disabled (e.g. benchmark mode where
            # we let the model think as long as it wants).  The model
            # produced only reasoning and no action — that's an honest
            # model-capability signal, not a harness rescue scenario.
            if not quiet:
                print(f"{DIM}  • reasoning-only turn ended naturally "
                      f"({loop_detail}); failsafes disabled, ending turn{RESET}")
        elif not quiet:
            print(f"{C.ERROR}  ✂ REASONING-ONLY RESCUE FAILED — gave up after "
                  f"{cut_count} stall "
                  f"attempt{'s' if cut_count != 1 else ''} "
                  f"× {loop_detail}; waiting for user input{RESET}")
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
            print(f"{DIM}  ↳ captured {len(reasoning_text)} chars of reasoning "
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
            print(f"{C.WARNING}  ✂ REASONING-ONLY STALL — {cut_msg}; forcing final "
                  f"answer (attempt {cut_count + 1}/{retry_limit}){RESET}")
        if reasoning_text.strip():
            messages.append({"role": "user", "content":
                "[Runtime note: Your reasoning has been preserved as your "
                "previous message. Do not continue private reasoning. Use "
                "the final_answer tool if available, or return a complete "
                "visible answer now based on what you reasoned. Keep it "
                "concise.]"})
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
        print(f"{C.WARNING}  ✂ REASONING-ONLY STALL — {cut_msg}; "
              f"{'nudging to wrap up' if wrapup_note else 'nudging to act'} "
              f"(attempt {cut_count + 1}/{retry_limit}){RESET}")
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
            "gets written. Emit the write_file or run_bash tool call now.]"})
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
        raise
    if stream is None:
        if not quiet:
            spinner.stop()
        if st._LAST_CONTEXT_OVERRUN:
            # Recoverable: _run_model_turn_loop() folds the older turns into a
            # summary and re-runs this turn. Not an error line — say what is
            # happening (once) so a long task doesn't just pause silently.
            if not quiet:
                print(f"{C.WARNING}  ✂ context window full — compressing and retrying"
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
    _ensure_interrupt_watcher()
    if not quiet:
        _USER_INTERRUPTED.clear()
        st._INTERRUPT_REDIRECT_TEXT = ""
        st._INTERRUPT_REDIRECT_SUBMITTED = False
    # Reset the per-turn bg-completion scan flag: the first tool call of this
    # new turn re-checks for finished background jobs (see run_tool()). A
    # quiet (async-agent) loop skips that scan entirely — run_tool() gates it
    # on _subagent_ctx(), because the notices belong to the PARENT's
    # transcript; the flag here just mirrors the reset for the parent turn.
    if not quiet:
        st._BG_SCAN_DONE = False
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
    if not quiet:
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
        # The watcher's inner select has a 0.1s timeout, so restore typically
        # completes in <0.1s. But if a bare Esc just fired, the watcher is
        # inside _drain_redirect_text, which has its own 2.0s deadline — wait
        # past that (not just 0.5s) so we never proceed while it still holds
        # raw termios mid-drain. NX_INTERRUPT_RESTORE_WAIT tunes the wait
        # (nx/_interrupt.py); keep the two call sites' effective values equal.
        if not quiet:
            _INTERRUPT_ARMED.clear()
            _INTERRUPT_RESTORED.wait(
                timeout=_env_float("NX_INTERRUPT_RESTORE_WAIT", 2.5))
            _USER_INTERRUPTED.clear()
    dropped_lines = getattr(stream, "dropped_lines", 0)
    if dropped_lines and not quiet:
        print(f"{C.WARNING}  ⚠ {dropped_lines} malformed SSE frame(s) dropped mid-stream — "
              f"any tool call in this turn may have truncated arguments{RESET}")
    # Account before every post-stream early return (stream error, user
    # interrupt, collapse/reasoning stall). The previous footer-owned hook
    # silently skipped all of those completed API calls.
    _record_usage(
        parser.usage, parser.timings,
        fallback_chars=parser.streamed_chars + sum(len(part) for part in parser.reasoning_parts),
        fallback_prompt_chars=sum(len(str(m.get("content") or "")) for m in messages),
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
                    if parser.mode in ("think", "say"):
                        print()
                    parser.close_think_block()
                    print(RESET)
                    delay = _step_backoff_delay(step_retries, None)
                    print(f"{C.WARNING}  ↻ STREAM TIMEOUT — server stopped delivering chunks; "
                          f"discarding partial output and retrying "
                          f"({step_retries + 1}/{_step_retry_limit()} in {delay:.1f}s){RESET}")
                return TURN_TIMED_OUT_RETRY
            if not quiet:
                if parser.mode in ("think", "say"):
                    print()
                parser.close_think_block()
                print(RESET)
                print(f"{C.ERROR}  ✗ STREAM TIMEOUT after {step_retries} "
                      f"retr{_plural_ies(step_retries)} — server stopped delivering chunks; "
                      f"the task did NOT complete{RESET}")
                print(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_TIMED_OUT
        retry_limit = max(0, st.MALFORMED_STREAM_RETRY_LIMIT)
        if malformed_stream_cut_count >= retry_limit:
            if not quiet:
                if parser.mode in ("think", "say"):
                    print()
                parser.close_think_block()
                print(RESET)
                print(f"{C.ERROR}  ✗ malformed stream from model/API after "
                      f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
                print(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_DONE
        if not quiet:
            print(f"{C.WARNING}  ✂ MALFORMED STREAM — discarded partial output/tool args; "
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
            print(f"{C.WARNING}  ↳ interrupted by user (Esc) after {elapsed:4.1f}s, "
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
    if not loop_cut and not text.strip() and not parser.tcs and parser.reasoning_only_chars > 0:
        loop_cut = True

    if loop_cut:
        return _handle_loop_cut(
            messages, forced_final, reasoning_loop_cut_count,
            parser.leaked_token_run, parser.repeat_chunk_run, parser.reasoning_only_start,
            parser.reasoning_only_chars, parser.reasoning_parts,
            malformed_stream_cut_count)

    # stats footer — prefer llama.cpp timings if present (server-measured tok/s);
    # otherwise divide the server's completion_tokens by the client-side
    # generation window (parser.gen_window(): first→last token span, not
    # wall-clock — the prefill head + render/teardown tail used to be included
    # and made short turns and just-read-a-file turns swing wildly). TTFT is
    # shown when available — for local it comes from llama.cpp timings, for
    # remote we measure it client-side.
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
            print(f"{C.WARNING}  ✂ FORCED FINAL ANSWER HIT TOKEN LIMIT — saved partial answer "
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

    if not text.strip():
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
                print(f"{C.WARNING}  ✂ EMPTY TURN — no content and no tool call"
                      f"{'; dangling tool result awaits a follow-up' if dangling else ''}"
                      f"; nudging the model to continue "
                      f"({empty_turn_count + 1}/{_subagent_scoped_retry(st.EMPTY_TURN_RETRY_LIMIT)}){RESET}")
            _nudge_current_user_turn(messages, EMPTY_TURN_NUDGE)
            return TURN_EMPTY
        if not forced_final and st.EMPTY_TURN_RETRY_LIMIT > 0:
            if not quiet:
                print(f"{C.WARNING}  ✂ EMPTY TURN — no content after {empty_turn_count} nudge"
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
            print(f"{C.WARNING}  ✂ TEXT-ONLY AFTER STALL NUDGE — model described "
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

# --- per-task wall-clock deadline (batch) -------------------------------------
# The batch harness arms this per worker thread just before its turn loop and
# clears it after. It must be a thread-local, not a module global -- with
# --parallel > 1 sibling workers would share and clobber each other's
# deadlines. The loop checks it at the on_step seam (every model-turn
# boundary) and aborts with a status-"error" summary rather than let one slow
# task hold a whole CI run open. A mid-tool-call cancel would be much messier
# and buys little: the deadline still fires at the next step boundary.
_TASK_DEADLINE_TLS = threading.local()


def _task_deadline_at(deadline):
    """Arm the calling worker's wall-clock deadline (None disarms it)."""
    _TASK_DEADLINE_TLS.deadline = deadline


def _task_deadline():
    """Return the armed deadline for this worker, or None if not armed."""
    return getattr(_TASK_DEADLINE_TLS, "deadline", None)


def _task_deadline_exceeded():
    d = getattr(_TASK_DEADLINE_TLS, "deadline", None)
    return d is not None and time.time() > d


def _task_timeout_summary(turns: int) -> dict:
    """The summary _run_model_turn_loop returns when a batch worker's
    wall-clock deadline lapses: status "error" so the task gets exit_code 4
    and the run stops scheduling (unless --continue-on-error), with
    "timed_out" so a harness can tell a timeout from a crash."""
    return {"status": "error",
            "error": (f"task timeout: exceeded the per-task wall-clock budget "
                      f"at turn {turns}"),
            "turns": turns,
            "timed_out": True}


def _report_length_cut(quiet: bool, used: int, limit: int) -> None:
    """Status line after appending a continuation resume note."""
    if not quiet:
        print(f"{C.WARNING}  ✂ RESPONSE CUT BY TOKEN LIMIT — continuing ({used}/{limit}){RESET}")


def _report_length_exhausted(quiet: bool, used: int) -> None:
    """Status line when the continuation budget ran out and the partial was closed out."""
    if not quiet:
        suffix = "s" if used != 1 else ""
        print(f"{C.WARNING}  ✂ RESPONSE STILL HIT TOKEN LIMIT after {used} continuation{suffix} "
              f"— saved partial answer with truncation marker{RESET}")


# --- P3-2 step-level transient retry (0.51.0) -------------------------------
# The HTTP layer (_openai_client._send) backs off 429/5xx/broken connections
# before the FIRST byte; a timeout after the stream has begun is structurally
# unreplayable there (a partially-consumed stream can't be re-sent), so the
# turn loop owns it: model_turn() returns TURN_TIMED_OUT_RETRY / TURN_TIMED_OUT
# and this loop applies the same ladder at the step boundary.

def _step_retry_limit() -> int:
    from ._openai_client import _HTTP_RETRIES_DEFAULT
    return max(0, _env_int("NX_TURN_RETRIES", _HTTP_RETRIES_DEFAULT))


def _step_backoff_delay(attempt: int, retry_after: float | None) -> float:
    """Same curve as the HTTP layer's pre-first-byte backoff (Retry-After wins,
    else base * 2**attempt, same cap) so a stream timeout and a connection
    retry behave identically."""
    from ._openai_client import _backoff_delay
    return _backoff_delay(attempt, retry_after)


def _plural_ies(n: int) -> str:
    return "y" if n == 1 else "ies"


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
    try:
        return _run_model_turn_loop_inner(messages, force_final, recovery_sampling,
                                          on_step, max_turns_override)
    finally:
        ctx = _recovery.current()
        if ctx is not None:
            _recovery.copy_back_and_reset(ctx)


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
        status = _m.model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                            empty_turn_count=empty_turn_cuts,
                            forced_final=force_final,
                            recovery_sampling=recovery_sampling,
                            malformed_call_count=malformed_call_cuts,
                            step_retries=timed_out_retries)
        if on_step is not None:
            on_step()
        if _task_deadline_exceeded():
            return _task_timeout_summary(steps + 1)
        was_forced = force_final  # TURN_TOOL after a forced-final turn is special-cased below
        force_final = False
        recovery_sampling = False
        if status == TURN_DONE:
            if not _subagent_active() and not _batch_active():
                # Only the parent loop fires NX_VERIFY_CMD — concurrent
                # sub-agents or batch-task threads hitting this line together
                # would race the settle log and double-launch the verify
                # command.
                _launch_verify_if_settled()  # model went quiet — fire NX_VERIFY_CMD if its settle window passed
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
                print(f"{DIM}  ↻ step retry {timed_out_retries}/"
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
            print(f"{C.WARNING}  ⚂ MODEL TURN LIMIT ({max_turns}) — "
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

