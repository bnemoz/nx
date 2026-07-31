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
    parse_text_calls_with_errors, _sanitize_tool_result, repair_json_candidates,
)
from ._compress import (
    _nudge_current_user_turn,
    _flush_pending_background_notice, _last_is_dangling_tool,
)
from ._env import _env_int
from . import _state as st
from . import (
    FINAL_ANSWER_TOOL, FINAL_ANSWER_TOOL_CHOICE, run_tool, _active_tools,
    REASONING_STALL_NUDGE, FORCED_FINAL_NUDGE, EMPTY_TURN_NUDGE,
    REASONING_ONLY_TIME_LIMIT, REASONING_CAPTURE_CHARS, _tool_args_parse_error,
    _CACHEABLE_READ_TOOLS, _tool_cache_key, DISPATCH,
)
from concurrent.futures import ThreadPoolExecutor
import os

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
    """Cache the just-seen prompt-token count in st._LAST_PROMPT_TOKENS.

    Split out of _print_stats_footer so it can run on EVERY settled stream —
    including the loop_cut give-up paths (collapse / exhausted reasoning-stall
    retries) that return before the footer is ever printed. Those turns still
    grow `messages` (captured reasoning, runtime-note nudges), so skipping this
    left _maybe_autocompress() checking a stale, smaller token count right when
    autocompress was most needed.
    """
    if timings and timings.get("predicted_n"):
        prompt_n = timings.get("prompt_n", 0)
        if prompt_n:
            st._LAST_PROMPT_TOKENS = prompt_n
    elif usage and (usage.prompt_tokens or 0):
        st._LAST_PROMPT_TOKENS = usage.prompt_tokens


def _print_stats_footer(usage: Any, timings: Optional[dict], streamed_chars: int, t_first: float | None,
                         elapsed: float, text: str, tcs: dict, messages: Optional[list[Message]] = None) -> None:
    """Render the per-turn stats footer.

    Prefers llama.cpp `timings` (real tok/s) when present; falls back to the
    standard streaming `usage` object (OpenAI/Z.ai); then to a char-based
    estimate; then to wall-clock only.
    Prompt-token caching itself lives in _capture_last_prompt_tokens(), called
    separately so it also runs on turns that never reach this footer.
    """
    if timings and timings.get("predicted_n"):
        prompt_n = timings.get("prompt_n", 0)
        gen_n = timings["predicted_n"]
        tps = timings.get("predicted_per_second", 0)
        ctx = _ctx_field(prompt_n)
        prompt_ms = timings.get("prompt_ms", 0)
        ttft = (prompt_ms / 1000.0) if prompt_ms else t_first
        parts = [f"{C.STATUS_BAR}{gen_n}{RESET}{DIM} tok", f"{C.STATUS_BAR}{tps:5.1f}{RESET}{DIM} tok/s", ctx]
        if st.MODEL:
            parts.append(f"{C.STATUS_BAR}{st.MODEL}{RESET}")
        if ttft:
            parts.append(f"{C.STATUS_BAR}{ttft*1000:4.0f}{RESET}{DIM}ms ttft")
        parts.append(f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall")
        print(f"{DIM}  └ {(f'{RESET}{DIM} · ').join(parts)}{RESET}")
    elif usage and (usage.completion_tokens or 0):
        gen_n = usage.completion_tokens or 0
        prompt_n = usage.prompt_tokens or 0
        gen_elapsed = elapsed - t_first if t_first else elapsed
        tps = gen_n / gen_elapsed if gen_elapsed > 0 else 0
        ctx = _ctx_field(prompt_n)
        parts = [f"{C.STATUS_BAR}{gen_n}{RESET}{DIM} tok", f"{C.STATUS_BAR}{tps:5.1f}{RESET}{DIM} tok/s", ctx]
        if st.MODEL:
            parts.append(f"{C.STATUS_BAR}{st.MODEL}{RESET}")
        if t_first:
            parts.append(f"{C.STATUS_BAR}{t_first*1000:4.0f}{RESET}{DIM}ms ttft")
        parts.append(f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall")
        print(f"{DIM}  └ {(f'{RESET}{DIM} · ').join(parts)}{RESET}")
    elif streamed_chars > 0:
        gen_n = max(1, streamed_chars // 4)
        gen_elapsed = elapsed - t_first if t_first else elapsed
        tps = gen_n / gen_elapsed if gen_elapsed > 0 else 0
        print(f"{DIM}  └ ≈{C.STATUS_BAR}{_abbr(gen_n)}{RESET}{DIM} tok · "
              f"{C.STATUS_BAR}{tps:5.1f}{RESET}{DIM} tok/s · "
              f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall{RESET}")
    elif text or tcs:
        print(f"{DIM}  └ {elapsed:4.1f}s wall{RESET}")


def _handle_forced_final_tool_call(messages: list[Message], tcs: dict) -> str:
    """Under forced-final mode, extract the `final_answer` tool call (if any) the
    model emitted and render its answer as visible assistant text. Returns a
    TURN_* status to bubble up from model_turn."""
    ordered = [tcs[i] for i in sorted(tcs)]
    for c in ordered:
        if c["name"] != "final_answer":
            continue
        try:
            args = json.loads(c["args"] or "{}")
        except json.JSONDecodeError:
            args = {}
        answer = str(args.get("answer") or "").strip()
        status = str(args.get("status") or "answered").strip()
        if answer:
            print(f"{C.TEXT}{strip_ansi_for_display(answer)}{RESET}")
            messages.append({"role": "assistant", "content": answer})
            return TURN_DONE
        print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER EMPTY — status={status or 'unknown'}; "
              f"returning to chat input{RESET}")
        return TURN_DONE
    names = ", ".join(c["name"] or "tool" for c in ordered)
    print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — model emitted {names}; "
          f"returning to chat input{RESET}")
    return TURN_DONE


def _handle_forced_final_text_calls(messages: list[Message], calls: list[tuple[str, dict]]) -> str:
    """Text-fallback counterpart to _handle_forced_final_tool_call(): under forced-final
    mode a text-protocol model can still emit <tool_call> blocks, so corral it into
    final_answer-only here instead of letting _handle_text_tool_calls execute arbitrary
    tools (which would defeat the stall-recovery safety valve for non-native backends)."""
    for n, a in calls:
        if n != "final_answer":
            continue
        answer = str(a.get("answer") or "").strip()
        status = str(a.get("status") or "answered").strip()
        if answer:
            print(f"{C.TEXT}{strip_ansi_for_display(answer)}{RESET}")
            messages.append({"role": "assistant", "content": answer})
            return TURN_DONE
        print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER EMPTY — status={status or 'unknown'}; "
              f"returning to chat input{RESET}")
        return TURN_DONE
    names = ", ".join(n or "tool" for n, _ in calls)
    print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — model emitted {names}; "
          f"returning to chat input{RESET}")
    return TURN_DONE


def _prefetch_cacheable_tools(ordered: list[dict], parsed_args: list[dict]) -> None:
    """Warm the read_file/list_dir cache concurrently for this turn's calls.

    run_tool() below still executes every call sequentially, in order, for
    printing/spinner/approval reasons — but if two-plus of them are cache-
    eligible reads, doing the actual blocking I/O in parallel here means
    run_tool()'s later cache hits are near-instant, cutting wall-clock without
    touching the tool schema or the sequential dispatch contract.
    """
    jobs = []
    seen_keys = set()
    for c, args in zip(ordered, parsed_args):
        if c["name"] not in _CACHEABLE_READ_TOOLS:
            continue
        cache_key = _tool_cache_key(c["name"], args)
        if cache_key in seen_keys or cache_key in st._TOOL_RESULT_CACHE:
            continue
        seen_keys.add(cache_key)
        jobs.append((c["name"], args, cache_key))
    if len(jobs) < 2:
        return

    def _run(job: tuple[str, dict, tuple[str, str]]) -> None:
        name, args, cache_key = job
        try:
            result = DISPATCH[name](**args)
        except Exception:
            return  # let the normal run_tool() call surface/report the error
        result = _sanitize_tool_result(result)
        cache_path = args.get("path", ".")
        if isinstance(cache_path, list):
            cache_path = cache_path[0] if cache_path else "."
        st._TOOL_RESULT_CACHE[cache_key] = (result, os.path.abspath(cache_path))

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        list(ex.map(_run, jobs))


def _handle_native_tool_calls(messages: list[Message], tcs: dict, text: str, finish_reasons: Any,
                              malformed_stream_cut_count: int) -> str:
    """Execute the native tool-calling path for one streamed turn.

    Parses each call's JSON args (retrying cleanly on malformed args rather
    than executing `{}`), records the assistant(tool_calls) turn, runs each
    tool, and appends matching tool results. If the user presses Esc at an
    approval, the remaining calls are marked CANCELLED/SKIPPED so the message
    history stays valid. Returns a TURN_* status for model_turn.
    """
    st._NATIVE_TOOL_CALLS_CONFIRMED = True  # server is actually using native tool_calls this session
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
        if obj is None:
            parse_error = (c, err)
            break
        parsed_args.append(obj)
    if parse_error is not None:
        c, err = parse_error
        if err is None:
            err = json.JSONDecodeError("no candidates", "", 0)
        detail = _tool_args_parse_error(c, err, finish_reasons)
        retry_limit = max(0, st.MALFORMED_STREAM_RETRY_LIMIT)
        if malformed_stream_cut_count >= retry_limit:
            print(f"{C.ERROR}  ✗ malformed tool call after "
                  f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
            print(f"{DIM}    {detail}{RESET}")
            return TURN_DONE
        print(f"{C.WARNING}  ✂ MALFORMED TOOL CALL — {detail}; "
              f"discarded partial tool args and retrying cleanly "
              f"({malformed_stream_cut_count + 1}/{retry_limit}){RESET}")
        _nudge_current_user_turn(
            messages,
            "Your previous native tool call had malformed or truncated JSON "
            "arguments, so it was discarded before execution. Retry the same "
            "task from the current conversation state, but emit a complete, "
            "valid tool call with all required arguments. For large file "
            "writes, emit only the tool call and skip any explanatory prose.")
        return TURN_STREAM_CUT
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
        print(f"{C.WARNING}  ↳ escaped approval of {esc_action!r} — back to chat input{RESET}")
        messages.append({"role": "user", "content":
            "[User pressed Esc at a tool approval prompt and returned to chat to "
            "add more input. Acknowledge briefly and wait for their next message.]"})
        return TURN_ESC
    _flush_pending_background_notice(messages)
    return TURN_TOOL


def _handle_text_tool_calls(messages: list[Message], text: str, malformed_call_count: int = 0) -> Optional[str]:
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
    if not calls:
        if errors and malformed_call_count < 1:
            print(f"{C.WARNING}  ✂ MALFORMED TOOL CALL — {errors[0]}; "
                  f"nudging correction (attempt {malformed_call_count + 1}/1){RESET}")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": (
                "[Runtime note: Your last tool call did not parse — "
                f"{errors[0]}. Re-emit it as a single well-formed "
                "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> "
                "block with valid JSON. For example:\n"
                "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"nx/_turn.py\"}}</tool_call>\n"
                "<tool_call>{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"TODO\", \"path\": \".\"}}</tool_call>\n"
                "<tool_call>{\"name\": \"run_bash\", \"arguments\": {\"cmd\": \"ls -la\"}}</tool_call>]"
            )})
            return TURN_MALFORMED_CALL
        if errors:
            print(f"{C.ERROR}  ✗ MALFORMED TOOL CALL — correction retry also failed; "
                  f"discarded invalid tool-call output and waiting for user input{RESET}")
            print(f"{DIM}    {errors[0]}{RESET}")
            # Do not retain `text`: it may be a truncated <tool_call> fragment
            # and would be replayed as assistant context on every later turn.
            messages.append({"role": "assistant", "content":
                "[Runtime: Discarded an invalid tool call after a failed correction retry. "
                "The requested tool action was not run.]"})
            return TURN_DONE
        return None
    messages.append({"role": "assistant", "content": text})
    obs = []
    esc_action = None
    for n, a in calls:
        try:
            obs.append(f"Observation ({n}): {run_tool(n, a)}")
        except _EscToChat as e:
            esc_action = e.args[0] if e.args else n
            obs.append(f"Observation ({n}): CANCELLED by user (Esc)")
            break
    if esc_action is not None:
        obs.append("[User pressed Esc at a tool approval prompt and returned to chat "
                   "to add more input. Acknowledge briefly and wait for their next message.]")
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
                     reasoning_only_chars: int, reasoning_parts: list[str]) -> str:
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
    if is_leak_collapse and not forced_final:
        collapse_type = ("special-token collapse" if leaked_token_run >= LEAKED_TOKEN_RUN_LIMIT
                         else "repetition collapse")
        print(f"{C.WARNING}  ✂ {collapse_type.upper()} — discarding degenerated "
              f"output and nudging clean recovery{RESET}")
        # Do NOT feed the degenerated reasoning/content back — it's spam
        # that would reinforce the collapse on the next turn.  Nudge fresh.
        _nudge_current_user_turn(messages,
            "Your previous output degenerated into a repetition collapse "
            "(repeated special tokens or identical chunks). This is a known "
            "failure mode — discard that output entirely. Re-orient: state "
            "what you were trying to do in one sentence, then emit the next "
            "concrete tool call (write_file, run_bash, or edit_file). Do not "
            "continue the previous stream. Start fresh.")
        return TURN_STREAM_CUT
    retry_limit = max(0, st.REASONING_ONLY_RETRY_LIMIT)
    cut_count = reasoning_loop_cut_count
    reasoning_secs = (time.time() - reasoning_only_start) if reasoning_only_start else 0
    if reasoning_only_start and REASONING_ONLY_TIME_LIMIT > 0 and \
            reasoning_secs >= REASONING_ONLY_TIME_LIMIT:
        loop_detail = (f"{reasoning_secs:.0f}s of reasoning "
                       f"({_abbr(reasoning_only_chars)} chars) without content/tool calls")
    else:
        loop_detail = (
            f"{_abbr(reasoning_only_chars)} reasoning chars without content/tool calls")
    if forced_final:
        print(f"{C.ERROR}  ✂ FORCED FINAL ANSWER FAILED — got {loop_detail}; "
              f"returning to chat input{RESET}")
        return TURN_DONE
    if cut_count >= retry_limit:
        if retry_limit == 0:
            # Failsafes deliberately disabled (e.g. benchmark mode where
            # we let the model think as long as it wants).  The model
            # produced only reasoning and no action — that's an honest
            # model-capability signal, not a harness rescue scenario.
            print(f"{DIM}  • reasoning-only turn ended naturally "
                  f"({loop_detail}); failsafes disabled, ending turn{RESET}")
        else:
            print(f"{C.ERROR}  ✂ REASONING-ONLY RESCUE FAILED — gave up after "
                  f"{cut_count} stall "
                  f"attempt{'s' if cut_count != 1 else ''} "
                  f"× {loop_detail}; waiting for user input{RESET}")
        return TURN_DONE
    if reasoning_only_start and REASONING_ONLY_TIME_LIMIT > 0 and \
            reasoning_secs >= REASONING_ONLY_TIME_LIMIT:
        cut_msg = (f"{reasoning_secs:.0f}s of reasoning with no "
                   f"answer/tool call (limit {REASONING_ONLY_TIME_LIMIT}s)")
    else:
        cut_msg = (f"{_abbr(reasoning_only_chars)} reasoning chars with no "
                   f"answer/tool call (limit {_abbr(st.REASONING_ONLY_CHAR_LIMIT)})")

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
    print(f"{C.WARNING}  ✂ REASONING-ONLY STALL — {cut_msg}; nudging to act "
          f"(attempt {cut_count + 1}/{retry_limit}){RESET}")
    if reasoning_text.strip():
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
        _nudge_current_user_turn(messages, REASONING_STALL_NUDGE)
    return TURN_STALL


# --- one model turn (streamed), returns TURN_* status -----------------------
def model_turn(messages: list[Message], reasoning_loop_cut_count: int = 0, malformed_stream_cut_count: int = 0,
               empty_turn_count: int = 0, forced_final: bool = False, recovery_sampling: bool = False,
               malformed_call_count: int = 0) -> str:
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
    spinner_label = (
        "forcing final answer · esc to interrupt"
        if forced_final else
        "recovering · esc to interrupt" if recovery_sampling else
        "thinking · esc to interrupt"
    )
    spinner = LifeSpinner(label=spinner_label)
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
        spinner.stop()
        raise
    if stream is None:
        spinner.stop()
        return TURN_DONE  # error already reported; REPL continues

    # Interrupt watcher: the singleton daemon thread watches stdin for bare Esc
    # while ARMED. On hit it sets _USER_INTERRUPTED; the main loop breaks out of
    # the chunk loop on its next iteration. We arm it BEFORE the spinner so the
    # user always has a moment to interrupt even if the first token takes a
    # while to arrive.
    _ensure_interrupt_watcher()
    _USER_INTERRUPTED.clear()
    st._INTERRUPT_REDIRECT_TEXT = ""
    st._INTERRUPT_REDIRECT_SUBMITTED = False
    # Clear the "termios restored" flag BEFORE arming so we can wait on it
    # after disarm. The watcher sets it in its inner finally once it has
    # restored cooked mode. Without this handshake the watcher's termios
    # restore is asynchronous (it may still be in select(0.1) when we
    # return), and the chatbox can capture the watcher's raw-mode termios
    # as its "old" state — leaving the terminal stuck in VMIN=0
    # with broken Enter and garbled typing.
    _INTERRUPT_RESTORED.clear()
    _INTERRUPT_ARMED.set()
    parser = StreamParser(_say_console, t0)
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
        # raw termios mid-drain.
        _INTERRUPT_ARMED.clear()
        _INTERRUPT_RESTORED.wait(timeout=2.5)
        _USER_INTERRUPTED.clear()
    dropped_lines = getattr(stream, "dropped_lines", 0)
    if dropped_lines:
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
        if parser.mode in ("think", "say"):
            print()
        parser.close_think_block()
        print(RESET)
        if isinstance(stream_error, httpx.TimeoutException):
            print(f"{C.WARNING}  ✂ STREAM TIMEOUT — server stopped delivering chunks; "
                  f"discarded partial output and returned to chat input{RESET}")
            print(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_DONE
        retry_limit = max(0, st.MALFORMED_STREAM_RETRY_LIMIT)
        if malformed_stream_cut_count >= retry_limit:
            print(f"{C.ERROR}  ✗ malformed stream from model/API after "
                  f"{malformed_stream_cut_count} recoveries — waiting for user input{RESET}")
            print(f"{DIM}    {type(stream_error).__name__}: {stream_error}{RESET}")
            return TURN_DONE
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
            parser.reasoning_only_chars, parser.reasoning_parts)

    # stats footer — prefer llama.cpp timings if present; otherwise fall back
    # to the standard streaming `usage` object (OpenAI, Z.ai, etc.); otherwise
    # fall back to wall-clock only. TTFT (time to first token) is shown when
    # available — for local it comes from llama.cpp timings, for remote we
    # measure it client-side.
    _print_stats_footer(parser.usage, parser.timings, parser.streamed_chars, parser.t_first,
                         elapsed, text, parser.tcs, messages=messages)

    if forced_final and parser.tcs:
        return _handle_forced_final_tool_call(messages, parser.tcs)

    if parser.tcs:  # native tool-calling path
        return _handle_native_tool_calls(messages, parser.tcs, text, parser.finish_reasons,
                                         malformed_stream_cut_count)

    if forced_final:
        calls, _errors = parse_text_calls_with_errors(text)
        if calls:
            return _handle_forced_final_text_calls(messages, calls)
    else:
        status = _handle_text_tool_calls(messages, text, malformed_call_count)  # text-fallback path
        if status is not None:
            return status

    if forced_final and text.strip() and "length" in parser.finish_reasons:
        print(f"{C.WARNING}  ✂ FORCED FINAL ANSWER HIT TOKEN LIMIT — saved partial answer "
              f"with truncation marker{RESET}")
        messages.append({
            "role": "assistant",
            "content": text.rstrip() + "\n\n[Truncated by token limit before completion.]",
        })
        return TURN_DONE

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
                and empty_turn_count < st.EMPTY_TURN_RETRY_LIMIT):
            dangling = _last_is_dangling_tool(messages)
            print(f"{C.WARNING}  ✂ EMPTY TURN — no content and no tool call"
                  f"{'; dangling tool result awaits a follow-up' if dangling else ''}"
                  f"; nudging the model to continue "
                  f"({empty_turn_count + 1}/{st.EMPTY_TURN_RETRY_LIMIT}){RESET}")
            _nudge_current_user_turn(messages, EMPTY_TURN_NUDGE)
            return TURN_EMPTY
        if not forced_final and st.EMPTY_TURN_RETRY_LIMIT > 0:
            print(f"{C.WARNING}  ✂ EMPTY TURN — no content after {empty_turn_count} nudge"
                  f"{'s' if empty_turn_count != 1 else ''}; forcing a final answer{RESET}")
            _nudge_current_user_turn(messages, FORCED_FINAL_NUDGE)
            return TURN_FORCE_FINAL
        return TURN_DONE
    messages.append({"role": "assistant", "content": text})
    return TURN_DONE


# Maximum number of model turns (tool calls + retries) per session before the
# loop breaks.  The real bound on work is the per-task timeout (900-3600s);
# each turn costs at minimum a few seconds (API round-trip + generation + tool
# exec), so even 200 turns in a 900s budget means turns averaging <4.5s — that's
# fast spinning, not productive work.  Individual pathologies (reasoning stalls,
# empty turns, malformed streams) already have their own retry limits.  This cap
# is just a total-work backstop so the loop can't run away *inside* the timeout.
# Overridable via env var.  Set very high (e.g. 99999) to effectively disable.
st.MAX_MODEL_TURNS = _env_int("NX_MAX_MODEL_TURNS", 200)

def _run_model_turn_loop(messages: list[Message], force_final: bool = False, recovery_sampling: bool = False,
                         on_step: Optional[Callable[[], None]] = None) -> dict:
    """Runs model_turn() to completion (or MAX_MODEL_TURNS). Returns a summary
    dict {"status": "done"|"esc"|"cap", "turns": int} — interactive callers
    ignore it, but headless one-shot mode (nx/_cli.py) uses it to report
    whether the model actually finished vs. hit the turn cap."""
    steps = 0
    reasoning_loop_cuts = 0
    malformed_stream_cuts = 0
    empty_turn_cuts = 0
    malformed_call_cuts = 0
    # Go through the top-level package (not a bare same-module reference) so
    # a test that monkeypatches `nx.model_turn` (to avoid a real model
    # round-trip) takes effect on every call this loop makes.
    import nx as _m
    while steps < st.MAX_MODEL_TURNS:  # cap runaway tool/retry loops
        status = _m.model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                            empty_turn_count=empty_turn_cuts,
                            forced_final=force_final,
                            recovery_sampling=recovery_sampling,
                            malformed_call_count=malformed_call_cuts)
        if on_step is not None:
            on_step()
        force_final = False
        recovery_sampling = False
        if status == TURN_DONE:
            return {"status": "done", "turns": steps + 1}
        if status == TURN_ESC:
            return {"status": "esc", "turns": steps + 1}  # user pressed Esc at an approval → drop to chat input
        if status == TURN_REDIRECT:
            # Esc-then-type: the user already supplied their next instruction
            # (appended to `messages` by model_turn()) — loop straight into
            # another turn instead of dropping to the REPL prompt.
            steps += 1
            continue
        steps += 1
        if status == TURN_STREAM_CUT:
            malformed_stream_cuts += 1
            recovery_sampling = True
            continue
        if status == TURN_EMPTY:
            empty_turn_cuts += 1
            recovery_sampling = True  # more entropy to escape the empty-output attractor
            continue
        if status == TURN_MALFORMED_CALL:
            malformed_call_cuts += 1
            continue
        if status == TURN_STALL:
            reasoning_loop_cuts += 1
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
            recovery_sampling = True
            continue
        if status == TURN_TOOL:
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
            malformed_stream_cuts = 0
            empty_turn_cuts = 0
            continue
    # Loop exited because steps >= st.MAX_MODEL_TURNS.  In interactive mode the
    # user can simply type another instruction; in headless mode (benchmarks)
    # the session would end with whatever partial work exists.  Give the model
    # one last forced-final turn so it can produce a visible answer from the
    # state so far — but only if we haven't already tried (force_final was
    # already consumed by the loop above).
    if steps >= st.MAX_MODEL_TURNS and not force_final:
        print(f"{C.WARNING}  ⚂ MODEL TURN LIMIT ({st.MAX_MODEL_TURNS}) — "
              f"forcing a final answer from current state{RESET}")
        final_status = _m.model_turn(messages, reasoning_loop_cuts, malformed_stream_cuts,
                   empty_turn_count=empty_turn_cuts,
                   forced_final=True,
                   recovery_sampling=recovery_sampling)
        if on_step is not None:
            on_step()
        if final_status == TURN_DONE:
            return {"status": "cap", "turns": steps + 1}
    return {"status": "cap", "turns": steps}

