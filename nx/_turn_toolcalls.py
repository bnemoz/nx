"""Tool-call handlers for the turn loop.

Split out of `_turn.py` (0.90.0 decomposition): the four `_handle_*` paths
(forced-final/native/text tool calls) plus the partial-answer close-out and
the unterminated-tool-call stripper. These consume a streamed turn's tool
calls and return a TURN_* status; `model_turn` in `_turn.py` is the caller.
`_turn.py` re-exports everything here, so the import surface (including test
attribute access like `nx._turn._handle_native_tool_calls`) is unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ._ansi import DIM, RESET, C, strip_ansi_for_display
from ._approval import _EscToChat
from ._interrupt import cancel_requested, _HARD_CANCEL
from . import _state as st
from . import run_tool, _tool_args_parse_error
from ._compress import _nudge_current_user_turn, _flush_pending_background_notice
from ._text_fallback import parse_text_calls_with_errors, repair_json_candidates, normalize_tool_args
from ._subagent import _subagent_quiet, _subagent_stream_cut_limit
from ._argsalvage import _malformed_args_nudge, _repair_log, _salvage_truncated_args
from ._tool_prefetch import _run_read_only_parallel, _should_parallelize, _prefetch_cacheable_tools
from ._turn_status import (
    TURN_DONE, TURN_TOOL, TURN_STREAM_CUT, TURN_ESC, TURN_MALFORMED_CALL,
    TRUNCATION_MARKER,
)

Message = dict[str, Any]  # one entry in the OpenAI-style `messages` array


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
                            f"({len(raw)} -> {len(json.dumps(salvaged))} chars)",
                            raw=raw)
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
        _repair_log("malformed_args", f"{tool_name}: {detail}",
                    raw=(c.get("arguments") if isinstance(c.get("arguments"), str)
                         else c.get("args") or ""))
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
        # C1 boundary check: once the hard-cancel ladder has fired, the whole
        # remaining group is cancelled — the model must see a cancelled
        # result for EVERY call (the transcript stays honest; a tool result
        # must never silently vanish).
        if cancel_requested():
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": "CANCELLED by user (hard cancel) — not started"})
            for c2 in ordered[idx + 1:]:
                messages.append({"role": "tool", "tool_call_id": c2["id"],
                                 "content": "SKIPPED (user hard-cancelled the turn)"})
            break
        try:
            result = run_tool(c["name"], args)
        except KeyboardInterrupt:
            # C1: Ctrl+C landing inside a tool body = cooperative hard cancel
            # (kill the live hook, skip the rest of the group) instead of a
            # raw turn unwind that would leave tool calls without results.
            from ._interrupt import request_hard_cancel
            request_hard_cancel("Ctrl+C in tool")
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": "CANCELLED by user (Ctrl+C hard cancel)"})
            for c2 in ordered[idx + 1:]:
                messages.append({"role": "tool", "tool_call_id": c2["id"],
                                 "content": "SKIPPED (user hard-cancelled the turn)"})
            break
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
    if cancel_requested() and esc_action is None:
        # The group consumed the cancel (this call + the skip-fill above):
        # clear so a later async wake/sub-agent turn doesn't inherit it. The
        # next interactive model_turn also clears it defensively.
        _HARD_CANCEL.clear()
        if not quiet:
            print(f"{C.WARNING}  ↳ turn hard-cancelled — remaining tools skipped{RESET}")
        messages.append({"role": "user", "content":
            "[User hard-cancelled this turn (double Esc / Ctrl+C). Any running "
            "tool was killed; the cancelled tool results above are final. "
            "Acknowledge briefly and wait for their next message.]"})
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
    for ci, (n, a) in enumerate(calls):
        if cancel_requested():  # C1 boundary check (see native path above)
            obs.append(f"Observation ({n}): CANCELLED by user (hard cancel) — not started")
            for n2, _a2 in calls[ci + 1:]:
                obs.append(f"Observation ({n2}): SKIPPED (user hard-cancelled the turn)")
            break
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
        except KeyboardInterrupt:
            from ._interrupt import request_hard_cancel
            request_hard_cancel("Ctrl+C in tool")
            obs.append(f"Observation ({n}): CANCELLED by user (Ctrl+C hard cancel)")
            break
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
    if cancel_requested():
        obs.append("[User hard-cancelled this turn (double Esc / Ctrl+C). Any running "
                   "tool was killed; the cancelled tool results above are final. "
                   "Acknowledge briefly and wait for their next message.]")
        _HARD_CANCEL.clear()  # consumed (see native-path note)
        if not quiet:
            print(f"{C.WARNING}  ↳ turn hard-cancelled — remaining tools skipped{RESET}")
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
