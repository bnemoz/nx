"""Malformed-args salvage: the fallback map, repair logging, the breaker nudge,
and the bracket-closure salvage for read tools.

Split out of `_turn.py` (which keeps the tool-call handlers that use it).
Consumed lazily by `_dispatch`/`_tool_gates` (`_malformed_args_nudge`) — the
call-time import is deliberate: `_turn` imports `_dispatch` at module level,
so those modules must not import `_turn` at theirs."""
from __future__ import annotations

import json

# Where the model should go when THIS tool's args keep getting truncated and
# the breaker drops it: the nearest sibling that can do the same job. A tool
# not listed here degrades the nudge to "use one of: <remaining tools>"
# (computed live, so it never names a tool the breaker itself dropped).
FALLBACK_FOR = {
    "multi_edit": "edit_file", "sed_edit": "edit_file",
    # apply_patch was retired into edit_file diff mode; a stale apply_patch
    # name still sitting in the breaker gets pointed at that mode explicitly
    # (plain "use edit_file" alone would invite a literal-mode call).
    "apply_patch": "edit_file diff mode (pass the unified diff as `diff`)",
    "read_image": "read_file", "delete_file": "edit_file",
    "plan_edit": "edit_file", "wait_background": "read_file",
    "memory_write": "memory_search", "memory_delete": "memory_search",
    "search_files": "list_dir", "list_dir": "search_files",
    "todo_write": "write_file",
}


def _repair_log(kind: str, what: str, raw: str = "") -> None:
    """0.30.0 observability: every applied repair and every repair-rejection
    gets one JSONL record in the session log (grep-able when tuning
    NX_ARG_REPAIR / NX_MALFORMED_TOOL_BREAKER). Best-effort — never raises.

    v1 usage metrics: the record now also carries the endpoint context
    (model/agent), the malformed `raw` args string (truncated to 8192) and
    the CTX_TAIL (the model's last words before the broken call), so a
    repair record alone answers "which model said what, and what did it
    emit". `raw` is optional: call sites that have no raw string (text-call
    rejections) just omit it. Still rides the EXISTING repair channel on
    llamacpp.log, not the usage stream.

    self-retro (nx #1): every repair ALSO mirrors onto the usage stream via
    `usage_repair` — the two channels stay separate (full detail here,
    compact counters there) but no repair is invisible to the friction
    counters anymore. The tool is recovered from the `<tool>: ...` `what`
    convention the call sites use, when present."""
    try:
        from ._logging import _log_event, _agent_tag, usage_repair
        from . import _state as _st
        from . import _usage_ctx
        _log_event("repair", {"kind": kind, "what": what,
                              "model": _st.MODEL,
                              "agent": _agent_tag(),
                              "raw": (raw or "")[:8192],
                              "ctx_tail": _usage_ctx.tail()})
        head = what.split(":", 1)[0].strip() if what else ""
        tool = head if head and head.replace("_", "").isalnum() else ""
        usage_repair(kind, what, tool)
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
    repair INVENTS content. On write_file/edit_file/multi_edit (diff mode
    included) the
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
