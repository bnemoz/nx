"""The tiered tool registry (0.62.0, PERF §6 items 1-2): gates, the active
schema list, and the tier-2 search_tools meta-tool.

A declarative (name -> gate) table replacing five sequential O(N) filter
passes over the schema list. `_active_tools()` is memoized on the gate-vector
(a tuple of booleans rebuilt per turn is one dict lookup + one list comp).

Tier 2 (meta-tool) is not yet occupied by any schema: `search_tools` lifts a
tier-2 tool's full schema into context on demand, so a future always-on tool
can ship at ~0 token rent instead of ~150 (the 0.62 feature-cost rule).

Split out of `_dispatch.py` (which keeps the DISPATCH table and run_tool)."""
from __future__ import annotations

import json
import re
from typing import Any

from . import _state as st
from ._plan import plan_active
from ._delegation_note import _delegation_enabled

_TOOL_GATES: dict[str, Any] = {
    # name -> callable returning True when the schema should be OFFERED this
    # turn. Unlisted names are always-on. The malformed-args breaker is a veto
    # applied after the gates (a broken tool is dropped, not gated in).
    "wait_background": lambda: st._SESSION_HAS_BACKGROUNDED,
    "spawn_agent": _delegation_enabled,
    "read_image": lambda: bool(_m_safe().NX_VISION) and st._NATIVE_TOOL_CALLS_CONFIRMED,
    "plan_edit": plan_active,
    # SoL-Pi ports (0.67.0): obs_recall only once a pack handle exists
    # (packing never happened otherwise — nothing to recall).
    "obs_recall": lambda: bool(_m_safe().NX_OBS),
    # web_fetch (0.70.0): keyless DuckDuckGo-lite search + page fetch. Tier-2
    # (0 schema tokens until a session needs the outside world) — surfaced by
    # search_tools; NX_WEBFETCH=0 disables the tool entirely (offline hosts).
    "web_fetch": lambda: bool(getattr(_m_safe(), "NX_WEBFETCH", 0)),
}

_TOOL_LIST_CACHE: dict[tuple, list] = {}


def _m_safe():
    import nx as _m
    return _m


def _gate_vector() -> tuple:
    import nx as _m
    from ._breaker import _breaker_views
    broken = _breaker_views()[1]
    return (
        bool(st._SESSION_HAS_BACKGROUNDED),
        bool(_m.NX_SUBAGENTS),
        bool(_m.NX_VISION and st._NATIVE_TOOL_CALLS_CONFIRMED),
        bool(plan_active()),
        bool(getattr(_m, "NX_OBS", 0)),
        bool(getattr(_m, "NX_WEBFETCH", 0)),
        tuple(sorted(broken)) if broken else (),
    )


def _active_tools() -> list[dict[str, Any]]:
    """The tool schema list to send this turn (0.62.0 tiered registry).

    wait_background is included only once run_bash has actually backgrounded
    this session — see run_tool(); spawn_agent (read-only sub-agent) only when
    NX_SUBAGENTS=1; read_image only when NX_VISION=1 AND native tool_calls have
    been confirmed this session (a text-protocol tool result can't carry a
    content-parts array); plan_edit only while a plan is active (/plan <goal> —
    see _plan.py). Memoized on the gate-vector: one tuple rebuild + dict hit
    per turn instead of five full list copies (PERF §6 item 1)."""
    import nx as _m
    from ._breaker import _breaker_views
    key = _gate_vector()
    cached = _TOOL_LIST_CACHE.get(key)
    if cached is not None:
        return cached
    broken = set(key[-1])
    tools = []
    for t in _m.TOOLS:
        name = t["function"]["name"]
        gate = _TOOL_GATES.get(name)
        if gate is not None and not gate():
            continue
        if name in broken:
            continue
        tools.append(t)
    _TOOL_LIST_CACHE.clear()  # keep exactly the current gate-vector hot
    _TOOL_LIST_CACHE[key] = tools
    return tools


def _invalidate_tool_list_cache() -> None:
    """Call when TOOLS membership itself changes (a schema is added/removed at
    runtime) — the gate-vector key cannot see that."""
    _TOOL_LIST_CACHE.clear()


def search_tools(pattern: str = "") -> str:
    """The tier-2 meta-tool (0.62.0, PERF §6 item 2): surfaces the full schema
    of a tool that `_active_tools()` withheld this turn (gated, broken, or
    tier-2-only), so the model can call it next turn without it costing tokens
    in sessions that never need it. The schema literal lives in __init__.py
    TOOLS (single source); this module defines the body."""
    import nx as _m
    active = {t["function"]["name"] for t in _active_tools()}
    listed = []
    for t in _m.TOOLS:
        name = t["function"]["name"]
        if name in active:
            continue
        if pattern:
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return f"invalid pattern {pattern!r}: {e}"
            blob = name + " " + json.dumps(t)
            if not rx.search(blob):
                continue
        listed.append(f"{name}: {json.dumps(t)}")
    if not listed:
        return "No withheld tools matched" + (f" {pattern!r}." if pattern else ".")
    return ("Withheld tool schemas (not offered this turn; a call to one next "
            "turn opens the gate or is refused with its reason):\n"
            + "\n".join(listed))


def _withheld_tool_error(name: str) -> str:
    """Refusal for a tool that exists but is NOT offered this turn: a session
    gate (sub-agents off, no backgrounded job yet, no plan active) or the
    malformed-args breaker. run_tool() applies it to RESOLVED names (0.40) —
    a repaired spelling like `multi_edits` -> `multi_edit` used to bypass both
    the early breaker check (which saw only the emitted name) and the gate
    checks, and the fall-through 'unknown tool' text then sent the model off to
    invent a substitute instead of fixing its spelling or its formatting.
    Empty for a tool that is in fact offered, so callers fall through."""
    from ._breaker import _breaker_broken
    if _breaker_broken(name):
        from ._turn import _malformed_args_nudge  # call-time: _turn imports this module
        return _malformed_args_nudge(name, broken_now=True)
    if name in {t["function"]["name"] for t in _active_tools()}:
        return ""
    return _unknown_tool_error(name)


def _unknown_tool_error(name: str) -> str:
    """Error text for a tool name that resolved to nothing in the dispatch table.

    Three distinct situations hide behind a failed lookup, and a flat "unknown
    tool" answers only the first: (a) the name really is made up (suggest the
    closest real names); (b) the tool exists but is withheld this session by a
    gate — the model never saw its schema this turn, so a bare "unknown" reads
    as "this capability does not exist at all" and the model abandons the task
    (small models do this; seen live with wait_background/read_image/spawn_agent);
    (c) the malformed-args circuit breaker dropped it — the model must know the
    tool died for cause, and that the failure is about ITS formatting, not the
    tool's existence. Never raises; unknown names fall to the generic wording."""
    import nx as _m
    known = {t["function"]["name"] for t in _m.TOOLS}
    known.update(t["function"]["name"] for t in getattr(_m, "_EXTRA_TOOL_SCHEMAS", ()) or ())
    if name not in known:
        import difflib
        suggestions = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.7)
        if suggestions:
            return (f"ERROR: unknown tool {name} — did you mean {', '.join(suggestions)}? "
                    f"Available tools: {', '.join(sorted(known))}")
        return f"ERROR: unknown tool {name} — Available tools: {', '.join(sorted(known))}"
    # Withheld this session: name the gate and the alternative, so the model
    # doesn't conclude the capability doesn't exist.
    if name == "wait_background":
        return (f"ERROR: {name} is not available in this session (no command has backgrounded "
                "yet — nothing to wait on). run_bash returns a '[background] PID' line when a "
                "command needs waiting on; read_file that log path in the meantime.")
    if name == "read_image":
        return (f"ERROR: {name} is not available in this session (NX_VISION is off, or native "
                "tool calls are not confirmed on this source). Describe images by other means "
                "or ask the user to paste the relevant details.")
    if name == "spawn_agent":
        return (f"ERROR: {name} is not available in this session (NX_SUBAGENTS is off). Do the "
                "investigation yourself with search_files/read_file/list_dir.")
    if name == "plan_edit":
        return (f"ERROR: {name} is not available in this session (no plan is active). Start one "
                "with /plan <goal>, or just proceed with the normal edit tools.")
    if name == "obs_recall":
        return (f"ERROR: {name} is not available in this session (NX_OBS is off — "
                "nothing was packed, so there is nothing to recall).")
    if name == "web_fetch":
        return (f"ERROR: {name} is not available in this session (NX_WEBFETCH is off, "
                "or the gate has not opened). Work from local files, or ask the user "
                "to paste the relevant page/search results.")
    return (f"ERROR: {name} is not available in this session (withheld by the current session "
            "configuration). Use one of: " + ", ".join(sorted(known)))
