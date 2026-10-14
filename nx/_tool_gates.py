"""The tiered tool registry (0.62.0, PERF §6 items 1-2): gates, the active
schema list, the tier-2 search_tools meta-tool, and the withheld-tool
refusals — the SINGLE SOURCE for everything "is this tool offered this turn".

A declarative (name -> gate) table replacing five sequential O(N) filter
passes over the schema list. `_active_tools()` is memoized on the gate-vector
(one dict lookup + one list comp per turn). `_gate_vector()` derives from
_TOOL_GATES itself, so adding a gate is a one-entry edit. The gate registry
also drives run_tool()'s pre-dispatch refusal re-check (`run_tool` calls
`_gate_closed()`) and `_unknown_tool_error()`'s per-gate wording — the three
places that used to need parallel edits per new gate.

Tier 2 (meta-tool) is not yet occupied by any schema: `search_tools` lifts a
tier-2 tool's full schema into context on demand, so a future always-on tool
can ship at ~0 token rent instead of ~150 (the 0.62 feature-cost rule).

Split out of `_dispatch.py` (which keeps the DISPATCH table and run_tool)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from . import _state as st
from ._plan import plan_active
from ._delegation_note import _delegation_enabled
from ._obs import obs_enabled as _obs_enabled  # noqa: F401 (gate callable)

def _floor_slimmed() -> bool:
    """NX_TOOL_FLOOR_SLIM=1 withholds the slimming-set schemas behind
    search_tools (swarm P13): every agent in a swarm pays the always-on
    floor ×N, so the tools a swarm member rarely calls on turn 1 (duckdb,
    read_image) cost 0 schema tokens until the model asks search_tools for
    them. Purely additive: withheld tools still dispatch via search_tools
    discovery + the normal refusal wording."""
    from ._env import _env_int
    return _env_int("NX_TOOL_FLOOR_SLIM", 0) == 1


# Tools withheld when the floor-slim gate is on (a gate entry below decides
# their per-turn visibility; this predicate is ANDed into it).
_SLIMMABLE = ("duckdb", "read_image")


_TOOL_GATES: dict[str, Any] = {
    # name -> callable returning True when the schema should be OFFERED this
    # turn. Unlisted names are always-on. The malformed-args breaker is a veto
    # applied after the gates (a broken tool is dropped, not gated in).
    # The per-gate refusal wording lives in _GATE_REFUSALS below; a gate entry
    # without a refusal falls back to the generic withheld wording.
    "wait_background": lambda: st._SESSION_HAS_BACKGROUNDED,
    # kill_background (0.113.2): same gate as wait_background — the tool can
    # only act on jobs THIS session started, so a session where nothing has
    # backgrounded has nothing to kill and shouldn't pay the schema rent.
    "kill_background": lambda: st._SESSION_HAS_BACKGROUNDED,
    "spawn_agent": _delegation_enabled,
    "read_image": lambda: (bool(_m_safe().NX_VISION)
                            and st._NATIVE_TOOL_CALLS_CONFIRMED
                            and not _floor_slimmed()),
    "plan_edit": plan_active,
    # SoL-Pi ports (0.67.0): obs_recall only once a pack handle exists
    # (packing never happened otherwise — nothing to recall).
    # obs_recall only once a pack handle exists (packing never happened
    # otherwise — nothing to recall). P10: the gate reads the env at call
    # time (via _obs.obs_enabled) instead of the import-time mirror.
    "obs_recall": lambda: _obs_enabled(),
    # web_fetch (0.70.0 tier-2 -> tier-1 always-on in 0.113.0): keyless
    # DuckDuckGo-lite search + page fetch, promoted because it is exactly the
    # tool a session needs on turn 1 (the outside world) and search_tools
    # discovery cost a round-trip exactly then. Not listed in _TOOL_GATES at
    # all — the schema is only appended to TOOLS when NX_WEBFETCH != 0, so
    # NX_WEBFETCH=0 (offline hosts) removes it from the catalog entirely.
    # duckdb (0.87.0, always-on by design): NX_TOOL_FLOOR_SLIM=1 demotes it
    # to tier-2 for swarms — a member rarely analyzes data on turn 1, and
    # ~310 tok ×N agents of rent is the point of the slimming knob.
    "duckdb": lambda: not _floor_slimmed(),
    # lint_check (0.94.0): tier-2, opt-in via NX_LINT=1. Read-only lint over
    # edited files without an approval-gated run_bash round-trip.
    "lint_check": lambda: bool(getattr(_m_safe(), "NX_LINT", 0)),
    # Stable gateways for project-local manifest tools. They are opt-in, so
    # ordinary sessions retain their existing prompt floor; once enabled the
    # two generic schemas never vary with the project's actual tool catalog.
    "search_project_tools": lambda: bool(os.environ.get("NX_PROJECT_TOOLS", "") not in ("", "0", "false", "False")),
    "call_project_tool": lambda: bool(os.environ.get("NX_PROJECT_TOOLS", "") not in ("", "0", "false", "False")),
}

# Per-gate refusal text (the "why it's closed and what to do instead" line the
# model needs so it doesn't conclude the capability doesn't exist). Keyed by
# tool name; a gated tool without an entry gets the generic withheld wording
# in _unknown_tool_error.
_GATE_REFUSALS: dict[str, str] = {
    "wait_background": "(no command has backgrounded yet — nothing to wait on). run_bash returns a '[background] PID' line when a command needs waiting on; read_file that log path in the meantime.",
    "kill_background": "(no command has backgrounded yet in this session — nothing to kill, and the tool only reaches jobs this session started).",
    "read_image": "(NX_VISION is off, native tool calls are not confirmed on this source, or NX_TOOL_FLOOR_SLIM=1 withholds it this session). Describe images by other means or ask the user to paste the relevant details.",
    "spawn_agent": "(NX_SUBAGENTS is off). Do the investigation yourself with search_files/read_file/run_bash.",
    "plan_edit": "(no plan is active). Start one with /plan <goal>, or just proceed with the normal edit tools.",
    "obs_recall": "(NX_OBS is off — nothing was packed, so there is nothing to recall).",
    "lint_check": "(NX_LINT is off — linting is disabled this session). Fix style issues yourself from the diff, or run the linter with run_bash if the user asks.",
    "duckdb": "(NX_TOOL_FLOOR_SLIM=1 — the SQL tool is tier-2 in this session to keep the shared floor small). Ask search_tools for its schema, or shell out to the duckdb CLI with run_bash.",
    "search_project_tools": "(NX_PROJECT_TOOLS is off). Set NX_PROJECT_TOOLS=1 only for a repository whose .nx/tools/ contents you trust.",
    "call_project_tool": "(NX_PROJECT_TOOLS is off). Set NX_PROJECT_TOOLS=1 only for a repository whose .nx/tools/ contents you trust.",
}

_TOOL_LIST_CACHE: dict[tuple, list] = {}


def _m_safe():
    import nx as _m
    return _m


def _gate_vector() -> tuple:
    """The memo key for _active_tools(): the evaluated state of EVERY gate,
    plus the breaker's broken set. Derived from _TOOL_GATES itself — adding a
    gate is a one-entry edit, not a parallel-tuple edit (the old parallel
    tuple of positional booleans silently mis-keyed the cache when a gate was
    added to one table and not the other)."""
    from ._breaker import _breaker_views
    broken = _breaker_views()[1]
    gate_state = tuple(sorted(
        (name, bool(gate())) for name, gate in _TOOL_GATES.items()))
    return (gate_state, tuple(sorted(broken)) if broken else ())


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
    # A sub-agent loop gets ONLY its tier's schemas (0.96.0): before this
    # branch the sub-loop was sent the parent's full active tool list while
    # run_tool enforced the restriction at dispatch time — so the agent saw
    # (and was tempted by) schemas for tools it could never call. Now the
    # per-thread sub-agent context selects the schema set, matching the
    # restricted dispatch table exactly. Memoization is bypassed on this path
    # (per-thread, not per-process state).
    from ._subagent_tiers import _subagent_ctx, _subagent_tools
    if _subagent_ctx() is not None:
        return _subagent_tools()
    key = _gate_vector()
    cached = _TOOL_LIST_CACHE.get(key)
    if cached is not None:
        return cached
    broken = set(key[1])
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
    # Retired-from-the-floor tools: dispatchable but never offered, so a model
    # emitting one gets the actionable "retired, use X instead" story — with
    # the native replacement named — instead of a bare unknown-tool error.
    _RETIRED_REFUSALS = {
        "list_dir": ("ERROR: list_dir is retired in this session — list a "
                     "directory with run_bash `ls [path]` (flags/pipes/filters "
                     "work), and search file NAMES + contents with search_files."),
        "git_status": ("ERROR: git_status is retired in this session — run "
                       "run_bash `git status --short -b` for branch + changed files."),
        "git_diff": ("ERROR: git_diff is retired in this session — run "
                     "run_bash `git diff` (or `git diff --staged`, `git diff -- <file>`)."),
        "git_log": ("ERROR: git_log is retired in this session — run "
                    "run_bash `git log --oneline` (scope with `-- <file>`)."),
        "sed_edit": ("ERROR: sed_edit is retired in this session — use "
                     "edit_file for a literal one-file replacement, edit_file "
                     "`diff` mode for a regex/multi-file restructure, or run_bash "
                     "for anything else."),
        # 0.158.11: apply_patch stopped being a tool at all — the same engine is
        # edit_file's `diff` mode, so the name is not even dispatchable now.
        "apply_patch": ("ERROR: apply_patch is retired in this session — pass the "
                        "same unified diff to edit_file as its `diff` argument "
                        "(atomic, every file the diff touches, `path` = the base "
                        "directory)."),
    }
    if name in _RETIRED_REFUSALS:
        try:
            from ._logging import usage_gate
            usage_gate(name, "retired")
        except Exception:  # noqa: BLE001 — telemetry must never break a refusal
            pass
        return _RETIRED_REFUSALS[name]
    if name not in known:
        import difflib
        suggestions = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.7)
        if suggestions:
            return (f"ERROR: unknown tool {name} — did you mean {', '.join(suggestions)}? "
                    f"Available tools: {', '.join(sorted(known))}")
        return f"ERROR: unknown tool {name} — Available tools: {', '.join(sorted(known))}"
    # Withheld this session: name the gate and the alternative, so the model
    # doesn't conclude the capability doesn't exist. The wording comes from
    # _GATE_REFUSALS (one entry per gate — the same registry the gate itself
    # lives in); a gated tool without an entry gets the generic wording.
    # v1 usage metrics: withheld-by-gate refusal — the model asked for a
    # tool whose schema was never offered this turn (missing-tool signal).
    if name in _GATE_REFUSALS:
        try:
            from ._logging import usage_gate
            usage_gate(name, "gate")
        except Exception:  # noqa: BLE001 — telemetry must never break a refusal
            pass
        return f"ERROR: {name} is not available in this session {_GATE_REFUSALS[name]}"
    # (The retired-tool table lives above the unknown-name check — a retired
    # name is by definition absent from TOOLS, so it must be answered there or
    # it degrades to a bare unknown-tool error. 0.158.11 removed the duplicate
    # copy that used to sit here; it was unreachable behind that first check.)
    try:
        from ._logging import usage_gate
        usage_gate(name, "withheld")
    except Exception:  # noqa: BLE001 — telemetry must never break a refusal
        pass
    return (f"ERROR: {name} is not available in this session (withheld by the current session "
            "configuration). Use one of: " + ", ".join(sorted(known)))


def _gate_closed(name: str) -> bool:
    """True when `name` has a gate in _TOOL_GATES and that gate is CLOSED this
    turn. run_tool()'s pre-dispatch re-check calls this instead of repeating
    each gate's condition inline — a new gate needs one entry in _TOOL_GATES,
    not a fourth hardcoded branch."""
    gate = _TOOL_GATES.get(name)
    return gate is not None and not gate()
