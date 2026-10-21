"""PTC Phase 4 rollout policy (docs/ptc-migration-spec.md §Phase 4).

The `NX_PTC=auto` default, kept in its own zero-import-weight module (the
spec's open item "Phase 4's per-model list needs a home" — `nx/_subagent_model.py`
is the model-ID-classification precedent this mirrors). Two pieces:

* `ptc_rollout_enabled(model_id)` — the per-model enablement heuristic, and
* `_ptc_enabled()` — the live resolver the `run_tools` GATE reads
  (`nx/_tool_gates._TOOL_GATES["run_tools"]`), which layers the session's
  fallback-ladder drop on top of it.

Resolution order (spec §Phase 4):

    NX_PTC=1   -> hard override ON. Beats everything, including the
                  fallback-ladder drop: this is the documented, reversible
                  "force it back on" knob (`NX_PTC=0` / `NX_PTC=auto` undo it).
    NX_PTC=0   -> hard override OFF.
    auto/unset -> the session fallback ladder first (`st._PTC_DISABLED`, set by
                  the double-trip hook in nx/_breaker.py), then the
                  `nx.NX_PTC` attribute — which nx/__init__.py sets to the
                  heuristic's answer for the STARTUP model, and which tests and
                  the REPL flip directly (the pre-Phase-4 contract every
                  existing PTC test relies on).
"""
from __future__ import annotations

import os

from . import _state as st

# The per-model `NX_PTC=auto` heuristic (spec §Phase 4; data: arXiv 2608.06370,
# BFCL v4, 14 models). Positive generations — the ones that track or beat JSON
# tool calling: ALL current Anthropic models and the recent OpenAI reasoning
# generations. Negative generations — the ones that REGRESS in the same data —
# win over the positive list, and an unknown id matches nothing and stays off
# (§Problem: "older/smaller models regress under PTC; JSON stays as the
# default-off-ramp"). Deliberately ONE small prefix-tuple pair rather than a
# registry: this classifies GENERATIONS, not models. Update it (and the matrix
# in tests/test_ptc_rollout.py) when a new generation ships.
_PTC_AUTO_ON_PREFIXES: tuple[str, ...] = (
    "claude",    # all current Anthropic models (claude-*)
    "gpt-5",     # GPT-5-class reasoning generations (gpt-5*)
    "o3",        # o3* reasoning generations
    "o4",        # o4* reasoning generations
)
_PTC_AUTO_OFF_PREFIXES: tuple[str, ...] = (
    "gpt-4o",    # gpt-4o* class: regresses under PTC (arXiv 2608.06370)
    "gpt-4.1",   # gpt-4.1* class: regresses under PTC (arXiv 2608.06370)
    "gpt-4",     # the rest of the gpt-4 class (catches gpt-45 / gpt-4-turbo too)
    "o1",        # first-gen reasoning: same failure class as the 4o family
)

# How many run_tools breaker TRIPS (a trip = the breaker OPENING for the tool,
# re-armed by a decay) drop the tool for the session. Spec §Phase 4: "twice".
PTC_TRIP_LIMIT = 2


def _normalize_ptc_setting(raw: object) -> str:
    """Any knob value (env string, bool, the legacy 0/1 int) -> "1"|"0"|"auto".
    Anything unrecognized reads as "auto" (the Phase 4 default), so a typo
    degrades to the per-model heuristic instead of silently disabling PTC."""
    if isinstance(raw, bool):
        return "1" if raw else "0"
    if isinstance(raw, (int, float)):
        return "1" if raw else "0"
    val = str(raw if raw is not None else "").strip().lower()
    if val in ("1", "true", "on", "yes"):
        return "1"
    if val in ("0", "false", "off", "no"):
        return "0"
    return "auto"


def ptc_rollout_enabled(model_id: str | None = None) -> bool:
    """The `NX_PTC=auto` decision for one model id: True when its generation is
    on the positive list and not on the negative one. Pure function of the id
    string, so the rollout matrix is testable with zero session state.
    `model_id=None` classifies the session's active model (st.MODEL); an empty
    id classifies as OFF (nothing proven). A provider-qualified id
    ("together:claude-sonnet-4-5", "zai-org/GLM-5.3") is classified on the
    final ':'/'/' segment."""
    mid = st.MODEL if model_id is None else model_id
    mid = str(mid or "").strip().lower()
    if not mid:
        return False
    fam = mid.replace(":", "/").split("/")[-1]
    if fam.startswith(_PTC_AUTO_OFF_PREFIXES):
        return False
    return fam.startswith(_PTC_AUTO_ON_PREFIXES)


def resolve_ptc_setting(model_id: str | None = None) -> int:
    """The value nx/__init__.py bakes into `nx.NX_PTC` at import (and what a
    model/source switch re-bakes): 1/0 for the given (or active) model, with an
    explicit env override winning over the heuristic."""
    setting = _normalize_ptc_setting(os.environ.get("NX_PTC", ""))
    if setting == "1":
        return 1
    if setting == "0":
        return 0
    return 1 if ptc_rollout_enabled(model_id) else 0


def _ptc_env_override() -> str:
    """The raw env knob, normalized: "1"/"0" when it is a hard override, else
    "" (read at CALL time, per the repo's no-import-time-freeze rule)."""
    setting = _normalize_ptc_setting(os.environ.get("NX_PTC", ""))
    return "" if setting == "auto" else setting


def _ptc_ladder_active() -> bool:
    """True when the session fallback ladder has dropped run_tools (2 breaker
    trips — see note_breaker_trip in nx/_breaker.py)."""
    return bool(getattr(st, "_PTC_DISABLED", False))


def ptc_ladder_disabled() -> bool:
    """Public read of the ladder flag (tests + the /status surface)."""
    return _ptc_ladder_active()


def clear_ptc_ladder() -> None:
    """Re-arm the session for run_tools: clears the drop and the trip counter
    and re-arms the breaker's open-edge watcher. Called by the `/toolcalls
    reset` path alongside the breaker reset (the ladder is reversible by
    design — spec §Phase 4: 'One knob, reversible'; NX_PTC=1 also defeats the
    drop without clearing it)."""
    st._PTC_DISABLED = False
    st._PTC_BREAKER_TRIPS = 0
    from ._breaker import rearm_run_tools_trip_watch
    rearm_run_tools_trip_watch()


def _ptc_enabled() -> bool:
    """True when the run_tools schema should be offered THIS turn.

    Order (spec §Phase 4): the env hard override, then the fallback ladder,
    then the `nx.NX_PTC` attribute (the heuristic's answer for the active
    model as of import / the last model switch, or whatever a test or the REPL
    set it to)."""
    override = _ptc_env_override()
    if override == "1":
        return True
    if override == "0":
        return False
    if _ptc_ladder_active():
        return False
    import nx as _m
    return bool(getattr(_m, "NX_PTC", 0))


def ptc_status_line() -> str:
    """One-line description of the rollout state (the /status + footer view:
    how the knob resolved, and whether the ladder dropped the tool)."""
    override = _ptc_env_override()
    if override:
        how = "forced on (NX_PTC=1)" if override == "1" else "forced off (NX_PTC=0)"
    elif _ptc_ladder_active():
        how = (f"dropped this session by the fallback ladder after "
               f"{getattr(st, '_PTC_BREAKER_TRIPS', 0)} run_tools breaker trips "
               f"(NX_PTC=1 forces it back on)")
    else:
        how = f"auto per-model heuristic (model={st.MODEL!r})"
    return f"run_tools: {'ON' if _ptc_enabled() else 'off'} — {how}"
