"""The malformed-args circuit breaker: per-endpoint evidence containers.

Split out of `_dispatch.py` (which keeps run_tool, the sole caller of
`_breaker_broken` in the gate path). 0.40: the breaker is evidence about ONE
endpoint — it used to be process-global, so switching to a healthy source left
the model permanently missing tools that endpoint handles fine. Every reader
goes through `_breaker_views()`: if the (source, model) identity changed since
the evidence was gathered, the streaks are dropped in place (clear, not
reassign — the conftest state rewind and the `nx.<name>` proxy both rely on
container identity)."""
from __future__ import annotations

from . import _state as st

# Tools whose breaker TRIP was already announced on the usage stream (v1
# metrics): one `breaker` event per open, re-armed when the tool decays or
# the evidence re-keys to a new endpoint.
_TRIP_ANNOUNCED: set[str] = set()


def _breaker_endpoint_key() -> tuple[str | None, str | None]:
    """Identity of the endpoint the malformed-args evidence was gathered
    against: (active source name, active model). A per-model key because
    `/source <name> <model>` can point one Source at any model on a
    multi-model host, and truncation is a property of the weights + server
    pair, not of the Source row."""
    return (st.ACTIVE.name if st.ACTIVE is not None else None, st.MODEL)


def _breaker_views() -> tuple[dict[str, int], set[str]]:
    """(failures, broken) — the breaker's live containers, after re-keying.

    0.40: the breaker is evidence about ONE endpoint. It used to be
    process-global, so switching to a healthy source left the model permanently
    missing tools that endpoint handles fine. Every reader goes through here:
    if the (source, model) identity changed since the evidence was gathered,
    the streaks are dropped in place (clear, not reassign — the conftest state
    rewind and `nx.<name>` proxy both rely on container identity).

    Lazy rather than a hook in switch_source(): tests (and future callers) may
    reassign st.ACTIVE/st.MODEL directly, and a read-time check cannot be
    bypassed by forgetting to call a notifier.

    The first observation just adopts the key without clearing — there is no
    prior-endpoint evidence to drop yet, and the containers may legitimately
    hold state installed by a caller (a test, or /toolcalls reset) rather than
    by an observed truncation."""
    key = _breaker_endpoint_key()
    observed = st._BREAKER_ENDPOINT_KEY
    if observed != key:
        if observed != st.BREAKER_UNOBSERVED:
            st._MALFORMED_TOOL_FAILURES.clear()
            st._MALFORMED_TOOL_BROKEN.clear()
            _TRIP_ANNOUNCED.clear()
        st._BREAKER_ENDPOINT_KEY = key
    _note_run_tools_opens(st._MALFORMED_TOOL_BROKEN)
    return st._MALFORMED_TOOL_FAILURES, st._MALFORMED_TOOL_BROKEN


# PTC Phase 4 (0.164.0): liveness of the run_tools breaker for the fallback
# ladder — True while the breaker currently HOLDS run_tools. Trip counting
# diffs the broken set here (the one function every trip source's effect
# passes through every turn — _tool_gates._gate_vector reads it, and both
# trip sites write into the container it returns), so it cannot double-count
# repeated failures against one open and works for the PTC site (which writes
# the containers directly) without editing it.
_RUN_TOOLS_BROKEN_SEEN = False


def _note_run_tools_opens(broken: set) -> None:
    """Count breaker OPENS for run_tools; the 2nd drop-for-the-session.

    docs/ptc-migration-spec.md §Phase 4 fallback ladder: "if a session's
    breaker trips twice on the run_tools key, drop run_tools from the gate
    vector for that session and nudge back to JSON". An OPEN is the
    false→true edge of the name in `broken` (decay/reset re-arms the edge);
    the drop sets st._PTC_DISABLED (read by nx/_ptc_rollout._ptc_enabled and
    mirrored onto nx.NX_PTC) and clears the memoized tool list via
    _tool_gates. NX_PTC=1 (env) is the documented hard override that forces
    the tool back on without clearing the drop; a model/source switch or a
    full /toolcalls reset re-arms the ladder. Never raises into the gate."""
    global _RUN_TOOLS_BROKEN_SEEN
    seen_now = "run_tools" in broken
    if seen_now == _RUN_TOOLS_BROKEN_SEEN:
        return
    _RUN_TOOLS_BROKEN_SEEN = seen_now
    if not seen_now:
        return  # decay/reset re-armed the edge; nothing to count
    st._PTC_BREAKER_TRIPS = getattr(st, "_PTC_BREAKER_TRIPS", 0) + 1
    if st._PTC_BREAKER_TRIPS < 2:
        return
    st._PTC_DISABLED = True
    try:
        from ._tool_gates import ptc_fallback_ladder_engaged
        ptc_fallback_ladder_engaged()
    except Exception:  # noqa: BLE001 — the ladder is a nudge, never a crash
        pass


def rearm_run_tools_trip_watch() -> None:
    """Re-arm the ladder's edge detector (used by clear_ptc_ladder and the
    test suite; the diff in _note_run_tools_opens also self-re-arms once the
    name leaves `broken`)."""
    global _RUN_TOOLS_BROKEN_SEEN
    _RUN_TOOLS_BROKEN_SEEN = False


def _breaker_broken(name: str) -> bool:
    """True if `name` is currently dropped by the malformed-args breaker.

    v1 usage metrics: doubles as the trip ANNOUNCE point. The trip itself
    happens in _turn_toolcalls (broken.add); this is the first reader every
    path hits right after (run_tool's refusal gate and _withheld_tool_error
    both call it in the same turn as the trip), so announcing here emits the
    `breaker` usage event exactly once per open — with trip_count read live
    from the failures streak — without restructuring the trip site."""
    failures, broken = _breaker_views()
    if name in broken and name not in _TRIP_ANNOUNCED:
        _TRIP_ANNOUNCED.add(name)
        try:
            from ._logging import usage_breaker
            usage_breaker(name, failures.get(name, 0))
        except Exception:  # noqa: BLE001 — telemetry must never break a gate
            pass
        from ._breadcrumbs import note
        note(f"breaker tripped: {name} dropped after "
             f"{failures.get(name, 0)} malformed-args attempts")
        return True
    return name in broken


def _breaker_decay(name: str) -> None:
    """0.29.9: one clean execution of `name` clears its malformed-args streak
    (and re-arms the tool if the breaker had dropped it). Two consecutive
    truncations on a flaky local server were routine; losing a tool for the
    whole session on that evidence was too heavy. A deliberately BROKEN tool
    (the model can't reach it anyway) only decays when the user re-offers it
    via /toolcalls reset — direct run_tool() calls (tests, hooks) count.

    0.40: reads through _breaker_views() first, so a clean dispatch on a NEW
    endpoint can never inherit a stale `broken` mark from the previous one."""
    failures, broken = _breaker_views()
    if failures.pop(name, None) is not None:
        was_broken = name in broken
        broken.discard(name)
        _TRIP_ANNOUNCED.discard(name)
        if was_broken:
            from ._breadcrumbs import note
            note(f"breaker reset: {name} re-armed after a clean execution")
