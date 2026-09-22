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
        st._BREAKER_ENDPOINT_KEY = key
    return st._MALFORMED_TOOL_FAILURES, st._MALFORMED_TOOL_BROKEN


def _breaker_broken(name: str) -> bool:
    """True if `name` is currently dropped by the malformed-args breaker."""
    return name in _breaker_views()[1]


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
        broken.discard(name)
