"""The per-thread sub-agent context slot, shared by the tiers and async
machinery — plus the two event-sink helpers keyed by job id.

Small dedicated home for `_SUBAGENT_TLS` / `_subagent_ctx` so `_subagent_tiers`
and `_subagent` (async workers) can share the one slot without a cycle —
the tiers module must not import the async-heavy `_subagent` at module level.
`tap_event` / `sink_tail` (EVENTSURFACE_SPEC Phase 2) live here for the same
reason: they are pure job-id -> sink plumbing with no heavy imports, so the
render paths (`_stream_parse`, `_dispatch`) and the /tabs snapshot can call them
without pulling in the sub-agent machinery.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._subagent_tiers import _SubagentContext

# Per-agent sub-loop context (see _subagent_tiers for why it's thread-local).
_SUBAGENT_TLS = threading.local()


def _subagent_ctx() -> "_SubagentContext | None":
    return getattr(_SUBAGENT_TLS, "ctx", None)


def tap_event(kind: str, text: str) -> None:
    """Phase 2 event-sink writer (EVENTSURFACE_SPEC §1): append a DISPLAY line
    to the CURRENT thread's agent sink, if one is attached.

    Lives here because the job-id resolution IS this module's job: the tap
    reads the per-thread context `_run_sync_agent_body` installed, which is
    what lets a fork inside the shared render paths land on the right agent
    with no `StreamParser` plumbing change.

    Zero cost in the common case: the registry dict is empty unless a sink was
    ever registered, so an agent with no dashboard open pays one truthiness
    check. `job_id` is None on the parent thread, on the `_batch.py`/
    `_telegram.py` quiet sites (quiet with no ctx) and on `_spec_reads`' pool
    threads (TLS does not propagate there) — all of which correctly emit
    nothing rather than erroring.

    Nothing here prints, and nothing here reaches `messages`: the sink is a
    display-only ring (see `nx/_eventsink.py`). A multi-line delta is split so
    one ring entry is one display line, as the ring's bound assumes.
    """
    if not text:
        return
    from . import _eventsink as _es
    if not _es._SINKS:  # zero-cost fast path: no sink ever attached
        return
    job_id = getattr(_subagent_ctx(), "job_id", None)
    sink = _es.sink_for(job_id)
    if sink is None or job_id is None:
        # sink_for() short-circuits a None id to None, so this arm is
        # unreachable in practice; it narrows `job_id` for sink.write's `str`.
        return
    for line in text.splitlines():
        if line.strip():
            sink.write(job_id, kind, line[:400])


def sink_tail(job_id: "str | None", lines: "int | None" = None) -> list[str]:
    """Display tail for `job_id`: the LIVE ring while the job runs, else the
    post-mortem retained copy. Empty when neither exists.

    One call because a /tabs row cannot tell "still running" from "just retired"
    apart: the worker's `finally` retires the sink the moment it stops, and a
    repaint landing between those two states must not render a blank row.
    Composed entirely from Phase 1's public API — `sink_for`'s bare dict read
    plus the sink's own lock, or the module lock behind `retained_tail` — so
    this adds no state and no new lock ordering.
    """
    if not job_id:
        return []
    from . import _eventsink as _es
    sink = _es.sink_for(job_id)
    if sink is not None:
        return sink.tail(job_id, lines)
    return _es.retained_tail(job_id, lines)
