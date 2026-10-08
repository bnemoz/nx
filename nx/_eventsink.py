"""Per-job ring-buffer event sink (EVENTSURFACE_SPEC Design §1, Phase 1).

One ring of bounded DISPLAY lines per spawned sub-agent, so a live /tabs
region can tail what a quiet agent is doing without ever touching the
console it shares with the parent turn. The sink is display-only: it
never enters `messages`, never reaches the model, and nothing here ever
prints. The only write is `write()`, the only read is `tail()`.

Locking (two levels, one order — never the reverse, so there is no cycle):

  _LOCK (module)   guards `_SINKS` (the registry) and `_RETAINED_TAILS`.
  sink._lock       guards that sink's ring: appends and tail snapshots
                   serialize on it, so a repaint never sees a torn line
                   and N concurrent writers never lose one.

`sink_for()` is the zero-cost path: a bare atomic dict read, NO lock.
Writers do `sink = sink_for(job_id)` and stop when it is None, so an
agent that never registers a sink pays one dict lookup per render event.

`drop_sink()` is retain-then-drop under a single _LOCK hold: it pops the
sink from the registry and copies its final tail into `_RETAINED_TAILS`
atomically, so a concurrent 4 fps dashboard repaint can never observe a
half-retired sink (line in neither place, or both). Retained tails are
post-mortem data for /tabs after the worker's finally ran; the dict is
capped at MAX_RETAINED_JOBS job ids (insertion order) so a long-lived
process cannot grow it without bound.

`kind` is one of "stream" (model answer deltas), "tool" (tool-call
start/finish), "status" (turn counters). An invalid kind is accepted and
stored verbatim — validating it is the caller's contract, not ours.
"""
from __future__ import annotations

import os
import threading
from collections import deque

# Ring capacity and retention cap. NX_EVENT_TAIL_LINES is read ONCE at
# first sink creation (absent/non-int/<=0 falls back to the default);
# MAX_RETAINED_JOBS is a fixed cap on post-mortem tails.
DEFAULT_TAIL_LINES = 200
MAX_RETAINED_JOBS = 16

# The one module-level lock: guards _SINKS AND _RETAINED_TAILS together,
# which is what makes drop_sink's retain-then-drop atomic.
_LOCK = threading.Lock()
_SINKS: dict[str, "_EventSink"] = {}
# Final tail of dropped jobs, newest-dropped last; insertion order is the
# eviction order. Values are the same rendered strings tail() returns.
_RETAINED_TAILS: dict[str, list[str]] = {}

# NX_EVENT_TAIL_LINES cache: None = not read yet. Tests reset it to None.
_TAIL_LINES: int | None = None


def _tail_lines() -> int:
    """Ring capacity in lines, read from the env once at first use.

    Absent, non-int, or <= 0 all fall back to DEFAULT_TAIL_LINES (same
    posture as other NX_* int tunables: a bad value must never disable
    the sink or make it unbounded)."""
    global _TAIL_LINES
    if _TAIL_LINES is None:
        try:
            n = int(os.environ.get("NX_EVENT_TAIL_LINES", "").strip())
        except ValueError:
            n = DEFAULT_TAIL_LINES
        _TAIL_LINES = n if n > 0 else DEFAULT_TAIL_LINES
    return _TAIL_LINES


class _EventSink:
    """Bounded ring of rendered event lines for one job id.

    One write = one ring line, rendered as `[kind] text`; callers that
    hold multi-line text split before writing (the ring counts LINES, and
    its bound is the display guarantee the /tabs region relies on)."""

    __slots__ = ("job_id", "_lock", "_lines")

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=_tail_lines())

    def write(self, job_id: str, kind: str, text: str) -> None:
        """Append one event line. `job_id` mirrors the spec's call shape
        (the tap already has it); the sink's own ring is the target.
        Invalid kinds are stored verbatim — see module docstring."""
        with self._lock:
            self._lines.append(f"[{kind}] {text}")

    def tail(self, job_id: str | None = None,
             lines: int | None = None) -> list[str]:
        """Most recent lines, oldest -> newest; `lines` caps the cut.
        Snapshot under the sink lock, so a repaint reads a coherent ring
        even with writers appending on other threads."""
        with self._lock:
            out = list(self._lines)
        if lines is None:
            return out
        return out[-lines:] if lines > 0 else []


def sink_for(job_id: str | None) -> "_EventSink | None":
    """The sink registered for `job_id`, or None. THE zero-cost path:
    one atomic dict read, no lock, no allocation — every Phase-2 writer
    sits behind `is not None` on this. None job id short-circuits to the
    same None result (the `_batch`/`_telegram` quiet sites have no id)."""
    if not job_id:
        return None
    return _SINKS.get(job_id)


def register_sink(job_id: str) -> "_EventSink":
    """Create and register the sink for a spawning job; returns it.
    Re-registering an id replaces its ring (job ids are unique per
    spawn, so this is a fresh-start guarantee, not a reset path)."""
    sink = _EventSink(job_id)
    with _LOCK:
        _SINKS[job_id] = sink
    return sink


def drop_sink(job_id: str) -> None:
    """Retire a finished job's sink: retain-then-drop in ONE _LOCK hold
    (spec §1) so a concurrent repaint cannot see a half-retired sink.
    The final tail moves to _RETAINED_TAILS for post-mortem /tabs reads;
    the cap keeps the retained set bounded (oldest-dropped evicted).
    Unknown id: no-op, nothing retained."""
    with _LOCK:
        sink = _SINKS.pop(job_id, None)
        if sink is None:
            return
        _RETAINED_TAILS[job_id] = sink.tail()
        while len(_RETAINED_TAILS) > MAX_RETAINED_JOBS:
            _RETAINED_TAILS.pop(next(iter(_RETAINED_TAILS)))


def retained_tail(job_id: str, lines: int | None = None) -> list[str]:
    """Final tail of a dropped job (same rendered strings, oldest ->
    newest), read under the same module lock that performed the retain
    — so it can never disagree with what drop_sink captured."""
    with _LOCK:
        out = list(_RETAINED_TAILS.get(job_id, ()))
    if lines is None:
        return out
    return out[-lines:] if lines > 0 else []
