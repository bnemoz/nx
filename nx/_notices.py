"""Per-thread notice slots (swarm audit P2, 0.98).

The pending runtime-note slots (`st._PENDING_BG_COMPLETION_NOTICE`,
`st._PENDING_SUBAGENT_NOTICE`, `st._PENDING_VERIFY_NOTICE`, plus the
`_BG_SCAN_DONE` flag and `_BG_NOTICE_SEEN` set) used to be one process-wide
value each. With parallel in-process agents (`nx run --parallel`, async
fan-out under --yolo) every agent staged into and flushed from the SAME slot,
so agent A's finished answer could be appended to agent B's transcript
(`_batch.py` documented the failure).

This module gives each running turn loop its own slots, resolved per-thread
via a contextvar — the same shape `_recovery.py` uses for the recovery state,
but with plain module-dict defaults instead of data descriptors (the notice
slots have few call sites, all in this package, and tests write
`st._PENDING_*` directly; a descriptor on `_state` would also re-arm the
CPython 3.12+ LOAD_ATTR_MODULE hazard `_recovery._install_descriptors`
documents).

Lifecycle: a turn loop calls `install_notice_ctx()` once at its top and
`pop_notice_ctx()` in its finally. Reads/writes of the five names route:
loop context (per-thread) → module dict (the pre-swarm default store).
Nesting seeds from the enclosing context and copies back on pop, so a sync
sub-loop stages into the parent the way it always did.
"""
from __future__ import annotations

import contextvars
from typing import Any

from . import _state as st

_MISSING = object()

SLOTS = tuple(st._NOTICE_DEFAULTS.keys())


def _frame_slots() -> dict:
    """A frame's slot dict, patched to the CURRENT SLOTS on first touch: a
    module reload / conftest eviction that adds a routed name must not
    KeyError a frame installed before it."""
    fr = _CURRENT.get()
    assert fr is not None  # callers only reach this inside an installed frame; narrows for the checker
    slots = fr.slots
    if len(slots) != len(SLOTS):
        d = st._NOTICE_DEFAULTS
        for name in SLOTS:
            if name not in slots:
                slots[name] = d.get(name)
    return slots

_CURRENT: contextvars.ContextVar["NoticeFrame | None"] = \
    contextvars.ContextVar("nx_notice_ctx", default=None)


class NoticeFrame:
    """One turn loop's private notice slots, seeded from its parent."""

    __slots__ = ("slots", "seed", "prev")

    def __init__(self, prev: "NoticeFrame | None") -> None:
        self.prev = prev
        seed = prev.slots if prev is not None else None
        vals: dict[str, Any] = {}
        for name in SLOTS:
            if seed is not None:
                cur = seed[name]
            else:
                cur = vars(st).get(name, st._NOTICE_DEFAULTS[name])
            vals[name] = cur.copy() if isinstance(cur, (dict, set, list)) else cur
        self.slots = vals
        self.seed = dict(vals)


def current() -> NoticeFrame | None:
    """This thread's notice frame, or None outside a turn loop."""
    return _CURRENT.get()


def install_notice_ctx() -> NoticeFrame:
    fr = NoticeFrame(_CURRENT.get())
    _CURRENT.set(fr)
    return fr


def pop_notice_ctx(fr: NoticeFrame) -> None:
    """Fold this frame's staged notes back into its parent (or the module
    defaults) and pop. Symmetric with install_notice_ctx()."""
    if _CURRENT.get() is not fr:
        return  # stale finally after a swapped frame
    prev = fr.prev
    # Fold back only slots THIS frame changed. A worker thread that finished
    # mid-turn has no frame, so its staged note landed in the module store
    # (or a parent frame); an unconditional overwrite with our stale seed
    # would destroy it — and the sub-agent's `announced` flag is already
    # burned, so the note could never be re-staged (0.104 wake-up fix).
    if prev is not None:
        for name in SLOTS:
            if fr.slots[name] == fr.seed[name]:
                continue  # untouched here — keep whatever the parent has
            if name == "_BG_NOTICE_SEEN":
                # The seen-set is shared with the wake poller thread (which
                # has no frame and writes the module store): marks made while
                # this frame ran must survive the fold-back, so MERGE sets
                # instead of overwriting with this frame's stale copy. The
                # overwrite erased them, the next sweep re-staged the same
                # "pid N exited" note, and each re-announcement woke another
                # "Nothing pending" turn — repeating for as long as the
                # session ran (0.109.5).
                merged = set(fr.slots[name] or set())
                merged.update(prev.slots[name] or set())
                prev.slots[name] = merged
                continue
            prev.slots[name] = fr.slots[name]
    else:
        for name in SLOTS:
            if fr.slots[name] == fr.seed[name]:
                # Untouched here — a worker may have staged into the module
                # store while we ran; keep whatever it wrote.
                continue
            cur = vars(st).get(name, st._NOTICE_DEFAULTS.get(name))
            if (cur and fr.seed[name] == st._NOTICE_DEFAULTS.get(name)
                    and isinstance(cur, str) and isinstance(fr.slots[name], str)):
                # Both a worker and this frame staged a note: concatenate so
                # neither is lost.
                setattr(st, name, cur + fr.slots[name])
                continue
            # Write the plain module-global default (no dict-entry dance here:
            # notice slots have no hot read site, and tests rewind vars(st)).
            setattr(st, name, fr.slots[name])
    _CURRENT.set(fr.prev)


def get(name: str) -> Any:
    fr = _CURRENT.get()
    if fr is not None:
        return _frame_slots().get(name, st._NOTICE_DEFAULTS.get(name))
    # vars(st) first (tests rewind/set it directly), then the import-time
    # default. The bare st._NOTICE_DEFAULTS lookup must not assume every name
    # exists on _state — a conftest rewind can drop names added later.
    v = vars(st).get(name, _MISSING)
    if v is not _MISSING:
        return v
    return st._NOTICE_DEFAULTS.get(name)


def set_(name: str, value: Any) -> None:
    fr = _CURRENT.get()
    if fr is not None:
        fr.slots[name] = value
    else:
        setattr(st, name, value)
