"""RecoveryContext — per-loop recovery state, thread-isolated (0.42, roadmap 1.2).

nx's self-healing stack runs on process-wide globals shared with sub-agent
threads: an async sub-agent runs its own `_run_model_turn_loop` on a worker
thread and writes the SAME module globals the parent reads, so a sub-agent's
oversized prompt could trigger the parent's autocompress and a sub-agent's
malformed args could drop a tool from the parent's tool list.

The fix is ownership: one RecoveryContext per `_run_model_turn_loop`
invocation, resolved per current thread via a contextvar (held in `_state` so
module reloads can't strand it). A contextvar beats threading.local here
because sync sub-agents run the sub-loop ON the parent's thread — nesting,
with restore-on-exit — while async agents get a fresh context automatically
on their worker thread. Nested loops seed from the enclosing loop's context
and copy their final values back INTO it; the outermost loop copies back to
the module globals, so session-scoped evidence (breaker streaks, native
tool-call confirmation) survives the whole tree.

Back-compat is total by construction: the module dict of `nx._state` stays the
DEFAULT store. Data descriptors installed on `st`'s class route every
`st._X` read/write through the current context when a loop is running, and
straight to the dict otherwise — so tests that set `st._TALLY_RETRIES = 0`
before a loop, conftest's in-place `__dict__` rewind (which bypasses
descriptors and therefore always rewinds the default store), and the
`nx.<name>` proxy in `__init__.py` all keep working unchanged.

What must NOT move: process-wide config (env-resolved once, read-only at loop
time) — MALFORMED_TOOL_BREAKER_LIMIT and friends stay plain module globals,
exactly the roadmap's "thin facade" carve-out.
"""
from __future__ import annotations

import contextvars
from typing import Any

from . import _state as st

# The fields one loop owns, mapped 1:1 from the old process-wide globals.
FIELDS = (
    "_LAST_PROMPT_TOKENS",           # int: prompt size the loop last saw
    "_TALLY_ARG_REPAIRS",            # int: repairs applied this turn (footer)
    "_TALLY_RETRIES",                # int: recovery nudges/cuts this turn (footer)
    "_STALL_EPISODE_ACTED",          # bool: a tool ran after this stall nudge
    "_LAST_PARTIAL_TEXT",            # str: partial answer of the last length cut
    "_NATIVE_TOOL_CALLS_CONFIRMED",  # bool: endpoint uses native tool_calls
    "_MALFORMED_TOOL_FAILURES",      # dict[str, int]: malformed-args streaks
    "_MALFORMED_TOOL_BROKEN",        # set[str]: tools dropped by the breaker
    "_BREAKER_ENDPOINT_KEY",         # (source, model) the breaker evidence describes
)


class RecoveryContext:
    """One turn loop's private recovery state, seeded from its parent (the
    enclosing loop's context, or the module globals at the outermost level)."""

    __slots__ = FIELDS

    def __init__(self, seed: "RecoveryContext | None" = None) -> None:
        for f in FIELDS:
            if seed is not None:
                cur = getattr(seed, f)
            else:
                cur = vars(st).get(f)
            # Copy session-scoped containers so the sub-loop SEES the parent's
            # breaker evidence but its writes stay private; the copy-back folds
            # final VALUES (not container identity) into the parent.
            setattr(self, f, cur.copy() if isinstance(cur, (dict, set)) else cur)

    def copy_back_into(self, target: Any) -> None:
        """Fold this loop's final state into `target` (a parent RecoveryContext
        or the module dict at the outermost level)."""
        for f in FIELDS:
            val = getattr(self, f)
            if isinstance(target, dict):
                target[f] = val
            else:
                setattr(target, f, val)


class _Frame:
    """One installed loop: its context plus the frame (or store) beneath it."""

    __slots__ = ("ctx", "prev")

    def __init__(self, ctx: RecoveryContext, prev: "RecoveryContext | None") -> None:
        self.ctx = ctx
        # The ENCLOSING loop's context (None at the outermost level). The seed
        # above reads it the same way, so copy-back returns to the layer the
        # loop actually seeded from.
        self.prev = prev


# Lives in _state (not here) so an importlib.reload of this module — or
# conftest's post-test module eviction — can't strand installed contexts.
_CURRENT: contextvars.ContextVar[_Frame | None] = st._RECOVERY_CURRENT


def current() -> RecoveryContext | None:
    """The calling thread's loop context, or None outside any loop."""
    fr = _CURRENT.get()
    return fr.ctx if fr is not None else None


def install() -> RecoveryContext:
    """Create + install a fresh context for the calling loop. The caller pairs
    this with copy_back_and_reset() in its finally."""
    parent = current()  # the enclosing loop's context, if any
    ctx = RecoveryContext(parent)
    _CURRENT.set(_Frame(ctx, parent))
    return ctx


def copy_back_and_reset(ctx: RecoveryContext) -> None:
    """Fold the context's final values into its parent (or the module globals
    at the outermost level) and pop the frame."""
    fr = _CURRENT.get()
    if fr is None or fr.ctx is not ctx:
        return  # stale finally after a swapped frame — nothing to unwind
    if fr.prev is not None:
        ctx.copy_back_into(fr.prev)
    else:
        ctx.copy_back_into(vars(st))
    _CURRENT.set(None)


def get(field: str) -> Any:
    """Read a recovery field: from the loop context when one is installed,
    else from the module globals (tests, REPL, import-time code)."""
    fr = _CURRENT.get()
    if fr is not None and fr.ctx is not None:
        return getattr(fr.ctx, field)
    return vars(st).get(field)


def set_(field: str, value: Any) -> None:
    """Write a recovery field (see get())."""
    fr = _CURRENT.get()
    if fr is not None and fr.ctx is not None:
        setattr(fr.ctx, field, value)
    else:
        vars(st)[field] = value


# --- descriptor wiring -------------------------------------------------------
# `st._X` must route without touching ~35 call sites: install data descriptors
# named exactly like the fields on the state module's class. Data descriptors
# win over dict entries for attribute access, while direct `__dict__` writes
# (conftest's rewind) still hit the default store underneath.

def _install_descriptors() -> None:
    base = type(st)
    if getattr(base, "_RECOVERY_ROUTED", False):
        return
    ns: dict[str, Any] = {"_RECOVERY_ROUTED": True}
    for f in FIELDS:
        ns[f] = property(
            (lambda _f: lambda self: get(_f))(f),
            (lambda _f: lambda self, v: set_(_f, v))(f),
        )
    st.__class__ = type(base.__name__ + "Routed", (base,), ns)


_install_descriptors()
