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
descriptors), and the `nx.<name>` proxy in `__init__.py` all keep working
unchanged. The no-loop default store is not IN the module dict, though: it is
an attribute of the descriptor class, because a dict entry is exactly what the
harness rewind clears in place (see `_store()` below) — conftest resets it
through `_reset_store()` instead.

What must NOT move: process-wide config (env-resolved once, read-only at loop
time) — MALFORMED_TOOL_BREAKER_LIMIT and friends stay plain module globals,
exactly the roadmap's "thin facade" carve-out.
"""
from __future__ import annotations

import contextvars
from typing import Any, cast

from . import _state as st

# The fields one loop owns, mapped 1:1 from the old process-wide globals.
FIELDS = (
    "_LAST_PROMPT_TOKENS",           # int: prompt size the loop last saw
    "_TALLY_ARG_REPAIRS",            # int: repairs applied this turn (footer)
    "_TALLY_RETRIES",                # int: recovery nudges/cuts this turn (footer)
    "_STALL_EPISODE_ACTED",          # bool: a tool ran after this stall nudge
    "_STALL_FORCE_NATIVE",           # bool: a stall/empty retry must open with tool_choice="auto"
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
                # getattr routes the descriptor: a value a no-frame set_() put
                # into vars(st) wins; the popped-name fallback is the
                # install-time snapshot in _ROUTED_DEFAULTS.
                cur = getattr(st, f)
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


# The pre-swap values of every routed field, snapshotted at class-install
# time. The routed names are popped out of the module dict (see
# _install_descriptors), so a no-frame fallback can't read them from vars(st)
# anymore; this is where "the module-global default" lives instead. A value
# written while no loop is running goes into vars(st) (set_) and takes
# precedence — the snapshot is only the seed, exactly like the plain module
# global it replaces.
#
# 0.104.1: the store lives on the state module's CLASS, not in a module-dict
# entry (st._RECOVERY_DEFAULTS, still defined for back-compat). A module dict
# entry is exactly what the test harness rewinds: conftest's per-test restore
# walks every nx.* module and does `_cur.clear(); _cur.update(_tmpl)` for dict
# values, which EMPTIED this store between tests while the already-installed
# descriptors kept reading it. The next test's first turn loop then seeded a
# RecoveryContext from get() -> None for all ten fields, and _run_model_turn_
# loop's `cur_tokens > 0` raised TypeError('NoneType' > 'int') (a single-
# process run of the batch/subagent files reproduces it every time). Class
# attributes are invisible to the module-dict rewind and survive both a
# _state reload and an eviction+re-import of this module, so every generation
# of descriptors, install(), and get()/set_() agree on one store.
_COPY_FIELDS = (dict, list, set)


def _copy_default(v: Any) -> Any:
    """A private copy of a container default, so in-place mutation of the
    live store can't corrupt the seed we repair it from."""
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, list):
        return list(v)
    if isinstance(v, set):
        return set(v)
    return v


def _store() -> dict[str, Any]:
    """The no-loop default store (see the note above) — an attribute of the
    state module's descriptor class, created by _install_descriptors()."""
    store = getattr(type(st), "_RECOVERY_STORE", None)
    if store is None:
        # The state module was re-executed (importlib.reload(nx._state)): its
        # fresh class is a plain module again, so re-install our class before
        # writing a store — an immutable `module` type rejects the write.
        _install_descriptors()
        store = getattr(type(st), "_RECOVERY_STORE", None)
    if store is None:  # pragma: no cover - only if the swap itself failed
        store = {}
    return store


def _reset_store() -> None:
    """Test-harness hook: rewind the no-loop store to the install-time seed.

    The store lives on the descriptor class SPECIFICALLY so a module-dict
    rewind can't reach it (see the note above) — which also means conftest's
    own per-test rewind no longer covers it for free. conftest calls this from
    _restore() so the isolation contract is unchanged: a test that leaves
    session-scoped recovery evidence in the global store (a breaker streak, a
    stale prompt-token count) still can't leak it into its neighbours.
    """
    store = getattr(type(st), "_RECOVERY_STORE", None)
    seed = getattr(type(st), "_RECOVERY_SEED", None)
    if store is None or seed is None:
        return  # descriptors not installed on this process yet
    store.clear()
    for _k, _v in seed.items():
        store[_k] = _copy_default(_v)


def _seed() -> dict[str, Any]:
    """The install-time values of every routed field: the truth a read falls
    back to when the store lost the key (harness rewind, a _state reload, or a
    descriptor generation that installed against an empty dict entry)."""
    seed = getattr(type(st), "_RECOVERY_SEED", None)
    if seed is None:
        _store()  # installs the class (and both containers) as a side effect
        seed = getattr(type(st), "_RECOVERY_SEED", None)
    if seed is None:  # pragma: no cover
        seed = {}
    return seed


def _default_of(field: str) -> Any:
    """The value a routed field reads when neither a loop context nor a
    vars(st) entry has one. Never None for a known field: an emptied store is
    repaired from the install-time seed instead, because a routed read that
    returns None for an int/bool/str field crashes its caller."""
    store = _store()
    if field in store:
        return store[field]
    seed = _seed()
    if field in seed:
        val = _copy_default(seed[field])
        store[field] = val
        return val
    return None


def current() -> RecoveryContext | None:
    """The calling thread's loop context, or None outside any loop."""
    fr = _CURRENT.get()
    return fr.ctx if fr is not None else None


def install() -> RecoveryContext:
    """Create + install a fresh context for the calling loop. The caller pairs
    this with copy_back_and_reset() in its finally."""
    # Re-pop the routed names on every install: conftest's between-test module
    # rewind restores the module dicts to a pre-_recovery snapshot, which puts
    # the plain defaults back into vars(st) — exactly the dict entries the
    # LOAD_ATTR_MODULE specialization hazard needs to serve stale values
    # (see _install_descriptors). Popping here keeps every in-loop access on
    # the descriptor path, whatever the test harness did to the dict in
    # between. A value the rewind restored is folded into _ROUTED_DEFAULTS
    # first so it isn't lost.
    store, seed = _store(), _seed()
    for f in FIELDS:
        if f in st.__dict__:
            _val = st.__dict__.pop(f)
            store[f] = _val
            # A value the rewind restored is the best default we have seen;
            # record it as the repair seed too (first write wins — the store
            # itself carries later runtime values).
            seed.setdefault(f, _copy_default(_val))
        elif f not in store:
            # Nothing in the dict and nothing in the store (a fresh descriptor
            # generation after a reload that lost the entry): repair from the
            # seed so the context below never seeds None.
            _default_of(f)
    parent = current()  # the enclosing loop's context, if any
    ctx = RecoveryContext(parent)
    _CURRENT.set(_Frame(ctx, parent))
    return ctx


def copy_back_and_reset(ctx: RecoveryContext) -> None:
    """Fold the context's final values into its parent (or the module globals
    at the outermost level) and pop the frame.

    The outermost fold deliberately writes through _recovery.set_ semantics
    rather than raw vars(st) assignment, and keeps the routed names OUT of the
    module dict (see _install_descriptors): a plain dict entry re-arms the
    CPython 3.12+ LOAD_ATTR_MODULE specialization hazard where a hot read site
    returns the dict value while descriptor-routed writes go to the context."""
    fr = _CURRENT.get()
    if fr is None or fr.ctx is not ctx:
        return  # stale finally after a swapped frame — nothing to unwind
    if fr.prev is not None:
        ctx.copy_back_into(fr.prev)
    else:
        for f in FIELDS:
            # Seed the next no-frame read without leaving a dict entry the
            # specialized module-attr path could latch onto: the snapshot is
            # the store; a test's own st.X = v write still lands in vars(st)
            # and (correctly) wins while present.
            _store()[f] = getattr(ctx, f)
            vars(st).pop(f, None)
    _CURRENT.set(None)


def get(field: str) -> Any:
    """Read a recovery field: from the loop context when one is installed,
    else from the module globals (tests, REPL, import-time code)."""
    fr = _CURRENT.get()
    if fr is not None and fr.ctx is not None:
        return getattr(fr.ctx, field)
    if field in vars(st):
        return vars(st)[field]
    return _default_of(field)


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
#
# 0.53.3: the dict entries are ALSO dropped after the swap. CPython 3.12+
# specializes LOAD_ATTR_MODULE for plain module attribute reads — a specialized
# site reads the module __dict__ directly and BYPASSES data descriptors on a
# swapped module class (stores have no such specialization, so they still hit
# the descriptor's __set__). A routed name left in the dict therefore made hot
# read sites (loop code) return the stale dict value while writes landed in the
# recovery context — st._LAST_PARTIAL_TEXT read '' back right after a turn wrote
# 'chunk 2 ' into the context. Deleting the dict entry forces every access
# (specialized or not) through the descriptor; the descriptor's fallback reads
# the same dict the old default lived in, so a no-frame access still sees the
# module-global value, and conftest's direct `__dict__` rewind keeps working
def _install_descriptors() -> None:
    base = type(st)
    if getattr(base, "_RECOVERY_ROUTED", False):
        return
    # Inherit the store/seed when an earlier generation installed them (None
    # checks, not `or {}` — an emptied dict must keep its identity, or the
    # generations would silently fork into separate stores).
    store = cast("dict[str, Any] | None", getattr(base, "_RECOVERY_STORE", None))
    seed = cast("dict[str, Any] | None", getattr(base, "_RECOVERY_SEED", None))
    if store is None:
        store = {}
    if seed is None:
        seed = {}
    ns: dict[str, Any] = {
        "_RECOVERY_ROUTED": True,
        # The no-loop default store + the install-time seed. They live on the
        # class — NOT in a module dict — so conftest's between-test module
        # rewind (which clears dict values in place) can't empty the store out
        # from under the descriptors, and so a later descriptor generation
        # inherits the same two objects instead of forking its own store.
        "_RECOVERY_STORE": store,
        "_RECOVERY_SEED": seed,
    }
    for f in FIELDS:
        ns[f] = property(
            (lambda _f: lambda self: get(_f))(f),
            (lambda _f: lambda self, v: set_(_f, v))(f),
        )
    st.__class__ = type(base.__name__ + "Routed", (base,), ns)  # pyrefly: ignore[missing-attribute]  # deliberate dynamic class swap
    for f in FIELDS:
        if f in st.__dict__:
            _val = st.__dict__.pop(f)
            store[f] = _val
            seed.setdefault(f, _copy_default(_val))


_install_descriptors()
