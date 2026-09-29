"""Off-critical-path session persistence (0.103.4): the async save writer.

After EVERY model turn the turn loop calls ``on_step()`` ->
``_repl_state._save_current()`` -> ``_sessions_fs._write_session()``, which is
expensive work inline on the agent's thread: a store-form copy of the whole
transcript, a SQLite read-merge-write of the entire blob, and a post-write
re-read. ``_SessionSaveWriter`` moves all of that onto one daemon thread so
``on_step()`` is a snapshot plus a queue hand-off and returns immediately —
the same structural trade ``_AsyncLogWriter`` (``nx/_logging.py``) makes for
the traffic log: caller never waits on disk I/O.

Coalescing, not a queue of writes: at most ONE snapshot per session id is
pending at a time — ``submit`` REPLACES a payload the writer has not picked up
yet, so the newest state always wins and a long turn's intermediate snapshots
collapse into a single write. (Per id, not one global slot: a save for session
B must never displace a pending save for session A.) Each snapshot is a full
transcript, not a delta, which is what makes replacing one safe.
``flush_pending()`` blocks until the in-flight write finishes and every queued
snapshot has landed.

Crash-safety window (the deliberate trade): the pre-0.103.4 behavior was
"every turn is durable the moment the turn ends", so a ``SIGKILL`` could lose
at most in-flight tokens. Now a sudden kill can additionally lose the last
snapshot — normally sub-second of wall time, up to the length of one SQLite
write. Every path that could observe the store closes that window first:
``atexit``, ``/quit``, ``/fork``, ``/resume``, ``/save``, ``/compress``,
``/delete``, and the read side (``_load_session`` / ``_list_sessions`` /
``_write_session`` / ``_delete_session`` all flush first). ``NX_ASYNC_SAVE=0``
restores the exact old synchronous behavior.

Test isolation: the singleton is created lazily inside ``submit_save()`` and
is *never* built at import time, so ``tests/conftest.py``'s per-test rewind of
every ``nx.*`` module dict (which restores the module-global ``_writer`` back
to its baseline ``None``) can't hand a live writer — or its queued snapshot —
to the next test. The process-wide atexit hook re-reads the global at call
time, so a writer a test created and then had rewound away is a no-op flush.

Only the writer thread performs a save; ``_sessiondb`` opens a fresh SQLite
connection per operation and serializes writes with ``BEGIN IMMEDIATE``, so no
extra lock is needed for the store itself. The module-global writer pointer is
guarded by ``_WRITER_LOCK`` so two threads can't create two writers.
"""
from __future__ import annotations

import atexit
import threading
from typing import Any, Callable

from ._env import _env_int
from ._sessions_fs import begin_exit as _begin_exit

# Kill switch: NX_ASYNC_SAVE=0 -> every save writes inline, exactly the
# pre-0.103.4 behavior (read at call time, like every other NX_* knob).
ASYNC_SAVE_DEFAULT = 1


def async_save_enabled() -> bool:
    """Whether saves may go to the writer thread. Read per call so a test (or
    a swarm harness) can flip NX_ASYNC_SAVE without re-importing."""
    return _env_int("NX_ASYNC_SAVE", ASYNC_SAVE_DEFAULT) != 0


class _SessionSaveWriter:
    """Single daemon writer thread holding at most one pending snapshot PER
    SESSION ID.

    `write` is an injected callable (``(session_id, messages, meta,
    update_index) -> None``) rather than a direct import of
    ``_sessions_fs._write_session_snapshot``: it keeps this module importable
    from anywhere (no import cycle with the sessions modules) and lets tests
    drive the writer with a recording stub.
    """

    def __init__(self, write: Callable[..., None]) -> None:
        self._write = write
        self._cond = threading.Condition()
        # Coalescing slots, keyed by session id: {sid: ((args), kwargs)}.
        # Keyed rather than one global slot because a snapshot for session B
        # must never DISPLACE a pending snapshot for session A (that would
        # drop a save outright); per id the newest always wins, which is
        # exactly the depth-1 coalescing the design asks for. In practice one
        # REPL has one active id, so this dict holds a single entry.
        self._pending: dict[Any, tuple[tuple, dict]] = {}
        self._busy = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="nx-session-save")
        self._thread.start()

    def submit(self, key: Any, args: tuple, kwargs: dict) -> None:
        """Hand the newest snapshot to the writer thread and return.

        Coalescing rule: a snapshot for the same `key` that the writer has not
        picked up yet is REPLACED (the newest transcript always wins), so the
        per-id queue depth stays at one and N autosaves in a burst collapse
        into one SQLite write. A snapshot already handed to the thread still
        lands first — the merged store is a prefix join, so that extra write
        costs one revision but never corrupts the transcript.
        """
        with self._cond:
            self._pending[key] = (args, kwargs)
            self._cond.notify_all()

    def _in_writer_thread(self) -> bool:
        """True when called from the writer thread itself. Compares against
        `self._thread.ident` (valid once started), NOT an ident captured in
        `__init__` — that one is the *constructing* thread's, which would make
        every flush from the main thread a silent no-op."""
        return threading.get_ident() == self._thread.ident

    def flush_pending(self) -> None:
        """Block until the pending snapshot has been written (and any write in
        flight has finished). Cheap no-op when nothing is outstanding, which
        is the common case — this is what the read paths call."""
        if self._in_writer_thread():
            return  # the writer's own store write re-reads the session: a
            # flush from there would wait on itself and deadlock.
        with self._cond:
            if not self._pending and not self._busy:
                return
            self._cond.notify_all()
            while self._pending or self._busy:
                self._cond.wait()

    def has_pending(self) -> bool:
        if self._in_writer_thread():
            return False
        with self._cond:
            return bool(self._pending) or self._busy

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                # FIFO across ids (dict order is insertion order), newest
                # content per id.
                key = next(iter(self._pending))
                item = self._pending.pop(key)
                self._busy = True
            try:
                args, kwargs = item
                self._write(*args, **kwargs)
            except Exception:
                # A failed save must never kill the writer thread: the next
                # save (or an explicit flush) has to still work. `_write_*`
                # already swallows its own errors; this is the backstop.
                pass
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()


def snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A snapshot of a live transcript that stays stable while the turn loop
    keeps appending.

    The transcript grows in place (`state.messages.append(...)`) and the turn
    loop mutates image parts in place
    (`_images._strip_image_parts` / `_strip_consumed_images` do
    `part.clear()` on the dicts inside a *list* content — see their
    docstrings, which rely on those very references). So the copy has to be:

    * a NEW list, plus a new dict per message — an append during the write can
      never shift or resize what the writer is serializing (which would raise
      "list changed size during iteration");
    * a new list with copied parts for any list-valued `content` — that is the
      only in-place mutation an image part sees, and only for the (rare)
      messages that actually carry image parts.

    Everything else is shared by reference, which is why this stays cheap: a
    message's string content and its `tool_calls` list are treated as
    immutable everywhere on the turn path, so no copy is needed. Costs one
    dict per message plus a pointer copy — orders of magnitude below the
    SQLite read-merge-write it replaces.
    """
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, list):
            d = dict(m)
            d["content"] = [dict(p) if isinstance(p, dict) else p for p in c]
            out.append(d)
        else:
            out.append(dict(m))
    return out


_writer: _SessionSaveWriter | None = None
_WRITER_LOCK = threading.Lock()


def _default_write(session_id: str, messages: list[dict[str, Any]],
                   meta: dict[str, Any] | None, update_index: bool) -> None:
    """Late-bound `_sessions_fs._write_session_snapshot` lookup: resolving the
    name at call time (rather than capturing it when the writer is built)
    keeps a test's monkeypatch of that function effective for an
    already-created writer."""
    from . import _sessions_fs
    return _sessions_fs._write_session_snapshot(
        session_id, messages, meta, update_index=update_index)


def _get_writer() -> _SessionSaveWriter:
    """Lazily start the writer thread on first use (never at import: see the
    module docstring's test-isolation note)."""
    global _writer
    w = _writer
    if w is not None:
        return w
    with _WRITER_LOCK:
        if _writer is None:
            _writer = _SessionSaveWriter(_default_write)
        return _writer


def submit_save(session_id: str, messages: list[dict[str, Any]],
                meta: dict[str, Any] | None, update_index: bool) -> None:
    """Enqueue a session save on the writer thread and return immediately.

    The session id doubles as the coalescing key, so a burst of autosaves for
    one session collapses into a single write while a save for a different
    session is never dropped."""
    _get_writer().submit(session_id, (session_id, messages, meta),
                         {"update_index": update_index})


def flush_pending() -> None:
    """Synchronously drain any pending save. No-op (never starts a thread)
    when nothing is queued — safe to call from any read path."""
    w = _writer
    if w is not None:
        w.flush_pending()


def has_pending() -> bool:
    """True while a snapshot is queued or being written."""
    w = _writer
    return bool(w is not None and w.has_pending())


def _flush_at_exit() -> None:
    """atexit hook: a normal shutdown never loses the last turn. Re-reads the
    module global so a writer rewound away by the test harness flushes as a
    no-op. Puts the save path in its cheap exit form first (0.103.28): the
    final write skips the post-write re-read and the GC sweeps, which the
    user is not waiting on at the moment they pressed Ctrl+Q."""
    try:
        _begin_exit()
    except Exception:
        pass
    w = _writer
    if w is not None:
        try:
            w.flush_pending()
        except Exception:
            pass


atexit.register(_flush_at_exit)
