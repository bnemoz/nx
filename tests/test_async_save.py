#!/usr/bin/env python3
"""Tests for `_SessionSaveWriter` — the off-critical-path session saver.

The turn loop calls `on_step()` after EVERY model step, and that used to pay a
full-transcript store-form copy + a SQLite read-merge-write + a post-write
re-read inline. Now it must hand a snapshot to the writer thread and return,
so these pin the four properties that make that safe:

  1. a save is QUEUED, not written — the store row is untouched right after
     `on_step()` returns (that's the whole point: no disk I/O on the turn
     thread);
  2. `flush_pending()` delivers it;
  3. coalescing keeps the NEWEST snapshot (the latest state always wins) and
     collapses a burst instead of queueing N writes;
  4. `NX_ASYNC_SAVE=0` restores the exact old synchronous behavior;
  5. the read paths flush first, so a reader never sees a stale transcript.
"""
import os
import sys
import tempfile
import threading

_tmp = tempfile.mkdtemp(prefix="nx-async-save-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _session_save, _sessiondb  # noqa: E402
from nx._repl_state import _ReplState, _flush_save, _save_current  # noqa: E402
from nx._sessions_fs import _load_session, _write_session_snapshot  # noqa: E402


class _Gate:
    """A writer `write` hook that parks inside the first write until released,
    so 'has the save landed yet?' is answered deterministically instead of by
    sleeping. `entered` fires once the writer thread is inside `write`, which
    is the moment later submits start coalescing."""

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, session_id, messages, meta, update_index=True):
        self.entered.set()
        assert self.release.wait(timeout=5), "writer never released"
        self.calls.append((session_id, [dict(x) for x in messages],
                           dict(meta or {})))
        _write_session_snapshot(session_id, messages, meta,
                               update_index=update_index)

    def install(self, monkeypatch):
        monkeypatch.setattr(_session_save, "_writer",
                            _session_save._SessionSaveWriter(self.write))
        return self


def _state(sid="s-async-save", n=1):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return _ReplState(msgs, sid, True, [])


def _row(sid):
    """The store's own read — deliberately WITHOUT the flush gate that
    `_load_session` applies, so a pending save stays pending."""
    return _sessiondb.load_session(sid)


# --- 1. the save is enqueued, not written ----------------------------------

def test_save_is_queued_and_does_not_touch_the_store(monkeypatch):
    gate = _Gate().install(monkeypatch)
    state = _state("s-queued")
    _save_current(state, update_index=False)
    # The writer thread is parked inside the write; nothing has landed yet.
    assert gate.entered.wait(timeout=5)
    assert _row("s-queued") is None
    assert _session_save.has_pending()
    gate.release.set()
    _session_save.flush_pending()
    assert len(gate.calls) == 1
    data = _row("s-queued")
    assert data is not None
    assert [msg["content"] for msg in data["messages"]] == [
        "sys", "question 0", "answer 0"]


def test_on_step_snapshot_survives_the_turn_loop_appending(monkeypatch):
    # The whole reason for the snapshot: `state.messages` keeps growing and
    # image parts get mutated in place while the writer is serializing.
    gate = _Gate().install(monkeypatch)
    state = _state("s-snapshot", n=1)
    live = state.messages
    _save_current(state, update_index=False)
    assert gate.entered.wait(timeout=5)
    # mutate the live transcript exactly like the turn loop does after a step
    live.append({"role": "user", "content": "appended after the snapshot"})
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    live.append({"role": "tool", "content": [part]})
    part.clear()          # what _images._strip_consumed_images does in place
    part["type"] = "text"
    part["text"] = "[image payload dropped after use]"
    gate.release.set()
    _session_save.flush_pending()
    snapshot = gate.calls[0][1]
    assert len(snapshot) == 3                      # the pre-append transcript
    assert all("appended after" not in str(x) for x in snapshot)


# --- 2/3. flush delivers, coalescing keeps the newest ----------------------

def test_coalescing_keeps_the_newest_snapshot(monkeypatch):
    gate = _Gate().install(monkeypatch)
    # First save: the writer picks it up and parks inside the write.
    _save_current(_state("s-coalesce", n=1), update_index=False)
    assert gate.entered.wait(timeout=5)
    # Everything submitted while that write is in flight collapses into ONE
    # pending snapshot — the newest one wins.
    for n in (2, 3, 4):
        _save_current(_state("s-coalesce", n=n), update_index=False)
    gate.release.set()
    _session_save.flush_pending()
    assert len(gate.calls) == 2                    # 4 saves -> 2 writes
    final = gate.calls[-1][1]
    assert len(final) == 1 + 2 * 4                 # the n=4 transcript
    assert final[-1]["content"] == "answer 3"
    data = _row("s-coalesce")
    assert data["messages"][-1]["content"] == "answer 3"


def test_flush_delivers_every_queued_session(monkeypatch):
    gate = _Gate().install(monkeypatch)
    gate.release.set()                             # no parking, just record
    for sid in ("s-a", "s-b", "s-c"):
        _save_current(_state(sid), update_index=False)
    _flush_save()
    assert not _session_save.has_pending()
    # Coalescing is PER SESSION ID: three different ids are three writes (a
    # save for B must never displace a pending save for A).
    assert len(gate.calls) == 3
    for sid in ("s-a", "s-b", "s-c"):
        assert _row(sid)["messages"][-1]["content"] == "answer 0"


def test_flush_pending_is_a_noop_without_a_writer(monkeypatch):
    monkeypatch.setattr(_session_save, "_writer", None)
    _session_save.flush_pending()                  # must not raise
    _flush_save()
    assert _session_save._writer is None           # and must not start one
    assert not _session_save.has_pending()


# --- 4. the kill switch restores the synchronous path ---------------------

def test_kill_switch_saves_inline(monkeypatch):
    gate = _Gate().install(monkeypatch)
    monkeypatch.setenv("NX_ASYNC_SAVE", "0")
    _save_current(_state("s-sync"), update_index=False)
    # No writer involvement at all — the save landed before we returned.
    assert gate.calls == [] and not gate.entered.is_set()
    assert not _session_save.has_pending()
    data = _row("s-sync")
    assert data is not None and data["messages"][-1]["content"] == "answer 0"


def test_async_is_the_default(monkeypatch):
    monkeypatch.delenv("NX_ASYNC_SAVE", raising=False)
    assert _session_save.async_save_enabled()
    monkeypatch.setenv("NX_ASYNC_SAVE", "0")
    assert not _session_save.async_save_enabled()
    monkeypatch.setenv("NX_ASYNC_SAVE", "1")
    assert _session_save.async_save_enabled()


# --- 5. the read side flushes first ---------------------------------------

def test_load_session_flushes_a_pending_save(monkeypatch):
    gate = _Gate().install(monkeypatch)
    _save_current(_state("s-read"), update_index=False)
    assert gate.entered.wait(timeout=5)
    assert _row("s-read") is None                  # still pending
    gate.release.set()
    data = _load_session("s-read")                 # the gated reader
    assert data is not None and data["messages"][-1]["content"] == "answer 0"
    assert not _session_save.has_pending()


def test_inline_write_session_drains_the_queue_first(monkeypatch):
    gate = _Gate().install(monkeypatch)
    _save_current(_state("s-order", n=1), update_index=False)
    assert gate.entered.wait(timeout=5)
    gate.release.set()
    # An inline save must not slip ahead of the queued snapshot.
    _write_session_call("s-order", _state("s-order", n=2).messages)
    assert _row("s-order")["messages"][-1]["content"] == "answer 1"


def _write_session_call(sid, messages):
    from nx._sessions_fs import _write_session
    _write_session(sid, messages, {"title": "inline"}, update_index=False)


def test_list_sessions_flushes(monkeypatch):
    gate = _Gate().install(monkeypatch)
    sid = "s-listing"
    _save_current(_state(sid), update_index=False)
    assert gate.entered.wait(timeout=5)
    gate.release.set()
    assert any(s["id"] == sid for s in m._list_sessions(limit=50))


# --- the snapshot helper itself -------------------------------------------

def test_snapshot_messages_is_independent_but_cheap():
    part = {"type": "image_url", "image_url": {"url": "data:,AA"}}
    msgs = [{"role": "user", "content": [part]}, {"role": "assistant",
            "content": "text"}]
    snap = _session_save.snapshot_messages(msgs)
    assert snap is not msgs
    assert snap[0] is not msgs[0]
    assert snap[0]["content"] is not msgs[0]["content"]
    assert snap[0]["content"][0] is not part
    part.clear()                                   # the in-place mutation
    assert snap[0]["content"][0].get("type") == "image_url"
    # a plain string-content message shares nothing mutable either
    assert snap[1] is not msgs[1]
    assert snap[1]["content"] == "text"


def test_writer_is_recreatable_per_test():
    # conftest rewinds nx._session_save._writer to None between tests, so the
    # singleton must never carry state across them.
    assert _session_save._writer is None
    w = _session_save._get_writer()
    assert isinstance(w, _session_save._SessionSaveWriter)
    assert _session_save._get_writer() is w
    w.flush_pending()                              # nothing queued: no hang


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
