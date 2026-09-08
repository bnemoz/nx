#!/usr/bin/env python3
"""RecoveryContext thread-isolation (0.42, roadmap 1.2).

The self-healing stack used to run on process-wide globals shared with
sub-agent threads: an async sub-agent running its own _run_model_turn_loop on
a worker thread wrote the SAME st.* the parent read, so a sub-agent's tally or
malformed-args breaker was the parent's. Now each loop owns a private context
(resolved per thread via a contextvar), and its final values fold back only
into ITS parent — a worker thread's copy-back lands in the module globals
while the parent loop's private context is untouched.
"""
import io
import os
import sys
import tempfile
import threading

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _recovery  # noqa: E402


def _quiet(monkeypatch):
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def _install_source():
    from nx import _sources as _sources_mod
    src = _sources_mod.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    m.st.SOURCES["fake"] = src
    m.st.SOURCE_ORDER.append("fake")
    m.st.ACTIVE = src
    return src


def test_agent_thread_writes_never_reach_the_parent_loop(monkeypatch):
    """The roadmap's headline hazard: a sub-agent's in-loop writes (tallies,
    stall marker, breaker) must not be visible to the parent's loop state."""
    _install_source()
    _quiet(monkeypatch)
    parent = _recovery.install()
    try:
        parent._TALLY_RETRIES = 0
        parent._STALL_EPISODE_ACTED = False
        parent._MALFORMED_TOOL_BROKEN = set()

        def agent_turn(messages, *a, **k):
            # The agent thread has NO parent context: its writes land in its
            # own loop context, seeded from the module globals.
            m.st._TALLY_RETRIES += 3
            m.st._STALL_EPISODE_ACTED = True
            m.st._MALFORMED_TOOL_BROKEN.add("multi_edit")
            return m.TURN_DONE

        monkeypatch.setattr(m, "model_turn", agent_turn)
        t = threading.Thread(
            target=m._run_model_turn_loop,
            args=([{"role": "user", "content": "agent task"}],),
            kwargs={"max_turns_override": 3})
        t.start()
        t.join(10)
        assert not t.is_alive(), "agent loop did not finish"

        # The parent's private state is untouched by the agent thread.
        assert parent._TALLY_RETRIES == 0, "agent's retries leaked into parent"
        assert parent._STALL_EPISODE_ACTED is False, "agent's stall flag leaked"
        assert parent._MALFORMED_TOOL_BROKEN == set(), "agent's breaker leaked"
        # And reads on the main thread still resolve to the parent context.
        assert m.st._TALLY_RETRIES == 0
        assert m.st._STALL_EPISODE_ACTED is False
        assert m.st._MALFORMED_TOOL_BROKEN == set()
    finally:
        _recovery.copy_back_and_reset(parent)


def test_agent_thread_session_evidence_lands_in_globals(monkeypatch):
    """copy_back folds the agent's final state into the module globals (its
    outermost parent) so session-scoped evidence still survives its thread."""
    _install_source()
    _quiet(monkeypatch)
    saved_confirmed = m.st._NATIVE_TOOL_CALLS_CONFIRMED
    try:
        m.st._NATIVE_TOOL_CALLS_CONFIRMED = False

        def agent_turn(messages, *a, **k):
            m.st._NATIVE_TOOL_CALLS_CONFIRMED = True
            return m.TURN_DONE

        monkeypatch.setattr(m, "model_turn", agent_turn)
        t = threading.Thread(
            target=m._run_model_turn_loop,
            args=([{"role": "user", "content": "probe native"}],),
            kwargs={"max_turns_override": 3})
        t.start()
        t.join(10)
        assert not t.is_alive()
        # Main thread: no loop context -> reads the globals, which the agent's
        # copy-back updated. (A native-tool confirmation IS session evidence.)
        assert m.st._NATIVE_TOOL_CALLS_CONFIRMED is True
    finally:
        m.st._NATIVE_TOOL_CALLS_CONFIRMED = saved_confirmed


def test_nested_sync_subloop_folds_back_into_parent(monkeypatch):
    """Sync sub-agents run the sub-loop on the parent's thread: the inner
    loop must seed from and copy back into the OUTER loop's context, not the
    module globals."""
    _quiet(monkeypatch)
    parent = _recovery.install()
    try:
        inner_seen = {}

        def inner_loop():
            ctx = _recovery.current()
            inner_seen["seeded_from_parent"] = ctx is not None
            if ctx is not None:
                ctx._TALLY_RETRIES = 7
            return {"status": "done", "turns": 1}

        monkeypatch.setattr(m._turn, "_run_model_turn_loop_inner",
                            lambda *a, **k: inner_loop())
        summary = m._run_model_turn_loop([{"role": "user", "content": "x"}])
        assert summary["status"] == "done"
        assert inner_seen["seeded_from_parent"], \
            "nested loop must resolve the enclosing loop's context"
        assert parent._TALLY_RETRIES == 7, \
            "inner loop's final state must fold back into the parent context"
        # (the outer loop's own finally then folds it into the globals)
    finally:
        _recovery.copy_back_and_reset(parent)


def test_globals_are_the_default_store_outside_any_loop(monkeypatch):
    _quiet(monkeypatch)
    assert _recovery.current() is None
    m.st._TALLY_RETRIES = 4
    try:
        assert vars(m.st)["_TALLY_RETRIES"] == 4, \
            "outside a loop, writes are plain module globals (conftest contract)"
        assert m.st._TALLY_RETRIES == 4
    finally:
        m.st._TALLY_RETRIES = 0
