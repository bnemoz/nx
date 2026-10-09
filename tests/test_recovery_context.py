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


# --- default-store durability (0.104.1) ------------------------------------------
# The store the descriptors fall back to must survive what the test harness does
# to module state between tests. It used to be the st._RECOVERY_DEFAULTS *dict
# entry*, which conftest's per-test rewind clears in place — so the FIRST turn
# loop of the next test on the same worker seeded its RecoveryContext from
# get() -> None for all ten fields, and `cur_tokens > 0` in _turn.py crashed with
# TypeError('>' not supported between 'NoneType' and 'int'). A single-process run
# of tests/test_batch_run.py + tests/test_subagent_*.py reproduced it every time.

def test_default_store_is_not_a_module_dict_entry():
    """Not reachable by the harness's module-dict rewind (the leak that made
    the suite order-dependent under -n auto)."""
    assert "_RECOVERY_STORE" not in vars(m.st), \
        "the default store must live on the descriptor class, not in the dict"
    assert type(m.st).__dict__["_RECOVERY_STORE"] is not None
    assert type(m.st).__dict__["_RECOVERY_SEED"] is not None


def test_emptied_store_is_repaired_from_the_install_time_seed(monkeypatch):
    """Whatever empties the store, a routed read falls back to the seeded
    default — never None for a field the callers treat as int/bool/str."""
    _quiet(monkeypatch)
    store = _recovery._store()
    saved = dict(store)
    saved_dict = {k: vars(m.st)[k] for k in _recovery.FIELDS if k in vars(m.st)}
    try:
        # Emulate the harness rewind of a dict-entry store + a fresh descriptor
        # generation that installed against it.
        store.clear()
        for k in _recovery.FIELDS:
            vars(m.st).pop(k, None)
        assert m.st._LAST_PROMPT_TOKENS == 0
        assert m.st._TALLY_RETRIES == 0
        assert m.st._LAST_PARTIAL_TEXT == ""
        assert m.st._MALFORMED_TOOL_FAILURES == {}
        assert m.st._MALFORMED_TOOL_BROKEN == set()
        assert m.st._STALL_EPISODE_ACTED is False
    finally:
        store.clear()
        store.update(saved)
        vars(m.st).update(saved_dict)


def test_reset_store_rewinds_to_the_seed(monkeypatch):
    """conftest's _restore() calls this every test, so session-scoped recovery
    evidence left in the global store can't leak into the next test.

    `_reset_store()` covers the class-anchored store only; the vars(st) half of
    the no-frame default is covered by conftest's own dict rewind (that's the
    pairing the harness relies on)."""
    _quiet(monkeypatch)
    store = _recovery._store()
    saved = dict(store)
    try:
        store["_TALLY_RETRIES"] = 7
        vars(m.st).pop("_TALLY_RETRIES", None)
        assert m.st._TALLY_RETRIES == 7
        _recovery._reset_store()
        assert m.st._TALLY_RETRIES == 0
    finally:
        store.clear()
        store.update(saved)
        vars(m.st).pop("_TALLY_RETRIES", None)

