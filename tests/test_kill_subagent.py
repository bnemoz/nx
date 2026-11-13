"""kill_subagent (0.171.6): cooperative cancel for background sub-agents.

Threads cannot be killed, so the tool follows the check-at-boundary contract
(nx/_interrupt.py): a per-job cancel event rides the record; the sub-loop
raises _SubagentKilled at the next tool-call boundary and _run_async_agent
finalizes as "killed" with the normal finalize path. A QUEUED job (parked on
the concurrency semaphore) is finalized immediately by kill_subagent itself —
its worker later finds a killed record and exits without running.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in [k for k in os.environ if k.startswith("NX_")]:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.environ["NX_SUBAGENT_ASYNC"] = "0"  # legacy foreground spawn path
    os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    monkeypatch.setattr(nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield nx


def test_unknown_job_id_returns_error_not_exception(nx_fresh):
    from nx._subagent import kill_subagent
    res = kill_subagent("agent-1-doesnotexist")
    assert res.startswith("[error]")
    assert "no such sub-agent job" in res


def test_queued_kill_finalizes_without_worker_running(nx_fresh, monkeypatch):
    """A kill of a QUEUED job finalizes it immediately; the parked worker must
    never run the agent (fake_loop would fail the test if it did)."""
    import nx as m
    from nx._subagent import (_ASYNC_AGENTS, _ASYNC_LOCK, AgentJob,
                              _KILLED_ANSWER, kill_subagent,
                              _stage_subagent_completions)

    def fail_loop(messages, *a, **kw):
        raise AssertionError("worker ran a killed queued job")

    monkeypatch.setattr(m, "_run_model_turn_loop", fail_loop)
    # Park the fan-out: hold the semaphore so the new job stays queued.
    sem = m._subagent._async_semaphore()
    held = []
    for _ in range(12):
        held.append(sem.acquire(blocking=False))
    try:
        job = m._subagent.spawn_agent_job("doomed", 5)
        with _ASYNC_LOCK:
            assert _ASYNC_AGENTS[job.job_id]["status"] == "queued"
        res = kill_subagent(job.job_id)
        assert res.startswith("[killed]")
        with _ASYNC_LOCK:
            rec = _ASYNC_AGENTS[job.job_id]
            assert rec["status"] == "killed"
            assert rec["answer"] == _KILLED_ANSWER
            assert rec["cancel"].is_set()
        # The note reaches the parent's completion sweep exactly once.
        _stage_subagent_completions()
        from nx import _notices
        note = _notices.get(m._subagent._PENDING_NOTICE_KEY)
        assert isinstance(note, str) and "KILLED" in note
        assert _stage_subagent_completions() is None  # idempotent
        assert isinstance(job, AgentJob) and job.done.is_set()
    finally:
        for _ in held:
            sem.release()
    # The parked worker exits without running; give it a beat to be safe.
    t = m._subagent._LIVE_WORKERS.get(job.job_id)
    if t is not None:
        t.join(timeout=5)
        assert not t.is_alive()


def test_running_agent_cancel_honored_at_boundary(nx_fresh, monkeypatch):
    """A running agent's loop aborts with _SubagentKilled at the next
    tool-call boundary once kill_subagent sets its cancel event."""
    from nx._subagent import (_ASYNC_AGENTS, _ASYNC_LOCK, _KILLED_ANSWER,
                              kill_subagent)
    import nx as m

    started = threading.Event()
    release = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        release.wait(10)  # hold the agent "running" until the test kills it
        # kill_subagent has fired: the real boundary check in _turn_toolcalls
        # raises; simulate it by checking the same event the boundary reads
        # (ctx.cancel via agent_cancel_requested).
        from nx._subagent_tls import agent_cancel_requested
        if agent_cancel_requested():
            from nx._subagent_tls import _SubagentKilled
            raise _SubagentKilled()
        messages.append({"role": "assistant", "content": "unused"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    job = m._subagent.spawn_agent_job("sleeper", 5)
    assert started.wait(10), "agent never started"
    res = kill_subagent(job.job_id)
    assert res.startswith("[killing]")
    release.set()
    answer = job.join(timeout=10)
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS.get(job.job_id, {})
    assert rec.get("status") == "killed"
    assert answer == _KILLED_ANSWER
    assert rec.get("exit_state") == "BLOCKED"


def test_bg_kill_routes_subagent_job_id(nx_fresh, monkeypatch, capsys):
    """/bg kill <job_id> routes to kill_subagent; an unknown id that matches
    neither a sub-agent nor a bash PID keeps the old not-tracked message."""
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    from nx._repl_settings_cmds import _cmd_bg
    from nx._repl_state import _ReplState
    from nx import _new_session_id

    with _ASYNC_LOCK:
        _ASYNC_AGENTS["agent-1-fake"] = {
            "instruction": "x", "status": "running", "started": time.time(),
            "cancel": threading.Event(), "handle": None, "announce": False,
        }
    monkeypatch.setattr("nx._subagent._async_running", lambda: True)
    try:
        _cmd_bg("/bg kill agent-1-fake",
                _ReplState([{"role": "user", "content": "hi"}],
                           _new_session_id(), True, []))
        out = capsys.readouterr().out
        assert "cancelling: agent-1-fake" in out or "[killing]" in out
        _cmd_bg("/bg kill agent-9-none",
                _ReplState([{"role": "user", "content": "hi"}],
                           _new_session_id(), True, []))
        out = capsys.readouterr().out
        assert "no such sub-agent job: agent-9-none" in out
    finally:
        with _ASYNC_LOCK:
            _ASYNC_AGENTS.pop("agent-1-fake", None)
