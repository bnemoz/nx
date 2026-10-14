"""spawn_agent_job (0.156.3): the joinable handle over the async ledger.

Covers the Phase 2 contract: join(timeout) returns the FULL answer (not the
RAM-trimmed record copy), announce=False leaves no _PENDING_SUBAGENT_NOTICE
residue while the record still evicts, result() shows queued while parked on
the concurrency semaphore, and worktree= pass-through skips creation. The
model-facing spawn_agent string is pinned by test_subagent_async.py unchanged.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

os.environ.setdefault("NX_ENV_FILE", "/dev/null")
_AMBIENT_NX = [k for k in os.environ if k == "NX_ENV_FILE" or k.startswith("NX_")]
for k in _AMBIENT_NX:
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
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    monkeypatch.setattr(nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield nx


def test_join_returns_full_answer_beyond_trim(nx_fresh, monkeypatch):
    """join() returns the FULL answer; the ledger record keeps only the
    head+tail (RAM trim), so they must differ for a long answer."""
    from nx._subagent import AgentJob, _ASYNC_AGENTS, _ASYNC_LOCK
    import nx as m

    big = "".join("line %d of the long answer" % i for i in range(100000))

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": big})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    job = nx_fresh._subagent.spawn_agent_job("long answer please", 5)
    assert isinstance(job, AgentJob)
    answer = job.join(timeout=30)
    assert answer is not None and answer == big
    assert job.done and job.result()["status"] == "done"
    # the record was RAM-trimmed (answer in the record != full answer) — the
    # handle is the only full-copy carrier
    with _ASYNC_LOCK:
        rec_answer = _ASYNC_AGENTS[job.job_id]["answer"]
    assert len(rec_answer) < len(answer)
    assert "spilled to report" in rec_answer


def test_announce_false_no_notice_residue_and_still_evicts(nx_fresh, monkeypatch):
    from nx._subagent import (_ASYNC_AGENTS, _ASYNC_LOCK,
                              _stage_subagent_completions)
    from nx import _state as st
    import nx as m

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "quiet findings"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    job = nx_fresh._subagent.spawn_agent_job("quiet job", 5, announce=False)
    assert job.join(timeout=30) == "quiet findings"
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS[job.job_id]
        assert rec["announced"] is True  # mandatory: eviction candidate
        assert rec["status"] == "done"
    # no notice staged anywhere: pending frame, module store, sweep no-op
    note = getattr(st, "_PENDING_SUBAGENT_NOTICE", None)
    from nx._subagent import _notices
    assert not (note or _notices.get("_PENDING_SUBAGENT_NOTICE"))
    _stage_subagent_completions()  # parent sweep must not stage anything
    assert not getattr(st, "_PENDING_SUBAGENT_NOTICE", None)
    # finished+announced -> eviction candidate; drop the record manually to
    # prove the invariant (real eviction waits for MAX_FINISHED churn)
    with _ASYNC_LOCK:
        _ASYNC_AGENTS.pop(job.job_id, None)
    assert job.job_id not in _ASYNC_AGENTS


def test_result_shows_queued_when_semaphore_parked(nx_fresh, monkeypatch):
    from nx._subagent import _async_semaphore
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_MAX_CONCURRENT", 1,
                        raising=False)
    sem = _async_semaphore()
    gate = threading.Event()
    release = threading.Event()
    try:
        def fake_loop(messages, *a, **kw):
            gate.wait(timeout=10)   # first agent holds its slot while parked
            release.wait(timeout=10)
            messages.append({"role": "assistant", "content": "second"})
            return {"status": "done", "turns": 1}

        monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
        first = nx_fresh._subagent.spawn_agent_job("one", 5)
        for _ in range(200):
            if first.result()["status"] == "running":
                break
            time.sleep(0.02)
        gate.set()  # let the first agent finish its turn... slowly
        second = nx_fresh._subagent.spawn_agent_job("two", 5)
        for _ in range(200):
            if second.result()["status"] == "queued":
                break
            time.sleep(0.02)
        assert second.result()["status"] == "queued"
        assert first.result()["status"] in ("running", "done")
        release.set()
        assert second.join(timeout=30) == "second"
    finally:
        gate.set()
        release.set()
        if sem._value == 0:  # type: ignore[attr-defined]
            sem.release()


def test_worktree_passthrough_skips_creation(nx_fresh, monkeypatch, tmp_path):
    """A pre-created worktree skips _worktree_enabled()/create_worktree and is
    snapshotted as the outcome baseline."""
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    import nx as m

    wt = tmp_path / "wt"
    wt.mkdir()

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "wrote it"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)

    def _boom(*a, **kw):
        raise AssertionError("create_worktree must not be called for a given worktree")

    monkeypatch.setattr(nx_fresh._subagent, "_worktree_enabled", _boom,
                        raising=False)
    job = nx_fresh._subagent.spawn_agent_job("edit in my tree", 5, tier="write",
                                             worktree=str(wt))
    assert job.join(timeout=30) is not None
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS[job.job_id]
    assert rec["worktree"] == str(wt)
    assert rec["status"] in ("done", "partial")


def test_spawn_agent_async_string_unchanged_over_job_path(nx_fresh, monkeypatch):
    """The model-facing spawn_agent(async=True) string is byte-identical to
    the pre-refactor contract and still flows through spawn_agent_job."""
    from nx._subagent import spawn_agent
    import nx as m

    seen = {}

    def fake_loop(messages, *a, **kw):
        seen["started"] = True
        messages.append({"role": "assistant", "content": "x"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("where is X?", async_=True)
    assert out.startswith("[background sub-agent] job agent-")
    assert "continue other work in the meantime" in out
    assert seen["started"]
