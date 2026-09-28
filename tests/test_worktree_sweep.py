"""0.103.7 worktree-leak fix: remove_worktree() was never called on the async
path, so crashed swarms leaked `nx-wt-agent-*` checkouts under the session
home forever. Two reapers now: sweep_stale_worktrees() (mtime TTL, run per
write-tier spawn before create_worktree) and _evict_finished_agents() (popped
records hand their tree to remove_worktree after the lock is released)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for _k in ("NX_SUBAGENT_WORKTREE_TTL_HOURS", "NX_SUBAGENT_WORKTREE"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

import nx._state as st  # noqa: E402
import nx._subagent as S  # noqa: E402
from nx._worktree import sweep_stale_worktrees  # noqa: E402


@pytest.fixture()
def sweep_home(tmp_path, monkeypatch):
    """A session home with a worktrees/ dir; PROJECT_ROOT=None so the sweep
    skips its final prune (no repo needed, no subprocess on the happy path)."""
    home = tmp_path / "nxhome"
    root = home / "worktrees"
    root.mkdir(parents=True)
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "PROJECT_ROOT", None)
    monkeypatch.delenv("NX_SUBAGENT_WORKTREE_TTL_HOURS", raising=False)
    return root


def _fake_tree(root, name, age_hours):
    d = root / name
    (d / "nx").mkdir(parents=True)
    old = time.time() - age_hours * 3600
    os.utime(d, (old, old))
    return str(d)


# --- sweep_stale_worktrees ----------------------------------------------------

def test_sweep_removes_only_stale_agent_trees(sweep_home):
    old = _fake_tree(sweep_home, "nx-wt-agent-1-aaaaaa", age_hours=48)
    fresh = _fake_tree(sweep_home, "nx-wt-agent-2-bbbbbb", age_hours=1)
    session = _fake_tree(sweep_home, "nx-wt-session-20260101-000000-abc123",
                         age_hours=4800)  # a live session's tree must survive
    other = _fake_tree(sweep_home, "nx-wt-job3", age_hours=48)
    sweep_stale_worktrees()
    assert not os.path.exists(old), "stale agent tree survived the sweep"
    assert os.path.isdir(fresh), "fresh agent tree was swept"
    assert os.path.isdir(session), "session worktree was swept"
    assert os.path.isdir(other), "non-agent name was swept"


def test_sweep_ttl_zero_never_sweeps(sweep_home, monkeypatch):
    monkeypatch.setenv("NX_SUBAGENT_WORKTREE_TTL_HOURS", "0")
    old = _fake_tree(sweep_home, "nx-wt-agent-3-cccccc", age_hours=24 * 365)
    sweep_stale_worktrees()
    assert os.path.isdir(old)


def test_sweep_honors_custom_ttl(sweep_home, monkeypatch):
    monkeypatch.setenv("NX_SUBAGENT_WORKTREE_TTL_HOURS", "1")
    d = _fake_tree(sweep_home, "nx-wt-agent-4-dddddd", age_hours=2)
    sweep_stale_worktrees()
    assert not os.path.exists(d)


def test_sweep_missing_root_is_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path / "nope"))
    monkeypatch.setattr(st, "PROJECT_ROOT", None)
    sweep_stale_worktrees()  # must not raise


# --- eviction removes the evicted record's tree -------------------------------

def test_eviction_removes_worktree_of_evicted_record(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "PROJECT_ROOT", None)  # remove_worktree: rmtree path
    monkeypatch.setattr(S, "_MAX_FINISHED_AGENTS", 1)
    monkeypatch.setattr(S, "_ASYNC_AGENTS", {})  # fresh ledger, restored after
    wt = tmp_path / "wt-victim"
    (wt / "nx").mkdir(parents=True)
    for i in range(3):
        rec = {"instruction": "q", "status": "done", "answer": "a",
               "started": i, "finished": i, "announced": True}
        if i == 0:
            rec["worktree"] = str(wt)
        S._ASYNC_AGENTS[f"agent-{i}-evict"] = rec
    S._evict_finished_agents()  # self-locking (0.103.7)
    assert not os.path.exists(wt), "evicted record's worktree survived"
    assert "agent-0-evict" not in S._ASYNC_AGENTS
    assert "agent-2-evict" in S._ASYNC_AGENTS  # the floor record stays


# --- the spawn sweeps before claiming a name ----------------------------------

def test_spawn_sweeps_before_creating_worktree(tmp_path, monkeypatch):
    calls = []
    # import INSIDE the test: conftest's per-test state rewind can evict
    # nx._worktree from sys.modules between tests, so a module-scope binding
    # may be an orphan -- _run_async_agent re-imports the live module at call
    # time, so patch that one.
    import nx._worktree as Wlive
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path))
    monkeypatch.setattr(st, "PROJECT_ROOT", None)
    monkeypatch.setattr(Wlive, "sweep_stale_worktrees", lambda: calls.append(1))
    monkeypatch.setattr(Wlive, "_worktree_enabled", lambda: True)
    monkeypatch.setattr(Wlive, "create_worktree", lambda jid: (None, "no repo"))
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None,
                        worktree=None: "ok")
    S._ASYNC_AGENTS["job-sweep"] = {"status": "running", "answer": None,
                                    "instruction": "x", "announced": False}
    try:
        S._run_async_agent("job-sweep", "x", 5, tier="write")
    finally:
        with S._ASYNC_LOCK:
            S._ASYNC_AGENTS.pop("job-sweep", None)
    assert calls == [1], "spawn did not sweep crash leftovers"
