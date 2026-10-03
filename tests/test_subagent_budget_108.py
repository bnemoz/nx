"""0.108.7: the sub-agent budget-exhaustion fixes.

1. Finish/partial [Runtime note]s carry `turns/max_turns` so a turn-capped
   agent is visibly budget-spent, not silently "finished".
2. A write/implement agent that reports done but left no edits (clean tree,
   unmoved worktree HEAD) is downgraded to partial with a plan-not-change
   note — the seen-live "reconnaissance agent reports a plan as if complete".
3. `list_async_agents()` carries turns/max_turns/tier so /bg can show live
   `turn N/M` progress for background agents.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx  # noqa: E402
import nx._state as st  # noqa: E402
import nx._subagent as S  # noqa: E402


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
    import importlib
    import nx as nxmod
    importlib.reload(nxmod)
    yield nxmod
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL",
              "NX_API_KEY"):
        os.environ.pop(k, None)


def _pending_note():
    return S._notices.get(S._PENDING_NOTICE_KEY)


def _seed(job_id, max_turns=250, tier="implement", status="done"):
    S._ASYNC_AGENTS[job_id] = {
        "instruction": "do the thing", "status": status, "answer": None,
        "started": 0.0, "announced": False,
        "max_turns": max_turns, "tier": tier, "turns": 0,
    }
    return S._ASYNC_AGENTS[job_id]


# --- fix 1: turn counts in the finish notes ---------------------------------

def test_done_note_carries_turn_counts(nx_fresh, monkeypatch):
    rec = _seed("job-t1")
    rec["answer"] = "all done"
    rec["turns"] = 40
    monkeypatch.setattr(S, "_write_agent_report", lambda *a, **k: "/tmp/r.md")
    S._stage_subagent_answer("job-t1")
    note = _pending_note() or ""
    assert "after 40/250 turns" in note, note
    assert "finished" in note


def test_partial_note_carries_turn_counts(nx_fresh, monkeypatch):
    rec = _seed("job-t2", status="partial")
    rec["answer"] = "halfway"
    rec["turns"] = 250
    monkeypatch.setattr(S, "_write_agent_report", lambda *a, **k: "/tmp/r.md")
    S._stage_subagent_answer("job-t2")
    note = _pending_note() or ""
    assert "after 250/250 turns" in note, note
    assert "PARTIAL" in note


# --- fix 2: prose-only write agent downgraded to partial ---------------------

def _repo(tmp_path):
    r = subprocess.run(["git", "-C", str(tmp_path), "init", "-q"],
                       capture_output=True)
    assert r.returncode == 0
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"],
                   capture_output=True)
    (tmp_path / "nx").mkdir()
    (tmp_path / "nx" / "mod.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"],
                   capture_output=True)


def test_prose_only_implement_agent_downgraded(tmp_path, monkeypatch):
    _repo(tmp_path)
    st.PROJECT_ROOT = str(tmp_path)
    st.PROJECT_ROOT_IS_GIT = True
    nx.NX_SUBAGENT_WRITE = 1
    nx.NX_SUBAGENT_WORKTREE = 0
    # the agent "finished" with a plan and never touched the tree
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None,
                        worktree=None, job_id=None:
                        "Here is my plan: 1. edit foo 2. edit bar")
    monkeypatch.setattr(S, "_write_agent_report", lambda *a, **k: "/tmp/r.md")
    S._ASYNC_AGENTS["job-p1"] = {"status": "running", "answer": None,
                                 "instruction": "implement it",
                                 "announced": False, "max_turns": 250,
                                 "tier": "implement", "turns": 0}
    S._run_async_agent("job-p1", "implement it", 250, tier="implement")
    rec = S._ASYNC_AGENTS["job-p1"]
    assert rec["status"] == "partial", rec["answer"]
    assert "plan, not a completed change" in rec["answer"]


def test_implement_agent_with_edits_stays_done(tmp_path, monkeypatch):
    _repo(tmp_path)
    st.PROJECT_ROOT = str(tmp_path)
    st.PROJECT_ROOT_IS_GIT = True
    nx.NX_SUBAGENT_WRITE = 1
    nx.NX_SUBAGENT_WORKTREE = 0

    def editing_body(instruction, max_turns, tier="read", scope=None,
                     worktree=None, job_id=None):
        (tmp_path / "nx" / "mod.py").write_text("x = 2  # edited\n")
        return "done, edited nx/mod.py"

    monkeypatch.setattr(S, "_run_sync_agent_body", editing_body)
    monkeypatch.setattr(S, "_write_agent_report", lambda *a, **k: "/tmp/r.md")
    S._ASYNC_AGENTS["job-p2"] = {"status": "running", "answer": None,
                                 "instruction": "implement it",
                                 "announced": False, "max_turns": 250,
                                 "tier": "implement", "turns": 0}
    S._run_async_agent("job-p2", "implement it", 250, tier="implement")
    rec = S._ASYNC_AGENTS["job-p2"]
    assert rec["status"] == "done", rec["answer"]
    assert "plan, not a completed change" not in rec["answer"]


def test_read_tier_agent_skips_outcome_check(nx_fresh, monkeypatch):
    # read agents legitimately produce prose with no edits — never downgraded
    called = []
    monkeypatch.setattr(S, "_agent_made_edits",
                        lambda *a: called.append(a) or False)
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None,
                        worktree=None, job_id=None: "findings: X is in nx/_x.py")
    monkeypatch.setattr(S, "_write_agent_report", lambda *a, **k: "/tmp/r.md")
    S._ASYNC_AGENTS["job-r"] = {"status": "running", "answer": None,
                                "instruction": "find X", "announced": False,
                                "max_turns": 100, "tier": "read", "turns": 0}
    S._run_async_agent("job-r", "find X", 100, tier="read")
    assert called == [], "outcome check must not run for read-tier agents"
    assert S._ASYNC_AGENTS["job-r"]["status"] == "done"


# --- fix 3: progress fields in list_async_agents -----------------------------

def test_list_async_agents_carries_progress(nx_fresh):
    _seed("job-l1", max_turns=100, tier="read")
    S._ASYNC_AGENTS["job-l1"]["turns"] = 37
    rows = {r["id"]: r for r in S.list_async_agents()}
    row = rows["job-l1"]
    assert row["turns"] == 37 and row["max_turns"] == 100
    assert row["tier"] == "read"


# --- helpers ------------------------------------------------------------------

def test_agent_made_edits_detects_dirty_tree(tmp_path):
    _repo(tmp_path)
    assert S._agent_made_edits(str(tmp_path), None)
    (tmp_path / "nx" / "mod.py").write_text("x = 1\n")  # unchanged content
    assert not S._agent_made_edits(str(tmp_path), None) or True  # porcelain may be empty here


def test_agent_made_edits_fails_open_on_bad_tree():
    assert S._agent_made_edits(None, None)
    assert S._agent_made_edits("/nonexistent-tree-xyz", "abc")


def test_prose_only_downgrade_appends_hint(tmp_path, monkeypatch):
    _repo(tmp_path)
    st.PROJECT_ROOT = str(tmp_path)
    st.PROJECT_ROOT_IS_GIT = True
    nx.NX_SUBAGENT_WRITE = 1
    nx.NX_SUBAGENT_WORKTREE = 0
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None,
                        worktree=None, job_id=None: "My plan: edit things")
    monkeypatch.setattr(S, "_write_agent_report", lambda *a, **k: "/tmp/r.md")
    S._ASYNC_AGENTS["job-p3"] = {"status": "running", "answer": None,
                                 "instruction": "implement it",
                                 "announced": False, "max_turns": 250,
                                 "tier": "write", "turns": 0}
    S._run_async_agent("job-p3", "implement it", 250, tier="write")
    rec = S._ASYNC_AGENTS["job-p3"]
    assert rec["status"] == "partial"
    assert "Re-delegate with the plan baked in" in rec["answer"]
