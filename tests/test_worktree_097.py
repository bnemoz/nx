"""0.97.0 headless swarms: per-agent git worktree isolation for background
write-tier subagents (NX_SUBAGENT_WORKTREE=1) + tier/scope reaching the async
worker at all (0.96.x dropped both)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx  # noqa: E402
import nx._state as st  # noqa: E402
import nx._subagent as S  # noqa: E402
import nx._subagent_tiers as T  # noqa: E402
from nx._worktree import (  # noqa: E402
    _worktree_enabled, create_worktree, in_worktree, remove_worktree,
    rewrite_path,
)


def _git(cwd, *args):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "nx").mkdir()
    (tmp_path / "nx" / "mod.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def _home(tmp_path):
    # session home OUTSIDE the repo (production: ~/.nx vs the project dir) —
    # a worktree nested inside its own parent repo is exactly the collision
    # create_worktree() refuses.
    home = tmp_path.parent / (tmp_path.name + "-home")
    home.mkdir(exist_ok=True)
    return str(home)


def _setup(tmp_path, monkeypatch):
    nx.SESSION_HOME = str(tmp_path / "home")
    st.SESSION_HOME = str(tmp_path / "home")
    st.PROJECT_ROOT = str(tmp_path)
    st.PROJECT_ROOT_IS_GIT = True
    nx._state.YOLO = True
    nx.NX_SUBAGENT_WRITE = 1
    nx.NX_SUBAGENT_SYNC_INLINE = 0
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None,
                        worktree=None, job_id=None: "ok")


# --- path rewriting ---------------------------------------------------------

def test_rewrite_path_rules():
    wt = "/tmp/wt-xyz"
    assert rewrite_path("nx/mod.py", wt) == os.path.join(wt, "nx/mod.py")
    # absolute inside → unchanged; absolute outside → unchanged (scope/fence
    # will refuse it); ~-paths are outside the tree (agent home), unchanged
    assert rewrite_path(wt + "/nx/mod.py", wt) == wt + "/nx/mod.py"
    assert rewrite_path("/etc/passwd", wt) == "/etc/passwd"
    assert rewrite_path("~/.nx", wt) == "~/.nx"
    assert rewrite_path("", wt) == ""
    assert rewrite_path("nx/mod.py", "") == "nx/mod.py"


def test_in_worktree():
    wt = os.path.realpath("/tmp/wt-abc")
    assert in_worktree(wt + "/nx/mod.py", wt)
    assert in_worktree(wt, wt)
    assert not in_worktree("/etc/passwd", wt)
    assert not in_worktree(wt + "-sibling/x", wt)  # prefix must be path-boundary


# --- worktree creation ------------------------------------------------------

def test_create_worktree_from_head(tmp_path):
    _repo(tmp_path)
    nx.SESSION_HOME = _home(tmp_path)
    st.SESSION_HOME = nx.SESSION_HOME
    st.PROJECT_ROOT = str(tmp_path)
    st.PROJECT_ROOT_IS_GIT = True
    wt, err = create_worktree("job1")
    assert err is None and wt and os.path.isdir(wt)
    try:
        assert (os.path.join(wt, "nx", "mod.py"))
        assert open(os.path.join(wt, "nx", "mod.py")).read() == "x = 1\n"
        # detached HEAD, registered as a linked worktree of the parent
        out = _git(tmp_path, "worktree", "list", "--porcelain")
        assert wt in out
    finally:
        remove_worktree(wt)
    assert not os.path.exists(wt)
    out = _git(tmp_path, "worktree", "list", "--porcelain")
    assert wt not in out


def test_create_worktree_refuses_non_git(tmp_path, monkeypatch):
    st.PROJECT_ROOT = str(tmp_path)
    st.PROJECT_ROOT_IS_GIT = False
    wt, err = create_worktree("job2")
    assert wt is None and err


# --- enabled flag -----------------------------------------------------------

def test_enabled_flag(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENT_WORKTREE", 0, raising=False)
    assert _worktree_enabled() is False
    monkeypatch.setattr(nx, "NX_SUBAGENT_WORKTREE", 1, raising=False)
    assert _worktree_enabled() is True
    monkeypatch.delattr(nx, "NX_SUBAGENT_WORKTREE", raising=False)
    monkeypatch.setenv("NX_SUBAGENT_WORKTREE", "1")
    assert _worktree_enabled() is True


# --- wrapper arg rewriting --------------------------------------------------

def test_write_wrapper_rewrites_paths_into_worktree(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    wt = str(tmp_path / "home" / "worktrees" / "nx-wt-j")
    os.makedirs(os.path.join(wt, "nx"), exist_ok=True)
    seen = {}

    def fake_body(**args):
        seen.update(args)
        return "written"

    monkeypatch.setitem(nx._dispatch.DISPATCH, "write_file", fake_body)
    ctx = T._SubagentContext({}, write=True, worktree=wt)
    T._SUBAGENT_TLS.ctx = ctx
    try:
        out = T._write_tier_wrapper("write_file")(path="nx/mod.py",
                                                  new="x = 2", old="x = 1")
    finally:
        T._SUBAGENT_TLS.ctx = None
    assert out == "written"
    assert seen["path"] == os.path.join(wt, "nx/mod.py")


def test_write_wrapper_without_worktree_leaves_paths(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    seen = {}

    def fake_body(**args):
        seen.update(args)
        return "written"

    monkeypatch.setitem(nx._dispatch.DISPATCH, "write_file", fake_body)
    T._SUBAGENT_TLS.ctx = T._SubagentContext({}, write=True)
    try:
        T._write_tier_wrapper("write_file")(path="nx/mod.py", new="x", old="y")
    finally:
        T._SUBAGENT_TLS.ctx = None
    assert seen["path"] == "nx/mod.py"


def test_git_commit_and_run_tests_default_to_worktree(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    wt = str(tmp_path / "home" / "worktrees" / "nx-wt-g")
    os.makedirs(wt, exist_ok=True)
    seen = {}
    monkeypatch.setitem(nx._dispatch.DISPATCH, "git_commit",
                        lambda **a: seen.setdefault("git", a) or "ok")
    monkeypatch.setitem(nx._dispatch.DISPATCH, "run_tests",
                        lambda **a: seen.setdefault("tests", a) or "ok")
    T._SUBAGENT_TLS.ctx = T._SubagentContext({}, write=True, worktree=wt)
    try:
        T._write_tier_wrapper("git_commit")(files="nx/mod.py", message="m")
        T._write_tier_wrapper("run_tests")()
    finally:
        T._SUBAGENT_TLS.ctx = None
    assert seen["git"]["path"] == wt
    assert seen["tests"]["cwd"] == wt


def test_scope_check_uses_worktree_root(tmp_path):
    wt = str(tmp_path / "wt")
    os.makedirs(os.path.join(wt, "nx"), exist_ok=True)
    T._SUBAGENT_TLS.ctx = T._SubagentContext({}, write=True,
                                             scope=["nx/*.py"], worktree=wt)
    try:
        assert T._scope_check(os.path.join(wt, "nx", "mod.py")) is None
        r = T._scope_check(os.path.join(wt, "other.py"))
        assert r and r.startswith("ERROR:")
    finally:
        T._SUBAGENT_TLS.ctx = None


# --- async worker carries tier/scope and builds the worktree ----------------

def test_async_worker_carries_tier_and_creates_worktree(tmp_path, monkeypatch):
    _repo(tmp_path)
    _setup(tmp_path, monkeypatch)
    nx.SESSION_HOME = _home(tmp_path)
    st.SESSION_HOME = nx.SESSION_HOME
    monkeypatch.setattr(nx, "NX_SUBAGENT_WORKTREE", 1, raising=False)
    captured = {}

    def fake_body(instruction, max_turns, tier="read", scope=None,
                  worktree=None, job_id=None):
        captured.update(tier=tier, scope=scope, worktree=worktree)
        return "all done"

    monkeypatch.setattr(S, "_run_sync_agent_body", fake_body)
    S._ASYNC_AGENTS["job-wt"] = {"status": "running", "answer": None,
                                 "instruction": "do it", "announced": False}
    S._run_async_agent("job-wt", "do it", 10, tier="implement",
                       scope=["nx/*.py"])
    assert captured["tier"] == "implement"
    assert captured["scope"] == ["nx/*.py"]
    assert captured["worktree"] and os.path.isdir(captured["worktree"])
    # the job record + answer carry the worktree path
    rec = S._ASYNC_AGENTS["job-wt"]
    assert rec["worktree"] == captured["worktree"]
    assert "[Worktree:" in rec["answer"] and captured["worktree"] in rec["answer"]
    remove_worktree(captured["worktree"])


def test_async_worker_read_tier_no_worktree(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WORKTREE", 1, raising=False)
    captured = {}
    monkeypatch.setattr(S, "_run_sync_agent_body",
                        lambda instruction, max_turns, tier="read", scope=None,
                        worktree=None, job_id=None: captured.update(tier=tier) or "done")
    S._run_async_agent("job-r", "look", 10, tier="read")
    assert captured["tier"] == "read"


def test_async_worker_worktree_failure_is_nonfatal(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    st.PROJECT_ROOT_IS_GIT = False  # no repo: creation fails
    monkeypatch.setattr(nx, "NX_SUBAGENT_WORKTREE", 1, raising=False)
    # _run_async_agent doesn't register the record (that's _spawn_async's job);
    # seed it the way the worker thread would have.
    S._ASYNC_AGENTS["job-f"] = {"status": "running", "answer": None,
                                "instruction": "do it", "announced": False}
    S._run_async_agent("job-f", "do it", 10, tier="write")
    rec = S._ASYNC_AGENTS["job-f"]
    assert rec["status"] == "done"  # the agent still ran, in the shared tree
    assert "worktree isolation unavailable" in rec["answer"]


def test_spawn_agent_async_passes_tier(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WORKTREE", 0, raising=False)
    got = {}

    def fake_spawn(instruction, max_turns, tier="read", scope=None):
        got.update(tier=tier, scope=scope)
        return "job-x"

    monkeypatch.setattr(S, "_spawn_async", fake_spawn)
    S.spawn_agent("x", tier="write", scope=["nx/*.py"], **{"async": "true"})
    assert got == {"tier": "write", "scope": ["nx/*.py"]}
