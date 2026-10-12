"""0.145.2 per-project store partitioning.

SESSION_HOME (NX_HOME / ~/.nx) is machine-global: agents/, bg-logs/ and
worktrees/ from every concurrently-running project landed in one flat dir,
so a session in project A surfaced project B's reports/worktrees. The fix
partitions each store under a per-project slug subdir —
<basename>-<sha256(realpath)[:8]> — keyed off st.PROJECT_ROOT (cwd
fallback). These tests pin: slug stability/uniqueness, the slug subdir in
each store path, and the legacy-drain sweeps."""
import hashlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx
import nx._state as st
from nx import _bash_jobs as bj
from nx._subagent import _agents_report_dir, _purge_old_agent_reports
from nx._worktree import create_worktree, sweep_stale_worktrees


def _slug_of(root):
    real = os.path.realpath(str(root))
    base = os.path.basename(real)
    digest = hashlib.sha256(real.encode()).hexdigest()[:8]
    return f"{base}-{digest}"


# --- the slug itself ----------------------------------------------------------

def test_slug_stable_and_shape(tmp_path, monkeypatch):
    root = tmp_path / "my-project"
    root.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    s1 = nx.project_store_slug()
    s2 = nx.project_store_slug()
    assert s1 == s2, "slug must be stable across calls"
    assert s1 == _slug_of(root)  # basename + first 8 sha256 hex chars


def test_slug_differs_for_different_roots(tmp_path, monkeypatch):
    a = tmp_path / "a" / "proj"
    b = tmp_path / "b" / "proj"  # same basename, different realpath
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    monkeypatch.setattr(st, "PROJECT_ROOT", str(a))
    sa = nx.project_store_slug()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(b))
    sb = nx.project_store_slug()
    assert sa != sb, "same-named projects must not share a store"


def test_slug_sanitizes_and_defaults(tmp_path, monkeypatch):
    weird = tmp_path / "we!rd name"
    weird.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(weird))
    s = nx.project_store_slug()
    assert all(c.isalnum() or c in "-_." for c in s), s
    # empty root + no cwd fallback -> "default"
    monkeypatch.setattr(st, "PROJECT_ROOT", "")
    monkeypatch.setattr(os, "getcwd", lambda: "")
    assert nx.project_store_slug() == "default"


# --- store paths carry the slug -----------------------------------------------

def test_agents_report_dir_under_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(st, "PROJECT_ROOT", str(tmp_path / "proj"))
    d = _agents_report_dir()
    assert d == os.path.join(str(tmp_path / "home"), "agents",
                             _slug_of(tmp_path / "proj"))


def test_bg_dir_under_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(st, "PROJECT_ROOT", str(tmp_path / "proj"))
    assert bj._bg_dir() == os.path.join(str(tmp_path / "home"), "bg-logs",
                                        _slug_of(tmp_path / "proj"))


def _git(cwd, *args):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_create_worktree_under_slug(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    home = tmp_path / "nxhome"
    home.mkdir()
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "PROJECT_ROOT", str(repo))
    monkeypatch.setattr(st, "PROJECT_ROOT_IS_GIT", True)
    wt, err = create_worktree("jobx")
    assert err is None and wt and os.path.isdir(wt)
    try:
        assert _slug_of(repo) in wt
        assert os.path.dirname(wt) == os.path.join(str(home), "worktrees",
                                                   _slug_of(repo))
    finally:
        from nx._worktree import remove_worktree
        remove_worktree(wt)


# --- legacy flat stores still drain --------------------------------------------

def test_purge_drains_legacy_flat_agents_dir(tmp_path, monkeypatch):
    home = tmp_path / "home"
    flat = home / "agents"
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "PROJECT_ROOT", str(tmp_path / "proj"))
    slug_dir = _agents_report_dir()  # the real per-project dir
    os.makedirs(slug_dir, exist_ok=True)
    other_dir = flat / "otherproj-99999999"  # a foreign project's store
    other_dir.mkdir(parents=True)

    def _aged(path, days):
        with open(path, "w") as f:
            f.write("x")
        old = time.time() - days * 86400
        os.utime(path, (old, old))

    stale_flat = flat / "agent-old.md"          # legacy file: must drain
    fresh_flat = flat / "agent-new.md"          # recent file: must survive
    stale_slug = os.path.join(slug_dir, "agent-old2.md")  # own dir: must drain
    foreign = other_dir / "agent-f.md"          # foreign slug dir: hands-off
    _aged(stale_flat, 400)
    _aged(fresh_flat, 0)
    _aged(stale_slug, 400)
    _aged(foreign, 400)
    _purge_old_agent_reports()
    assert not os.path.exists(stale_flat)
    assert not os.path.exists(stale_slug)
    assert os.path.exists(fresh_flat)
    assert os.path.exists(foreign), "other projects' slug dirs must never be purged"
    assert os.path.isdir(slug_dir), "slug subdirs in the parent must never be purged"


def test_sweep_reaps_slug_dir_and_legacy_parent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    flat = home / "worktrees"
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "PROJECT_ROOT", None)  # skip the git prune step
    monkeypatch.delenv("NX_SUBAGENT_WORKTREE_TTL_HOURS", raising=False)
    slug = nx.project_store_slug()
    mine = flat / slug

    def _aged_tree(d, days):
        (d / "nx").mkdir(parents=True)
        old = time.time() - days * 3600 * 24
        os.utime(d, (old, old))
        return str(d)

    stale_slug = _aged_tree(mine / "nx-wt-agent-1-aaaa", 30)
    stale_legacy = _aged_tree(flat / "nx-wt-agent-2-bbbb", 30)
    other_slug_tree = _aged_tree(flat / "otherproj-99999999" / "nx-wt-agent-3-cccc", 30)
    sweep_stale_worktrees()
    assert not os.path.exists(stale_slug), "stale tree in own slug dir survived"
    assert not os.path.exists(stale_legacy), "stale tree in legacy parent survived"
    assert os.path.isdir(other_slug_tree), \
        "sweep must never touch other projects' slug dirs"


# --- critique read-scope nudge --------------------------------------------------

def test_critique_instruction_confines_reads_to_project_root(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True)
    seen = {}

    def fake_spawn(instruction, **kwargs):
        seen["instruction"] = instruction
        return "spawned"

    import nx._subagent as S
    monkeypatch.setattr(S, "spawn_agent", fake_spawn)
    root = os.path.realpath(str(os.getcwd()))
    monkeypatch.setattr(st, "PROJECT_ROOT", root)
    out = S.critique("the retry loop", paths="nx/_turn.py")
    assert out == "spawned"
    text = seen["instruction"]
    assert root in text, "reviewer must be told the project root"
    assert "~/.nx/agents" in text, "reviewer must be told to stay out of ~/.nx"
