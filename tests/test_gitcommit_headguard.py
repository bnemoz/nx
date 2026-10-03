"""HEAD-moved preflight guard for git_commit.

Contract under test:
- _stage_index returns the temp index AND the HEAD hash it seeded from;
- immediately before `git commit`, _git_commit_with_index re-reads HEAD and
  refuses if it moved (another session committed between staging and commit);
- the refusal names the new HEAD commit(s) that landed (old..new) so the model
  can re-inspect and retry deliberately;
- a refusal commits nothing and leaves the SHARED index untouched (staging ran
  in the private temp index, which the caller unlinks).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A real git repo with one committed file (mirrors test_git_commit.py)."""
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.chdir(tmp_path)
    import importlib
    import nx
    importlib.reload(nx)
    nx.st.PROJECT_ROOT = str(tmp_path)
    nx.st.PROJECT_ROOT_IS_GIT = True
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True,
                   capture_output=True)
    for k, v in (("user.email", "nx@example.com"), ("user.name", "nx"),
                 ("commit.gpgsign", "false")):
        subprocess.run(["git", "config", k, v], cwd=tmp_path, check=True,
                       capture_output=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path,
                   check=True, capture_output=True)
    yield nx, tmp_path
    for k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(k, None)


def _git_count(repo_path):
    r = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo_path,
                       capture_output=True, text=True)
    return int(r.stdout.strip())


def _advancing_confirm(repo_path):
    """Confirm that lands another commit mid-flight: the exact race the guard
    exists for. It runs AFTER staging but before our final rev-parse."""
    def fake_confirm(action, preview=None):
        (repo_path / "other.py").write_text("y = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "other.py"], cwd=repo_path, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "other session"],
                       cwd=repo_path, check=True, capture_output=True)
        return True
    return fake_confirm


def test_stage_index_returns_head_hash(repo):
    nx, rp = repo
    idx, head = nx._gitcommit._stage_index(str(rp))
    try:
        expected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rp,
                                  capture_output=True, text=True).stdout.strip()
        assert head == expected
        assert os.path.exists(idx)
    finally:
        os.unlink(idx)


def test_head_moved_aborts_commit(repo, monkeypatch):
    nx, rp = repo
    monkeypatch.setattr(nx._bash_jobs, "_confirm",
                        _advancing_confirm(rp))
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert result.startswith("REFUSED"), result
    assert "HEAD moved" in result
    assert "other session" in result, "must name the commit that landed"
    # nothing was committed by us: HEAD is the OTHER session's commit.
    log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=rp,
                         capture_output=True, text=True).stdout.strip()
    assert "other session" in log
    assert "fix: bump x" not in log
    # shared index untouched: our private-index staging never leaked into it,
    # so the shared index entry for a.py still equals HEAD's blob (whatever the
    # worktree holds is a plain unstaged change, not something we staged).
    idx_blob = subprocess.run(["git", "ls-files", "-s", "--", "a.py"], cwd=rp,
                              capture_output=True, text=True).stdout.split()[1]
    head_blob = subprocess.run(["git", "rev-parse", "HEAD:a.py"], cwd=rp,
                               capture_output=True, text=True).stdout.strip()
    assert idx_blob == head_blob, "our staging leaked into the shared index"


def test_head_moved_guard_passes_when_head_stable(repo, monkeypatch):
    nx, rp = repo
    monkeypatch.setattr(nx._bash_jobs, "_confirm", lambda a, preview=None: True)
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert "[git_commit]" in result
    assert _git_count(rp) == 2


def test_head_moved_names_multiple_landed_commits(repo, monkeypatch):
    nx, rp = repo

    def two_commits(action, preview=None):
        for i, msg in enumerate(("first landed", "second landed")):
            (rp / f"c{i}.py").write_text("z = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", f"c{i}.py"], cwd=rp, check=True,
                           capture_output=True)
            subprocess.run(["git", "commit", "-q", "-m", msg], cwd=rp,
                           check=True, capture_output=True)
        return True

    monkeypatch.setattr(nx._bash_jobs, "_confirm", two_commits)
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert result.startswith("REFUSED")
    assert "first landed" in result and "second landed" in result
