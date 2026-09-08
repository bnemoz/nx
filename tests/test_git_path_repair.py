"""Regression tests: git_* tools survive arg misuse instead of erroring (0.39.2).

The model keeps (a) passing a FILE where the `path` directory belongs —
git_diff(path="pyproject.toml") died with "fatal: cannot change to
'pyproject.toml': Not a directory" — and (b) passing an absolute path as
`file`, which git rejects with "fatal: ... is outside repository". Both now
repair in-flight inside _run_git: a file-as-`path` runs in the file's
directory and restricts the command to that file; absolute pathspecs after
`--` are made repo-relative. Genuinely unrelated paths are still left for
git to report.
"""
import importlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _mkrepo(tmp_path):
    """Throwaway repo: pkg/mod.py committed, then edited, plus other.py."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("x = 1\n")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=str(tmp_path / "gitconfig"),
               GIT_CONFIG_SYSTEM="/dev/null", HOME=str(tmp_path),
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    for a in (["init", "-q"], ["add", "."], ["commit", "-qm", "one"]):
        subprocess.run(["git", *a], cwd=repo, env=env, check=True, capture_output=True)
    (repo / "pkg" / "mod.py").write_text("x = 2\n")
    (repo / "other.py").write_text("y = 1\n")
    return repo


@pytest.fixture()
def nx_mod(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True
    yield nx
    nx.YOLO = False
    nx._state._TOOL_RESULT_CACHE.clear()


def test_git_diff_file_passed_as_path(tmp_path):
    """The transcript failure: git_diff(path=<a file>) used to be
    'fatal: cannot change to ...: Not a directory'; now it diffs that file."""
    from nx._tools import git_diff
    repo = _mkrepo(tmp_path)
    out = git_diff(path=str(repo / "pkg" / "mod.py"))
    assert "Not a directory" not in out
    assert "+x = 2" in out


def test_git_diff_absolute_file(tmp_path):
    from nx._tools import git_diff
    repo = _mkrepo(tmp_path)
    out = git_diff(file=str(repo / "pkg" / "mod.py"))
    assert "outside repository" not in out
    assert "+x = 2" in out


def test_git_diff_staged_absolute_file(tmp_path):
    from nx._tools import git_diff
    repo = _mkrepo(tmp_path)
    subprocess.run(["git", "add", "pkg/mod.py"], cwd=repo, check=True, capture_output=True)
    out = git_diff(staged=True, file=str(repo / "pkg" / "mod.py"))
    assert "outside repository" not in out
    assert "+x = 2" in out


def test_git_log_absolute_file(tmp_path):
    from nx._tools import git_log
    repo = _mkrepo(tmp_path)
    out = git_log(max_count=5, file=str(repo / "pkg" / "mod.py"))
    assert "outside repository" not in out
    assert "one" in out


def test_git_status_with_file_path_still_runs(tmp_path):
    from nx._tools import git_status
    repo = _mkrepo(tmp_path)
    out = git_status(path=str(repo / "pkg" / "mod.py"))
    assert "Not a directory" not in out
    assert out.startswith("##")          # branch line: status ran, unrestricted


def test_unrelated_absolute_file_still_errors(tmp_path):
    """Repair only reroutes commands toward the named file's own directory;
    a nonexistent one still surfaces a git ERROR (no silent swallow)."""
    from nx._tools import git_diff
    repo = _mkrepo(tmp_path)
    out = git_diff(file=str(tmp_path / "elsewhere" / "nope.py"))
    assert out.startswith("ERROR:")
    assert "+x = 2" not in out


def test_run_tool_dispatch_caches_file_only_git_diff(nx_mod, tmp_path):
    """End-to-end through run_tool: file-only args (no `path`) work and are
    cache-keyed verbatim like any other read-only call."""
    repo = _mkrepo(tmp_path)
    abs_file = str(repo / "pkg" / "mod.py")
    first = nx_mod._dispatch.run_tool("git_diff", {"file": abs_file})
    assert "+x = 2" in first
    second = nx_mod._dispatch.run_tool("git_diff", {"file": abs_file})
    assert second == first
