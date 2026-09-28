"""ROADMAP P3-6 (0.51.0): git_commit with staged-diff review.

Contract under test:
- staging + commit run as the fixed argv shapes (`git add -- <files>`,
  `git commit -m <message>`), never free-text shell;
- the staged diff (post-stage `git diff --staged`) is what the user reviews:
  it is passed to _confirm as the approval preview;
- the same text is secret-scanned BEFORE any confirm round-trip — a hit
  refuses without committing;
- an empty staged diff commits nothing;
- denial commits nothing and leaves the SHARED index untouched (staging ran
  in a private GIT_INDEX_FILE; since 0.103.0 another session's or the user's
  already-staged work can never be swept into a preview, and a denied commit
  never leaves stray staged changes);
- the shared index is re-synced to the commit's tree afterwards (0.103.7):
  a multi-file commit must leave `git status` clean, not phantom-MM;
- the action string "git commit: <subject>" has a local medium risk hint (no
  classifier round-trip), and _allowlist_tool maps it to git_commit.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A real git repo with one committed file, so every stage/diff/commit is
    genuine git behavior (offline, local binary only)."""
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.setenv("NX_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.chdir(tmp_path)
    import importlib
    import nx
    importlib.reload(nx)
    # _state is not re-executed by reload(nx): its import-time PROJECT_ROOT
    # still points at the real checkout. Point it at this test repo.
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


def _last_commit(repo_path):
    r = subprocess.run(["git", "log", "--oneline", "-1"], cwd=repo_path,
                       capture_output=True, text=True)
    return r.stdout.strip()


def _git_count(repo_path):
    r = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=repo_path,
                       capture_output=True, text=True)
    return int(r.stdout.strip())


def test_commit_reviews_staged_diff_and_commits(repo, monkeypatch):
    nx, rp = repo
    seen = {}

    def fake_confirm(action, preview=None):
        seen["action"] = action
        seen["preview"] = preview
        return True

    monkeypatch.setattr(nx._bash_jobs, "_confirm", fake_confirm)
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py", message="fix: bump x")
    assert "committed" in result or "[git_commit]" in result
    assert seen["action"] == "git commit: fix: bump x"
    # the user reviewed the POST-STAGE diff, not the argv
    assert "+x = 2" in seen["preview"]
    assert "-x = 1" in seen["preview"]
    assert "bump x" in _last_commit(rp)
    assert _git_count(rp) == 2
    # nothing left staged
    r = subprocess.run(["git", "diff", "--staged", "--name-only"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == ""


def test_denial_commits_nothing(repo, monkeypatch):
    nx, rp = repo

    def fake_confirm(action, preview=None):
        return False

    monkeypatch.setattr(nx._bash_jobs, "_confirm", fake_confirm)
    (rp / "a.py").write_text("x = 3\n", encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py", message="nope")
    assert result.startswith("DENIED")
    assert _git_count(rp) == 1  # no commit
    # the SHARED index was never touched (staging ran in a private index):
    # nothing left staged, and the denial can't leak into other sessions.
    r = subprocess.run(["git", "diff", "--staged", "--name-only"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == ""


def test_secret_in_diff_refuses_before_confirm(repo, monkeypatch):
    nx, rp = repo
    calls = []

    def fake_confirm(action, preview=None):
        calls.append(1)
        return True

    monkeypatch.setattr(nx._bash_jobs, "_confirm", fake_confirm)
    (rp / "a.py").write_text('key = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"\n',
                             encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py", message="oops")
    assert result.startswith("REFUSED")
    assert "secret" in result.lower()
    assert not calls, "secret scan must fire before any confirm round-trip"
    assert _git_count(rp) == 1


def test_empty_staged_diff_commits_nothing(repo, monkeypatch):
    nx, rp = repo
    monkeypatch.setattr(nx._bash_jobs, "_confirm", lambda a, preview=None: True)
    result = nx._gitcommit.git_commit(files="a.py", message="nothing changed")
    assert "nothing staged" in result
    assert _git_count(rp) == 1


def test_missing_message_and_files_are_arg_errors(repo):
    nx, _ = repo
    assert nx._gitcommit.git_commit(files="", message="m").startswith("ERROR")
    assert nx._gitcommit.git_commit(files="a.py", message="").startswith("ERROR")


def test_multi_file_commit_leaves_status_clean(repo, monkeypatch):
    """0.103.7 regression: _sync_shared_index used to emit ONE --cacheinfo
    flag for ALL specs; git consumes one triple per flag, the extras parsed
    as positional paths, update-index aborted, and every committed path
    showed a phantom `MM` in `git status` afterwards."""
    nx, rp = repo
    monkeypatch.setattr(nx._bash_jobs, "_confirm", lambda a, preview=None: True)
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    (rp / "b.py").write_text("y = 1\n", encoding="utf-8")
    result = nx._gitcommit.git_commit(files="a.py b.py", message="fix: two")
    assert "[git_commit]" in result
    assert _git_count(rp) == 2
    r = subprocess.run(["git", "status", "--short"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", f"phantom status after commit: {r.stdout!r}"


def test_non_repo_path_is_refused(repo, tmp_path):
    nx, _ = repo
    other = tmp_path / "elsewhere"
    other.mkdir()
    result = nx._gitcommit.git_commit(files=".", message="m", path=str(other))
    assert result.startswith("ERROR")
    assert "not a git repository" in result


def test_action_string_has_local_risk_hint(repo):
    nx, _ = repo
    from nx._risk import _local_risk_hint
    assert _local_risk_hint("git commit: fix: bump x") == (
        "medium", "commit with staged-diff review")


def test_allowlist_tool_maps_commit_action(repo):
    nx, _ = repo
    from nx._allowed import _allowlist_tool
    assert _allowlist_tool("git commit: fix: bump x") == "git_commit"


def test_dispatch_and_write_tier_contain_git_commit(repo):
    nx, _ = repo
    assert "git_commit" in nx._dispatch.DISPATCH
    assert "git_commit" in nx._plan.PLAN_GATE_TOOLS
    assert "git_commit" in nx._subagent._WRITE_TIER_TOOL_NAMES
    names = [t["function"]["name"] for t in nx.TOOLS]
    assert names.count("git_commit") == 1


def test_schema_required_matches_body_signature(repo):
    nx, _ = repo
    import inspect
    body = nx._dispatch.DISPATCH["git_commit"]
    sig = inspect.signature(body)
    for prop in nx._gitcommit.__doc__ or []:
        pass
    tool = next(t for t in nx.TOOLS
                if t["function"]["name"] == "git_commit")
    required = set(tool["function"]["parameters"]["required"])
    accepted = {p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    assert required <= accepted
    assert required == {"files", "message"}
