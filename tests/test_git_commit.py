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
  a multi-file commit must leave `git status` clean, not phantom-MM; the
  sync re-hashes the worktree blobs and only resets entries that match HEAD
  (0.103.9), so a stale pre-change entry with worktree == HEAD is cleared
  while live WIP entries are never clobbered;
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


def _index_blob(repo_path, path):
    r = subprocess.run(["git", "ls-files", "-s", "--", path], cwd=repo_path,
                       capture_output=True, text=True)
    return r.stdout.split()[1]


def _head_blob(repo_path, path):
    r = subprocess.run(["git", "rev-parse", f"HEAD:{path}"], cwd=repo_path,
                       capture_output=True, text=True)
    return r.stdout.strip()


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
    (rp / "a.py").write_text(f'key = "ghp_{"A" * 36}"\n',
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


def test_sync_resets_stale_entry_when_worktree_matches_head(repo):
    """Residual phantom-MM (0.103.8 follow-up): an aborted update-index can
    leave the shared index holding a PRE-change blob while the worktree is
    byte-identical to HEAD. _sync_shared_index must reset that entry from
    HEAD (not cacheinfo a stale hash), leaving `git status` clean."""
    nx, rp = repo
    # Advance HEAD past the pre-change blob, then poison the index with that
    # PRE-change blob while the worktree keeps HEAD's (new) content: exactly
    # the residual phantom-MM state (stale index entry, worktree == HEAD).
    old = _index_blob(rp, "a.py")
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=rp, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=rp,
                   check=True, capture_output=True)
    subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                    f"100644,{old},a.py"], cwd=rp, check=True,
                   capture_output=True)  # poison: stale pre-change blob
    r = subprocess.run(["git", "status", "--short"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "MM a.py", "setup failed: poison not visible"
    nx._gitcommit._sync_shared_index(str(rp), ["a.py"], 60)
    r = subprocess.run(["git", "status", "--short"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", f"phantom status after sync: {r.stdout!r}"


def test_sync_does_not_clobber_live_wip_entry(repo):
    """Worktree differs from HEAD after a commit (0.103.9 residual bug): the
    shared index holds the PRE-commit blob and the worktree has FURTHER edits,
    so `git status` shows a phantom MM — and the old guard SKIPPED the path
    entirely. Sync must write the WORKTREE blob: WIP stays staged, no phantom,
    and the worktree file is never touched."""
    nx, rp = repo
    (rp / "a.py").write_text("x = 2\n", encoding="utf-8")
    old = _index_blob(rp, "a.py")  # pre-change blob (v1)
    subprocess.run(["git", "add", "a.py"], cwd=rp, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=rp, check=True,
                   capture_output=True)
    # Poison the index with the PRE-commit blob: the exact stale state a
    # git_commit leaves when its sync skipped a worktree-dirty path.
    subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                    f"100644,{old},a.py"], cwd=rp, check=True,
                   capture_output=True)
    (rp / "a.py").write_text("x = 3\n", encoding="utf-8")
    r = subprocess.run(["git", "status", "--short"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "MM a.py", "setup failed: MM not visible"
    nx._gitcommit._sync_shared_index(str(rp), ["a.py"], 60)
    r = subprocess.run(["git", "status", "--short"], cwd=rp,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "M  a.py", \
        f"expected staged WIP without phantom, got: {r.stdout!r}"
    wtblob = subprocess.run(["git", "hash-object", "--", "a.py"], cwd=rp,
                            capture_output=True, text=True).stdout.strip()
    assert _index_blob(rp, "a.py") == wtblob, "index not set to worktree blob"
    assert (rp / "a.py").read_text(encoding="utf-8") == "x = 3\n", \
        "sync modified the worktree file"


def test_sync_keeps_live_wip_staged(repo):
    """A path with staged live WIP (index blob differs from both HEAD and the
    worktree) keeps its WIP entry; the worktree file is never touched."""
    nx, rp = repo
    (rp / "a.py").write_text("x = 9\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=rp, check=True,
                   capture_output=True)  # staged in the shared index
    staged = _index_blob(rp, "a.py")
    nx._gitcommit._sync_shared_index(str(rp), ["a.py"], 60)
    assert _index_blob(rp, "a.py") != _head_blob(rp, "a.py"), \
        "sync unstaged live WIP (index reset to HEAD)"
    assert (rp / "a.py").read_text(encoding="utf-8") == "x = 9\n"


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
