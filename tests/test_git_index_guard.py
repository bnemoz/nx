#!/usr/bin/env python3
"""Regression tests for the git-index guard (run_bash).

The incident: a session ran `git add … && git commit` through run_bash and its
commit swept the SHARED index — including ANOTHER session's staged revert of a
milestone commit. nx's git_commit tool is immune (0.103.0 stages into a private
GIT_INDEX_FILE, so a foreign staged set is never picked up); raw shell git has
no such isolation, and /undo cannot un-commit work that was staged by someone
else. The guard therefore joins the clobber guard's shape in run_bash: a
deterministic check that forces its OWN approval prompt even in YOLO, naming
the staged paths, and refuses outright in a background sub-agent (which has no
terminal to prompt on).

Attribution is deliberately conservative — git records no owner in the index,
so ANY staged entry is enough to ask. The common flow stays free: the check
runs before the command, so `git add mine.py && git commit -m x` from a clean
index never trips it.

`NX_GIT_INDEX_GUARD=0` is the kill-switch.
"""
import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for _k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from nx._clobber import (  # noqa: E402
    git_index_guard, git_index_guard_message, git_mutating_index_op,
    staged_index_paths,
)


def _git(repo, *args):
    env = dict(os.environ)
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.com"),
                 ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.com")):
        env.setdefault(k, v)
    return subprocess.run(["git", *args], cwd=str(repo), check=True,
                          capture_output=True, text=True, env=env)


@pytest.fixture()
def repo(tmp_path):
    """A real git repo with one commit and a clean index."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(r, "add", "seed.txt")
    _git(r, "commit", "-q", "-m", "seed")
    return r


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True  # the guard must fire anyway — that is the whole point
    yield nx, tmp_path
    nx.YOLO = False
    for k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# 1. static scan: which commands mutate the git index
# ---------------------------------------------------------------------------
def test_mutating_ops_are_detected():
    for cmd in ("git add .", "git add -A", "git add mine.py && git commit -m x",
                "git commit -m done", "git commit -am x", "git reset --hard HEAD~1",
                "git reset HEAD~1", "git restore --staged file.py",
                "git merge --squash feature", "cd /tmp && git commit -m x",
                "GIT_DIR=/x git commit -m y", "sudo git commit -m x",
                "(git add .)"):
        assert git_mutating_index_op(cmd) is not None, cmd


def test_non_index_git_and_non_git_are_exempt():
    for cmd in ("git status", "git log --oneline", "git diff --cached",
                "git status --porcelain", "git restore file.py",  # worktree only
                "git merge feature",  # no --squash: its own commit path
                "git commit-tree abc",  # plumbing, not a subcommand match
                "gitstatus", "digitgit commit", "echo 'git commit'",
                "rg 'git commit'",  # quoted: a search pattern, not an invocation
                "echo \"git commit\""):
        assert git_mutating_index_op(cmd) is None, cmd


def test_unquoted_search_for_the_words_is_an_accepted_false_positive():
    # `rg git commit` reads as an invocation to a static scan; the cost is at
    # most one prompt, and only while the index is dirty (the shell-parsing
    # limitation the write fence's docstring records)
    assert git_mutating_index_op("rg git commit") is not None


def test_repo_target_is_extracted_for_minus_C():
    # the guard must check the repo the command names, not the process cwd
    assert git_mutating_index_op("git -C /x/y commit -m m") == ("commit", "/x/y")
    assert git_mutating_index_op("git -C=/x/y commit -m m") == ("commit", "/x/y")
    assert git_mutating_index_op("git --git-dir=/x/.git add .") == ("add", "/x")
    assert git_mutating_index_op("git commit -m m") == ("commit", None)
    # -c is a config setting, not a path
    assert git_mutating_index_op("git -c core.editor=vim commit") == ("commit", None)


def test_flagged_ops_report_their_flag():
    assert git_mutating_index_op("git restore --staged a.py")[0] == "restore --staged"
    assert git_mutating_index_op("git merge --squash br")[0] == "merge --squash"


# ---------------------------------------------------------------------------
# 2. reading the SHARED staged set
# ---------------------------------------------------------------------------
def test_clean_index_is_empty(repo):
    assert staged_index_paths(str(repo)) == []


def test_untracked_and_unstaged_are_not_staged(repo):
    (repo / "new.py").write_text("n\n", encoding="utf-8")
    (repo / "seed.txt").write_text("changed\n", encoding="utf-8")
    assert staged_index_paths(str(repo)) == []


def test_staged_entries_are_reported(repo):
    (repo / "other.py").write_text("o\n", encoding="utf-8")
    _git(repo, "add", "other.py")
    (repo / "seed.txt").write_text("mine\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    assert staged_index_paths(str(repo)) == ["other.py", "seed.txt"]


def test_deletions_and_renames_are_reported(repo):
    _git(repo, "mv", "seed.txt", "renamed.txt")
    assert staged_index_paths(str(repo)) == ["renamed.txt"]
    _git(repo, "commit", "-q", "-m", "rename")
    _git(repo, "rm", "renamed.txt")
    assert staged_index_paths(str(repo)) == ["renamed.txt"]


def test_shared_index_is_read_even_with_a_private_git_index_file(repo):
    """git_commit runs with GIT_INDEX_FILE pointed at a private copy; the
    guard's question is about the SHARED index, so it must survive that env."""
    (repo / "foreign.py").write_text("f\n", encoding="utf-8")
    _git(repo, "add", "foreign.py")
    os.environ["GIT_INDEX_FILE"] = str(repo / ".git" / "nx-index-nonexistent")
    try:
        assert staged_index_paths(str(repo)) == ["foreign.py"]
    finally:
        del os.environ["GIT_INDEX_FILE"]


def test_non_repo_is_nothing_checkable(tmp_path):
    assert staged_index_paths(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 3. the composed check + its kill switch + the message
# ---------------------------------------------------------------------------
def test_guard_fires_only_for_the_conjunction(repo):
    (repo / "foreign.py").write_text("f\n", encoding="utf-8")
    _git(repo, "add", "foreign.py")
    # mutating op + foreign staged entry → the entry is at risk
    assert git_index_guard("git commit -m mine", cwd=str(repo)) == ["foreign.py"]
    # staged work alone is nobody's business until a mutating op arrives
    assert git_index_guard("git status", cwd=str(repo)) == []
    assert git_index_guard("ls -la", cwd=str(repo)) == []


def test_clean_index_keeps_commit_prompt_free(repo):
    (repo / "mine.py").write_text("m\n", encoding="utf-8")
    # the check runs BEFORE the command, so stage-then-commit from a clean
    # index is exempt even though the same command would trip a dirty one
    assert git_index_guard("git add mine.py && git commit -m mine",
                           cwd=str(repo)) == []


def test_guard_names_the_staged_paths(repo):
    (repo / "milestone.md").write_text("revert\n", encoding="utf-8")
    _git(repo, "add", "milestone.md")
    msg = git_index_guard_message("git commit -m mine", ["milestone.md"])
    assert "milestone.md" in msg and "git commit" in msg
    assert "[Y/n/esc]" in msg
    many = [f"f{i}.py" for i in range(5)]
    msg2 = git_index_guard_message("git add -A", many)
    assert "(+2 more)" in msg2


def test_kill_switch(repo):
    (repo / "foreign.py").write_text("f\n", encoding="utf-8")
    _git(repo, "add", "foreign.py")
    os.environ["NX_GIT_INDEX_GUARD"] = "0"
    try:
        assert git_index_guard("git commit -m mine", cwd=str(repo)) == []
    finally:
        del os.environ["NX_GIT_INDEX_GUARD"]


# ---------------------------------------------------------------------------
# 4. run_bash wiring: its own prompt under YOLO, refuse in a sub-agent
# ---------------------------------------------------------------------------
def _stage_foreign(repo, name="foreign.py"):
    """Another session's staged work, as it appears in the shared index."""
    (repo / name).write_text("someone else's staged revert\n", encoding="utf-8")
    _git(repo, "add", name)


def test_yolo_still_prompts_and_denial_changes_nothing(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    prompts = []

    def fake_ask(prompt, allow_hint=None):
        prompts.append(prompt)
        return "n"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval", fake_ask)
    monkey.setattr(nx, "_confirm", lambda *a, **k: pytest.fail(
        "the guard prompts on its own; _confirm must not also run"))
    try:
        out = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert out == "DENIED by user"
    assert len(prompts) == 1, prompts
    assert "foreign.py" in prompts[0]
    # denial must leave the foreign staged work exactly as it was
    assert staged_index_paths(str(repo)) == ["foreign.py"]
    assert _git(repo, "log", "--oneline").stdout.count("\n") == 1


def test_approval_lets_the_commit_run(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    prompts = []

    def fake_ask(prompt, allow_hint=None):
        prompts.append(prompt)
        return "y"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval", fake_ask)
    try:
        out = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert len(prompts) == 1
    assert "[exit 0]" in out, out
    # that is the swept commit: the foreign entry landed in it (the guard
    # asks, it does not veto)
    assert "foreign.py" in _git(repo, "show", "--stat", "--oneline", "HEAD").stdout


def test_esc_drops_to_chat(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval", lambda prompt, allow_hint=None: "esc")
    try:
        with pytest.raises(nx._EscToChat):
            nx.run_bash("git add -A", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert staged_index_paths(str(repo)) == ["foreign.py"]


def test_background_sub_agent_refuses(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    import nx._subagent as sa
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sa, "_subagent_quiet", lambda: True)
    monkey.setattr(nx, "_ask_approval", lambda *a, **k: pytest.fail(
        "a detached agent has no terminal to prompt on"))
    try:
        out = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert out.startswith("ERROR:")
    assert "foreign.py" in out and "refused in a background sub-agent" in out
    assert "let the parent run it" in out
    # and it really did not run
    assert staged_index_paths(str(repo)) == ["foreign.py"]


def test_sub_agent_refusal_is_grammatical_for_several_paths(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo, "a.py")
    _stage_foreign(repo, "b.py")
    import nx._subagent as sa
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sa, "_subagent_quiet", lambda: True)
    try:
        out = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert "a.py, b.py are staged" in out, out


def test_clean_index_commit_never_prompts(nx_fresh, repo):
    """The false-positive guard: this is the flow every session uses."""
    nx, _tmp = nx_fresh
    (repo / "mine.py").write_text("m\n", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval", lambda *a, **k: pytest.fail(
        "a clean index must not prompt"))
    monkey.setattr(nx, "_confirm", lambda *a, **k: True)
    try:
        out = nx.run_bash("git add mine.py && git commit -q -m mine",
                          timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert "[exit 0]" in out, out
    assert "mine" in _git(repo, "log", "--oneline", "-1").stdout


def test_read_only_git_is_untouched_under_yolo(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval", lambda *a, **k: pytest.fail(
        "git status mutates nothing"))
    try:
        out = nx.run_bash("git status --porcelain", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert "foreign.py" in out and "[exit 0]" in out


def test_kill_switch_end_to_end(nx_fresh, repo):
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    os.environ["NX_GIT_INDEX_GUARD"] = "0"
    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval", lambda *a, **k: pytest.fail(
        "NX_GIT_INDEX_GUARD=0 must be silent"))
    try:
        out = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
        del os.environ["NX_GIT_INDEX_GUARD"]
    assert "[exit 0]" in out, out


def test_minus_C_target_is_the_index_checked(nx_fresh, repo, tmp_path):
    """A dirty repo named with -C trips the guard even though the process cwd
    (tmp_path) is no repo at all — and the reverse stays quiet."""
    nx, _tmp = nx_fresh
    _stage_foreign(repo)
    prompts = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval",
                   lambda prompt, allow_hint=None: prompts.append(prompt) or "n")
    try:
        out = nx.run_bash(f"git -C {repo} commit -m mine", timeout=20)
        assert out == "DENIED by user" and len(prompts) == 1
        # same command shape, clean target repo → no prompt
        clean = tmp_path / "clean"
        clean.mkdir()
        _git(clean, "init", "-q")
        _git(clean, "config", "user.email", "t@example.com")
        _git(clean, "config", "user.name", "t")
        (clean / "s.txt").write_text("s\n", encoding="utf-8")
        _git(clean, "add", "s.txt")
        _git(clean, "commit", "-q", "-m", "seed")
        (clean / "mine.txt").write_text("m\n", encoding="utf-8")
        prompts.clear()
        out2 = nx.run_bash(f"git -C {clean} add mine.txt && git -C {clean} "
                           f"commit -q -m mine", timeout=20)
    finally:
        monkey.undo()
    assert prompts == [] and "[exit 0]" in out2, (prompts, out2)


# ---------------------------------------------------------------------------
# 5. session-scoped staged-path claims (0.125.0): the false-positive fix
# ---------------------------------------------------------------------------
@pytest.fixture()
def no_claims():
    """The claim registry is process-global; start every claim test empty and
    leave it empty so test order can't leak a claim into another test."""
    import nx._stagedclaim as sc
    with sc._LOCK:
        saved = set(sc._CLAIMED)
        sc._CLAIMED.clear()
    yield sc
    with sc._LOCK:
        sc._CLAIMED.clear()
        sc._CLAIMED.update(saved)


def test_claimed_paths_are_subtracted_from_the_guard(repo, no_claims):
    """A path THIS session staged is its own business; a foreign one is not."""
    sc = no_claims
    _stage_foreign(repo, "foreign.py")
    (repo / "mine.py").write_text("m\n", encoding="utf-8")
    _git(repo, "add", "mine.py")
    assert git_index_guard("git commit -m mine", cwd=str(repo)) == ["foreign.py",
                                                                    "mine.py"]
    sc.mark_staged([str(repo / "mine.py")])
    assert git_index_guard("git commit -m mine", cwd=str(repo)) == ["foreign.py"]
    # the message lists ONLY the foreign path
    msg = git_index_guard_message("git commit -m mine",
                                  git_index_guard("git commit -m mine",
                                                  cwd=str(repo)))
    assert "foreign.py" in msg and "mine.py" not in msg


def test_claim_matching_survives_a_subdir_cwd(repo, no_claims):
    """git status prints repo-root-relative paths whatever the cwd, so the
    claim (an absolute path) must still match when the guard runs from a
    subdirectory."""
    sc = no_claims
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("m\n", encoding="utf-8")
    _git(repo, "add", "pkg/mod.py")
    assert git_index_guard("git commit -m mine", cwd=str(sub)) == ["pkg/mod.py"]
    sc.mark_staged([str(repo / "pkg" / "mod.py")])
    assert git_index_guard("git commit -m mine", cwd=str(sub)) == []


def test_unclaimed_and_unkillable_semantics_hold(repo, no_claims):
    """The pinned semantics are claim-independent: a non-mutating command and
    the NX_GIT_INDEX_GUARD=0 kill switch stay silent even with unclaimed work
    staged, and an empty claim set changes nothing."""
    _stage_foreign(repo)
    assert git_index_guard("git status", cwd=str(repo)) == []
    assert git_index_guard("ls -la", cwd=str(repo)) == []
    os.environ["NX_GIT_INDEX_GUARD"] = "0"
    try:
        assert git_index_guard("git commit -m mine", cwd=str(repo)) == []
    finally:
        del os.environ["NX_GIT_INDEX_GUARD"]


# ---------------------------------------------------------------------------
# 6. claim persistence (0.128.x): a claim survives a session restart, but only
#    while the shared index still holds the SAME blob
# ---------------------------------------------------------------------------
def test_claim_survives_process_restart(repo, no_claims):
    """mark_staged writes <repo>/.nx/staged-claims.json; a FRESH registry (the
    restarted session's empty _CLAIMED) still matches the unchanged staged
    blob, so the guard no longer re-prompts on its own prior work."""
    sc = no_claims
    (repo / "mine.py").write_text("m\n", encoding="utf-8")
    _git(repo, "add", "mine.py")
    sc.mark_staged([str(repo / "mine.py")])
    # simulate a restart: the in-process set is empty, only the file remains
    with sc._LOCK:
        saved, sc._CLAIMED = sc._CLAIMED, set()
    try:
        assert git_index_guard("git commit -m mine", cwd=str(repo)) == []
        assert os.path.exists(repo / ".nx" / "staged-claims.json")
    finally:
        with sc._LOCK:
            sc._CLAIMED = saved


def test_restarted_claim_expires_when_blob_changes(repo, no_claims):
    """A prior session's claim matches only the recorded blob: restaging
    DIFFERENT content for the same path (another session or the user) invalidates
    it and the guard prompts again."""
    sc = no_claims
    (repo / "mine.py").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "mine.py")
    sc.mark_staged([str(repo / "mine.py")])
    with sc._LOCK:
        saved, sc._CLAIMED = sc._CLAIMED, set()
    try:
        (repo / "mine.py").write_text("v2\n", encoding="utf-8")
        _git(repo, "add", "mine.py")
        assert git_index_guard("git commit -m mine", cwd=str(repo)) == ["mine.py"]
    finally:
        with sc._LOCK:
            sc._CLAIMED = saved


def test_foreign_staged_path_without_claim_still_prompts(repo, no_claims):
    """Persistence must not widen trust: a staged path with no claim file entry
    (never staged by nx) keeps prompting."""
    _stage_foreign(repo)
    assert git_index_guard("git commit -m mine", cwd=str(repo)) == ["foreign.py"]


def test_staged_deletion_claim_is_conservative(repo, no_claims):
    """A staged deletion leaves no index entry, so a RESTARTED session cannot
    verify the blob and stays conservative (prompts); the same-session claim
    (empty registry cleared here is NOT the same-session case) — pin the
    restart semantics: no verifiable blob ⇒ prompt."""
    sc = no_claims
    (repo / "gone.txt").write_text("g\n", encoding="utf-8")
    _git(repo, "add", "gone.txt")
    _git(repo, "commit", "-q", "-m", "add gone")
    os.unlink(repo / "gone.txt")
    _git(repo, "add", "gone.txt")  # stages the deletion: index entry removed
    sc.mark_staged([str(repo / "gone.txt")])
    with sc._LOCK:
        saved, sc._CLAIMED = sc._CLAIMED, set()
    try:
        assert git_index_guard("git commit -m mine",
                               cwd=str(repo)) == ["gone.txt"]
    finally:
        with sc._LOCK:
            sc._CLAIMED = saved


def test_mark_staged_ignores_failures(no_claims):
    """Bookkeeping never raises into a commit or a shell run."""
    sc = no_claims
    sc.mark_staged([])
    sc.mark_staged(["", None])  # type: ignore[list-item]
    assert sc.claimed_staged([]) == set()
    assert sc.claimed_staged(["", None]) == set()  # type: ignore[list-item]


def test_gitcommit_staged_wip_does_not_prompt_this_session(nx_fresh, repo,
                                                           no_claims):
    """The reported false positive: `git_commit` stages in a private index,
    then `_sync_shared_index` re-caches the committed paths — and LEAVES one
    staged whenever the worktree differs from HEAD at sync time (live WIP,
    simulated here as an edit landing inside the approval window). The next raw
    git op of the SAME session must not prompt about that WIP; a path staged by
    someone else afterwards still prompts, alone in the message."""
    nx, _tmp = nx_fresh
    monkey = pytest.MonkeyPatch()

    def confirm_edited(action, preview=None):
        # the worktree moves AFTER git_commit staged (v1) but before its sync:
        # HEAD takes v1, the sync re-caches the worktree blob (v2), leaving
        # wip.py STAGED in the shared index — the residual WIP of the bug
        (repo / "wip.py").write_text("v2 uncommitted\n", encoding="utf-8")
        return True

    monkey.setattr(nx._bash_jobs, "_confirm", confirm_edited)
    (repo / "wip.py").write_text("v1\n", encoding="utf-8")
    out = nx._gitcommit.git_commit(files="wip.py", message="add wip",
                                   path=str(repo))
    assert "[git_commit]" in out, out
    assert staged_index_paths(str(repo)) == ["wip.py"], "sync left WIP staged"

    prompts = []
    monkey.setattr(nx, "_ask_approval",
                   lambda prompt, allow_hint=None: prompts.append(prompt) or "y")
    monkey.setattr(nx, "_confirm", lambda *a, **k: True)
    monkey.setattr(nx._bash_jobs, "_confirm", lambda *a, **k: True)
    try:
        # a mutating op whose ONLY staged path is this session's own WIP:
        # no prompt at all (this is the false positive being removed)
        out2 = nx.run_bash("git add wip.py", timeout=20, _cwd=str(repo))
        assert prompts == [] and "[exit 0]" in out2, (prompts, out2)
        assert staged_index_paths(str(repo)) == ["wip.py"]
        # a foreign staged path arriving afterwards still prompts…
        _stage_foreign(repo)
        out3 = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert len(prompts) == 1, prompts
    assert "foreign.py" in prompts[0] and "wip.py" not in prompts[0]
    assert "[exit 0]" in out3, out3


def test_raw_git_add_then_later_op_does_not_prompt(nx_fresh, repo, no_claims):
    """run_bash's own `git add foo` is claimed, so a later git op asks about
    foreign staged work only. `foo` staged BEFORE the add (by someone else)
    still prompts — the diff, not the command, decides attribution."""
    nx, _tmp = nx_fresh
    _stage_foreign(repo)  # present before this session staged anything
    (repo / "foo.py").write_text("f\n", encoding="utf-8")
    prompts = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(nx, "_ask_approval",
                   lambda prompt, allow_hint=None: prompts.append(prompt) or "y")
    monkey.setattr(nx, "_confirm", lambda *a, **k: True)
    monkey.setattr(nx._bash_jobs, "_confirm", lambda *a, **k: True)
    try:
        out = nx.run_bash("git add foo.py", timeout=20, _cwd=str(repo))
        assert "[exit 0]" in out, out
        # the pre-existing foreign entry is unclaimed → this add itself prompts
        assert len(prompts) == 1 and "foreign.py" in prompts[0], prompts
        assert "foo.py" not in prompts[0]
        # a LATER mutating op: foo.py is this session's, foreign.py is not
        prompts.clear()
        out2 = nx.run_bash("git commit -m mine", timeout=20, _cwd=str(repo))
    finally:
        monkey.undo()
    assert len(prompts) == 1, prompts
    assert "foreign.py" in prompts[0]
    assert "foo.py" not in prompts[0]
    assert "[exit 0]" in out2, out2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
