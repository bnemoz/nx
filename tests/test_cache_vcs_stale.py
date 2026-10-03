"""Regression tests: git_* cache entries must track actual git state.

git_status/git_diff/git_log are cacheable reads whose backing path is the repo
root directory. Nothing git does — commit, checkout, reset — mutates that
directory's own mtime (git writes under .git/), and the path-based invalidation
on write_file/edit_file drops only entries whose stored path == the edited
file. So a cached "clean working tree" was served for the rest of the session:
after edits, after run_bash("git commit -am …"), after anything. Entries now
carry a git_state_stamp (mtimes of .git/HEAD, index, ORIG_HEAD) checked on
every hit, and are dropped outright by any edit tool or run_bash.
"""
import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True
    yield nx, tmp_path
    nx.YOLO = False
    nx._state._TOOL_RESULT_CACHE.clear()
    nx._state._SESSION_HAS_BACKGROUNDED = False
    nx._state._PENDING_BACKGROUND_NOTICE = None
    for k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(k, None)


def _git(repo, *args):
    env = dict(os.environ)
    env.setdefault("GIT_AUTHOR_NAME", "t")
    env.setdefault("GIT_AUTHOR_EMAIL", "t@example.com")
    env.setdefault("GIT_COMMITTER_NAME", "t")
    env.setdefault("GIT_COMMITTER_EMAIL", "t@example.com")
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, env=env)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _count_calls(monkeypatch, nx, tool):
    """Wrap DISPATCH[tool] to count body executions."""
    from nx import _dispatch as disp
    calls = []
    real = disp.DISPATCH[tool]

    def counting(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setitem(disp.DISPATCH, tool, counting)
    return calls


def _git_dir(path) -> str:
    """The effective git dir for *path*. In a linked worktree `.git` is a
    `gitdir: …` pointer file rather than a directory."""
    git = os.path.join(str(path), ".git")
    if os.path.isfile(git):
        with open(git) as fh:
            line = fh.readline().strip()
        if line.startswith("gitdir:"):
            target = line[len("gitdir:"):].strip()
            git = target if os.path.isabs(target) else os.path.join(str(path), target)
    return git


# Offset used to force an mtime onto a different tick than the stored one.
# 10s clears the coarsest granularity any real filesystem uses (FAT records
# mtime with 2s resolution, HFS+ with 1s), so the value cannot be truncated
# back onto the tick the entry already stored — which is why the smaller sleeps
# these tests used to take were not enough on such a mount.
_MTIME_BUMP_NS = 10_000_000_000


def _force_stamp_move(path) -> None:
    """Move .git/HEAD's mtime to a different tick than the one the cache stored.

    HEAD is always part of git_state_stamp, so bumping it changes the stamp for
    sure. Only used as a fallback when a real git write landed inside a single
    mtime tick of the stored stamp (a filesystem coarser-grained than the sleep
    these tests used to take): the old 200x5ms poll loop waited on a condition
    such a clock may never satisfy, while the assert after it still had to hold,
    which is exactly how these flaked. Driving the mtime with os.utime removes
    the clock-granularity dependency."""
    head = os.path.join(_git_dir(path), "HEAD")
    t = os.stat(head).st_mtime_ns - _MTIME_BUMP_NS
    os.utime(head, ns=(t, t))


def test_write_file_invalidates_cached_git_status(nx_fresh, monkeypatch):
    """Cached git_status must not survive a write_file: the entry is keyed on
    the repo root dir, whose mtime never moves, so only explicit invalidation
    catches this."""
    nx, tmp = nx_fresh
    repo = _git_repo(tmp)
    from nx._dispatch import run_tool

    first = run_tool("git_status", {"path": str(repo)})
    assert "a.py" not in first
    key = ("git_status", '{"path": "' + str(repo) + '"}')
    assert key in nx._state._TOOL_RESULT_CACHE

    calls = _count_calls(monkeypatch, nx, "git_status")
    run_tool("write_file", {"path": str(repo / "a.py"), "content": "x = 1\n"})
    assert calls == [], "write_file should not re-run git_status"

    second = run_tool("git_status", {"path": str(repo)})
    assert second != first, "stale git_status served after write_file"
    assert "a.py" in second, f"unexpected fresh status: {second!r}"
    assert len(calls) == 1, "git_status body did not re-run after write_file"


def test_run_bash_commit_invalidates_cached_git_log(nx_fresh, monkeypatch):
    """run_bash("git commit …") must drop cached git_log: the stamp sees the
    commit, but the explicit VCS drop must too (stamp can miss, e.g. .git
    removed, and drop_changed_entries only re-stats the root dir)."""
    _nx, tmp = nx_fresh
    repo = _git_repo(tmp)
    from nx._dispatch import run_tool

    first = run_tool("git_log", {"path": str(repo)})
    assert "seed" in first

    run_tool("run_bash", {"command": "git -C " + str(repo) + " commit --allow-empty -m second"})
    second = run_tool("git_log", {"path": str(repo)})
    assert "second" in second, f"stale git_log served after a commit: {second!r}"
    assert second != first


def test_git_stamp_detects_commit_behind_cache(nx_fresh):
    """Stamp-level regression: HEAD's mtime moves on commit, so a cached
    git_status entry whose stamp predates a commit is invalid on the next hit
    even if no run_bash/edit went through run_tool (e.g. the user ran git in
    another terminal)."""
    _nx, tmp = nx_fresh
    repo = _git_repo(tmp)
    from nx import _state as st
    from nx._dispatch import _tool_cache_key, run_tool
    from nx._tool_cache import cache_valid, git_state_stamp

    run_tool("git_status", {"path": str(repo)})
    key = _tool_cache_key("git_status", {"path": str(repo)})
    entry = st._TOOL_RESULT_CACHE[key]
    assert entry.git_stamp is not None, "git_status entry stored without a stamp"
    assert cache_valid(entry), "freshly stored entry should be valid"

    _git(repo, "commit", "--allow-empty", "-m", "external")
    # the commit moves a watched mtime on a fine-grained clock; if the fs wrote
    # it in the same tick as the stored stamp, force it forward instead of
    # polling for a change that may never arrive
    if git_state_stamp(str(repo)) == entry.git_stamp:
        _force_stamp_move(repo)
    assert git_state_stamp(str(repo)) != entry.git_stamp, "test void: stamp did not move"
    assert not cache_valid(entry), "stamp unchanged across a commit"

    fresh = run_tool("git_status", {"path": str(repo)})
    assert "??" not in fresh, f"unexpected untracked output after empty commit: {fresh!r}"


def test_run_bash_edit_does_not_touch_git_status_of_untouched_repo(nx_fresh):
    """The explicit VCS drop is unconditional; this guards the flip side — a
    plain run_bash with no git still drops git_* entries (cheap, and no worse
    than the old whole-cache clear), while a read_file entry whose file is
    untouched stays (the pre-existing behavior this cache exists for)."""
    _nx, tmp = nx_fresh
    repo = _git_repo(tmp)
    f = tmp / "plain.txt"
    f.write_text("plain\n")
    from nx import _state as st
    from nx._dispatch import _tool_cache_key, run_tool

    run_tool("git_status", {"path": str(repo)})
    run_tool("read_file", {"path": str(f)})
    git_key = _tool_cache_key("git_status", {"path": str(repo)})
    read_key = _tool_cache_key("read_file", {"path": str(f)})

    run_tool("run_bash", {"command": "true"})
    assert git_key not in st._TOOL_RESULT_CACHE, "run_bash kept a git_* entry"
    assert read_key in st._TOOL_RESULT_CACHE, "run_bash dropped an unchanged read_file entry"


def test_git_state_stamp_unit(nx_fresh, monkeypatch):
    """git_state_stamp: None outside a repo; a string inside one; unchanged
    without git activity; moved by os.utime on .git/HEAD."""
    _nx, tmp = nx_fresh
    from nx._tool_cache import git_state_stamp

    outside = tmp / "nogit"
    outside.mkdir()
    assert git_state_stamp(str(outside)) is None

    repo = _git_repo(tmp)
    stamp = git_state_stamp(str(repo))
    assert isinstance(stamp, str) and "HEAD:" in stamp
    # deep subdirectory walks up to the repo root
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    assert git_state_stamp(str(sub)) == stamp

    # drive HEAD's mtime to a different tick than the one `stamp` recorded (a
    # bare os.utime() stamps "now", which a coarse-granularity filesystem may
    # already have used for `stamp`)
    head = os.path.join(str(repo), ".git", "HEAD")
    t = os.stat(head).st_mtime_ns - _MTIME_BUMP_NS
    os.utime(head, ns=(t, t))
    assert git_state_stamp(str(repo)) != stamp
    # ORIG_HEAD doesn't exist in a fresh repo and is reported as "-"
    assert "ORIG_HEAD:-" in git_state_stamp(str(repo))


def test_git_state_stamp_worktree(nx_fresh):
    """A linked worktree's .git is a FILE ("gitdir: …"), so stamping it
    directly would yield a constant string that never detects a commit. The
    pointer must be resolved to the real gitdir first."""
    _nx, tmp = nx_fresh
    from nx._tool_cache import git_state_stamp

    main = _git_repo(tmp)
    wt = tmp / "wt"
    _git(main, "worktree", "add", "-q", str(wt), "-b", "wtbranch")
    stamp = git_state_stamp(str(wt))
    assert stamp is not None and "HEAD:-" not in stamp, (
        f"worktree stamp not resolved through the gitdir pointer: {stamp!r}"
    )

    _git(wt, "commit", "--allow-empty", "-m", "wt-commit")
    if git_state_stamp(str(wt)) == stamp:
        _force_stamp_move(wt)
    assert git_state_stamp(str(wt)) != stamp, "commit in a worktree did not move the stamp"


def test_read_file_entries_never_call_git_state_stamp(nx_fresh, monkeypatch):
    """Non-VCS entries are unaffected: storing/validating a read_file entry
    must not consult git state (a non-repo cwd would otherwise slow every
    read-cache hit)."""
    _nx, tmp = nx_fresh
    f = tmp / "f.txt"
    f.write_text("content\n")
    import nx._tool_cache as tc
    from nx._dispatch import run_tool

    def boom(path):
        raise AssertionError("git_state_stamp called for read_file")

    monkeypatch.setattr(tc, "git_state_stamp", boom)
    result = run_tool("read_file", {"path": str(f)})
    assert "content" in result
    assert run_tool("read_file", {"path": str(f)}) == result  # cache hit path


def test_prefetch_stamps_git_entries(nx_fresh):
    """The mixed-turn prefetch dispatches through run_tool(), so its git_*
    entries carry the same stamp run_tool stores (read_file keep git_stamp
    None) — one store path for both."""
    nx, tmp = nx_fresh
    repo = _git_repo(tmp)
    f = tmp / "a.txt"
    f.write_text("A\n")
    from nx._turn import _prefetch_cacheable_tools

    _prefetch_cacheable_tools(
        [{"name": "read_file"}, {"name": "git_status"}],
        [{"path": str(f)}, {"path": str(repo)}],
    )
    cache = nx._state._TOOL_RESULT_CACHE
    read_entry = cache[("read_file", '{"path": "%s"}' % f)]
    git_entry = cache[("git_status", '{"path": "%s"}' % repo)]
    assert read_entry.git_stamp is None
    assert git_entry.git_stamp is not None
    assert read_entry.mtime_ns == os.stat(str(f)).st_mtime_ns
    assert "A" in read_entry.result
