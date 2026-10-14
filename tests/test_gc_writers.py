"""Phase-3 artifact-GC, writer side: the four leak fixes that live in the
modules that WRITE the artifacts.

Each fix keeps the file that produces an artifact bounded at the point of
writing, rather than relying on a separate sweeper:

1. `nx/_webfetch.py`  — web-cache/cache.jsonl is rewritten on every save; over
   `NX_WEBCACHE_MAX_MB` the rewrite applies the module's own size policy
   (expired entries first, then newest-first) in memory AND on the file it is
   about to replace, so the file cannot outgrow the cap.
2. `nx/_logging.py`   — usage.jsonl now rotates to usage.jsonl.1 (replacing an
   older .1) under `NX_USAGE_LOG_MAX_MB`, using the writer's existing policy.
3. `nx/_undo.py`      — the persisted undo file of a deleted session is
   removed by `_sessions_fs._delete_session`, the single deletion entry point
   `/delete` and the fullscreen browser both funnel through.
4. `nx/_worktree.py`  — the `*-dirty` trees `remove_worktree` renames aside get
   their own (longer) TTL so the sweep eventually reaps them, and a clean
   `remove_session_worktree` deletes the session's `nx/session-<sid>` branch.

No network. Everything points NX state at a tmp dir and monkeypatches the
module globals the writer reads.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

import nx
import nx._state as st
from nx import _logging as L
from nx import _undo as U
from nx import _webfetch as W
from nx import _worktree as WT

TTL_7D = 7 * 24 * 3600


# --- 1. web-cache size policy (nx/_webfetch.py) --------------------------------

def _cache(text: str, age_s: float) -> tuple[float, str, str, str]:
    """An in-memory `_CACHE` value: (monotonic ts, text, etag, last-modified)."""
    return (time.monotonic() - age_s, text, "", "")


def _entries_bytes(*keyed_texts: tuple[str, str]) -> int:
    """Exact encoded size of the given (key, text) entries, via the module's
    own counter — the only honest way to set a byte cap in a test."""
    entries = [(k, time.time(), t, "", "") for k, t in keyed_texts]
    return W._cache_entries_size(entries)


def test_webcache_save_drops_expired_entries_over_cap(tmp_path, monkeypatch):
    """Over the cap, the save keeps only what the loader would have kept: the
    expired entry leaves both the file and the in-memory cache."""
    fresh = {"new1": _cache("y" * 20, 0), "new2": _cache("z" * 20, 0)}
    cache = {"stale": _cache("x" * 20, 10 * 24 * 3600), **fresh}
    cap = _entries_bytes(("new1", "y" * 20), ("new2", "z" * 20)) + 5
    monkeypatch.setattr(W, "_CACHE", dict(cache))
    monkeypatch.setattr(W, "CACHE_TTL_SECS", TTL_7D)
    monkeypatch.setattr(W, "WEB_CACHE_MAX_BYTES", cap)
    monkeypatch.setattr(W, "_cache_dir", lambda: str(tmp_path / "web-cache"))

    W._save_cache()

    text = (tmp_path / "web-cache" / "cache.jsonl").read_text()
    assert "stale" not in text, "expired entry survived an over-cap save"
    assert "new1" in text and "new2" in text
    assert set(W._CACHE) == {"new1", "new2"}, \
        "the dropped entry must leave _CACHE or the next save rewrites it"


def test_webcache_save_keeps_everything_under_cap(tmp_path, monkeypatch):
    cache = {"stale": _cache("x" * 20, 10 * 24 * 3600),
             "new": _cache("y" * 20, 0)}
    monkeypatch.setattr(W, "_CACHE", dict(cache))
    monkeypatch.setattr(W, "CACHE_TTL_SECS", TTL_7D)
    monkeypatch.setattr(W, "WEB_CACHE_MAX_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(W, "_cache_dir", lambda: str(tmp_path / "web-cache"))

    W._save_cache()

    text = (tmp_path / "web-cache" / "cache.jsonl").read_text()
    # Under the cap there is no trimming at all — even the expired entry is
    # written (the loader is what drops it, on the next start).
    assert "stale" in text and "new" in text


def test_webcache_truncates_newest_first_when_all_fresh():
    entries = [("k%d" % i, time.time() - i * 60, "v" * 20, "", "") for i in range(4)]
    cap = _entries_bytes(("k0", "v" * 20), ("k1", "v" * 20)) + 5
    kept = W._trim_cache_entries(list(entries), cap, TTL_7D)
    assert [e[0] for e in kept] == ["k0", "k1"], "kept entries must be the NEWEST"
    assert W._cache_entries_size(kept) <= cap


def test_webcache_max_mb_zero_disables_the_policy(tmp_path, monkeypatch):
    entry = ("k0", time.time() - 99 * 24 * 3600, "v" * 20, "", "")
    path = tmp_path / "cache.jsonl"
    path.write_text("\n".join(W._cache_line(entry) for _ in range(10)) + "\n")
    before = path.read_bytes()
    monkeypatch.setattr(W, "WEB_CACHE_MAX_BYTES", 0)

    assert W._trim_cache_entries([entry], 0, TTL_7D) == [entry]  # 0 = disabled
    W._trim_persisted_cache(str(path))
    assert path.read_bytes() == before, "cap 0 must leave the file alone"


def test_webcache_file_trim_clears_a_fully_expired_file(tmp_path, monkeypatch):
    """An over-cap file whose every entry is past the TTL holds nothing the
    loader would keep — it must be emptied, not left as dead weight."""
    old = ("gone", time.time() - 30 * 24 * 3600, "x" * 40, "", "")
    path = tmp_path / "cache.jsonl"
    path.write_text("\n".join(W._cache_line(old) for _ in range(5)) + "\n")
    monkeypatch.setattr(W, "CACHE_TTL_SECS", TTL_7D)
    monkeypatch.setattr(W, "WEB_CACHE_MAX_BYTES", 10)

    W._trim_persisted_cache(str(path))

    assert path.read_text() == ""


def test_webcache_file_trim_leaves_an_entirely_fresh_over_cap_file(tmp_path, monkeypatch):
    """cache.jsonl can hold entries this process never loaded (a second session
    appended them). The file-level trim must bound those on its own."""
    stale = ("old", time.time() - 10 * 24 * 3600, "x" * 20, "", "")
    current = ("cur", time.time(), "y" * 20, "", "")
    path = tmp_path / "cache.jsonl"
    path.write_text("\n".join(W._cache_line(e) for e in (stale, current)) + "\n")
    cap = W._cache_entries_size([current]) + 5
    monkeypatch.setattr(W, "CACHE_TTL_SECS", TTL_7D)
    monkeypatch.setattr(W, "WEB_CACHE_MAX_BYTES", cap)
    assert path.stat().st_size > cap

    W._trim_persisted_cache(str(path))

    text = path.read_text()
    assert "old" not in text and "cur" in text
    assert path.stat().st_size <= cap


# --- 2. usage.jsonl rotation (nx/_logging.py) ----------------------------------

def test_usage_log_rotates_past_max_bytes(tmp_path):
    path = str(tmp_path / "usage.jsonl")
    w = L._AsyncLogWriter(path, max_bytes=10)
    w.write("0123456789\n")   # 11 bytes: at/over the cap on the NEXT write
    w.write("next-line\n")
    w.flush()
    assert os.path.exists(path + ".1")
    with open(path + ".1") as f:
        assert f.read() == "0123456789\n"
    with open(path) as f:
        assert f.read() == "next-line\n"


def test_usage_rotation_replaces_older_backup(tmp_path):
    path = tmp_path / "usage.jsonl"
    path.with_name("usage.jsonl.1").write_text("ancient\n")
    path.write_text("0123456789\n")
    w = L._AsyncLogWriter(str(path), max_bytes=10)
    w.write("next-line\n")
    w.flush()
    with open(str(path) + ".1") as f:
        assert f.read() == "0123456789\n", "the older .1 must be replaced, not kept"


def test_usage_log_is_wired_to_its_own_mb_cap():
    """The usage stream rotates under NX_USAGE_LOG_MAX_MB, not the request
    log's NX_LOG_MAX_BYTES — the two writers must not share one cap."""
    assert L.USAGE_PATH.endswith(os.path.join("logs", "usage.jsonl"))
    assert L._ulog._path == L.USAGE_PATH
    assert L._ulog._max_bytes == L.USAGE_MAX_BYTES == 5 * 1024 * 1024
    assert L.LOG_MAX_BYTES == L._env_int("NX_LOG_MAX_BYTES", 10000000)


def test_usage_log_max_mb_zero_disables_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_USAGE_LOG_MAX_MB", "0")
    assert L._env_int("NX_USAGE_LOG_MAX_MB", 5) * 1024 * 1024 == 0
    path = str(tmp_path / "usage.jsonl")
    w = L._AsyncLogWriter(path, max_bytes=0)
    for i in range(5):
        w.write(f"line{i}\n")
    w.flush()
    assert not os.path.exists(path + ".1")
    with open(path) as f:
        assert f.read().count("\n") == 5


# --- 3. undo file GC on session delete (nx/_undo.py + _sessions_fs) -----------

def test_undo_file_removed_on_session_delete(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "SESSION_ID", None)
    sid = "20260101-000000-abc123"
    undo_dir = home / "undo"
    undo_dir.mkdir(parents=True)
    dead = undo_dir / f"{sid}.json"
    dead.write_text('{"version": 1, "entries": []}')
    keeper = undo_dir / "someone-else.json"
    keeper.write_text("{}")

    from nx import _sessions_fs
    _sessions_fs._delete_session(sid)

    assert not dead.exists(), "the deleted session's undo snapshots survived"
    assert keeper.exists(), "another session's undo file must not be touched"


def test_delete_persisted_for_targets_the_dead_sid(tmp_path, monkeypatch):
    """The per-sid keying the /delete hook relies on (not st.SESSION_ID)."""
    home = tmp_path / "home"
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "SESSION_ID", "live-session")
    (home / "undo").mkdir(parents=True)
    dead = home / "undo" / "dead-sid.json"
    dead.write_text("{}")
    U.delete_persisted_for("dead-sid")
    assert not dead.exists()
    assert not (home / "undo" / "live-session.json").exists()


def test_orphaned_undo_files_gc(tmp_path, monkeypatch):
    """The batch form (for deletion paths that do NOT go through
    `_delete_session`: the retention sweep, a hand rm, a crash) removes exactly
    the files whose session the store no longer holds."""
    home = tmp_path / "home"
    undo_dir = home / "undo"
    undo_dir.mkdir(parents=True)
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    for sid in ("live-1", "live-2", "dead-1", "dead-2"):
        (undo_dir / f"{sid}.json").write_text("{}")
    (undo_dir / "notes.txt").write_text("not an undo file")

    removed = U.gc_orphaned_undo_files(["live-1", "live-2"])

    assert removed == 2
    assert sorted(p.name for p in undo_dir.iterdir()) == [
        "live-1.json", "live-2.json", "notes.txt"]


def test_orphaned_undo_gc_missing_dir_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path / "absent"))
    assert U.gc_orphaned_undo_files([]) == 0
    assert U.gc_orphaned_undo_files(["a"]) == 0


def test_delete_session_is_the_hook_for_undo_gc():
    """Contract guard: the single deletion entry point both `/delete` and the
    fullscreen browser call must keep routing into `_undo`. If a refactor drops
    the import, this fails here instead of silently leaking snapshots."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "nx" / "_sessions_fs.py").read_text()
    body = src.split("def _delete_session(", 1)[1].split("\ndef ", 1)[0]
    assert "_undo.delete_persisted_for(session_id)" in body, \
        "_delete_session no longer GCs the dead session's undo file"


# --- 4a. dirty worktree TTL (nx/_worktree.py) ---------------------------------

def _agent_tree(root, name: str, age_hours: float, content: str = "wip\n"):
    d = root / name
    (d / "nx").mkdir(parents=True)
    (d / "nx" / "work.txt").write_text(content)
    old = time.time() - age_hours * 3600
    os.utime(d, (old, old))
    return d


@pytest.fixture()
def sweep_home(tmp_path, monkeypatch):
    """A tmp SESSION_HOME with the real slug layout, no git repo in play."""
    home = tmp_path / "home"
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "PROJECT_ROOT", None)   # skips the git prune step
    monkeypatch.delenv("NX_SUBAGENT_WORKTREE_TTL_HOURS", raising=False)
    monkeypatch.delenv("NX_SUBAGENT_DIRTY_WORKTREE_TTL_HOURS", raising=False)
    root = home / "worktrees" / nx.project_store_slug()
    root.mkdir(parents=True)
    return root


def test_dirty_worktree_reaped_past_its_own_ttl(sweep_home):
    reaped = _agent_tree(sweep_home, "nx-wt-agent-1-aaaa-dirty", 100)
    kept = _agent_tree(sweep_home, "nx-wt-agent-2-bbbb-dirty", 1)
    WT.sweep_stale_worktrees()
    assert not reaped.exists(), "a *-dirty tree past 72h must be removed"
    assert kept.exists(), "a fresh *-dirty tree must survive"


def test_dirty_worktree_outlives_the_live_tree_ttl(sweep_home):
    """The dirty TTL is deliberately LONGER: a tree between 24h and 72h old is
    still someone's unrecovered work, and remove_worktree would only rename it
    aside again (so it would never be reaped at all)."""
    between = _agent_tree(sweep_home, "nx-wt-agent-3-cccc-dirty", 30)
    WT.sweep_stale_worktrees()
    assert between.exists(), "dirty work must outlive NX_SUBAGENT_WORKTREE_TTL_HOURS"


def test_dirty_ttl_zero_disables_the_sweep(sweep_home, monkeypatch):
    monkeypatch.setenv("NX_SUBAGENT_DIRTY_WORKTREE_TTL_HOURS", "0")
    d = _agent_tree(sweep_home, "nx-wt-agent-4-dddd-dirty", 400)
    WT.sweep_stale_worktrees()
    assert d.exists(), "0 = never sweep dirty trees"


def test_live_agent_tree_ttl_unchanged(sweep_home):
    stale = _agent_tree(sweep_home, "nx-wt-agent-5-eeee", 30)
    fresh = _agent_tree(sweep_home, "nx-wt-agent-6-ffff", 1)
    WT.sweep_stale_worktrees()
    assert not stale.exists(), "a live tree past 24h is still reaped"
    assert fresh.exists()


def test_live_ttl_zero_still_disables_the_whole_sweep(sweep_home, monkeypatch):
    """The pre-existing contract survives the dirty-TTL addition: 0 on the
    live-tree TTL means no sweep at all, dirty trees included."""
    monkeypatch.setenv("NX_SUBAGENT_WORKTREE_TTL_HOURS", "0")
    live = _agent_tree(sweep_home, "nx-wt-agent-7-aaaa", 400)
    dirty = _agent_tree(sweep_home, "nx-wt-agent-8-bbbb-dirty", 400)
    WT.sweep_stale_worktrees()
    assert live.exists() and dirty.exists()


# --- 4b. session branch deletion on clean removal -----------------------------

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A real tmp git repo + tmp session home, the pattern the existing
    worktree tests use (they exercise real `git worktree`/`git branch`)."""
    home = tmp_path / "nxhome"
    monkeypatch.setenv("NX_HOME", str(home))
    monkeypatch.setattr(st, "SESSION_HOME", str(home))
    monkeypatch.setattr(st, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(st, "PROJECT_ROOT_IS_GIT", True)
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path,
                   check=True, capture_output=True)
    for k, v in (("user.email", "nx@example.com"), ("user.name", "nx")):
        subprocess.run(["git", "config", k, v], cwd=tmp_path, check=True,
                       capture_output=True)
    (tmp_path / "a.txt").write_text("base\n")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path,
                   check=True, capture_output=True)
    return tmp_path


def test_session_branch_deleted_on_clean_removal(repo):
    wt, err = WT.create_session_worktree("sid-branch")
    assert err is None and wt
    assert _git(repo, "branch", "--list", "nx/session-sid-branch").stdout.strip()

    WT.remove_session_worktree(wt)

    assert not os.path.isdir(wt)
    assert _git(repo, "branch", "--list", "nx/session-sid-branch").stdout.strip() == "", \
        "the session branch leaked on a clean exit"


def test_session_branch_deleted_for_detached_fallback(repo):
    """create_session_worktree's fallback checks the tree out DETACHED (reusing
    a stale branch), so `rev-parse --abbrev-ref HEAD` reports nothing — the
    branch name must come from the path, or it leaks forever."""
    branch = "nx/session-sid-det"
    assert _git(repo, "branch", branch).returncode == 0
    home = st.SESSION_HOME
    wt = os.path.join(home, "worktrees", nx.project_store_slug(),
                      "nx-wt-session-sid-det")
    os.makedirs(os.path.dirname(wt), exist_ok=True)
    assert _git(repo, "worktree", "add", "--detach", wt, branch).returncode == 0
    assert WT._branch_of(wt) is None, "precondition: the tree is detached"

    WT.remove_session_worktree(wt)

    assert not os.path.isdir(wt)
    assert _git(repo, "branch", "--list", branch).stdout.strip() == ""


def test_session_branch_name_derived_from_path():
    assert WT._session_branch_for("/tmp/x/nx-wt-session-sid-9") == "nx/session-sid-9"
    assert WT._session_branch_for("/tmp/x/nx-wt-agent-1-aaaa") == ""
    assert WT._session_branch_for("/tmp/x/nx-wt-session-") == ""


def test_dirty_session_tree_is_still_never_destroyed(repo):
    """The branch deletion must not weaken the dirty-tree refusal."""
    wt, _ = WT.create_session_worktree("sid-dirty")
    with open(os.path.join(wt, "dirty.txt"), "w") as f:
        f.write("unsaved\n")
    WT.remove_session_worktree(wt)
    assert os.path.isdir(wt)
    assert _git(repo, "branch", "--list", "nx/session-sid-dirty").stdout.strip()
