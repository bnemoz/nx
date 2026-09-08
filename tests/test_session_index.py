#!/usr/bin/env python3
"""Tests for the consolidated session store index (`.store.sqlite`, 2.3b)
that lets `_list_sessions()` / `nx sessions [query]` list and search saved
sessions without json.load-ing every full transcript on disk.

2.3b folded the per-session `.index/<sid>.json` shards and the lazy
`.search.sqlite` FTS cache into one SQLite db written by every indexed
save — these tests pin the surface `_load_index`/`_write_index` kept and
the degradation paths (corrupt store, missing rows) that fall back to a
full transcript parse.
"""
import os
import shutil
import sqlite3
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-session-index-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m


def _use_test_sessions_dir():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    return m._sessions_dir()


def setup_function(_function):
    _use_test_sessions_dir()


def _clear_test_sessions_dir():
    d = _use_test_sessions_dir()
    assert os.path.basename(d).startswith("nx-session-index-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    return d


def _msg(role, content):
    return {"role": role, "content": content}


def test_write_session_populates_index_entry():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx001"
    m._write_session(sid, [_msg("user", "hello there")], {"title": "greeting"})
    idx = m._load_index()
    assert sid in idx
    assert idx[sid]["title"] == "greeting"
    assert idx[sid]["preview"] == "hello there"
    assert idx[sid]["n"] == 1


def test_index_dir_is_excluded_from_session_listing():
    _clear_test_sessions_dir()
    m._write_session("20250101-120000-idx002", [_msg("user", "hi")])
    files = m._session_files_newest()
    assert all(not f.startswith(".") for f in files)
    assert ".index" not in files


def test_list_sessions_uses_index_even_if_file_is_corrupted():
    """The whole point of the index: listing must not require a full parse
    of every transcript. Prove it by corrupting the on-disk file directly
    (bypassing nx) after the index has a valid entry, and checking the
    listing is unaffected."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx003"
    m._write_session(sid, [_msg("user", "important preview text")],
                      {"title": "corruption-proof"})
    path = m._session_path(sid)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json at all")

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "corruption-proof"
    assert match[0]["preview"] == "important preview text"


def test_missing_index_entry_falls_back_to_full_parse():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx004"
    m._write_session(sid, [_msg("user", "fallback preview")],
                      {"title": "fallback-title"})
    # Simulate an index written by an older nx (or lost/corrupted) by
    # dropping this session's entry.
    idx = m._load_index()
    idx.pop(sid, None)
    m._write_index(idx)

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "fallback-title"
    assert match[0]["preview"] == "fallback preview"


def test_delete_session_removes_index_entry():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx005"
    m._write_session(sid, [_msg("user", "bye")])
    assert sid in m._load_index()
    assert m._delete_session(sid) is True
    assert sid not in m._load_index()


def test_missing_index_dir_degrades_gracefully():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx006"
    m._write_session(sid, [_msg("user", "still works")], {"title": "no-index"})
    os.remove(m._storeindex._store_path())

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "no-index"
    # a fresh store should now exist again after the next write
    m._write_session(sid, [_msg("user", "still works")], {"title": "no-index"})
    assert os.path.exists(m._storeindex._store_path())


def test_corrupt_store_degrades_gracefully():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx007"
    m._write_session(sid, [_msg("user", "resilient")], {"title": "resilient-title"})
    with open(m._storeindex._store_path(), "w", encoding="utf-8") as f:
        f.write("not a sqlite database at all")

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "resilient-title"
    # the unusable file was set aside, not left in the way
    assert os.path.exists(m._storeindex._store_path() + ".corrupt")


def test_legacy_shard_store_migrates_into_consolidated_db():
    """An upgraded ≤0.45 sessions dir (`.index/` shards, `.search.sqlite`)
    is folded into .store.sqlite on the first indexed save/list, the stale
    shard whose transcript is gone is NOT imported, and both legacy files
    are retired."""
    _clear_test_sessions_dir()
    d = m._sessions_dir()
    live_sid = "20250101-120000-idx008"
    dead_sid = "20250101-120000-idx009"
    os.makedirs(m._index_dir(), exist_ok=True)
    for sid, title in ((live_sid, "migrated"), (dead_sid, "ghost")):
        with open(m._index_entry_path(sid), "w", encoding="utf-8") as f:
            import json
            json.dump({"id": sid, "short": sid.rsplit("-", 1)[-1], "title": title,
                       "description": None, "preview": "p", "updated_at": 1.0,
                       "n": 1, "model": None, "source": None, "cwd": None,
                       "sha256s": []}, f)
    # only the live session still has a transcript
    m._write_session(live_sid, [_msg("user", "fresh turn")], {"title": "migrated"})
    assert os.path.exists(os.path.join(d, ".store.sqlite"))
    assert not os.path.isdir(m._index_dir())          # legacy shards retired
    idx = m._load_index()
    assert live_sid in idx and idx[live_sid]["title"] == "migrated"
    assert dead_sid not in idx                         # no transcript → not imported as live
    # the FTS half came along: body search finds the migrated session
    from nx._sessions_search import search_sessions
    hits = search_sessions("fresh turn")
    assert [h["id"] for h in hits] == [live_sid]


def test_stale_fts_row_cannot_offer_deleted_session():
    """The 2.3a bug pinned: a body-search row for a session deleted out-of-
    band (crash, manual rm) must not surface in /sessions <query>."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx010"
    m._write_session(sid, [_msg("user", "quokka in the body")], {"title": "doomed"})
    from nx._sessions_fs import _prune_sessions_store
    from nx._sessions_search import search_sessions
    assert [h["id"] for h in search_sessions("quokka")] == [sid]
    # out-of-band deletion: transcript removed without /delete
    os.remove(m._session_path(sid))
    stats = _prune_sessions_store()
    assert stats["rows_pruned"] == 1
    assert search_sessions("quokka") == []             # stale FTS row pruned
    assert sid not in m._load_index()                  # stale listing row pruned


def test_store_file_is_private():
    _clear_test_sessions_dir()
    m._write_session("20250101-120000-idx011", [_msg("user", "hi")])
    path = m._storeindex._store_path()
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600, oct(mode)
    assert os.stat(os.path.dirname(path)).st_mode & 0o777 == 0o700


def test_rollback_flag_does_not_stop_the_store(monkeypatch):
    """NX_SESSION_IMAGES=0 restores inline storage; the listing/store index
    must keep working under it (it is not an image-store flag)."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx012"
    monkeypatch.setenv("NX_SESSION_IMAGES", "0")
    m._write_session(sid, [_msg("user", "inline now")], {"title": "inline"})
    assert m._load_index().get(sid, {}).get("title") == "inline"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
