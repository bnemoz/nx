#!/usr/bin/env python3
"""Tests for the session store (`sessions.sqlite`, 0.101.0) that lets
`_list_sessions()` / `nx sessions [query]` list and search saved sessions
without parsing every full transcript. The transcripts live in the DB; the
per-session JSON files are a legacy mirror kept only as backup.

These pin the degradation paths (corrupt store, self-healing re-import)
that fall back gracefully.
"""
import os
import shutil
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


def test_listing_is_column_only_and_self_heals_null_rows():
    """0.129 regression: a listing never reads event logs (iter_rows /
    _read_events are not called), and pre-0.129 rows with NULL listing
    columns self-heal once — backfilled to the real values in the same
    listing call."""
    _clear_test_sessions_dir()
    sids = []
    for i in range(5):
        sid = f"20250101-1400{i:02d}-col{i:02d}"
        m._write_session(sid, [_msg("user", f"coltest {i}")],
                         {"updated_at": 300 + i})
        sids.append(sid)

    from nx import _sessiondb
    calls = []
    orig_read = _sessiondb._read_events

    def spy_read(conn, session_id):
        calls.append(session_id)
        return orig_read(conn, session_id)

    _sessiondb._read_events = spy_read
    try:
        sessions = m._list_sessions()
    finally:
        _sessiondb._read_events = orig_read

    assert len(sessions) == 5
    assert not calls, f"listing read {len(calls)} event logs: {calls}"
    assert sessions[0]["preview"] == "coltest 4"  # newest first
    assert sessions[0]["n"] == 1

    # Pre-0.129 row: manually NULL the listing columns, then list.
    victim = sids[2]
    conn = _sessiondb._open()
    try:
        conn.execute("UPDATE sessions SET preview = NULL, n_msg = NULL"
                     " WHERE session_id = ?", (victim,))
        conn.commit()
    finally:
        conn.close()

    _sessiondb._read_events = spy_read
    try:
        healed = m._list_sessions()
    finally:
        _sessiondb._read_events = orig_read

    row = [s for s in healed if s["id"] == victim][0]
    assert row["preview"] == "coltest 2"
    assert row["n"] == 1
    assert len(calls) == 1 and calls[0] == victim  # bounded: just that row
    # And the backfill persisted — a re-list is column-only again.
    after = [s for s in m._list_sessions() if s["id"] == victim][0]
    assert after["preview"] == "coltest 2" and after["n"] == 1


def test_missing_index_entry_falls_back_to_full_parse():
    """0.101.0: the listing IS the transcript table, so dropping a summary
    entry cannot lose the session — _write_index only drops ids the caller
    omitted, and this test pins that a reconstruction pass that omits one
    id removes exactly that row (the old 'fall back to a full parse' path
    has no file to fall back to)."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx004"
    other = "20250101-120000-idx040"
    m._write_session(sid, [_msg("user", "fallback preview")],
                      {"title": "fallback-title"})
    m._write_session(other, [_msg("user", "kept")], {"title": "kept-title"})
    idx = m._load_index()
    idx.pop(sid, None)
    m._write_index(idx)
    assert sid not in m._load_index()
    assert other in m._load_index()  # only the omitted id is dropped


def test_delete_session_removes_index_entry():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx005"
    m._write_session(sid, [_msg("user", "bye")])
    assert sid in m._load_index()
    assert m._delete_session(sid) is True
    assert sid not in m._load_index()


def test_missing_store_self_heals_from_mirror_files():
    """A deleted store is rebuilt: the legacy mirror files (written by an
    older nx, or kept as a backup) are re-imported by the lazy migration
    on the next listing/load."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx006"
    m._write_session(sid, [_msg("user", "still works")], {"title": "no-index"})
    # out-of-band: keep the legacy mirror file, drop the store
    import json as _json
    d = m._sessions_dir()
    with open(m._session_path(sid), "w", encoding="utf-8") as f:
        _json.dump({"id": sid, "title": "no-index", "description": None,
                    "created_at": 1.0, "updated_at": 2.0, "revision": 3,
                    "messages": [{"role": "user", "content": "still works"}]}, f)
    os.remove(m._sessiondb._store_path())
    m._sessiondb._MIGRATED.clear()

    sessions = m._list_sessions()
    match = [s for s in sessions if s["id"] == sid]
    assert len(match) == 1
    assert match[0]["title"] == "no-index"
    # a fresh store should now exist again after the next write
    m._write_session(sid, [_msg("user", "still works")], {"title": "no-index"})
    assert os.path.exists(m._sessiondb._store_path())


def test_corrupt_store_degrades_gracefully():
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx007"
    m._write_session(sid, [_msg("user", "resilient")], {"title": "resilient-title"})
    with open(m._sessiondb._store_path(), "w", encoding="utf-8") as f:
        f.write("not a sqlite database at all")

    sessions = m._list_sessions()  # degrades to empty, never raises
    assert isinstance(sessions, list)
    # the unusable file was set aside, not left in the way
    assert os.path.exists(m._sessiondb._store_path() + ".corrupt")
    # and the next write works on a fresh store
    m._write_session(sid, [_msg("user", "resilient")], {"title": "resilient-title"})
    assert m._load_session(sid) is not None


def test_legacy_shard_store_migrates_into_consolidated_db():
    """An upgraded ≤0.45 sessions dir (.index/ shards) still folds into the
    current store via the shard-migration path, and the shard whose
    transcript is gone is NOT imported as a live session."""
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
    assert os.path.exists(m._sessiondb._store_path())
    assert not os.path.isdir(m._index_dir())          # legacy shards retired
    idx = m._load_index()
    assert live_sid in idx and idx[live_sid]["title"] == "migrated"
    # the dead shard's transcript is gone (no mirror file), but its listing
    # summary survives as a stub — the user's history list doesn't lose rows
    assert dead_sid in idx and idx[dead_sid]["title"] == "ghost"
    # ...and it is not resumable content: the stub has no messages
    assert m._load_session(dead_sid)["messages"] == []
    # the FTS half came along: body search finds the migrated session
    from nx._sessions_search import search_sessions
    hits = search_sessions("fresh turn")
    assert [h["id"] for h in hits] == [live_sid]


def test_stale_fts_row_cannot_offer_deleted_session():
    """The 2.3a bug pinned: a body-search row for a session deleted out-of-
    band (crash, manual DB surgery) must not surface in /sessions <query>."""
    _clear_test_sessions_dir()
    sid = "20250101-120000-idx010"
    m._write_session(sid, [_msg("user", "quokka in the body")], {"title": "doomed"})
    from nx._sessions_fs import _prune_sessions_store
    from nx._sessions_search import search_sessions
    assert [h["id"] for h in search_sessions("quokka")] == [sid]
    # out-of-band deletion: transcript row dropped without /delete, FTS row left
    import sqlite3 as _sq
    conn = _sq.connect(m._sessiondb._store_path())
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    conn.commit()
    conn.close()
    stats = _prune_sessions_store()
    assert stats["rows_pruned"] == 1
    assert search_sessions("quokka") == []             # stale FTS row pruned
    assert sid not in m._load_index()                  # no transcript row either


def test_store_file_is_private():
    _clear_test_sessions_dir()
    m._write_session("20250101-120000-idx011", [_msg("user", "hi")])
    path = m._sessiondb._store_path()
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
