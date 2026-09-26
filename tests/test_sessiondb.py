#!/usr/bin/env python3
"""Tests for nx/_sessiondb.py — the 0.101.0 SQLite session store.

Pins: round-trip through the DB, lazy file→DB migration (idempotent,
files kept as backup), revision counting, corrupt-DB set-aside and
self-healing re-import, listing order, delete, prune.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-sessiondb-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _sessiondb as db  # noqa: E402


def _use_test_sessions_dir():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    return m._sessions_dir()


def setup_function(_function):
    _use_test_sessions_dir()


def _clear():
    d = _use_test_sessions_dir()
    assert os.path.basename(d).startswith("nx-sessiondb-"), d
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    db._MIGRATED.clear()
    return d


def _msg(role, content):
    return {"role": role, "content": content}


def test_store_session_roundtrip_and_revision():
    _clear()
    sid = "20250101-120000-dbaa1"
    assert db.store_session(sid, [_msg("user", "hello")], {"title": "hi"})
    data = db.load_session(sid)
    assert data["title"] == "hi"
    assert data["revision"] == 1
    assert data["messages"] == [_msg("user", "hello")]
    # second write bumps revision and keeps created_at
    db.store_session(sid, [_msg("user", "hello"), _msg("assistant", "yo")])
    data2 = db.load_session(sid)
    assert data2["revision"] == 2
    assert data2["created_at"] == data["created_at"]
    assert data2["updated_at"] >= data["updated_at"]


def test_store_session_merges_stale_writer():
    _clear()
    sid = "20250101-120000-dbaa2"
    base = [_msg("user", "start")]
    db.store_session(sid, base)
    db.store_session(sid, base + [_msg("assistant", "first")])
    # a stale snapshot that never saw "first" must not erase it
    db.store_session(sid, base + [_msg("user", "second")])
    data = db.load_session(sid)
    assert data["messages"] == base + [_msg("assistant", "first"),
                                       _msg("user", "second")]
    assert data["revision"] == 3


def test_migrate_files_imports_legacy_and_is_idempotent():
    _clear()
    d = _sessions_dir = _use_test_sessions_dir()
    legacy = {"20250101-120000-dbb1": [_msg("user", "old one")],
              "20250101-120000-dbb2": [_msg("user", "old two")]}
    for sid, msgs in legacy.items():
        with open(os.path.join(d, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"id": sid, "messages": msgs, "updated_at": 100.0,
                       "revision": 4, "title": f"t-{sid[-4:]}"}, f)
    n = db.migrate_files()
    assert n == 2
    # files are left in place (backup, never deleted)
    assert os.path.exists(os.path.join(d, "20250101-120000-dbb1.json"))
    # idempotent: a second pass imports nothing
    assert db.migrate_files(force=True) == 0
    # and the loaded transcripts match
    assert db.load_session("20250101-120000-dbb1")["messages"] == legacy[
        "20250101-120000-dbb1"]
    assert db.load_session("20250101-120000-dbb1")["revision"] == 4


def test_lazy_migration_runs_when_store_empty():
    """The self-heal path: mirror files present + empty store → next
    load/list imports them without an explicit migrate call."""
    _clear()
    d = _use_test_sessions_dir()
    sid = "20250101-120000-dbb5"
    with open(os.path.join(d, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump({"id": sid, "messages": [_msg("user", "resurrected")],
                   "updated_at": 5.0, "revision": 1}, f)
    assert m._load_session(sid) is not None
    assert m._load_session(sid)["messages"] == [_msg("user", "resurrected")]


def test_corrupt_store_set_aside_and_recovers():
    _clear()
    sid = "20250101-120000-dbc1"
    db.store_session(sid, [_msg("user", "keep me")], {"title": "t"})
    with open(db._store_path(), "w", encoding="utf-8") as f:
        f.write("not a database")
    assert db.load_session(sid) is None          # degrades, never raises
    assert os.path.exists(db._store_path() + ".corrupt")
    # fresh store accepts a new write
    assert db.store_session("20250101-120000-dbc2", [_msg("user", "fresh")])
    assert db.load_session("20250101-120000-dbc2")["revision"] == 1


def test_list_ids_newest_order():
    _clear()
    for i, ts in enumerate((300.0, 100.0, 200.0)):
        db.store_session(f"20250101-12000{i}-ord{i}", [_msg("user", f"s{i})")])
        # rewrite updated_at via a direct row edit to pin pure column order
        conn = sqlite3.connect(db._store_path())
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                     (ts, f"20250101-12000{i}-ord{i}"))
        conn.commit()
        conn.close()
    rows = db.list_ids_newest()
    assert [sid for sid, _ in rows] == [
        "20250101-120000-ord0", "20250101-120002-ord2", "20250101-120001-ord1"]


def test_delete_session_row():
    _clear()
    sid = "20250101-120000-dbd1"
    db.store_session(sid, [_msg("user", "bye")])
    assert db.load_session(sid) is not None
    assert db.delete_session_row(sid) is True
    assert db.load_session(sid) is None


def test_refresh_fts_and_prune_stale():
    _clear()
    sid = "20250101-120000-dbe1"
    db.store_session(sid, [_msg("user", "quokka habitat")], {"title": "q"})
    db.refresh_fts(sid, [_msg("user", "quokka habitat")], title="q")
    conn = sqlite3.connect(db._store_path())
    n = conn.execute("SELECT COUNT(*) FROM sessions_fts").fetchone()[0]
    conn.close()
    assert n == 1
    # transcript row gone → the FTS row is stale and prune removes it
    conn = sqlite3.connect(db._store_path())
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    conn.commit()
    conn.close()
    assert db.prune_stale() == 1
    conn = sqlite3.connect(db._store_path())
    n = conn.execute("SELECT COUNT(*) FROM sessions_fts").fetchone()[0]
    conn.close()
    assert n == 0


def test_store_file_private():
    _clear()
    db.store_session("20250101-120000-dbf1", [_msg("user", "hi")])
    mode = os.stat(db._store_path()).st_mode & 0o777
    assert mode == 0o600, oct(mode)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
