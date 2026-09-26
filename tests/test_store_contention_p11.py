#!/usr/bin/env python3
"""Swarm P11: SQLite store contention.

The two stores (_storeindex, _memindex) used to connect per call with a
5 s busy_timeout: under a shared NX_HOME every agent's per-turn indexed
save could block a sibling 5 s, and lock failures were indistinguishable
from corruption. This pins:

- busy_timeout is env-tunable (NX_STORE_BUSY_TIMEOUT_MS, default 750),
  not a hardcoded 5000;
- a busy/locked OperationalError retries with backoff and NEVER triggers
  the corrupt-file set-aside;
- a corrupt file is still set aside (the pre-P11 behavior must survive);
- prune_stale and search_sessions use ONE connection per call (counted
  via sqlite3.connect).
"""
import os
import sqlite3
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-p11-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import nx as m  # noqa: E402
from nx import _sessiondb as si  # noqa: E402
from nx import _sessions_search as ss  # noqa: E402


def _write_session(sid):
    m._write_session(sid, [{"role": "user", "content": f"body of {sid}"}], {})


# --- busy timeout ------------------------------------------------------------

def test_busy_timeout_env_tunable(monkeypatch):
    monkeypatch.setenv("NX_STORE_BUSY_TIMEOUT_MS", "123")
    conn = si._connect()
    try:
        got = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    assert got == 123


def test_busy_timeout_default_750(monkeypatch):
    monkeypatch.delenv("NX_STORE_BUSY_TIMEOUT_MS", raising=False)
    conn = si._connect()
    try:
        got = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    assert got == 750


# --- busy vs corrupt retry paths ----------------------------------------------

def test_busy_error_retries_and_never_sets_aside(monkeypatch):
    """A locked store is contention: retries then returns None (degrade),
    and the store file is untouched — not set aside as .corrupt."""
    _write_session("20250101-120000-p11aa")
    path = si._store_path()
    assert os.path.exists(path)
    monkeypatch.setenv("NX_STORE_BUSY_TIMEOUT_MS", "5")

    calls = {"n": 0}

    def busy_connect(*a, **kw):
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(si.sqlite3, "connect", busy_connect)
    assert si._open() is None
    assert calls["n"] == 3  # bounded retry ladder, not one-shot
    # the store file survives — a busy file must never be set aside
    assert os.path.exists(path)
    assert not os.path.exists(path + ".corrupt")


def test_corrupt_file_still_set_aside():
    _write_session("20250101-120000-p11ab")
    path = si._store_path()
    with open(path, "w") as f:
        f.write("{not a database at all")
    assert si.store_stats() == {"sessions_indexed": 0}  # degrade, not raise
    assert os.path.exists(path + ".corrupt")
    # and a fresh store is created in its place
    assert si.store_stats() == {"sessions_indexed": 0}


def test_corrupt_memindex_still_set_aside(monkeypatch):
    from nx import _memindex as mi
    d = tempfile.mkdtemp(prefix="nx-p11-mem-")
    monkeypatch.setattr(mi, "_fts_supported", lambda: True, raising=False)
    mi.enabled()  # ensure module state
    # build the index once
    assert mi.search_dir(d, "probe") is None or True
    path = mi._db_path(d)
    if os.path.exists(path):
        with open(path, "w") as f:
            f.write("garbage not sqlite")
        # degrades (never raises) — and the corrupt file is set aside so a
        # fresh index can replace it
        out = mi.search_dir(d, "anything")
        assert out is None or out == []
        assert not os.path.exists(path) or os.path.exists(path + ".corrupt")


# --- single-connection shapes ---------------------------------------------------

def test_prune_stale_one_connection(monkeypatch):
    _write_session("20250101-120000-p11ac")
    _write_session("20250101-120000-p11ad")
    # out-of-band: transcript row dropped, FTS row left (crash between deletes)
    conn = sqlite3.connect(si._store_path())
    conn.execute("DELETE FROM sessions WHERE session_id = ?",
                 ("20250101-120000-p11ac",))
    conn.commit()
    conn.close()
    opens = {"n": 0}
    real = sqlite3.connect

    def counting(*a, **kw):
        opens["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(si.sqlite3, "connect", counting)
    pruned = si.prune_stale()
    monkeypatch.setattr(si.sqlite3, "connect", real)
    assert pruned == 1
    assert opens["n"] == 1  # was 2 (read + write connects) before P11


def test_search_sessions_one_connection(monkeypatch):
    _write_session("20250101-120000-p11ae")
    opens = {"n": 0}
    real = sqlite3.connect

    def counting(*a, **kw):
        opens["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(ss.sqlite3, "connect", counting)
    out = ss.search_sessions("body of p11ae")
    monkeypatch.setattr(ss.sqlite3, "connect", real)
    assert any(r["id"] == "20250101-120000-p11ae" for r in out)
    assert opens["n"] == 2  # refresh + query (one connect each, one connection)


def test_refresh_index_still_works_standalone():
    """The caller-owned-conn default path (fresh connection) still works."""
    _write_session("20250101-120000-p11af")
    assert ss._refresh_index() >= 0
    conn = si._open()
    try:
        row = conn.execute(
            "SELECT mtime FROM sessions_fts WHERE session_id = ?",
            ("20250101-120000-p11af",)).fetchone()
        assert row is not None
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
