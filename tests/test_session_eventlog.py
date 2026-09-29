#!/usr/bin/env python3
"""Regression tests for the append-only event-log session store.

0.104.x: the authoritative transcript is the session_events/session_chain
append-only log inside sessions.sqlite; the sessions.data blob is a lagging
cache (rewritten only on merge/compaction/fork/migration). These tests pin
the fast append, the lagging-cache property, the compaction collapse, the
collision fallback, and the legacy blob -> event-log migration.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-session-eventlog-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_") and k not in ("NX_HOME", "NX_SESSIONS_DIR"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _sessiondb as sdb  # noqa: E402


def _clear_test_sessions_dir():
    d = os.environ["NX_SESSIONS_DIR"]
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _msg(role, content):
    return {"role": role, "content": content}


def _base_history(n=4):
    msgs = [{"role": "system", "content": "You are nx."}]
    for i in range(n):
        msgs.append(_msg("user", f"u{i}"))
        msgs.append(_msg("assistant", f"a{i}"))
    return msgs


def _compress_snapshot(existing, keep=4):
    """Reproduce the shape compress() writes via messages[:] = sys + [summary]
    + tail (nx/_compress.py, end of compress())."""
    sys_msg = [existing[0]] if existing[0].get("role") == "system" else []
    summarized = len(existing) - len(sys_msg) - keep
    summary = _msg(
        "user",
        f"[Compressed context — {summarized} earlier turns summarized;"
        f" last {keep} turns kept verbatim]\n\nearlier work summarized here",
    )
    return sys_msg + [summary] + existing[-keep:]


def _conn():
    path = os.path.join(os.environ["NX_SESSIONS_DIR"], "sessions.sqlite")
    return sqlite3.connect(path)


def _chain_len(sid):
    with _conn() as c:
        row = c.execute(
            "SELECT chain_len FROM session_chain WHERE session_id = ?",
            (sid,)).fetchone()
    return None if row is None else int(row[0])


def _event_count(sid):
    with _conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ?",
            (sid,)).fetchone()[0]


def _blob_messages(sid):
    with _conn() as c:
        row = c.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            (sid,)).fetchone()
    return json.loads(row[0]).get("messages") if row else None


# --- 1. fast append ---------------------------------------------------------

def test_fast_append_extends_chain_without_rewriting_blob():
    _clear_test_sessions_dir()
    sid = "20250101-120000-evlog1"
    base = _base_history(4)
    assert m._write_session(sid, base) is None  # caller returns None; just save
    saved = sdb.store_session(sid, base)
    assert saved is not None  # warm the chain via the direct seam
    grown = base + [_msg("user", "u9"), _msg("assistant", "a9"),
                    _msg("user", "u10")]
    out = sdb.store_session(sid, grown)
    assert out == grown
    assert _chain_len(sid) == len(grown)
    assert _event_count(sid) == len(grown)
    loaded = m._load_session(sid)["messages"]
    assert loaded == grown


# --- 2. lagging cache -------------------------------------------------------

def test_blob_is_lagging_cache_after_append():
    _clear_test_sessions_dir()
    sid = "20250101-120000-evlog2"
    base = _base_history(4)
    sdb.store_session(sid, base)
    n_base = len(base)
    grown = base + [_msg("user", "later")]
    sdb.store_session(sid, grown)
    blob_msgs = _blob_messages(sid)
    assert isinstance(blob_msgs, list)
    assert len(blob_msgs) == n_base  # cache not rewritten by the fast append
    loaded = m._load_session(sid)["messages"]
    assert loaded == grown  # the event log is the authority


# --- 3. compaction collapses the chain --------------------------------------

def test_compaction_collapses_chain():
    _clear_test_sessions_dir()
    sid = "20250101-120000-evlog3"
    base = _base_history(10)
    sdb.store_session(sid, base)
    incoming = _compress_snapshot(base, keep=4)
    assert len(incoming) < len(base)
    out = sdb.store_session(sid, incoming)
    assert out == incoming
    assert _chain_len(sid) == len(incoming)
    assert _event_count(sid) == len(incoming)
    assert m._load_session(sid)["messages"] == incoming


# --- 4. collision fallback --------------------------------------------------

def test_collision_falls_back_to_rebuild():
    _clear_test_sessions_dir()
    sid = "20250101-120000-evlog4"
    base = _base_history(4)
    sdb.store_session(sid, base)
    grown = base + [_msg("user", "new turn")]
    # Pre-insert the exact event the fast path would write at seq L (the
    # concurrent-writer collision): the INSERT must fail, the txn roll back,
    # and the rebuild path must still produce a consistent chain.
    with _conn() as c:
        c.execute(
            "INSERT INTO session_events(session_id, seq, msg) VALUES (?,?,?)",
            (sid, len(base), json.dumps(grown[len(base)], ensure_ascii=False)))
    out = sdb.store_session(sid, grown)
    assert out is not None
    assert _chain_len(sid) == _event_count(sid)  # no orphan/dup slots
    loaded = m._load_session(sid)["messages"]
    assert loaded == grown
    # the other writer's turn is not lost and not duplicated
    assert [x.get("content") for x in loaded].count("new turn") == 1


# --- 5. legacy migration ----------------------------------------------------

def test_legacy_blob_row_migrates_to_event_log():
    _clear_test_sessions_dir()
    sid = "20250101-120000-evlog5"
    base = _base_history(4)
    data = {"id": sid, "messages": base, "updated_at": 1234.0,
            "revision": 3, "title": "legacy"}
    sdb._ensure_migrated()  # create the store file + schema first
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO sessions"
            " (session_id, data, updated_at, revision, title, cwd, model,"
            "  source) VALUES (?,?,?,?,?,?,?,?)",
            (sid, json.dumps(data, ensure_ascii=False), 1234.0, 3, "legacy",
             None, None, None))
    grown = base + [_msg("user", "appended post-migration")]
    out = sdb.store_session(sid, grown)
    assert out is not None
    assert _chain_len(sid) == len(grown)
    assert _event_count(sid) == len(grown)
    loaded = m._load_session(sid)["messages"]
    assert loaded == grown
