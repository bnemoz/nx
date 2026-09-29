#!/usr/bin/env python3
"""Regression tests for opt-in age x size session retention.

0.104.x: `_sessiondb.retention_sweep` applies two independent, OFF-by-default
gates over the store — small-session expiry (age + event count) and a hard
floor that protects big sessions from gate 1 (and can itself expire by age).
A store that silently deletes must never be the default, so every flag
defaults to 0 and the sweep fast-paths without touching the DB.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="nx-session-retention-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_") and k not in ("NX_HOME", "NX_SESSIONS_DIR"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _sessiondb as sdb  # noqa: E402

_DAY = 86400.0


def _clear_test_sessions_dir():
    d = os.environ["NX_SESSIONS_DIR"]
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _conn():
    path = os.path.join(os.environ["NX_SESSIONS_DIR"], "sessions.sqlite")
    return sqlite3.connect(path)


def _msgs(n):
    out = [{"role": "system", "content": "You are nx."}]
    for i in range(max(0, n - 1)):
        out.append({"role": "user", "content": f"u{i}"})
    return out[:n]


def _seed(sid, n_msgs, age_days):
    """Seed a real session (chain row created) then age it via the indexed
    `updated_at` column the sweep reads."""
    sdb.store_session(sid, _msgs(n_msgs), meta={"title": sid})
    with _conn() as c:
        c.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                  (time.time() - age_days * _DAY, sid))


def _seed_stub(sid, age_days):
    """A legacy stub row with no session_chain row (chain_len counts as 0)."""
    sdb._ensure_migrated()
    data = {"id": sid, "messages": [], "updated_at": time.time() - age_days * _DAY,
            "revision": 0, "title": sid}
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO sessions"
            " (session_id, data, updated_at, revision, title, cwd, model,"
            "  source) VALUES (?,?,?,?,?,?,?,?)",
            (sid, json.dumps(data), time.time() - age_days * _DAY, 0, sid,
             None, None, None))


def _clear_flags():
    for k in ("NX_SESSION_RETAIN_SMALL_DAYS", "NX_SESSION_RETAIN_SMALL_MAX_MSGS",
              "NX_SESSION_RETAIN_FLOOR_MSGS", "NX_SESSION_RETAIN_FLOOR_DAYS"):
        os.environ.pop(k, None)


# --- 1. small gate ----------------------------------------------------------

def test_small_gate_deletes_only_small_and_old():
    _clear_test_sessions_dir()
    _clear_flags()
    _seed("small-old", 3, age_days=40)
    _seed("small-new", 3, age_days=1)
    _seed("big-old", 12, age_days=40)
    os.environ["NX_SESSION_RETAIN_SMALL_DAYS"] = "30"
    out = sdb.retention_sweep()
    assert out == {"expired_small": 1, "expired_floor": 0}
    assert m._load_session("small-old") is None
    assert m._load_session("small-new")["messages"] == _msgs(3)
    assert m._load_session("big-old")["messages"] == _msgs(12)


# --- 2. floor protects gate 1 ----------------------------------------------

def test_floor_protects_big_session_from_gate1():
    _clear_test_sessions_dir()
    _clear_flags()
    _seed("big-old", 12, age_days=400)
    _seed("small-old", 3, age_days=400)
    os.environ["NX_SESSION_RETAIN_SMALL_DAYS"] = "30"
    # A lowered max would have swept the big row too, but the floor protects it.
    os.environ["NX_SESSION_RETAIN_SMALL_MAX_MSGS"] = "100"
    os.environ["NX_SESSION_RETAIN_FLOOR_MSGS"] = "10"
    out = sdb.retention_sweep()
    assert out == {"expired_small": 1, "expired_floor": 0}
    assert m._load_session("big-old")["messages"] == _msgs(12)
    assert m._load_session("small-old") is None


# --- 3. floor_days expires big sessions by age -----------------------------

def test_floor_days_deletes_big_sessions_by_age():
    _clear_test_sessions_dir()
    _clear_flags()
    _seed("big-old", 50, age_days=120)
    _seed("big-new", 50, age_days=10)
    os.environ["NX_SESSION_RETAIN_SMALL_DAYS"] = "0"
    os.environ["NX_SESSION_RETAIN_FLOOR_DAYS"] = "90"
    out = sdb.retention_sweep()
    assert out == {"expired_small": 0, "expired_floor": 1}
    assert m._load_session("big-old") is None
    assert m._load_session("big-new")["messages"] == _msgs(50)


# --- 4. default OFF ---------------------------------------------------------

def test_default_off_deletes_nothing():
    _clear_test_sessions_dir()
    _clear_flags()
    _seed("tiny-ancient", 1, age_days=9999)
    out = sdb.retention_sweep()
    assert out == {"expired_small": 0, "expired_floor": 0}
    assert m._load_session("tiny-ancient") is not None


# --- 5. prune CLI wiring ----------------------------------------------------

def test_prune_store_runs_retention():
    _clear_test_sessions_dir()
    _clear_flags()
    from nx import _sessions_fs
    _seed("small-old", 2, age_days=40)
    os.environ["NX_SESSION_RETAIN_SMALL_DAYS"] = "30"
    stats = _sessions_fs._prune_sessions_store()
    assert stats["expired_small"] == 1
    assert stats["expired_floor"] == 0
    assert m._load_session("small-old") is None


# --- 6. legacy stub (no chain row) -----------------------------------------

def test_legacy_stub_without_chain_is_small_candidate():
    _clear_test_sessions_dir()
    _clear_flags()
    _seed_stub("legacy-stub", age_days=40)
    os.environ["NX_SESSION_RETAIN_SMALL_DAYS"] = "30"
    out = sdb.retention_sweep()
    assert out == {"expired_small": 1, "expired_floor": 0}
    assert m._load_session("legacy-stub") is None
