"""The session store (0.101.0): one SQLite database, the source of truth.

Replaces the one-JSON-file-per-session layout (`<sessions_dir>/<id>.json`,
written atomically under an flock since the beginning) with a single
`sessions.sqlite` in the sessions dir:

    sessions      — one row per session; `data` is the full transcript
                    JSON (exactly the dict the old per-session file held:
                    id/messages/meta/metrics/...), plus indexed columns
                    (updated_at, revision, title, cwd, model, source) for
                    listing without a JSON parse
    sessions_fts  — FTS5 over title/description/body (search backend,
                    same shape the 2.3b consolidated cache used)

The transcript is the `data` blob, period: there is no shadow file. What
was the "cache, never the source of truth" rule becomes "the DB is the
store"; the migration keeps the old files on disk untouched as a backup
(imported once, idempotently, when the DB holds no transcript rows).

Contracts carried over from the file store:
- 0600 file, 0700 dir (nx/_private.py) — transcripts hold tool output.
- WAL + busy_timeout (NX_STORE_BUSY_TIMEOUT_MS) + a bounded retry ladder;
  a *busy* lock is contention and is never treated as corruption, while a
  corrupt file is set aside as `sessions.sqlite.corrupt` and a fresh one
  is created.
- Writers merge, not replace: BEGIN IMMEDIATE + the stored `revision` is
  the optimistic-concurrency check that the old flock spanned, and
  _merge_session_messages (in _sessions_fs, still the merge authority)
  resolves a stale writer against the committed transcript.
- Every helper degrades instead of raising into listing/GC paths.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from typing import Any, Optional

from ._env import _env_int
from ._logging import _log_event
from ._private import open_private, private_dir

STORE_FILENAME = "sessions.sqlite"


def _sessions_dir() -> str:
    # Local import: _sessions_fs is the store's primary *caller* and reads
    # st.SESSION_HOME lazily; a top-level import would be circular.
    from ._sessions_fs import _sessions_dir as _d
    return _d()


def _store_path() -> str:
    return os.path.join(_sessions_dir(), STORE_FILENAME)


def _set_aside_corrupt() -> None:
    """Move an unusable store aside so a fresh one can be created. The
    mirror files are not touched here — the next migration pass re-imports
    anything the fresh store is missing (see _ensure_migrated)."""
    path = _store_path()
    try:
        os.replace(path, path + ".corrupt")
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass


def _busy_timeout_ms() -> int:
    return max(1, _env_int("NX_STORE_BUSY_TIMEOUT_MS", 750))


def _connect() -> sqlite3.Connection:
    d = _sessions_dir()
    try:
        private_dir(d)
    except OSError:
        pass  # a read-only dir still serves existing rows
    path = _store_path()
    if not os.path.exists(path):
        try:
            open_private(path)  # 0600
        except OSError:
            pass
    timeout = _busy_timeout_ms() / 1000.0
    conn = sqlite3.connect(path, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_busy_timeout_ms()}")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions("
        "session_id TEXT PRIMARY KEY,"
        "data TEXT NOT NULL,"
        "updated_at REAL, revision INTEGER,"
        "title TEXT, cwd TEXT, model TEXT, source TEXT)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at)")
    # FTS5 ships in every CPython build nx supports, but degrade to a
    # listing-only store if this one lacks it.
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5("
            "session_id UNINDEXED, title, description, body, mtime UNINDEXED)")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def _open() -> Optional[sqlite3.Connection]:
    """An open, schema-correct connection — or None when the store can't
    serve this call. Every caller treats None as "fall back / skip".

    Busy ("database is locked") is contention, not corruption: retried
    with a small jittered backoff, never a set-aside. A corrupt file is
    set aside and retried once on the fresh store."""
    last_err: Optional[Exception] = None
    for attempt in range(3):
        if attempt:
            time.sleep(0.02 * (2 ** (attempt - 1)) + random.random() * 0.01)
        try:
            conn = _connect()
            _ensure_schema(conn)
            return conn
        except sqlite3.OperationalError as e:
            last_err = e
            msg = str(e)
            if "locked" in msg or "busy" in msg:
                continue  # contention: bounded retry, never a set-aside
            if attempt >= 1 or "not a database" not in msg:
                _log_event("error", {"where": "sessiondb", "error": msg})
                return None
            _set_aside_corrupt()
        except sqlite3.DatabaseError:
            _set_aside_corrupt()
    _log_event("error", {"where": "sessiondb", "error": str(last_err)})
    return None


# --- migration: legacy <id>.json files -> rows ---------------------------------

_MIGRATED: dict[str, int] = {}  # sessions dir -> import attempt count
# (plain dict — tests/conftest.py rewinds module dicts by design)


def migrate_files(force: bool = False) -> int:
    """Import legacy per-session JSON files into the store. Idempotent.

    Runs lazily (via _ensure_migrated) whenever the store currently holds
    no transcript rows — that covers first-run migration AND self-healing
    after the DB file was deleted or set aside as corrupt. `force=True`
    (nx sessions --migrate) scans even when rows exist; files already
    imported are skipped, and files are never deleted (backup).
    Returns the number of sessions imported."""
    d = _sessions_dir()
    if not force and _MIGRATED.get(d):
        return 0
    try:
        names = [n for n in os.listdir(d)
                 if n.endswith(".json") and not n.startswith(".")]
    except OSError:
        _MIGRATED[d] = _MIGRATED.get(d, 0) + 1
        return 0
    conn = _open()
    if conn is None:
        return 0
    n = 0
    try:
        have = {r[0] for r in conn.execute("SELECT session_id FROM sessions")}
        if have and not force:
            _MIGRATED[d] = _MIGRATED.get(d, 0) + 1
            return 0
        with conn:
            for name in sorted(names):
                sid = name[:-len(".json")]
                if sid in have:
                    continue
                try:
                    with open(os.path.join(d, name), encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(data, dict) or not isinstance(
                        data.get("messages"), list):
                    continue
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO sessions"
                        " (session_id, data, updated_at, revision, title,"
                        "  cwd, model, source) VALUES (?,?,?,?,?,?,?,?)",
                        _data_to_row(sid, data))
                except sqlite3.Error:
                    continue
                n += 1
            n += _import_legacy_shards(conn, d, have)
    except sqlite3.Error:
        return n
    finally:
        conn.close()
    _MIGRATED[d] = _MIGRATED.get(d, 0) + 1
    return n


def _import_legacy_shards(conn: sqlite3.Connection, d: str,
                          have: set[str]) -> int:
    """Pre-2.3b `.index/<sid>.json` shards: a transcript we no longer have
    can't be resurrected, but the listing summary should survive — import
    each shard as a listing stub (messages: []), then retire the dir.
    Returns shards imported."""
    index_dir = os.path.join(d, ".index")
    try:
        shard_names = [n for n in os.listdir(index_dir) if n.endswith(".json")]
    except OSError:
        return 0
    n = 0
    with conn:
        for name in sorted(shard_names):
            sid = name[:-len(".json")]
            if sid in have:
                continue
            try:
                with open(os.path.join(index_dir, name), encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(summary, dict):
                continue
            data = {
                "id": sid,
                "title": summary.get("title"),
                "description": summary.get("description"),
                "cwd": summary.get("cwd"),
                "model": summary.get("model"),
                "source": summary.get("source"),
                "created_at": float(summary.get("updated_at") or 0.0),
                "updated_at": float(summary.get("updated_at") or 0.0),
                "revision": 0,
                "messages": [],  # transcript lost pre-store — listing stub only
            }
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions"
                    " (session_id, data, updated_at, revision, title,"
                    "  cwd, model, source) VALUES (?,?,?,?,?,?,?,?)",
                    _data_to_row(sid, data))
            except sqlite3.Error:
                continue
            n += 1
    import shutil
    try:
        shutil.rmtree(index_dir)
    except OSError:
        pass
    return n


def _ensure_migrated() -> None:
    """Lazy migration hook, called from the read/write facade paths."""
    try:
        migrate_files(force=False)
    except Exception:
        pass  # never raise into a save/load path


def _data_to_row(session_id: str, data: dict[str, Any]) -> tuple:
    return (
        session_id,
        json.dumps(data, ensure_ascii=False),
        float(data.get("updated_at") or time.time()),
        int(data.get("revision") or 0),
        data.get("title"),
        data.get("cwd"),
        data.get("model"),
        data.get("source"),
    )


def _row_to_data(session_id: str, blob: str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("id", session_id)
    msgs = data.get("messages")
    if not isinstance(msgs, list):
        return None
    return data


def _store_path_exists() -> bool:
    return os.path.exists(_store_path())


# --- load ------------------------------------------------------------------

def load_session(session_id: str) -> Optional[dict[str, Any]]:
    """The transcript dict for `session_id`, or None."""
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            (session_id,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row:
        return None
    return _row_to_data(session_id, row[0])


# --- write -----------------------------------------------------------------

def store_session(
    session_id: str,
    messages: list[dict[str, Any]],
    meta: Optional[dict[str, Any]] = None,
) -> bool:
    """Insert/replace one session row — the old _write_session critical
    section, in one BEGIN IMMEDIATE transaction (the writer lock the old
    per-file flock provided).

    The committed transcript is re-read *inside* the transaction and
    merged against the caller's snapshot via _merge_session_messages, so a
    stale writer (e.g. a background description refresh) appends its turns
    instead of clobbering another process's. Meta keys the caller supplies
    win over stored ones; created_at/revision come from the stored row.

    Returns True on success (False = store unavailable; callers treat the
    session as unsaved-but-alive, same as the old failed file write)."""
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return False
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing: dict[str, Any] = {}
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,)).fetchone()
            if row:
                parsed = _row_to_data(session_id, row[0])
                if parsed:
                    existing = parsed
            now = time.time()
            from ._sessions_fs import _merge_session_messages
            merged = _merge_session_messages(
                existing.get("messages") or [], messages)
            data = dict(existing)
            data["id"] = session_id
            data["messages"] = merged
            data["created_at"] = existing.get("created_at", now)
            data["updated_at"] = now
            data["revision"] = int(existing.get("revision") or 0) + 1
            if meta:
                for k, v in meta.items():
                    if v is not None:
                        data[k] = v
            conn.execute(
                "INSERT OR REPLACE INTO sessions"
                " (session_id, data, updated_at, revision, title, cwd,"
                "  model, source) VALUES (?,?,?,?,?,?,?,?)",
                _data_to_row(session_id, data))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


# --- search index (FTS5) ------------------------------------------------------

def refresh_fts(session_id: str, messages: list[dict[str, Any]],
                title: str | None = None, description: str | None = None,
                conn: Optional[sqlite3.Connection] = None) -> None:
    """Insert/replace one session's FTS body row. Building the body scales
    with the transcript, so the write path only calls this on indexed saves
    (update_index=True) — mid-turn autosaves skip it, exactly like the 2.3b
    design where the FTS body was backfilled lazily by the next search."""
    own = conn is None
    if own:
        conn = _open()
        if conn is None:
            return
    try:
        from ._sessions_io import _flatten_message_text
        parts = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            txt = _flatten_message_text(m)
            if txt:
                parts.append(f"[{m.get('role', '?')}] {txt}")
        body = "\n".join(parts)
        with conn:
            conn.execute("DELETE FROM sessions_fts WHERE session_id = ?",
                         (session_id,))
            try:
                conn.execute(
                    "INSERT INTO sessions_fts (session_id, title, description,"
                    " body, mtime) VALUES (?,?,?,?,?)",
                    (session_id, title or "", description or "", body,
                     time.time()))
            except sqlite3.OperationalError:
                pass  # no FTS5 here — the transcript row is the part that matters
    finally:
        if own:
            conn.close()


def delete_session_row(session_id: str) -> bool:
    """Drop one session's rows (transcript + FTS). Best-effort."""
    conn = _open()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?",
                         (session_id,))
            try:
                conn.execute("DELETE FROM sessions_fts WHERE session_id = ?",
                             (session_id,))
            except sqlite3.OperationalError:
                pass
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


# --- listing -----------------------------------------------------------------

_LIST_COLS = ("session_id, data, updated_at, revision, title, cwd,"
              " model, source")


def list_ids_newest(limit: Optional[int] = None) -> list[tuple[str, float]]:
    """(session_id, updated_at) newest-first. `_ensure_migrated` runs first
    so a fresh store picks up legacy files before the first listing."""
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return []
    try:
        sql = "SELECT session_id, updated_at FROM sessions ORDER BY updated_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        return [(r[0], float(r[1] or 0.0)) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def all_ids() -> set[str]:
    conn = _open()
    if conn is None:
        return set()
    try:
        return {r[0] for r in conn.execute("SELECT session_id FROM sessions")}
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


def iter_rows() -> list[tuple[str, Optional[dict[str, Any]]]]:
    """(session_id, parsed transcript) for every row — the live-set scan
    (image sha256s) and pruning read the whole store in one query. A
    corrupt blob yields (sid, None) so the caller can fall back."""
    conn = _open()
    if conn is None:
        return []
    try:
        rows = conn.execute("SELECT session_id, data FROM sessions").fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [(sid, _row_to_data(sid, blob)) for sid, blob in rows]


def prune_stale() -> int:
    """0.101.0 DB-truth: there are no per-session files to stat, so the
    only staleness inside the store is an FTS row whose transcript row is
    gone (a crash between the two deletes). Returns rows pruned."""
    conn = _open()
    if conn is None:
        return 0
    try:
        try:
            fts_ids = [r[0] for r in conn.execute(
                "SELECT session_id FROM sessions_fts").fetchall()]
        except sqlite3.OperationalError:
            return 0  # no FTS5 here
        have = {r[0] for r in conn.execute("SELECT session_id FROM sessions")}
        doomed = [sid for sid in fts_ids if sid not in have]
        if not doomed:
            return 0
        with conn:
            for sid in doomed:
                conn.execute(
                    "DELETE FROM sessions_fts WHERE session_id = ?", (sid,))
        return len(doomed)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def store_stats() -> dict[str, int]:
    conn = _open()
    if conn is None:
        return {"sessions_indexed": 0}
    try:
        n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    except sqlite3.Error:
        return {"sessions_indexed": 0}
    finally:
        conn.close()
    return {"sessions_indexed": n}
