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
    # Append-only event log (0.104.x): the authoritative transcript lives in
    # per-message rows; the sessions.data blob becomes a lagging cache (only
    # rebuilt on merge/compaction/fork/migration). session_chain is the
    # per-session index into the log.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_events("
        "session_id TEXT NOT NULL,"
        "seq INTEGER NOT NULL,"
        "msg TEXT NOT NULL,"
        "PRIMARY KEY(session_id, seq)) WITHOUT ROWID")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_chain("
        "session_id TEXT PRIMARY KEY,"
        "chain_len INTEGER NOT NULL,"
        "updated_at REAL)")
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
    """The transcript dict for `session_id`, or None.

    0.104.x: when a session_chain row exists, the messages come from the
    append-only event log (the authoritative store) while the metadata
    (title, revision, ...) comes from the blob cache with the live
    revision/updated_at taken from the indexed columns (the blob cache is
    stale between rebuilds); a missing blob degrades to a synthesized dict.
    """
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT data, revision, updated_at FROM sessions"
            " WHERE session_id = ?", (session_id,)).fetchone()
        chain = _read_chain(conn, session_id)
        events = _read_events(conn, session_id) if chain is not None else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if chain is not None:
        out = _row_to_data(session_id, row[0]) if row else {}
        if out is None:
            out = {}
        out["id"] = session_id
        out["messages"] = events or []
        if row:
            if row[1] is not None:
                out["revision"] = int(row[1])
            if row[2] is not None:
                out["updated_at"] = float(row[2])
        out.setdefault("updated_at", 0.0)
        return out
    if not row:
        return None
    return _row_to_data(session_id, row[0])


def _session_meta(session_id: str) -> dict[str, Any]:
    """Cheap metadata-only view (title/description/cwd/...) for the save
    path's FTS refresh — a JSON parse of the one blob, no event-log reads
    (the caller already holds the authoritative merged message list)."""
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return {}
    try:
        row = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            (session_id,)).fetchone()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    if not row:
        return {}
    return _row_to_data(session_id, row[0]) or {}


# --- event log helpers ------------------------------------------------------

def _read_chain(conn: sqlite3.Connection, session_id: str) -> Optional[int]:
    """The committed chain length for a session (None = no chain row)."""
    row = conn.execute(
        "SELECT chain_len FROM session_chain WHERE session_id = ?",
        (session_id,)).fetchone()
    return int(row[0]) if row else None


def _read_events(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """The committed event-log messages, oldest first. A corrupt/unparseable
    event degrades that turn to a skipped slot (a rebuild heals the gap)."""
    out: list[dict[str, Any]] = []
    for (blob,) in conn.execute(
            "SELECT msg FROM session_events WHERE session_id = ?"
            " ORDER BY seq", (session_id,)):
        try:
            m = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if isinstance(m, dict):
            out.append(m)
    return out


# --- write ------------------------------------------------------------------

def _store_session_txn(
    session_id: str,
    messages: list[dict[str, Any]],
    meta: Optional[dict[str, Any]],
    force_rebuild: bool = False,
) -> Optional[list[dict[str, Any]]]:
    """One BEGIN IMMEDIATE attempt. Returns the authoritative merged message
    list, or None when the store is unavailable."""
    conn = _open()
    if conn is None:
        return None
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = time.time()
            from ._sessions_fs import _merge_session_messages
            chain_len = _read_chain(conn, session_id)
            row = conn.execute(
                "SELECT data, revision FROM sessions WHERE session_id = ?",
                (session_id,)).fetchone()
            existing: dict[str, Any] = {}
            if row:
                parsed = _row_to_data(session_id, row[0])
                if parsed:
                    existing = parsed
            # the indexed revision is the live counter (the blob cache is
            # stale between rebuilds)
            live_revision = int(row[1] or 0) if row else 0

            fast = False
            if (not force_rebuild and chain_len is not None
                    and 0 < chain_len <= len(messages)):
                tail = conn.execute(
                    "SELECT msg FROM session_events"
                    " WHERE session_id = ? AND seq = ?",
                    (session_id, chain_len - 1)).fetchone()
                if tail is not None:
                    try:
                        fast = json.loads(tail[0]) == messages[chain_len - 1]
                    except (ValueError, TypeError):
                        fast = False

            if fast:
                # Fast append: new events only; the data blob is a lagging
                # cache and is deliberately NOT rewritten.
                for i, m in enumerate(messages[chain_len:], start=chain_len):
                    conn.execute(
                        "INSERT INTO session_events(session_id, seq, msg)"
                        " VALUES (?,?,?)",
                        (session_id, i,
                         json.dumps(m, ensure_ascii=False)))
                merged = list(messages)
            else:
                # Full rebuild (no chain / compaction / fork / shorter than
                # chain / tail mismatch / collision retry): merge, collapse
                # the chain, rewrite the blob cache.
                if chain_len is not None:
                    existing_msgs: list[dict[str, Any]] = _read_events(
                        conn, session_id)
                else:
                    existing_msgs = existing.get("messages") or []
                merged = _merge_session_messages(existing_msgs, messages)
                conn.execute(
                    "DELETE FROM session_events WHERE session_id = ?",
                    (session_id,))
                for i, m in enumerate(merged):
                    conn.execute(
                        "INSERT INTO session_events(session_id, seq, msg)"
                        " VALUES (?,?,?)",
                        (session_id, i,
                         json.dumps(m, ensure_ascii=False)))
                data = dict(existing)
                data["id"] = session_id
                data["messages"] = merged
                data["created_at"] = existing.get("created_at", now)
                data["updated_at"] = now
                data["revision"] = live_revision + 1
                if meta:
                    for k, v in meta.items():
                        if v is not None:
                            data[k] = v
                conn.execute(
                    "INSERT OR REPLACE INTO sessions"
                    " (session_id, data, updated_at, revision, title, cwd,"
                    "  model, source) VALUES (?,?,?,?,?,?,?,?)",
                    _data_to_row(session_id, data))

            # Shared tail: chain upsert + indexed meta columns. On a rebuild
            # the blob INSERT OR REPLACE already set the indexed columns, so
            # only the chain row is written; on a fast append the blob cache
            # is stale by design and only the indexed columns are touched.
            conn.execute(
                "INSERT OR REPLACE INTO session_chain"
                " (session_id, chain_len, updated_at) VALUES (?,?,?)",
                (session_id, len(merged), now))
            if fast:
                # Only the columns the caller actually supplied may change —
                # the blob cache's values are stale, so defaulting to them
                # here could regress a title another writer just set.
                sets = ["updated_at = ?", "revision = COALESCE(revision, 0) + 1"]
                vals: list[Any] = [now]
                for col in ("title", "cwd", "model", "source"):
                    v = (meta or {}).get(col)
                    if v is not None:
                        sets.append(f"{col} = ?")
                        vals.append(v)
                vals.append(session_id)
                conn.execute(
                    f"UPDATE sessions SET {', '.join(sets)}"
                    " WHERE session_id = ?", vals)
            conn.commit()
            return merged
        except Exception:
            conn.rollback()
            raise
    except sqlite3.IntegrityError:
        # fast-path insert collided with a concurrent writer's event: let
        # store_session roll back and retry as a full rebuild
        raise
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def store_session(
    session_id: str,
    messages: list[dict[str, Any]],
    meta: Optional[dict[str, Any]] = None,
) -> Optional[list[dict[str, Any]]]:
    """Insert/append one session's transcript — the old _write_session critical
    section, in one BEGIN IMMEDIATE transaction (the writer lock the old
    per-file flock provided).

    0.104.x: the authoritative transcript is the append-only event log
    (session_events + session_chain); the sessions.data blob is a lagging
    cache. A fast append (the committed chain tail matches the caller's
    prefix) inserts only the new events and updates the indexed columns —
    the blob is NOT rewritten. A full rebuild (compaction, fork, migration,
    shorter than the chain, tail mismatch) merges via
    _merge_session_messages, collapses the chain, and rewrites the blob.

    Returns the authoritative merged message list on success (None = store
    unavailable; old truthiness checks still work)."""
    _ensure_migrated()
    try:
        merged = _store_session_txn(session_id, messages, meta)
    except sqlite3.Error:
        # A collision on the fast-path insert (another writer extended the
        # chain after our tail check): retry once as a full rebuild — the
        # tail still matches, so the fast path must be disabled or it would
        # collide with the same committed event a second time.
        return _store_session_txn(session_id, messages, meta, force_rebuild=True)
    return merged



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
            conn.execute("DELETE FROM session_events WHERE session_id = ?",
                         (session_id,))
            conn.execute("DELETE FROM session_chain WHERE session_id = ?",
                         (session_id,))
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
        chains = {r[0] for r in conn.execute(
            "SELECT session_id FROM session_chain")}
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out: list[tuple[str, Optional[dict[str, Any]]]] = []
    for sid, blob in rows:
        data = _row_to_data(sid, blob)
        if sid in chains:
            # Event-log-backed: materialize messages from the events so the
            # live-set sees appended-but-not-yet-cached turns.
            try:
                conn2 = _open()
                if conn2 is not None:
                    try:
                        events = _read_events(conn2, sid)
                    finally:
                        conn2.close()
                else:
                    events = None
            except sqlite3.Error:
                events = None
            if events is not None:
                base = dict(data or {})
                base["id"] = sid
                base["messages"] = events
                data = base
        out.append((sid, data))
    return out


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
