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
from typing import Any, Iterator, Optional

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


_FTS_OLD_MARKER = "session_id UNINDEXED"


def _create_external_fts(conn: sqlite3.Connection) -> None:
    """The 0.115.0 FTS trio. Message bodies live in the real `fts_msgs`
    table (one row per transcript message); `msgs_fts` references them by
    rowid and stores only the inverted index — no second copy of the text.
    `meta_fts` is a tiny per-session index over title/description so a
    title-only query still matches. Created with IF NOT EXISTS, so a call
    on an up-to-date store is a cheap no-op."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fts_msgs("
        "sid TEXT NOT NULL, seq INTEGER NOT NULL,"
        "role TEXT, body TEXT, mtime REAL)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fts_meta("
        "sid TEXT PRIMARY KEY, title TEXT, description TEXT, mtime REAL)")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS msgs_fts USING fts5("
            "body, content='fts_msgs', content_rowid='rowid')")
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS meta_fts USING fts5("
            "title, description, content='fts_meta',"
            " content_rowid='rowid')")
    except sqlite3.OperationalError:
        pass  # no FTS5 here — fts_msgs still serves the delete/prune paths


def _fts_is_current(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'msgs_fts'"
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None and "content='fts_msgs'" in (row[0] or "")


def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
    """Create the 0.115.0 per-message FTS pair, migrating a pre-0.115
    store (external-content sessions_fts over fts_body, or the legacy
    standalone sessions_fts) in place: the old index tables are dropped
    and the per-message index is rebuilt lazily by search (see
    _sessions_search._refresh_index — a blocking backfill of 2.2M events
    must never run on open). One transaction; any error rolls back to the
    previous shape and search keeps working."""
    if _fts_is_current(conn):
        _create_external_fts(conn)
        return
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for name in ("sessions_fts", "fts_body"):
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {name}")
                except sqlite3.Error:
                    pass
            _create_external_fts(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    except sqlite3.Error:
        pass  # keep whatever shape exists; search degrades but still works


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
    # listing-only store if this one lacks it. 0.104.x: external-content
    # mode — the bodies live in the real `fts_body` table and the FTS
    # index holds only the inverted index, not its own content copy.
    _ensure_fts_schema(conn)
    # 0.108.6: per-session image sha set, maintained on every write so the
    # sidecar-GC live-set scan never has to parse transcript blobs.
    _ensure_shas_column(conn)
    # 0.129.x: listing columns (preview text + message count) maintained on
    # every write so a listing never has to read the event log at all.
    _ensure_listing_columns(conn)
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
    # Cross-process file-claim protocol (nx/_fileclaim.py): one row per claimed
    # file, keyed by path so two processes contend on a single PRIMARY KEY.
    # `cwd` is the claiming process's working directory — a claim only guards
    # the repo it was taken from (see _fileclaim's liveness rules).
    # `pid_start_time` (0.119.0) is the holder's Linux start-time tick, so a
    # recycled pid can't read as a live holder.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS file_claims("
        "path TEXT PRIMARY KEY,"
        "owner_id TEXT NOT NULL,"
        "pid INTEGER NOT NULL,"
        "claimed_at REAL,"
        "cwd TEXT,"
        "pid_start_time TEXT)")
    _ensure_file_claims_cwd(conn)
    _ensure_file_claims_pid_start_time(conn)
    conn.commit()


def _ensure_file_claims_cwd(conn: sqlite3.Connection) -> None:
    """Add the `file_claims.cwd` column to a store created before claims were
    scoped by working directory.

    The column shipped after `file_claims` did, so an existing DB has no
    `cwd`: `ALTER TABLE ADD COLUMN` is the entire migration (SQLite backfills
    NULL for existing rows, and _fileclaim treats a NULL/foreign cwd as
    "not this repo" — such a row stops blocking and the next writer reclaims
    it in place). Idempotent via table_info; any failure just leaves the old
    table shape, and claim_file's own `except` fails open (the edit proceeds)
    on the missing column rather than raising into a write tool."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(file_claims)")}
    except sqlite3.Error:
        return
    if not cols or "cwd" in cols:
        return
    try:
        conn.execute("ALTER TABLE file_claims ADD COLUMN cwd TEXT")
    except sqlite3.OperationalError:
        pass  # duplicate column or a contended store: degrade, never raise


def _ensure_file_claims_pid_start_time(conn: sqlite3.Connection) -> None:
    """Add the `file_claims.pid_start_time` column to a store created before
    claims carried a PID-reuse guard (0.119.0).

    Same shape as _ensure_file_claims_cwd: `ALTER TABLE ADD COLUMN` is the whole
    migration (SQLite backfills NULL), idempotent via table_info, and any
    failure merely leaves the old table shape — _fileclaim reads a NULL start
    time as "no guard possible" and falls back to bare pid liveness, so an
    un-migrated row still blocks a live holder instead of being stolen. The
    column also keeps its NOT NULL-free shape deliberately: pre-0.119.0 rows
    have nothing to backfill it with."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(file_claims)")}
    except sqlite3.Error:
        return
    if not cols or "pid_start_time" in cols:
        return
    try:
        conn.execute("ALTER TABLE file_claims ADD COLUMN pid_start_time TEXT")
    except sqlite3.OperationalError:
        pass  # duplicate column or a contended store: degrade, never raise


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
                        "  cwd, model, source, image_shas, preview, n_msg)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
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
                # keep the shard's own preview/n so the stub lists as-is
                "preview": summary.get("preview"),
                "n_msg": int(summary.get("n") or 0),
            }
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions"
                    " (session_id, data, updated_at, revision, title,"
                    "  cwd, model, source, image_shas, preview, n_msg)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
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


def _ensure_shas_column(conn: sqlite3.Connection) -> None:
    """Add the sessions.image_shas column to a pre-0.108.6 store (rows read
    as NULL and fall back to parsing, same as before the column existed)."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
        if "image_shas" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN image_shas TEXT")
    except sqlite3.Error:
        pass  # the NULL fallback still serves the scan


def _ensure_listing_columns(conn: sqlite3.Connection) -> None:
    """Add the sessions.preview / sessions.n_msg listing columns to a
    pre-0.129 store (rows read as NULL and fall back to a bounded heal in
    the listing path, same as before the columns existed)."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
        if "preview" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN preview TEXT")
        if "n_msg" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN n_msg INTEGER")
    except sqlite3.Error:
        pass  # the NULL fallback still serves the listing


def _preview_of(messages: list[dict[str, Any]]) -> Optional[str]:
    """The indexed preview: the first user message's content, truncated the
    same way _summarize_session_data truncates it (None when there is no
    user message — the listing falls back to the title)."""
    try:
        from ._sessions_io import _safe_title
        for msg in messages or []:
            if isinstance(msg, dict) and msg.get("role") == "user" \
                    and msg.get("content"):
                return _safe_title(msg["content"]) or None
    except Exception:
        return None
    return None


def _n_msg_of(messages: list[dict[str, Any]]) -> int:
    """The indexed turn count: non-system messages (matches `n` in the
    listing summary)."""
    try:
        return sum(1 for m in messages or []
                   if isinstance(m, dict) and m.get("role") != "system")
    except Exception:
        return 0


def heal_listing_row(session_id: str, preview: str, n_msg: int) -> bool:
    """Best-effort backfill of a pre-0.129 row's listing columns (computed
    once, by the listing's bounded self-heal, not on every open)."""
    conn = _open()
    if conn is None:
        return False
    try:
        with conn:
            conn.execute(
                "UPDATE sessions SET preview = ?, n_msg = ?"
                " WHERE session_id = ?", (preview or None, int(n_msg),
                                          session_id))
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _shas_of(data: dict[str, Any]) -> Optional[str]:
    """The sorted image shas referenced by `data`'s messages, as a JSON list
    string — the indexed form of the sidecar-GC live-set."""
    try:
        from ._images import _collect_message_image_refs
        shas = _collect_message_image_refs(data.get("messages") or [])
        return json.dumps(sorted(shas)) if shas else None
    except Exception:
        return None


# The unescaped JSON substring every text_ref event contains — the LIKE
# marker for the GC live-set scan (the escaped form of "type": "text_ref").
TEXT_REF_MARKER = '"type": "text_ref"'


def _data_to_row(session_id: str, data: dict[str, Any]) -> tuple:
    msgs = data.get("messages") or []
    # an explicit preview/n_msg (legacy-shard stubs) wins over a recompute
    preview = data["preview"] if "preview" in data else _preview_of(msgs)
    n_msg = data["n_msg"] if data.get("n_msg") is not None else _n_msg_of(msgs)
    return (
        session_id,
        json.dumps(data, ensure_ascii=False),
        float(data.get("updated_at") or time.time()),
        int(data.get("revision") or 0),
        data.get("title"),
        data.get("cwd"),
        data.get("model"),
        data.get("source"),
        _shas_of(data),
        preview,
        n_msg,
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

    0.115.0: the event log is the only transcript copy — messages always
    come from session_events, metadata from the sessions row's indexed
    columns. A pre-0.104 session (blob, no events) is inflated from its
    blob once, lazily, right here."""
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT data, revision, updated_at, title, cwd, model, source"
            " FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        # Lazy pre-0.104 migration: inflate events from the blob once.
        ev_count = conn.execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ?",
            (session_id,)).fetchone()[0]
        if not ev_count and row[0]:
            parsed = _row_to_data(session_id, row[0])
            if parsed and parsed.get("messages"):
                conn.executemany(
                    "INSERT OR IGNORE INTO session_events(session_id, seq,"
                    " msg) VALUES (?,?,?)",
                    [(session_id, i, json.dumps(m, ensure_ascii=False))
                     for i, m in enumerate(parsed["messages"])])
                conn.commit()
        events = _read_events(conn, session_id)
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    out: dict[str, Any] = {}
    if row and row[0]:
        try:
            shell = json.loads(row[0])
        except (ValueError, TypeError):
            shell = None
        if isinstance(shell, dict):
            out = dict(shell)
    out["id"] = session_id
    out["messages"] = events
    if row:
        if row[1] is not None:
            out["revision"] = int(row[1])
        if row[2] is not None:
            out["updated_at"] = float(row[2])
        # 0.115.0: metadata lives in the indexed columns (the stored blob is
        # a small-meta shell); the columns win over any stale shell keys.
        for k, v in zip(("title", "cwd", "model", "source"), row[3:7]):
            if v is not None:
                out[k] = v
    # 0.115.0: created_at lived only in the removed transcript blob — the
    # first-save updated_at is the closest surviving creation marker.
    out.setdefault("created_at", out.get("updated_at", 0.0))
    out.setdefault("updated_at", 0.0)
    return out


def _session_meta(session_id: str) -> dict[str, Any]:
    """Cheap metadata-only view (title/description/cwd/...) for the save
    path's FTS refresh.

    0.115.0: the blob is gone; description lives in the indexed `fts_meta`
    row (written by refresh_fts) or falls back to nothing — never a blob
    parse."""
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return {}
    try:
        row = conn.execute(
            "SELECT s.title, m.description FROM sessions s"
            " LEFT JOIN fts_meta m ON m.sid = s.session_id"
            " WHERE s.session_id = ?", (session_id,)).fetchone()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    if not row:
        return {}
    return {"id": session_id, "title": row[0],
            "description": row[1]}


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
            # 0.115.0: the blob cache is gone — the event log is the only
            # transcript copy. A fast append never reads any transcript;
            # a rebuild reads events (the authoritative copy) as its merge
            # base. The sessions row holds metadata + indexed columns only.
            row = conn.execute(
                "SELECT revision FROM sessions WHERE session_id = ?",
                (session_id,)).fetchone()
            live_revision = int(row[0] or 0) if row else 0

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

            # Small non-transcript meta (todos/plan/wip/metrics) rides in the
            # shell blob — the event log holds only messages, so this is the
            # one place per-save state can persist. Cheap: a few short keys.
            def _shell_blob() -> str:
                small = {k: v for k, v in (meta or {}).items()
                         if k not in ("id", "title", "description", "cwd",
                                      "model", "source", "updated_at",
                                      "revision", "messages", "created_at")
                         and v is not None}
                if not small:
                    return "{}"
                return json.dumps(small, ensure_ascii=False)

            shell_blob = _shell_blob()

            existing: dict[str, Any] = {}
            if not fast:
                # Lazy pre-0.104 migration: a session with a data blob but
                # no events inflates its events from the blob once.
                if chain_len is None:
                    ev_rows = conn.execute(
                        "SELECT COUNT(*) FROM session_events"
                        " WHERE session_id = ?", (session_id,)).fetchone()
                    if not ev_rows[0]:
                        blob_row = conn.execute(
                            "SELECT data FROM sessions WHERE session_id = ?",
                            (session_id,)).fetchone()
                        if blob_row and blob_row[0]:
                            parsed = _row_to_data(session_id, blob_row[0])
                            if parsed and parsed.get("messages"):
                                conn.executemany(
                                    "INSERT INTO session_events(session_id,"
                                    " seq, msg) VALUES (?,?,?)",
                                    [(session_id, i, json.dumps(m, ensure_ascii=False))
                                     for i, m in enumerate(parsed["messages"])])

            if fast:
                # Fast append: new events only. The sessions row is
                # metadata + indexed columns; there is no blob to rewrite.
                for i, m in enumerate(messages[chain_len:], start=chain_len):
                    conn.execute(
                        "INSERT INTO session_events(session_id, seq, msg)"
                        " VALUES (?,?,?)",
                        (session_id, i,
                         json.dumps(m, ensure_ascii=False)))
                merged = list(messages)
            else:
                # Full rebuild (no chain / compaction / fork / shorter than
                # chain / tail mismatch / collision retry): merge against
                # the event log, collapse the chain, rewrite the events.
                existing_msgs = _read_events(conn, session_id)
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
                # 0.115.0: metadata + indexed columns only — the transcript
                # JSON is never written here (the event log is the store).
                # Small meta (todos/plan/wip/metrics) rides in the shell.
                conn.execute(
                    "INSERT INTO sessions (session_id, data, updated_at,"
                    " revision, title, cwd, model, source, image_shas,"
                    " preview, n_msg)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(session_id) DO UPDATE SET"
                    " data=excluded.data,"
                    " updated_at=excluded.updated_at,"
                    " revision=excluded.revision,"
                    " title=COALESCE(excluded.title, sessions.title),"
                    " cwd=COALESCE(excluded.cwd, sessions.cwd),"
                    " model=COALESCE(excluded.model, sessions.model),"
                    " source=COALESCE(excluded.source, sessions.source),"
                    " image_shas=excluded.image_shas,"
                    " preview=excluded.preview,"
                    " n_msg=excluded.n_msg",
                    (session_id, shell_blob,
                     float((meta or {}).get("updated_at") or now),
                     live_revision + 1,
                     (meta or {}).get("title") or existing.get("title"),
                     (meta or {}).get("cwd") or existing.get("cwd"),
                     (meta or {}).get("model") or existing.get("model"),
                     (meta or {}).get("source") or existing.get("source"),
                     _shas_of({"messages": merged}),
                     _preview_of(merged),
                     _n_msg_of(merged)))
                # The FTS mirror of a rebuilt transcript is stale at every
                # seq — drop it; the next indexed save (or the next search's
                # lazy backfill) rebuilds it.
                try:
                    conn.execute(
                        "DELETE FROM fts_msgs WHERE sid = ?", (session_id,))
                except sqlite3.OperationalError:
                    pass  # no FTS5 here

            # Shared tail: chain upsert. On a rebuild the sessions upsert
            # above already set the indexed columns; on a fast append the
            # UPDATE below touches only the columns the caller supplied.
            conn.execute(
                "INSERT OR REPLACE INTO session_chain"
                " (session_id, chain_len, updated_at) VALUES (?,?,?)",
                (session_id, len(merged), now))
            if fast:
                # 0.129.x: the indexed listing columns must track an append:
                # n_msg grows by the newly appended non-system messages;
                # preview is stable (first user message never changes) and
                # is set only if a pre-0.129 row never computed one.
                added = _n_msg_of(messages[chain_len:])
                sets = ["updated_at = ?",
                        "revision = COALESCE(revision, 0) + 1",
                        "n_msg = COALESCE(n_msg, 0) + ?"]
                vals: list[Any] = [
                    float((meta or {}).get("updated_at") or now), added]
                for col in ("title", "cwd", "model", "source"):
                    v = (meta or {}).get(col)
                    if v is not None:
                        sets.append(f"{col} = ?")
                        vals.append(v)
                vals.append(session_id)
                conn.execute(
                    f"UPDATE sessions SET {', '.join(sets)}"
                    " WHERE session_id = ?", vals)
                pv = _preview_of(messages)
                if pv is not None:
                    try:
                        conn.execute(
                            "UPDATE sessions SET preview = ?"
                            " WHERE session_id = ? AND preview IS NULL",
                            (pv, session_id))
                    except sqlite3.Error:
                        pass
                # Fast path: the shell's small meta must still refresh —
                # metrics totals and the plan/wip block change every turn.
                if shell_blob != "{}":
                    try:
                        conn.execute(
                            "UPDATE sessions SET data = ?"
                            " WHERE session_id = ?", (shell_blob, session_id))
                    except sqlite3.Error:
                        pass
            if fast:
                # 0.108.6: keep the indexed sha set current — a union with
                # the old value stays correct (events are append-only; a sha
                # once referenced by a committed event stays referenced until
                # the GC deletes that sidecar, which only happens when it is
                # in the live-set).
                try:
                    new_shas = _shas_of({"messages": messages})
                    old = conn.execute(
                        "SELECT image_shas FROM sessions WHERE session_id = ?",
                        (session_id,)).fetchone()
                    both = set(json.loads(old[0])) if old and old[0] else set()
                    if new_shas:
                        both |= set(json.loads(new_shas))
                    conn.execute(
                        "UPDATE sessions SET image_shas = ? WHERE session_id = ?",
                        (json.dumps(sorted(both)) if both else None, session_id))
                except (sqlite3.Error, ValueError, TypeError):
                    pass  # scan falls back to parsing
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
    """Sync one session's per-message FTS index (0.115.0).

    State-driven, mirroring the events fast path: the indexed row set is
    compared against the transcript by count and tail content, and only
    the genuinely new tail is inserted. Building a full body scales with
    the transcript, so the write path only calls this on indexed saves
    (update_index=True) — mid-turn autosaves skip it, exactly like the
    2.3b design where the FTS body was backfilled lazily by the next
    search.

    A full reindex (session shrank, index gap, tail mismatch — the
    compaction/fork cases) deletes and rebuilds every row for the sid,
    same worst case as the pre-0.115 whole-body rewrite; the steady state
    (turn appended) is O(new messages).
    """
    own = conn is None
    if own:
        conn = _open()
        if conn is None:
            return
    try:
        from ._sessions_io import _flatten_message_text

        def _body_of(m: Any) -> str:
            if not isinstance(m, dict):
                return ""
            txt = _flatten_message_text(m)
            return f"[{m.get('role', '?')}] {txt}" if txt else ""

        now = time.time()
        with conn:
            try:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(MAX(seq), -1) FROM fts_msgs"
                    " WHERE sid = ?", (session_id,)).fetchone()
                cnt, max_seq = int(row[0]), int(row[1])
                n_msgs = len(messages or [])
                rebuild = True
                if cnt == 0:
                    pass  # fresh index: full insert is the append
                elif max_seq != cnt - 1:
                    pass  # gap/corruption: reindex
                elif max_seq >= n_msgs:
                    pass  # transcript shrank (compaction): reindex
                else:
                    tail = conn.execute(
                        "SELECT body FROM fts_msgs WHERE sid = ? AND seq = ?",
                        (session_id, max_seq)).fetchone()
                    # Same trick as the events fast path: a matching tail
                    # proves the shared prefix is intact, so rows
                    # [max_seq+1, n_msgs) are the only additions.
                    rebuild = (tail[0] or "") != _body_of(
                        (messages or [])[max_seq])
                if rebuild:
                    # 0.116.1: index-first delete — the external-content
                    # postings must be dropped while fts_msgs rows still
                    # identify them (same bug _fts_delete_conn had).
                    stale = [r[0] for r in conn.execute(
                        "SELECT rowid FROM fts_msgs WHERE sid = ?",
                        (session_id,))]
                    try:
                        for rid in stale:
                            conn.execute(
                                "DELETE FROM msgs_fts WHERE rowid = ?",
                                (rid,))
                    except sqlite3.OperationalError:
                        pass  # no FTS5 here
                    conn.execute(
                        "DELETE FROM fts_msgs WHERE sid = ?", (session_id,))
                    rows = [(session_id, i,
                             str(m.get("role", "?"))
                             if isinstance(m, dict) else "?",
                             _body_of(m), now)
                            for i, m in enumerate(messages or [])
                            if _body_of(m)]
                    conn.executemany(
                        "INSERT INTO fts_msgs(sid, seq, role, body, mtime)"
                        " VALUES (?,?,?,?,?)", rows)
                    # External-content sync: the index must be given the row
                    # text explicitly (a bare rowid insert indexes nothing).
                    try:
                        conn.execute(
                            "INSERT INTO msgs_fts(rowid, body)"
                            " SELECT rowid, body FROM fts_msgs WHERE sid = ?",
                            (session_id,))
                    except sqlite3.OperationalError:
                        pass  # no FTS5 here
                else:
                    conn.executemany(
                        "INSERT INTO fts_msgs(sid, seq, role, body, mtime)"
                        " VALUES (?,?,?,?,?)",
                        [(session_id, i, str(m.get("role", "?")),
                          _body_of(m), now)
                         for i, m in enumerate(messages or [])
                         if i > max_seq and _body_of(m)])
                    try:
                        conn.execute(
                            "INSERT INTO msgs_fts(rowid, body)"
                            " SELECT rowid, body FROM fts_msgs"
                            " WHERE sid = ? AND seq > ?",
                            (session_id, max_seq))
                    except sqlite3.OperationalError:
                        pass  # no FTS5 here
                # The tiny metadata index is upserted whole — title and
                # description are two short strings.
                try:
                    mrow = conn.execute(
                        "SELECT rowid FROM fts_meta WHERE sid = ?",
                        (session_id,)).fetchone()
                    if mrow:
                        conn.execute(
                            "DELETE FROM meta_fts WHERE rowid = ?", (mrow[0],))
                        conn.execute(
                            "DELETE FROM fts_meta WHERE sid = ?",
                            (session_id,))
                    cur = conn.execute(
                        "INSERT INTO fts_meta(sid, title, description, mtime)"
                        " VALUES (?,?,?,?)",
                        (session_id, title or "", description or "", now))
                    conn.execute(
                        "INSERT INTO meta_fts(rowid, title, description)"
                        " SELECT rowid, title, description FROM fts_meta"
                        " WHERE rowid = ?", (cur.lastrowid,))
                except sqlite3.OperationalError:
                    pass  # no FTS5 here — fts_meta still serves deletes
            except sqlite3.OperationalError:
                pass  # no FTS5 here — the event log is the part that matters
    finally:
        if own:
            conn.close()


def _fts_delete_conn(conn: sqlite3.Connection, session_id: str) -> None:
    """Delete one session's FTS index rows (per-message + metadata) on the
    caller's connection/transaction. Best-effort: no FTS5 here degrades to
    deleting the real tables only.

    0.116.1: the external-content index rows must be deleted BEFORE the
    content rows — the subselect keys off fts_msgs, so a content-first
    delete left every posting in msgs_fts_data orphaned forever (a
    delete-everything session sweep kept a 500 MB index for 2 sessions
    of content)."""
    try:
        # capture the indexed rowids while the content rows still exist
        msg_rowids = [r[0] for r in conn.execute(
            "SELECT rowid FROM fts_msgs WHERE sid = ?", (session_id,))]
        meta_rowids = [r[0] for r in conn.execute(
            "SELECT rowid FROM fts_meta WHERE sid = ?", (session_id,))]
        try:
            for rid in msg_rowids:
                conn.execute("DELETE FROM msgs_fts WHERE rowid = ?", (rid,))
            for rid in meta_rowids:
                conn.execute("DELETE FROM meta_fts WHERE rowid = ?", (rid,))
        except sqlite3.OperationalError:
            pass  # no FTS5 here — the content deletes below still apply
        conn.execute("DELETE FROM fts_msgs WHERE sid = ?", (session_id,))
        conn.execute("DELETE FROM fts_meta WHERE sid = ?", (session_id,))
    except sqlite3.OperationalError:
        pass  # no FTS5 here — the transcript rows are what matter


def _delete_session_conn(conn: sqlite3.Connection, session_id: str) -> None:
    """Delete one session's rows (transcript + FTS + events + chain) on the
    caller's connection. The caller owns the transaction; this never opens or
    closes a connection, so a sweep can reuse one conn for many deletes."""
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    _fts_delete_conn(conn, session_id)
    conn.execute("DELETE FROM session_events WHERE session_id = ?",
                 (session_id,))
    conn.execute("DELETE FROM session_chain WHERE session_id = ?",
                 (session_id,))


def delete_session_row(session_id: str) -> bool:
    """Drop one session's rows (transcript + FTS). Best-effort."""
    conn = _open()
    if conn is None:
        return False
    try:
        with conn:
            _delete_session_conn(conn, session_id)
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


def list_summaries(limit: Optional[int] = None, offset: int = 0,
                   ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Column-only listing summaries, newest-first — the whole point of the
    listing columns: NO event-log read, NO transcript parse, O(page) work.

    Returns dicts in the exact `_summarize_session_data` shape. Rows whose
    preview/n_msg are NULL (pre-0.129 stores) come back with `heal` set to
    True — the caller may compute and backfill those bounded rows."""
    from ._sessions_io import _short_id
    _ensure_migrated()
    conn = _open()
    if conn is None:
        return []
    try:
        sql = ("SELECT s.session_id, s.updated_at, s.title, s.preview,"
               " s.n_msg, s.model, s.source, s.cwd, s.image_shas,"
               " m.description FROM sessions s"
               " LEFT JOIN fts_meta m ON m.sid = s.session_id")
        params: list[Any] = []
        if ids is not None:
            if not ids:
                return []
            sql += (" WHERE s.session_id IN ("
                    + ",".join("?" * len(ids)) + ")")
            params.extend(ids)
        sql += " ORDER BY s.updated_at DESC"
        if offset:
            sql += f" OFFSET {int(offset)}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        out: list[dict[str, Any]] = []
        for (sid, updated_at, title, preview, n_msg, model, source, cwd,
             shas, description) in conn.execute(sql, params):
            out.append({
                "id": sid,
                "short": _short_id(sid),
                "title": title or preview or "(empty)",
                "description": description,
                "preview": preview or "",
                "updated_at": float(updated_at or 0.0),
                "n": int(n_msg or 0),
                "model": model,
                "source": source,
                "cwd": cwd,
                "sha256s": sorted(json.loads(shas)) if shas else [],
                "heal": preview is None or n_msg is None,
            })
        return out
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


def iter_rows() -> Iterator[tuple[str, Optional[dict[str, Any]]]]:
    """Yield (session_id, transcript dict) for every row — the listing index
    builder, the live-set scan fallback, and pruning read the whole store.

    0.115.0: metadata comes from the indexed columns (the blob is an empty
    shell) and messages always come from the event log — a listing never
    parses a transcript blob. Rows whose events fail to read yield
    (sid, None) so the caller can fall back."""
    conn = _open()
    if conn is None:
        return
    try:
        try:
            cur = conn.execute(
                "SELECT session_id, data, updated_at, revision, title, cwd,"
                " model, source FROM sessions")
            while True:
                rows = cur.fetchmany(64)
                if not rows:
                    break
                for sid, blob, updated_at, revision, title, cwd, model, source in rows:
                    base: dict[str, Any]
                    if blob:
                        parsed = _row_to_data(sid, blob)
                        base = parsed if parsed is not None else {}
                    else:
                        base = {}
                    base["id"] = sid
                    if revision is not None:
                        base["revision"] = int(revision)
                    if updated_at is not None:
                        base["updated_at"] = float(updated_at)
                    for k, v in (("title", title), ("cwd", cwd),
                                 ("model", model), ("source", source)):
                        if v is not None:
                            base[k] = v
                    try:
                        events = _read_events(conn, sid)
                    except sqlite3.Error:
                        events = None
                    if events is not None:
                        base["messages"] = events
                        data: Optional[dict[str, Any]] = base
                    else:
                        data = None
                    yield sid, data
        except sqlite3.Error:
            return
    finally:
        conn.close()


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
                "SELECT sid FROM fts_meta").fetchall()]
            msg_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT sid FROM fts_msgs").fetchall()]
        except sqlite3.OperationalError:
            return 0  # no FTS5 here
        have = {r[0] for r in conn.execute("SELECT session_id FROM sessions")}
        doomed = [sid for sid in set(fts_ids) | set(msg_ids)
                  if sid not in have]
        if not doomed:
            return 0
        with conn:
            for sid in doomed:
                _fts_delete_conn(conn, sid)
        return len(doomed)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def image_live_shas() -> tuple[set[str], set[str]]:
    """(live sha set, covered session ids) for the sidecar GC, read from the
    indexed `image_shas` column — one cheap column scan, no transcript
    parsing (0.108.6: the old iter_rows()-based scan peaked at ~1.5 GB RSS
    because every row's blob + event log was materialized just to collect
    image_ref parts).

    Rows with a NULL column (pre-0.108.6 writes, legacy stubs) fall back to
    parsing just those transcripts via iter_rows(); a corrupt/unreadable row
    contributes nothing, exactly as before. sqlite3.Error degrades to
    ({}, set()) — the caller's 60s sidecar age grace covers the miss."""
    conn = _open()
    if conn is None:
        return set(), set()
    live: set[str] = set()
    covered: set[str] = set()
    null_rows: list[str] = []
    try:
        for sid, shas in conn.execute(
                "SELECT session_id, image_shas FROM sessions"):
            covered.add(sid)
            if not shas:
                null_rows.append(sid)
                continue
            try:
                live |= set(json.loads(shas))
            except (ValueError, TypeError):
                null_rows.append(sid)
    except sqlite3.Error:
        return set(), set()
    finally:
        conn.close()
    if null_rows:
        # Parse only the rows the column can't answer — a pre-0.108.6 store
        # upgrades lazily as sessions are saved. Each target is read with
        # its own single-row SELECT (load_session-like), never a whole-table
        # cursor: a store with a few permanent NULLs (legacy rows never
        # re-saved, listing stubs) must not drag every big blob past the
        # parser on every sweep just to reach its 20 targets.
        conn = _open()
        if conn is not None:
            try:
                for sid in null_rows:
                    row = conn.execute(
                        "SELECT data FROM sessions WHERE session_id = ?",
                        (sid,)).fetchone()
                    if not row:
                        covered.discard(sid)  # no row: let the file fallback try
                        continue
                    data = _row_to_data(sid, row[0])
                    if data is None:
                        covered.discard(sid)  # unreadable: file fallback
                        continue
                    live |= _shas_live(data)
            except sqlite3.Error:
                pass
            finally:
                conn.close()
    return live, covered


def _shas_live(data: dict[str, Any]) -> set[str]:
    try:
        from ._images import _collect_message_image_refs
        return _collect_message_image_refs(data.get("messages") or [])
    except Exception:
        return set()


def retention_sweep(conn: Optional[sqlite3.Connection] = None) -> dict[str, int]:
    """Opt-in age x size session retention (0.104.x). Disabled by default.

    A store that silently deletes must never be the default, so every gate is
    OFF when its env value is 0 (the default). Two independent gates, both
    checked at sweep time against each session's `updated_at` and event
    count (the committed `session_chain.chain_len`; a legacy stub row with no
    chain row counts as 0 messages, so it is a gate-1 candidate when aged):

    1. Small-session expiry — NX_SESSION_RETAIN_SMALL_DAYS (default 0 =
       disabled): sessions with fewer than NX_SESSION_RETAIN_SMALL_MAX_MSGS
       events (default 10) AND older than N days are deleted.
    2. Hard floor — NX_SESSION_RETAIN_FLOOR_MSGS (default 0 = disabled):
       sessions with >= this many events are NEVER auto-deleted by gate 1,
       no matter their age (the "100 messages, don't touch them" rule).
       Optionally NX_SESSION_RETAIN_FLOOR_DAYS (default 0 = disabled): even
       big sessions expire after this many days; 0 = keep forever. The floor
       protects gate 1 only — gate 2 applies regardless of size.

    Returns {"expired_small": n, "expired_floor": n}. With every gate
    disabled this returns zeros without touching the DB. sqlite3.Error never
    propagates: a failing sweep returns zeros.
    """
    small_days = _env_int("NX_SESSION_RETAIN_SMALL_DAYS", 0)
    small_max = _env_int("NX_SESSION_RETAIN_SMALL_MAX_MSGS", 10)
    floor_msgs = _env_int("NX_SESSION_RETAIN_FLOOR_MSGS", 0)
    floor_days = _env_int("NX_SESSION_RETAIN_FLOOR_DAYS", 0)
    # Fast path: nothing to do, don't even open the store.
    if small_days <= 0 and floor_days <= 0:
        return {"expired_small": 0, "expired_floor": 0}
    if small_days > 0 and small_max <= 0:
        small_days = 0  # a non-positive size threshold disables gate 1
    own = conn is None
    if own:
        conn = _open()
    if conn is None:
        return {"expired_small": 0, "expired_floor": 0}
    expired_small = 0
    expired_floor = 0
    now = time.time()
    try:
        rows = conn.execute(
            "SELECT s.session_id, c.chain_len, s.updated_at FROM sessions s"
            " LEFT JOIN session_chain c USING (session_id)").fetchall()
        small: list[str] = []
        floor: list[str] = []
        for sid, chain_len, updated_at in rows:
            n_msgs = int(chain_len) if chain_len is not None else 0
            age_days = (now - float(updated_at or 0.0)) / 86400.0
            if (floor_days > 0 and age_days >= floor_days):
                floor.append(sid)
                continue
            if (small_days > 0 and n_msgs < small_max
                    and age_days >= small_days):
                if not (floor_msgs > 0 and n_msgs >= floor_msgs):
                    small.append(sid)
        if not small and not floor:
            return {"expired_small": 0, "expired_floor": 0}
        with conn:
            for sid in small:
                _delete_session_conn(conn, sid)
                expired_small += 1
            for sid in floor:
                _delete_session_conn(conn, sid)
                expired_floor += 1
    except sqlite3.Error:
        return {"expired_small": 0, "expired_floor": 0}
    finally:
        if own:
            conn.close()
    return {"expired_small": expired_small, "expired_floor": expired_floor}


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
