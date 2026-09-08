"""The consolidated session-store index (2.3b): one SQLite database.

Replaces two stores that never reconciled:
- the `.index/<session_id>.json` shards (per-session listing summaries,
  written by every indexed save), and
- the `.search.sqlite` FTS5 cache (bodies, written only lazily by a
  search — so a deleted session's rows were never pruned and
  `/sessions <query>` could offer a dead id).

Both now live in `<sessions_dir>/.store.sqlite`:

    sessions      — one row per session: the listing summary (exactly the
                    fields `_summarize_session_data` produced for the
                    shards, including the 2.3a `sha256s` image live-set)
                    plus the file mtime the row mirrors
    sessions_fts  — FTS5 over title/description/body, keyed by session_id

Everything here is a cache, never the source of truth: the session JSON
files on disk are the store. A missing/corrupt/locked database degrades to
the pre-index behavior (full per-file parse) — helpers return empty or
swallow, never raise, into listing/GC paths. A corrupt file is set aside
(`.store.sqlite.corrupt`) and a fresh one is created rather than letting
one bad write poison every later listing.

Stale-row pruning: because the store is written from `_write_session`
(indexed saves) and read for GC live-sets, rows whose session JSON is gone
are pruned opportunistically on indexed saves (`prune_stale`) and by hand
via `nx sessions --prune`. Deletion (`/delete`) removes the row directly.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

from ._logging import _log_event
from ._private import open_private, private_dir

STORE_FILENAME = ".store.sqlite"

# Lexicographic timestamp marker for a row whose FTS body has never been
# built (search backfills it on demand, exactly like the .search.sqlite
# cache did).
_FTS_UNINDEXED_MTIME = 0.0


def _sessions_dir() -> str:
    # Local import: _sessions_fs is the store's primary *caller* and reads
    # st.SESSION_HOME lazily; a top-level import would be circular.
    from ._sessions_fs import _sessions_dir as _d
    return _d()


def _store_path() -> str:
    return os.path.join(_sessions_dir(), STORE_FILENAME)


def _open_file_private(path: str) -> None:
    """Create `path` as a 0600 file so the umask can't leak session
    metadata (titles, previews — potentially secrets) to other users."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)


def _set_aside_corrupt() -> None:
    """Move an unusable store aside so a fresh one can be created. The
    data is a cache — nothing is lost that a parse fallback can't rebuild."""
    path = _store_path()
    try:
        os.replace(path, path + ".corrupt")
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass


def _connect() -> sqlite3.Connection:
    d = _sessions_dir()
    try:
        private_dir(d)
    except OSError:
        pass  # a read-only dir still serves existing rows
    path = _store_path()
    if not os.path.exists(path):
        try:
            _open_file_private(path)
        except OSError:
            pass
    conn = sqlite3.connect(path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions("
        "session_id TEXT PRIMARY KEY,"
        "title TEXT, description TEXT, preview TEXT, short TEXT,"
        "updated_at REAL, n INTEGER, model TEXT, source TEXT, cwd TEXT,"
        "sha256s TEXT, mtime REAL)")
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
    serve this call (unreadable dir, locked/lost file, no FTS5). Every
    caller treats None as "fall back to a full parse"."""
    for attempt in (1, 2):
        try:
            conn = _connect()
            _ensure_schema(conn)
            return conn
        except sqlite3.OperationalError as e:
            # locked / disk error / no such dir — one retry after setting a
            # corrupt file aside (a message like "file is not a database")
            if attempt == 2 or "not a database" not in str(e):
                _log_event("error", {"where": "storeindex", "error": str(e)})
                return None
            _set_aside_corrupt()
        except sqlite3.DatabaseError:
            _set_aside_corrupt()
    return None


# --- row <-> summary shaping --------------------------------------------------

_ROW_FIELDS = ("title", "description", "preview", "short", "updated_at",
               "n", "model", "source", "cwd")


def _row_to_summary(sid: str, row: tuple) -> dict[str, Any]:
    (title, description, preview, short, updated_at, n, model, source,
     cwd, sha256s) = row
    try:
        shas = json.loads(sha256s) if sha256s else []
        if not isinstance(shas, list):
            shas = []
    except (ValueError, TypeError):
        shas = []
    return {
        "id": sid,
        "short": short or (sid.rsplit("-", 1)[-1] if sid and "-" in sid else sid),
        "title": title,
        "description": description,
        "preview": preview or "",
        "updated_at": updated_at or 0,
        "n": n or 0,
        "model": model,
        "source": source,
        "cwd": cwd,
        "sha256s": [s for s in shas if isinstance(s, str) and s],
    }


def _summary_to_row(summary: dict[str, Any]) -> tuple:
    return (
        summary.get("title"),
        summary.get("description"),
        summary.get("preview"),
        summary.get("short"),
        summary.get("updated_at", 0),
        summary.get("n", 0),
        summary.get("model"),
        summary.get("source"),
        summary.get("cwd"),
        json.dumps(summary.get("sha256s") or [], ensure_ascii=False),
    )


_UPSERT_SQL = (
    "INSERT INTO sessions (session_id, title, description, preview, short,"
    " updated_at, n, model, source, cwd, sha256s, mtime)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")


# --- public surface ------------------------------------------------------------

def update_session(session_id: str, data: dict[str, Any], mtime: float,
                   conn: Optional[sqlite3.Connection] = None) -> None:
    """Refresh one session's listing row AND its FTS row from the full
    session `data`. Replaces the shard write of the old `_update_index`
    plus the lazily-built FTS row: one save keeps both in step."""
    from ._sessions_io import _summarize_session_data
    from ._sessions_search import _session_body
    summary = _summarize_session_data(session_id, data)
    own = conn is None
    if own:
        conn = _open()
        if conn is None:
            return
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?",
                         (session_id,))
            conn.execute(_UPSERT_SQL,
                         (session_id,) + _summary_to_row(summary) + (mtime,))
            try:
                conn.execute("DELETE FROM sessions_fts WHERE session_id = ?",
                             (session_id,))
                conn.execute(
                    "INSERT INTO sessions_fts (session_id, title, description,"
                    " body, mtime) VALUES (?, ?, ?, ?, ?)",
                    (session_id, summary.get("title") or "",
                     summary.get("description") or "",
                     _session_body(data), mtime))
            except sqlite3.OperationalError:
                pass  # no FTS5 here — the listing row is the part that matters
    finally:
        if own:
            conn.close()

def load_index() -> dict[str, Any]:
    """The listing index: {session_id: summary dict} — the shape the
    `.index/` shards used to return, straight from the store. Callers
    (_list_sessions) still verify against the on-disk file list."""
    conn = _open()
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT session_id, title, description, preview, short,"
            " updated_at, n, model, source, cwd, sha256s"
            " FROM sessions").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {row[0]: _row_to_summary(row[0], row[1:]) for row in rows}


def image_live_set() -> tuple[set[str], set[str]]:
    """(live shas, session ids whose row answered) — the GC live-set from
    the rows, replacing the shard scan. A session with no row (legacy
    file, mid-write) is simply absent from `covered` so the caller parses
    that transcript, exactly the old shard fallback."""
    conn = _open()
    if conn is None:
        return set(), set()
    try:
        rows = conn.execute("SELECT session_id, sha256s FROM sessions").fetchall()
    except sqlite3.Error:
        return set(), set()
    finally:
        conn.close()
    live: set[str] = set()
    covered: set[str] = set()
    for sid, sha256s in rows:
        try:
            shas = json.loads(sha256s) if sha256s else []
        except (ValueError, TypeError):
            shas = []
        if isinstance(shas, list):
            live.update(s for s in shas if isinstance(s, str) and s)
        covered.add(sid)
    return live, covered


def delete_session_row(session_id: str) -> None:
    """Drop one session's rows (listing + FTS) — the store-side half of
    `/delete`. Best-effort: a failed delete just leaves a stale row that
    prune_stale / --prune sweeps later."""
    conn = _open()
    if conn is None:
        return
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?",
                         (session_id,))
            try:
                conn.execute("DELETE FROM sessions_fts WHERE session_id = ?",
                             (session_id,))
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()


def write_full_index(idx: dict[str, Any]) -> None:
    """Rebuild the listing table from a full {id: summary} dict (tests and
    reconstruction after loss; no FTS rows — search backfills on demand).
    Ids in the store that `idx` omits are dropped."""
    conn = _open()
    if conn is None:
        return
    try:
        with conn:
            conn.execute("DELETE FROM sessions")
            for session_id, summary in idx.items():
                conn.execute(
                    _UPSERT_SQL,
                    (session_id,) + _summary_to_row(summary) + (0.0,))
    finally:
        conn.close()


def prune_stale() -> int:
    """Delete rows whose session JSON is gone (crash / manual rm — /delete
    already removes its own row). Returns rows pruned. O(#rows) os.stat
    calls, so it runs on indexed saves and from --prune."""
    conn = _open()
    if conn is None:
        return 0
    doomed: list[str] = []
    try:
        rows = conn.execute("SELECT session_id FROM sessions").fetchall()
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
    d = _sessions_dir()
    for (sid,) in rows:
        if not os.path.exists(os.path.join(d, f"{sid}.json")):
            doomed.append(sid)
    if not doomed:
        return 0
    conn = _open()
    if conn is None:
        return 0
    try:
        with conn:
            for sid in doomed:
                conn.execute("DELETE FROM sessions WHERE session_id = ?",
                             (sid,))
                try:
                    conn.execute("DELETE FROM sessions_fts WHERE session_id = ?",
                                 (sid,))
                except sqlite3.OperationalError:
                    pass
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
    return len(doomed)


def store_stats() -> dict[str, int]:
    """Row counts for the CLI status line."""
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


# --- migration -----------------------------------------------------------------

def _read_legacy_shards() -> dict[str, dict[str, Any]]:
    """The pre-2.3b `.index/<sid>.json` shards, as {sid: summary}. Local
    copy of the old read path (the one in _sessions_fs now reads THIS
    module's store), tolerant of corrupt shards."""
    out: dict[str, dict[str, Any]] = {}
    from ._sessions_fs import _index_dir
    try:
        with os.scandir(_index_dir()) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    with open(entry.path, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, IOError, json.JSONDecodeError):
                    continue
                if isinstance(data, dict) and entry.name[:-5]:
                    out[entry.name[:-5]] = data
    except OSError:
        return {}
    return out


def migrate_legacy() -> int:
    """Import whatever `.index/` shards (and any `.search.sqlite` rows)
    still exist, then retire both legacy files. Runs whenever legacy files
    are present: the first save/list after an upgrade from ≤0.45 folds them
    in; afterwards the files are gone and this is a no-op stat pair."""
    d = _sessions_dir()
    has_shards = os.path.isdir(os.path.join(d, ".index"))
    has_search = os.path.exists(os.path.join(d, ".search.sqlite"))
    if not has_shards and not has_search:
        return 0
    n = 0
    shards = _read_legacy_shards() if has_shards else {}
    if shards:
        conn = _open()
        if conn is not None:
            try:
                have = {r[0] for r in conn.execute(
                    "SELECT session_id FROM sessions").fetchall()}
                d_path = d
                with conn:
                    for sid, summary in shards.items():
                        if sid in have:
                            continue
                        mtime = 0.0
                        try:
                            mtime = os.path.getmtime(
                                os.path.join(d_path, f"{sid}.json"))
                        except OSError:
                            mtime = float(summary.get("updated_at") or 0.0)
                        conn.execute(_UPSERT_SQL,
                                     (sid,) + _summary_to_row(summary) + (mtime,))
                        n += 1
            finally:
                conn.close()
    # FTS bodies can't be lifted from the old FTS db without trusting its
    # session_id column; the on-demand backfill in search_sessions rebuilds
    # them from the transcripts, so drop the old file rather than carry it.
    try:
        os.remove(os.path.join(d, ".search.sqlite"))
    except OSError:
        pass
    for suffix in ("-wal", "-shm"):
        try:
            os.remove(os.path.join(d, ".search.sqlite" + suffix))
        except OSError:
            pass
    if has_shards:
        import shutil
        try:
            shutil.rmtree(os.path.join(d, ".index"))
        except OSError:
            pass
    return n
