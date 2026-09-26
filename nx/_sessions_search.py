"""Session content search (FTS5) and transcript export.

`nx sessions <query>` and `/sessions <query>` have always filtered by title/
description substring only — the answer to "which session touched _turn.py?"
was scrolling. `search_sessions()` queries the FTS5 table of the session
store (`<sessions_dir>/sessions.sqlite`, see _sessiondb — 0.101.0 the
transcripts and their search index share one database), so a content query
scans every saved chat in milliseconds. Pure stdlib (sqlite3 ships FTS5 in
every CPython ≥3.7 build nx supports).

The FTS body is written by the indexed save path (`_write_session` →
`refresh_fts`) and lazily backfilled for sessions that predate it, so a
fresh store still answers content queries. Deleting the sessions.sqlite
file is always safe (the next save/list re-imports any legacy mirror
files, and search rebuilds from there).

`export_markdown()` renders a session JSON as a readable markdown transcript
for /export md.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Optional

from . import _state as st

_INDEX_FILENAME = "sessions.sqlite"


def _index_path() -> str:
    from ._sessions_fs import _sessions_dir
    return os.path.join(_sessions_dir(), _INDEX_FILENAME)


def _connect() -> sqlite3.Connection:
    # The FTS table lives in the session store — nx._sessiondb owns the
    # connection/schema so transcript and search index cannot drift apart.
    from . import _sessiondb
    return _sessiondb._connect()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5("
        "session_id UNINDEXED, title, description, body, mtime UNINDEXED)")
    conn.commit()


def _session_body(data: dict[str, Any]) -> str:
    """The searchable text of a session: every message, flattened."""
    from ._sessions_io import _flatten_message_text
    parts = []
    for m in data.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        txt = _flatten_message_text(m)
        if txt:
            parts.append(f"[{role}] {txt}")
    return "\n".join(parts)


def index_session(session_id: str, data: dict[str, Any], mtime: float,
                  conn: Optional[sqlite3.Connection] = None) -> None:
    """Insert/replace one session's FTS row. Caller owns the connection
    unless given none (then a short-lived one is used). The listing row is
    updated separately by _storeindex.update_session from _write_session."""
    own = conn is None
    if own:
        conn = _open_store()
        if conn is None:
            return
    try:
        title = str(data.get("title") or "")
        desc = str(data.get("description") or "")
        body = _session_body(data)
        conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (session_id,))
        conn.execute(
            "INSERT INTO sessions_fts (session_id, title, description, body, mtime) "
            "VALUES (?, ?, ?, ?, ?)", (session_id, title, desc, body, mtime))
        conn.commit()
    finally:
        if own:
            conn.close()


def _open_store() -> Optional[sqlite3.Connection]:
    """Connection with the FTS schema ensured, or None when the store
    can't serve this call (callers degrade to substring search)."""
    conn = _connect()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
    except sqlite3.DatabaseError:
        conn.close()
        return None
    return conn


def _iter_session_files() -> list[tuple[str, float]]:
    from ._sessions_fs import _sessions_dir
    d = _sessions_dir()
    out = []
    try:
        for name in os.listdir(d):
            if not name.endswith(".json"):
                continue
            p = os.path.join(d, name)
            try:
                out.append((name[:-5], os.path.getmtime(p)))
            except OSError:
                continue
    except OSError:
        return []
    return out


def _indexed_mtime(conn: sqlite3.Connection, session_id: str) -> Optional[float]:
    row = conn.execute(
        "SELECT mtime FROM sessions_fts WHERE session_id = ?",
        (session_id,)).fetchone()
    return row[0] if row else None


def _refresh_index(force: bool = False,
                   conn: Optional[sqlite3.Connection] = None) -> int:
    """Backfill FTS rows for sessions that have a transcript row but no
    body yet (indexed saves write the body themselves; legacy imports and
    autosave-only writes do not). Also drops FTS rows whose transcript row
    is gone — /sessions <query> must never offer a deleted id.

    0.101.0: no per-file mtimes to compare — freshness is simply "does an
    FTS row exist for this transcript row". Returns sessions indexed.

    Swarm P11: accepts a caller-owned connection so search_sessions' old
    open-refresh-close + open-again shape becomes one."""
    from . import _sessiondb
    own = conn is None
    if own:
        conn = _open_store()
        if conn is None:
            return 0
    n = 0
    try:
        try:
            fts_ids = {r[0] for r in conn.execute(
                "SELECT session_id FROM sessions_fts").fetchall()}
        except sqlite3.OperationalError:
            return 0  # no FTS5 here
        # prune FTS rows whose transcript row is gone
        have = {r[0] for r in conn.execute(
            "SELECT session_id FROM sessions").fetchall()}
        stale = fts_ids - have
        for sid in stale:
            conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (sid,))
        if stale:
            conn.commit()
        # backfill missing bodies from the store itself
        for sid, data in _sessiondb.iter_rows():
            if not data or sid in fts_ids:
                continue
            _sessiondb.refresh_fts(
                sid, data.get("messages") or [],
                title=data.get("title"), description=data.get("description"),
                conn=conn)
            n += 1
    finally:
        if own:
            conn.close()
    return n


def search_sessions(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Full-text search over saved session bodies. Returns matches shaped like
    _list_sessions() rows plus `snippet` (an FTS5 highlight around the first
    body hit). Falls back to an empty list (never raises) when FTS5 is missing
    or the query is invalid FTS syntax — callers degrade to substring search.
    A lazy FTS backfill for store-only sessions runs on the same connection
    first (see _refresh_index)."""
    query = (query or "").strip()
    if not query:
        return []
    # Swarm P11: one connection serves the refresh AND the query.
    conn = _open_store()
    if conn is None:
        return []
    try:
        try:
            _refresh_index(conn=conn)
        except sqlite3.OperationalError:
            return []
        try:
            rows = conn.execute(
                "SELECT session_id, title, description, body, "
                "highlight(sessions_fts, 3, '[[', ']]') "
                "FROM sessions_fts WHERE sessions_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []  # bad FTS query (operators, quotes) — caller falls back
        out = []
        for sid, title, desc, _body, hl in rows:
            snippet = (hl or "").strip()
            # trim around the first highlight for display
            k = snippet.find("[[")
            if k > 80:
                snippet = "…" + snippet[k - 60:]
            out.append({"id": sid, "title": title, "description": desc,
                        "snippet": snippet, "matched": True})
        return out
    finally:
        conn.close()


def index_stats() -> dict[str, int]:
    """{sessions_indexed} for the /sessions status line."""
    try:
        conn = _open_store()
        if conn is None:
            return {"sessions_indexed": 0}
        try:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return {"sessions_indexed": row[0] if row else 0}
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return {"sessions_indexed": 0}


# --- /export md -------------------------------------------------------------

def export_markdown(data: dict[str, Any], max_tool_chars: int = 2000) -> str:
    """Render a session JSON as a readable markdown transcript."""
    from ._sessions_io import _flatten_message_text
    lines = [f"# {data.get('title') or '(untitled session)'}", ""]
    meta = []
    if data.get("id"):
        meta.append(f"id `{data['id']}`")
    if data.get("created_at"):
        meta.append(f"created {time.strftime('%Y-%m-%d %H:%M', time.localtime(data['created_at']))}")
    if data.get("source"):
        meta.append(f"source {data['source']}")
    if data.get("model"):
        meta.append(f"model {data['model']}")
    if data.get("description"):
        meta.append(f"_{data['description']}_")
    if meta:
        lines += [" · ".join(meta), ""]
    for m in data.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        if role == "system":
            continue
        txt = _flatten_message_text(m)
        tcs = m.get("tool_calls") or []
        if not txt and tcs:
            txt = "\n".join(
                f"→ {c.get('function', {}).get('name')}({c.get('function', {}).get('arguments', '')})"
                for c in tcs)
        elif tcs:
            txt += "\n" + "\n".join(
                f"→ {c.get('function', {}).get('name')}({c.get('function', {}).get('arguments', '')})"
                for c in tcs)
        if len(txt) > max_tool_chars and role in ("tool", "user"):
            txt = txt[:max_tool_chars] + "\n… [truncated]"
        if not txt:
            continue
        label = {"user": "🧑 user", "assistant": "🤖 assistant",
                 "tool": "🔧 tool"}.get(role, role)
        lines += [f"### {label}", "", txt, ""]
    usage = data.get("usage") or {}
    if usage:
        parts = [f"{k.replace('_', ' ')}: {v}" for k, v in sorted(usage.items()) if isinstance(v, (int, float))]
        if parts:
            lines += ["---", "", " · ".join(parts), ""]
    return "\n".join(lines)
