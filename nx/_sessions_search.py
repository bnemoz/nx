"""Session content search (FTS5) and transcript export.

`nx sessions <query>` and `/sessions <query>` have always filtered by title/
description substring only — the answer to "which session touched _turn.py?"
was scrolling. `search_sessions()` builds (and incrementally maintains) a
SQLite FTS5 index over session bodies under `<sessions_dir>/.search.sqlite`,
so a content query scans every saved chat in milliseconds. Pure stdlib
(sqlite3 ships FTS5 in every CPython ≥3.7 build nx supports).

The index is a cache, never the source of truth: any session file newer than
its indexed row is re-indexed on the next search, a missing row is backfilled
on demand, and deleting the .search.sqlite file is always safe.

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

_INDEX_FILENAME = ".search.sqlite"


def _index_path() -> str:
    from ._sessions_fs import _sessions_dir
    return os.path.join(_sessions_dir(), _INDEX_FILENAME)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_index_path())
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


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
    """Insert/replace one session's row. Caller owns the connection unless
    given none (then a short-lived one is used)."""
    own = conn is None
    if own:
        conn = _connect()
        _ensure_schema(conn)
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


def _refresh_index(force: bool = False) -> int:
    """Re-index session files that are newer than their indexed row. Returns
    the number of sessions (re)indexed. Cheap: untouched sessions are skipped
    after a single mtime comparison."""
    from ._sessions_fs import _load_session
    conn = _connect()
    _ensure_schema(conn)
    n = 0
    try:
        for sid, mtime in _iter_session_files():
            if not force:
                seen = _indexed_mtime(conn, sid)
                if seen is not None and seen >= mtime:
                    continue
            data = _load_session(sid)
            if not data or not isinstance(data.get("messages"), list):
                continue
            index_session(sid, data, mtime, conn)
            n += 1
    finally:
        conn.close()
    return n


def search_sessions(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Full-text search over saved session bodies. Returns matches shaped like
    _list_sessions() rows plus `snippet` (an FTS5 highlight around the first
    body hit). Falls back to an empty list (never raises) when FTS5 is missing
    or the query is invalid FTS syntax — callers degrade to substring search."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        n = _refresh_index()
    except sqlite3.OperationalError:
        return []
    conn = _connect()
    _ensure_schema(conn)
    try:
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
        conn = _connect()
        _ensure_schema(conn)
        try:
            row = conn.execute("SELECT COUNT(*) FROM sessions_fts").fetchone()
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
