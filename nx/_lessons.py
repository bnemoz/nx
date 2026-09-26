"""Cross-session learning recall: BM25 over the session store's FTS index.

`recall_lessons` answers "has a past session already learned how to do X?" by
running an FTS5 MATCH over `sessions_fts` (nx/_sessiondb) and ranking the hits
by FTS5's built-in bm25() — so the "learning" is full-text relevance, not
title substring. The tool is read-only (no approval, like memory_search) and
degrades to `[]` on every failure shape (locked store, bad query syntax, no
FTS5) — a recall miss must never raise into a turn.

The result cap (NX_LESSONS_LIMIT, default 5) bounds the context a recall can
dump into a turn, the same contract memory_search has with
NX_MEMORY_SEARCH_LIMIT. `scope` narrows a hit to the current project
(relevance + cwd match) or widens it to every session in the store.
"""
from __future__ import annotations

import os
from typing import Any

from . import _state as st

DEFAULT_LIMIT = 5

_VALID_SCOPES = ("all", "project", "global")

# A hit's body is bounded before it enters context: a whole transcript is
# megabytes, and the model only needs the region around the match.
_SNIPPET_CHARS = 240


def _lessons_limit() -> int:
    """Result cap for recall_lessons. 0 = no cap. Read at call time so tests
    (and the /lessons handler) can set the env."""
    import nx as _m
    try:
        return max(0, _m._env_int("NX_LESSONS_LIMIT", DEFAULT_LIMIT))
    except Exception:
        return DEFAULT_LIMIT


def _snippet(body: str) -> str:
    """A bounded one-line-ish preview of a hit's FTS body: the first
    _SNIPPET_CHARS of the earliest non-empty role line. FTS5's snippet() is
    token-bounded and truncated badly (see _memindex), so the preview is built
    from the body text the row already carries."""
    for ln in body.splitlines():
        s = ln.strip()
        if s:
            out = s[:_SNIPPET_CHARS]
            if len(s) > _SNIPPET_CHARS:
                out += "…"
            return out
    return ""


def search_lessons(query: str, limit: int | None = None,
                   scope: str = "all") -> list[dict[str, Any]]:
    """BM25-ranked full-text search over every saved session transcript.

    Query syntax is FTS5 MATCH (AND by default, OR/quotes/NEAR supported);
    a query FTS5 rejects returns [] rather than raising. `limit=None` (the
    default) applies the NX_LESSONS_LIMIT cap (0 = unbounded); one connection
    serves the whole call — the lazy FTS backfill for store-only sessions
    runs on the same connection first (see _sessions_search._refresh_index)."""
    from . import _sessions_search
    query = (query or "").strip()
    if not query:
        return []
    if limit is None:
        limit = _lessons_limit()
    try:
        conn = _sessions_search._open_store()
    except Exception:
        return []  # corrupt/locked store — degrade, never raise into a turn
    if conn is None:
        return []
    try:
        _sessions_search._refresh_index(conn=conn)
    except Exception:
        pass  # a failing backfill degrades this call, never raises
    try:
        rows = conn.execute(
            "SELECT sessions_fts.session_id, sessions_fts.title, sessions.cwd, "
            "sessions_fts.body, bm25(sessions_fts) AS rank "
            "FROM sessions_fts JOIN sessions USING (session_id) "
            "WHERE sessions_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit if limit > 0 else -1)).fetchall()
    except Exception:
        return []  # bad FTS syntax / no FTS5 / locked store → degrade
    root = None
    if scope == "project":
        _root = getattr(st, "PROJECT_ROOT", None)
        if _root:
            try:
                root = os.path.realpath(str(_root))
            except (TypeError, ValueError):
                root = None
    out = []
    for sid, title, cwd, body, rank in rows:
        if root and os.path.realpath(str(cwd or "")) != root:
            continue
        out.append({
            "id": sid,
            "title": title or "",
            "snippet": _snippet(body or ""),
            "score": round(float(rank), 3),
        })
    return out


def recall_lessons(query: str, scope: str = "all", **_: Any) -> str:
    """The tool body. Returns the BM25-ranked hits (or a quiet [no ...]
    note) — never raises into the turn loop."""
    query = (query or "").strip()
    if not query:
        return "ERROR: recall_lessons: `query` is empty — pass what to recall"
    if scope not in _VALID_SCOPES:
        scope = "all"
    hits = search_lessons(query, limit=_lessons_limit(), scope=scope)
    if not hits:
        return (f"[no past session matching {query!r} — nothing learned to "
                "recall yet]")
    out = [f"past sessions matching {query!r} (best first):"]
    for h in hits:
        title = h["title"] or "(untitled)"
        out.append(f"· {h['id'][:12]}  {title}  (bm25 {h['score']})")
        if h["snippet"]:
            out.append(f"    {h['snippet']}")
    out.append("read one with read_file on its store mirror, or resume it "
               "with /resume <short-id>")
    return "\n".join(out)
