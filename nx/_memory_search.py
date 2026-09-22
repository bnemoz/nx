"""Memory search: the ranking, the scanner that serves `memory_search` /
`memory_delete` and `/memory remember`, and the model-callable search/delete
tools.

Split out of `_memory_cmd.py` (which keeps the `/memory` slash-command UI).
`memory_search` accelerates its per-dir scan with the FTS5 index in
`nx/_memindex.py` (a self-healing cache; NX_MEMORY_FTS=0, or a sqlite build
without FTS5/trigram, falls back to the plain scan); hits from either path are
re-scored by the same `_rank_hit`, so ranking, the cap and the overflow note
are identical whichever path served the rows.

Searches keep the contract: every query term must appear, title hits rank
above body hits, newest wins ties, and results are capped at
NX_MEMORY_SEARCH_LIMIT so a big store can't dump an unbounded context blob
into a turn. `/memory remember` shares the scan (unlimited) for its previews."""
from __future__ import annotations

import os
from typing import Any, NamedTuple

from . import _state as st
from ._memory_store import (
    _iter_memory_dirs, _iter_notes, _memory_confirm_preview, _read_note_preview,
)

# Cap on memory_search results. A store that outgrows the cap says so
# in-band ("… N more not shown") instead of silently hiding or flooding.
MEMORY_SEARCH_DEFAULT_LIMIT = 8


def _memory_search_limit() -> int:
    """Result cap for memory_search (/memory remember stays unlimited).
    0 = no cap. Read via nx._env_int at call time so tests can set the env."""
    import nx as _m
    try:
        return max(0, _m._env_int("NX_MEMORY_SEARCH_LIMIT", MEMORY_SEARCH_DEFAULT_LIMIT))
    except Exception:
        return MEMORY_SEARCH_DEFAULT_LIMIT


class _MemoryHit(NamedTuple):
    mtime: float
    name: str
    title: str
    snippet: str
    label: str
    score: float


def _rank_hit(content: str, query: str) -> float:
    """Score one memory against a multi-term query (the caller has already
    verified every term appears somewhere): a term in the `# Title` outweighs
    body occurrences, and body occurrences are capped so a term repeated all
    over one long note can't dominate ranking."""
    low = content.lower()
    title_low = None
    for ln in content.splitlines():
        if ln.startswith("# "):
            title_low = ln[2:].strip().lower()
            break
    score = 0.0
    for term in query.lower().split():
        if title_low and term in title_low:
            score += 3.0
        else:
            score += min(2.0, 0.5 * low.count(term))
    return score


def _memory_snippet(content: str, query: str, max_lines: int = 4) -> str:
    """First `max_lines` of a memory body (title line stripped) as a preview."""
    lines = content.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("# ") and not out:
            continue  # skip the H1 title (already shown separately)
        if not ln.strip():
            continue
        out.append(ln.strip())
        if len(out) >= max_lines:
            break
    return " / ".join(out)


def _memory_scan(
    query: str, scope: str, limit: int
) -> tuple[list[_MemoryHit], int, int]:
    """Scan the memory dirs for `query`. Every term must appear; ranking is
    title-hits above body-hits, then newest first. Returns
    (hits, overflow, total): overflow = how many ranked hits the cap removed
    (0 when unlimited or nothing cut), total = matches before the cap.
    limit=0 → unlimited. Empty query → newest-first index of every note.

    Per dir, `nx._memindex.search_dir` (an mtime-reconciled FTS5 cache over
    the notes) supplies candidate rows without reading any file; the rows
    still pass the same lowercase-substring predicate and `_rank_hit`
    scoring, so results are identical to the plain scan up to whitespace
    folding between query characters (trigram matching is character-adjacent,
    not whitespace-exact) — the accelerator can only widen recall slightly.
    When it is off, unsupported, or returns no rows for a dir, the plain
    `_iter_notes` scan runs for that dir.
    """
    terms = query.lower().split()
    hits: list[_MemoryHit] = []
    for d, label in _iter_memory_dirs(scope):
        fts_rows = None
        if terms and all(len(t) >= 3 for t in terms):
            try:
                from ._memindex import search_dir
                fts_rows = search_dir(d, query)
            except Exception:
                fts_rows = None
        if fts_rows is not None:
            for name, mtime, title, body in sorted(
                    fts_rows, key=lambda r: r[0], reverse=True):
                low = body.lower()
                if not all(t in low for t in terms):
                    continue
                # Same preview builder as the plain scan (see _memindex: the
                # old FTS `snippet()` was token-bounded and truncated to ~12
                # chars, so the accelerator used to degrade every hit).
                hits.append(_MemoryHit(
                    mtime, name, title, _memory_snippet(body, query), label,
                    _rank_hit(body, query)))
            continue
        for name, _path, mtime, title, content in _iter_notes(d):
            if terms:
                low = content.lower()
                if not all(t in low for t in terms):
                    continue
                score = _rank_hit(content, query)
            else:
                score = 0.0
            hits.append(_MemoryHit(
                mtime, name, title, _memory_snippet(content, query), label, score))
    hits.sort(key=lambda h: (-h.score, -h.mtime))
    total = len(hits)
    if limit and total > limit:
        return hits[:limit], total - limit, total
    return hits, 0, total


def _scan_memory_entries(scope: str = "project") -> list[_MemoryHit]:
    """Every saved memory as a hit — no query, no snippet, no limit. The
    resolver behind memory_delete and `/memory delete`."""
    hits: list[_MemoryHit] = []
    for d, label in _iter_memory_dirs(scope):
        for name, _path, mtime, title, _content in _iter_notes(d, titles_only=True):
            hits.append(_MemoryHit(mtime, name, title, "", label, 0.0))
    hits.sort(key=lambda h: -h.mtime)  # newest first, same order memory_search uses
    return hits


def memory_search(query: str, scope: str = "project", **_: Any) -> str:
    """Keyword-search saved memories (global + project): title + snippet per
    match, best match first — title hits above body hits, newer above older.
    Read-only; no approval. Use this to recall prior learnings before
    re-deriving them.

    scope="project" searches both project and global memories; scope="global"
    searches only global. Every query term must appear. An empty query returns
    the newest titles as a quick index (so the model can discover what
    exists). Results are capped at NX_MEMORY_SEARCH_LIMIT (default 8); the
    response says how many were cut.
    """
    query = (query or "").strip()
    hits, overflow, total = _memory_scan(query, scope, _memory_search_limit())
    if not hits:
        return f"[no memories matching {query!r}]" if query else "[no memories saved yet]"
    if query:
        head = (f"memories matching {query!r} ({len(hits)} of {total}):" if overflow
                else f"memories matching {query!r} ({total}):")
    else:
        head = f"memories ({len(hits)} of {total}):" if overflow else f"memories ({total}):"
    out = [head]
    for hit in hits:
        out.append(f"[{hit.label}] {hit.title} ({hit.name})")
        out.append(f"  {hit.snippet}")
    if overflow:
        out.append(f"  … {overflow} more not shown — raise NX_MEMORY_SEARCH_LIMIT to see them")
    dirs: list[str] = []
    for d, _label in _iter_memory_dirs(scope):
        if d in dirs:
            continue
        if any(os.path.isfile(os.path.join(d, h.name)) for h in hits):
            dirs.append(d)
    if dirs:
        out.append("read a note with read_file <absolute path> — the (name) above is a bare filename; the files live in: "
                   + "; ".join(dirs))
    return "\n".join(out)


def memory_delete(target: str, scope: str = "project", **_: Any) -> str:
    """Delete one saved memory note by exact title, filename, or unique title
    prefix. An ambiguous or unknown target returns candidate titles instead of
    deleting. Goes through the approval gate like memory_write; on success it
    drops any cached memory_search results (see run_tool)."""
    target = (target or "").strip()
    if not target:
        return "ERROR: memory_delete requires a non-empty `target`"
    if scope not in ("project", "global"):
        scope = "project"
    entries = _scan_memory_entries(scope)
    if not entries:
        return "[no memories saved yet]"

    low = target.lower()
    exact = [h for h in entries
             if h.name.lower() == low or h.name[:-3].lower() == low
             or h.title.lower() == low]
    if len(exact) == 1:
        chosen = exact[0]
    else:
        cands = exact if exact else [
            h for h in entries if h.title.lower().startswith(low)]
        if len(cands) > 1:
            listing = "\n".join(f"  - {h.title} ({h.name})" for h in cands[:10])
            return (f"AMBIGUOUS: {target!r} matches {len(cands)} memories — "
                    f"delete by exact title or filename:\n{listing}")
        if not cands:
            recent = "\n".join(f"  - {h.title} ({h.name})" for h in entries[:5])
            return (f"no memory matching {target!r} — recent memories:\n{recent} "
                    f"(memory_search with an empty query lists all)")
        chosen = cands[0]

    path = None
    for d, _label in _iter_memory_dirs(scope):
        p = os.path.join(d, chosen.name)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return f"ERROR: memory file for {chosen.title!r} vanished"

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if not _confirm_memory(
            f"delete memory {path} ({size} bytes)",
            preview=_memory_confirm_preview(chosen.title, _read_note_preview(path))):
        return "DENIED by user"
    try:
        os.unlink(path)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    return f"deleted {chosen.label} memory: {chosen.title}\nfile: {path}"


def _confirm_memory(action: str, preview: str | None = None) -> bool:
    """`_confirm` is monkeypatched by tests via the top-level module, so call
    it through `import nx as _m` — same pattern as _edit_tools. (Thin alias of
    the store's helper, kept here so this module is self-contained for its
    two gate sites.)"""
    import nx as _m
    return _m._confirm(action, preview=preview)
