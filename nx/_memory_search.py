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

from ._memory_store import (
    _annotate_drift, _drift_demote, _drift_root, _iter_memory_dirs, _iter_notes,
    _memory_confirm_preview, _parse_drift, _read_note_preview, _stamp_note_now,
)

# Cap on memory_search results. A store that outgrows the cap says so
# in-band ("… N more not shown") instead of silently hiding or flooding.
# With include_sessions=True it caps the FUSED list (notes + sessions
# together), so the cross-arm recall can't double the context a call dumps.
MEMORY_SEARCH_DEFAULT_LIMIT = 8

# Reciprocal rank fusion constant (0.158.x, `include_sessions=True`): a doc at
# 1-based position `rank` of a result list contributes 1 / (RRF_K + rank) to its
# fused score, so a doc ranked high in BOTH arms accumulates and rises.
# Fusion is rank-based *by necessity*: the two arms' score scales are not
# comparable — notes carry a hand-rolled relevance score (3.0 per title hit,
# 0.5 per body occurrence capped at 2.0, drift-demoted), sessions carry raw
# SQLite bm25() values (negative-is-better, magnitude scales with corpus size).
# No score crosses arms; only list position does. k=60 is the standard RRF
# constant (it damps the top ranks' dominance and makes rank 1 vs 2 a small
# gap), and **no per-list weighting is applied, deliberately**: weighting one
# arm would re-introduce exactly the calibration RRF exists to avoid. If one
# arm ever needs to dominate, that is a ranking fix inside that arm, not a
# fusion weight.
RRF_K = 60


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
    # Drift stamping: `score` stays the RAW relevance score (what _rank_hit /
    # the FTS path produced); freshness rides alongside it — `eff` is the score
    # actually sorted on (demoted), `ann` the short tag rendered on the hit.
    # Both default so older constructors and tests keep working.
    eff: float = 0.0
    ann: str = ""


def _hit_from(content: str, score: float, can_stamp: bool = False) -> tuple:
    """(eff_score, annotation) for one hit body: the raw score divided by
    (1 + drift weight) so stale notes sink, and the short freshness tag.

    `can_stamp=True` (this dir is the CURRENT project's own memory dir, which
    the scanner backfills) makes a note that predates stamping count as freshly
    stamped at HEAD — the same verdict the `_iter_notes` scanner reaches, so
    the FTS accelerator and the plain scan annotate identically. Global notes
    never get that treatment: another repo's HEAD would be a lie."""
    stamp = _parse_drift(content)
    if stamp is None and can_stamp:
        backfilled = _stamp_note_now(content, _drift_root())
        if backfilled is not None:
            stamp = _parse_drift(backfilled[0])
    return score / (1.0 + _drift_demote(stamp)), _annotate_drift(stamp)


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
                score = _rank_hit(body, query)
                eff, ann = _hit_from(body, score, label == "project")
                hits.append(_MemoryHit(
                    mtime, name, title, _memory_snippet(body, query), label,
                    score, eff, ann))
            continue
        for name, _path, mtime, title, content in _iter_notes(d):
            if terms:
                low = content.lower()
                if not all(t in low for t in terms):
                    continue
                score = _rank_hit(content, query)
            else:
                score = 0.0
            eff, ann = _hit_from(content, score, label == "project")
            hits.append(_MemoryHit(
                mtime, name, title, _memory_snippet(content, query), label, score,
                eff, ann))
    # Ranking: the demoted score (a note whose own files moved 40 commits ago
    # sinks below a fresh one of equal relevance), newest first on ties.
    hits.sort(key=lambda h: (-h.eff, -h.mtime))
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


class _FusedRow(NamedTuple):
    """One line of the fused (notes + sessions) presentation ranking.

    `key` is the row's **per-arm** identity — ("note", label, filename) or
    ("session", session id). A note and a session are different object kinds,
    so they never collapse into each other: a note titled after a session and
    that session itself both appear, each in its own arm's visual convention.
    The identity is per-arm on purpose; RRF here joins two lists into ONE
    presentation ranking, it does not decide that two kinds are the same doc.
    """
    key: tuple
    kind: str  # "note" | "session"
    note: Any = None     # _MemoryHit
    session: Any = None  # _lessons.search_lessons hit dict


def _rrf_fuse(
    lists: list[list[_FusedRow]], limit: int
) -> tuple[list[_FusedRow], int, int]:
    """Reciprocal rank fusion over already-ranked lists (see RRF_K).

    Each list is best-first; the item at 1-based position i accumulates
    1 / (RRF_K + i). An item appearing in more than one list therefore scores
    higher than either single-arm contribution alone and rises — that is the
    whole mechanism, and it needs no calibration between arms. Equal fused
    scores keep first-list-then-list-order (a stable, reproducible tie-break).
    Returns (fused rows, hidden, total). limit=0 → unlimited, hidden=0.
    """
    scores: dict[tuple, float] = {}
    order: dict[tuple, int] = {}
    items: dict[tuple, _FusedRow] = {}
    for rows in lists:
        for rank, row in enumerate(rows, start=1):
            scores[row.key] = scores.get(row.key, 0.0) + 1.0 / (RRF_K + rank)
            if row.key not in items:
                order[row.key] = len(order)
                items[row.key] = row
    ranked = sorted(scores, key=lambda k: (-scores[k], order[k]))
    total = len(ranked)
    kept = ranked if not limit else ranked[:limit]
    return [items[k] for k in kept], total - len(kept), total


def _fused_lines(row: _FusedRow) -> list[str]:
    """Render one fused row in the conventions its own arm already uses:
    notes keep the `[global]`/`[project]` + snippet-line shape of
    memory_search, sessions keep the `· short-id  title  (bm25 …)` shape of
    recall_lessons — so a fused block reads exactly like the two tools' own
    output, interleaved."""
    if row.kind == "note":
        h = row.note
        return [f"[{h.label}] {h.title} ({h.name}) {h.ann}".rstrip(),
                f"  {h.snippet}"]
    h = row.session or {}
    line = f"· {str(h.get('id') or '')[:12]}  {h.get('title') or '(untitled)'}"
    if h.get("score") is not None:
        line += f"  (bm25 {h['score']})"
    out = [line]
    if h.get("snippet"):
        out.append(f"    {h['snippet']}")
    return out


def _fused_memory_search(query: str, scope: str, limit: int) -> str:
    """The include_sessions=True arm: rank the note list and the session list
    for the SAME query, RRF-fuse them into one presentation ranking, and render
    it as one block. Cross-arm recall is what this buys: the curated note about
    a learning and the past session that produced it both rise when each ranks
    high in its own arm, instead of being buried in two separate tool results
    the model has to reconcile by hand.

    What fusion does HERE, concretely: a note and a session are different object
    kinds and are never identified with each other (see _FusedRow), so no row
    earns contributions from both lists on this path — a row's fused score is
    its own 1/(k + rank-in-its-arm), and the merge therefore *interleaves* the
    two arms by rank (note#1, session#1, note#2, …) rather than appending one
    arm below the other. That interleave is the point: a rank-1 session outranks
    a rank-3 note, so the note about a learning and the session that produced it
    land adjacent near the top of one block. The accumulation branch in
    _rrf_fuse stays general (any two lists that share keys fuse by summing), so
    introducing a cross-arm identity later — or fusing two note-side lists —
    needs no change here.
    """
    from ._lessons import search_lessons
    # The note arm stays uncapped (the scan already reads every note, so an
    # uncut list costs nothing and lets the "(N of M)" head count every real
    # note match). The session arm's window is the fused budget: it is a SQL
    # bound, and a candidate past rank `limit` could only ever land below the
    # limit-th ranked row of the other arm anyway. This deliberately overrides
    # NX_LESSONS_LIMIT for this call — the fused list has ONE budget (see the
    # MEMORY_SEARCH_DEFAULT_LIMIT note); NX_MEMORY_SEARCH_LIMIT=0 leaves both
    # arms unbounded, which is what "0 = no cap" already means here.
    note_hits, _arm_overflow, _note_total = _memory_scan(query, scope, 0)
    lesson_scope = scope if scope in ("project", "global") else "project"
    try:
        lesson_hits = search_lessons(query, limit=limit, scope=lesson_scope)
    except Exception:
        lesson_hits = []  # a locked/corrupt store must not fail the note search
    lists = [[
        # Keyed per arm AND per label: a note file of the same name in the
        # global and the project dir is two distinct docs (the plain path lists
        # both), so the key must not let them collapse into one row.
        _FusedRow(("note", h.label, h.name), "note", note=h) for h in note_hits
    ], [
        _FusedRow(("session", h.get("id")), "session", session=h)
        for h in lesson_hits
    ]]
    rows, hidden, total = _rrf_fuse(lists, limit)
    if not rows:
        return f"[no memories or past sessions matching {query!r}]"
    head = (f"memories + past sessions matching {query!r} "
            f"({len(rows)} of {total}):" if hidden else
            f"memories + past sessions matching {query!r} ({total}):")
    out = [head]
    for row in rows:
        out.extend(_fused_lines(row))
    if hidden:
        out.append(f"  … {hidden} more not shown — raise NX_MEMORY_SEARCH_LIMIT "
                   "to see them")
    shown_notes = [r.note for r in rows if r.kind == "note"]
    if shown_notes:
        dirs_note = _memory_dirs_note(scope, shown_notes)
        if dirs_note:
            out.append(dirs_note)
    if any(r.kind == "session" for r in rows):
        out.append("read a session with read_file on its store mirror, or "
                   "resume it with /resume <short-id>")
    return "\n".join(out)


def _memory_dirs_note(scope: str, hits: list) -> str:
    """The trailing 'where the files actually live' pointer: only the dirs that
    contributed a shown hit. "" when no shown hit resolves to a file (the same
    silence the plain path keeps in that case)."""
    dirs: list[str] = []
    for d, _label in _iter_memory_dirs(scope):
        if d in dirs:
            continue
        if any(os.path.isfile(os.path.join(d, h.name)) for h in hits):
            dirs.append(d)
    if not dirs:
        return ""
    return ("read a note with read_file <absolute path> — the (name) above is a "
            "bare filename; the files live in: " + "; ".join(dirs))


def memory_search(query: str, scope: str = "project",
                  include_sessions: bool = False, **_: Any) -> str:
    """Keyword-search saved memories (global + project): title + snippet per
    match, best match first — title hits above body hits, newer above older.
    Read-only; no approval. Use this to recall prior learnings before
    re-deriving them.

    scope="project" searches both project and global memories; scope="global"
    searches only global. Every query term must appear. An empty query returns
    the newest titles as a quick index (so the model can discover what
    exists). Results are capped at NX_MEMORY_SEARCH_LIMIT (default 8); the
    response says how many were cut.

    include_sessions=True additionally runs the session-transcript arm
    (recall_lessons' BM25 index) with the same query + scope and returns the
    two lists FUSED by reciprocal rank (see RRF_K): a curated note and the past
    session that produced it rise together when each ranks high in its own arm,
    in one block instead of two tool results. The cap then bounds the fused
    list. Default False = notes only. No fusion for an empty query — there is
    nothing to rank.
    """
    query = (query or "").strip()
    if include_sessions and query:
        try:
            return _fused_memory_search(query, scope, _memory_search_limit())
        except Exception:
            pass  # the fused arm degrades to the plain note search, never a failed one
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
        out.append(f"[{hit.label}] {hit.title} ({hit.name}) {hit.ann}".rstrip())
        out.append(f"  {hit.snippet}")
    if overflow:
        out.append(f"  … {overflow} more not shown — raise NX_MEMORY_SEARCH_LIMIT to see them")
    if hits:
        dirs_note = _memory_dirs_note(scope, hits)
        if dirs_note:
            out.append(dirs_note)
    return "\n".join(out)


def memory_delete(target: str, scope: str = "project", confirm: bool = True,
                  **_: Any) -> str:
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
    if confirm and not _confirm_memory(
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
