"""FTS5 accelerator over the saved memory notes (2.6).

`memory_search` reads every note in every scope on every call. Fine at ten
notes; at a few hundred it is a full-store read per query. This module keeps
one SQLite FTS5 (trigram) index per memories dir — `<dir>/.memindex.sqlite` —
mirroring every note's body, and answers `search_dir()` from it after a cheap
mtime reconciliation (new/changed notes re-indexed, deleted notes pruned).

It is a cache, never the source of truth, with the same posture as the 2.3b
session store (`nx/_storeindex.py`): every public helper returns None/[] on
any problem (locked, corrupt, no FTS5, disabled) and the caller falls back to
the plain substring scan in `_memory_cmd._memory_scan` — which stays the
definition of memory-search semantics. The index can only change WHICH rows a
query finds before the substring re-check, never how they are ranked or
capped: hits are re-scored with `_rank_hit` over the indexed body.

Trigram tokenization makes the FTS MATCH a fixed-substring filter: the query
is phrase-quoted (`"x-header"`), so a hit requires the raw characters to
appear — the same shape the substring scan checks, modulo whitespace folding
between the characters. `search_dir` therefore also re-checks the lowercase
terms against the body before returning, so both paths answer from the same
predicate.

No `snippet()` column is selected (see `search_dir`): FTS5's snippet() is
token-bounded and returned a ~12-char fragment, so previews come from the body
the caller already has.

`compact()` is the one GC seam: the index is a pure cache, so when the store
shrinks (an optimize run archived notes) the orphaned pages are reclaimed
instead of bloating the file forever. It runs opportunistically rather than on
every reconcile, because a VACUUM rewrites the whole file.
"""
from __future__ import annotations

import os
import random
import sqlite3
import time
from typing import Optional

from ._logging import _log_event
from ._private import private_dir

INDEX_FILENAME = ".memindex.sqlite"

# Lexicographic marker matching _storeindex's: a row whose body was built
# before this feature (mtime 0) is simply re-indexed on the next reconcile.
_UNINDEXED_MTIME = 0.0

# Set once, per process, when this sqlite build cannot serve the virtual
# table (rare: every CPython build nx supports ships FTS5; trigram needs
# sqlite >= 3.34). Never un-set — but conftest rewinds module state per test,
# so a test simulating the failure can clear it.
_FTS_UNAVAILABLE = False


def _fts_supported() -> bool:
    """One in-memory probe for the exact DDL we depend on."""
    global _FTS_UNAVAILABLE
    if _FTS_UNAVAILABLE:
        return False
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE _probe USING fts5("
                "x, tokenize='trigram')")
        finally:
            conn.close()
        return True
    except sqlite3.Error:
        _FTS_UNAVAILABLE = True
        _log_event("error", {"where": "memindex", "error": "fts5/trigram unavailable"})
        return False


def enabled() -> bool:
    """The NX_MEMORY_FTS gate (default on; 0 = always the plain scan).
    Read at call time via the top-level module so tests can set the env."""
    import nx as _m
    try:
        return _m._env_int("NX_MEMORY_FTS", 1) != 0
    except Exception:
        return True


def _db_path(memories_dir: str) -> str:
    return os.path.join(memories_dir, INDEX_FILENAME)


def _set_aside_corrupt(path: str) -> None:
    """Move an unusable index aside so a fresh one is built. Nothing is lost:
    the notes on disk are the store and the next search re-indexes them."""
    try:
        os.replace(path, path + ".corrupt")
    except OSError:
        try:
            os.remove(path)
        except OSError:
            pass


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
        "name UNINDEXED, title, body, mtime UNINDEXED, "
        "tokenize='trigram')")
    conn.commit()


def _connect(memories_dir: str) -> sqlite3.Connection:
    path = _db_path(memories_dir)
    try:
        private_dir(memories_dir)
        if not os.path.exists(path):
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
    except OSError:
        pass  # a read-only dir still serves an existing index
    # Swarm P11: same contention posture as _storeindex — a shorter busy
    # wait (everything here degrades to a plain scan) instead of a 5 s block
    # per memory_search under a shared NX_HOME.
    ms = _busy_timeout_ms()
    conn = sqlite3.connect(path, timeout=ms / 1000.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={ms}")
    return conn


def _busy_timeout_ms() -> int:
    from ._env import _env_int
    return max(1, _env_int("NX_STORE_BUSY_TIMEOUT_MS", 750))


def _open(memories_dir: str) -> Optional[sqlite3.Connection]:
    """An open, schema-correct connection — or None when this dir's index
    cannot serve the call. One retry after setting a corrupt file aside."""
    if not _fts_supported() or not enabled():
        return None
    path = _db_path(memories_dir)
    conn = None
    last_err: Optional[Exception] = None
    # Swarm P11: busy-lock contention retries with jittered backoff (never a
    # set-aside); corrupt files keep the set-aside + retry path.
    for attempt in range(3):
        if attempt:
            time.sleep(0.02 * (2 ** (attempt - 1)) + random.random() * 0.01)
        try:
            conn = _connect(memories_dir)
            _ensure_schema(conn)
            return conn
        except sqlite3.OperationalError as e:
            last_err = e
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            msg = str(e)
            if "locked" in msg or "busy" in msg:
                continue  # contention: bounded retry, never a set-aside
            if attempt >= 1 or "not a database" not in msg:
                _log_event("error", {"where": "memindex", "error": msg})
                return None
            _set_aside_corrupt(path)
        except sqlite3.DatabaseError:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            _set_aside_corrupt(path)
    _log_event("error", {"where": "memindex", "error": str(last_err)})
    return None


# --- reconcile -----------------------------------------------------------------

def _fts_phrase(term: str) -> str:
    """One query term as a trigram-MATCH phrase. The wrapping quotes make the
    match a raw substring requirement; an inner quote would break out of the
    phrase, so it is stripped."""
    return '"%s"' % term.replace('"', "")


def _list_notes_on_disk(d: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    try:
        for name in os.listdir(d):
            if not name.endswith(".md"):
                continue
            try:
                out.append((name, os.path.getmtime(os.path.join(d, name))))
            except OSError:
                continue
    except OSError:
        return []
    out.sort(reverse=True)
    return out


def _index_note(conn: sqlite3.Connection, d: str, name: str, mtime: float) -> None:
    """Read one note and put its row into the index (or drop the row when the
    note became unreadable — the scan fallback wouldn't have served it)."""
    from ._memory_cmd import _read_memory_file
    read = _read_memory_file(os.path.join(d, name), name)
    conn.execute("DELETE FROM memories_fts WHERE name = ?", (name,))
    if read is None:
        return
    title, content = read
    conn.execute(
        "INSERT INTO memories_fts (name, title, body, mtime) VALUES (?, ?, ?, ?)",
        (name, title, content, mtime))


def reconcile(conn: sqlite3.Connection, d: str) -> None:
    """Make the index mirror the dir: re-index new/changed notes, drop rows
    whose note is gone. One transaction; a failing reconcile leaves the old
    rows in place (their mtime check simply repeats next search)."""
    rows = conn.execute("SELECT name, mtime FROM memories_fts").fetchall()
    indexed = {}
    for name, mtime in rows:
        try:
            indexed[str(name)] = float(mtime)
        except (TypeError, ValueError):
            indexed[str(name)] = _UNINDEXED_MTIME
    with conn:
        for name, mtime in _list_notes_on_disk(d):
            seen = indexed.pop(name, None)
            if seen is not None and seen >= mtime:
                continue
            _index_note(conn, d, name, mtime)
        for name in indexed:
            conn.execute("DELETE FROM memories_fts WHERE name = ?", (name,))


# --- query -----------------------------------------------------------------------

def search_dir(d: str, query: str) -> Optional[list[tuple[str, float, str, str]]]:
    """Answer `query` from this dir's index: a list of (name, mtime, title,
    body) tuples. The body is included so the caller can apply the exact
    substring predicate, `_rank_hit` scoring AND `_memory_snippet` previewing
    without touching the filesystem.

    Deliberately no `snippet()` column: FTS5's snippet() is token-, not
    context-, bounded, so at any cheap context size it returned a ~12-char
    fragment where the plain scan's `_memory_snippet` returned four body lines.
    The snippet is the only body content the model gets, so the accelerator was
    silently degrading recall; the caller now builds the preview from the body
    it already has.

    Returns None when the caller must fall back to the plain scan (FTS
    disabled/unsupported, index unusable, no query terms, or the FTS query
    itself failing). The reconciliation runs here, so a stale or missing index
    self-heals on every search that uses it.
    """
    terms = (query or "").lower().split()
    if not terms:
        return None
    conn = _open(d)
    if conn is None:
        return None
    try:
        reconcile(conn, d)
        match = " ".join(_fts_phrase(t) for t in terms)
        try:
            rows = conn.execute(
                "SELECT name, mtime, title, body "
                "FROM memories_fts WHERE memories_fts MATCH ?",
                (match,)).fetchall()
        except sqlite3.OperationalError as e:
            # a MATCH the trigram table cannot serve (e.g. a term with no
            # trigrams at all) — degrade, never fail the search
            _log_event("error", {"where": "memindex", "error": str(e)})
            return None
        out = []
        for name, mtime, title, body in rows:
            try:
                mtime_f = float(mtime)
            except (TypeError, ValueError):
                mtime_f = 0.0
            out.append((str(name), mtime_f, title or "", body or ""))
        return out
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# --- gc ------------------------------------------------------------------------

# Reclaim the index once dead pages outweigh this share of the file. Measured
# rationale: a 300 KB store produced a 1.4 MB index (76/335 pages free) after
# an optimize run archived notes, because reconcile deletes rows but SQLite
# never shrinks the file on its own.
_COMPACT_FREE_RATIO = 0.25


def _freq_fold(conn: sqlite3.Connection) -> None:
    """Merge FTS5's pending b-tree leaves into the segment index (the
    `merge=4` maintenance op), so the freed pages VACUUM reclaims are the
    whole table's, not just freshly-deleted rows'. Best effort."""
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
    except sqlite3.Error:
        return
    if sqlite3.sqlite_version_info >= (3, 42, 0):
        try:  # fold the segment index's frequencies into their leaves
            conn.execute(
                "INSERT INTO memories_fts(memories_fts, rank) VALUES('merge', 4)")
        except sqlite3.Error:
            pass


def compact(d: str, *, force: bool = False) -> bool:
    """Reclaim this dir's index after the store shrank. Returns True when a
    VACUUM ran. `force=True` (opt-in `nx memories --compact`) skips the
    threshold check; the default only fires when free pages exceed
    _COMPACT_FREE_RATIO of the file, so an ordinary reconcile never pays for
    a full rewrite. Uses a DELETE journal for the duration (VACUUM cannot run
    inside a WAL transaction; the journal mode reverts to WAL after)."""
    if not _fts_supported() or not enabled():
        return False
    # Never CREATE an index just to compact it: a dir that has no index has no
    # dead pages, and _open/_connect would mkdir + create an empty .sqlite.
    if not os.path.exists(_db_path(d)):
        return False
    conn = _open(d)
    if conn is None:
        return False
    try:
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0] or 0)
        free = int(conn.execute("PRAGMA freelist_count").fetchone()[0] or 0)
        if not force:
            if page_count <= 0 or free < 1 or free < _COMPACT_FREE_RATIO * page_count:
                return False
        _freq_fold(conn)
        conn.commit()
        conn.execute("PRAGMA journal_mode=DELETE")
        try:
            conn.execute("VACUUM")
        finally:
            conn.execute("PRAGMA journal_mode=WAL")
        return True
    except sqlite3.Error as e:
        _log_event("error", {"where": "memindex", "error": f"compact: {e}"})
        return False
    finally:
        conn.close()
