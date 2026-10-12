"""Validity + bounds for the read-only tool result cache (st._TOOL_RESULT_CACHE).

Cache entries store os.stat().st_mtime_ns of the backing file (or directory,
for list_dir/search_files calls) alongside the result. A cache hit re-stats: if
the mtime is unchanged the entry is still valid — even after a run_bash call,
which previously dropped the whole cache because it might touch anything. A
trivial `ls` no longer invalidates every cached read_file for the rest of the
session. Entries whose file changed (by any path: bash, edit tools — which
already invalidate by path explicitly — or an external editor, fixing a
staleness bug) are re-read or dropped.

Anchors (PERF.md PF-18, 0.61.0). A cache key is the *whole* args JSON, so a
cached result can be backed by more than one path: `read_file` accepts a list
and a `search_files` result reflects a whole tree. Both used to anchor on
`path_arg[0]` alone, so editing file #2 of a multi-path read never invalidated
it and a new file in a search_files subdirectory never invalidated the search.
Entries now carry **every** backing path (`paths` + their `mtimes`), and a
`search_files` entry additionally carries a `tree_stamp`: a bounded walk of the
subtree recording each directory's and file's mtime, so adding/removing/editing
anything under the searched root invalidates it.

Bounded (PF-16, invariant I-8: every cache has an explicit bound and an
eviction path). The dict is an LRU: a hit touches the key, a store evicts from
the front until both `NX_TOOL_CACHE_MAX_ENTRIES` and `NX_TOOL_CACHE_MAX_BYTES`
hold, so a long session's RSS stays flat instead of growing with every read.

VCS-backed entries (git_status/git_diff/git_log) need a second signal: the
backing path is usually the repo root directory, and nothing git does — commit,
checkout, reset — mutates that directory's own mtime (git writes under .git/).
So their entries additionally carry a git_state_stamp fingerprint (mtimes of
.git/HEAD, index, ORIG_HEAD, the resolved branch ref, packed-refs and the HEAD
reflog) and are valid only while it matches, plus
they are dropped outright by any write_file/edit_file/multi_edit/delete_file
and by any successful run_bash (git state can change without a stamp change —
e.g. worktree edits alter status, and `rm -rf .git` makes the stamp None).

Non-file-backed entries (no usable path arg — e.g. memory_search, or git_* in
a bare repo) anchor at the cwd and keep whole-result validity: they were
never meaningful to invalidate per-path, and memory_write / memory_delete
already drop memory_search entries explicitly.
"""
from __future__ import annotations

import os
from typing import Any

from ._env import _env_int

# Tools whose results reflect git state (see module docstring).
_VCS_TOOLS = frozenset({"git_status", "git_diff", "git_log"})

# Tools whose result reflects a whole subtree rather than one path.
_SEARCH_TREE_TOOLS = frozenset({"search_files"})

# Directories a tree_stamp walk skips: build/VCS noise that search_files
# itself never reports (mirrors _tools.search_files' NX_SEARCH_EXCLUDE_DIRS
# default, kept here as a frozen set so the stamp stays cheap).
_TREE_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".eggs",
})

# Value shape for st._TOOL_RESULT_CACHE. (Originally a bare 2-tuple; a
# NamedTuple keeps indexing-by-position working for any code that still does
# entry[0]/entry[1] — result, path — while adding the anchor/bound fields.)
try:
    from typing import NamedTuple
    class _CacheEntry(NamedTuple):
        result: str
        path: str
        mtime_ns: int | None
        git_stamp: str | None = None  # set only for VCS-backed entries
        paths: tuple = ()             # every backing path (absolute), PF-18
        mtimes: tuple = ()            # their mtimes at store time
        tree_stamp: str | None = None  # set only for search_files entries
        hits: int = 0                  # times this entry served a cache hit
except ImportError:  # pragma: no cover
    _CacheEntry = None  # ty: ignore[invalid-assignment]  # pyrefly: ignore[invalid-assignment, not-callable, missing-attribute]  # py3.6- fallback


def _mtime_ns_of(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _abs_paths(path_arg: Any) -> tuple[str, ...]:
    """Normalize an args['path'] value (str, list, or absent) to the set of
    absolute backing paths a cache entry is anchored on. A list keeps every
    element (PF-18: editing file #2 must invalidate the entry); absent/empty
    falls back to the cwd, matching the pre-0.61 anchor."""
    if isinstance(path_arg, (list, tuple)):
        items = [p for p in path_arg if p]
    elif path_arg:
        items = [path_arg]
    else:
        items = ["."]
    out: list[str] = []
    for p in items:
        try:
            out.append(os.path.abspath(p))
        except (TypeError, ValueError):
            continue
    return tuple(out) or (os.path.abspath("."),)


def _git_dir_stamp(git_dir: str) -> str:
    """mtimes of the state files git writes, joined. A plain commit never
    rewrites HEAD (a symref, only rewritten on branch switches) nor the index
    (an empty commit rewrites nothing there), so the stamp also covers the
    files that DO move: the resolved branch ref (loose file under the shared
    commondir, or packed-refs) and the HEAD reflog (commit/checkout/reset all
    append to logs/HEAD when reflogs are on, the default outside bare repos)."""
    parts = []
    watched = ["HEAD", "index", "ORIG_HEAD", "logs/HEAD", "packed-refs"]
    try:
        with open(os.path.join(git_dir, "HEAD")) as fh:
            line = fh.readline().strip()
        if line.startswith("ref:"):
            ref = line[len("ref:"):].strip()
            # A linked worktree's HEAD symrefs a branch whose loose ref lives in
            # the shared commondir (the main .git), not the per-worktree dir.
            shared = git_dir
            try:
                with open(os.path.join(git_dir, "commondir")) as cfh:
                    shared = os.path.normpath(os.path.join(git_dir, cfh.readline().strip()))
            except OSError:
                pass
            watched.append(os.path.relpath(os.path.join(shared, ref), git_dir))
    except OSError:
        pass  # unreadable HEAD: the base files still get fingerprinted
    for rel in watched:
        try:
            parts.append(f"{rel}:{os.stat(os.path.join(git_dir, rel)).st_mtime_ns}")
        except OSError:
            parts.append(f"{rel}:-")
    return "|".join(parts)


def git_state_stamp(path: str) -> str | None:
    """A cheap fingerprint of git state under *path*: mtime_ns of .git/HEAD,
    index, ORIG_HEAD, the resolved branch ref, packed-refs and the HEAD reflog
    (whatever exists), joined. Returns None when no .git
    is reachable, in which case the entry keeps whole-result validity."""
    real = os.path.realpath(path)
    for _ in range(6):  # walk up like git does
        git = os.path.join(real, ".git")
        if os.path.exists(git):
            if os.path.isfile(git):  # linked worktree: .git points at the real gitdir
                try:
                    with open(git) as fh:
                        line = fh.readline().strip()
                    if line.startswith("gitdir:"):
                        target = line[len("gitdir:"):].strip()
                        git = target if os.path.isabs(target) else os.path.join(real, target)
                except OSError:
                    pass
            return _git_dir_stamp(git)
        parent = os.path.dirname(real)
        if parent == real:
            break
        real = parent
    return None


def tree_stamp(root: str) -> str | None:
    """A bounded fingerprint of every path under *root*: each directory's
    mtime + entry count and each file's mtime, walked in sorted order.

    This is what makes a cached `search_files` honest (PF-18): the searched
    root's *own* mtime does not move when a file in a subdirectory changes or
    a new one appears, so mtime-on-the-root "validity" served stale hits
    forever. The walk stops after NX_TOOL_CACHE_TREE_FILES entries (default
    5000) — past that the stamp is a bounded lower bound and a change beyond
    the cut can be missed, which is the documented trade for keeping a cache
    hit cheap. Returns None if the root is unreadable (treated as invalid)."""
    cap = _env_int("NX_TOOL_CACHE_TREE_FILES", 5000)
    if cap <= 0:
        return None
    parts: list[str] = []
    seen = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in _TREE_SKIP_DIRS)
            parts.append(f"{dirpath}:{_mtime_ns_of(dirpath)}:{len(filenames)}")
            seen += 1
            for fn in sorted(filenames):
                parts.append(f"{fn}:{_mtime_ns_of(os.path.join(dirpath, fn))}")
                seen += 1
                if seen >= cap:
                    return "|".join(parts)
    except OSError:
        return None
    return "|".join(parts)


def cache_store_entry(
    result: str,
    path_arg: Any,
    git_stamp: str | None = None,
) -> _CacheEntry:
    """Build a cache entry for a completed read-only tool call. path_arg is the
    raw args["path"] value (str, list -> every element, or absent)."""
    paths = _abs_paths(path_arg)  # always non-empty: _abs_paths falls back to "."
    mtimes = tuple(_mtime_ns_of(p) for p in paths)
    # pyrefly: ignore [not-callable]
    return _CacheEntry(result, paths[0], mtimes[0], git_stamp, paths, mtimes, None)


def cache_store_tree_entry(
    result: str,
    root: Any,
    git_stamp: str | None = None,
) -> _CacheEntry:
    """Build a cache entry for a subtree-wide read (search_files): one anchor
    path plus the bounded tree_stamp that sees changes anywhere below it."""
    root_paths = _abs_paths(root)
    root_abs = root_paths[0]
    m = _mtime_ns_of(root_abs)
    # pyrefly: ignore [not-callable]
    return _CacheEntry(result, root_abs, m, git_stamp, (root_abs,), (m,),
                       tree_stamp(root_abs))


def cache_valid(entry: _CacheEntry) -> bool:
    """Re-stat every backing path; a hit whose mtimes are unchanged is still
    valid. A path stored as absent (mtime None) stays valid only while it is
    still absent — its appearance is a change. A tree-stamped entry re-walks
    the subtree. OSError/unreadable -> invalid."""
    # pyrefly: ignore [missing-attribute]
    paths = getattr(entry, "paths", None) or (entry.path,)
    mtimes = getattr(entry, "mtimes", None)
    if not mtimes or len(mtimes) != len(paths):
        # Entries built positionally by older callers/tests carry only the
        # single anchor; degrade to the pre-0.61 check.
        # pyrefly: ignore [missing-attribute]
        mtimes = (entry.mtime_ns,) * len(paths)
    for p, m in zip(paths, mtimes):
        if _mtime_ns_of(p) != m:
            return False
    stamp = getattr(entry, "tree_stamp", None)
    if stamp is not None:
        return tree_stamp(entry.path) == stamp  # pyrefly: ignore[missing-attribute]  # py3.6- fallback makes _CacheEntry None
    if entry.git_stamp is not None:  # pyrefly: ignore[missing-attribute]  # py3.6- fallback makes _CacheEntry None
        # pyrefly: ignore [missing-attribute]
        return git_state_stamp(entry.path) == entry.git_stamp
    return True


def store_cacheable_result(
    cache: dict,
    cache_key: Any,
    name: str,
    args: dict,
    result: str,
) -> None:
    """Build and store the right entry shape for a cacheable read. run_tool()
    is the single writer — the mixed-turn prefetch (_turn._prefetch_cacheable_tools)
    dispatches through run_tool() too, so every entry (stamped git_* included)
    comes from here and both paths produce identical entries by construction.
    The cache is then re-bounded (PF-16): oldest entry first until both the
    entry and byte caps hold."""
    cache_path = args.get("path", ".")
    if isinstance(cache_path, list):
        first = cache_path[0] if cache_path else "."
        stamp = git_state_stamp(first) if name in _VCS_TOOLS else None
    else:
        stamp = git_state_stamp(cache_path) if name in _VCS_TOOLS else None
    if name in _SEARCH_TREE_TOOLS:
        cache[cache_key] = cache_store_tree_entry(result, cache_path, stamp)
    else:
        cache[cache_key] = cache_store_entry(result, cache_path, stamp)
    bound_cache(cache)


# --- bounds (PF-16, invariant I-8) ---------------------------------------------

def _cache_caps() -> tuple[int, int]:
    """(max entries, max result bytes) read at call time so a test can set the
    knobs per case. 0 disables that half of the bound."""
    entries = _env_int("NX_TOOL_CACHE_MAX_ENTRIES", 512)
    mb = _env_int("NX_TOOL_CACHE_MAX_BYTES", 16 * 1024 * 1024)
    return max(0, entries), max(0, mb)


def cache_touch(cache: dict, key: Any) -> None:
    """Mark *key* as most-recently-used so eviction is LRU, not FIFO.

    A plain dict has no move_to_end, so recency is expressed by re-inserting
    the key: dicts preserve insertion order, which makes the front of the
    iteration order the oldest entry — exactly what bound_cache evicts."""
    if key in cache:
        e = cache[key]
        # Count the hit before re-inserting (0.66.1): a rapid repeat of the
        # same call is how a stuck model burns context — each hit re-bills the
        # full cached result as a fresh tool message. _dispatch turns the
        # second hit into a one-line stub pointing at the first result.
        try:
            cache[key] = e._replace(hits=e.hits + 1)
        except Exception:
            pass
        try:
            # pyrefly: ignore [missing-attribute]
            cache.move_to_end(key)
        except AttributeError:
            cache[key] = cache.pop(key)  # pyrefly: ignore[missing-attribute, unsupported-operation]  # plain dict: re-insert for recency


def _entry_cost(e: _CacheEntry) -> int:
    """RAM cost of an entry: the result plus its tree_stamp (a search entry's
    stamp holds up to NX_TOOL_CACHE_TREE_FILES "name:mtime" tokens — hundreds
    of KB per entry, unmetered before 0.108.3, so a search-heavy swarm could
    blow the byte cap several times over)."""
    # pyrefly: ignore [missing-attribute]
    return len(e.result or "") + len(getattr(e, "tree_stamp", None) or "")


def bound_cache(cache: dict) -> None:
    """Evict oldest entries until both caps hold. Always leaves at least one
    entry, so a single oversize result is still cacheable (and re-readable)."""
    max_entries, max_bytes = _cache_caps()
    if max_entries and len(cache) > max_entries:
        for key in list(cache)[: len(cache) - max_entries]:
            del cache[key]
    if max_bytes:
        total = sum(_entry_cost(e) for e in cache.values())
        for key in list(cache):
            if total <= max_bytes or len(cache) <= 1:
                break
            entry = cache.pop(key)
            total -= _entry_cost(entry)


def drop_changed_entries(cache: dict, full_tree_walk: bool | None = None) -> None:
    """Drop only entries whose backing paths changed since they were stored —
    the mtime-aware replacement for the old whole-cache clear on run_bash.

    Per-path anchors are re-statted (one os.stat each — cheap). Tree-stamped
    entries (search_files) used to re-walk up to NX_TOOL_CACHE_TREE_FILES
    entries *per entry* here, which after every edit/run_bash cost tens of
    thousands of stats across a populated cache (swarm audit, P1). The sweep
    is only eager cleanup — the honest check is cache_valid's tree walk on
    *hit*, which stays in place. So the sweep now uses a fast path: a
    tree-stamped entry is dropped iff its root directory's mtime moved
    (create/delete/rename under the root moves it; a pure content edit does
    not, and is caught on the next hit's full walk). Set
    NX_TOOL_CACHE_SWEEP_TREE_WALK=1 to restore the legacy eager walk.

    Deletion is pop(key, None): the mixed-turn prefetch pool inserts
    concurrently (run_tool), so a check-then-del raced a KeyError into the
    dispatcher (swarm audit, P1)."""
    if full_tree_walk is None:
        full_tree_walk = bool(_env_int("NX_TOOL_CACHE_SWEEP_TREE_WALK", 0))
    for key, entry in list(cache.items()):
        if getattr(entry, "tree_stamp", None) is not None and not full_tree_walk:
            # Fast path: one stat on the searched root decides eager eviction.
            if _mtime_ns_of(entry.path) != entry.mtime_ns:
                cache.pop(key, None)
            continue
        if not cache_valid(entry):
            cache.pop(key, None)


def drop_vcs_entries(cache: dict) -> None:
    """Drop every git_status/git_diff/git_log entry unconditionally — used by
    run_bash and the edit tools. Git state can change in ways no stamp sees:
    working-tree edits alter `git status`/`git diff` without touching .git,
    and `rm -rf .git` flips the stamp to None (where mtime-only "validity"
    would otherwise happily serve a repo's results forever)."""
    for key in [k for k in cache if k[0] in _VCS_TOOLS]:
        del cache[key]
