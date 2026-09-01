"""Mtime-aware validity for the read-only tool result cache (st._TOOL_RESULT_CACHE).

Cache entries store os.stat().st_mtime_ns of the backing file (or directory,
for list_dir/search_files calls) alongside the result. A cache hit re-stats: if
the mtime is unchanged the entry is still valid — even after a run_bash call,
which previously dropped the whole cache because it might touch anything. A
trivial `ls` no longer invalidates every cached read_file for the rest of the
session. Entries whose file changed (by any path: bash, edit tools — which
already invalidate by path explicitly — or an external editor, fixing a
staleness bug) are re-read or dropped.

VCS-backed entries (git_status/git_diff/git_log) need a second signal: the
backing path is usually the repo root directory, and nothing git does — commit,
checkout, reset — mutates that directory's own mtime (git writes under .git/).
So their entries additionally carry a git_state_stamp fingerprint (mtimes of
.git/HEAD, .git/index, ORIG_HEAD) and are valid only while it matches, plus
they are dropped outright by any write_file/edit_file/multi_edit/delete_file
and by any successful run_bash (git state can change without a stamp change —
e.g. worktree edits alter status, and `rm -rf .git` makes the stamp None).

Non-file-backed entries (no usable path arg — e.g. memory_search, or git_* in
a bare repo) store st_mtime_ns None and keep whole-result validity: they were
never meaningful to invalidate per-path, and memory_write / memory_delete
already drop memory_search entries explicitly.
"""
from __future__ import annotations

import os
from typing import Any

# Tools whose results reflect git state (see module docstring).
_VCS_TOOLS = frozenset({"git_status", "git_diff", "git_log"})

# Value shape for st._TOOL_RESULT_CACHE. (Previously a bare 2-tuple; a
# NamedTuple keeps indexing-by-position working for any code that still does
# entry[0]/entry[1] — result, path — while adding the mtime field.)
try:
    from typing import NamedTuple
    class _CacheEntry(NamedTuple):
        result: str
        path: str
        mtime_ns: int | None
        git_stamp: str | None = None  # set only for VCS-backed entries
except ImportError:  # pragma: no cover
    _CacheEntry = None  # ty: ignore[invalid-assignment]  # py3.6- fallback


def _mtime_ns_of(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _git_dir_stamp(git_dir: str) -> str:
    """mtimes of HEAD/index/ORIG_HEAD inside one git dir, joined."""
    parts = []
    for rel in ("HEAD", "index", "ORIG_HEAD"):
        try:
            parts.append(f"{rel}:{os.stat(os.path.join(git_dir, rel)).st_mtime_ns}")
        except OSError:
            parts.append(f"{rel}:-")
    return "|".join(parts)


def git_state_stamp(path: str) -> str | None:
    """A cheap fingerprint of git state under *path*: mtime_ns of .git/HEAD,
    index and ORIG_HEAD (whatever exists), joined. Returns None when no .git
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


def cache_store_entry(
    result: str,
    path_arg: Any,
    git_stamp: str | None = None,
) -> _CacheEntry:
    """Build a cache entry for a completed read-only tool call. path_arg is the
    raw args["path"] value (str, or list -> first element, or absent)."""
    if isinstance(path_arg, list):
        path_arg = path_arg[0] if path_arg else "."
    path = os.path.abspath(path_arg if path_arg else ".")
    return _CacheEntry(result, path, _mtime_ns_of(path), git_stamp)


def cache_valid(entry: _CacheEntry) -> bool:
    """Re-stat the entry's backing path; a hit whose mtime is unchanged is
    still valid. mtime_ns None (non-file-backed) -> always valid — unless the
    entry carries a git stamp, which is checked against live state. OSError ->
    invalid (file vanished since it was read)."""
    if entry.mtime_ns is not None:
        try:
            if os.stat(entry.path).st_mtime_ns != entry.mtime_ns:
                return False
        except OSError:
            return False
    if entry.git_stamp is not None:
        return git_state_stamp(entry.path) == entry.git_stamp
    return True


def store_cacheable_result(
    cache: dict,
    cache_key: Any,
    name: str,
    args: dict,
    result: str,
) -> None:
    """Build and store the right entry shape for a cacheable read. Shared by
    run_tool()'s post-call store and _turn._prefetch_cacheable_tools so both
    paths always produce identical entries (notably: git_* get a stamp)."""
    cache_path = args.get("path", ".")
    if isinstance(cache_path, list):
        cache_path = cache_path[0] if cache_path else "."
    stamp = git_state_stamp(cache_path) if name in _VCS_TOOLS else None
    cache[cache_key] = cache_store_entry(result, cache_path, stamp)


def drop_changed_entries(cache: dict) -> None:
    """Drop only entries whose backing file changed since they were stored —
    the mtime-aware replacement for the old whole-cache clear on run_bash."""
    for key, entry in list(cache.items()):
        if not cache_valid(entry):
            del cache[key]


def drop_vcs_entries(cache: dict) -> None:
    """Drop every git_status/git_diff/git_log entry unconditionally — used by
    run_bash and the edit tools. Git state can change in ways no stamp sees:
    working-tree edits alter `git status`/`git diff` without touching .git,
    and `rm -rf .git` flips the stamp to None (where mtime-only "validity"
    would otherwise happily serve a repo's results forever)."""
    for key in [k for k in cache if k[0] in _VCS_TOOLS]:
        del cache[key]
