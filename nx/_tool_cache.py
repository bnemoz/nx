"""Mtime-aware validity for the read-only tool result cache (st._TOOL_RESULT_CACHE).

Cache entries store os.stat().st_mtime_ns of the backing file (or directory,
for list_dir/search_files/git_status-style calls) alongside the result. A cache
hit re-stats: if the mtime is unchanged the entry is still valid — even after a
run_bash call, which previously dropped the whole cache because it might touch
anything. A trivial `ls` no longer invalidates every cached read_file for the
rest of the session. Entries whose file changed (by any path: bash, edit tools
— which already invalidate by path explicitly — or an external editor, fixing a
staleness bug) are re-read or dropped.

Non-file-backed entries (no usable path arg — e.g. memory_search, or git_* in
a bare repo) store st_mtime_ns None and keep whole-result validity: they were
never meaningful to invalidate per-path, and memory_write / memory_delete
already drop memory_search entries explicitly.
"""
from __future__ import annotations

import os
from typing import Any, Optional

# Value shape for st._TOOL_RESULT_CACHE. (Previously a bare 2-tuple; a
# NamedTuple keeps indexing-by-position working for any code that still does
# entry[0]/entry[1] — result, path — while adding the mtime field.)
try:
    from typing import NamedTuple
    class _CacheEntry(NamedTuple):
        result: str
        path: str
        mtime_ns: Optional[int]
except ImportError:  # pragma: no cover
    _CacheEntry = None


def _mtime_ns_of(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def cache_store_entry(result: str, path_arg: Any) -> "_CacheEntry":
    """Build a cache entry for a completed read-only tool call. path_arg is the
    raw args["path"] value (str, or list -> first element, or absent)."""
    if isinstance(path_arg, list):
        path_arg = path_arg[0] if path_arg else "."
    path = os.path.abspath(path_arg if path_arg else ".")
    return _CacheEntry(result, path, _mtime_ns_of(path))


def cache_valid(entry: "_CacheEntry") -> bool:
    """Re-stat the entry's backing path; a hit whose mtime is unchanged is
    still valid. mtime_ns None (non-file-backed) -> always valid. OSError ->
    invalid (file vanished since it was read)."""
    if entry.mtime_ns is None:
        return True
    try:
        return os.stat(entry.path).st_mtime_ns == entry.mtime_ns
    except OSError:
        return False


def drop_changed_entries(cache: dict) -> None:
    """Drop only entries whose backing file changed since they were stored —
    the mtime-aware replacement for the old whole-cache clear on run_bash."""
    for key, entry in list(cache.items()):
        if not cache_valid(entry):
            del cache[key]
