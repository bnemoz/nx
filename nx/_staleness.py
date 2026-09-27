"""Cross-session staleness bookkeeping for file writes.

The per-tool mtime guard (`_edit_tools`) only covers the approval window: a
file changed by ANOTHER nx session (or a user's editor) before this session's
model last read it gets silently clobbered — the edit is internally consistent
against content that is no longer on disk. This module records, per session,
the mtime_ns every read_file observed, and the write tools consult it before
applying: if the file's current mtime differs from the last READ mtime this
session recorded, the write is refused with an explicit "changed underneath
you — re-read" error instead of overwriting.

Bounded (PF-16 invariant): one dict entry per read path, capped by
NX_STALENESS_MAX_PATHS (default 512), LRU-evicted oldest first.
"""
from __future__ import annotations

import os
import threading

from ._env import _env_int

_LOCK = threading.Lock()
# path (real, absolute) -> mtime_ns at last read by THIS session
_LAST_READ_MTIME: dict[str, int | None] = {}
_ORDER: list[str] = []


def _max_paths() -> int:
    return max(0, _env_int("NX_STALENESS_MAX_PATHS", 512))


def record_read(path: str) -> None:
    """Note the file's current mtime as the freshness anchor for this session.
    Called by read_file after a successful read (and NOT by cache hits — a
    cache hit re-stats anyway and means "unchanged since the read")."""
    cap = _max_paths()
    if cap <= 0 or not path:
        return
    try:
        rp = os.path.realpath(path)
        mtime = os.stat(rp).st_mtime_ns
    except OSError:
        return
    with _LOCK:
        if rp in _LAST_READ_MTIME:
            _ORDER.remove(rp)
        _LAST_READ_MTIME[rp] = mtime
        _ORDER.append(rp)
        while len(_ORDER) > cap:
            del _LAST_READ_MTIME[_ORDER.pop(0)]


def staleness_error(path: str) -> str | None:
    """A refusal message when the file changed on disk after this session last
    read it; None when no record exists or the file is unchanged (also None
    inside a sub-agent/batch loop, which may legitimately act on parent-side
    reads, and when the fence target was never read as text)."""
    if not path:
        return None
    try:
        rp = os.path.realpath(path)
        mtime = os.stat(rp).st_mtime_ns
    except OSError:
        return None
    with _LOCK:
        seen = _LAST_READ_MTIME.get(rp)
    if seen is None:
        return None
    if mtime == seen:
        return None
    return (f"ERROR: {path} changed on disk since this session last read it "
            f"(mtime {mtime} != {seen}, likely another session or editor) — "
            "re-read the file and retry; refusing to overwrite possibly-newer "
            "content")
