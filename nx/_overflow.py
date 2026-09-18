"""Tool-result spillover: keep oversized tool output on disk, not lost.

When a tool result exceeds NX_TOOL_RESULT_CHARS, _sanitize_tool_result head/tail-
caps it — grep/find output the model can't know it's missing. The full text is
also written to ``SESSION_HOME/overflow/<sha256>.txt`` and the returned result
gains head + tail + the path, so the model can read_file/search_files that path
for the middle. Files purge after NX_OVERFLOW_TTL_DAYS like _delete_old_bg_logs.
"""
import hashlib
import os
import time
from typing import Optional

from . import _state as st
from ._env import _env_int
from ._private import open_private, private_dir

# Purge threshold for stale overflow files, matching bg-log trimming (_bash_jobs).
OVERFLOW_TTL_DAYS = _env_int("NX_OVERFLOW_TTL_DAYS", 7)

# Cap on how much of the full text we put back in the message: keep it far below
# TOOL_RESULT_CHARS (the message budget) so the path note itself always fits.
MAX_SPILL_ECHO = _env_int("NX_OVERFLOW_ECHO_CHARS", 2000)

# Marker prefixed to read_file results that came from an overflow spill
# file. _sanitize_tool_result uses it to serve the read verbatim (hard cap
# only) instead of re-spilling the read — a stub of a stub points at a fresh
# file and the model chases pointers forever.
OVERFLOW_READ_MARKER = "[overflow spill read: served verbatim]"

_MAX_SPILL_ECHO_LINES = 5


def overflow_dir() -> str:
    """Directory holding spilled tool output (st.SESSION_HOME/overflow)."""
    return os.path.join(st.SESSION_HOME, "overflow")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _path_for(name: str) -> str:
    return os.path.join(overflow_dir(), name)


def _purge_old_overflow(max_age_days: int = OVERFLOW_TTL_DAYS) -> None:
    """Remove overflow files older than max_age_days (best-effort, never raises).

    Mirrors the age-pruning half of _delete_old_bg_logs; no count cap is needed
    since filenames are content hashes, so repeated identical output dedups.
    """
    d = overflow_dir()
    try:
        now = time.time()
        for name in os.listdir(d):
            p = _path_for(name)
            try:
                if (now - os.path.getmtime(p)) > max_age_days * 86400:
                    os.remove(p)
            except OSError:
                continue
    except OSError:
        pass


def is_overflow_path(path: str) -> bool:
    """True if `path` is inside the overflow spill dir (absolute or not, with
    or without the .txt suffix — read_file callers pass whatever the stub
    note showed them)."""
    if not path:
        return False
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    return os.path.dirname(real) == os.path.realpath(overflow_dir())


def spill_to_disk(full: str) -> Optional[str]:
    """Write the complete, untruncated `full` text to the overflow dir and
    return its path, or None if spilling isn't viable (I/O error).

    The filename is the first 16 hex chars of the sha256 of the contents, so
    re-seeing the same output rewrites the same file (content-addressed, no
    duplicate pile-up) and bumps its mtime (refreshing its TTL). Writes go
    through a temp file + os.replace so a concurrent reader never sees a
    half-written spill, and the file is created 0600 regardless of umask —
    a spill is the one place nx persists MORE than the model was sent.
    """
    d = overflow_dir()
    tmp = None
    try:
        private_dir(d)
        path = os.path.join(d, _short_hash(full) + ".txt")
        tmp = f"{path}.{os.getpid()}.tmp"
        # This file holds the COMPLETE untruncated tool result — strictly more
        # than the model itself was sent. Creating it 0600 (not via plain
        # open(), which honours a group-writable umask) is what keeps a grep
        # dump over a credential out of other users' hands; write_private is
        # also atomic, so a concurrent read_file on the spill never sees a
        # half-written file.
        with open_private(tmp) as f:
            f.write(full)
        os.replace(tmp, path)
        _purge_old_overflow()
        return path
    except OSError:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return None


def _echo_lines(full: str) -> tuple:
    """Pick head/tail *line* samples of `full` bounded by MAX_SPILL_ECHO chars.

    Returns (head_block, tail_block, elided_line_count). Takes up to
    _MAX_SPILL_ECHO_LINES lines from each end, then shrinks toward the edges if
    the char budget is exceeded. A single giant line degrades gracefully: it is
    itself head/tail-truncated to the char budget rather than omitted.
    """
    lines = full.split("\n")
    budget = max(MAX_SPILL_ECHO, 0)
    if budget == 0:
        return "", "", max(len(lines) - 2, 0)

    head_n = min(_MAX_SPILL_ECHO_LINES, len(lines))
    tail_n = min(_MAX_SPILL_ECHO_LINES, max(len(lines) - head_n, 0))
    head_lines = lines[:head_n]
    tail_lines = lines[len(lines) - tail_n:] if tail_n else []

    def _cost():
        return sum(len(x) + 1 for x in head_lines + tail_lines)

    while head_lines and _cost() > budget:
        if len(head_lines) > 1:
            head_lines.pop()
        elif tail_lines:
            tail_lines.pop(0)
        else:
            break

    def _fmt(block, other_len):
        text = "\n".join(block)
        if len(text) + other_len > budget and text:
            cut = max(budget - other_len, 0)
            text = text[:cut] + ("..." if cut < len(text) else "")
        return text

    head = _fmt(head_lines, len("\n".join(tail_lines)) + 1 if tail_lines else 0)
    tail = _fmt(tail_lines, len(head) + 1)
    return head, tail, max(len(lines) - len(head_lines) - len(tail_lines), 0)


def cap_overflow_read(text: str) -> str:
    """Bound a verbatim spill-file read without re-spilling it: hard head/tail
    cap (no pointer — the file IS the pointer), then strip the marker so the
    model never sees the plumbing."""
    from ._text_fallback import TOOL_RESULT_CHARS  # local: avoids import cycle
    budget = TOOL_RESULT_CHARS
    if budget > 0 and len(text) > budget:
        head = budget * 2 // 3
        tail = budget - head
        elided = len(text) - head - tail
        text = (text[:head]
                + f"\n... [{elided} chars elided to bound context; "
                + "re-run with offset/limit if you need the rest]\n"
                + text[-tail:])
    return text.replace(OVERFLOW_READ_MARKER, "")


def build_overflow_result(full: str, path: str) -> str:
    """Compose the message-body replacement: real head + real tail + the path.

    Overflow replaces the old content-free elision marker with *content*: the
    first few lines, the last few lines, and where the rest lives — so a grep
    dump over the budget still shows its best matches up front instead of a
    bare "[N chars elided]" and a shrug.
    """
    head, tail, elided_lines = _echo_lines(full)
    parts = []
    if head:
        parts.append(head)
    if tail:
        parts.append(tail)
    note = (f"... [{elided_lines} lines elided; full output ({len(full)} chars)"
            f" saved to: {path} — read_file that path for the middle]")
    return "\n".join(parts + [note])
