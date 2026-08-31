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

# Purge threshold for stale overflow files, matching bg-log trimming (_bash_jobs).
OVERFLOW_TTL_DAYS = _env_int("NX_OVERFLOW_TTL_DAYS", 7)

# Cap on how much of the full text we put back in the message: keep it far below
# TOOL_RESULT_CHARS (the message budget) so the path note itself always fits.
MAX_SPILL_ECHO = _env_int("NX_OVERFLOW_ECHO_CHARS", 2000)
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


def spill_to_disk(full: str) -> Optional[str]:
    """Write the complete, untruncated `full` text to the overflow dir and
    return its path, or None if spilling isn't viable (I/O error).

    The filename is the first 16 hex chars of the sha256 of the contents, so
    re-seeing the same output rewrites the same file (content-addressed, no
    duplicate pile-up) and bumps its mtime (refreshing its TTL). Writes go
    through a temp file + os.replace so a concurrent reader never sees a
    half-written spill.
    """
    d = overflow_dir()
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, _short_hash(full) + ".txt")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", errors="replace") as f:
            f.write(full)
        os.replace(tmp, path)
        _purge_old_overflow()
        return path
    except OSError:
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
