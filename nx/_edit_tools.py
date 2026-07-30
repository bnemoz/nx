"""File-mutation tool bodies: write_file, edit_file, multi_edit, sed_edit.

Split out of __init__.py as a focused submodule for the tools that write to
disk (as opposed to nx/_tools.py's read-only ones). Each still goes through
_confirm()/risk classification from _approval.py.

`_confirm` is monkeypatched by tests via `m._confirm = fake` (the top-level
`nx` module attribute), so it's called through a lazy `import nx as _m;
_m._confirm(...)` rather than bound directly at import time — see the note
at the top of _repl.py for the same pattern.
"""
from __future__ import annotations

import glob
import os
import re
import tempfile
from typing import Any

from ._ansi import DIM, RESET
from ._approval import _diff_preview, _diff_hunk
from ._tools import _strip_line_numbers
from ._undo import push_checkpoint as _undo_push


def _confirm(action: str, preview: str | None = None) -> bool:
    import nx as _m
    return _m._confirm(action, preview=preview)


def _atomic_write(path: str, content: str) -> None:
    """Write *content* to *path* without ever leaving a truncated file on
    disk: write to a sibling temp file first, then rename over the target.
    `os.replace` is atomic on POSIX, so a crash mid-write leaves either the
    old file or the new one, never a half-written one."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".nx-tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_with_mtime(path: str) -> tuple[str, float | None]:
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        mtime = None
    with open(path) as f:
        return f.read(), mtime


def write_file(path: str, content: str, **_: Any) -> str:
    try:
        before, before_mtime = _read_with_mtime(path)
    except (FileNotFoundError, IsADirectoryError):
        before, before_mtime = "", None
    if not _confirm(f"write {path} ({len(content)} bytes)", preview=_diff_preview(before, content)):
        return "DENIED by user"
    try:
        after_mtime = os.stat(path).st_mtime
    except OSError:
        after_mtime = None
    if after_mtime != before_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
    _atomic_write(path, content)
    return f"wrote {len(content)} bytes to {path}"


def edit_file(path: str, old: str, new: str, **_: Any) -> str:
    src, src_mtime = _read_with_mtime(path)
    if src.count(old) != 1:
        # The model may have pasted read_file's numbered output into `old`.
        # Only as a fallback (exact match already failed), retry with the
        # numbering stripped — and strip `new` the same way so we never write
        # line numbers into the file. Normal edits hit the exact match above
        # and never reach here, so genuine content is never touched.
        stripped_old = _strip_line_numbers(old)
        if stripped_old != old and src.count(stripped_old) == 1:
            old, new = stripped_old, _strip_line_numbers(new)
    if src.count(old) != 1:
        return f"ERROR: `old` matched {src.count(old)} times (need exactly 1)"
    new_src = src.replace(old, new)
    if new_src == src:
        return f"ERROR: no change to {path} (`old` and `new` are identical)"
    if not _confirm(f"edit {path}", preview=_diff_preview(src, new_src)):
        return "DENIED by user"
    try:
        now_mtime = os.stat(path).st_mtime
    except OSError:
        now_mtime = None
    if now_mtime != src_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
    _atomic_write(path, new_src)
    return f"edited {path}\n{_diff_hunk(src, new_src)}"


def multi_edit(path: str, edits: list[dict[str, str]], **_: Any) -> str:
    """Apply multiple exact find-and-replace edits to one file atomically:
    either every edit lands or none do (a failed match rolls back the whole
    batch — the file is only written after all edits match). Each edit is an
    object {"old": <str>, "new": <str>}; `old` must match exactly once in the
    running text (edits apply in list order, so a later edit may target text an
    earlier edit produced). Pasted `read_file` line-number prefixes are stripped
    automatically, like `edit_file`. One approval covers the whole batch — a
    five-line refactor becomes one tool call, not five.
    """
    src, src_mtime = _read_with_mtime(path)
    if not isinstance(edits, list) or not edits:
        return "ERROR: `edits` must be a non-empty list of {old, new} objects"
    work = src
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"ERROR: edit #{i + 1} is not an object"
        old = edit.get("old")
        new = edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return f"ERROR: edit #{i + 1} needs string `old` and `new`"
        # Tolerate pasted read_file line numbers exactly like edit_file.
        if work.count(old) != 1:
            stripped = _strip_line_numbers(old)
            if stripped != old and work.count(stripped) == 1:
                old, new = stripped, _strip_line_numbers(new)
        count = work.count(old)
        if count != 1:
            return (f"ERROR: edit #{i + 1} `old` matched {count} times "
                    f"(need exactly 1); no edits applied")
        work = work.replace(old, new, 1)
    if work == src:
        return f"no change to {path} (edits were no-ops)"
    if not _confirm(f"edit {path} ({len(edits)} edits)", preview=_diff_preview(src, work)):
        return "DENIED by user"
    try:
        now_mtime = os.stat(path).st_mtime
    except OSError:
        now_mtime = None
    if now_mtime != src_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
    _atomic_write(path, work)
    return f"edited {path} ({len(edits)} edits applied)\n{_diff_hunk(src, work)}"


def sed_edit(pattern: str, replacement: str, path: str, count: int = 0, flags: str = "", **_: Any) -> str:
    """Regex find-and-replace (Python `re` syntax) over one file or a glob of
    files (e.g. "src/**/*.py", recursive `**` supported). `flags` is a string
    of single-letter re flags: i=IGNORECASE, m=MULTILINE, s=DOTALL. `count`
    caps replacements per file (0 = unlimited). One approval covers every
    matched file, with a combined diff preview.
    """
    try:
        reflags = 0
        for ch in flags:
            reflags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}.get(ch, 0)
        rx = re.compile(pattern, reflags)
    except re.error as e:
        return f"ERROR: invalid regex: {e}"
    files = sorted(glob.glob(path, recursive=True)) if glob.has_magic(path) else [path]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return f"ERROR: no files matched {path}"
    changes = []
    previews = []
    for f in files:
        with open(f) as fh:
            src = fh.read()
        # replacement is always literal text, never a backreference template:
        # escape backslashes so `\1`, `\U`, a Windows path, etc. don't get
        # interpreted as re.subn's replacement-string syntax.
        literal_replacement = replacement.replace("\\", "\\\\")
        new_src, n = rx.subn(literal_replacement, src, count=count or 0)
        if n == 0:
            continue
        diff = _diff_preview(src, new_src)
        if diff:
            previews.append(f"{DIM}--- {f} ---{RESET}\n{diff}")
        changes.append((f, src, new_src, n))
    if not changes:
        return f"no matches for pattern in {len(files)} file(s)"
    total = sum(n for _, _, _, n in changes)
    action = f"sed_edit {len(changes)} file(s), {total} replacement(s): " + ", ".join(f for f, _, _, _ in changes)
    if not _confirm(action, preview="\n".join(previews)):
        return "DENIED by user"
    _undo_push("sed_edit", [{"path": f, "existed": True, "content": src} for f, src, _new_src, _n in changes])
    for f, _src, new_src, _n in changes:
        _atomic_write(f, new_src)
    hunks = "\n".join(
        f"--- {f} ---\n{_diff_hunk(src, new_src)}" for f, src, new_src, _n in changes)
    return (f"edited {len(changes)} file(s), {total} replacement(s): "
            + ", ".join(f"{f} ({n})" for f, _, _, n in changes)
            + "\n" + hunks)
