"""The `archive/` side of the memory store: listing and restoring retired
notes.

Split out of `_memory_cmd.py` (which keeps the `/memory` slash-command
dispatch). Archived notes are exactly what `/memory optimize` moved into
`<dir>/archive/<run>/`, so this is the read side of the reversible-by-`mv`
promise the archive makes; restoring goes through the same one writer
(`_write_memory_file`) and drops the archived original.

Notes sitting FLAT at the archive root (no run dir — a writer bug, not the
layout the store produces) are surfaced too, under the `_UNBATCHED_RUN`
pseudo-run, so they stay listable and restorable until retention ages them
out; see `_memory_optimize.prune_archive`, which bounds them."""
from __future__ import annotations

import os

from ._ansi import C, DIM, RESET
from ._memory_store import (
    _MEMORY_FOOTER_RE, _confirm_memory, _iter_memory_dirs, _note_derived,
    _read_memory_title, _write_memory_file,
)


# The pseudo-run label for notes sitting at the archive ROOT with no run dir
# (see _iter_archive_notes): they are real, restorable notes, just unbatched —
# the label says so instead of inventing a stamp they never had.
_UNBATCHED_RUN = "unbatched"


def _iter_archive_notes(d: str) -> list[tuple[str, str, float, str, str]]:
    """Every archived note under `<d>/archive/` as the same shape the live
    scanner yields, (run, name, mtime, title, path). Archived notes are exactly
    what `/memory optimize` moved to `<d>/archive/<run>/`, so this is the read
    side of the reversible-by-`mv` promise the archive makes. Newest run first.

    Flat files at the archive ROOT (no run subdir — a writer bug) are reported
    too, under the `_UNBATCHED_RUN` pseudo-run and after every run dir, so they
    stay visible to `/memory archived` and can still be restored before
    retention removes them. Only `.md` files are ever reported, never a dir and
    never a non-`.md` stray."""
    out: list[tuple[str, str, float, str, str]] = []
    archive = os.path.join(d, "archive")
    try:
        names = sorted(os.listdir(archive), reverse=True)
    except OSError:
        return out
    for run in names:
        run_dir = os.path.join(archive, run)
        if not os.path.isdir(run_dir):
            continue
        try:
            inner = sorted(os.listdir(run_dir), reverse=True)
        except OSError:
            continue
        for name in inner:
            if not name.endswith(".md"):
                continue
            path = os.path.join(run_dir, name)
            head = _read_memory_title(path, name)
            if head is None:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            out.append((run, name, mtime, head[0], path))
    for name in sorted(names, reverse=True):
        if not name.endswith(".md"):
            continue
        path = os.path.join(archive, name)
        if not os.path.isfile(path):
            continue
        head = _read_memory_title(path, name)
        if head is None:
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        out.append((_UNBATCHED_RUN, name, mtime, head[0], path))
    return out


def _cmd_memory_archived(target: str) -> None:
    """`/memory archived [target]` — list archived (retired) notes, or restore
    one by title / filename / unique prefix. Restoring moves the file back into
    its scope dir with a fresh name, so it re-enters the store (and the next
    search re-indexes it)."""
    entries = []
    for d, label in _iter_memory_dirs("project"):
        for run, name, mtime, title, path in _iter_archive_notes(d):
            entries.append((run, name, mtime, title, path, label, d))

    if not target:
        if not entries:
            print(f"{DIM}  no archived memories{RESET}")
            return
        print(f"{DIM}  archived memories ({len(entries)}):{RESET}")
        for run, name, _mtime, title, _path, label, _d in sorted(
                entries, key=lambda x: x[2], reverse=True):
            print(f"  {C.SUCCESS}·{RESET} [{label}] {title}  "
                  f"{DIM}({run} · {name}){RESET}")
        print(f"{DIM}  restore one with: /memory archived <title|filename|prefix>{RESET}")
        return

    low = target.lower()
    exact = [e for e in entries
             if e[1].lower() == low or e[1][:-3].lower() == low or e[3].lower() == low]
    if len(exact) == 1:
        chosen = exact[0]
    else:
        cands = exact if exact else [e for e in entries if e[3].lower().startswith(low)]
        if len(cands) > 1:
            listing = "\n".join(f"  - {e[3]} ({e[1]})" for e in cands[:10])
            print(f"{C.WARNING}  {target!r} matches {len(cands)} archived notes — "
                  f"use an exact title or filename:{RESET}\n{listing}")
            return
        if not cands:
            print(f"{DIM}  no archived memory matching {target!r}{RESET}")
            return
        chosen = cands[0]

    _run, name, _mtime, title, path, label, scope_dir = chosen
    if not _confirm_memory(f"restore memory {path} → {scope_dir}"):
        return
    # Publish with a fresh timestamped name via the one writer, then drop the
    # archived original — the restore goes through the same 0600/no-clobber path
    # as any other write, so a restored note is byte-conventional.
    try:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", "replace")
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't read {path}: {e}{RESET}")
        return
    # Strip the archived note's own footer — the writer appends a fresh one.
    # (The regex anchors on `_scope:` and DOTALLs to EOF, so a `_derived:` field
    # riding that line is stripped with it: the restore can never produce a
    # doubled footer.) Provenance is CARRIED ACROSS, not dropped: a derived note
    # comes back derived, so the pair stays identifiable after an archive.
    derived = _note_derived(content) or None
    body = _MEMORY_FOOTER_RE.sub("", content).rstrip()
    written = _write_memory_file(title, body, label, gate=False, derived=derived)
    if written is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass  # the restored copy exists; a stale archive file is harmless
    print(f"  {C.SUCCESS}restored{RESET} {title}  {DIM}→ {written[0]}{RESET}")
