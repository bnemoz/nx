"""Checkpoint/undo stack for file-touching tools.

`run_tool()` in `nx/__init__.py` snapshots a file's prior contents just
before `write_file`/`edit_file`/`multi_edit` apply, and pushes the snapshot
here once the tool call actually succeeds (denied/errored calls never touched
the file, so they push nothing). `sed_edit` can touch a whole glob of files
in one call, so it snapshots each matched file itself and pushes them as one
entry (see nx/__init__.py). Either way, one stack entry = one tool call, and
it may cover several files: the `/undo` REPL command pops the most recent
entry and restores every file in it in one step, so undoing a `multi_edit`
or `sed_edit` call reverts all the files it touched together, not one at a
time.
"""
from __future__ import annotations

import os
import time
from typing import Optional, TypedDict

from ._env import _env_int
from . import _state as st

MAX_UNDO = _env_int("NX_MAX_UNDO", 20)


class FileSnapshot(TypedDict):
    path: str
    existed: bool
    content: Optional[str]


class Checkpoint(TypedDict):
    tool: str
    ts: float
    files: list[FileSnapshot]


def _read_prior(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def snapshot(path: Optional[str]) -> Optional[FileSnapshot]:
    """Capture a file's state before a write/edit tool touches it."""
    if not path:
        return None
    existed = os.path.exists(path)
    content = _read_prior(path) if existed else None
    # Unreadable-as-text existing file (binary, permissions) — content is None
    # but existed is True; skip so undo never overwrites it with garbage.
    if existed and content is None:
        return None
    return {"path": path, "existed": existed, "content": content}


def push_checkpoint(tool: str, snaps: list[Optional[FileSnapshot]]) -> None:
    """`snaps` is a list of per-file snapshot dicts (as from `snapshot()`);
    entries with a None snapshot are dropped. Pushes nothing if the list ends
    up empty (e.g. every file was unreadable)."""
    files = [s for s in snaps if s is not None]
    if not files:
        return
    st._UNDO_STACK.append({"tool": tool, "ts": time.time(), "files": files})
    if len(st._UNDO_STACK) > MAX_UNDO:
        del st._UNDO_STACK[0]


def list_checkpoints() -> list[Checkpoint]:
    """Pending checkpoints, most recent first — for `/undo --list`."""
    return list(reversed(st._UNDO_STACK))


def undo_last() -> tuple[Optional[Checkpoint], str]:
    """Revert every file in the most recent checkpointed call. Returns (entry, message)."""
    if not st._UNDO_STACK:
        return None, "nothing to undo"
    entry: Checkpoint = st._UNDO_STACK.pop()
    reverted: list[str] = []
    errors: list[str] = []
    for snap in entry["files"]:
        path = snap["path"]
        try:
            if snap["existed"]:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(snap["content"])
            elif os.path.exists(path):
                os.remove(path)
            reverted.append(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    if errors:
        msg = f"reverted {entry['tool']} on {len(reverted)} file(s); FAILED: " + "; ".join(errors)
    elif len(reverted) == 1:
        msg = f"reverted {entry['tool']} on {reverted[0]}"
    else:
        msg = f"reverted {entry['tool']} on {len(reverted)} file(s): {', '.join(reverted)}"
    return entry, msg
