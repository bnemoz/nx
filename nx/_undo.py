"""Checkpoint/undo stack for file-touching tools.

`run_tool()` in `nx/__init__.py` snapshots a file's prior contents just
before `write_file`/`edit_file`/`multi_edit` apply, and pushes the snapshot
here once the tool call actually succeeds (denied/errored calls never touched
the file, so they push nothing). The `/undo` REPL command pops the most
recent entry and restores it — this is deliberately per-file, not a batch
transaction: undoing a `multi_edit` reverts the whole file to its pre-edit
state in one step, same as undoing a single `edit_file`.
"""
import os
import time

from ._env import _env_int
from . import _state as st

MAX_UNDO = _env_int("NX_MAX_UNDO", 20)


def _read_prior(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def snapshot(path):
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


def push_checkpoint(tool, snap):
    if snap is None:
        return
    st._UNDO_STACK.append({"tool": tool, "ts": time.time(), **snap})
    if len(st._UNDO_STACK) > MAX_UNDO:
        del st._UNDO_STACK[0]


def list_checkpoints():
    """Pending checkpoints, most recent first — for `/undo --list`."""
    return list(reversed(st._UNDO_STACK))


def undo_last():
    """Revert the most recent checkpointed write/edit. Returns (entry, message)."""
    if not st._UNDO_STACK:
        return None, "nothing to undo"
    entry = st._UNDO_STACK.pop()
    path = entry["path"]
    try:
        if entry["existed"]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(entry["content"])
            msg = f"reverted {entry['tool']} on {path}"
        else:
            if os.path.exists(path):
                os.remove(path)
            msg = f"reverted {entry['tool']} on {path} (file did not exist before, removed)"
        return entry, msg
    except OSError as e:
        return entry, f"ERROR: failed to undo {path}: {e}"
