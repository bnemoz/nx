"""Session-scoped registry of clobber targets THIS session was approved for.

The clobber guard (`_clobber.clobber_targets`) forces an approval prompt the
first time a run_bash command truncates an existing, non-empty file — even in
YOLO, because a truncated file is unrecoverable. But without memory of the
approval, the SAME file re-prompts on every later truncation of it in the
session (a model rewriting a probe file five times = five identical prompts).
The user's Y answered the question once; re-asking trains Y-mashing, which is
exactly what the guard's own docstring says a guard must not do.

This module records the approved targets so the guard can subtract them, the
same shape as `_stagedclaim` does for the git-index guard. In-process only,
deliberately: an approval is this session's consent, not a property of the
file. A path approved here keeps prompting for OTHER sessions — and nx's
staleness guard, /undo and the unified diff of the write tools are untouched.

One deliberate asymmetry with the write tools: the approval covers the file,
not the command, and never expires — because the danger (silently destroying
current contents) was already disclosed and accepted for this session. A file
that didn't exist when approved (the command completed and deleted it, or was
denied) drops out naturally: `clobber_targets` only reports existing,
non-empty files.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterable

_LOCK = threading.Lock()
# realpath of every clobber target this session's user already approved
_APPROVED: set[str] = set()


def mark_clobbered(paths: Iterable[str]) -> None:
    """Record `paths` as clobber-approved for this session. Best-effort: a
    path that can't be normalized is skipped, never raised into the caller."""
    keys: list[str] = []
    for p in paths:
        if not p:
            continue
        try:
            keys.append(os.path.realpath(p))
        except OSError:
            continue
    if keys:
        with _LOCK:
            _APPROVED.update(keys)


def approved_clobbers(paths: Iterable[str]) -> set[str]:
    """Which of `paths` this session already approved (the SAME strings it was
    given, so the caller can subtract them from its own list)."""
    with _LOCK:
        approved = set(_APPROVED)
    out: set[str] = set()
    for p in paths:
        if not p:
            continue
        try:
            if os.path.realpath(p) in approved:
                out.add(p)
        except OSError:
            continue
    return out
