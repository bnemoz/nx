"""Session-scoped registry of paths THIS session staged into the shared index.

The git-index guard (`_clobber.git_index_guard`) asks before any raw git op
while the SHARED index holds staged paths, because git records no owner in the
index and a raw `git add`/`git commit` sweeps whatever is staged into its own
result. But a session legitimately leaves its OWN work staged:

  * `git_commit` commits through a private GIT_INDEX_FILE, then
    `_gitcommit._sync_shared_index` re-caches the committed paths into the
    shared index — where they stay STAGED whenever the worktree still differs
    from HEAD (live WIP);
  * a raw `git add foo` earlier in the same run_bash session.

Without attribution every LATER git op of that same session prompts on its own
staged paths. This module records what this session staged, so the guard can
subtract it and ask only about paths it never touched.

In-process only, deliberately (same shape as `_staleness`: a module-level set
behind a lock, no persistence, no cross-session signalling). A claimant we
cannot see is a claimant we must still ask about: staged work from ANOTHER
session (or the user's own terminal) has no entry here, so it keeps prompting.
The registry can therefore only ever REDUCE prompts for this session's own
paths — never widen, never silence a foreign staged set.

Keys are `os.path.realpath` of absolute paths: pass staged paths resolved
against the repo root (see `_clobber.staged_abs_paths`) so a repo-relative path
from `git status` and the same path named absolutely compare equal.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterable

_LOCK = threading.Lock()
# realpath of every path THIS session staged (claimed as its own to commit)
_CLAIMED: set[str] = set()


def mark_staged(paths: Iterable[str]) -> None:
    """Record `paths` as staged by this session. Best-effort: a path that
    can't be normalized is skipped, never raised into the caller's commit or
    shell run."""
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
            _CLAIMED.update(keys)


def claimed_staged(paths: Iterable[str]) -> set[str]:
    """Which of `paths` this session marked (the SAME strings it was given, so
    the caller can subtract them from its own list)."""
    with _LOCK:
        claimed = set(_CLAIMED)
    out: set[str] = set()
    for p in paths:
        if not p:
            continue
        try:
            if os.path.realpath(p) in claimed:
                out.add(p)
        except OSError:
            continue
    return out
