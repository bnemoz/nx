"""Registry of paths staged into the shared index by nx sessions.

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

Persistence (0.128.x): the registry lives per-repo in
``<repo>/.nx/staged-claims.json``, so a claim SURVIVES a session restart — the
common case of staging work, crashing/restarting, then committing. A claim is
not a blanket trust: each entry stores the index BLOB hash the claimant staged,
and `claimed_staged` subtracts a path only while the shared index still holds
THAT blob. Someone (another session, the user) restaging different content for
the same path changes the blob, the claim no longer matches, and the guard
prompts again — attribution can only ever REDUCE prompts for identical staged
content, never silence foreign work.

In-process claims (`_CLAIMED`) stay authoritative for the current session (no
git round-trip needed); the on-disk entries serve prior sessions. Keys are
`os.path.realpath` of absolute paths; values are index blob hashes.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Iterable

_LOCK = threading.Lock()
# realpath of every path THIS session staged (claimed as its own to commit)
_CLAIMED: set[str] = set()

_GIT_TIMEOUT = 5


def _repo_root(path: str) -> str | None:
    """The git repo containing `path`, or None (not a repo / no git / slow)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.path.dirname(path) or ".", capture_output=True, text=True,
            timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return None
    try:
        return os.path.realpath(out)
    except OSError:
        return None


def _claims_path(repo: str) -> str:
    return os.path.join(repo, ".nx", "staged-claims.json")


def _load(repo: str) -> dict[str, str]:
    try:
        with open(_claims_path(repo), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(repo: str, claims: dict[str, str]) -> None:
    """Merge-write, best-effort: read-modify-write under one lock hold keeps
    two concurrent nx sessions from losing each other's entries; any failure
    just costs a prompt later."""
    try:
        os.makedirs(os.path.dirname(_claims_path(repo)), exist_ok=True)
        merged = _load(repo)
        merged.update(claims)
        tmp = _claims_path(repo) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        os.replace(tmp, _claims_path(repo))
    except OSError:
        pass


def _index_blob(repo: str, path: str) -> str | None:
    """The blob hash the shared index holds for `path` (repo-relative), or
    None when the path is not in the index (deleted-from-index staging shows
    as absent too — unclaimable, which only costs a prompt)."""
    try:
        rel = os.path.relpath(path, repo)
    except ValueError:
        return None
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-s", "--", rel],
            cwd=repo, capture_output=True, text=True, timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or "").split()
    # "<mode> <blob> <stage>\t<path>" — stage > 0 (a merge conflict) means no
    # single authoritative blob; treat as unclaimed.
    if proc.returncode != 0 or len(out) < 3 or out[2] != "0":
        return None
    return out[1] or None


def mark_staged(paths: Iterable[str]) -> None:
    """Record `paths` as staged by this session. Best-effort: a path that
    can't be normalized is skipped, never raised into the caller's commit or
    shell run. Each path's CURRENT index blob is recorded with it, so a later
    restage of different content invalidates the claim."""
    by_repo: dict[str, dict[str, str]] = {}
    for p in paths:
        if not p:
            continue
        try:
            real = os.path.realpath(p)
        except OSError:
            continue
        with _LOCK:
            _CLAIMED.add(real)
        if os.path.isfile(real) or not os.path.exists(real):
            # a staged DELETION leaves no worktree file: the index blob still
            # identifies the entry, but only via the repo root — handled below
            pass
        repo = _repo_root(real)
        if repo is None:
            continue
        blob = _index_blob(repo, real)
        if blob is None:
            continue
        by_repo.setdefault(repo, {})[real] = blob
    for repo, claims in by_repo.items():
        _save(repo, claims)


def claimed_staged(paths: Iterable[str]) -> set[str]:
    """Which of `paths` a claim covers (the SAME strings it was given, so the
    caller can subtract them from its own list). In-session claims match
    unconditionally; on-disk claims from prior sessions match only while the
    shared index still holds the recorded blob."""
    with _LOCK:
        claimed = set(_CLAIMED)
    out: set[str] = set()
    candidates: dict[str, list[tuple[str, str]]] = {}
    for p in paths:
        if not p:
            continue
        try:
            real = os.path.realpath(p)
        except OSError:
            continue
        if real in claimed:
            out.add(p)
            continue
        repo = _repo_root(real)
        if repo is not None:
            candidates.setdefault(repo, []).append((p, real))
    for repo, pairs in candidates.items():
        claims = _load(repo)
        if not claims:
            continue
        for p, real in pairs:
            blob = claims.get(real)
            if blob is not None and blob == _index_blob(repo, real):
                out.add(p)
    return out
