"""The cross-process content-addressed tool-result store (swarm P10, 0.98.1).

One CAS layer under ``<NX_HOME>/obsstore/`` that every large-tool-result
sink can point at, so N concurrent agents sharing NX_HOME dedup identical
bytes on disk instead of each holding/writing their own copy:

    obsstore/
      objects/<h2>/<sha256>.txt        object bytes (0600, no-clobber publish)
      meta/<h2>/<sha256>.json          {bytes, created, kind, tool} (miss-only)

Object identity is the sha256 of the CONTENT ALONE — not of
(tool, call_key, content) the way `_obs.make_observation_id` keyed it. That
was what defeated cross-process dedup: the same `read_file` output seen by
two swarm agents (different call keys) produced two files. Handles stay
byte-compatible: ``obs_`` + first 24 hex of the content hash, so a 24-hex
prefix of the old scheme's *content* portion still resolves (old objects
written by pre-0.98.1 nx keep working through the `_obs` legacy path; new
writes are pure-content).

Concurrency: publication is `atomic_write_private_noreplace` (`os.link`) —
exactly one of N racing writers publishes, the others see FileExistsError
and skip the write (one `stat` on the hot duplicate path). No locks, no
ledger: the directory IS the index.

GC (amortized behind `_sweep_due("obs_gc")`, the P6 gate; also wired into
`nx sessions --prune`):
  - TTL: objects older than NX_OBS_STORE_TTL_DAYS (default 7) are removed
    on touch of the sweep.
  - Quota: past NX_OBS_STORE_MAX_MB (default 512, 0 = off), oldest objects
    go first — every consumer already degrades gracefully on a miss
    (obs_recall: "unknown observation id"; overflow read: a normal read
    error), so eviction is safe-by-contract, and the 60s grace keeps a
    concurrently-written sibling's object safe mid-save.

Env (all read at call time, per CONTRIBUTING's no-import-time-env rule):
  NX_OBS_STORE_TTL_DAYS        7      — age prune
  NX_OBS_STORE_MAX_MB          512    — byte quota (0 = off)

Fail-open everywhere: any OSError degrades the caller to "store it
yourself" semantics — the store is an optimization, never a dependency.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Optional

from . import _state as st
from ._env import _env_int
from ._private import atomic_write_private_noreplace, private_dir

STORE_DIR_NAME = "obsstore"
AGE_GRACE_S = 60.0  # mirrors the image sidecar grace (concurrent writers)


def store_dir() -> str:
    return os.path.join(st.SESSION_HOME, STORE_DIR_NAME)


def objects_dir() -> str:
    return os.path.join(store_dir(), "objects")


def _h2(sha: str) -> str:
    return sha[:2]


def content_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def handle_id(text: str) -> str:
    """The `obs_...` handle for `text`: obs_ + first 24 hex of the CONTENT
    sha256. Byte-compatible with the pre-P10 24-hex id shape, but keyed on
    content alone so identical bytes dedup across tools, args, sessions,
    and processes."""
    return "obs_" + content_sha(text)[:24]


def object_path(sha: str) -> str:
    return os.path.join(objects_dir(), _h2(sha), sha + ".txt")


def meta_path(sha: str) -> str:
    return os.path.join(store_dir(), "meta", _h2(sha), sha + ".json")


# --- put / get ------------------------------------------------------------------


def put(text: str, kind: str = "obs", tool: str = "") -> str | None:
    """Store `text` content-addressed; return the object's sha, or None on
    an I/O failure (fail-open: the caller keeps its own copy). Duplicate
    writes cost one stat + one mkdir-exist_ok — the no-clobber publish
    makes N racing writers safe with no lock."""
    try:
        sha = content_sha(text)
        opath = object_path(sha)
        if os.path.exists(opath):
            _touch_meta(sha, kind, tool)  # atime-ish refresh, best-effort
            return sha
        private_dir(os.path.dirname(opath))
        try:
            atomic_write_private_noreplace(opath, text)
        except FileExistsError:
            return sha  # a sibling published it first — that IS the dedup win
        _write_meta(sha, kind, tool, len(text.encode("utf-8", "replace")))
        return sha
    except OSError:
        return None


def get(sha: str) -> str | None:
    """The object's text, or None (caller fails open / reports a miss)."""
    try:
        with open(object_path(sha), "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def exists(sha: str) -> bool:
    return os.path.exists(object_path(sha))


def expand_prefix(prefix: str) -> str | None:
    """The full sha whose object file starts with `prefix` (24-hex handles),
    or None. One listdir of a 2-entry fanout dir; unambiguous by sha256
    construction (a real prefix collision is astronomically unlikely, and
    deterministic first-match keeps it non-failing)."""
    if not prefix or len(prefix) < 2:
        return None
    sub = os.path.join(objects_dir(), _h2(prefix))
    try:
        for name in os.listdir(sub):
            if name.endswith(".txt") and name[:-len(".txt")].startswith(prefix):
                return name[:-len(".txt")]
    except OSError:
        return None
    return None


def resolve_handle(obs_id: str) -> Optional[tuple[str, str]]:
    """(sha, path) for an `obs_...` handle, or None. The id must be exactly
    obs_ + hex; the path must live inside objects/ (no traversal)."""
    if not obs_id.startswith("obs_"):
        return None
    sha = obs_id[len("obs_"):]
    if not sha or len(sha) > 64 or not all(c in "0123456789abcdef" for c in sha):
        return None
    path = object_path(sha)
    root = os.path.realpath(objects_dir())
    try:
        real = os.path.realpath(path)
    except OSError:
        return None
    # containment (not dirname-equality): objects live one fanout level down
    if os.path.dirname(real) != root and not real.startswith(root + os.sep):
        return None
    return sha, path


def _write_meta(sha: str, kind: str, tool: str, nbytes: int) -> None:
    """Miss-only metadata: bytes/created/kind/tool for GC ordering + /prune
    reporting. Duplicate objects never rewrite it (first writer wins)."""
    import json
    try:
        mpath = meta_path(sha)
        if os.path.exists(mpath):
            return
        private_dir(os.path.dirname(mpath))
        atomic_write_private_noreplace(mpath, json.dumps({
            "bytes": nbytes, "created": time.time(),
            "kind": kind, "tool": tool,
        }))
    except OSError:
        pass


def _touch_meta(sha: str, kind: str, tool: str) -> None:
    """Refresh an existing object's mtime (its TTL lease) without rewriting
    meta: the hot duplicate path is one utime, not a read + write."""
    try:
        os.utime(object_path(sha), None)
    except OSError:
        pass


# --- GC ---------------------------------------------------------------------------


def _iter_objects() -> list[tuple[float, str, int]]:
    """(mtime, path, size) for every object, oldest first. Unreadable
    entries are skipped; a missing dir is an empty store."""
    out: list[tuple[float, str, int]] = []
    d = objects_dir()
    try:
        for h2 in os.listdir(d):
            sub = os.path.join(d, h2)
            if not os.path.isdir(sub):
                continue
            for name in os.listdir(sub):
                if not name.endswith(".txt"):
                    continue
                p = os.path.join(sub, name)
                try:
                    stt = os.stat(p)
                    out.append((stt.st_mtime, p, stt.st_size))
                except OSError:
                    continue
    except OSError:
        return []
    out.sort()
    return out


def gc(now: float | None = None, grace_s: float = AGE_GRACE_S) -> dict[str, int]:
    """TTL prune + byte quota over the store. Returns removed/kept counts.
    Amortized callers gate this behind `_sweep_due("obs_gc")`; `--prune`
    calls it directly with grace_s=0."""
    now = now if now is not None else time.time()
    ttl_days = _env_int("NX_OBS_STORE_TTL_DAYS", 7)
    quota_mb = _env_int("NX_OBS_STORE_MAX_MB", 512)
    objs = _iter_objects()
    removed = 0

    def _drop(path: str) -> bool:
        nonlocal removed
        sha = os.path.basename(path)[:-len(".txt")]
        try:
            os.remove(path)
            try:
                os.remove(meta_path(sha))
            except OSError:
                pass
            removed += 1
            return True
        except OSError:
            return False

    # 1. TTL prune (with the concurrency grace — a sibling may have just
    #    published this object and not yet flushed its handle to a
    #    transcript).
    if ttl_days > 0:
        cutoff = now - ttl_days * 86400
        for mtime, path, _sz in list(objs):
            if mtime < cutoff and (now - mtime) > grace_s:
                if _drop(path):
                    objs.remove((mtime, path, _sz))

    # 2. Byte quota, oldest first (the image-store ordering; consumers
    #    degrade on a miss by contract).
    if quota_mb > 0:
        limit = quota_mb * 1024 * 1024
        total = sum(sz for _m, _p, sz in objs)
        for mtime, path, sz in list(objs):
            if total <= limit:
                break
            if (now - mtime) <= grace_s:
                continue  # never quota-evict a just-written object
            if _drop(path):
                total -= sz
                objs.remove((mtime, path, sz))

    return {"removed": removed, "kept": len(objs),
            "bytes": sum(sz for _m, _p, sz in objs)}


def store_stats() -> dict[str, int]:
    objs = _iter_objects()
    return {"objects": len(objs),
            "bytes": sum(sz for _m, _p, sz in objs)}
