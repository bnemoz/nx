"""Phase-1 artifact garbage collection: declarative age/size rules over the
durable side-effect files nx writes under NX_HOME (``st.SESSION_HOME``, see
nx/_state.py) that nothing ever cleaned.

What grows forever without this module:

- ``agents/<project-slug>/<job_id>.md`` — sub-agent reports (nx/_subagent.py);
  also drains the pre-0.145.2 flat ``agents/`` residue.
- ``web-cache/cache.jsonl`` — append/rewrite JSONL of fetched-page text
  (nx/_webfetch.py), which holds full page bodies (potentially from internal
  hosts).
- ``logs/usage.jsonl`` — append-only per-tool-call telemetry (nx/_logging.py).
- ``undo/<session-id>.json`` + the legacy ``undo.json`` — whole-file pre-edit
  snapshots (nx/_undo.py): up to the undo byte cap per file, often secrets.
- ``input_history.json`` — cross-session recall ring buffer
  (nx/_input_history.py).

Design: ONE declarative table (:data:`RULES`) of ``(path resolver, kind,
retention rule)`` entries, three kinds:

``file-age``
    delete the file when its mtime is older than N days.
``dir-age``
    walk the directory recursively, delete files whose mtime is older than N
    days (directories themselves stay — the writers recreate them lazily).
``size-cap``
    keep the file under N MB. Two trim strategies: ``tail`` (an append-only
    log: keep the NEWEST bytes, cut at a line boundary, rewritten IN PLACE so
    a live ``O_APPEND`` writer's fd keeps working) and ``jsonl-ttl`` (a cache:
    drop entries past their TTL first, then the oldest ones until under cap).

Every knob is an env int read at sweep time (so a mid-session change and the
test-suite's monkeypatch both take effect immediately), and ``0`` disables
that one rule — the same off-by-zero convention as the other retention flags
(NX_SESSION_RETAIN_*, NX_IMAGES_QUOTA_MB, NX_OBS_STORE_MAX_MB).

A sweep never raises: it runs on the amortized session-save path
(nx/_sessions_fs.py's ``_maybe_gc_artifacts``), where throwing would be worse
than leaving a file behind. Every removal is reported as a human-readable
line (``gc_sweep(dry_run=True)`` reports without touching anything), which is
what makes the sweep reviewable before it is trusted.

Deliberately NOT here in Phase 1: the cascade rule that would delete an
``agents/<slug>/<job>.md`` whose owning session transcript is gone (it needs a
job→session mapping the report files do not currently carry), and any change
to the obsstore / overflow / bg-logs TTLs, which already have their own GCs.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable

from . import _state as st
from ._env import _env_int
from ._private import atomic_write_private

DAY_SECONDS = 86400.0

# Default interval for the sweep's amortized gate, in seconds. The gate itself
# is the session-store sweep machinery (nx/_sessions_fs.py calls
# `_sweep_due("gc", NX_GC_SWEEP_INTERVAL)` from its save path), so the sweep
# runs off the existing once-per-save-hook plumbing but on this coarser
# interval — the walk covers whole directories. `gc_sweep()` itself is
# ungated: it applies the rules and reports, which is what a dry run and the
# tests need.
SWEEP_INTERVAL_DEFAULT = 86400

# --- path resolvers ----------------------------------------------------------
# Resolved LAZILY from st.SESSION_HOME at sweep time: NX_HOME is applied to
# SESSION_HOME after import (nx/__init__.py) and tests repoint it per test, so
# a path bound at import time would GC the wrong home.


def _agents_root() -> list[str]:
    return [os.path.join(st.SESSION_HOME, "agents")]


def _undo_paths() -> list[str]:
    # The per-session dir AND the legacy flat file (both written by _undo.py).
    return [os.path.join(st.SESSION_HOME, "undo"),
            os.path.join(st.SESSION_HOME, "undo.json")]


def _undo_dir() -> list[str]:
    return [os.path.join(st.SESSION_HOME, "undo")]


def _undo_legacy_file() -> list[str]:
    return [os.path.join(st.SESSION_HOME, "undo.json")]


def _usage_log() -> list[str]:
    return [os.path.join(st.SESSION_HOME, "logs", "usage.jsonl")]


def _webcache() -> list[str]:
    return [os.path.join(st.SESSION_HOME, "web-cache", "cache.jsonl")]


def _input_history() -> list[str]:
    return [os.path.join(st.SESSION_HOME, "input_history.json")]


def _webcache_ttl_s() -> float:
    """The web cache's own TTL, so an entry the fetcher considers expired is
    also garbage on disk (resolved live: tests and /reload can move it)."""
    try:
        from . import _webfetch
        return float(getattr(_webfetch, "CACHE_TTL_SECS", 3600.0))
    except Exception:  # noqa: BLE001 — a failed import must not skip GC entirely
        return 3600.0


@dataclass(frozen=True)
class Rule:
    """One declarative retention rule. Age rules read `days_env` (default
    `default_days`, 0/negative = rule disabled); size rules read `mb_env`
    (default `default_mb`, 0/negative = rule disabled)."""

    name: str
    kind: str                              # "file-age" | "dir-age" | "size-cap"
    resolve: Callable[[], list[str]]       # paths to consider, resolved at sweep
    days_env: str = ""
    default_days: int = 0
    mb_env: str = ""
    default_mb: int = 0
    trim: str = ""                          # "tail" | "jsonl-ttl"
    ttl_s: Callable[[], float] | None = None


RULES: tuple[Rule, ...] = (
    # Sub-agent reports: the fastest-growing artifact (every async agent leaves
    # one .md, plus intent records) and the least durable value — the answer
    # already reached the parent transcript. Recursive, so both the per-project
    # slug dirs and the legacy flat files drain under one rule.
    Rule("agent-reports", "dir-age", _agents_root,
         days_env="NX_GC_AGENT_REPORT_DAYS", default_days=14),
    # Undo snapshots hold whole pre-edit file contents (often secrets) for
    # sessions that may never be resumed again. Both undo kinds share the one
    # age knob.
    Rule("undo-dir", "dir-age", _undo_dir,
         days_env="NX_GC_UNDO_DAYS", default_days=30),
    Rule("undo-legacy", "file-age", _undo_legacy_file,
         days_env="NX_GC_UNDO_DAYS", default_days=30),
    # Recall buffer: an entry nobody scrolled back to in a month is not going
    # to be. Shares the undo age knob (small per-session state files).
    Rule("input-history", "file-age", _input_history,
         days_env="NX_GC_UNDO_DAYS", default_days=30),
    # Append-only telemetry: bounded by BYTES, not age (the file is written
    # continuously, so its mtime is always fresh — an age rule never fires).
    Rule("usage-log", "size-cap", _usage_log,
         mb_env="NX_USAGE_LOG_MAX_MB", default_mb=5, trim="tail"),
    # Fetched-page text cache: bounded by BYTES and by the fetcher's TTL.
    Rule("web-cache", "size-cap", _webcache,
         mb_env="NX_WEBCACHE_MAX_MB", default_mb=5,
         trim="jsonl-ttl", ttl_s=_webcache_ttl_s),
)


# --- reporting helpers -------------------------------------------------------


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _removed_line(path: str, size: int, dry_run: bool) -> str:
    verb = "would remove" if dry_run else "removed"
    return f"{verb} {path} ({_fmt_size(size)})"


def _trimmed_line(path: str, freed: int, kept: int, dry_run: bool) -> str:
    verb = "would trim" if dry_run else "trimmed"
    return f"{verb} {path} (freed {_fmt_size(freed)}, kept {_fmt_size(kept)})"


def _size_or_zero(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def _open_for_trim(path: str):
    """The write-phase open of :func:`_trim_tail`, factored out as a seam so a
    test can land a rotation exactly between the stat and the open (the race
    that module documents). No locking on purpose — the inode check is the
    fix; a lock around an in-place trim would only widen the window."""
    return open(path, "r+b")


# --- rule implementations ----------------------------------------------------


def _age_days(rule: Rule) -> int:
    return _env_int(rule.days_env, rule.default_days) if rule.days_env else 0


def _expired(mtime: float, cutoff: float) -> bool:
    return mtime <= cutoff


def _apply_file_age(rule: Rule, dry_run: bool) -> list[str]:
    days = _age_days(rule)
    if days <= 0:
        return []
    cutoff = time.time() - days * DAY_SECONDS
    out: list[str] = []
    for path in rule.resolve():
        try:
            stt = os.stat(path)
        except OSError:
            continue  # absent (or vanished mid-sweep): nothing to do
        if not os.path.isfile(path) or not _expired(stt.st_mtime, cutoff):
            continue
        size = int(stt.st_size)
        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                continue
        out.append(_removed_line(path, size, dry_run))
    return out


def _apply_dir_age(rule: Rule, dry_run: bool) -> list[str]:
    days = _age_days(rule)
    if days <= 0:
        return []
    cutoff = time.time() - days * DAY_SECONDS
    out: list[str] = []
    for root in rule.resolve():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    stt = os.lstat(path)
                except OSError:
                    continue
                if not _expired(stt.st_mtime, cutoff):
                    continue
                size = int(stt.st_size)
                if not dry_run:
                    try:
                        os.remove(path)
                    except OSError:
                        continue
                out.append(_removed_line(path, size, dry_run))
    return out


def _trim_tail(rule: Rule, path: str, limit: int, dry_run: bool) -> list[str]:
    """Append-only log: keep the newest `limit` bytes, cut at the next line
    boundary so the file still parses as JSONL. Rewritten IN PLACE (read tail,
    rewind, write, truncate) rather than tmp+rename: the async log writer
    (nx/_logging._AsyncLogWriter) holds this path open with O_APPEND, and an
    os.replace would silently orphan its fd onto the deleted inode.

    The flip side of in-place: that writer ALSO rotates this very path
    (os.replace to ``.1`` + recreate, _AsyncLogWriter._run), so between the
    stat here and the rewrite open the path can name a FRESH, short inode.
    Seeking the old size into it and truncating would erase everything
    appended after the rotation (or splice the old inode's tail into the new
    file). Guard: capture st_ino together with the size, re-verify it via
    fstat on the r+b handle, and re-derive the trim offset from THAT size so
    the write phase is self-consistent; on inode mismatch abort silently —
    the next sweep trims the rotated-to file (it owns its own ``.1`` cap)."""
    try:
        stt = os.stat(path)  # size AND inode, captured together
    except OSError:
        return []
    size = int(stt.st_size)
    if size <= limit:
        return []
    start = max(0, size - limit)
    try:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
    except OSError:
        return []
    nl = data.find(b"\n")
    if nl < 0:
        return []  # one unbreakable record larger than the cap: leave it alone
    start += nl + 1
    freed = start
    kept = size - start
    if freed <= 0:
        return []
    if dry_run:
        return [_trimmed_line(path, freed, kept, dry_run)]
    try:
        with _open_for_trim(path) as f:
            fst = os.fstat(f.fileno())
            if fst.st_ino != stt.st_ino:
                return []  # rotated out from under us: this isn't our file
            size = int(fst.st_size)
            if size <= limit:
                return []  # no longer over the cap: nothing to rewrite
            start = max(0, size - limit)
            f.seek(start)
            tail = f.read()
            nl = tail.find(b"\n")
            if nl < 0:
                return []  # same reasoning as above; leave the file alone
            start += nl + 1
            freed, kept = start, size - start
            f.seek(0)
            f.write(tail[nl + 1:])
            f.truncate()
    except OSError:
        return []
    return [_trimmed_line(path, freed, kept, False)]


def _trim_jsonl_ttl(rule: Rule, path: str, limit: int,
                    dry_run: bool) -> list[str]:
    """Cache JSONL (``{"k","t","v"}`` per line, nx/_webfetch.py): drop entries
    past the fetcher's TTL, corrupt lines, then — only if still over the byte
    cap — the oldest survivors. Kept lines are written back atomically (the
    cache writer uses tmp+os.replace too, so a reader never sees a partial
    file)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return []
    size = _size_or_zero(path)
    ttl = rule.ttl_s() if rule.ttl_s is not None else 0.0
    now = time.time()
    kept: list[tuple[float, int, str]] = []
    dropped = 0
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            dropped += 1
            continue
        try:
            ts = float(json.loads(s)["t"])
        except Exception:  # noqa: BLE001 — corrupt/partial line is garbage
            dropped += 1
            continue
        if ttl > 0 and now - ts >= ttl:
            dropped += 1
            continue
        kept.append((ts, len(line) + 1, line))
    kept.sort(key=lambda item: item[0])  # oldest first
    total = sum(n for _ts, n, _line in kept)
    while total > limit and len(kept) > 1:
        _ts, n, _line = kept.pop(0)
        total -= n
        dropped += 1
    if dropped == 0:
        return []
    freed = max(0, size - total)
    if dry_run:
        return [_trimmed_line(path, freed, max(0, total), dry_run)]
    try:
        atomic_write_private(
            path, "".join(line + "\n" for _ts, _n, line in kept))
    except OSError:
        return []
    return [_trimmed_line(path, freed, max(0, total), False)]


def _apply_size_cap(rule: Rule, dry_run: bool) -> list[str]:
    mb = _env_int(rule.mb_env, rule.default_mb) if rule.mb_env else 0
    if mb <= 0:
        return []
    limit = mb * 1024 * 1024
    out: list[str] = []
    for path in rule.resolve():
        if not os.path.isfile(path):
            continue
        if rule.trim == "jsonl-ttl":
            out.extend(_trim_jsonl_ttl(rule, path, limit, dry_run))
        else:
            out.extend(_trim_tail(rule, path, limit, dry_run))
    return out


_APPLIERS: dict[str, Callable[[Rule, bool], list[str]]] = {
    "file-age": _apply_file_age,
    "dir-age": _apply_dir_age,
    "size-cap": _apply_size_cap,
}


def gc_sweep(dry_run: bool = False) -> list[str]:
    """Apply every retention rule over the artifact dirs under SESSION_HOME;
    return one human-readable line per removal/trim (``removed <path> (<size>)``,
    ``trimmed <path> (freed …, kept …)``, or the ``would …`` form when `dry_run`).
    Ungated on purpose: amortization lives at the caller (the save path's
    ``_sweep_due("gc", …)`` in nx/_sessions_fs.py), so a dry run, a test and a
    future `/gc` command all get the full sweep. Never raises: a rule that
    throws contributes nothing and the next rule still runs."""
    lines: list[str] = []
    for rule in RULES:
        apply_rule = _APPLIERS.get(rule.kind)
        if apply_rule is None:
            continue
        try:
            lines.extend(apply_rule(rule, dry_run))
        except Exception:  # noqa: BLE001 — GC must never break the save path
            continue
    return lines
