"""`nx gc` — the artifact-GC command, layered on nx/_gc.gc_sweep.

The Docker-prune model: a bare `nx gc` is a DRY RUN that prints what the
retention rules would reclaim and how much space that frees, and `nx gc --all`
is the only surface that deletes anything. No confirmation prompt on `--all`:
the age/size rules in _gc.py are conservative by design (14/30-day ages, byte
caps on append-only logs), so the dry run is the review step and `--all` is the
opt-in.

Every knob the sweep honors (NX_GC_*, NX_USAGE_LOG_MAX_MB, NX_WEBCACHE_MAX_MB)
applies here automatically because the command only calls gc_sweep(); the
command never gates the sweep itself (the amortized save-path gate lives in
nx/_sessions_fs._maybe_gc_artifacts, not here). Exit 0 on success; nonzero only
if something outside gc_sweep raises — the sweep itself never does.
"""
from __future__ import annotations

import re
import sys

_USAGE = """\
Usage: nx gc [--all]

Report (default) or apply the artifact retention rules of nx/_gc.py over the
durable side-effect files under NX_HOME — sub-agent reports, undo snapshots,
the input-history buffer, logs/usage.jsonl and the web cache.

  (no flags)   dry run: print one line per artifact the rules would reclaim,
               a size summary, and exit without deleting anything
  --all        apply the rules for real (this is the only destructive flag)

-h, --help     print this usage and exit

Retention is configured by env vars (see docs/REFERENCE.md's env table):
NX_GC_AGENT_REPORT_DAYS, NX_GC_UNDO_DAYS, NX_USAGE_LOG_MAX_MB,
NX_WEBCACHE_MAX_MB; each rule is disabled by 0.

Exit codes: 0 = done, 2 = bad flags, 1 = unexpected error.
"""

# A gc_sweep line is either
#   "removed <path> (<size>)"                       (or "would remove …")
#   "trimmed <path> (freed <size>, kept <size>)"    (or "would trim …")
# — one artifact per line. The summary re-reads those sizes rather than
# re-walking the tree, so its numbers always match the lines above it.
_LINE_RE = re.compile(
    r"^(?P<verb>would remove|would trim|removed|trimmed) (?P<path>.+) "
    r"\((?:freed (?P<freed>[0-9.]+[BK]?MB|B)|(?P<size>[0-9.]+[BK]?MB|B))")


def _parse_size(token: str) -> int:
    """Inverse of nx/_gc._fmt_size (whole-byte/KB/MB magnitudes)."""
    t = token.strip()
    try:
        if t.endswith("MB"):
            return int(float(t[:-2]) * 1024 * 1024)
        if t.endswith("KB"):
            return int(float(t[:-2]) * 1024)
        if t.endswith("B"):
            return int(float(t[:-1]))
    except ValueError:
        return 0
    return 0


def _totals(lines: list[str]) -> tuple[int, int]:
    """(artifact count, bytes freed) straight out of the report lines."""
    count = 0
    freed = 0
    for line in lines:
        m = _LINE_RE.match(line)
        if not m:
            count += 1  # an unrecognized report line is still one artifact
            continue
        count += 1
        token = m.group("freed") or m.group("size") or ""
        freed += _parse_size(token)
    return count, freed


def _summary(count: int, freed: int, dry_run: bool) -> str:
    verb = "would be removed" if dry_run else "removed"
    return (f"{count} artifact(s) {verb}, "
            f"{freed / (1024 * 1024):.2f} MB reclaimed")


def _cli_gc(args: list[str]) -> int:
    """`nx gc [--all]` — dry-run by default, `--all` applies. Returns the exit
    code (0 success, 2 misuse, 1 unexpected error)."""
    for a in args:
        if a in ("-h", "--help"):
            sys.stdout.write(_USAGE)
            return 0
    dry_run = True
    for a in args:
        if a == "--all":
            dry_run = False
        elif a.startswith("-"):
            sys.stderr.write(f"nx gc: unknown flag {a!r} — try 'nx gc --help'\n")
            return 2
    try:
        from ._gc import gc_sweep
        lines = gc_sweep(dry_run=dry_run)
    except Exception as e:  # noqa: BLE001 — gc_sweep never raises; this is the
        # caller's own surface (an import blowup, a closed stdout), and the
        # contract is a nonzero exit rather than a traceback.
        sys.stderr.write(f"nx gc: {type(e).__name__}: {e}\n")
        return 1
    for line in lines:
        print(line)
    count, freed = _totals(lines)
    print(_summary(count, freed, dry_run))
    if dry_run:
        print("run `nx gc --all` to apply")
    return 0
