"""Lightweight resource monitor for the current `nx` process — CPU% and RSS,
read straight from /proc (Linux) with an os.times()-based fallback elsewhere.
No psutil dependency: this is a couple of small numbers, not worth a new dep.
"""
from __future__ import annotations

import os
import platform
import time

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None  # ty: ignore[invalid-assignment]


def _read_proc_stat_times() -> tuple[int, int]:
    with open("/proc/self/stat") as f:
        parts = f.read().split()
    # fields 14/15 (1-indexed) = utime/stime, in clock ticks
    return int(parts[13]), int(parts[14])


def _sample_cpu_percent(interval: float) -> float:
    try:
        u1, s1 = _read_proc_stat_times()
        t1 = time.time()
        time.sleep(interval)
        u2, s2 = _read_proc_stat_times()
        t2 = time.time()
        ticks = os.sysconf("SC_CLK_TCK")
        cpu_time = (u2 - u1 + s2 - s1) / ticks
    except (OSError, ValueError):
        times1 = os.times()
        t1 = time.time()
        time.sleep(interval)
        times2 = os.times()
        t2 = time.time()
        cpu_time = (times2.user - times1.user) + (times2.system - times1.system)
    wall = t2 - t1
    return (cpu_time / wall) * 100.0 if wall > 0 else 0.0


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    if resource is not None:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KB, macOS in bytes.
        return maxrss / 1024 if sys_is_linux() else maxrss / (1024 * 1024)
    return None


def _peak_rss_mb() -> float | None:
    """Lifetime peak RSS of this process (ru_maxrss; children excluded)."""
    if resource is None:
        return None
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KB, macOS in bytes.
    return maxrss / 1024 if sys_is_linux() else maxrss / (1024 * 1024)


def _tree_rss_mb() -> dict[str, float | None] | None:
    """RSS of this process plus all its descendants, via /proc (Linux only).

    Spawns (shells, test runners, helper tools) are separate processes with
    their own RSS; a system monitor that "shows nx" usually aggregates them,
    so the tree total is the number that matches what the user sees.
    """
    if not sys_is_linux():
        return None
    try:
        ppid: dict[int, int] = {}
        rss: dict[int, int] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    parts = f.read().rsplit(")", 1)[-1].split()
                    ppid[pid] = int(parts[1])  # ppid: first field after comm
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss[pid] = int(line.split()[1])
                            break
            except (OSError, ValueError, IndexError):
                continue
        # collect descendants of self by walking up from every pid
        tree: set[int] = {os.getpid()}
        for pid in ppid:
            seen: set[int] = set()
            cur = pid
            while cur > 0 and cur not in tree and cur not in seen:
                seen.add(cur)
                cur = ppid.get(cur, 0)
            if cur in tree:
                tree |= seen
        if len(tree) <= 1:
            return None  # no children — the self RSS is the whole story
        self_mb = _rss_mb() or 0.0
        other = sum(rss[p] for p in tree if p != os.getpid() and p in rss)
        return {
            "self_mb": self_mb,
            "tree_mb": (other / 1024) + self_mb,
            "count": len(tree) - 1,
        }
    except OSError:
        return None


def sys_is_linux() -> bool:
    return platform.system() == "Linux"


def process_stats(interval: float = 0.2) -> dict[str, int | float | None]:
    """Sample this process's CPU% (over `interval` seconds, blocking) and RSS."""
    return {
        "pid": os.getpid(),
        "cpu_percent": _sample_cpu_percent(interval),
        "rss_mb": _rss_mb(),
        "peak_rss_mb": _peak_rss_mb(),
        "tree": _tree_rss_mb(),
    }
