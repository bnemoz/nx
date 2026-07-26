"""Lightweight resource monitor for the current `nx` process — CPU% and RSS,
read straight from /proc (Linux) with an os.times()-based fallback elsewhere.
No psutil dependency: this is a couple of small numbers, not worth a new dep.
"""
import os
import time

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None


def _read_proc_stat_times():
    with open("/proc/self/stat") as f:
        parts = f.read().split()
    # fields 14/15 (1-indexed) = utime/stime, in clock ticks
    return int(parts[13]), int(parts[14])


def _sample_cpu_percent(interval):
    try:
        u1, s1 = _read_proc_stat_times()
        t1 = time.time()
        time.sleep(interval)
        u2, s2 = _read_proc_stat_times()
        t2 = time.time()
        ticks = os.sysconf("SC_CLK_TCK")
        cpu_time = (u2 - u1 + s2 - s1) / ticks
    except (OSError, ValueError):
        t1 = os.times()
        w1 = time.time()
        time.sleep(interval)
        t2 = os.times()
        t2_w = time.time()
        cpu_time = (t2.user - t1.user) + (t2.system - t1.system)
        t1, t2 = w1, t2_w
    wall = t2 - t1
    return (cpu_time / wall) * 100.0 if wall > 0 else 0.0


def _rss_mb():
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


def sys_is_linux():
    return os.uname().sysname == "Linux" if hasattr(os, "uname") else False


def process_stats(interval=0.2):
    """Sample this process's CPU% (over `interval` seconds, blocking) and RSS."""
    return {
        "pid": os.getpid(),
        "cpu_percent": _sample_cpu_percent(interval),
        "rss_mb": _rss_mb(),
    }
