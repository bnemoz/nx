"""Shared pytest bootstrap for nx's test suite.

Two jobs, both about keeping the suite fast and hermetic:

1. OFFLINE BY DEFAULT — NX_SKIP_STARTUP_PROBES=1 makes every `import nx` /
   `importlib.reload(nx)` skip nx's startup probes (resolve_model()'s
   /v1/models round-trip, the capability + thinking background probes, the
   ctx overrun probe). Without it, pytest merely *importing* the standalone
   assertion-style test files fired real HTTP at configured sources and
   burned 30-60s of TCP/TLS timeouts inside "collection" (see the audit note
   in .nx/memories). Probe-dependent tests override this deliberately —
   e.g. test_context_window.py sets it to "0" because its Test 6b asserts a
   silently-clamped overrun probe fires exactly once.

2. SHRUNKEN TIMING BUDGETS — several nx internals wait on fixed multi-second
   windows that only exist to absorb real-terminal/human latency (the
   interrupt-restore handshake after Esc, the watchdog reaper's poll tail,
   the hung-hook timeout). The default budget is correct for interactive use;
   tests shrink each via its NX_* tunable (read at call time, so these
   setdefaults apply even to the standalone files that reload nx).
"""
import os

_DEFAULTS = {
    # --- 1. no startup probes at import --------------------------------------
    "NX_SKIP_STARTUP_PROBES": "1",
    # --- 2. shrink interactive-latency budgets -------------------------------
    # interrupt watcher restore handshake: default 2.5s (covers the 2.0s Esc
    # redirect-drain window); the handshake itself completes in <0.1s under pytest
    "NX_INTERRUPT_RESTORE_WAIT": "0.05",
    # bash watchdog reaper poll tail: default 0.5s; tests poll _WATCHDOG_PIDS
    "NX_WATCHDOG_POLL_SECONDS": "0.05",
    # SIGTERM→SIGKILL grace: default 5s (SIGTERM rarely ignored by shells);
    # tests don't need the full courtesy period
    "NX_WATCHDOG_KILL_GRACE": "0.1",
    # pre/post tool-hook subprocess timeout: default 10s; the hung-hook test uses
    # a `sleep 30` script and only needs the veto message
    "NX_HOOK_TIMEOUT": "1",
}

for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)


def pytest_runtest_setup(item):
    """Re-assert the defaults before every test.

    Several standalone test files scrub ALL NX_* vars at module scope while
    pytest is still collecting (e.g. test_sources.py's clear_nx_env()), which
    would silently undo the setdefaults above for the whole execution phase.
    setdefault keeps any value a test deliberately set (test_context_window
    opts back into the overrun probe).
    """
    for _k, _v in _DEFAULTS.items():
        os.environ.setdefault(_k, _v)
