"""Shared pytest bootstrap for nx's test suite.

Three jobs, all about keeping the suite fast, hermetic, and xdist-safe:

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

3. PER-TEST STATE RESTORE (xdist-safe) — `pytest -n auto` runs files in
   arbitrary order per worker, so cross-file leakage of NX_* env vars and of
   process-global `nx` module state (the interrupt-watcher singleton, session
   caches, ...) makes the suite order-dependent. Before every test we rewind
   (a) all NX_* env vars to a scrubbed baseline, and (b) every `nx.*` module
   dict to its post-import snapshot. Files that set NX_* at module scope are
   honored: each module's import-time env assignments are captured (against a
   clean env, see pytest_collectstart) and re-applied for that module's tests.
"""
import atexit
import os
import shutil
import sys
import tempfile
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    # The 2.1 hard write fence is OFF in the suite's baseline: most test files
    # exercise tool BODIES directly (unfenced by design — the fence lives in
    # run_tool), and the dispatch-level ones write to tmp dirs outside the
    # imported nx's project root. tests/test_write_fence.py opts back in by
    # setting NX_WRITE_FENCE=1 in its autouse fixture, so the shipped default
    # (on) is what's actually tested.
    "NX_WRITE_FENCE": "0",
}
# Baseline every test's env is rewound to (AGENTS.md: tests must be immune to
# the developer's real NX_* credentials).
_SAFE_NX = dict(_DEFAULTS)
_SAFE_NX["NX_ENV_FILE"] = "/dev/null"

# Session tests and REPL-driving tests (which take the repo advisory lock via
# _repl.main -> _sessionlock.acquire) must never touch the developer's real
# ~/.nx or the repo's .nx/ — point NX_HOME at a per-process tempdir for ALL
# runs, not only xdist workers (serial runs appended to the real ~/.nx logs
# and input_history.json via _logging/_session_save).
# NX_SESSION_LOCK=0: a test suite is exactly the "many fake sessions in one
# repo" case the 0.103.0 advisory lock is not for — without it, REPL-driving
# tests flock the real repo's .nx/session.lock and can hang prompting on a
# contended lock.
_worker_home = tempfile.mkdtemp(prefix="nx-test-%d-" % os.getpid())
atexit.register(shutil.rmtree, _worker_home, ignore_errors=True)
_SAFE_NX["NX_HOME"] = _worker_home
_SAFE_NX["NX_SESSION_LOCK"] = "0"

# 4. TEMP-DIR REAPER — ~100 test modules do `tempfile.mkdtemp(prefix="nx-test-…")`
# at module scope and never clean up; each of the thousands of suite runs this
# machine has seen leaked a directory per module per xdist worker (80k /tmp/nx-*
# dirs when finally noticed). Wrap mkdtemp so every dir any test process creates
# is registered here and rmrf'd at interpreter exit (runs once per xdist worker,
# which is exactly the process boundary that owns the dirs).
_real_mkdtemp = tempfile.mkdtemp
_mkdtmp_registry: list[str] = []

def _tracking_mkdtemp(*a, **kw):
    d = _real_mkdtemp(*a, **kw)
    _mkdtmp_registry.append(d)
    return d

tempfile.mkdtemp = _tracking_mkdtemp

def _sweep_registered_tmpdirs():
    for d in _mkdtmp_registry:
        shutil.rmtree(d, ignore_errors=True)

atexit.register(_sweep_registered_tmpdirs)


def _scrub_env():
    """Drop every NX_* var, then lay down the safe baseline."""
    for _k in [k for k in os.environ if k.startswith("NX_")]:
        del os.environ[_k]
    os.environ.update(_SAFE_NX)


# ---------------------------------------------------------------------------
# 3. pristine baseline + per-module import-time env intents
# ---------------------------------------------------------------------------
_scrub_env()

import nx  # noqa: E402,F401  (baseline import; probes skipped)


def _is_nx(name):
    return name == "nx" or name.startswith("nx.")


# env baseline (NX_* plus a small allowlist of non-NX keys) and nx.* module
# dicts, both taken before any test module has been imported — the pristine
# state each test rewinds to. Module dicts are copied one level deep so tests
# mutating module-level containers in place (st._TOOL_RESULT_CACHE.clear()
# etc.) can't corrupt the baseline.
# Non-NX keys: _sources auto-registers built-in providers when these are set,
# so a module that sets one at import (test_sources.py) must not leak it into
# later modules' source builds on the same worker.

def _copy_val(v):
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, list):
        return list(v)
    if isinstance(v, set):
        return set(v)
    return v


_OTHER_ENV_KEYS = ("TOGETHER_API_KEY", "OPENROUTER_API_KEY", "ZAI_API_KEY")
_BASELINE_ENV = {k: v for k, v in os.environ.items()
                 if k.startswith("NX_") or k in _OTHER_ENV_KEYS}
_BASELINE_MODS = {
    name: {k: _copy_val(v) for k, v in mod.__dict__.items()}
    for name, mod in sys.modules.items()
    if _is_nx(name)
}

# module file path -> {NX_* var: value} assigned while that module was being
# imported (its module-scope `os.environ[...] = ...` lines).
_INTENTS = {}


def pytest_collectstart(collector):
    """Import each test module against a scrubbed NX_* env.

    pytest imports a test file when its Module node starts collecting, so
    scrubbing here means the module-scope env assignments we later read back
    in pytest_collectreport are exactly that module's own intents — never
    dirt inherited from a file collected earlier on the same worker.
    """
    if isinstance(collector, pytest.Module):
        _scrub_env()


def pytest_collectreport(report):
    if report.when != "collect" or "::" in report.nodeid:
        return  # only the Module-level report carries import-time state
    # Module-scope env *deletions* are intents too: files that scrub all NX_*
    # at import (test_tool_hooks.py, test_sources.py) must not silently erase
    # the conftest defaults (NX_HOOK_TIMEOUT etc.) for every later test.
    intents = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("NX_") and _BASELINE_ENV.get(k) != v
    }
    # Deletable intents: never treat the per-process isolation home or the
    # lock-off baseline as a module's deletable intent.
    intents.update({
        k: None for k in _BASELINE_ENV
        if k not in os.environ and k not in ("NX_HOME", "NX_SESSION_LOCK")
    })
    _INTENTS[os.path.abspath(report.nodeid)] = intents
    _scrub_env()  # next module's import starts clean too


def _restore(fspath):
    """Rewind NX_* env, cwd, and nx.* module state to the pristine baseline.

    Runs before every test's setup (so no state leaks in from earlier tests,
    whether serial or xdist-scrambled) and after every test's teardown (so a
    test that forgets to clean up can't poison its neighbors).
    """
    # cwd: a fixture that os.chdir(tmp_path) without restoring it must not
    # leave later tests resolving relative paths ("README.md", "foo.py")
    # against a temp dir — that was the serial-run failure in
    # test_breaker_ergonomics / test_local_risk_hint (0.95.0).
    os.chdir(_ROOT)
    # NOTE: do NOT set _INTERRUPT_ARMED/_INTERRUPT_EXIT here. Setting ARMED
    # wakes a parked watcher thread, whose wake path pays the interrupt-restore
    # handshake — doing that 2x per test added ~2 minutes to the serial run.
    # Events are snapshotted/restored as plain state below instead.

    # env: NX_* (+ the non-NX source keys) -> baseline, then this module's own
    # import-time deltas (value None = the module deleted it at import)
    for _k in [k for k in os.environ
               if k.startswith("NX_") or k in _OTHER_ENV_KEYS]:
        del os.environ[_k]
    os.environ.update(_BASELINE_ENV)
    for _k, _v in _INTENTS.get(str(fspath).split("::")[0], {}).items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v

    # module state: rewind each snapshotted nx.* module. Mutable containers
    # are cleared+refilled IN PLACE so identity shared across nx modules
    # (e.g. nx._state._TOOL_RESULT_CACHE re-exported by nx/__init__) survives;
    # names a test added that weren't in the baseline are dropped.
    for _name, _mod in list(sys.modules.items()):
        if not _is_nx(_name):
            continue
        # A daemon worker thread (subagent tier / worktree prefetch: e.g.
        # nx._subagent._run_job does a lazy `from ._worktree import ...`) can be
        # mid-`import nx._worktree` while this runs. Touching a module whose
        # loader hasn't finished is not safe: importlib's _load_unlocked ends
        # with an UNGUARDED `module = sys.modules.pop(spec.name)` (cpython 3.12
        # importlib/_bootstrap.py:946 — the failure-path `del` a few lines above
        # it does catch KeyError, this one does not), so deleting the entry out
        # from under that importer makes the thread raise KeyError('nx._worktree'),
        # which surfaces as PytestUnhandledThreadExceptionWarning. Rewinding a
        # half-executed module is wrong the same way (it wipes the globals the
        # body is still executing against). `__spec__._initializing` is True for
        # exactly that window (set before the sys.modules insert, cleared after
        # the pop), so skipping these closes the race; they get reaped by the
        # next restore pass (restore runs before setup and after teardown of
        # every test, so a lingering module is gone within one test).
        if getattr(getattr(_mod, "__spec__", None), "_initializing", False):
            continue
        _snap = _BASELINE_MODS.get(_name)
        if _snap is None:
            # Identity check, not pop-by-name: between the list() snapshot above
            # and here, an importer thread can have replaced this entry with its
            # own in-flight module (first import finished and was reaped, a
            # second worker started a fresh one). Popping by name alone would
            # yank *that* import — the exact KeyError this guard exists to
            # prevent. `sys.modules.get(_name) is _mod` closes that ABA window.
            if sys.modules.get(_name) is _mod:
                sys.modules.pop(_name, None)
            continue
        _d = _mod.__dict__
        for _k in [k for k in _d if k not in _snap]:
            del _d[_k]
        for _k, _tmpl in _snap.items():
            _cur = _d.get(_k)
            if isinstance(_tmpl, dict):
                if isinstance(_cur, dict):
                    _cur.clear()
                    _cur.update(_tmpl)
                else:
                    _d[_k] = dict(_tmpl)
            elif isinstance(_tmpl, list):
                if isinstance(_cur, list):
                    _cur[:] = _tmpl
                else:
                    _d[_k] = list(_tmpl)
            elif isinstance(_tmpl, set):
                if isinstance(_cur, set):
                    _cur.clear()
                    _cur.update(_tmpl)
                else:
                    _d[_k] = set(_tmpl)
            elif isinstance(_tmpl, threading.Event):
                if isinstance(_cur, threading.Event):
                    (_cur.set if _tmpl.is_set() else _cur.clear)()
                else:
                    _d[_k] = threading.Event()
                    if _tmpl.is_set():
                        _d[_k].set()
            else:
                _d[_k] = _tmpl

    # 0.143.0: the four homes the module-dict rewind above can't see — they
    # live on a threading.local, a ContextVar, or were added by a test that
    # reloaded nx (so the live module objects are forks the baseline never
    # snapshotted). Resolve every module via sys.modules AT RESET TIME.
    _sa = sys.modules.get("nx._subagent")
    _tls = sys.modules.get("nx._subagent_tls")
    if _tls is not None:
        _tls._SUBAGENT_TLS.__dict__.clear()      # subagent ctx + residue
    if _sa is not None:
        # quiet is assigned, never deleted, by production — clear the whole
        # TLS dict rather than popping one key (prod re-creates it anyway).
        _sa._ASYNC_QUIET_SLOTS.__dict__.clear()
        # Drain live workers BEFORE clearing their registry: _worker() re-reads
        # its _ASYNC_AGENTS record under the lock, so clearing with a worker
        # mid-flight loses the staged note and skips worktree removal.
        for _t in _sa._live_worker_threads():
            _t.join(timeout=2)
        with _sa._ASYNC_LOCK:
            _sa._ASYNC_AGENTS.clear()
    _no = sys.modules.get("nx._notices")
    if _no is not None:
        _no._CURRENT.set(None)
    # A Live leaked by a dashboard test (test_tabs_live / test_panel) stays on
    # the shared console's stack, which reroutes every later cprint through
    # rich — invisible to the suite's builtins.print captures (a source of the
    # test_preview_visibility "auto-allow [rule:" flake under -n auto).
    _rich = sys.modules.get("nx._rich")
    if _rich is not None:
        try:
            _stack = getattr(_rich.shared_console(), "_live_stack", None)
            if _stack:
                _stack.clear()
        except Exception:
            pass

    # 0.104.1: rewind nx._recovery's no-loop default store too. The ten
    # per-loop recovery fields (_LAST_PROMPT_TOKENS, the breaker evidence, ...)
    # route through descriptors on the state module's class, and that store
    # deliberately lives ON the class rather than in a module-dict entry —
    # conftest's module rewind used to clear it in place between tests, which
    # left the descriptors reading None and crashed the first turn loop of the
    # next test on the same worker (`cur_tokens > 0` with cur_tokens None,
    # nx/_turn.py:1086). Being out of dict reach also means the rewind above
    # no longer covers it, so reset it explicitly (cheap: ten entries).
    # Only when already imported — nx._recovery is not imported by `import nx`,
    # and importing it here would add a module to _BASELINE_MODS forever.
    _rec = sys.modules.get("nx._recovery")
    if _rec is not None:
        _reset = getattr(_rec, "_reset_store", None)
        if _reset is not None:
            _reset()

    # Re-derive the SESSION_HOME mirror from the just-applied env intents.
    # The module-state rewind above restores nx.SESSION_HOME / st.SESSION_HOME
    # to the baseline snapshot (real ~/.nx, since the baseline import has no
    # NX_HOME). A test module that sets NX_HOME at import gets that env value
    # re-applied, but without this the mirror still points at the real ~/.nx —
    # so REPL-driving tests appended "hello"/"world" to the user's real
    # input_history.json (and _write_session could too), trimming real history
    # to the ring-buffer cap every run.
    if "nx" in sys.modules:
        _nx = sys.modules["nx"]
        _home = os.path.expanduser(os.environ.get("NX_HOME", "~/.nx"))
        _nx.SESSION_HOME = _home
        if "nx._state" in sys.modules:
            sys.modules["nx._state"].SESSION_HOME = _home


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    """Rewind state, then re-assert the timing/probe defaults.

    Hookwrapper so the rewind lands before *all* of the item's setup (both
    autouse fixtures and xunit setup_function), not interleaved with it.
    Several standalone test files scrub ALL NX_* vars at module scope (e.g.
    test_sources.py's clear_nx_env()), which would silently undo the
    setdefaults above; setdefault keeps any value a test deliberately set
    (test_context_window opts back into the overrun probe).
    """
    _restore(str(item.fspath))
    for _k, _v in _DEFAULTS.items():
        os.environ.setdefault(_k, _v)
    yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    # post-yield: after fixture finalization (monkeypatch.undo et al.) has
    # restored what it patched, wipe whatever the test left behind.
    _restore(str(item.fspath))


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_auto_num_workers(config):
    """Cap `-n auto` at 32 workers.

    Each xdist worker re-imports all ~70 test modules during collection
    (~0.25s of nx import + module scope), so past ~32 workers the wall time
    *rises* — 17s at 128 workers vs 9.6s at 32 on this 128-core box.
    Firstresult: returning non-None here beats xdist's cpu_count(); returning
    None (explicit PYTEST_XDIST_AUTO_NUM_WORKERS set) defers to it. Only
    consulted for -n auto; explicit -n N bypasses this hook entirely.
    """
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None
    return 32
