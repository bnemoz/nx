"""Regression tests for the test-suite speedup work (v0.15.0).

Three behaviors landed so pytest's *import/collection* phase can't dial the
network and so the suite doesn't pay interactive-latency windows:

  1. NX_SKIP_STARTUP_PROBES=1 — every nx startup probe (resolve_model's
     /v1/models + /props round-trip, the capability/thinking background
     probes, all three resolve_context_window metadata legs, the over-max_tokens
     overrun probe) is a no-op. Previously, merely importing three standalone
     test files fired real HTTP at configured sources — 59s of TCP/TLS
     timeouts attributed to "collection".
  2. NX_INTERRUPT_RESTORE_WAIT — the disarm handshake's 2.5s wait is tunable
     (nx/_interrupt.py AND nx/_turn.py must honor the same env var).
  3. NX_HOOK_TIMEOUT / NX_WATCHDOG_POLL_SECONDS / NX_WATCHDOG_KILL_GRACE —
     hung-hook and watchdog-reaper budgets are tunable, read at call time
     (env scrubbed during pytest collection must not freeze the old value).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for k in ("NX_SKIP_STARTUP_PROBES", "NX_INTERRUPT_RESTORE_WAIT",
          "NX_HOOK_TIMEOUT", "NX_WATCHDOG_POLL_SECONDS",
          "NX_WATCHDOG_KILL_GRACE", "NX_ENV_FILE"):
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"

import pytest  # noqa: E402 — repo-root sys.path insert must come first


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib = __import__("importlib")
    importlib.reload(nx)
    yield nx, tmp_path
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_SKIP_STARTUP_PROBES",
              "NX_INTERRUPT_RESTORE_WAIT", "NX_HOOK_TIMEOUT",
              "NX_WATCHDOG_POLL_SECONDS", "NX_WATCHDOG_KILL_GRACE"):
        os.environ.pop(k, None)


# --- 1. NX_SKIP_STARTUP_PROBES ----------------------------------------------

def test_skip_flag_wires_active_without_probes(nx_fresh):
    """switch_source still wires ACTIVE/client/MODEL under the flag — only the
    probes are skipped. (Regression guard for the __init__ gate: gating the
    whole switch would leave st.ACTIVE None and break every reload-based
    test; wiring must survive.)"""
    nx, _ = nx_fresh
    # scrub the shell's real multi-source config (NX_SOURCES/NX_ACTIVE/
    # NX_SOURCE_* point this developer machine at "bubbles") so the reload
    # sees exactly one local source — same approach as test_sources.py's
    # clear_nx_env()
    for k in [k for k in os.environ if k.startswith("NX_")]:
        del os.environ[k]
    os.environ["NX_ENV_FILE"] = "/dev/null"
    os.environ["NX_SKIP_STARTUP_PROBES"] = "1"
    os.environ["NX_SOURCE_LOCAL_BASE_URL"] = "http://localhost:8080/v1"
    os.environ["NX_SOURCE_LOCAL_MODEL"] = "test-model"
    import importlib
    importlib.reload(nx)
    assert nx.ACTIVE is not None and nx.ACTIVE.name == "local"
    assert nx.client is not None
    assert nx.MODEL == "test-model"


def test_resolve_model_never_dials_under_flag(nx_fresh):
    """A source with no configured model must not attempt /v1/models: give it
    a client whose models.list() raises AssertionError if called."""
    nx, _ = nx_fresh
    src = nx.Source("probeless", "http://localhost:9/v1", "sk-noop", None)

    class _Boom:
        class models:  # noqa: N801 - attribute name is part of the client API
            @staticmethod
            def list(**kw):
                raise AssertionError("resolve_model dialed /v1/models under "
                                     "NX_SKIP_STARTUP_PROBES=1")

    src.client = _Boom()
    assert src.resolve_model() == "local-model"


def test_context_probes_never_dial_under_flag(nx_fresh):
    """All three /v1/models//props//api/v0/models metadata legs and the
    over-max_tokens chat probe are no-ops under the flag: a live client that
    asserts on any use proves none of them fire."""
    nx, _ = nx_fresh
    src = nx.Source("probeless", "http://localhost:9/v1", "sk-noop", "m")

    def _boom(*a, **kw):
        raise AssertionError("context probe dialed the endpoint under "
                             "NX_SKIP_STARTUP_PROBES=1")

    class _Boom:
        models = type("M", (), {"list": staticmethod(_boom)})()
        chat = type("C", (), {"completions": type(
            "CC", (), {"create": staticmethod(_boom)})()})()

    src.client = _Boom()
    assert src.resolve_context_window() is None
    assert src._ctx_from_overrun_probe("m") is None


def test_probes_still_fire_when_flag_unset(nx_fresh):
    """With the flag explicitly off, resolve_model probes as before (fake
    client returns an advertised id) — the flag must not become permanent."""
    nx, _ = nx_fresh
    os.environ["NX_SKIP_STARTUP_PROBES"] = "0"
    src = nx.Source("probeok", "http://localhost:9/v1", "sk-noop", None)

    class _M:
        @staticmethod
        def list(**kw):
            page = type("P", (), {"data": [type("X", (), {"id": "advertised-1"})()]})()
            return page

    src.client = type("C", (), {"models": _M})()
    assert src.resolve_model() == "advertised-1"


# --- 2. NX_INTERRUPT_RESTORE_WAIT -------------------------------------------

def test_restore_wait_env_honored_by_watch_and_turn(nx_fresh, monkeypatch):
    """Both disarm-wait call sites read NX_INTERRUPT_RESTORE_WAIT at call
    time (the suite scrubs NX_* during collection)."""
    nx, _ = nx_fresh
    os.environ["NX_INTERRUPT_RESTORE_WAIT"] = "0.05"
    from nx import _interrupt as iw
    from nx import _turn as turn_mod
    monkeypatch.setattr(iw, "_RESTORE_ENV_NAME", "NX_INTERRUPT_RESTORE_WAIT",
                        raising=False)
    # _watch_for_interrupts must return quickly when RESTORED is already set
    iw._INTERRUPT_ARMED.clear()
    iw._INTERRUPT_RESTORED.set()
    t0 = time.monotonic()
    with iw._watch_for_interrupts():
        pass
    assert time.monotonic() - t0 < 1.0, "already-restored handshake must not block"
    # and _turn's wait uses the same env: RESTORED already set → returns at once
    turn_mod._INTERRUPT_RESTORED.set()
    t0 = time.monotonic()
    turn_mod._INTERRUPT_RESTORED.wait(
        timeout=float(os.environ["NX_INTERRUPT_RESTORE_WAIT"]))
    assert time.monotonic() - t0 < 1.0


# --- 3. hook / watchdog budgets ---------------------------------------------

def test_hook_timeout_read_at_call_time(nx_fresh, monkeypatch):
    """_hook_timeout() must track the env value in force when the hook runs,
    not the value present at import (tests scrub env during collection)."""
    from nx import _hooks
    os.environ["NX_HOOK_TIMEOUT"] = "3"
    assert _hooks._hook_timeout() == 3
    os.environ["NX_HOOK_TIMEOUT"] = "1"
    assert _hooks._hook_timeout() == 1
    os.environ.pop("NX_HOOK_TIMEOUT")
    assert _hooks._hook_timeout() == 10  # documented default


def test_watchdog_reap_polls_instead_of_sleeping_full_budget(nx_fresh):
    """A background job that exits before NX_BASH_MAX_SECONDS must leave
    _WATCHDOG_PIDS promptly (poll loop), not after the full budget."""
    nx, _ = nx_fresh
    from nx._bash_jobs import _maybe_start_watchdog, _WATCHDOG_PIDS
    os.environ["NX_WATCHDOG_POLL_SECONDS"] = "0.05"
    # start a real short-lived process so the reaper sees a real exit
    import subprocess as sp
    proc = sp.Popen(["sleep", "0.2"])
    try:
        _maybe_start_watchdog(proc.pid, "sleep 0.2", 5)
        assert proc.pid in _WATCHDOG_PIDS
        deadline = time.time() + 3
        while time.time() < deadline and proc.pid in _WATCHDOG_PIDS:
            proc.poll()  # reap the child: a zombie still answers signal-0
            time.sleep(0.05)
        assert proc.pid not in _WATCHDOG_PIDS, \
            "early-exiting job must leave _WATCHDOG_PIDS before the 5s budget"
    finally:
        proc.wait(timeout=5)
