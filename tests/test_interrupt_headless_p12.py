#!/usr/bin/env python3
"""Swarm P12: the stdin interrupt watcher must not run for headless/quiet
processes.

Invariants:
- a quiet loop (async sub-agent, batch worker) never STARTS the watcher
  thread (it never arms it — starting it is pure per-agent overhead);
- NX_INTERRUPT_WATCHER=0 disables the thread and the whole arm handshake
  for the process (headless density: no stdin thief, no termios snapshot);
- the default (unset env) behavior is unchanged: a non-quiet turn starts
  and arms the watcher exactly as before;
- _watch_for_interrupts under =0 is a no-op that never waits on the
  restore handshake.
"""
import os
import sys
import tempfile
import threading

_tmp = tempfile.mkdtemp(prefix="nx-p12-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import nx as m  # noqa: E402
import nx._interrupt as I  # noqa: E402
import nx._state as st  # noqa: E402
from nx._subagent_tls import _SUBAGENT_TLS  # noqa: E402
from nx._subagent_tiers import _SubagentContext  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    saved = st._INTERRUPT_THREAD
    st._INTERRUPT_THREAD = None
    yield
    st._INTERRUPT_THREAD = saved


def _quiet_ctx():
    return _SubagentContext({}, quiet=True)


def test_watcher_disabled_env(monkeypatch):
    monkeypatch.setenv("NX_INTERRUPT_WATCHER", "0")
    assert I._watcher_disabled() is True
    monkeypatch.delenv("NX_INTERRUPT_WATCHER", raising=False)
    assert I._watcher_disabled() is False
    monkeypatch.setenv("NX_INTERRUPT_WATCHER", "1")
    assert I._watcher_disabled() is False


def test_quiet_loop_does_not_start_thread():
    _SUBAGENT_TLS.ctx = _quiet_ctx()
    try:
        I._ensure_interrupt_watcher()
        assert st._INTERRUPT_THREAD is None  # not started
    finally:
        _SUBAGENT_TLS.ctx = None


def test_disabled_env_does_not_start_thread(monkeypatch):
    monkeypatch.setenv("NX_INTERRUPT_WATCHER", "0")
    I._ensure_interrupt_watcher()
    assert st._INTERRUPT_THREAD is None


def test_nonquiet_still_starts_thread(monkeypatch):
    """Default behavior unchanged: a non-quiet (stdin-owning) turn starts
    the singleton. The thread body exits immediately without a tty (pytest
    capture) — we only assert it was STARTED."""
    monkeypatch.delenv("NX_INTERRUPT_WATCHER", raising=False)
    I._ensure_interrupt_watcher()
    try:
        assert st._INTERRUPT_THREAD is not None
    finally:
        if st._INTERRUPT_THREAD is not None:
            I._INTERRUPT_EXIT.set()
            st._INTERRUPT_THREAD = None


def test_watch_for_interrupts_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("NX_INTERRUPT_WATCHER", "0")
    with I._watch_for_interrupts():
        assert not I._INTERRUPT_ARMED.is_set()
        assert not I._USER_INTERRUPTED.is_set()
    assert st._INTERRUPT_THREAD is None
