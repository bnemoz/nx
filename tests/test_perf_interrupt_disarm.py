#!/usr/bin/env python3
"""PF-06 (0.56.0, T-8): the turn-end interrupt handshake must not wait out the
watcher's poll interval.

Before 0.56.0 a turn's `finally` cleared `_INTERRUPT_ARMED` and then blocked on
`_INTERRUPT_RESTORED` while the watcher finished its `wait_ready(fd, 0.1)` poll
— up to 100ms of dead time *after the last token and before the prompt
returned*, exactly the gap a user reads as "it's still thinking". The fix is a
wake pipe: `_disarm_interrupt_armed()` pokes it so the watcher's `select()`
returns at once and it restores cooked input mode in single-digit ms.

These tests drive the real armed-window loop (`_armed_loop`) over a real pipe —
no tty needed — and assert the disarm latency is well under the old poll
interval. They are plain pytest tests with no timing tunable: the whole point is
that the latency is a property of the mechanism, not of an env var.
"""
import os
import sys
import threading
import time
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _arming_thread_interrupt(monkeypatch):
    """Import the interrupt module and neutralise the termios primitives so an
    armed window can run over a plain pipe instead of a tty."""
    from nx import _interrupt as iw
    monkeypatch.setattr(iw, "apply_interrupt_mode", lambda fd, old: None)
    monkeypatch.setattr(iw, "restore_input_mode", lambda fd, old: None)
    return iw


def test_disarm_wakes_the_watcher_immediately(monkeypatch):
    """Clearing the arm flag + poking the wake pipe ends the armed window in
    single-digit ms, not after the 0.1s poll."""
    iw = _arming_thread_interrupt(monkeypatch)
    if iw._WAKE_R is None:
        print("SKIP — no wake pipe on this platform")
        return
    r, w = os.pipe()
    try:
        iw._INTERRUPT_ARMED.set()
        iw._USER_INTERRUPTED.clear()
        iw._INTERRUPT_RESTORED.clear()
        t = threading.Thread(target=iw._armed_loop, args=(r, ("posix", [])),
                             daemon=True)
        t.start()
        time.sleep(0.05)  # let the loop block in select([stdin, wake])
        t0 = time.monotonic()
        iw._disarm_interrupt_armed()
        assert iw._INTERRUPT_RESTORED.wait(timeout=1.0), \
            "watcher never restored input mode after disarm"
        elapsed = time.monotonic() - t0
        t.join(timeout=1.0)
        assert elapsed < 0.05, (
            f"disarm→restore took {elapsed * 1000:.0f}ms; the wake pipe did not "
            f"cut the watcher's 0.1s poll short")
    finally:
        iw._INTERRUPT_ARMED.clear()
        iw._INTERRUPT_RESTORED.set()
        os.close(r)
        os.close(w)
    print(f"PASS — disarm woke the watcher in {elapsed * 1000:.1f}ms")


def test_stale_wake_bytes_do_not_spin_the_loop(monkeypatch):
    """A leftover poke from a previous disarm must be drained before the next
    armed window blocks, or the loop would spin on a permanently-readable
    wake fd."""
    iw = _arming_thread_interrupt(monkeypatch)
    if iw._WAKE_R is None:
        return
    iw._wake_watcher()  # leave an undrained byte in the pipe
    r, w = os.pipe()
    try:
        iw._INTERRUPT_ARMED.set()
        iw._USER_INTERRUPTED.clear()
        iw._INTERRUPT_RESTORED.clear()
        # Sync on the loop actually reaching its select (after the entry
        # _drain_wake consumed the stale byte) instead of a blind sleep —
        # under -n auto load 0.05s can elapse before the thread even starts.
        first_select = threading.Event()
        thread_err = []
        real_select = iw._select_stdin_or_wake
        def _select_mark(fd, timeout):
            first_select.set()
            return real_select(fd, timeout)
        monkeypatch.setattr(iw, "_select_stdin_or_wake", _select_mark)
        t = threading.Thread(target=iw._armed_loop, args=(r, ("posix", [])),
                             daemon=True)
        t.start()
        # If the thread dies before select, the join below ends early and the
        # message says so instead of timing out blind.
        first_select.wait(timeout=10.0)
        if not first_select.is_set():
            t.join(timeout=1.0)
        assert first_select.is_set(), (
            "armed loop never reached select (thread alive: "
            f"{t.is_alive()})")
        # If the stale byte were not drained the loop would have exited early
        # (treating the wake as a disarm). It must still be armed.
        assert not iw._INTERRUPT_RESTORED.is_set(), \
            "stale wake byte made the armed window exit immediately"
        iw._disarm_interrupt_armed()
        assert iw._INTERRUPT_RESTORED.wait(timeout=1.0)
        t.join(timeout=1.0)
    finally:
        iw._INTERRUPT_ARMED.clear()
        iw._INTERRUPT_RESTORED.set()
        os.close(r)
        os.close(w)
    print("PASS — stale wake byte drained, armed window still blocked on stdin")


if __name__ == "__main__":
    import types
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            class _MP:
                def setattr(self, obj, attr, val, raising=True):
                    old = getattr(obj, attr, None)
                    setattr(obj, attr, val)
                    self._undo.append((obj, attr, old))

                def __init__(self):
                    self._undo = []

                def undo(self):
                    for o, a, v in self._undo:
                        setattr(o, a, v)

            mp = _MP()
            try:
                fn(mp)
            finally:
                mp.undo()
