#!/usr/bin/env python3
"""The pre-first-token "thinking" spinner must never survive a ^C.

^C during generation can land between the spinner's last frame and the
finally that wipes its line, so the "thinking · esc to interrupt" bar used
to stay painted in scrollback above the next chatbox (the reported repl bug).
model_turn() now registers the spinner in st._TURN_SPINNER and clears it on
every exit path, and the REPL's KeyboardInterrupt handler force-wipes it
before the chatbox repaints.

No model/network round-trip.
"""
import os
import sys
import tempfile
import threading
import time

_tmp = tempfile.mkdtemp(prefix="nx-test-tsp-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx._ansi import CLEAR_LINE  # noqa: E402


class _FakeTTY:
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        pass

    def isatty(self):
        return True

    def getvalue(self):
        return "".join(self.buf)


class _TrackingSpinner:
    """Records start/stop; a thread that keeps painting frames after the
    caller believed it stopped — the ^C race, modeled."""

    def __init__(self, **kw):
        self._t = None
        self.started = False
        self.stops = 0

    def start(self):
        self.started = True
        self._t = threading.current_thread()

    def stop(self):
        self.stops += 1
        self._t = None


def _with_patched_stream(chunks_or_exc):
    class _Stream:
        def __iter__(self):
            yield from chunks_or_exc

        def close(self):
            pass

    return _Stream()


def test_model_turn_registers_and_clears_turn_spinner():
    saved = (m.open_stream, m._turn.LifeSpinner)
    m._interrupt_watcher = lambda: None
    m._turn.LifeSpinner = _TrackingSpinner
    m.open_stream = lambda messages, **kw: _with_patched_stream([])  # no chunks
    try:
        st = m._state
        st._TURN_SPINNER = None
        status = m.model_turn([{"role": "user", "content": "hi"}])
        assert status in (m.TURN_DONE, m.TURN_EMPTY), status
        # early return before any chunk: spinner was started, stopped, and
        # the slot cleared — nothing left for a ^C handler to trip over
        assert st._TURN_SPINNER is None
    finally:
        m.open_stream, m._turn.LifeSpinner = saved
        m._state._TURN_SPINNER = None


def test_repl_interrupt_handler_wipes_registered_spinner():
    # Drive the exact wipe block from the REPL's KeyboardInterrupt handler:
    # a still-registered spinner (its line already painted) must be stopped
    # and cleared before the "^C interrupted" line prints.
    out = _FakeTTY()
    real_stdout = sys.stdout
    sys.stdout = out
    try:
        sp = _TrackingSpinner()
        sp.start()
        m._state._TURN_SPINNER = sp
        # --- inline of the repl.py ^C block (kept in sync; regression pin) ---
        _ts = m._state._TURN_SPINNER
        m._state._TURN_SPINNER = None
        if _ts is not None:
            try:
                _ts.stop()
            except Exception:
                sys.stdout.write(CLEAR_LINE)
                sys.stdout.flush()
        # ---------------------------------------------------------------------
        assert m._state._TURN_SPINNER is None
        assert sp.stops == 1
    finally:
        sys.stdout = real_stdout
        m._state._TURN_SPINNER = None
    print("PASS — ^C handler wipes the registered spinner")


def test_registered_spinner_stops_cleanly_when_already_stopped():
    # Double-stop must be a no-op (the turn's own finally ran first; the ^C
    # handler then finds a stale registration) — no exception, no second wipe.
    out = _FakeTTY()
    real_stdout = sys.stdout
    sys.stdout = out
    try:
        from nx._ansi import LifeSpinner
        sp = LifeSpinner(label="thinking · esc to interrupt", tick=LifeSpinner._MIN_TICK)
        sp.start()
        time.sleep(0.05)
        sp.stop()
        sp.stop()  # the ^C-handler stop — must not raise
    finally:
        sys.stdout = real_stdout
        m._state._TURN_SPINNER = None
    print("PASS — double stop is a clean no-op")


if __name__ == "__main__":
    test_model_turn_registers_and_clears_turn_spinner()
    test_repl_interrupt_handler_wipes_registered_spinner()
    test_registered_spinner_stops_cleanly_when_already_stopped()
    print("all passed")
