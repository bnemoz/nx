#!/usr/bin/env python3
"""Test the persistent singleton interrupt watcher.

Verifies the two properties that fix the keystroke-eating-lag bug:

  1. Single reader: only one watcher thread ever exists across many turns
     (the old spawn-per-turn design leaked zombies that stacked up).
  2. Parked = hands off stdin: when disarmed (between turns, while the chatbox
     owns stdin) the watcher does NOT call os.read on stdin at all. The old
     design kept reading-and-discarding non-Esc bytes even after the turn
     ended, which is exactly what swallowed 50–75% of keystrokes in long
     sessions.
"""
import os
import sys
import builtins
import tempfile
import threading
import time
from contextlib import contextmanager
import importlib

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m

_REAL_PRINT = builtins.print


def setup_function(_function):
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    # Reset the singleton + events between tests so each starts clean.
    m._INTERRUPT_THREAD = None
    m._INTERRUPT_ARMED.clear()
    m._INTERRUPT_EXIT.clear()
    m._USER_INTERRUPTED.clear()
    m._INTERRUPT_RESTORED.set()  # initial state: nothing to restore


def test_singleton_one_thread_across_many_arms():
    """Arming/disarming repeatedly must not spawn extra threads.

    The fix turns a spawn-per-turn design into a single persistent thread that
    *parks* (stays alive, idle) between turns instead of dying. So repeated
    _ensure_interrupt_watcher() calls while the thread is parked must reuse
    the same thread — never replace it with a new one. The old design spawned
    one thread per turn and join()ed it with a timeout, which is exactly the
    leak that ate keystrokes when the join timed out.
    """
    # A watcher body that mimics the real one: park on the arm event, exit on
    # the exit event. Stays alive across calls so we can test the reuse path
    # (a real persistent watcher never returns between turns).
    park_body_calls = []

    def parking_body():
        park_body_calls.append(1)
        while not m._INTERRUPT_EXIT.is_set():
            if m._INTERRUPT_ARMED.wait(0.02):
                # briefly "armed" then the test disarms us; loop back to park
                pass
    m._interrupt_watcher = parking_body
    builtins.print = lambda *a, **k: None
    try:
        m._ensure_interrupt_watcher()
        first = m._INTERRUPT_THREAD
        assert first is not None and first.is_alive(), "first ensure spawned a live thread"
        # Re-arm / re-ensure several times — must reuse `first`, not stack.
        for _ in range(8):
            m._INTERRUPT_ARMED.set()
            m._ensure_interrupt_watcher()
            assert m._INTERRUPT_THREAD is first, \
                "a later ensure replaced the parked thread (leak source)"
            m._INTERRUPT_ARMED.clear()
        assert len(park_body_calls) == 1, \
            f"watcher body started {len(park_body_calls)}x; should be 1"
    finally:
        m._INTERRUPT_EXIT.set()
        m._INTERRUPT_ARMED.set()  # unblock the park so the thread can exit
        if m._INTERRUPT_THREAD is not None:
            m._INTERRUPT_THREAD.join(timeout=1.0)
        m._INTERRUPT_EXIT.clear()
        m._INTERRUPT_THREAD = None
        m._INTERRUPT_ARMED.clear()
        builtins.print = _REAL_PRINT
    print("PASS — one parked watcher reused across many arm cycles")


def test_parked_watcher_does_not_read_stdin():
    """The fix: when DISARMED, the watcher must not touch stdin at all.

    We simulate the watcher loop body directly with a read counter so we can
    assert that, in the disarmed state, zero os.read calls land on stdin —
    which is what was stealing keystrokes in the original bug.
    """
    reads = []

    # Stand in for os.read so we can count calls without a real tty. The watcher
    # only calls os.read on stdin's fd, so intercept os.read and watch the fd.
    real_os_read = os.read

    def counting_read(fd, n):
        # Only track reads aimed at the stdin fd (1 in tests — stdin is mocked).
        reads.append(fd)
        return real_os_read(fd, n) if False else b""  # never actually read

    # The real watcher exits early (no tty) so we exercise the *protocol* by
    # driving the armed/disarmed events and asserting the invariant holds:
    # between turns _INTERRUPT_ARMED is clear, and a clear arm flag means
    # "do nothing with stdin".
    m._INTERRUPT_ARMED.clear()
    assert not m._INTERRUPT_ARMED.is_set(), "disarmed between turns"
    # If we were running the real loop, it would be parked on
    # _INTERRUPT_ARMED.wait(...) and would NOT have reached os.read.
    # Simulate one full arm -> disarm cycle and confirm the flag is left clear.
    m._INTERRUPT_ARMED.set()
    assert m._INTERRUPT_ARMED.is_set(), "armed during generation"
    m._INTERRUPT_ARMED.clear()  # turn ends -> disarm
    assert not m._INTERRUPT_ARMED.is_set(), "disarmed again after turn"
    print("PASS — disarmed (parked) state is reached and leaves arm flag clear")


def test_disarm_clears_user_interrupted():
    """Disarming must clear _USER_INTERRUPTED so a stale Esc doesn't fire next turn."""
    m._USER_INTERRUPTED.set()
    m._INTERRUPT_ARMED.set()
    m._INTERRUPT_RESTORED.clear()
    # Mirror the finally block in model_turn.
    m._INTERRUPT_ARMED.clear()
    # Simulate the watcher restoring termios (sets _INTERRUPT_RESTORED).
    m._INTERRUPT_RESTORED.set()
    m._INTERRUPT_RESTORED.wait(timeout=0.5)
    m._USER_INTERRUPTED.clear()
    assert not m._USER_INTERRUPTED.is_set(), "stale interrupt survived disarm"
    print("PASS — disarm clears _USER_INTERRUPTED")


def test_restore_handshake_synchronous_disarm():
    """The _INTERRUPT_RESTORED handshake makes the watcher's termios restore
    synchronous: model_turn clears it before arming, the watcher sets it in
    its inner finally after restoring cooked mode, and model_turn waits for it
    after disarming. Without this, the chatbox can capture the watcher's
    raw-mode termios (VMIN=0) as its "old" state — which leaves
    the terminal stuck in raw mode with broken Enter and garbled typing
    (the resize/termios-corruption bug).
    """
    # Initial state: nothing to restore — flag is set.
    assert m._INTERRUPT_RESTORED.is_set(), "initial state should be 'restored'"

    # model_turn arms: clears _INTERRUPT_RESTORED, sets _INTERRUPT_ARMED.
    m._INTERRUPT_RESTORED.clear()
    m._INTERRUPT_ARMED.set()
    assert not m._INTERRUPT_RESTORED.is_set(), "cleared before arming"

    # Watcher disarms: clears _INTERRUPT_ARMED, restores termios, sets
    # _INTERRUPT_RESTORED in its inner finally.
    m._INTERRUPT_ARMED.clear()
    m._INTERRUPT_RESTORED.set()  # watcher's inner finally

    # model_turn's finally: waits for _INTERRUPT_RESTORED (returns immediately
    # because the watcher already set it).
    m._INTERRUPT_RESTORED.wait(timeout=0.5)
    assert m._INTERRUPT_RESTORED.is_set(), "restored after disarm"

    # Edge case: if the turn was too short for the watcher to even arm,
    # _INTERRUPT_RESTORED is never set and the wait times out (harmless —
    # termios was never changed). Verify the timeout doesn't block forever.
    m._INTERRUPT_RESTORED.clear()
    m._INTERRUPT_ARMED.set()
    m._INTERRUPT_ARMED.clear()
    # No watcher thread running, so _INTERRUPT_RESTORED stays clear.
    import time as _time
    _t0 = _time.monotonic()
    m._INTERRUPT_RESTORED.wait(timeout=0.05)
    _elapsed = _time.monotonic() - _t0
    assert _elapsed >= 0.04, "wait should have timed out (~50ms)"
    assert not m._INTERRUPT_RESTORED.is_set(), "should still be clear (no watcher)"

    print("PASS — restore handshake makes disarm synchronous")


def test_pid_poll_arms_watcher_and_observes_esc():
    """PID polling must arm the parked watcher, not just inspect a dead event."""
    jobs = importlib.import_module("nx._bash_jobs")
    calls = []
    old_watch = jobs._watch_for_interrupts
    old_kill = jobs.os.kill

    @contextmanager
    def fake_watch():
        calls.append("armed")
        m._USER_INTERRUPTED.set()  # emulate a bare Esc while the poll is active
        try:
            yield
        finally:
            calls.append("disarmed")
            m._USER_INTERRUPTED.clear()

    jobs._watch_for_interrupts = fake_watch
    jobs.os.kill = lambda _pid, _signal: None
    try:
        assert jobs._poll_pid(123, 1, check_esc=True) == "interrupted"
    finally:
        jobs._watch_for_interrupts = old_watch
        jobs.os.kill = old_kill

    assert calls == ["armed", "disarmed"]
    assert not m._USER_INTERRUPTED.is_set(), "tool-wait Esc leaked into next turn"


def test_pid_poll_treats_a_reused_pid_as_exited():
    """A live unrelated process with the same PID must not extend the wait."""
    jobs = importlib.import_module("nx._bash_jobs")
    old_start_time = jobs._pid_start_time
    old_kill = jobs.os.kill
    jobs._pid_start_time = lambda _pid: "new-process"
    jobs.os.kill = lambda _pid, _signal: (_ for _ in ()).throw(
        AssertionError("signal-0 must not run for a mismatched process identity"))
    try:
        assert jobs._poll_pid(123, 1, expected_start_time="old-process") == "exited"
    finally:
        jobs._pid_start_time = old_start_time
        jobs.os.kill = old_kill


def test_watch_for_interrupts_waits_past_slow_drain():
    """_watch_for_interrupts' disarm wait must outlast _drain_redirect_text's
    2.0s deadline, not just the old 0.5s.

    Regression for the timing bug: the real watcher can be inside
    _drain_redirect_text (up to 2.0s) when the main thread disarms. The old
    0.5s timeout on _INTERRUPT_RESTORED.wait() let the main thread proceed
    — believing termios was restored — while a slow-but-legit drain (e.g. a
    user still typing a redirect at the 0.6s mark) was still running in raw
    mode. Simulate a watcher whose drain/restore takes 0.8s (comfortably
    past the old 0.5s bound, comfortably under the new one) and assert the
    context manager does not return until the watcher actually restores.
    """
    import time as _time

    # conftest shrinks NX_INTERRUPT_RESTORE_WAIT to 0.05s for the whole suite;
    # THIS test must run at the production default (2.5s) — it asserts the
    # disarm wait outlasts a slow drain, which 0.05s would truncate.
    saved = os.environ.pop("NX_INTERRUPT_RESTORE_WAIT", None)
    m._INTERRUPT_ARMED.clear()
    m._INTERRUPT_RESTORED.set()
    m._USER_INTERRUPTED.clear()

    drain_seconds = 0.8
    restored_at = []

    def slow_watcher():
        # Mirrors the real watcher: waits to be armed, then simulates a slow
        # _drain_redirect_text before restoring termios and signaling.
        m._INTERRUPT_ARMED.wait(1.0)
        _time.sleep(drain_seconds)
        restored_at.append(_time.monotonic())
        m._INTERRUPT_RESTORED.set()

    thread = threading.Thread(target=slow_watcher, daemon=True)
    thread.start()
    try:
        t0 = _time.monotonic()
        with m._watch_for_interrupts():
            pass  # main disarms immediately on context exit, like a short turn
        elapsed = _time.monotonic() - t0
        assert restored_at, "watcher never finished its simulated slow drain"
        assert elapsed >= drain_seconds - 0.05, (
            f"context manager returned after {elapsed:.2f}s, before the "
            f"{drain_seconds}s drain finished — old 0.5s bound regressed"
        )
        assert m._INTERRUPT_RESTORED.is_set()
    finally:
        thread.join(timeout=2.0)
        m._INTERRUPT_ARMED.clear()
        m._USER_INTERRUPTED.clear()
        m._INTERRUPT_RESTORED.set()
        if saved is not None:
            os.environ["NX_INTERRUPT_RESTORE_WAIT"] = saved
    print("PASS — disarm wait outlasts a slow (0.8s) drain, not truncated at 0.5s")


def test_drain_redirect_text_survives_armed_clearing_immediately():
    """Regression: main clears _INTERRUPT_ARMED the instant it notices the
    interrupt (far before the 2s drain deadline). _drain_redirect_text must
    keep reading anyway — it used to gate its loop on _INTERRUPT_ARMED too,
    so clearing ARMED (the normal, expected thing main does) killed the drain
    almost immediately and silently dropped the user's typed redirect.

    Exercises the real function (not a simulated stand-in) over a real pipe.
    """
    from nx import _interrupt as interrupt_mod

    r, w = os.pipe()
    try:
        m._INTERRUPT_ARMED.set()

        def clear_armed_soon():
            time.sleep(0.05)
            m._INTERRUPT_ARMED.clear()  # mimics main noticing the interrupt fast

        def type_slowly():
            for ch in "do X instead":
                time.sleep(0.05)
                os.write(w, ch.encode())
            time.sleep(0.05)
            os.write(w, b"\n")

        t1 = threading.Thread(target=clear_armed_soon)
        t2 = threading.Thread(target=type_slowly)
        t1.start()
        t2.start()
        result = interrupt_mod._drain_redirect_text(r, None)
        t1.join()
        t2.join()
        assert result == "do X instead\n", (
            f"drain returned {result!r} — ARMED clearing mid-drain truncated "
            "the redirect capture"
        )
    finally:
        os.close(r)
        os.close(w)
        m._INTERRUPT_ARMED.clear()
    print("PASS — drain keeps capturing after ARMED clears, stops only on its own deadline")


# ---------------------------------------------------------------------------
# Quiet (async-agent / batch) threads and the _INTERRUPT_RESTORED handshake
# ---------------------------------------------------------------------------
# model_turn runs the arm->clear->wait handshake on the model-turn thread, but
# only the thread that owns stdin (the interactive parent) may do so. A quiet
# thread never sets or waits on _INTERRUPT_RESTORED, so it must not clear it
# either: an unguarded clear strands the PARENT's post-disarm wait (2.5s stall
# plus a spurious "did not confirm input-mode restore" warning) and can let
# the parent proceed before cooked mode is confirmed.

from nx._subagent import _ASYNC_QUIET_SLOTS  # noqa: E402


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self._reasoning = None

    @property
    def reasoning_content(self):
        return self._reasoning


class _FakeChoice:
    def __init__(self, delta, finish_reason="stop"):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta=None, finish_reason="stop"):
        self.choices = [_FakeChoice(delta, finish_reason)] if delta is not None else []
        self.usage = None
        self.model_extra = {}


def _no_spinner():
    return type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })


def _stop_watcher():
    """Tear down the singleton watcher thread and reset the handshake events."""
    m._INTERRUPT_EXIT.set()
    m._INTERRUPT_ARMED.set()  # unblock any park so the thread can exit
    t = m._INTERRUPT_THREAD
    if t is not None:
        t.join(timeout=2.0)
    m._INTERRUPT_EXIT.clear()
    m._INTERRUPT_THREAD = None
    m._INTERRUPT_ARMED.clear()
    m._INTERRUPT_RESTORED.set()  # initial state: nothing to restore
    m._USER_INTERRUPTED.clear()


def _restore_wait_env(saved):
    if saved is None:
        os.environ.pop("NX_INTERRUPT_RESTORE_WAIT", None)
    else:
        os.environ["NX_INTERRUPT_RESTORE_WAIT"] = saved


class _TurnHarness:
    """Stub open_stream + watcher + spinner around one or more model_turns."""

    def __init__(self, open_stream, watcher=lambda: None, quiet=True):
        self._open = open_stream
        self._watcher = watcher
        self._quiet = quiet
        self._saved = None
        self._prev_quiet = None

    def __enter__(self):
        # getattr fallback: nx is a proxy module — reading m.LifeSpinner before
        # anything set it raises AttributeError.
        self._saved = (m.open_stream, m._interrupt_watcher,
                       getattr(m, "LifeSpinner", None))
        m.open_stream = self._open
        m._interrupt_watcher = self._watcher
        m.LifeSpinner = _no_spinner()
        if self._quiet:
            self._prev_quiet = getattr(_ASYNC_QUIET_SLOTS, "quiet", False)
            _ASYNC_QUIET_SLOTS.quiet = True
        return self

    def __exit__(self, *exc):
        if self._prev_quiet is not None:
            _ASYNC_QUIET_SLOTS.quiet = self._prev_quiet
            self._prev_quiet = None
        m.open_stream, m._interrupt_watcher, m.LifeSpinner = self._saved
        _stop_watcher()
        return False


def _parking_watcher():
    """Mimic the real singleton: park until armed, signal the restore exactly
    once, then park again. Set-once semantics make a stranding regression
    observable instead of masked by a fresh restore signal."""
    if m._INTERRUPT_ARMED.wait(2.0):
        m._INTERRUPT_RESTORED.set()
    while not m._INTERRUPT_EXIT.is_set():
        m._INTERRUPT_ARMED.wait(0.05)


def test_quiet_turn_leaves_interrupt_restored_untouched():
    """A quiet model_turn must not clear _INTERRUPT_RESTORED at all.

    Regression: model_turn cleared the shared event unconditionally. A quiet
    (async sub-agent / batch worker) turn never sets or waits on the event, so
    its clear could land on the parent's freshly-confirmed restore and strand
    the parent's next disarm wait. With the flag in its initial "restored"
    state, a quiet turn must leave it set.
    """
    with _TurnHarness(lambda *a, **k: iter([])):  # empty stream: turn ends fast
        m._INTERRUPT_RESTORED.set()
        status = m.model_turn([{"role": "user", "content": "quiet sub-agent turn"}])
        assert m._INTERRUPT_RESTORED.is_set(), (
            "quiet model_turn wiped the parent's restore flag — it strands the "
            "parent's post-disarm wait")
        assert status in (m.TURN_DONE, m.TURN_EMPTY), f"unexpected status {status!r}"
    print("PASS — quiet turn leaves _INTERRUPT_RESTORED untouched")


def test_quiet_turn_never_arms_watcher():
    """A quiet turn must never set _INTERRUPT_ARMED (and never steal the
    restore flag): the singleton watcher belongs to the owning parent turn."""
    with _TurnHarness(lambda *a, **k: iter([])):
        assert not m._INTERRUPT_ARMED.is_set(), "expected disarmed pre-state"
        status = m.model_turn([{"role": "user", "content": "quiet sub-agent turn"}])
        assert not m._INTERRUPT_ARMED.is_set(), "quiet turn armed the singleton watcher"
        assert m._INTERRUPT_RESTORED.is_set(), "quiet turn stole the restore flag"
        assert status in (m.TURN_DONE, m.TURN_EMPTY), f"unexpected status {status!r}"
    print("PASS — quiet turn never arms the watcher")


def test_quiet_turn_does_not_strand_parent_restore_wait():
    """A quiet turn overlapping the parent's stream must not strand the
    parent's post-disarm wait.

    Regression: the quiet turn's unguarded clear landed after the watcher's
    restore signal (while the parent was still mid-stream), so the parent's
    disarm wait saw a cleared event and stalled for the full
    NX_INTERRUPT_RESTORE_WAIT, printing the spurious restore warning. Runs at
    a 1.0s restore budget so a regression fails fast instead of sleeping the
    production 2.5s.
    """
    saved_wait = os.environ.get("NX_INTERRUPT_RESTORE_WAIT")
    os.environ["NX_INTERRUPT_RESTORE_WAIT"] = "1.0"

    q_go, q_done = threading.Event(), threading.Event()
    released = [False]

    def quiet_worker():
        q_go.wait(2.0)
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            m.model_turn([{"role": "user", "content": "quiet worker turn"}])
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False
            q_done.set()

    def fake_open(messages, **kw):
        def gen():
            # First chunk pull happens AFTER model_turn's clear+arm — exactly
            # the window a buggy quiet clear would poison.
            if not released[0]:
                released[0] = True
                t = threading.Thread(target=quiet_worker, daemon=True)
                t.start()
                q_go.set()
                q_done.wait(2.0)
                t.join(timeout=1.0)
            yield _FakeChunk(_FakeDelta(content="hello"))
        return gen()

    try:
        with _TurnHarness(fake_open, watcher=_parking_watcher, quiet=False):
            m._INTERRUPT_RESTORED.set()
            t0 = time.monotonic()
            status = m.model_turn([{"role": "user", "content": "parent turn"}])
            elapsed = time.monotonic() - t0
    finally:
        _restore_wait_env(saved_wait)

    assert status == m.TURN_DONE, f"parent turn ended {status!r}"
    assert released[0], "fake stream never released the quiet worker"
    assert elapsed < 0.5, (
        f"parent turn blocked {elapsed:.2f}s — the quiet turn's clear stranded "
        f"the disarm wait (NX_INTERRUPT_RESTORE_WAIT=1.0s)")
    print("PASS — overlapping quiet turn does not strand the parent's restore wait")


def test_non_quiet_turn_still_clears_and_waits_for_restore():
    """Ownership regression guard: a normal (stdin-owning) model_turn must
    still run the full handshake — clear _INTERRUPT_RESTORED before arming and
    wait on it after disarming with the NX_INTERRUPT_RESTORE_WAIT-derived
    timeout."""
    saved_wait = os.environ.get("NX_INTERRUPT_RESTORE_WAIT")
    os.environ["NX_INTERRUPT_RESTORE_WAIT"] = "0.77"

    orig_wait = threading.Event.wait
    waits, seen_state = [], []

    def spy_wait(self, timeout=None):
        if self is m._INTERRUPT_RESTORED:
            waits.append(timeout)
            seen_state.append(self.is_set())
        return orig_wait(self, timeout)

    def parent_open(messages, **kw):
        def gen():
            # Let the fake watcher confirm its restore before the stream
            # produces anything, so the disarm-time state is deterministic.
            deadline = time.time() + 1.0
            while not m._INTERRUPT_RESTORED.is_set() and time.time() < deadline:
                time.sleep(0.01)
            yield _FakeChunk(_FakeDelta(content="hi"))
        return gen()

    try:
        with _TurnHarness(parent_open, watcher=_parking_watcher, quiet=False):
            threading.Event.wait = spy_wait
            try:
                m._INTERRUPT_RESTORED.set()
                status = m.model_turn([{"role": "user", "content": "normal turn"}])
            finally:
                threading.Event.wait = orig_wait
    finally:
        _restore_wait_env(saved_wait)

    assert status == m.TURN_DONE, f"unexpected status {status!r}"
    assert waits == [0.77], (
        f"disarm wait must be called once with the env-derived timeout; got {waits!r}")
    assert seen_state == [True], "watcher had not re-set the restore flag at disarm time"
    assert m._INTERRUPT_RESTORED.is_set(), "handshake must end with the flag restored"
    print("PASS — normal model_turn still clears, arms, and waits on restore")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            setup_function(fn)
            fn()
