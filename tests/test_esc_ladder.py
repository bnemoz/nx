#!/usr/bin/env python3
"""UIUX_SPEC C1 (0.136.0): the Esc escalation ladder + hard tool cancellation.

Acceptance per the spec:
  * a `sleep 300` job launched through the real run_bash pipeline is killed by
    the ladder (SIGTERM -> SIGKILL after the grace to the process group) and
    marked `killed` in the /bg map;
  * the model receives a cancelled tool result (never a silent vanish);
  * no thread leak: _ensure_interrupt_watcher stays the singleton it always
    was, and the SIGKILL-escalation thread is bounded (one per kill, exits
    once the group is gone).

The watcher ladder itself is driven over a real pipe with the same harness
tests/test_perf_interrupt_disarm.py uses (no tty needed).
"""
import os
import sys
import tempfile
import threading
import time

_tmp = tempfile.mkdtemp(prefix="nx-ladder-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _bash_jobs as bj  # noqa: E402
from nx import _interrupt as iw  # noqa: E402
from nx import _state as st  # noqa: E402


def setup_function(_f):
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    m.SESSION_HOME = _tmp
    iw._HARD_CANCEL.clear()
    iw.clear_tool_cancel()
    iw._LAST_ESC_AT[0] = 0.0
    m._USER_INTERRUPTED.clear()


def _threads_named(prefix):
    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


# ------------------------------------------------------------- kill primitive

def test_cancel_kill_group_terminates_then_marks_killed():
    """A real `sleep 300` detached job: SIGTERM kills it, the map says
    killed, and the escalation thread exits (bounded — no leak)."""
    pid, log_path = bj._run_bash("sleep 300")
    assert pid.isdigit(), "run_bash must return a real pid"
    base = len(_threads_named("nx-cancel-kill"))
    bj.cancel_kill_group(pid, "sleep 300")
    deadline = time.time() + 5
    while bj._pid_is_alive(int(pid)) and time.time() < deadline:
        time.sleep(0.05)
    assert not bj._pid_is_alive(int(pid)), f"sleep {pid} survived the hard cancel"
    # marked killed in the /bg map
    jobs = {j["pid"]: j for j in bj.list_bg_jobs()}
    assert jobs[pid]["killed"] is True
    assert jobs[pid]["alive"] is False
    # escalation thread finished (sleep dies to SIGTERM immediately, so the
    # thread exits well before the grace period ends)
    deadline = time.time() + 3
    while len(_threads_named("nx-cancel-kill")) > base and time.time() < deadline:
        time.sleep(0.05)
    assert len(_threads_named("nx-cancel-kill")) <= base, "escalation thread leaked"
    bj._delete_old_bg_logs(max_logs=0, max_age_days=0)


def test_cancel_kill_group_survives_sigterm_ignorers():
    """A job trapping SIGTERM is SIGKILLed after the (shrunk) grace."""
    os.environ["NX_CANCEL_KILL_GRACE"] = "0.3"
    try:
        import importlib
        importlib.reload(bj)
        assert bj.CANCEL_KILL_GRACE == 0.3
        pid, _ = bj._run_bash("trap '' TERM; sleep 300; exit 0")
        time.sleep(0.3)  # let the trap install
        bj.cancel_kill_group(pid, "trap TERM; sleep 300")
        deadline = time.time() + 5
        while bj._pid_is_alive(int(pid)) and time.time() < deadline:
            time.sleep(0.05)
        assert not bj._pid_is_alive(int(pid)), "SIGKILL escalation did not land"
    finally:
        os.environ.pop("NX_CANCEL_KILL_GRACE", None)
        importlib.reload(bj)


# ------------------------------------------------------------ poll integration

def test_poll_pid_returns_cancelled():
    """A hard cancel outranks the plain wait: _poll_pid returns "cancelled"
    immediately instead of waiting out the timeout."""
    pid, log_path = bj._run_bash("sleep 300")
    try:
        iw._HARD_CANCEL.set()
        t0 = time.time()
        status = bj._poll_pid(pid, 30, check_esc=False)
        assert status == "cancelled"
        assert time.time() - t0 < 1.0
    finally:
        iw._HARD_CANCEL.clear()
        bj.cancel_kill_group(pid)
        bj._delete_old_bg_logs(max_logs=0, max_age_days=0)


def test_run_bash_cancelled_returns_model_facing_result_and_kills(monkeypatch):
    """Full run_bash: mid-poll escalation kills the job and the model gets a
    cancelled result (never [background])."""
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None, **kw: True)
    real_poll = bj._poll_pid
    fired = {"done": False}

    def poll_then_cancel(pid, timeout, **kw):
        status = real_poll(pid, 0.3, **{k: v for k, v in kw.items()
                                        if k != "expected_start_time"})
        # Escalate right after the short initial window: the hook is live.
        if not fired["done"]:
            fired["done"] = True
            assert iw.request_hard_cancel("test"), "cancel hook was not registered"
        return "cancelled"

    monkeypatch.setattr(bj, "_poll_pid", poll_then_cancel)
    try:
        out = bj.run_bash("sleep 300")
        assert out.startswith("[cancelled]"), out
        assert "killed" in out
        # hook slot cleared after the tool finished
        assert iw._CANCEL_FN is None
    finally:
        monkeypatch.undo()
        iw._HARD_CANCEL.clear()
        for j in bj.list_bg_jobs():
            if j["alive"]:
                bj.cancel_kill_group(j["pid"])
        bj._delete_old_bg_logs(max_logs=0, max_age_days=0)


def test_run_bash_ctrl_c_converts_to_cancelled(monkeypatch):
    """Ctrl+C (KeyboardInterrupt) inside the poll window = hard cancel: the
    job is killed and a cancelled result comes back — no turn unwind."""
    monkeypatch.setattr(m, "_confirm", lambda action, preview=None, **kw: True)

    def poll_raises(pid, timeout, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(bj, "_poll_pid", poll_raises)
    try:
        out = bj.run_bash("sleep 300")
        assert out.startswith("[cancelled]"), out
        assert iw.cancel_requested()  # sticky for the rest of the group
    finally:
        iw._HARD_CANCEL.clear()
        monkeypatch.undo()
        for j in bj.list_bg_jobs():
            if j["alive"]:
                bj.cancel_kill_group(j["pid"])
        bj._delete_old_bg_logs(max_logs=0, max_age_days=0)


# ----------------------------------------------------------------- the ladder

def test_double_esc_fires_the_ladder_and_hook(monkeypatch):
    """Two bare Esc inside the window (driven through the REAL watcher loop
    over a pipe) sets _HARD_CANCEL and fires the registered hook; two Esc
    FAR apart fire twice but do NOT escalate."""
    monkeypatch.setattr(iw, "apply_interrupt_mode", lambda fd, old: None)
    monkeypatch.setattr(iw, "restore_input_mode", lambda fd, old: None)
    monkeypatch.setattr(iw, "_ladder_window", lambda: 0.8)
    st._TURN_ACTIVE = False
    hook_calls = []
    iw.register_tool_cancel(lambda: hook_calls.append(1))
    r, w = os.pipe()
    try:
        # --- within the window: escalate
        iw._INTERRUPT_ARMED.set()
        iw._USER_INTERRUPTED.clear()
        iw._INTERRUPT_RESTORED.clear()
        iw._HARD_CANCEL.clear()
        t = threading.Thread(target=iw._armed_loop, args=(r, ("posix", [])),
                             daemon=True)
        t.start()
        time.sleep(0.05)
        os.write(w, b"\x1b")
        time.sleep(0.15)
        assert iw._USER_INTERRUPTED.is_set(), "first Esc must still interrupt"
        assert not iw._HARD_CANCEL.is_set()
        os.write(w, b"\x1b")  # second Esc inside 800ms
        deadline = time.time() + 1.0
        while not iw._HARD_CANCEL.is_set() and time.time() < deadline:
            time.sleep(0.02)
        assert iw._HARD_CANCEL.is_set(), "second Esc inside the window did not escalate"
        assert hook_calls == [1], f"hook fired {hook_calls}"
        iw._INTERRUPT_ARMED.clear()
        assert iw._INTERRUPT_RESTORED.wait(timeout=2.5)
        t.join(timeout=2.0)

        # --- outside the window: interrupt twice, never escalate
        iw._INTERRUPT_ARMED.set()
        iw._USER_INTERRUPTED.clear()
        iw._INTERRUPT_RESTORED.clear()
        iw._HARD_CANCEL.clear()
        iw._LAST_ESC_AT[0] = 0.0
        hook_calls.clear()
        t2 = threading.Thread(target=iw._armed_loop, args=(r, ("posix", [])),
                              daemon=True)
        t2.start()
        time.sleep(0.05)
        os.write(w, b"\x1b")
        time.sleep(1.1)          # > the 0.8s window
        os.write(w, b"\x1b")
        time.sleep(0.2)
        assert iw._USER_INTERRUPTED.is_set()
        assert not iw._HARD_CANCEL.is_set(), "Esc outside the window must not kill"
        assert hook_calls == []
        iw._INTERRUPT_ARMED.clear()
        assert iw._INTERRUPT_RESTORED.wait(timeout=2.5)
        t2.join(timeout=2.0)
    finally:
        iw._INTERRUPT_ARMED.clear()
        iw._HARD_CANCEL.clear()
        iw._INTERRUPT_RESTORED.set()
        iw.clear_tool_cancel()
        os.close(r)
        os.close(w)


# ------------------------------------------------ thread-leak regression (C1)

def test_ladder_adds_no_watcher_threads(monkeypatch):
    """_ensure_interrupt_watcher stays the singleton: arming/cancelling many
    times must never stack watcher threads (the 0.x leak history at
    _interrupt.py:17-35 is the regression this guards)."""
    calls = []

    def parking_body():
        calls.append(1)
        while not iw._INTERRUPT_EXIT.is_set():
            iw._INTERRUPT_ARMED.wait(0.02)

    m._interrupt_watcher = parking_body
    m._INTERRUPT_THREAD = None
    iw._INTERRUPT_EXIT.clear()
    try:
        for _ in range(10):
            iw._ensure_interrupt_watcher()
            iw._INTERRUPT_ARMED.set()
            iw.request_hard_cancel("spin")
            iw._HARD_CANCEL.clear()
            iw._INTERRUPT_ARMED.clear()
        assert len(calls) == 1, f"watcher body started {len(calls)}x"
        watchers = [t for t in threading.enumerate() if t.name == "nx-interrupt"]
        assert len(watchers) == 1, f"{len(watchers)} watcher threads stacked"
    finally:
        iw._INTERRUPT_EXIT.set()
        iw._INTERRUPT_ARMED.clear()
        iw._INTERRUPT_EXIT.clear()


# ------------------------------------------------------- dispatch-level skips

def test_tool_group_skips_after_hard_cancel(monkeypatch):
    """Once the cancel fired mid-group, the remaining calls get explicit
    CANCELLED/SKIPPED results and the turn returns TURN_ESC."""
    from nx import _turn_toolcalls as tt
    calls_seen = []

    def fake_run(name, args):
        calls_seen.append(name)
        if name == "boom":
            iw._HARD_CANCEL.set()
        return f"{name}:ok"

    monkeypatch.setattr(tt, "run_tool", fake_run)
    messages = []
    tcs = {
        0: {"id": "c0", "name": "boom", "args": "{}"},
        1: {"id": "c1", "name": "second", "args": "{}"},
        2: {"id": "c2", "name": "third", "args": "{}"},
    }
    try:
        status = tt._handle_native_tool_calls(messages, tcs, "", {}, 0)
        assert status == tt.TURN_ESC
        assert calls_seen == ["boom"], "later tools must NOT run after the cancel"
        bodies = [msg.get("content") or "" for msg in messages]
        assert any("CANCELLED" in b and "not started" in b for b in bodies)
        assert sum("SKIPPED" in b for b in bodies) == 1  # third only
        # the group consumed the cancel — flag cleared for later turns
        assert not iw.cancel_requested()
    finally:
        iw._HARD_CANCEL.clear()


def test_ctrl_c_inside_tool_converts(monkeypatch):
    """KeyboardInterrupt out of run_tool = hard cancel, results stay
    complete (every call gets a result), TURN_ESC returned."""
    from nx import _turn_toolcalls as tt

    def fake_run(name, args):
        if name == "boom":
            raise KeyboardInterrupt
        raise AssertionError("must not run after Ctrl+C")

    monkeypatch.setattr(tt, "run_tool", fake_run)
    messages = []
    tcs = {
        0: {"id": "c0", "name": "boom", "args": "{}"},
        1: {"id": "c1", "name": "second", "args": "{}"},
    }
    try:
        status = tt._handle_native_tool_calls(messages, tcs, "", {}, 0)
        assert status == tt.TURN_ESC
        roles = [msg for msg in messages if msg["role"] == "tool"]
        assert len(roles) == 2, "every tool call needs a result"
        assert "Ctrl+C" in roles[0]["content"]
        assert "SKIPPED" in roles[1]["content"]
    finally:
        iw._HARD_CANCEL.clear()
