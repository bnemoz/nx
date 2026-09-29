#!/usr/bin/env python3
"""PF-07 / PF-08 / PF-09 (0.56.0) — the per-turn latency tax.

Three fixes, each pinned by a test that fails on the pre-0.56.0 behaviour:

- **PF-07** — the optional dashboard push ran `urlopen(timeout=1.5)` INLINE on
  the REPL thread; a down endpoint cost up to ~4.5s of dead time before it
  self-disabled. `_Metrics.flush()` must now be a queue put.
- **PF-08** — `_finish_bg_job` paid a flat `time.sleep(0.2)` settle on every
  short `run_bash`. It must wait on the flushed `[exit: N]` trailer instead
  (bounded, and faster when the trailer is already there).
- **PF-09** — session-description generation made a 20s-timeout model call
  INLINE every SESSION_DESC_REFRESH turns on the REPL thread. It must now run on
  a daemon thread, with the REPL call returning immediately.
"""
import importlib
import os
import sys
import tempfile
import threading
import time

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------- PF-07 -------

def test_metrics_flush_does_not_block_on_a_dead_endpoint(monkeypatch):
    """flush() must enqueue, not POST. A URL pointing at a black hole would
    burn the full 1.5s timeout inline if the push were still synchronous."""
    from nx import _metrics as mm

    calls = []
    # A urlopen that would block forever if it were ever called inline.
    def never_returns(req, timeout=None):
        calls.append(timeout)
        time.sleep(30)
    monkeypatch.setattr(mm.urllib.request, "urlopen", never_returns)

    m = mm._Metrics("sid", model="x", source="s", push_url="http://10.255.255.1:9/push")
    t0 = time.monotonic()
    m.flush()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, (
        f"flush() blocked {elapsed:.2f}s — the push is still synchronous")
    assert not calls, "urlopen was called on the caller's thread"
    print("PASS — metrics flush enqueues without touching the network inline")


def test_async_pusher_posts_and_self_disables():
    """The background worker actually delivers a payload and latches off after
    the first failure (so a missing dashboard never slows the REPL)."""
    from nx import _metrics as mm

    seen = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    orig = mm.urllib.request.urlopen
    try:
        mm.urllib.request.urlopen = lambda req, timeout=None: (
            seen.append(req) or _Resp())
        pusher = mm._AsyncPusher()
        owner = type("O", (), {"_push_disabled": False})()
        pusher.submit("http://x/push", {"a": 1}, owner)
        pusher._q.join()
        assert len(seen) == 1, "payload was never POSTed by the worker"
        assert not pusher.disabled and not owner._push_disabled

        # Now make it fail — the pusher latches off and marks its owner.
        def boom(req, timeout=None):
            raise OSError("dashboard down")
        mm.urllib.request.urlopen = boom
        pusher.submit("http://x/push", {"a": 2}, owner)
        pusher._q.join()
        assert pusher.disabled, "pusher did not latch off after a failure"
        assert owner._push_disabled, "owner was not told the push is disabled"
        # A latched-off pusher drops further submits without queueing.
        pusher.submit("http://x/push", {"a": 3}, owner)
        assert pusher._q.qsize() == 0
    finally:
        mm.urllib.request.urlopen = orig
    print("PASS — async pusher delivers, then self-disables on failure")


# --------------------------------------------------------------- PF-08 -------

def test_finish_bg_job_waits_on_trailer_not_a_fixed_sleep(monkeypatch):
    """With the [exit: N] trailer already on disk the common case is a few ms,
    not the old flat 200ms settle."""
    jobs = importlib.import_module("nx._bash_jobs")
    path = os.path.join(_tmp, "bg-settle.log")
    with open(path, "w") as f:
        f.write("hello\n[exit: 0]\n")

    slept = []
    real_sleep = time.sleep
    monkeypatch.setattr(jobs.time, "sleep", lambda s: slept.append(s))
    t0 = time.monotonic()
    out = jobs._finish_bg_job(path)
    elapsed = time.monotonic() - t0
    monkeypatch.undo()
    real_sleep(0)

    assert out == "[exit 0]\nhello", out
    assert not os.path.exists(path), "finished job's log must be deleted"
    assert elapsed < 0.05, f"took {elapsed * 1000:.0f}ms — still paying the settle"
    assert 0.2 not in slept, "the flat 0.2s settle sleep is still there"
    print(f"PASS — _finish_bg_job returned in {elapsed * 1000:.1f}ms (was ~200ms)")


def test_wait_for_log_trailer_waits_for_a_late_marker():
    """When the trailer is not there yet, the helper waits for it (bounded) so
    we never read a half-flushed log — the real reason the old sleep existed."""
    import threading
    jobs = importlib.import_module("nx._bash_jobs")
    path = os.path.join(_tmp, "bg-late.log")
    with open(path, "w") as f:
        f.write("partial\n")

    def finish_later():
        time.sleep(0.04)
        with open(path, "a") as f:
            f.write("[exit: 3]\n")
    threading.Thread(target=finish_later, daemon=True).start()

    t0 = time.monotonic()
    code = jobs._wait_for_log_trailer(path, timeout=1.0)
    elapsed = time.monotonic() - t0
    assert code == 3, f"expected the late trailer's exit code, got {code!r}"
    assert elapsed >= 0.03, "returned before the trailer landed"
    print(f"PASS — waited {elapsed * 1000:.0f}ms for the late trailer, got exit 3")


def test_wait_for_log_trailer_is_bounded_when_marker_never_lands():
    """A truncated log (killed mid-command, no trailer) must not hang — the
    helper gives up after its timeout and _read_log still reports what it has."""
    jobs = importlib.import_module("nx._bash_jobs")
    path = os.path.join(_tmp, "bg-truncated.log")
    with open(path, "w") as f:
        f.write("no trailer here\n")
    t0 = time.monotonic()
    assert jobs._wait_for_log_trailer(path, timeout=0.05) is None
    assert 0.04 <= time.monotonic() - t0 < 0.5, "bound not honoured"
    print("PASS — truncated log returns None after a bounded wait")


# --------------------------------------------------------------- PF-09 -------

def test_desc_refresh_returns_immediately_and_calls_in_background(monkeypatch):
    """_maybe_refresh_description must return at once and do the model call on a
    daemon thread — the pre-0.56.0 code blocked the REPL for the call."""
    from nx import _sessions_fs as sf

    done = threading.Event()

    def slow_call(*a, **k):
        time.sleep(0.3)
        done.set()
    monkeypatch.setattr(sf, "_run_desc_refresh", slow_call)
    monkeypatch.setattr(sf, "_load_session", lambda sid: {"desc_turns": 0})

    import nx as _nx
    monkeypatch.setattr(_nx, "SESSION_DESC_REFRESH", 3)
    monkeypatch.setattr(sf.st, "client", object())
    monkeypatch.setattr(sf.st, "MODEL", "m")

    messages = [{"role": "user", "content": f"turn {i}"} for i in range(3)]
    t0 = time.monotonic()
    result = sf._maybe_refresh_description("sid", messages)
    elapsed = time.monotonic() - t0
    assert result is None
    # Non-blocking means returning well before the 0.3s stub finishes, not an
    # absolute wall-clock bound (schedulers can stretch even a fast return).
    assert elapsed < 0.25, f"returned after {elapsed:.2f}s — the call is still inline"
    # The background thread is running the (slow) stub.
    assert done.wait(timeout=5.0), "the background refresh never ran"
    print("PASS — description refresh returns at once, model call runs in background")


def test_desc_refresh_skips_below_threshold(monkeypatch):
    """No background thread before the first SESSION_DESC_REFRESH turns."""
    from nx import _sessions_fs as sf
    import nx as _nx
    monkeypatch.setattr(_nx, "SESSION_DESC_REFRESH", 6)
    monkeypatch.setattr(sf.st, "client", object())
    monkeypatch.setattr(sf.st, "MODEL", "m")
    called = []
    monkeypatch.setattr(sf, "_run_desc_refresh", lambda *a: called.append(1))
    assert sf._maybe_refresh_description(
        "sid", [{"role": "user", "content": "only one turn"}]) is None
    assert not called
    print("PASS — no refresh below the turn threshold")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
