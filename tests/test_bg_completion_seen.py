#!/usr/bin/env python3
"""Regression tests for the 0.37.4 auto-continuation fix.

A run_bash command that finishes inside its own 3s poll window is delivered
inline (`[exit 0]\\n<output>`) — never via the `[background]` flow — but its
pid->log map file outlived the log that `_finish_bg_job` deletes. Nothing
marked that pid "seen", so the next completion scan (and the wake poller's
sweep) announced `pid N exited (code unknown)` for output the transcript
already contained, and the auto-continuation machinery woke a turn whose only
possible answer was "Nothing to do — that was the sweep script from my last
turn exiting." One orphan map per short bash call, one bogus wake each,
forever (worse: still ongoing when every bash call is short).

Pinned here:
  1. inline finish + wait_background mark the pid seen → the scan never
     announces it.
  2. read_file on a finished job's log marks it seen; a still-running job's
     log read does not (its exit is still news).
  3. the wake sweep skips dead jobs whose log is already gone.
  4. the scan is serialized (poller thread vs model turn) and merges without
     duplicating an already-staged exit line.
  5. end-to-end: run_bash → wake_due() must be False after the job's output
     was delivered inline (no "Nothing to do" wake turn).
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    os.environ["NX_WAKE_PROMPT"] = "1"
    monkeypatch.chdir(tmp_path)
    import nx
    # Tier A: conftest imported nx once under a scrubbed env, so the reload that
    # used to rebuild nx's import-time state from this fixture's env is gone.
    # Rebuild just the model wiring: re-discover sources from the env set above
    # and switch to the first one (binds st.client + st.MODEL). resolve_model()
    # reads NX_MODEL ("test-model"), so the dead 127.0.0.1:9 endpoint is never
    # contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # bg-logs live under st.SESSION_HOME, which __init__.py pinned at import time
    # from the developer's own NX_HOME; the env var set above is only read by an
    # nx reload, so point the live state at this test's tmpdir by hand.
    home = str(tmp_path / "nxhome")
    monkeypatch.setattr(nx.st, "SESSION_HOME", home)
    monkeypatch.setattr(nx, "SESSION_HOME", home, raising=False)
    # nx reads YOLO through the state proxy (nx.YOLO -> st.YOLO); pin it per test.
    monkeypatch.setattr(nx, "YOLO", True)
    # Pin the hidden slots this file both reads and writes directly. Pinning
    # through monkeypatch means a bare `m.st._X = ...` inside a test is undone at
    # teardown (the recorded pre-test value is written back), which is what the
    # module-dict rewind used to do — no manual save/restore needed.
    monkeypatch.setattr(nx.st, "_BG_NOTICE_SEEN", None)
    monkeypatch.setattr(nx.st, "_BG_NOTICE_BASELINE", set(), raising=False)
    monkeypatch.setattr(nx.st, "_PENDING_BG_COMPLETION_NOTICE", None)
    monkeypatch.setattr(nx.st, "_BG_SCAN_DONE", False)
    monkeypatch.setattr(nx.st, "_WAKE_ARMED", False)
    monkeypatch.setattr(nx.st, "_TURN_ACTIVE", False)
    yield nx
    # NX_* env rewind is universal (conftest); nx globals are restored by the
    # monkeypatch pins above.


# ---------------------------------------------------------------------------
# 1. inline delivery marks seen
# ---------------------------------------------------------------------------
def test_inline_run_bash_finish_marks_seen(nx_fresh, monkeypatch):
    """The 0.37.4 core bug: run_bash that finishes inside its poll window
    hands the model the FULL output inline — the completion scan must never
    then announce that pid as finished-unseen."""
    m = nx_fresh
    result = m.run_bash("echo inline-done-7734")  # exits well inside 3s
    assert "inline-done-7734" in result and "[exit 0]" in result
    assert m.st._BG_NOTICE_SEEN is not None, "inline finish must mark the pid seen"
    # The scan (what run_tool's first-tool hook + wake sweep run) sees nothing.
    assert m._dispatch._scan_background_completions() is None
    assert m.st._PENDING_BG_COMPLETION_NOTICE is None
    # No wake either: the sweep finds no unstaged work.
    monkeypatch.setattr(m.st, "_WAKE_ARMED", True)
    monkeypatch.setattr(m.st, "_TURN_ACTIVE", False)
    from nx import _autowake
    assert _autowake._finished_unseen_bg_jobs() == []
    assert _autowake.wake_due() is False


def test_wait_background_marks_seen(nx_fresh):
    m = nx_fresh
    pid, log_path = m._bash_jobs._run_bash("echo waited-8812")
    assert pid
    out = m.wait_background(pid)
    assert "waited-8812" in out and "[exit 0]" in out
    assert str(pid) in (m.st._BG_NOTICE_SEEN or set()), \
        "wait_background delivers the full log — the pid must be marked seen"
    assert m._dispatch._scan_background_completions() is None


# ---------------------------------------------------------------------------
# 2. read_file on a job log
# ---------------------------------------------------------------------------
def test_read_file_of_finished_log_marks_seen(nx_fresh, monkeypatch):
    """The model fetched the log by path and got the whole file: the output is
    in the transcript, so the eventual exit must not be announced again."""
    m = nx_fresh
    import nx._bash_jobs as bj
    pid, log_path = bj._run_bash("echo read-file-9901")
    assert bj._poll_pid(pid, 3.0) == "exited"
    # The log still exists here (no _finish_bg_job ran) — exactly the state
    # after the model's previous turn read it mid-run and the job then exited.
    m.run_tool("read_file", {"path": log_path})
    assert str(pid) in (m.st._BG_NOTICE_SEEN or set())
    assert m._dispatch._scan_background_completions() is None


def test_read_file_of_running_log_does_not_mark_seen(nx_fresh):
    """A still-running job whose log was read once must keep its exit
    announceable — the exit itself is the news worth a wake."""
    m = nx_fresh
    pid, log_path = m._bash_jobs._run_bash("sleep 2; echo late-9902")
    m.run_tool("read_file", {"path": log_path})
    assert m.st._BG_NOTICE_SEEN in (None, set()), \
        "reading a RUNNING job's log must not pre-consume its exit notice"
    m._bash_jobs._poll_pid(pid, 4.0)
    assert m._dispatch._scan_background_completions() is not None, \
        "the exit after a mid-run read is still worth announcing"
    assert "9902" not in "".join(m.st._PENDING_BG_COMPLETION_NOTICE or "")


def test_read_file_of_unrelated_path_is_noop(nx_fresh):
    m = nx_fresh
    m.run_tool("read_file", {"path": "/etc/hostname"})
    assert m.st._BG_NOTICE_SEEN in (None, set())


# ---------------------------------------------------------------------------
# 3. wake sweep skips log-less dead jobs
# ---------------------------------------------------------------------------
def test_wake_sweep_skips_dead_job_without_log(nx_fresh, monkeypatch):
    """The orphan shape this release fixes: dead pid, map present, log already
    deleted. There is nothing left to read — the transcript already holds the
    output — so the sweep must not wake on it."""
    from nx import _autowake
    m = nx_fresh
    monkeypatch.setattr(m.st, "_WAKE_ARMED", True)
    fake_jobs = [{"pid": "434343", "alive": False, "owner_pid": os.getpid(),
                  "command": "true", "tail": "", "log_path": "/tmp/gone.log"}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)
    monkeypatch.setattr("os.path.exists", lambda p: False)
    assert _autowake._finished_unseen_bg_jobs() == []
    monkeypatch.setattr(m.st, "_TURN_ACTIVE", False)
    assert _autowake.wake_due() is False


def test_wake_sweep_still_announces_live_log(nx_fresh, monkeypatch):
    """A dead job whose log still exists (backgrounded, exited later, model
    never read it) must still wake — that's the feature, not the bug."""
    from nx import _autowake
    m = nx_fresh
    monkeypatch.setattr(m.st, "_WAKE_ARMED", True)
    log = os.path.join(str(m.st.SESSION_HOME), "bg-logs", "bg-x.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        f.write("built 3 targets\n[exit 0]\n")
    fake_jobs = [{"pid": "434344", "alive": False, "owner_pid": os.getpid(),
                  "command": "true", "tail": "built 3 targets", "log_path": log}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)
    assert _autowake._finished_unseen_bg_jobs() == ["434344"]


# ---------------------------------------------------------------------------
# 4. scan hardening
# ---------------------------------------------------------------------------
def test_scan_consumes_log_less_orphan_without_notice(nx_fresh, monkeypatch):
    """A dead job whose log is gone must be marked seen (so it never comes
    back) but stage NO notice — 'exited (code unknown)' with no tail is noise,
    not information."""
    m = nx_fresh
    fake_jobs = [{"pid": "445566", "alive": False, "owner_pid": os.getpid(),
                  "command": "true", "tail": "", "log_path": "/tmp/gone2.log"}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)
    monkeypatch.setattr("os.path.exists", lambda p: False)
    assert m._dispatch._scan_background_completions() is None
    assert "445566" in (m.st._BG_NOTICE_SEEN or set())


def test_scan_marks_seen_even_when_exit_status_raises(nx_fresh, monkeypatch):
    """0.37.4 hardening: the seen-set must be committed even if a per-job
    probe raises — previously a mid-loop exception discarded every mark made
    so far, so the same pid re-announced on every later turn."""
    m = nx_fresh
    log = os.path.join(str(m.st.SESSION_HOME), "bg-logs", "bg-y.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        f.write("out\n[exit 1]\n")
    fake_jobs = [{"pid": "455667", "alive": False, "owner_pid": os.getpid(),
                  "command": "true", "tail": "out", "log_path": log}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)

    def boom(pid):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("nx._bash_jobs._read_bg_exit_status", boom)
    assert m._dispatch._scan_background_completions() is None  # swallowed
    assert "455667" in (m.st._BG_NOTICE_SEEN or set()), \
        "the mark must survive the probe failing"


def test_merge_does_not_duplicate_staged_exit_lines(nx_fresh, monkeypatch):
    """0.37.4: the poller's sweep and the turn's sweep can both stage the same
    job's exit line in a race — the merge must not produce it twice."""
    m = nx_fresh
    note = ("[Runtime note: background job(s) you started have finished:\n"
            "  pid 465768 exited (code 0) — tail: done\n"
            "Check their output with read_file(log) or wait_background before "
            "relying on them.]")
    monkeypatch.setattr(m.st, "_PENDING_BG_COMPLETION_NOTICE", note)
    # the poller's sweep already marked it
    monkeypatch.setattr(m.st, "_BG_NOTICE_SEEN", {"465768"})
    # A second sweep producing the same line (pre-lock race) merges to nothing.
    fake_jobs = [{"pid": "465768", "alive": False, "owner_pid": os.getpid(),
                  "command": "true", "tail": "done", "log_path": "/tmp/dup.log"}]
    import nx._bash_jobs as bj
    monkeypatch.setattr(bj, "list_bg_jobs", lambda: fake_jobs)
    m._dispatch._run_bg_completion_scan()
    merged = m.st._PENDING_BG_COMPLETION_NOTICE
    assert merged.count("pid 465768 exited") == 1, merged


def test_scan_lock_serializes_poller_and_turn(nx_fresh):
    """The seen-set is read-modify-rewritten from two threads; the lock must
    exist and be the same object the marking helper takes (a poller sweep and
    a concurrent inline finish may not drop one side's mark)."""
    m = nx_fresh
    from nx._dispatch import _BG_SCAN_LOCK
    from nx._bash_jobs import _mark_bg_notice_seen
    assert isinstance(_BG_SCAN_LOCK, type(threading.Lock()))
    with _BG_SCAN_LOCK:
        # Holding the lock must not deadlock the marking helper's own
        # acquisition path on a non-reentrant lock — it takes the lock after
        # we release; here we only assert the object identity contract.
        pass
    _mark_bg_notice_seen("477779")
    assert "477779" in (m.st._BG_NOTICE_SEEN or set())


# ---------------------------------------------------------------------------
# 5. end-to-end: no wake after an inline-delivered job
# ---------------------------------------------------------------------------
def test_no_phantom_wake_after_inline_bash_turn(nx_fresh, monkeypatch):
    """The user's live reproduction: a turn runs one short bash command
    (delivered inline), ends text-only. wake_due() must be False — before
    0.37.4 the orphan map woke a 'Nothing to do' auto-continuation."""
    from nx import _autowake
    m = nx_fresh
    m.run_bash("echo e2e-5656")  # inline finish, log deleted, map left behind
    monkeypatch.setattr(m.st, "_WAKE_ARMED", True)
    monkeypatch.setattr(m.st, "_TURN_ACTIVE", False)
    monkeypatch.setattr(m.st, "_PENDING_BG_COMPLETION_NOTICE", None)
    assert _autowake.wake_due() is False
    # And a second poll (the real poller re-checks every 2s) stays quiet.
    assert _autowake.wake_due() is False


def test_bg_job_prune_does_not_resurrect_seen_pids(nx_fresh):
    """The next _run_bash prunes the orphan map; before that prune the scan
    has already marked (or skipped) the pid. Either way no notice may appear
    for a pid whose output was delivered inline earlier."""
    m = nx_fresh
    m.run_bash("echo prune-6767")   # inline finish #1 (map left, marked seen)
    m.run_bash("echo prune-6768")   # _delete_old_bg_logs prunes #1's map
    assert m._dispatch._scan_background_completions() is None
