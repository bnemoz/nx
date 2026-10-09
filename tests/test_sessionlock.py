"""nx/_sessionlock.py (0.103.0, session-isolation phase 2): the per-repo
advisory lock warns a second REPL that another session is editing here."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("NX_HOME", str(tmp_path / "nxhome"))
    monkeypatch.chdir(tmp_path)
    import nx
    nx.st.PROJECT_ROOT = str(tmp_path)
    nx.st.SESSION_ID = "20260101-000000-aaa111"
    from nx import _sessionlock as sl
    sl._HELD.clear()
    sl._HELD_PATHS.clear()
    yield sl, tmp_path
    sl.release()
    nx.st.SESSION_ID = None


def test_first_session_acquires(repo):
    sl, rp = repo
    assert sl.acquire() is True
    lock = rp / ".nx" / "session.lock"
    assert lock.exists(), sorted(p.name for p in (rp / ".nx").iterdir()) if (rp / ".nx").is_dir() else "no .nx"
    data = json.loads(lock.read_text())
    assert data["session_id"] == "20260101-000000-aaa111"
    assert data["pid"] == os.getpid()


def test_second_session_sees_holder(repo):
    sl, rp = repo
    assert sl.acquire() is True
    # A second process can't take the flock: verify via subprocess flock probe.
    r = subprocess.run(
        ["flock", "-n", str(rp / ".nx" / "session.lock.lock"), "true"],
        capture_output=True)
    assert r.returncode != 0  # lock is held


def test_contention_refuses_without_aska(repo):
    sl, rp = repo
    assert sl.acquire() is True
    # Simulate a second process by taking the flock in a child that sleeps.
    holder = subprocess.Popen(
        ["flock", str(rp / ".nx" / "session.lock.lock"), "-c", "sleep 5"])
    try:
        # wait until the child holds it
        import time
        for _ in range(50):
            r = subprocess.run(
                ["flock", "-n", str(rp / ".nx" / "session.lock.lock"), "true"],
                capture_output=True)
            if r.returncode != 0:
                break
            time.sleep(0.05)
        assert sl.acquire(ask_confirm=None) is False
        asked = []
        assert sl.acquire(ask_confirm=lambda m: asked.append(m) or False) is False
        assert asked and "another nx session" in asked[0]
        assert sl.acquire(ask_confirm=lambda m: True) is True
    finally:
        holder.terminate()
        holder.wait()


def test_release_clears(repo):
    sl, rp = repo
    assert sl.acquire() is True
    sl.release()
    assert not (rp / ".nx" / "session.lock").exists()
    r = subprocess.run(
        ["flock", "-n", str(rp / ".nx" / "session.lock.lock"), "true"],
        capture_output=True)
    assert r.returncode == 0  # lock released


# --- reacquire: the /resume-into-another-repo path (0.129.0) -----------------

def test_reacquire_noop_without_lock(repo):
    sl, rp = repo
    assert sl._HELD == []
    assert sl.reacquire() is True
    assert not (rp / ".nx" / "session.lock").exists()


def test_reacquire_after_chdir(repo, monkeypatch, tmp_path):
    sl, old = repo
    assert sl.acquire() is True
    new = tmp_path / "other-repo"
    new.mkdir()
    monkeypatch.chdir(new)
    import nx
    nx.st.PROJECT_ROOT = str(new)
    assert sl.reacquire() is True
    # The NEW repo now holds the lock, advertising our session.
    data = json.loads((new / ".nx" / "session.lock").read_text())
    assert data["session_id"] == nx.st.SESSION_ID
    r = subprocess.run(
        ["flock", "-n", str(new / ".nx" / "session.lock.lock"), "true"],
        capture_output=True)
    assert r.returncode != 0  # we hold it
    # The OLD repo's flock is gone (file deliberately left — see release).
    r = subprocess.run(
        ["flock", "-n", str(old / ".nx" / "session.lock.lock"), "true"],
        capture_output=True)
    assert r.returncode == 0


def test_reacquire_contended_declines(repo, tmp_path, monkeypatch):
    sl, old = repo
    assert sl.acquire() is True
    new = tmp_path / "busy-repo"
    (new / ".nx").mkdir(parents=True)  # the holder's lock file needs its dir
    monkeypatch.chdir(new)
    import nx
    nx.st.PROJECT_ROOT = str(new)
    holder = subprocess.Popen(
        ["flock", str(new / ".nx" / "session.lock.lock"), "-c", "sleep 5"])
    try:
        # wait until the child actually holds the flock
        import time
        for _ in range(50):
            r = subprocess.run(
                ["flock", "-n", str(new / ".nx" / "session.lock.lock"), "true"],
                capture_output=True)
            if r.returncode != 0:
                break
            time.sleep(0.05)
        # No way to ask → the new repo is declined (matching acquire's
        # contract: an advisory lock must never fence a session out silently
        # shared, and the old lock was already dropped by the switch).
        assert sl.reacquire(ask_confirm=None) is False
    finally:
        holder.terminate()
        holder.wait()
