"""Background PID→log resolution must never guess.

Regression tests for the wrong-log bug: the setsid launcher's stdout is
validated before a ``bg-pid-*.map`` file is written (no ``bg-pid-.map`` /
``bg-pid-<garbage>.map`` for an empty or invalid PID), the 1KB
content-scan fallback in ``_find_bg_log_for_pid`` is gone (a PID substring
in some log's head must not resolve a model-supplied pid to *another*
job's log), and ``wait_background`` answers a non-numeric pid with an
``[error]`` result instead of an unhandled ValueError.
"""
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
    from nx import _state as st
    # Tier A: no importlib.reload(nx) here. The reload rebuilt two things from
    # this fixture's env, and both are now rebuilt explicitly:
    #  1. the model source (st.client / st.MODEL) — re-discover sources from the
    #     env we just set and switch to the first one. resolve_model() reads
    #     NX_MODEL ("test-model"), so no network round-trip happens; the dead
    #     127.0.0.1:9 endpoint is never contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    #  2. the state home: every bg-logs path is derived from st.SESSION_HOME at
    #     call time, so pin it at this test's NX_HOME. Without the reload the
    #     attr keeps conftest's per-worker home, which every test in the worker
    #     shares — and since owner_pid filtering compares against the shared
    #     worker pid, a previous file's maps would show up as "mine". Pinning
    #     gives each test a private bg-logs dir (both spellings of the attr;
    #     production reads the _state one, some callers the top-level one).
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path / "nxhome"), raising=False)
    monkeypatch.setattr(nx, "SESSION_HOME", str(tmp_path / "nxhome"), raising=False)
    monkeypatch.setattr(nx, "YOLO", True, raising=False)
    # Background-job notice state the module-dict rewind used to wipe (and this
    # fixture's teardown used to reset by hand): pin fresh values per test so a
    # previous test's staged notice / scan latch cannot leak in. monkeypatch
    # auto-restores them, so no manual teardown is needed.
    monkeypatch.setattr(st, "_SESSION_HAS_BACKGROUNDED", False, raising=False)
    monkeypatch.setattr(st, "_PENDING_BACKGROUND_NOTICE", None, raising=False)
    monkeypatch.setattr(st, "_BG_SCAN_DONE", False, raising=False)
    monkeypatch.setattr(st, "_PENDING_BG_COMPLETION_NOTICE", None, raising=False)
    monkeypatch.setattr(st, "_BG_NOTICE_SEEN", None, raising=False)
    monkeypatch.setattr(st, "_PENDING_SUBAGENT_NOTICE", None, raising=False)
    yield nx, tmp_path
    # env rewind is universal (conftest), so no manual pop here; nx globals a
    # test mutates go through monkeypatch (per-test, auto-restored).


def _bg_dir(nx):
    # per-project slug subdir (0.145.2); the same dir _run_bash writes into
    from nx import _bash_jobs as bj
    d = bj._bg_dir()
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
    assert data["cwd"] == os.getcwd()  # the launch dir, for the /bg all view


# --- list_bg_jobs is scoped to THIS process (bg-logs is machine-shared) -------

def _write_map(bg_dir, pid, *, owner, command="sleep 600", started=None,
               with_cwd=True):
    """Forge a bg-pid-<pid>.map the way _run_bash writes one."""
    log = os.path.join(bg_dir, f"bg-{pid}.log")
    with open(log, "w") as f:
        f.write("fake output\n")
    data = {"log_path": log, "command": command,
            "started": started if started is not None else 1.0}
    if owner is not None:
        data["owner_pid"] = owner
        if with_cwd:
            data["cwd"] = "/tmp/foreign-project"
    with open(os.path.join(bg_dir, f"bg-pid-{pid}.map"), "w") as f:
        json.dump(data, f)
    return log


def test_list_bg_jobs_hides_foreign_owner_by_default(nx_fresh):
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    _write_map(bg, 41001, owner=os.getpid() + 987654)   # another nx process
    mine = _write_map(bg, 41002, owner=os.getpid())     # this process
    pids = [j["pid"] for j in nx.list_bg_jobs()]
    assert pids == ["41002"]
    assert nx.list_bg_jobs()[0]["log_path"] == mine


def test_list_bg_jobs_mine_only_false_shows_everything(nx_fresh):
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    _write_map(bg, 41001, owner=os.getpid() + 987654)
    _write_map(bg, 41002, owner=os.getpid())
    pids = {j["pid"] for j in nx.list_bg_jobs(mine_only=False)}
    assert pids == {"41001", "41002"}


def test_list_bg_jobs_legacy_map_without_owner_is_foreign(nx_fresh):
    """Maps written before owner_pid existed have no owner to match — with
    mine_only they are treated as foreign (deliberately no liveness
    fallback: the filter is exactly process-based)."""
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    _write_map(bg, 41003, owner=None)
    assert nx.list_bg_jobs() == []
    legacy = nx.list_bg_jobs(mine_only=False)
    assert [j["pid"] for j in legacy] == ["41003"]
    assert legacy[0]["owner_pid"] is None


def test_bg_kill_refuses_foreign_pid_and_points_at_bg_all(nx_fresh, capsys):
    from nx._repl_settings_cmds import _cmd_bg
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    _write_map(bg, 41004, owner=os.getpid() + 987654)
    _cmd_bg("/bg kill 41004", None)
    out = capsys.readouterr().out
    assert "another session" in out and "/bg all" in out


def test_bg_all_lists_foreign_jobs_labelled(nx_fresh, capsys):
    from nx._repl_settings_cmds import _cmd_bg
    nx, _ = nx_fresh
    bg = _bg_dir(nx)
    _write_map(bg, 41005, owner=os.getpid() + 987654, command="other-project-build")
    _write_map(bg, 41006, owner=os.getpid(), command="my-build")
    _cmd_bg("/bg", None)
    out = capsys.readouterr().out
    assert "my-build" in out and "41006" in out
    assert "other-project-build" not in out  # default view is session-scoped
    _cmd_bg("/bg all", None)
    out = capsys.readouterr().out
    assert "other-project-build" in out and "41005" in out and "my-build" in out
    assert "EVERY nx session" in out  # says so in the output
    assert "/tmp/foreign-project" in out  # cwd shown for foreign jobs


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
