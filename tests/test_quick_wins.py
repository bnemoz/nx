"""Quick-win regressions: the reasoning-stall nudge must not name benchmark
paths (/app/…), run_bash docs must match the setsid launcher (not the retired
nohup one), sleep-duration inference must cover fractional/suffixed sleeps,
/approval set-default persists to the env file like /theme, and
wait_background's schema must not advertise the ignored log_path arg.
"""
import importlib
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_APPROVAL", "NX_THEME",
          "NX_SOURCES", "NX_ACTIVE", "NX_HOME"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    import nx
    importlib.reload(nx)
    yield nx
    os.environ["NX_ENV_FILE"] = "/dev/null"
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_APPROVAL", "NX_THEME"):
        os.environ.pop(k, None)


# --- REASONING_STALL_NUDGE has no leaked benchmark paths ---------------------

def test_stall_nudge_no_benchmark_paths(nx_fresh):
    text = nx_fresh.REASONING_STALL_NUDGE
    assert "/app/" not in text
    assert "/app/solution.py" not in text
    # still nudges toward writing a file + running it
    assert "write_file" in text and "run_bash" in text


# --- run_bash / guidance docs describe session detachment, not nohup ---------

def test_run_bash_docstring_says_detached(nx_fresh):
    from nx import _bash_jobs
    doc = _bash_jobs.run_bash.__doc__ or ""
    # detached in its own session; the mechanism is start_new_session, and
    # macOS has no external setsid binary so it must not be the documented way
    assert "detached" in doc and "own session" in doc
    assert "run via nohup" not in doc
    assert "launched via nohup" not in doc


def test_background_guidance_says_detached(nx_fresh):
    assert "detached" in nx_fresh.BACKGROUND_GUIDANCE


# --- sleep inference covers fractional + suffixed durations ------------------

def test_sleep_inference_fractional_and_suffixed(nx_fresh):
    from nx._bash_jobs import _infer_timeout_from_sleep, _sleep_seconds
    f = _infer_timeout_from_sleep
    assert f("sleep 30 && cat x", 3) == 40
    # fractional sleeps used to fall through to the 3 s default and background
    assert f("sleep 0.5; ls", 3) == 10
    # suffixed durations (GNU sleep) used to fall through too
    assert f("sleep 2m", 3) == 130
    # over-1h waits clamp to the default NX_BASH_SLEEP_CAP (3600) + slack
    assert f("sleep 1h30m && echo done", 3) == 3610
    assert f("sleep 1h", 3) == 3610
    # trailing bare number is seconds (GNU semantics: 1h30 = 1h + 30s)
    assert f("sleep 1h30", 3) == 3610
    # unchanged behaviors
    assert f("ls && sleep 5", 3) == 15
    assert f("echo x", 3) == 3
    assert f("sleep 0", 3) == 3      # zero → not a wait worth extending
    assert f("sleep $N", 3) == 3     # unknowable → default
    assert f("sleep .5", 3) == 3     # no leading digit → not sleep's grammar
    # non-duration junk still falls back (2M=2 megabytes is not a sleep unit)
    assert f("sleep 2M", 3) == 3
    assert f("sleep 1x", 3) == 3
    # newline and subshell boundaries used to fall through to the 3 s default;
    # quoted sleeps are not boundaries (just an echo argument) and must not match
    boundary_cmds = {
        "cmd\nsleep 30 && x": 40,
        "(sleep 30 && x)": 40,
        "x\n\nsleep 30": 40,        # blank line still counts
        "echo 'sleep 30'": 3,
    }
    for cmd, want in boundary_cmds.items():
        assert f(cmd, 3) == want, cmd


def test_sleep_inference_cap(nx_fresh, monkeypatch):
    from nx._bash_jobs import _infer_timeout_from_sleep as f
    monkeypatch.setenv("NX_BASH_SLEEP_CAP", "120")
    # sleeps under the cap keep the old inferred window
    assert f("sleep 30 && cat x", 3) == 40
    # at the cap: cap + slack
    assert f("sleep 5m", 3) == 130
    # beyond the cap the turn no longer blocks for the full wait
    for cmd in ("sleep 1h", "sleep 1d", "sleep 1h30m"):
        assert f(cmd, 3) == 130
    # cap = 0 disables the ceiling (old unbounded behavior)
    monkeypatch.setenv("NX_BASH_SLEEP_CAP", "0")
    assert f("sleep 1d", 3) == 86410


def test_sleep_seconds_token_parser(nx_fresh):
    from nx._bash_jobs import _sleep_seconds
    assert _sleep_seconds("30") == 30.0
    assert _sleep_seconds("0.5") == 0.5
    assert _sleep_seconds("2m") == 120.0
    assert _sleep_seconds("1h30m") == 5400.0
    assert _sleep_seconds("1h30") == 3630.0
    assert _sleep_seconds("1d") == 86400.0
    # junk → 0.0 (caller falls back to the default poll window)
    assert _sleep_seconds("2M") == 0.0
    assert _sleep_seconds("1x") == 0.0
    assert _sleep_seconds("abc") == 0.0
    assert _sleep_seconds("") == 0.0
    # a real `sleep 2m` command must get a poll window ≥ the wait + slack
    from nx._bash_jobs import _infer_timeout_from_sleep
    assert _infer_timeout_from_sleep("sleep 2m && echo ok", 3) >= 120


# --- /approval set-default persists to the env file (parity with /theme) -----

def test_approval_set_default_persists(nx_fresh, capsys):
    from nx._repl_settings_cmds import _cmd_approval
    from nx._repl_state import _ReplState
    env_file = os.environ["NX_ENV_FILE"]
    state = _ReplState([], "t", False, [])
    try:
        _cmd_approval("/approval set-default low", state)
        out = capsys.readouterr().out
        assert "saved to ~/.env" in out
        assert os.environ["NX_APPROVAL"] == "low"
        with open(env_file, encoding="utf-8") as fh:
            body = fh.read()
        assert "NX_APPROVAL=low" in body
        # yolo persists as the literal "yolo"
        _cmd_approval("/approval set-default yolo", state)
        capsys.readouterr()
        with open(env_file, encoding="utf-8") as fh:
            assert "NX_APPROVAL=yolo" in fh.read()
        # set-default with no arg persists the current display mode
        _cmd_approval("/approval medium", state)
        capsys.readouterr()
        assert nx_fresh.APPROVE_LEVEL == "medium"
        _cmd_approval("/approval set-default", state)
        capsys.readouterr()
        with open(env_file, encoding="utf-8") as fh:
            assert "NX_APPROVAL=medium" in fh.read()
    finally:
        # don't leak the approval default into later tests in this process
        nx_fresh.DEFAULT_APPROVE_LEVEL = None
        nx_fresh.APPROVE_LEVEL = None
        nx_fresh.YOLO = False


def test_approval_bare_level_stays_session_scoped(nx_fresh, capsys):
    from nx._repl_settings_cmds import _cmd_approval
    from nx._repl_state import _ReplState
    env_file = os.environ["NX_ENV_FILE"]
    state = _ReplState([], "t", False, [])
    try:
        _cmd_approval("/approval medium", state)
        capsys.readouterr()
        assert nx_fresh.APPROVE_LEVEL == "medium"
        # session-scoped: nothing written to the env file
        assert not os.path.exists(env_file)
    finally:
        nx_fresh.DEFAULT_APPROVE_LEVEL = None
        nx_fresh.APPROVE_LEVEL = None
        nx_fresh.YOLO = False


# --- wait_background schema no longer advertises log_path --------------------

def test_wait_background_schema_drops_log_path(nx_fresh):
    props = nx_fresh._WAIT_BACKGROUND_TOOL["function"]["parameters"]["properties"]
    assert "log_path" not in props
    assert set(props) == {"pid", "timeout"}
    assert nx_fresh._WAIT_BACKGROUND_TOOL in nx_fresh.TOOLS
