#!/usr/bin/env python3
"""Tests for post-edit verification (/verify, NX_VERIFY_CMD).

Covers: the free in-process ast.parse appended to .py edit tool results (OK and
SYNTAX ERROR forms, non-.py paths untouched), the settle-log recording, the
detached NX_VERIFY_CMD launch after a settled edit batch (and no launch when
unsettled / unset / already in flight), and the pass/fail outcome staging into
the pending runtime-note slot.

No live model or terminal; _confirm is monkeypatched so writes don't block.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-verify-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ.pop("NX_VERIFY_CMD", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _verify as v  # noqa: E402
from nx import _dispatch as d  # noqa: E402


def _write(name, text):
    path = os.path.join(_tmp, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def _reset(verify_cmd=None):
    m._confirm = lambda action, preview=None: True
    v.st._VERIFY_EDIT_LOG.clear()
    v.st._VERIFY_INFLIGHT = False
    v.st._PENDING_VERIFY_NOTICE = None
    v.st._VERIFY_SEEN.clear()
    if verify_cmd:
        os.environ["NX_VERIFY_CMD"] = verify_cmd
    else:
        os.environ.pop("NX_VERIFY_CMD", None)


def test_ast_check_ok_and_syntax_error():
    _reset()
    assert v._ast_check(_write("ok.py", "def f():\n    return 1\n")) is None
    bad = _write("bad.py", "def f(:\n    pass\n")
    msg = v._ast_check(bad)
    assert msg is not None and msg.startswith("SYNTAX ERROR"), msg
    assert "bad.py" in msg and "line 1" in msg


def test_ast_check_syntax_error_includes_line_text():
    _reset()
    path = _write("bad2.py", "x = 1\ny = = 2\n")
    msg = v._ast_check(path)
    assert "line 2" in msg and "y = = 2" in msg


def test_edit_tool_result_carries_py_compile_verdict():
    _reset()
    path = _write("t_ok.py", "a = 1\n")
    res = m.run_tool("edit_file", {"path": path, "old": "a = 1", "new": "a = 2\nb = 3\n"})
    assert res.startswith("edited") and "py_compile OK" in res
    # the broken case: an edit that lands but leaves invalid python
    res2 = m.run_tool("edit_file", {"path": path, "old": "a = 2", "new": "a = )(\n"})
    assert res2.startswith("edited") and "SYNTAX ERROR" in res2
    assert "t_ok.py" in res2


def test_edit_tool_result_clean_for_non_py():
    _reset()
    path = _write("notes.txt", "alpha\nbeta\n")
    res = m.run_tool("edit_file", {"path": path, "old": "alpha", "new": "gamma"})
    assert res.startswith("edited") and "py_compile OK" not in res
    assert "SYNTAX ERROR" not in res


def test_record_py_edit_appends_settle_log():
    _reset()
    path = _write("log.py", "z = 0\n")
    assert v.st._VERIFY_EDIT_LOG == []
    v.record_py_edit(path)
    assert len(v.st._VERIFY_EDIT_LOG) == 1
    v.record_py_edit(path)
    assert len(v.st._VERIFY_EDIT_LOG) == 2


def _fake_dispatch_run_bash(calls):
    def fake_run_bash(command):
        calls.append(command)
        return "[background] pid 424242 — read_file(log) to poll"
    return fake_run_bash


def _stub_run_bash(calls):
    # _launch_verify resolves run_bash through DISPATCH (the same table
    # run_tool uses), so patch the table, not the module attribute.
    d.DISPATCH["run_bash"] = _fake_dispatch_run_bash(calls)


def _restore_run_bash():
    from nx._bash_jobs import run_bash as real
    d.DISPATCH["run_bash"] = real


def test_settled_batch_launches_verify_cmd():
    _reset(verify_cmd="uv run pytest -q")
    calls = []
    d.DISPATCH["run_bash"] = _fake_dispatch_run_bash(calls)  # type: ignore[assignment]
    try:
        v.st._VERIFY_EDIT_LOG.append(0.0)  # ancient timestamp → settled
        v._launch_verify_if_settled()
    finally:
        _restore_run_bash()
    assert calls == ["uv run pytest -q"]
    assert v.st._VERIFY_EDIT_LOG == []  # cleared so it can't double-fire
    assert v.st._VERIFY_INFLIGHT


def test_unsettled_batch_does_not_launch():
    _reset(verify_cmd="uv run pytest -q")
    calls = []
    d.DISPATCH["run_bash"] = _fake_dispatch_run_bash(calls)  # type: ignore[assignment]
    try:
        import time as _t
        v.st._VERIFY_EDIT_LOG.append(_t.monotonic())  # just now → not settled
        v._launch_verify_if_settled()
    finally:
        _restore_run_bash()
    assert calls == []
    assert v.st._VERIFY_EDIT_LOG  # untouched


def test_no_verify_cmd_no_launch():
    _reset()  # NX_VERIFY_CMD unset
    calls = []
    d.DISPATCH["run_bash"] = _fake_dispatch_run_bash(calls)  # type: ignore[assignment]
    try:
        v.st._VERIFY_EDIT_LOG.append(0.0)
        v._launch_verify_if_settled()
    finally:
        _restore_run_bash()
    assert calls == []


def test_inflight_never_double_launches():
    _reset(verify_cmd="uv run pytest -q")
    calls = []
    d.DISPATCH["run_bash"] = _fake_dispatch_run_bash(calls)  # type: ignore[assignment]
    try:
        v.st._VERIFY_EDIT_LOG.append(0.0)
        v.st._VERIFY_INFLIGHT = True
        v._launch_verify_if_settled()
    finally:
        _restore_run_bash()
    assert calls == []


def test_stage_verify_outcome_pass_and_fail():
    _reset(verify_cmd="uv run pytest -q")
    # Build fake bg-job entries the way list_bg_jobs would (this process owns
    # them; _read_bg_exit_status is stubbed so no real pid is probed).
    fake_jobs = [{"pid": "999001", "alive": False, "command": "uv run pytest -q",
                  "owner_pid": os.getpid(), "log_path": "/tmp/v.log",
                  "started": 1.0, "tail": "1 passed"}]
    import nx._bash_jobs as bj
    bj_list, bj_exit = bj.list_bg_jobs, bj._read_bg_exit_status
    bj.list_bg_jobs = lambda: fake_jobs
    try:
        bj._read_bg_exit_status = lambda pid: (True, 0, "")
        v._stage_verify_outcome()
        note = v.st._PENDING_VERIFY_NOTICE
        assert note and note.startswith("[Runtime note: NX_VERIFY_CMD"), note
        assert "passed" in note and "999001" not in note  # success is terse
        # once staged, a re-scan doesn't duplicate
        v._stage_verify_outcome()
        assert v.st._PENDING_VERIFY_NOTICE == note
        v.st._PENDING_VERIFY_NOTICE = None
        v.st._VERIFY_SEEN.clear()  # a re-run of the same command is a new verdict
        bj._read_bg_exit_status = lambda pid: (True, 2, "")
        v._stage_verify_outcome()
        fail_note = v.st._PENDING_VERIFY_NOTICE
        assert fail_note and "FAILED (exit 2)" in fail_note and "/tmp/v.log" in fail_note
    finally:
        bj.list_bg_jobs, bj._read_bg_exit_status = bj_list, bj_exit
    del fake_jobs[:]


def test_stage_verify_skips_other_commands_and_owners():
    _reset(verify_cmd="uv run pytest -q")
    import nx._bash_jobs as bj
    bj_list, bj_exit = bj.list_bg_jobs, bj._read_bg_exit_status
    try:
        bj.list_bg_jobs = lambda: [
            {"pid": "1", "alive": False, "command": "docker pull x", "owner_pid": os.getpid()},
            {"pid": "2", "alive": False, "command": "uv run pytest -q", "owner_pid": 12345},
            {"pid": "3", "alive": True, "command": "uv run pytest -q", "owner_pid": os.getpid()},
        ]
        bj._read_bg_exit_status = lambda pid: (True, 0, "")
        v._stage_verify_outcome()
        assert v.st._PENDING_VERIFY_NOTICE is None
        assert v.st._VERIFY_SEEN == set()
    finally:
        bj.list_bg_jobs, bj._read_bg_exit_status = bj_list, bj_exit


def test_flush_pending_verify_notice():
    _reset(verify_cmd="uv run pytest -q")
    v.st._PENDING_VERIFY_NOTICE = "[Runtime note: NX_VERIFY_CMD test]"
    msgs = [{"role": "user", "content": "hi"}]
    m._flush_pending_background_notice(msgs)
    assert msgs[-1]["content"] == "[Runtime note: NX_VERIFY_CMD test]"
    assert v.st._PENDING_VERIFY_NOTICE is None  # flushed exactly once
    msgs2 = [{"role": "user", "content": "hi"}]
    m._flush_pending_background_notice(msgs2)
    assert len(msgs2) == 1


def test_verify_enabled_reads_env_lazily():
    _reset()
    assert v.verify_enabled() is False
    os.environ["NX_VERIFY_CMD"] = "true"
    try:
        assert v.verify_enabled() is True
    finally:
        os.environ.pop("NX_VERIFY_CMD", None)
    assert v.verify_enabled() is False


def test_launch_verify_failure_is_not_fatal():
    _reset(verify_cmd="true")
    calls = []
    def failing_run_bash(command):
        calls.append(command)
        return "ERROR: no shell on PATH"
    d.DISPATCH["run_bash"] = failing_run_bash  # type: ignore[assignment]
    try:
        v.st._VERIFY_EDIT_LOG.append(0.0)
        v._launch_verify_if_settled()  # must not raise
    finally:
        _restore_run_bash()
    assert calls == ["true"]
    assert v.st._VERIFY_INFLIGHT  # settled log consumed either way


def test_run_tool_wraps_ast_check_failure_safely():
    _reset()
    path = _write("wrap.py", "q = 1\n")
    orig = v.record_py_edit
    def boom(p):
        raise RuntimeError("disk exploded")
    v.record_py_edit = boom  # type: ignore[assignment]
    try:
        res = m.run_tool("edit_file", {"path": path, "old": "q = 1", "new": "q = 2\n"})
    finally:
        v.record_py_edit = orig
    assert res.startswith("edited") and "SYNTAX ERROR" not in res and "py_compile OK" not in res
