#!/usr/bin/env python3
"""Private-sink guarantees (roadmap 1.6, 0.40): everything nx writes under
NX_HOME is created 0600/0700 regardless of the process umask.

Two hostile-umask mechanisms are exercised for real, because chmod-after-write
does not save you there:

- a pre-existing group-writable DIRECTORY is re-chmodded by private_dir()
- a file that a SHELL redirection creates (bg logs) must already exist at 0600
  before the redirection opens it — the shell honors the umask, not us
"""
import json
import os
import stat
import sys
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _bash_jobs as bj  # noqa: E402
from nx import _private as pv  # noqa: E402


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _hostile_dir(path):
    """Create a directory a pre-0.40 nx (or a stray mkdir) may have left
    group/world-writable."""
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o770)
    return path


# --- the helpers ---------------------------------------------------------------

def test_private_dir_repairs_existing_group_writable_dir():
    d = os.path.join(_tmp, "repair-me")
    os.chmod(d, 0o755) if os.path.isdir(d) else None
    os.makedirs(d, exist_ok=True)
    os.chmod(d, 0o775)
    pv.private_dir(d)
    assert _mode(d) == 0o700


def test_open_private_creates_0600_under_hostile_umask():
    old = os.umask(0o077)  # would normally yield 0666 on plain open()
    try:
        p = os.path.join(_tmp, "openme", "f.txt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with pv.open_private(p) as f:
            f.write("x")
        assert _mode(p) == 0o600
    finally:
        os.umask(old)


def test_touch_private_defeats_shell_redirection_umask():
    old = os.umask(0o007)  # shell `>` would create the file 0660
    try:
        p = pv.touch_private(os.path.join(_tmp, "bg", "precreated.log"))
        assert _mode(p) == 0o600
        # the redirection itself now appends into the pre-created file
        with open(p, "a") as f:
            f.write("shell wrote this\n")
        assert _mode(p) == 0o600, "redirection must not re-create or re-mode the log"
        assert "shell wrote this" in open(p).read()
    finally:
        os.umask(old)


def test_atomic_write_private_is_private_and_leaves_no_tmp():
    p = os.path.join(_tmp, "atom", "data.json")
    pv.atomic_write_private(p, json.dumps({"k": "v"}))
    assert _mode(p) == 0o600
    assert json.load(open(p)) == {"k": "v"}
    assert not [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")]


# --- the real sinks ------------------------------------------------------------

def test_bg_log_and_pid_map_are_private():
    pid, log_path = bj._run_bash("echo private-sink-4411")
    assert log_path and os.path.exists(log_path)
    assert _mode(log_path) == 0o600, "bg log must be 0600 even under a lax umask"
    d = os.path.dirname(log_path)
    assert _mode(d) == 0o700
    map_path = os.path.join(d, f"bg-pid-{pid}.map")
    assert os.path.exists(map_path)
    assert _mode(map_path) == 0o600


def test_overflow_spill_is_private_and_atomic(monkeypatch):
    monkeypatch.setattr(m._text_fallback, "TOOL_RESULT_CHARS", 200)
    out = m._sanitize_tool_result("line " + "secret" * 300)  # > cap -> spills
    spill_path = out.rsplit("saved to: ", 1)[1].split(" —", 1)[0].strip()
    assert os.path.exists(spill_path)
    assert _mode(spill_path) == 0o600
    assert _mode(os.path.dirname(spill_path)) == 0o700
    assert not [f for f in os.listdir(os.path.dirname(spill_path))
                if f.endswith(".tmp")]


def test_subagent_report_is_private():
    from nx import _subagent as sa
    path = sa._write_agent_report("job-priv-4412", "instruction", "done", "answer")
    assert path and os.path.exists(path)
    assert _mode(path) == 0o600
    assert _mode(os.path.dirname(path)) == 0o700


def test_allowed_json_is_private():
    from nx import _allowed as al
    old = os.environ.get("NX_ALLOWED_FILE")
    os.environ["NX_ALLOWED_FILE"] = os.path.join(_tmp, "allowed", "allowed.json")
    try:
        al.add_rule("run_bash", "run: pytest*", scope="project")
        path = al._allowed_path()
        assert os.path.exists(path)
        assert _mode(path) == 0o600
        assert _mode(os.path.dirname(path)) == 0o700
    finally:
        if old is None:
            os.environ.pop("NX_ALLOWED_FILE", None)
        else:
            os.environ["NX_ALLOWED_FILE"] = old


def test_memory_and_history_sinks_stay_private():
    # memory write path goes through private_dir + atomic_write_private now
    from nx import _memory_cmd as mc
    note = mc._write_memory_file("Priv Sink", "body", scope="global", gate=False)
    assert note, "memory write should succeed"
    note_path = os.path.abspath(note[0])
    assert _mode(note_path) == 0o600
    assert _mode(os.path.dirname(note_path)) == 0o700

    from nx import _input_history as ih
    hist = os.path.join(_tmp, "hist", "history.json")
    os.environ["NX_INPUT_HISTORY_FILE"] = hist
    try:
        ih._append_to_file("some input")
        assert _mode(hist) == 0o600
        assert _mode(os.path.dirname(hist)) == 0o700
    finally:
        os.environ.pop("NX_INPUT_HISTORY_FILE", None)


def test_session_file_and_index_are_private():
    from nx import _sessions_fs as sf
    sid = sf._new_session_id() if hasattr(sf, "_new_session_id") else "s-priv-4413"
    sf._write_session(sid, [{"role": "user", "content": "hi"}])
    assert _mode(sf._session_path(sid)) == 0o600
    sf._write_index_entry(sid, {"title": "t", "mtime": 0})
    assert _mode(sf._index_entry_path(sid)) == 0o600
    assert _mode(sf._index_dir()) == 0o700
