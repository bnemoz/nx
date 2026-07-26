#!/usr/bin/env python3
"""Tests for `edit_file`, the single find-and-replace tool.

No live model or terminal. Exercises edit_file directly and monkeypatches
_confirm so writes don't block on the approval prompt. Covers the no-op
short-circuit (old == new never prompts or writes) and the ambiguous/absent
match error.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-editfile-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def _write(name, text):
    path = os.path.join(_tmp, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def _allow_edits():
    m._confirm = lambda action, preview=None: True


def _deny_edits():
    m._confirm = lambda action, preview=None: False


def test_no_op_edit_errors_without_prompting():
    saved_confirm = m._confirm
    calls = []
    m._confirm = lambda action, preview=None: calls.append(1) or True
    try:
        path = _write("noop.py", "def add(a, b):\n    return a + b\n")
        res = m.edit_file(path, "def add(a, b):\n    return a + b\n",
                           "def add(a, b):\n    return a + b\n")
        assert res.startswith("ERROR"), res
        assert not calls, "no-op edit should not prompt for confirmation"
        with open(path) as f:
            assert f.read() == "def add(a, b):\n    return a + b\n"
    finally:
        m._confirm = saved_confirm


def test_real_edit_still_prompts_and_writes():
    saved_confirm = m._confirm
    try:
        _allow_edits()
        path = _write("real.py", "def add(a, b):\n    return a + b\n")
        res = m.edit_file(path, "return a + b", "return a + b  # commented differently")
        assert res.startswith("edited"), res
        with open(path) as f:
            assert "commented differently" in f.read()
    finally:
        m._confirm = saved_confirm


def test_ambiguous_match_errors():
    path = _write("dup.py", "x = 1\nx = 1\n")
    res = m.edit_file(path, "x = 1", "x = 2")
    assert res.startswith("ERROR"), res
    assert "matched 2 times" in res


if __name__ == "__main__":
    test_no_op_edit_errors_without_prompting()
    test_real_edit_still_prompts_and_writes()
    test_ambiguous_match_errors()
    print("all tests passed")
