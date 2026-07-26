#!/usr/bin/env python3
"""Tests for `multi_edit`, the batch atomic edit tool.

No live model or terminal. Exercises multi_edit directly and monkeypatches
_confirm so writes don't block on the approval prompt. Covers the atomicity
guarantee (all-or-nothing), ordered application, line-number tolerance, the
no-op short-circuit, and that a single failed match leaves the file untouched.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-multiedit-")
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


def test_applies_multiple_edits_in_order():
    _allow_edits()
    path = _write("me1.py", "def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    res = m.multi_edit(path, [
        {"old": "return 1", "new": "return 10"},
        {"old": "return 2", "new": "return 20"},
    ])
    assert res.startswith(f"edited {path} (2 edits applied)")
    with open(path) as f:
        src = f.read()
    assert "return 10" in src and "return 20" in src
    assert "return 1\n" not in src and "return 2\n" not in src


def test_later_edit_targets_text_from_earlier_edit():
    _allow_edits()
    path = _write("me2.txt", "alpha\n")
    res = m.multi_edit(path, [
        {"old": "alpha", "new": "alpha-beta"},
        {"old": "alpha-beta", "new": "gamma"},
    ])
    assert res.startswith(f"edited {path} (2 edits applied)")
    with open(path) as f:
        assert f.read() == "gamma\n"


def test_failed_match_rolls_back_atomically():
    _allow_edits()
    path = _write("me3.py", "def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    res = m.multi_edit(path, [
        {"old": "return 1", "new": "return 10"},
        {"old": "NONEXISTENT", "new": "x"},
    ])
    assert res.startswith("ERROR")
    assert "edit #2" in res
    # the file must be untouched — edit #1 was rolled back
    with open(path) as f:
        src = f.read()
    assert "return 1\n" in src, "atomic rollback must restore original content"
    assert "return 10" not in src


def test_ambiguous_old_rolls_back():
    _allow_edits()
    path = _write("me4.txt", "dup\ndup\n")
    res = m.multi_edit(path, [{"old": "dup", "new": "x"}])
    assert res.startswith("ERROR")
    assert "matched 2" in res
    with open(path) as f:
        assert f.read() == "dup\ndup\n"


def test_strips_pasted_line_numbers():
    _allow_edits()
    path = _write("me5.py", "def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    win = m.read_file(path, offset=1, limit=2)
    old_block = win.split("\n", 1)[1].rstrip("\n")  # numbered
    new_block = "     1\tdef foo():\n     2\t    return 100"
    res = m.multi_edit(path, [{"old": old_block, "new": new_block}])
    assert res.startswith("edited")
    with open(path) as f:
        src = f.read()
    assert "return 100" in src
    assert "\t" not in src, "line-number prefixes must not leak into the file"
    assert "return 2" in src, "untouched section remains"


def test_no_op_edits_do_not_write_or_prompt():
    calls = {"n": 0}
    m._confirm = lambda action, preview=None: calls.__setitem__("n", calls["n"] + 1) or True
    path = _write("me6.txt", "hello\n")
    res = m.multi_edit(path, [{"old": "hello", "new": "hello"}])
    assert "no change" in res
    assert calls["n"] == 0, "a no-op batch must not prompt for approval"


def test_denied_approval_leaves_file_unchanged():
    _deny_edits()
    path = _write("me7.txt", "a\nb\n")
    res = m.multi_edit(path, [
        {"old": "a", "new": "X"},
        {"old": "b", "new": "Y"},
    ])
    assert res == "DENIED by user"
    with open(path) as f:
        assert f.read() == "a\nb\n"


def test_empty_edits_is_error():
    _allow_edits()
    path = _write("me8.txt", "x\n")
    assert m.multi_edit(path, []).startswith("ERROR")
    assert m.multi_edit(path, "not a list").startswith("ERROR")


def test_malformed_edit_object_is_error():
    _allow_edits()
    path = _write("me9.txt", "x\n")
    assert m.multi_edit(path, [{"old": "x"}]).startswith("ERROR")
    assert m.multi_edit(path, [{"old": 1, "new": 2}]).startswith("ERROR")
    assert m.multi_edit(path, ["not a dict"]).startswith("ERROR")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
