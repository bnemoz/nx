#!/usr/bin/env python3
"""Tests for `sed_edit`, the regex find-and-replace tool.

No live model or terminal. Exercises sed_edit directly and monkeypatches
_confirm so writes don't block on the approval prompt. Covers single-file
and glob replacement, the no-match no-op, denied approval, and invalid regex.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-sededit-")
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


def test_single_file_replace():
    _allow_edits()
    path = _write("se1.py", "foo = 1\nfoo = 2\n")
    res = m.sed_edit(r"foo", "bar", path)
    assert res.startswith("edited 1 file(s), 2 replacement(s)")
    with open(path) as f:
        assert f.read() == "bar = 1\nbar = 2\n"


def test_glob_replace_across_files():
    _allow_edits()
    p1 = _write("se2a.py", "foo\n")
    p2 = _write("se2b.py", "foo\n")
    res = m.sed_edit(r"foo", "bar", os.path.join(_tmp, "se2*.py"))
    assert "2 file(s)" in res and "2 replacement(s)" in res
    with open(p1) as f:
        assert f.read() == "bar\n"
    with open(p2) as f:
        assert f.read() == "bar\n"


def test_count_caps_replacements_per_file():
    _allow_edits()
    path = _write("se3.txt", "x x x\n")
    res = m.sed_edit(r"x", "y", path, count=2)
    assert "2 replacement(s)" in res
    with open(path) as f:
        assert f.read() == "y y x\n"


def test_flags_ignorecase():
    _allow_edits()
    path = _write("se4.txt", "Foo foo FOO\n")
    res = m.sed_edit(r"foo", "bar", path, flags="i")
    assert "3 replacement(s)" in res
    with open(path) as f:
        assert f.read() == "bar bar bar\n"


def test_no_match_is_noop_and_does_not_prompt():
    calls = {"n": 0}
    m._confirm = lambda action, preview=None: calls.__setitem__("n", calls["n"] + 1) or True
    path = _write("se5.txt", "hello\n")
    res = m.sed_edit(r"nomatch", "x", path)
    assert "no matches" in res
    assert calls["n"] == 0


def test_denied_approval_leaves_file_unchanged():
    _deny_edits()
    path = _write("se6.txt", "a b\n")
    res = m.sed_edit(r"a", "X", path)
    assert res == "DENIED by user"
    with open(path) as f:
        assert f.read() == "a b\n"


def test_invalid_regex_is_error():
    _allow_edits()
    path = _write("se7.txt", "x\n")
    res = m.sed_edit(r"(", "x", path)
    assert res.startswith("ERROR: invalid regex")


def test_no_glob_match_is_error():
    _allow_edits()
    res = m.sed_edit(r"x", "y", os.path.join(_tmp, "nonexistent*.zzz"))
    assert res.startswith("ERROR: no files matched")


def test_replacement_with_backslash_is_literal_not_re_escape():
    _allow_edits()
    path = _write("se8.txt", "foo\n")
    res = m.sed_edit(r"foo", r"C:\Users\1new", path)
    assert res.startswith("edited 1 file(s), 1 replacement(s)")
    with open(path) as f:
        assert f.read() == "C:\\Users\\1new\n"


def test_replacement_with_backreference_syntax_is_literal():
    _allow_edits()
    path = _write("se9.txt", "foo\n")
    res = m.sed_edit(r"foo", r"\1 literal", path)
    assert res.startswith("edited 1 file(s), 1 replacement(s)")
    with open(path) as f:
        assert f.read() == "\\1 literal\n"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
