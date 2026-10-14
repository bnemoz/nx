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


# --- diff mode (the consolidated apply_patch engine, 0.158.11) -----------------

def _diff(a="a", b="b"):
    return (f"--- {os.path.join(_tmp, 'dm.py')}\n"
            f"+++ b/dm.py\n@@ -1 +1 @@\n-{a}\n+{b}\n")


def test_diff_mode_applies_through_edit_file(tmp_path, monkeypatch):
    saved_confirm = m._confirm
    try:
        _allow_edits()
        (tmp_path / "dm.py").write_text("a\n")
        monkeypatch.chdir(tmp_path)
        res = m.edit_file(path=str(tmp_path), diff=_diff())
        assert "applied patch" in res, res
        assert (tmp_path / "dm.py").read_text() == "b\n"
    finally:
        m._confirm = saved_confirm


def test_diff_mode_refuses_mixed_modes():
    res = m.edit_file("x.py", old="a", new="b", diff="--- a/x\n+++ b/x\n")
    assert res.startswith("ERROR") and "not both" in res


def test_diff_mode_denial_writes_nothing(tmp_path, monkeypatch):
    saved_confirm = m._confirm
    try:
        _deny_edits()
        (tmp_path / "dm.py").write_text("a\n")
        monkeypatch.chdir(tmp_path)
        res = m.edit_file(path=str(tmp_path), diff=_diff())
        assert res.startswith("DENIED"), res
        assert (tmp_path / "dm.py").read_text() == "a\n"
    finally:
        m._confirm = saved_confirm


def test_diff_mode_staleness_guard_refuses(tmp_path, monkeypatch):
    """Diff mode crosses the same cross-session staleness guard the literal
    mode uses: a file read this session and changed underneath refuses the
    whole patch before the dry-run/approval flow (0.158.11 guard routing)."""
    from nx import _staleness
    saved_confirm = m._confirm
    try:
        _allow_edits()
        f = tmp_path / "dm.py"
        f.write_text("a\n")
        _staleness.record_read(str(f))          # anchor at the current mtime
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # newer
        monkeypatch.chdir(tmp_path)
        res = m.edit_file(path=str(tmp_path), diff=_diff())
        assert res.startswith("ERROR"), res
        assert "changed on disk since this session last read it" in res
        assert (tmp_path / "dm.py").read_text() == "a\n"
    finally:
        m._confirm = saved_confirm


def test_apply_patch_name_no_longer_dispatches():
    from nx._dispatch import DISPATCH
    assert "apply_patch" not in DISPATCH
    assert "apply_patch" not in {t["function"]["name"] for t in m.TOOLS}
    # the engine itself stays, as edit_file's diff-mode implementation
    from nx._edit_tools import apply_patch  # noqa: F401


if __name__ == "__main__":
    test_no_op_edit_errors_without_prompting()
    test_real_edit_still_prompts_and_writes()
    test_ambiguous_match_errors()
    print("all tests passed")
