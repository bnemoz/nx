#!/usr/bin/env python3
"""Tests for `_atomic_write` metadata preservation.

The shared write path behind write_file/edit_file/multi_edit/sed_edit must
(1) keep the existing file's permission bits (mkstemp creates 0600, which
used to silently strip the executable bit off scripts on every edit) and
(2) follow symlinks — rewriting the link itself used to replace it with a
regular file, breaking symlink-farm dotfile management.

No live model or terminal; _confirm is bypassed by patching nx._confirm.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-atomicwrite-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def _path(name):
    return os.path.join(_tmp, name)


def _allow():
    orig = m._confirm
    def yes(action, preview=None):
        return True
    m._confirm = yes
    return orig


def test_mode_preserved_on_edit():
    saved = _allow()
    try:
        _mode_preserved_on_edit()
    finally:
        m._confirm = saved


def _mode_preserved_on_edit():
    path = _path("hook.sh")
    with open(path, "w") as f:
        f.write("echo hi\n")
    os.chmod(path, 0o755)
    res = m.edit_file(path, "hi", "bye")
    assert res.startswith("edited"), res
    assert (os.stat(path).st_mode & 0o7777) == 0o755, "executable bit stripped"
    with open(path) as f:
        assert f.read() == "echo bye\n"


def test_symlink_followed_not_replaced():
    saved = _allow()
    try:
        _symlink_followed_not_replaced()
    finally:
        m._confirm = saved


def _symlink_followed_not_replaced():
    target = _path("real-config.toml")
    with open(target, "w") as f:
        f.write("orig\n")
    link = _path("linked-config.toml")
    os.symlink(target, link)
    res = m.write_file(link, "new\n")
    assert res.startswith("wrote"), res
    assert os.path.islink(link), "symlink was replaced by a regular file"
    with open(target) as f:
        assert f.read() == "new\n", "write did not go through to the target"
    with open(link) as f:
        assert f.read() == "new\n"


def test_new_file_still_writable():
    saved = _allow()
    try:
        _new_file_still_writable()
    finally:
        m._confirm = saved


def _new_file_still_writable():
    path = _path("brand-new.txt")
    res = m.write_file(path, "hello\n")
    assert res.startswith("wrote"), res
    with open(path) as f:
        assert f.read() == "hello\n"


if __name__ == "__main__":
    test_mode_preserved_on_edit()
    test_symlink_followed_not_replaced()
    test_new_file_still_writable()
    print("all tests passed")
