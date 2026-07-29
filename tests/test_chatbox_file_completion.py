#!/usr/bin/env python3
"""Regression coverage for on-demand @file completion."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-chatbox-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _chatbox, _risk  # noqa: E402


def test_completion_reads_the_typed_deep_directory():
    root = os.path.join(_tmp, "project")
    nested = os.path.join(root, "one", "two", "three", "four")
    os.makedirs(nested)
    with open(os.path.join(nested, "target.py"), "w") as f:
        f.write("# target\n")

    original = _risk._detect_project_root
    _risk._detect_project_root = lambda cwd=None: root
    try:
        assert _chatbox._match_file_paths("one/two/three/four/tar") == [
            "one/two/three/four/target.py"
        ]
    finally:
        _risk._detect_project_root = original


def test_completion_does_not_include_skipped_directories():
    root = os.path.join(_tmp, "skip-project")
    os.makedirs(os.path.join(root, "node_modules"))
    os.makedirs(os.path.join(root, "src"))

    original = _risk._detect_project_root
    _risk._detect_project_root = lambda cwd=None: root
    try:
        assert _chatbox._match_file_paths("") == ["src/"]
    finally:
        _risk._detect_project_root = original


def test_accepting_a_directory_keeps_the_mention_token_open():
    assert _chatbox._accept_file_completion("@sr", 3, 0, "src/") == ("@src/", 5)
    assert _chatbox._accept_file_completion("@fi", 3, 0, "file.py") == ("@file.py ", 9)


def test_raw_readers_raise_eoferror_for_a_closed_input_fd():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        try:
            _chatbox._raw_read_key(read_fd)
        except EOFError:
            pass
        else:
            raise AssertionError("_raw_read_key() did not raise EOFError")

        try:
            _chatbox._raw_read_available(read_fd)
        except EOFError:
            pass
        else:
            raise AssertionError("_raw_read_available() did not raise EOFError")
    finally:
        os.close(read_fd)
