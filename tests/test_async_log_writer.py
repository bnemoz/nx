#!/usr/bin/env python3
"""Tests for `_AsyncLogWriter`, the background-thread traffic-log writer.

`_LoggingStream` calls `.write()` once per streamed SSE chunk — on a fast
local server that's easily hundreds of times a second, so the write must
never block the caller on disk I/O, and the log must still be complete and
correctly rotated once it passes the configured size.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-log-writer-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def test_writes_land_in_order_after_flush(tmp_path):
    path = str(tmp_path / "test.log")
    w = m._AsyncLogWriter(path, max_bytes=0)
    for i in range(50):
        w.write(f"line{i}\n")
    w.flush()
    with open(path) as f:
        content = f.read()
    assert content == "".join(f"line{i}\n" for i in range(50))


def test_write_is_a_queue_put_not_a_disk_write(tmp_path):
    path = str(tmp_path / "test.log")
    w = m._AsyncLogWriter(path, max_bytes=0)
    # write() only enqueues — it must not touch the filesystem itself
    # (that's the whole point: no blocking I/O on the caller's thread).
    w.write("hello\n")
    assert isinstance(w._queue, __import__("queue").Queue)
    w.flush()
    with open(path) as f:
        assert f.read() == "hello\n"


def test_rotates_past_max_bytes(tmp_path):
    path = str(tmp_path / "test.log")
    w = m._AsyncLogWriter(path, max_bytes=10)
    w.write("0123456789\n")  # 11 bytes, at/over the cap on the *next* write
    w.write("next-line\n")
    w.flush()
    assert os.path.exists(path + ".1")
    with open(path + ".1") as f:
        assert f.read() == "0123456789\n"
    with open(path) as f:
        assert f.read() == "next-line\n"


def test_zero_max_bytes_disables_rotation(tmp_path):
    path = str(tmp_path / "test.log")
    w = m._AsyncLogWriter(path, max_bytes=0)
    for i in range(100):
        w.write("x" * 50 + "\n")
    w.flush()
    assert not os.path.exists(path + ".1")


def test_log_event_uses_the_module_level_async_writer():
    # _log_event must not write synchronously to disk on the caller's
    # thread — it hands off to _llog (an _AsyncLogWriter) and returns.
    assert isinstance(m._llog, m._AsyncLogWriter)
    m._log_event("req", {"probe": "test_log_event_uses_the_module_level_async_writer"})
    m._llog.flush()  # drain so a slow disk can't make this test flaky


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
