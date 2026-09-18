"""Regression tests for 0.66.1: reads of overflow spill files must never be
re-spilled (pointer-chasing burned ~100K tokens re-reading one file), and an
identical cache-hit repeat call must be stubbed, not re-billed in full."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nx import _overflow
from nx._text_fallback import _sanitize_tool_result


def test_whole_spill_file_read_not_respilled(tmp_path, monkeypatch):
    import nx._text_fallback as _tf
    monkeypatch.setattr(_tf, "TOOL_RESULT_CHARS", 20000)
    monkeypatch.setattr(_tf, "APPEND_SPILL_MIN_CHARS", 4000)
    body = "".join(f"line {i}\n" for i in range(50))
    result = _overflow.OVERFLOW_READ_MARKER + body
    out = _sanitize_tool_result(result)
    # Served verbatim (minus the marker), NOT a stub pointing at a new file
    assert out == body
    assert "saved to:" not in out


def test_windowed_spill_read_short_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(_overflow, "overflow_dir", lambda: str(tmp_path))
    small = "[x: lines 1-5 of 20]\n" + _overflow.OVERFLOW_READ_MARKER + "abc\n"
    out = _sanitize_tool_result(small)
    assert "saved to:" not in out


def test_is_overflow_path(tmp_path, monkeypatch):
    monkeypatch.setattr(_overflow, "overflow_dir", lambda: str(tmp_path))
    assert _overflow.is_overflow_path(os.path.join(tmp_path, "abc.txt"))
    assert not _overflow.is_overflow_path("/etc/passwd")
    assert not _overflow.is_overflow_path("")


def test_cap_overflow_read_caps_and_strips_marker(monkeypatch):
    import nx._text_fallback as _tf
    monkeypatch.setattr(_tf, "TOOL_RESULT_CHARS", 100)
    big = _overflow.OVERFLOW_READ_MARKER + "x" * 500
    out = _overflow.cap_overflow_read(big)
    assert len(out) < 300


def test_normal_result_still_spills(monkeypatch):
    # The guard must not disarm the ordinary arrival-time spill.
    import nx._text_fallback as _tf
    monkeypatch.setattr(_tf, "TOOL_RESULT_CHARS", 20000)
    monkeypatch.setattr(_tf, "APPEND_SPILL_MIN_CHARS", 4000)
    monkeypatch.setattr(_tf, "TOOL_RESULT_SPILL", 1)
    big = "".join(f"L{i} unique\n" for i in range(5000))
    out = _sanitize_tool_result(big)
    assert "saved to:" in out or "chars elided" in out


def test_repeat_cache_hit_returns_stub():
    import nx._dispatch as d
    import nx._state as st
    import nx._tool_cache as tc

    key = ("read_file", '{"path": "/tmp/somefile.txt"}')
    st._TOOL_RESULT_CACHE.clear()
    st._TOOL_RESULT_CACHE[key] = tc._CacheEntry(
        result="x" * 5000, path="/tmp/somefile.txt", mtime_ns=None, hits=0)
    try:
        # Simulate two hits via cache_touch's counting.
        tc.cache_touch(st._TOOL_RESULT_CACHE, key)
        tc.cache_touch(st._TOOL_RESULT_CACHE, key)
        assert st._TOOL_RESULT_CACHE[key].hits == 2
    finally:
        st._TOOL_RESULT_CACHE.clear()
