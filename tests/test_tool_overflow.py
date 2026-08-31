#!/usr/bin/env python3
"""Regression tests for tool-result spillover (nx/_overflow.py).

Oversized tool results used to be silently head/tail-cut by the
TOOL_RESULT_CHARS cap — the model had no way to know (or recover) the elided
middle. Now the full text is spilled to SESSION_HOME/overflow/<sha256>.txt and
the result carries head + tail + that path.
"""
import os
import sys
import tempfile
import time


_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _text_fallback as tf  # noqa: E402
from nx import _overflow as ov  # noqa: E402


def _set_n(monkeypatch, value):
    monkeypatch.setattr(tf, "TOOL_RESULT_CHARS", value)


def _set_spill(monkeypatch, value):
    monkeypatch.setattr(tf, "TOOL_RESULT_SPILL", value)


def _body(tag="line", n=50):
    # >1000 chars: below that _sanitize_tool_result early-returns (no dedup/cap).
    return "\n".join(f"{tag} {i} {'y' * 30}" for i in range(n))


def _spill_path(out: str) -> str:
    """Extract the spill path from a result's '[... saved to: <path> — ...]' note."""
    return out.rsplit("saved to: ", 1)[1].split(" —", 1)[0].strip()


def test_oversized_result_spills_and_names_path(monkeypatch):
    _set_n(monkeypatch, 200)
    body = "\n".join(f"line {i} {'x' * 40}" for i in range(30))  # ~1.4KB

    out = m._sanitize_tool_result(body)

    assert "overflow" in out and "read_file that path for the middle" in out
    # the path named exists and holds the FULL text
    path = _spill_path(out)
    assert path.endswith(".txt") and os.path.exists(path), path
    with open(path) as f:
        assert f.read() == body


def test_small_result_does_not_spill(monkeypatch):
    _set_n(monkeypatch, 20000)
    out = m._sanitize_tool_result("hello\nworld")

    assert "overflow" not in out
    assert out == "hello\nworld"


def test_spill_disabled_falls_back_to_plain_elision(monkeypatch):
    _set_n(monkeypatch, 200)
    _set_spill(monkeypatch, 0)
    body = _body()

    out = m._sanitize_tool_result(body)

    assert "chars elided to bound context" in out
    assert "overflow" not in out


def test_spill_io_failure_falls_back_to_plain_elision(monkeypatch):
    _set_n(monkeypatch, 200)
    _set_spill(monkeypatch, 1)

    def _boom(_text):
        raise OSError("disk full")

    monkeypatch.setattr(tf, "spill_to_disk", _boom)  # bound into tf's namespace
    out = m._sanitize_tool_result(_body())

    assert "chars elided to bound context" in out
    assert "overflow" not in out


def test_spill_file_is_content_addressed_and_deduped(monkeypatch):
    _set_n(monkeypatch, 200)
    for n in os.listdir(ov.overflow_dir()):  # isolate from earlier tests' spills
        os.remove(ov._path_for(n))
    body = _body()
    first = m._sanitize_tool_result(body)
    path = _spill_path(first)

    m._sanitize_tool_result(body)

    m._sanitize_tool_result(body)

    names = [n for n in os.listdir(ov.overflow_dir()) if n.endswith(".txt")]
    assert len(names) == 1, names
    assert os.path.basename(path) == ov._short_hash(body) + ".txt"


def test_spill_purges_stale_files(monkeypatch):
    _set_n(monkeypatch, 200)
    body = _body()
    out = m._sanitize_tool_result(body)
    path = _spill_path(out)

    old = time.time() - (ov.OVERFLOW_TTL_DAYS + 1) * 86400
    os.utime(path, (old, old))
    m._sanitize_tool_result(_body(tag="other"))

    assert not os.path.exists(path), "stale spill should have been purged"


def test_overflow_echo_bounded_by_char_budget(monkeypatch):
    huge = "\n".join("z" * 500 for _ in range(20))
    head, tail, _ = ov._echo_lines(huge)

    assert len(head) + len(tail) <= ov.MAX_SPILL_ECHO + 10
    assert head  # real content, not a bare marker
    assert tail


def test_single_giant_line_still_echoes(monkeypatch):
    head, tail, elided = ov._echo_lines("y" * 9000)

    assert len(head) + len(tail) <= ov.MAX_SPILL_ECHO + 10
    assert head or tail
