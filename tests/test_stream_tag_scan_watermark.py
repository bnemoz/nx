#!/usr/bin/env python3
"""Regression coverage for the streaming tool-tag scan watermark (PERF.md PF-02).

`_first_tool_tag_index(full)` used to re-scan the ENTIRE accumulated content
string on every SSE chunk: per-chunk cost grew 2.4 us -> 77.2 us as the buffer
grew (500 -> 8000 chunks), 618 ms of pure CPU on a 288 KB answer, on the SSE
read thread. The fix resumes each scan at most `_MAX_OPENER_LEN - 1` bytes
before the previous chunk boundary, so the total scan is O(answer) not O(n^2).
These tests pin both the flat cost and the split-opener correctness the
watermark must preserve.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"
sys.argv = ["nx.py"]

import nx  # noqa: E402,F401
import nx._stream_parse as sp  # noqa: E402
from nx._stream_parse import StreamParser  # noqa: E402


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None
        self.model_extra = {}
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta
        self.finish_reason = None


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(_FakeDelta(content))]
        self.usage = None
        self.model_extra = {}


class _NoLive:
    def __call__(self):
        return self

    is_terminal = False
    width = 80


def test_scan_window_is_bounded_per_chunk():
    """Every resumption of the tag scan starts within _MAX_OPENER_LEN bytes of
    the previous boundary -- i.e. the scan never re-reads the whole buffer."""
    starts = []
    real = sp._first_tool_tag_index

    def spy(text, start=0):
        starts.append((start, len(text)))
        return real(text, start)

    sp._first_tool_tag_index = spy
    p = StreamParser(_NoLive(), 0.0, quiet=True)
    try:
        for i in range(500):
            p.process(_FakeChunk(f"chunk-{i} " * 10))  # ~90 bytes each
    finally:
        sp._first_tool_tag_index = real
    assert len(starts) > 100
    for start, length in starts:
        assert length - start <= 200, (
            f"scan window was {length - start} bytes at offset {start} -- the "
            "watermark is not holding, this is the O(n^2) regression")


def test_per_chunk_scan_cost_is_flat():
    """The last chunk's scan must not be far more expensive than the first's
    (T-1 analogue): 4x the buffer must not cost 4x per chunk."""
    import time

    def scan_cost(n_chunks):
        real = sp._first_tool_tag_index
        t0 = time.perf_counter()
        p = StreamParser(_NoLive(), 0.0, quiet=True)
        for i in range(n_chunks):
            p.process(_FakeChunk(f"chunk-{i} " * 10))
        sp._first_tool_tag_index = real
        return time.perf_counter() - t0

    small = scan_cost(500)
    big = scan_cost(2000)
    # 4x the chunks should be ~4x the work, not ~16x. Generous bound so a
    # loaded CI box does not flap.
    assert big / max(small, 1e-6) < 8, (
        f"4x buffer cost {big / small:.1f}x the time -- quadratic scan is back")


def test_tag_split_across_chunks_is_still_detected():
    """Correctness the watermark must not break: an opener arriving in pieces
    across several chunks is still found, and prose before it is committed."""
    p = StreamParser(_NoLive(), 0.0, quiet=True)
    p.process(_FakeChunk("prose before "))
    p.process(_FakeChunk("<tool"))
    p.process(_FakeChunk('_call>{"name": "run_bash"}'))
    assert p.in_text_tool_tag, "a split <tool_call> opener was missed"
    assert p._content_str.startswith("prose before ")


def test_late_tag_after_a_long_answer_is_found():
    """The watermark must not skip a tag that opens deep in a long answer."""
    p = StreamParser(_NoLive(), 0.0, quiet=True)
    for i in range(2000):
        p.process(_FakeChunk("filler text " * 8))
    assert not p.in_text_tool_tag
    p.process(_FakeChunk("[nx_tool_call]"))
    assert p.in_text_tool_tag, "tag after 2000 chunks was missed by the watermark"


if __name__ == "__main__":
    test_scan_window_is_bounded_per_chunk()
    test_per_chunk_scan_cost_is_flat()
    test_tag_split_across_chunks_is_still_detected()
    test_late_tag_after_a_long_answer_is_found()
    print("PASS — all tool-tag scan watermark tests")
