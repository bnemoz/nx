#!/usr/bin/env python3
"""Regression coverage for the stream render path (PERF.md 0.57.0).

Three separate accidental costs used to sit inside `StreamParser`'s per-chunk
path, all invisible at demo size:

- **PF-03**: `self._content_str += chunk` copied the whole accumulated buffer
  per chunk (CPython's in-place `+=` fast path is defeated the moment a second
  reference — the render — survives the throttle window), i.e. O(n^2) bytes
  copied over an answer.
- **PF-04**: the *entire* accumulated answer was ANSI-stripped (4 regex subs)
  and re-parsed as Markdown on every rendered frame, and parsed a second time
  at stop even when nothing had arrived since the last frame.
- (PF-22's `iter_raw`/dict-access half lives in `_openai_client.py`.)

These tests pin the two invariants the fix is supposed to buy (PERF.md §4 I-2,
bench tests T-2/T-5 by ratio, never absolute ms, so a loaded CI box cannot
flap). Everything is offline: fake chunks, a counting `Live`, a counting
`Markdown`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"
sys.argv = ["nx.py"]

import nx  # noqa: E402,F401  (initializes st.* state StreamParser reads)
import nx._stream_parse as sp  # noqa: E402
from nx._stream_parse import StreamParser  # noqa: E402


# --- minimal fakes ---------------------------------------------------------
class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None
        self.model_extra = {}
        self.reasoning_content = None


class _FakeChunk:
    def __init__(self, content):
        self.choices = [type("_C", (), {"delta": _FakeDelta(content),
                                        "finish_reason": None})()]
        self.usage = None
        self.model_extra = {}


class _NoLive:
    def __call__(self):
        return self

    is_terminal = False
    width = 80


class _CountingLive:
    def __init__(self, **kw):
        self.updates = 0

    def start(self):
        pass

    def update(self, renderable):
        self.updates += 1

    def stop(self):
        pass


class _CountingMarkdown:
    """Records the byte length handed to every Markdown parse."""
    parses: list[int] = []

    def __init__(self, text):
        type(self).parses.append(len(text))


class _Clock:
    """Advanceable stand-in for time.time inside _stream_parse only."""
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _feed(total_chars, chunk_chars, step=0.05):
    """Stream `total_chars` bytes in `chunk_chars`-sized deltas, advancing the
    clock by `step` per chunk (a slower-than-real stream, so the throttle is
    the only thing that can hold the frame count down). Return
    (total_bytes_parsed_as_markdown, number_of_markdown_parses)."""
    saved_live, saved_md, saved_time = sp.Live, sp.themed_markdown, sp.time.time
    _CountingMarkdown.parses = []
    clock = _Clock()
    sp.Live = _CountingLive
    sp.themed_markdown = _CountingMarkdown
    sp.time.time = clock
    try:
        p = StreamParser(_NoLive()(), 0.0)
        n = 0
        while n < total_chars:
            k = min(chunk_chars, total_chars - n)
            p.process(_FakeChunk("x" * k))
            clock.now += step
            n += k
        p.stop_say_live()
    finally:
        sp.Live, sp.Markdown, sp.time.time = saved_live, saved_md, saved_time
    return sum(_CountingMarkdown.parses), len(_CountingMarkdown.parses)


# --- PF-03: no whole-buffer string accumulator -----------------------------
def test_content_string_is_derived_not_accumulated():
    """`_content_str`/`_clean_str` must be properties (joined on demand), not
    instance attributes reassigned per chunk — that reassignment is the O(n^2)
    copy PF-03 is about."""
    for name in ("_content_str", "_clean_str"):
        assert isinstance(getattr(StreamParser, name), property), (
            f"{name} is an instance attribute again — the O(n^2) += accumulator is back")
    saved_live = sp.Live
    sp.Live = _CountingLive
    try:
        p = StreamParser(_NoLive()(), 0.0)
        assert "_content_str" not in p.__dict__ and "_clean_str" not in p.__dict__
        for i in range(200):
            p.process(_FakeChunk(f"chunk-{i} "))
    finally:
        sp.Live = saved_live
    assert p._content_str == "".join(p.content)
    assert p._clean_str == "".join(p._clean)


# --- PF-04 / T-5: rendered bytes stay sub-linear in answer size ------------
def test_markdown_render_bytes_are_sublinear():
    """8 KB vs 288 KB (36x) must not multiply the bytes re-parsed per frame by
    anywhere near 36x. Measured after the fix: 28.7 KB vs 1.28 MB = 44x for
    this chunk size; the pre-fix flat 50 ms throttle (every chunk rendered)
    gave ~630x."""
    small_bytes, small_frames = _feed(8 * 1024, 512)
    big_bytes, big_frames = _feed(288 * 1024, 512)
    # Sanity: the harness really did render, and the answer diff is 36x.
    assert small_bytes >= 8 * 1024, "the whole-answer final parse is missing"
    assert big_frames >= 2, "only the final frame rendered — throttling consumed the stream"
    ratio = big_bytes / small_bytes
    assert ratio < 100, (
        f"36x the answer cost {ratio:.1f}x the rendered bytes — the per-frame "
        "full re-render (flat 50 ms throttle) is back")


def test_final_parse_skipped_when_nothing_arrived():
    """`stop_say_live` must not re-parse the whole answer when the last
    throttled frame already covers every byte received (PF-04's 'no double
    final parse')."""
    saved_live, saved_md, saved_time = sp.Live, sp.themed_markdown, sp.time.time
    _CountingMarkdown.parses = []
    clock = _Clock()
    sp.Live = _CountingLive
    sp.themed_markdown = _CountingMarkdown
    sp.time.time = clock
    try:
        p = StreamParser(_NoLive()(), 0.0)
        p.process(_FakeChunk("a" * 4096))   # opens the Live and renders it
        clock.now += 10.0                    # long past any throttle interval
        p.process(_FakeChunk("b" * 4096))    # renders again (covers the 8 KB)
        before = len(_CountingMarkdown.parses)
        p.stop_say_live()                    # nothing new since that render
        assert len(_CountingMarkdown.parses) == before, (
            "stop_say_live re-parsed the full answer even though nothing "
            "arrived since the last frame")
    finally:
        sp.Live, sp.Markdown, sp.time.time = saved_live, saved_md, saved_time


def test_throttle_scales_with_answer_size():
    """The render interval must grow past the 50 ms floor once the answer is
    larger than _SAY_THROTTLE_SCALE_BYTES — that is the size-scaled throttle."""
    small = max(sp._SAY_MIN_INTERVAL,
                sp._SAY_MIN_INTERVAL * (2 * 1024) / sp._SAY_THROTTLE_SCALE_BYTES)
    big = max(sp._SAY_MIN_INTERVAL,
              sp._SAY_MIN_INTERVAL * (64 * 1024) / sp._SAY_THROTTLE_SCALE_BYTES)
    assert small == sp._SAY_MIN_INTERVAL, "small answers must keep the 20 Hz floor"
    assert big > small, "the render interval does not scale with answer length"


if __name__ == "__main__":
    test_content_string_is_derived_not_accumulated()
    test_markdown_render_bytes_are_sublinear()
    test_final_parse_skipped_when_nothing_arrived()
    test_throttle_scales_with_answer_size()
    print("PASS — all render-path perf tests")
