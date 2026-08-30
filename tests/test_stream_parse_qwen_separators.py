#!/usr/bin/env python3
"""Tests for nx/_stream_parse.py — Qwen-style inter-tool-call content
separators.

Qwen chat templates (qwen3.8-27b, Qwen3.8-flash-next, ...) emit a literal
"\\n" as a CONTENT delta between each native tool_call and after the last
one. StreamParser used to treat any content delta as "back to prose": it
wiped the generating-tool-call status line and — because the first
tool-call delta had already stopped the Live panel — opened a brand-new
Live re-rendering the ENTIRE accumulated preamble, then the next tool-call
delta froze that full text on screen again. Net effect on screen: the
preamble printed once per separator (2 calls -> doubled text, 3 ->
tripled). The byte-identical <=40-char separators also fed
REPEAT_CHUNK_LIMIT, so a turn with ~6 calls was misclassified as a
repetition collapse and discarded.

Offline: StreamParser is driven directly with fake SSE chunks (same shape
as tests/test_esc_approval.py). No model endpoint needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"
sys.argv = ["nx.py"]

import nx  # noqa: E402,F401  (importing the package initializes st.* state StreamParser reads)
from nx._stream_parse import StreamParser  # noqa: E402


# --- minimal fake SSE chunk objects ----------------------------------------
class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeTC:
    def __init__(self, index, id_, name, args):
        self.index = index
        self.id = id_
        self.function = _FakeFunction(name, args)


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self._reasoning = reasoning

    @property
    def reasoning_content(self):
        return self._reasoning


class _FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta=None, finish_reason=None):
        self.choices = [_FakeChoice(delta, finish_reason)] if delta is not None else []
        self.usage = None
        self.model_extra = {}


class _NoLive:
    """Console stand-in whose Live panels are counted, not rendered."""
    def __init__(self):
        self.live_starts = 0

    def __call__(self):
        return self

    @property
    def is_terminal(self):
        return False

    @property
    def width(self):
        return 80


class _CountingLive:
    instances = 0

    def __init__(self, console=None, refresh_per_second=12, transient=False):
        _CountingLive.instances += 1
        self.updates = 0
        self.last_render = None
        self.stopped = False

    def start(self):
        pass

    def update(self, renderable):
        self.updates += 1
        self.last_render = renderable

    def stop(self):
        self.stopped = True


def _parser():
    """A StreamParser wired to count Live-panel activity instead of rendering.
    Patches sp.Live for the caller's scope — restore via the returned
    `_restore` callable (context-manager style) so other tests see real Live."""
    import nx._stream_parse as sp
    saved_live = sp.Live
    _CountingLive.instances = 0
    sp.Live = _CountingLive
    return StreamParser(_NoLive()(), 0.0), saved_live


def _qwen_turn(p, n_calls=3, preamble="Preamble prose before the calls."):
    """Feed a Qwen-shaped turn: reasoning, preamble, then tool calls each
    separated by a literal "\n" content delta, plus trailing ones."""
    p.process(_FakeChunk(_FakeDelta(reasoning="thinking hard")))
    p.process(_FakeChunk(_FakeDelta(content=preamble)))
    for i in range(n_calls):
        p.process(_FakeChunk(_FakeDelta(
            tool_calls=[_FakeTC(i, f"call_{i}", "run_bash", '{"command": "true"}')])))
        p.process(_FakeChunk(_FakeDelta(content="\n")))  # the separator
    p.process(_FakeChunk(_FakeDelta(content="\n")))  # trailing separator(s)
    p.process(_FakeChunk(_FakeDelta(content="\n")))
    p.process(_FakeChunk(_FakeDelta()))


def test_separator_never_reopens_live():
    """The core regression: after tool-call deltas start, "\\n" separators must
    not open a fresh Live panel (which re-rendered the whole preamble — the
    double/triple line-repeat bug)."""
    p, saved_live = _parser()
    import nx._stream_parse as sp
    try:
        _qwen_turn(p, n_calls=3)
    finally:
        sp.Live = saved_live
    assert _CountingLive.instances == 1, (
        f"expected exactly 1 Live panel (the preamble), got {_CountingLive.instances} "
        "— every separator opened another one, re-rendering the full text")
    assert p.tcs and len(p.tcs) == 3, "all three tool calls must still be captured"
    assert p.content == ["Preamble prose before the calls."], (
        "separators must not be appended as content")


def test_separator_does_not_trip_repeat_collapse():
    """Byte-identical <=40-char separators (2, 3, ... 6+ calls) must not count
    toward REPEAT_CHUNK_LIMIT — a 6-call Qwen turn is not a repetition
    collapse."""
    p, saved_live = _parser()
    import nx._stream_parse as sp
    try:
        _qwen_turn(p, n_calls=8)
    finally:
        sp.Live = saved_live
    assert not p.loop_cut, "separators must not trigger the collapse cut"
    assert p.repeat_chunk_run < 6, (
        f"repeat_chunk_run={p.repeat_chunk_run} — separators fed the "
        "repetition-collapse detector")


def test_content_between_calls_still_renders():
    """Non-whitespace content interleaved after a tool call (some servers put
    real commentary there) must still be kept as content, not dropped."""
    p, saved_live = _parser()
    import nx._stream_parse as sp
    try:
        p.process(_FakeChunk(_FakeDelta(reasoning="thinking")))
        p.process(_FakeChunk(_FakeDelta(content="first line")))
        p.process(_FakeChunk(_FakeDelta(
            tool_calls=[_FakeTC(0, "call_0", "run_bash", '{"command": "true"}')])))
        p.process(_FakeChunk(_FakeDelta(content="\n")))
        p.process(_FakeChunk(_FakeDelta(
            tool_calls=[_FakeTC(1, "call_1", "run_bash", '{"command": "true"}')])))
        p.process(_FakeChunk(_FakeDelta(content="mid-turn commentary")))
        p.process(_FakeChunk(_FakeDelta()))
    finally:
        sp.Live = saved_live
    assert p.content[-1] == "mid-turn commentary", (
        "real (non-whitespace) interleaved content must be kept")
    assert len(p.tcs) == 2


def test_text_tool_tag_live_never_reopens():
    """Second facet of the same rendering bug: after a text-protocol
    <tool_call> tag froze the Live panel, later content must not open a new
    one either (previously any content after the tag re-rendered the whole
    accumulated string)."""
    import nx._stream_parse as sp
    p, saved_live = _parser()
    try:
        p.process(_FakeChunk(_FakeDelta(content="prose before tag ")))
        p.process(_FakeChunk(_FakeDelta(
            content='[nx_tool_call]{"name": "run_bash"}[/nx_tool_call]')))
        p.process(_FakeChunk(_FakeDelta(content=" tail after tag")))
    finally:
        sp.Live = saved_live
    assert _CountingLive.instances == 1, (
        f"expected 1 Live panel for the tag-prefix, got {_CountingLive.instances}")
    assert "prose before tag" in p._content_str
    assert p.in_text_tool_tag


if __name__ == "__main__":
    test_separator_never_reopens_live()
    test_separator_does_not_trip_repeat_collapse()
    test_content_between_calls_still_renders()
    test_text_tool_tag_live_never_reopens()
    print("PASS — all stream-parse separator tests")
