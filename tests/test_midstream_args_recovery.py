#!/usr/bin/env python3
"""Regression tests: a stream cut while tool-call args were mid-flight.

The model was actively streaming native tool-call args ("generating tool call
edit_file args 1.98K chars") when the stream ended abnormally, but the
post-stream recovery classified the turn as EMPTY TURN — no content and no tool
call — because the empty-turn guard only inspected the accumulated content
text, never the buffered tool-call args in parser.tcs.

Two paths are pinned:
  1. A collapse (loop_cut) with buffered args routes to the malformed-args
     recovery (the same nudge family as _handle_native_tool_calls), not the
     reasoning-stall ladder.
  2. A turn that somehow reaches the empty-turn check with a populated tcs is
     NOT classified empty — it falls through to the native handler.
"""
import os
import sys
import builtins
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402

_REAL_PRINT = builtins.print


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
    def __init__(self, delta, finish_reason="tool_calls"):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, delta=None, finish_reason="tool_calls"):
        self.choices = [_FakeChoice(delta, finish_reason)] if delta is not None else []
        self.usage = None
        self.model_extra = {}


class _FakeTC:
    def __init__(self, index, id_, name, args):
        self.index = index
        self.id = id_
        self.function = type("F", (), {"name": name, "arguments": args})()


def _install_quiet_stubs(m):
    """Silence the interactive-only pieces model_turn touches."""
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None


def _restore(m, saved_open, saved_watcher, saved_spinner, saved_run_tool=None):
    m.open_stream = saved_open
    m._interrupt_watcher = saved_watcher
    if saved_spinner is not None:
        m.LifeSpinner = saved_spinner
    else:
        m.__dict__.pop("LifeSpinner", None)
    if saved_run_tool is not None:
        m.run_tool = saved_run_tool
    builtins.print = _REAL_PRINT


def test_loop_cut_with_buffered_args_uses_malformed_args_recovery():
    """A collapse cut mid-args nudges like a malformed tool call, not a stall."""
    saved_open = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = getattr(m, "LifeSpinner", None)
    saved_run_tool = m.run_tool
    _install_quiet_stubs(m)
    # Args stream in, then a repetition-collapse-cut chunk (identical short
    # chunks beyond REPEAT_CHUNK_LIMIT set parser.loop_cut mid-stream).
    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "call_1", "edit_file", '{"path":"a.txt","old":"x","new":"y")}'),
        ]), finish_reason="length"),
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    m.run_tool = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("truncated args must not execute"))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "edit a.txt"}]
        # Force the loop-cut handoff: reasoning-loop counter > 0 makes the
        # stall ladder eligible; a bufferd-args cut must bypass it instead.
        status = m.model_turn(messages, reasoning_loop_cut_count=1)
        assert status == m.TURN_STREAM_CUT, f"got {status!r}"
        # The malformed-args nudge (not the reasoning-stall nudge) was appended.
        notes = [x for x in messages if x.get("role") == "user"]
        assert notes and notes[-1]["content"].startswith("[Runtime note: ")
        assert "malformed or truncated JSON arguments" in notes[-1]["content"], \
            f"expected malformed-args recovery, got: {notes[-1]['content'][:200]}"
        assert "reasoning" not in notes[-1]["content"].lower()
    finally:
        _restore(m, saved_open, saved_watcher, saved_spinner, saved_run_tool)
    _REAL_PRINT("PASS — loop cut with buffered args recovers as malformed args")


def test_empty_turn_guard_ignores_buffered_tcs():
    """A populated tcs must never classify the turn as EMPTY TURN."""
    saved_open = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = getattr(m, "LifeSpinner", None)
    saved_run_tool = m.run_tool
    _install_quiet_stubs(m)
    # Args arrive but no finish_reason/content; tcs is populated at stream end
    # with args that fail to parse → the native malformed-args path owns it.
    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "call_1", "write_file", '{"path":"x","content":"unterminated")}'),
        ]), finish_reason="length"),
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    m.run_tool = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("truncated args must not execute"))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "tool", "tool_call_id": "tc0", "content": "[exit 0]\n"}]
        status = m.model_turn(messages, empty_turn_count=0)
        assert status == m.TURN_STREAM_CUT, f"got {status!r}, not the empty-turn path"
        assert not any("EMPTY TURN" in (x.get("content") or "")
                       for x in messages if x.get("role") == "user")
    finally:
        _restore(m, saved_open, saved_watcher, saved_spinner, saved_run_tool)
    _REAL_PRINT("PASS — buffered tcs is never classified as an empty turn")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
