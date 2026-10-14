#!/usr/bin/env python3
"""Regression tests for the REASONING-ONLY STALL / forced-final recovery ladder.

Covers the 0.22.1 behavior change: a forced-final turn in which the model
emits a real (non-`final_answer`) tool call now EXECUTES that call and keeps
the turn going (TURN_TOOL) instead of discarding it and dropping the user to
the chat prompt mid-task.

No live model or terminal needed — streams and the tool dispatcher are faked
the same way tests/test_esc_approval.py does it.
"""
import os
import sys
import builtins
import json
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
import nx._turn as _turn  # noqa: E402

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


class _Patched:
    """Silence + fake-stream boilerplate shared by the model_turn tests."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.saved = {}

    def __enter__(self):
        self.saved = {
            "open_stream": m.open_stream,
            "watcher": _turn._ensure_interrupt_watcher,
            "spinner": _turn.LifeSpinner,
            "print": builtins.print,
        }
        _turn._ensure_interrupt_watcher = lambda: None
        _turn.LifeSpinner = type("NoSpinner", (), {
            "__init__": lambda self, **kw: None,
            "start": lambda self: None,
            "stop": lambda self: None,
            "_t": None,
        })
        builtins.print = lambda *a, **k: None
        m.open_stream = lambda messages, **kw: iter(self.chunks)
        return self

    def __exit__(self, *exc):
        m.open_stream = self.saved["open_stream"]
        _turn._ensure_interrupt_watcher = self.saved["watcher"]
        _turn.LifeSpinner = self.saved["spinner"]
        builtins.print = self.saved["print"]
        return False


def _messages():
    return [{"role": "system", "content": m.SYSTEM},
            {"role": "user", "content": "answer me"}]


# ---------------------------------------------------------------------------
# 1. forced-final + native run_bash tool call → executed, TURN_TOOL
# ---------------------------------------------------------------------------
def test_forced_final_actionable_tool_call_executes():
    with _Patched([_FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "call_1", "run_bash", json.dumps({"command": "echo hi"}))]))]):
        messages = _messages()
        start = len(messages)          # hermetic: inspect only the appended slice
        status = m.model_turn(messages, forced_final=True)
    assert status == m.TURN_TOOL, f"expected TURN_TOOL, got {status!r}"
    appended = messages[start:]
    # answer to the issued call, located by tool_call_id — never by position
    tool = [x for x in appended if x.get("role") == "tool"
            and x.get("tool_call_id") == "call_1"]
    assert len(tool) == 1, appended
    assert "hi" in tool[0]["content"], tool[0]["content"][:200]
    # the assistant turn carrying the call was appended too
    assert any(x.get("role") == "assistant" and
               any(tc.get("id") == "call_1" for tc in x.get("tool_calls") or [])
               for x in appended), appended
    # no unexpected appended messages: only assistant/tool kinds
    assert all(x.get("role") in ("assistant", "tool") for x in appended), appended


# ---------------------------------------------------------------------------
# 2. forced-final + text-protocol actionable tool call → executed, TURN_TOOL
# ---------------------------------------------------------------------------
def test_forced_final_text_actionable_tool_call_executes():
    opener = chr(0x3C) + "tool_call" + chr(0x3E)
    closer = chr(0x3C) + "/tool_call" + chr(0x3E)
    text = (opener + '{"name": "run_bash", "arguments": '
            '{"command": "echo hi"}}' + closer)
    with _Patched([_FakeChunk(_FakeDelta(content=text))]):
        messages = _messages()
        start = len(messages)          # hermetic: inspect only the appended slice
        status = m.model_turn(messages, forced_final=True)
    assert status == m.TURN_TOOL, f"expected TURN_TOOL, got {status!r}"
    appended = messages[start:]
    assert any(x.get("role") == "assistant" and "run_bash" in (x.get("content") or "")
               for x in appended), "assistant tool-call turn missing"
    # the observation landed as a user turn — asserted by kind, not index
    assert any(x.get("role") == "user" and "Observation (run_bash)" in (x.get("content") or "")
               for x in appended), appended
    # no unexpected appended messages: only assistant/user kinds
    assert all(x.get("role") in ("assistant", "user") for x in appended), appended


# ---------------------------------------------------------------------------
# 3. mixed empty final_answer + actionable → actionable wins, TURN_TOOL
# ---------------------------------------------------------------------------
def test_forced_final_mixed_calls_execute_actionable():
    empty_final = json.dumps({"answer": ""})
    with _Patched([_FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "c1", "final_answer", empty_final),
            _FakeTC(1, "c2", "run_bash", json.dumps({"command": "echo hi"}))]))]):
        messages = _messages()
        start = len(messages)          # hermetic: inspect only the appended slice
        status = m.model_turn(messages, forced_final=True)
    assert status == m.TURN_TOOL, f"expected TURN_TOOL, got {status!r}"
    appended = messages[start:]
    # the actionable call was executed: the tool message answers call c2,
    # located by tool_call_id — never by position
    tool = [x for x in appended if x.get("role") == "tool"
            and x.get("tool_call_id") == "c2"]
    assert len(tool) == 1, appended
    assert "hi" in tool[0]["content"], tool[0]["content"][:200]
    # no unexpected appended messages: only assistant/tool kinds
    assert all(x.get("role") in ("assistant", "tool") for x in appended), appended


# ---------------------------------------------------------------------------
# 4. loop: TURN_TOOL from a forced-final turn keeps going (not TURN_DONE)
# ---------------------------------------------------------------------------
def test_run_loop_continues_after_forced_final_tool_turn():
    scripted = iter([
        m.TURN_STALL,        # reasoning-only stall → nudge
        m.TURN_FORCE_FINAL,  # last rung → force final
        m.TURN_TOOL,         # forced turn emitted an actionable call — keep going
        m.TURN_DONE,         # wrap-up
    ])
    calls = []

    def fake_model_turn(messages, *a, forced_final=False, **k):
        calls.append({"forced": forced_final})
        return next(scripted)

    with mock.patch.object(m, "model_turn", fake_model_turn), \
            mock.patch.object(builtins, "print", lambda *a, **k: None):
        summary = m._run_model_turn_loop(_messages())
    assert summary["status"] == "done", summary
    assert summary["turns"] == 4, summary
    assert [c["forced"] for c in calls] == [False, False, True, False], calls


# ---------------------------------------------------------------------------
# 5. loop: actionable forced-final tool call keeps the stall counter pinned
# ---------------------------------------------------------------------------
def test_run_loop_pin_reasoning_counter_after_forced_tool_turn():
    # Ordering, asserted through the captured kwargs: cut 0 and cut 1 get
    # nudge turns, cut 2 == retry_limit-1 opens a FORCED turn, and that
    # forced turn then ACTS (TURN_TOOL) instead of ending at the prompt. The
    # stall counter must stay pinned at the last rung, so the next stall
    # (call 5) opens ANOTHER forced turn instead of re-climbing the ladder.
    calls = []

    def fake_model_turn(messages, reasoning_loop_cut_count=0, *a,
                        forced_final=False, **k):
        calls.append({"forced": forced_final, "cuts": reasoning_loop_cut_count})
        n = len(calls)
        if n == 1:
            return m.TURN_STALL          # first stall → nudge to act
        if n == 2:
            return m.TURN_STALL          # second stall → nudge again
        if n == 3:
            return m.TURN_FORCE_FINAL    # last rung reached
        if n == 4:
            assert forced_final, "4th call must be the forced-final turn"
            # TURN_FORCE_FINAL bumps the counter once more before the forced
            # call, so the forced turn itself reports 3 (printed as 3/3)
            assert reasoning_loop_cut_count == 3, calls
            return m.TURN_TOOL           # forced turn emitted an actionable call
        if n == 5:
            # A fresh stall right after the forced tool turn arrives UNforced
            # with the counter pinned at the last rung (2), not reset to 0.
            # Real model_turn at cut 2 returns TURN_FORCE_FINAL immediately.
            assert not forced_final, "5th call is a normal turn"
            assert reasoning_loop_cut_count == 2, calls
            return m.TURN_FORCE_FINAL    # pinned ⇒ escalates on first fresh stall
        if n == 6:
            assert forced_final, "6th call must be forced (counter pinned)"
            return m.TURN_DONE

    with mock.patch.object(m, "model_turn", fake_model_turn), \
            mock.patch.object(_turn, "_launch_verify_if_settled", lambda *a, **k: None), \
            mock.patch.object(builtins, "print", lambda *a, **k: None):
        summary = m._run_model_turn_loop(_messages())
    assert summary["status"] == "done", summary
    assert len(calls) == 6, calls
    assert [c["forced"] for c in calls] == [False, False, False, True, False, True], calls


# ---------------------------------------------------------------------------
# 6. plain final_answer under forced-final still ends the turn (unchanged)
# ---------------------------------------------------------------------------
def test_forced_final_answer_still_done():
    args = json.dumps({"answer": "the answer", "status": "answered"})
    with _Patched([_FakeChunk(_FakeDelta(
            tool_calls=[_FakeTC(0, "f1", "final_answer", args)]))]):
        messages = _messages()
        status = m.model_turn(messages, forced_final=True)
    assert status == m.TURN_DONE, f"expected TURN_DONE, got {status!r}"
    assert messages[-1]["content"] == "the answer"


# ---------------------------------------------------------------------------
# 7. empty final_answer, nothing else → honest give-up (0.154.7: nothing was
# produced, so headless callers must not score the silent non-answer as done)
# ---------------------------------------------------------------------------
def test_forced_final_empty_answer_gives_up():
    from nx import _turn
    with _Patched([_FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "f1", "final_answer", json.dumps({"answer": ""}))]))]):
        messages = _messages()
        status = m.model_turn(messages, forced_final=True)
    assert status == m.TURN_GAVE_UP, f"expected TURN_GAVE_UP, got {status!r}"
    assert "no final_answer" in _turn._LAST_GIVE_UP_CAUSE


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS — {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL — {name}: {e}")
    sys.exit(1 if fails else 0)
