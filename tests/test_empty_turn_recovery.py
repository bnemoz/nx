#!/usr/bin/env python3
"""Regression tests for the empty-turn auto-continue path.

An assistant turn that produces no content and no tool calls used to return
TURN_DONE, silently dropping the user to the chat prompt mid-task. The fix adds
a TURN_EMPTY status: the loop nudges the model and retries (with recovery
sampling), escalating to a forced final answer if it keeps stalling. These
tests pin that behavior against `model_turn`'s empty-turn branch and the
`_run_model_turn_loop` driver.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def test_empty_turn_returns_turn_empty_when_enabled():
    """A genuinely empty (no text, no tool) turn signals TURN_EMPTY, not DONE."""
    # Force a clean message layout: the last message is a tool result, which is
    # exactly the dangling-tool situation where an empty turn is most suspicious.
    messages = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "run_bash", "arguments": '{"command": "true"}'}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "[exit 0]\n"},
    ]
    before = [dict(x) for x in messages]

    # Bypass open_stream: simulate the model returning nothing.
    orig_open = m.open_stream
    m.open_stream = lambda *a, **k: iter([])  # no chunks → empty content/tcs
    try:
        status = m.model_turn(messages, empty_turn_count=0)
    finally:
        m.open_stream = orig_open

    assert status == m.TURN_EMPTY, f"expected TURN_EMPTY, got {status!r}"
    # A runtime note must have been appended so the next turn sees the nudge.
    notes = [x for x in messages if x.get("role") == "user"
             and "[Runtime note:" in x.get("content", "")]
    assert notes, "empty turn should nudge the current user turn"
    # The dangling tool result is preserved (history stays valid for the retry).
    assert messages[-1].get("role") != "assistant"  # no empty assistant turn appended
    assert any(x.get("role") == "tool" for x in messages)
    # No assistant message was appended for the empty turn.
    assert len([x for x in messages if x.get("role") == "assistant"]) == 1


def test_empty_turn_respects_disabled_retry_limit():
    """EMPTY_TURN_RETRY_LIMIT <= 0 falls back to TURN_DONE (opt-out)."""
    saved = m.EMPTY_TURN_RETRY_LIMIT
    m.EMPTY_TURN_RETRY_LIMIT = 0
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "run_bash", "arguments": '{"command": "true"}'}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "[exit 0]\n"},
    ]
    orig_open = m.open_stream
    m.open_stream = lambda *a, **k: iter([])
    try:
        status = m.model_turn(messages, empty_turn_count=0)
    finally:
        m.open_stream = orig_open
        m.EMPTY_TURN_RETRY_LIMIT = saved

    assert status == m.TURN_DONE, "disabled retries → empty turn is DONE"


def test_empty_turn_forced_final_is_done():
    """Under forced_final an empty turn stops (a failed forced answer), never auto-continues."""
    messages = [
        {"role": "user", "content": "answer now"},
    ]
    orig_open = m.open_stream
    m.open_stream = lambda *a, **k: iter([])
    try:
        status = m.model_turn(messages, empty_turn_count=0, forced_final=True)
    finally:
        m.open_stream = orig_open

    assert status == m.TURN_DONE, "empty forced-final turn should be DONE"


def test_run_loop_nudges_then_escalates_to_forced_final():
    """The loop nudges on each empty turn, then escalates to TURN_FORCE_FINAL.

    We drive model_turn with a stub that returns a scripted sequence of statuses
    so we can assert the loop's retry bookkeeping without touching the network.
    """
    calls = {"empty": 0, "force_final": 0, "tool": 0}

    # Scripted model_turn: empty, empty, empty → then escalate (model_turn
    # returns TURN_FORCE_FINAL on the 4th empty), then a final DONE.
    script = [
        m.TURN_EMPTY, m.TURN_EMPTY, m.TURN_EMPTY,
        m.TURN_FORCE_FINAL, m.TURN_DONE,
    ]
    state = {"i": 0}

    def fake_model_turn(messages, rl, ms, empty_turn_count=0,
                        forced_final=False, recovery_sampling=False,
                        malformed_call_count=0, step_retries=0):
        i = state["i"]
        state["i"] += 1
        if i < len(script):
            st = script[i]
        else:
            st = m.TURN_DONE
        if st == m.TURN_EMPTY:
            calls["empty"] += 1
            assert empty_turn_count == calls["empty"] - 1, \
                "empty_turn_count must be threaded through"
        if st == m.TURN_FORCE_FINAL:
            calls["force_final"] += 1
        return st

    orig = m.model_turn
    m.model_turn = fake_model_turn
    messages = [{"role": "user", "content": "go"}]
    try:
        m._run_model_turn_loop(messages)
    finally:
        m.model_turn = orig

    assert calls["empty"] == 3, f"expected 3 empty nudges, got {calls['empty']}"
    assert calls["force_final"] == 1, "should escalate to forced final once"


def test_successful_tool_resets_empty_counter():
    """A TURN_TOOL mid-sequence resets the empty-turn counter (mirrors other resets)."""
    seen = []

    def fake_model_turn(messages, rl, ms, empty_turn_count=0,
                        forced_final=False, recovery_sampling=False,
                        malformed_call_count=0, step_retries=0):
        seen.append(empty_turn_count)
        # empty, empty, TOOL, empty, DONE. The empty *after* the TOOL must see
        # count 0 — that's the reset we're proving (the final DONE call carries
        # the post-increment count of 1).
        seq = [m.TURN_EMPTY, m.TURN_EMPTY, m.TURN_TOOL, m.TURN_EMPTY, m.TURN_DONE]
        return seq[len(seen) - 1] if len(seen) <= len(seq) else m.TURN_DONE

    orig = m.model_turn
    m.model_turn = fake_model_turn
    try:
        m._run_model_turn_loop([{"role": "user", "content": "go"}])
    finally:
        m.model_turn = orig

    # The count climbs 0→1→2 through the empty turns, then resets to 0 right after
    # the TOOL, then climbs again (1) for the empty's retry before DONE.
    assert seen[:4] == [0, 1, 2, 0], f"empty counter not reset after tool: {seen}"
    assert seen[3] == 0, "the empty turn after a TOOL must start from a reset counter"


def test_recovery_sampling_enabled_on_empty_turn():
    """An empty turn should arm recovery sampling for the next attempt."""
    sampled = []

    def fake_model_turn(messages, rl, ms, empty_turn_count=0,
                        forced_final=False, recovery_sampling=False,
                        malformed_call_count=0, step_retries=0):
        sampled.append(recovery_sampling)
        return m.TURN_EMPTY if len(sampled) == 1 else m.TURN_DONE

    orig = m.model_turn
    m.model_turn = fake_model_turn
    try:
        m._run_model_turn_loop([{"role": "user", "content": "go"}])
    finally:
        m.model_turn = orig

    # First call: no recovery yet. Second call (the retry): recovery armed.
    assert sampled == [False, True], f"recovery not armed after empty: {sampled}"


def test_last_is_dangling_tool():
    assert m._last_is_dangling_tool([{"role": "tool", "content": "x"}]) is True
    assert m._last_is_dangling_tool([{"role": "user", "content": "x"}]) is False
    assert m._last_is_dangling_tool([{"role": "assistant", "content": "x"}]) is False
    assert m._last_is_dangling_tool([]) is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))