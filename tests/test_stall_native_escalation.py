#!/usr/bin/env python3
"""Regression tests for the default tool_choice="auto" posture (0.54.0) and
the stall/empty-turn → native escalation that backs it up when native is
explicitly disabled.

glm-5.3 / qwen3.8-flash-next behind a vLLM proxy only ever act through the
native tool_calls channel. The pre-0.54 default posture — tool_choice="none"
until native tool-calling was confirmed — suppressed that channel, so a fresh
session's first prompt (which regularly beat the startup capability probe)
stalled reasoning-only with no action path, and the ladder climbed to a
forced final whose salvage printed the model's reasoning as the answer.

0.54.0 flips the default: tool_choice="auto" from the very first request,
opted out only via /toolcalls text or NX_NATIVE_TOOLS=0. The escalation
(armed by a stall/empty nudge while "none" is in effect) remains as the
recovery for the opted-out case.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
for k in list(os.environ):
    if k.startswith("NX_"):
        del os.environ[k]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


class _Chunk:
    """Minimal SSE chunk shape for StreamParser.process()."""

    class _D:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.model_extra = {}
            self.content = kw.get("content")
            self.tool_calls = kw.get("tool_calls")
            self.reasoning_content = kw.get("reasoning_content")

    class _Choice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    def __init__(self, delta=None, finish_reason=None, usage=None):
        self.usage = usage
        self.choices = [self._Choice(self._D(**(delta or {})), finish_reason)] \
            if (delta or finish_reason) else []


def _reset_state():
    from nx import _compress
    m.st._NATIVE_TOOL_CALLS_CONFIRMED = False
    m.st._STALL_FORCE_NATIVE = False
    _compress.set_toolcalls_mode("auto")
    if m.st.ACTIVE is not None:
        m.st.ACTIVE._capability = None


def test_default_posture_is_auto():
    """The whole point of 0.54.0: a fresh session's first request already
    carries tool_choice="auto" — nothing to confirm, no probe to race."""
    _reset_state()
    assert m._default_tool_choice() == "auto"


def test_toolcalls_text_pins_none():
    """/toolcalls text (the rung past the malformed-args breaker) restores
    tool_choice="none" for the session."""
    _reset_state()
    from nx import _compress
    assert _compress.set_toolcalls_mode("text") is True
    assert m._default_tool_choice() == "none"
    assert _compress.set_toolcalls_mode("auto") is True
    assert m._default_tool_choice() == "auto"


def test_stall_does_not_escalate_under_auto_posture():
    """With "auto" already the posture, a stall is a model problem, not a
    posture problem — the (now pointless) escalation must not arm."""
    _reset_state()
    messages = [{"role": "user", "content": "do the thing"}]

    orig = m.open_stream
    m.open_stream = lambda *a, **k: iter(
        [_Chunk({"reasoning_content": "thinking about it"})])
    try:
        status = m.model_turn(messages, reasoning_loop_cut_count=0)
    finally:
        m.open_stream = orig

    assert status == m.TURN_STALL, f"expected TURN_STALL, got {status!r}"
    assert m.st._STALL_FORCE_NATIVE is False


def test_stall_escalates_when_native_disabled():
    """NX_NATIVE_TOOLS=0 restores the old "none" posture — and the stall
    ladder re-arms the native channel rather than nudging unsatisfiably."""
    _reset_state()
    from nx import _compress
    saved = _compress.NX_NATIVE_TOOLS
    _compress.NX_NATIVE_TOOLS = True  # True now MEANS "opted out" (0.54.0)
    assert m._default_tool_choice() == "none"
    messages = [{"role": "user", "content": "do the thing"}]
    try:
        orig = m.open_stream
        m.open_stream = lambda *a, **k: iter(
            [_Chunk({"reasoning_content": "thinking about it"})])
        try:
            status = m.model_turn(messages, reasoning_loop_cut_count=0)
        finally:
            m.open_stream = orig
    finally:
        _compress.NX_NATIVE_TOOLS = saved

    assert status == m.TURN_STALL
    assert m.st._STALL_FORCE_NATIVE is True, \
        "a stall under the opted-out none posture must arm the escalation"


def test_escalated_retry_opens_with_auto():
    """Once armed, the retry opens with tool_choice="auto" — overriding even
    the NX_NATIVE_TOOLS=0 opt-out (the stall is evidence it went wrong)."""
    _reset_state()
    m.st._STALL_FORCE_NATIVE = True
    assert m._default_tool_choice() == "auto"


def test_user_forced_text_protocol_is_respected():
    """/toolcalls text is a user decision — a stall under it must NOT
    re-enable native behind their back."""
    _reset_state()
    from nx import _compress
    assert _compress.set_toolcalls_mode("text") is True
    messages = [{"role": "user", "content": "go"}]

    orig = m.open_stream
    m.open_stream = lambda *a, **k: iter(
        [_Chunk({"reasoning_content": "thinking"})])
    try:
        status = m.model_turn(messages, reasoning_loop_cut_count=0)
    finally:
        m.open_stream = orig
        _compress.set_toolcalls_mode("auto")

    assert status == m.TURN_STALL
    assert m.st._STALL_FORCE_NATIVE is False, \
        "user-forced text protocol must not be overridden by the escalation"


def test_empty_turn_arms_escalation_when_disabled():
    """An empty turn (no reasoning, no content, no call) arms it too, under
    the opted-out posture."""
    _reset_state()
    from nx import _compress
    saved = _compress.NX_NATIVE_TOOLS
    _compress.NX_NATIVE_TOOLS = True
    messages = [{"role": "user", "content": "go"}]
    try:
        orig = m.open_stream
        m.open_stream = lambda *a, **k: iter([])
        try:
            status = m.model_turn(messages, empty_turn_count=0)
        finally:
            m.open_stream = orig
    finally:
        _compress.NX_NATIVE_TOOLS = saved

    assert status == m.TURN_EMPTY, f"expected TURN_EMPTY, got {status!r}"
    assert m.st._STALL_FORCE_NATIVE is True


def test_settled_turn_clears_escalation():
    """A turn that completes normally clears the flag."""
    _reset_state()
    m.st._STALL_FORCE_NATIVE = True

    def fake_model_turn(*a, **k):
        return m.TURN_DONE

    orig = m.model_turn
    m.model_turn = fake_model_turn
    try:
        m._run_model_turn_loop([{"role": "user", "content": "go"}])
    finally:
        m.model_turn = orig

    assert m.st._STALL_FORCE_NATIVE is False


def test_tool_turn_clears_escalation():
    """A successful tool call clears the per-episode flag (the session-sticky
    _NATIVE_TOOL_CALLS_CONFIRMED takes over if the call was native)."""
    _reset_state()
    m.st._STALL_FORCE_NATIVE = True

    def fake_model_turn(*a, **k):
        return m.TURN_TOOL

    orig = m.model_turn
    m.model_turn = fake_model_turn
    try:
        m._run_model_turn_loop([{"role": "user", "content": "go"}])
    finally:
        m.model_turn = orig

    assert m.st._STALL_FORCE_NATIVE is False


def test_probe_native_budget_is_reasoning_sized():
    """The capability probe's max_tokens must exceed a reasoning model's
    thinking phase — the old 64 cut the probe mid-think, misclassifying
    native servers as TEXT (which pinned tool_choice=none all session)."""
    from nx import _capability
    assert _capability._PROBE_MAX_TOKENS >= 512, (
        "probe budget too small for reasoning models' thinking phase")


def test_thinking_probe_budget_is_reasoning_sized():
    """Ditto for the thinking-separation probe (old 48)."""
    from nx import _thinking
    assert _thinking._PROBE_MAX_TOKENS >= 128


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
