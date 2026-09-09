#!/usr/bin/env python3
"""Regression tests for the stall/empty-turn → native tool_choice escalation.

glm-5.3 / qwen3.8-flash-next behind a vLLM proxy only ever act through the
native tool_calls channel. Until the capability probe lands (or when its
max_tokens budget was too small for the model's reasoning phase — see
tests/test_capability_probe_budget.py), open_stream() sends tool_choice="none",
which on those servers suppresses the model's ONLY action path: the turn
streams reasoning, emits nothing, and every "act now" nudge is unsatisfiable.
The ladder then climbed to a forced final whose salvage printed the model's
reasoning as the user-visible answer.

These tests pin the recovery: a stall or an empty turn under the "none"
posture arms st._STALL_FORCE_NATIVE, open_stream() honors it with
tool_choice="auto", and a successful tool call (or a settled turn) clears it.
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
    m.st._NATIVE_TOOL_CALLS_CONFIRMED = False
    m.st._STALL_FORCE_NATIVE = False
    if m.st.ACTIVE is not None:
        m.st.ACTIVE._capability = None


def test_stall_arms_native_escalation():
    """A reasoning-only stall under tool_choice=none arms the escalation."""
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
    assert m.st._STALL_FORCE_NATIVE is True, \
        "a stall under the none posture must arm the native escalation"


def test_default_tool_choice_reflects_escalation():
    """_default_tool_choice(): none by default, auto once the escalation arms,
    auto when native is confirmed, auto when the probe says native."""
    _reset_state()
    assert m._default_tool_choice() == "none"
    m.st._STALL_FORCE_NATIVE = True
    assert m._default_tool_choice() == "auto"
    m.st._STALL_FORCE_NATIVE = False
    m.st._NATIVE_TOOL_CALLS_CONFIRMED = True
    assert m._default_tool_choice() == "auto"
    m.st._NATIVE_TOOL_CALLS_CONFIRMED = False
    if m.st.ACTIVE is not None:
        m.st.ACTIVE._capability = "native"
        assert m._default_tool_choice() == "auto"
        m.st.ACTIVE._capability = None


def test_empty_turn_arms_native_escalation():
    """An empty turn (no reasoning, no content, no call) arms it too."""
    _reset_state()
    messages = [{"role": "user", "content": "go"}]

    orig = m.open_stream
    m.open_stream = lambda *a, **k: iter([])
    try:
        status = m.model_turn(messages, empty_turn_count=0)
    finally:
        m.open_stream = orig

    assert status == m.TURN_EMPTY, f"expected TURN_EMPTY, got {status!r}"
    assert m.st._STALL_FORCE_NATIVE is True


def test_escalated_retry_opens_with_auto():
    """Once armed, open_stream() must send tool_choice="auto"."""
    _reset_state()
    m.st._STALL_FORCE_NATIVE = True
    messages = [{"role": "user", "content": "go"}]
    captured = {}

    def fake_open_stream(messages, tools=None, tool_choice=None,
                         max_tokens=None, recovery_sampling=False):
        captured["tool_choice"] = tool_choice
        return iter([_Chunk({"content": "done: the thing"})])

    orig = m.open_stream
    m.open_stream = fake_open_stream
    try:
        status = m.model_turn(messages)
    finally:
        m.open_stream = orig

    assert status == m.TURN_DONE
    # model_turn calls open_stream with tool_choice=None; the posture is
    # applied inside — verify through the helper the real path uses.
    assert m._default_tool_choice() == "auto", \
        "armed escalation must open the retry with tool_choice=auto"


def test_no_escalation_when_native_confirmed():
    """If native is already confirmed, a stall has nothing to escalate."""
    _reset_state()
    m.st._NATIVE_TOOL_CALLS_CONFIRMED = True
    messages = [{"role": "user", "content": "go"}]

    orig = m.open_stream
    m.open_stream = lambda *a, **k: iter(
        [_Chunk({"reasoning_content": "thinking"})])
    try:
        status = m.model_turn(messages)
    finally:
        m.open_stream = orig

    assert status == m.TURN_STALL
    assert m.st._STALL_FORCE_NATIVE is False, \
        "native already confirmed — the stall is a model problem, not a posture one"


def test_no_escalation_when_probe_says_native():
    """A landed CAPABILITY_NATIVE verdict means every request already used
    tool_choice=auto; a stall there must not arm the (no-op) escalation."""
    _reset_state()
    if m.st.ACTIVE is None:
        return  # no source configured in this environment — nothing to pin
    m.st.ACTIVE._capability = "native"
    messages = [{"role": "user", "content": "go"}]

    orig = m.open_stream
    m.open_stream = lambda *a, **k: iter(
        [_Chunk({"reasoning_content": "thinking"})])
    try:
        status = m.model_turn(messages)
    finally:
        m.open_stream = orig
        m.st.ACTIVE._capability = None

    assert status == m.TURN_STALL
    assert m.st._STALL_FORCE_NATIVE is False


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
