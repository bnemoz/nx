#!/usr/bin/env python3
"""Test the Esc-at-approval → back-to-chat behavior.

No live model or terminal needed — we monkeypatch the risk classifier and the
approval prompt so we can drive the control-flow deterministically, then assert
that:
  1. `_confirm` raises `_EscToChat` when the user picks Esc.
  2. `_confirm` returns True/False for Y/n as before.
  3. `run_tool` lets `_EscToChat` propagate (doesn't swallow it as a string).
  4. `model_turn` returns `TURN_ESC` when a tool is escaped, and fills in
     results for remaining tool calls so the message history stays valid.
  5. The native tool-call path: when Esc fires on the 1st of 2 tool calls,
     both calls get result messages (CANCELLED / SKIPPED).
"""
import os
import sys
import builtins
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m

# Capture the REAL print once at import time so tests can silence + restore it
# without the "builtins.print = print" self-reference bug (after reassigning,
# the bare name `print` resolves to the new value, not the original builtin).
_REAL_PRINT = builtins.print


def _use_test_sessions_dir():
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp


def setup_function(_function):
    _use_test_sessions_dir()


# ---------------------------------------------------------------------------
# 1. _confirm raises _EscToChat on "esc"
# ---------------------------------------------------------------------------
def test_confirm_esc_raises():
    saved_ask, saved_assess = m._ask_approval, m._assess_risk
    m._ask_approval = lambda prompt, allow_hint=None: "esc"
    m._assess_risk = lambda action: ("high", "because test")
    try:
        try:
            m._confirm("write foo.py (10 bytes)")
        except m._EscToChat:
            print("PASS — _confirm raised _EscToChat on esc")
            return
        raise AssertionError("_confirm did NOT raise _EscToChat on esc")
    finally:
        m._ask_approval, m._assess_risk = saved_ask, saved_assess


# ---------------------------------------------------------------------------
# 2. _confirm still returns True/False for Y/n
# ---------------------------------------------------------------------------
def test_confirm_y_n():
    saved_ask, saved_assess = m._ask_approval, m._assess_risk
    m._assess_risk = lambda action: ("high", "because test")
    builtins.print = lambda *a, **k: None  # hush
    try:
        m._ask_approval = lambda prompt, allow_hint=None: "y"
        assert m._confirm("write foo.py") is True
        m._ask_approval = lambda prompt, allow_hint=None: "n"
        assert m._confirm("write foo.py") is False
    finally:
        m._ask_approval, m._assess_risk = saved_ask, saved_assess
        builtins.print = _REAL_PRINT
    print("PASS — _confirm returns True/False for Y/n")


# ---------------------------------------------------------------------------
# 3. run_tool propagates _EscToChat (doesn't swallow it)
# ---------------------------------------------------------------------------
def test_run_tool_propagates_esc():
    saved_confirm = m._confirm
    # make write_file's _confirm raise
    m._confirm = lambda action, **kw: (_ for _ in ()).throw(m._EscToChat(action))
    builtins.print = lambda *a, **k: None
    # suppress the spinner so we don't spawn a thread
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
    })
    raised = False
    try:
        try:
            m.run_tool("write_file", {"path": "x", "content": "y"})
        except m._EscToChat:
            raised = True
    finally:
        m._confirm = saved_confirm
        builtins.print = _REAL_PRINT
    if raised:
        print("PASS — run_tool propagated _EscToChat")
        return
    raise AssertionError("run_tool swallowed _EscToChat")


# ---------------------------------------------------------------------------
# 4. model_turn returns TURN_ESC + keeps history valid
#    Drive it with a fake stream that emits two tool calls; escape the first.
# ---------------------------------------------------------------------------
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
    """Minimal stand-in for a streamed tool-call delta."""
    def __init__(self, index, id_, name, args):
        self.index = index
        self.id = id_
        self.function = type("F", (), {"name": name, "arguments": args})()


def test_model_turn_native_esc():
    saved_open_stream = m.open_stream
    saved_confirm = m._confirm
    saved_watcher = m._interrupt_watcher
    m._interrupt_watcher = lambda: None  # don't spawn the thread
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None

    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(0, "call_1", "write_file", '{"path":"a","content":"x"}')])),
        _FakeChunk(_FakeDelta(tool_calls=[_FakeTC(1, "call_2", "edit_file", '{"path":"b","old":"o","new":"n"}')])),
        _FakeChunk(),  # final empty-choices chunk
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    # Esc on the FIRST tool; the second should be skipped, not re-prompted.
    m._confirm = lambda action, **kw: (_ for _ in ()).throw(m._EscToChat(action))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "do it"}]
        status = m.model_turn(messages)
        assert status == m.TURN_ESC, f"expected TURN_ESC, got {status!r}"
        # The assistant(tool_calls) turn must be present…
        asst = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")]
        assert len(asst) == 1, f"expected 1 assistant(tool_calls), got {len(asst)}"
        ids = [tc["id"] for tc in asst[0]["tool_calls"]]
        assert ids == ["call_1", "call_2"], f"tool call ids wrong: {ids}"
        # …and EVERY tool_call must have a matching tool result, or the chat
        # template will reject the context on the next request.
        tool_msgs = [msg for msg in messages if msg.get("role") == "tool"]
        assert len(tool_msgs) == 2, f"expected 2 tool results, got {len(tool_msgs)}"
        result_ids = {tm["tool_call_id"] for tm in tool_msgs}
        assert result_ids == {"call_1", "call_2"}, f"missing results: {result_ids}"
        # first = CANCELLED, second = SKIPPED
        contents = {tm["tool_call_id"]: tm["content"] for tm in tool_msgs}
        assert "CANCELLED" in contents["call_1"], contents["call_1"]
        assert "SKIPPED" in contents["call_2"], contents["call_2"]
        # a trailing user note telling the model what happened
        assert messages[-1]["role"] == "user"
        assert "Esc" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._confirm = saved_confirm
        m._interrupt_watcher = saved_watcher
        builtins.print = _REAL_PRINT
    print("PASS — model_turn returns TURN_ESC, history stays valid (all calls have results)")


# ---------------------------------------------------------------------------
# 5. Malformed native tool-call args are retried, not executed as {}
# ---------------------------------------------------------------------------
def test_model_turn_malformed_native_tool_args_retries_cleanly():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_run_tool = m.run_tool
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [
        _FakeChunk(_FakeDelta(tool_calls=[
            _FakeTC(0, "call_bad", "write_file",
                    '{"path":"big.html","content":"unterminated')
        ]), finish_reason="length"),
    ]
    m.open_stream = lambda messages, **kw: iter(chunks)
    m.run_tool = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("malformed tool args must not execute"))
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "write a large file"}]
        status = m.model_turn(messages)
        assert status == m.TURN_STREAM_CUT, f"expected TURN_STREAM_CUT, got {status!r}"
        assert len(messages) == 2, f"partial assistant/tool history leaked: {messages!r}"
        assert messages[-1]["role"] == "user"
        assert "malformed or truncated JSON arguments" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.run_tool = saved_run_tool
        builtins.print = _REAL_PRINT
    print("PASS — malformed native tool args retry cleanly without executing {}")


# ---------------------------------------------------------------------------
# 6. REPL drops back to chat input on TURN_ESC (no extra model turn)
# ---------------------------------------------------------------------------
def test_repl_breaks_on_esc():
    saved_read_multiline = m.read_multiline
    saved_model_turn = m.model_turn
    saved_banner = m._banner
    m._banner = lambda: ""
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    call_count = {"n": 0}
    def fake_model_turn(messages, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
                        empty_turn_count=0, forced_final=False, recovery_sampling=False,
                        malformed_call_count=0):
        call_count["n"] += 1
        # first turn: model emits a tool call, user escapes it
        if call_count["n"] == 1:
            return m.TURN_ESC
        return m.TURN_DONE
    prompts = iter(["hello", "again", "/quit"])
    m.read_multiline = lambda history=None, initial=None: next(prompts)
    m.model_turn = fake_model_turn
    try:
        m.main()
        # model_turn should have been called exactly once per *real* user turn
        # (the TURN_ESC must break the inner loop, not trigger a retry). "hello"
        # → 1 call (esc, break), "again" → 1 call (done, break) = 2 total.
        # If TURN_ESC were mishandled (fell through to the steps++ branch) the
        # first turn would loop and call model_turn many more times.
        assert call_count["n"] == 2, f"expected 2 model_turn calls, got {call_count['n']}"
    finally:
        m.read_multiline = saved_read_multiline
        m.model_turn = saved_model_turn
        m._banner = saved_banner
        builtins.print = _REAL_PRINT
    print("PASS — REPL breaks inner loop on TURN_ESC, no extra model turn")


# ---------------------------------------------------------------------------
# 7. Reasoning-only stalls use the forced-finalizer path
# ---------------------------------------------------------------------------
def test_reasoning_only_stall_requests_forced_final():
    # Two-tier recovery: the LAST allowed stall (cut_count == retry_limit - 1)
    # forces a final answer. Pin retry_limit to 1 so the first (and only)
    # stall is also the last one, isolating that tier from the "nudge to
    # act" tier tested separately below.
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_limit = m.REASONING_ONLY_CHAR_LIMIT
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_CHAR_LIMIT = 3
    m.REASONING_ONLY_RETRY_LIMIT = 1
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [_FakeChunk(_FakeDelta(reasoning="abcd"))]
    m.open_stream = lambda messages, **kw: iter(chunks)
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_FORCE_FINAL, f"expected TURN_FORCE_FINAL, got {status!r}"
        assert "complete visible answer" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_CHAR_LIMIT = saved_limit
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — reasoning-only stall requests forced final answer")


def test_reasoning_only_stall_nudges_to_act_before_forcing_final():
    # First (non-last) stall: two-tier recovery nudges the model to act with
    # FULL tools still available (TURN_STALL), rather than immediately
    # restricting it to the final-answer tool. Pin retry_limit to 2 so
    # cut_count=0 is NOT the last retry (that's index 1).
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_limit = m.REASONING_ONLY_CHAR_LIMIT
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_CHAR_LIMIT = 3
    m.REASONING_ONLY_RETRY_LIMIT = 2
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [_FakeChunk(_FakeDelta(reasoning="abcd"))]
    m.open_stream = lambda messages, **kw: iter(chunks)
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages, reasoning_loop_cut_count=0)
        assert status == m.TURN_STALL, f"expected TURN_STALL, got {status!r}"
        assert messages[-1]["role"] == "user"
        assert "write_file" in messages[-1]["content"] or "run_bash" in messages[-1]["content"], \
            "first-tier nudge should push the model toward a concrete tool call"
        # The captured reasoning is preserved as its own assistant turn ahead
        # of the nudge, not silently dropped.
        assistant_msgs = [msg for msg in messages if msg.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert "abcd" in assistant_msgs[0]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_CHAR_LIMIT = saved_limit
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — reasoning-only stall nudges to act before forcing final")


# ---------------------------------------------------------------------------
# 8. Forced finalizer tool call becomes visible assistant text
# ---------------------------------------------------------------------------
def test_forced_final_tool_call_becomes_assistant_text():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    captured = {}

    def fake_open_stream(messages, **kw):
        captured.update(kw)
        args = '{"answer":"forced answer works","status":"answered"}'
        return iter([_FakeChunk(_FakeDelta(
            tool_calls=[_FakeTC(0, "final_1", "final_answer", args)]
        ))])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages, forced_final=True)
        assert status == m.TURN_DONE, f"expected TURN_DONE, got {status!r}"
        assert captured["tools"] == [m.FINAL_ANSWER_TOOL]
        assert captured["tool_choice"] == m.FINAL_ANSWER_TOOL_CHOICE
        assert captured["max_tokens"] == m.FORCED_FINAL_MAX_TOKENS
        assert messages[-1] == {"role": "assistant", "content": "forced answer works"}
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        builtins.print = _REAL_PRINT
    print("PASS — forced final_answer tool call becomes assistant text")


def test_forced_final_length_marks_partial_answer():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    m._interrupt_watcher = lambda: None
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None

    def fake_open_stream(messages, **kw):
        return iter([_FakeChunk(_FakeDelta(content="partial final"), finish_reason="length")])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages, forced_final=True)
        assert status == m.TURN_DONE, f"expected TURN_DONE, got {status!r}"
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].startswith("partial final")
        assert "Truncated by token limit" in messages[-1]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        builtins.print = _REAL_PRINT
    print("PASS — forced final length stop is marked in history")


def test_reasoning_only_empty_completion_forces_final_without_empty_assistant():
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_limit = m.REASONING_ONLY_CHAR_LIMIT
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_CHAR_LIMIT = 9999
    m.REASONING_ONLY_RETRY_LIMIT = 1
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    m.open_stream = lambda messages, **kw: iter([
        _FakeChunk(_FakeDelta(reasoning="I should answer soon but never emit content."))
    ])
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_FORCE_FINAL, f"expected forced final, got {status!r}"
        assert messages[-1]["role"] == "user"
        assert "complete visible answer" in messages[-1]["content"]
        # Reasoning capture (REASONING_ONLY_CHAR_LIMIT default) legitimately adds
        # ONE assistant message holding the preserved reasoning text — that's not
        # the empty/vacuous assistant turn this test guards against. What must
        # never happen is an empty assistant message sneaking in alongside it.
        assistant_msgs = [msg for msg in messages if msg.get("role") == "assistant"]
        assert len(assistant_msgs) == 1, f"expected exactly one captured-reasoning assistant message, got {len(assistant_msgs)}"
        assert "answer soon" in assistant_msgs[0]["content"]
        assert all((msg.get("content") or "").strip() for msg in assistant_msgs), \
            "no empty assistant message should be appended"
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_CHAR_LIMIT = saved_limit
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — empty reasoning-only turns force final without empty assistant")


def test_repl_recover_command_forces_visible_checkpoint():
    saved_read_multiline = m.read_multiline
    saved_model_turn = m.model_turn
    saved_banner = m._banner
    m._banner = lambda: ""
    builtins.print = lambda *a, **k: None
    calls = []

    def fake_model_turn(messages, reasoning_loop_cut_count=0, malformed_stream_cut_count=0,
                        empty_turn_count=0, forced_final=False, recovery_sampling=False,
                        malformed_call_count=0):
        calls.append({
            "forced_final": forced_final,
            "recovery_sampling": recovery_sampling,
            "last_message": messages[-1]["content"],
        })
        return m.TURN_DONE

    prompts = iter(["/recover save the audio diagnosis", "/quit"])
    m.read_multiline = lambda history=None, initial=None: next(prompts)
    m.model_turn = fake_model_turn
    try:
        m.main()
        assert len(calls) == 1, calls
        assert calls[0]["forced_final"] is True
        assert calls[0]["recovery_sampling"] is True
        assert "Manual recovery requested" in calls[0]["last_message"]
        assert "save the audio diagnosis" in calls[0]["last_message"]
    finally:
        m.read_multiline = saved_read_multiline
        m.model_turn = saved_model_turn
        m._banner = saved_banner
        builtins.print = _REAL_PRINT
    print("PASS — /recover forces a low-temp visible checkpoint")


# ---------------------------------------------------------------------------
# 9. Text-only after stall nudge: real-work summary is accepted, not re-nudged
# ---------------------------------------------------------------------------
def test_text_only_after_stall_accepts_final_summary_after_toolwork():
    # Regression for a too-aggressive "TEXT-ONLY AFTER STALL NUDGE". Once a
    # reasoning-stall nudge has fired, the stall counter is kept across later
    # tool calls (it does not reset — see _run_model_turn_loop). So after the
    # model DOES act (a tool call lands, leaving a `tool` result as the last
    # message) and then returns a text-only final summary, the old guard saw
    # reasoning_loop_cut_count > 0 and nudged anyway — rejecting a correct
    # finished answer and wasting turns. The last-message-is-tool-result check
    # must suppress the nudge here: the summary is a real final answer.
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_RETRY_LIMIT = 3
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    # A text-only reply (content, no tool calls): the model finished and is
    # summarizing completed work.
    chunks = [_FakeChunk(_FakeDelta(content="All done — the fix is in."), finish_reason="stop")]
    m.open_stream = lambda messages, **kw: iter(chunks)
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"},
                    # A stall nudge fired earlier in the conversation ...
                    {"role": "assistant", "content": "[stalled reasoning]"},
                    {"role": "user", "content": "[Runtime note: nudge to act]"},
                    # ... the model then acted (tool call) ...
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "content": "result of the action"}]
        # reasoning_loop_cut_count=1 > 0 (a stall already happened), under the
        # limit of 3 — exactly the condition that used to spuriously nudge.
        status = m.model_turn(messages, reasoning_loop_cut_count=1)
        assert status == m.TURN_DONE, f"expected TURN_DONE, got {status!r}"
        # No second nudge appended: the last message stays the tool result,
        # and the model's summary is preserved as the assistant turn.
        assert messages[-1] == {"role": "assistant", "content": "All done — the fix is in."}
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — text-only summary after toolwork accepted, not re-nudged")


def test_text_only_after_stall_nudges_when_nudge_is_last_message():
    # The legitimate case the guard was written for: the model stalls, gets
    # nudged, and replies with a plain restatement of intent ("Now I'll
    # implement...") and STILL no tool call. The last message IS the nudge
    # (a user/runtime-note), so the guard must still fire and re-nudge.
    saved_open_stream = m.open_stream
    saved_watcher = m._interrupt_watcher
    saved_spinner = m.LifeSpinner
    saved_retries = m.REASONING_ONLY_RETRY_LIMIT
    m._interrupt_watcher = lambda: None
    m.REASONING_ONLY_RETRY_LIMIT = 3
    m.LifeSpinner = type("NoSpinner", (), {
        "__init__": lambda self, **kw: None,
        "start": lambda self: None,
        "stop": lambda self: None,
        "_t": None,
    })
    builtins.print = lambda *a, **k: None
    chunks = [_FakeChunk(_FakeDelta(content="Now I'll write the file next."), finish_reason="stop")]
    m.open_stream = lambda messages, **kw: iter(chunks)
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"},
                    {"role": "assistant", "content": "[stalled reasoning]"},
                    {"role": "user", "content": "[Runtime note: nudge to act]"}]
        status = m.model_turn(messages, reasoning_loop_cut_count=1)
        assert status == m.TURN_STALL, f"expected TURN_STALL, got {status!r}"
        # The intent text is preserved as the assistant turn (now last), and
        # _nudge_current_user_turn rewrites the prior user message with a fresh
        # re-nudge (it edits in place rather than appending a new message).
        assert messages[-1] == {"role": "assistant", "content": "Now I'll write the file next."}
        assert messages[-2]["role"] == "user"
        assert "Runtime note" in messages[-2]["content"]
    finally:
        m.open_stream = saved_open_stream
        m._interrupt_watcher = saved_watcher
        m.LifeSpinner = saved_spinner
        m.REASONING_ONLY_RETRY_LIMIT = saved_retries
        builtins.print = _REAL_PRINT
    print("PASS — text-only intent after nudge (no tool) still re-nudged")


if __name__ == "__main__":
    test_confirm_esc_raises()
    test_confirm_y_n()
    test_run_tool_propagates_esc()
    test_model_turn_native_esc()
    test_model_turn_malformed_native_tool_args_retries_cleanly()
    test_repl_breaks_on_esc()
    test_reasoning_only_stall_requests_forced_final()
    test_forced_final_tool_call_becomes_assistant_text()
    test_forced_final_length_marks_partial_answer()
    test_reasoning_only_empty_completion_forces_final_without_empty_assistant()
    test_repl_recover_command_forces_visible_checkpoint()
    test_text_only_after_stall_accepts_final_summary_after_toolwork()
    test_text_only_after_stall_nudges_when_nudge_is_last_message()
    print("\nALL ESC-APPROVAL TESTS PASSED")
