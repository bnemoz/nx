#!/usr/bin/env python3
"""finish_reason="length" auto-continuation.

qwen3.8-flash-next (and any other model behind a per-response token cap)
regularly ends its turn with finish_reason="length": the reply stops
mid-sentence and the old behavior just dropped the user back to the prompt
with a truncated answer in the transcript. Now the turn loop auto-continues:
the partial text becomes a real assistant turn, a short resume note is
appended, and the stream re-opens — budgeted by st.MAX_CONTINUATIONS
(NX_MAX_CONTINUATIONS), after which the partial is closed out with the
truncation marker.

Standalone-runnable (python3 tests/test_length_continuation.py) as well as
pytest-collectable, mirroring the test_esc_approval.py harness style:
open_stream and friends are monkeypatched so the tests are offline.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
os.environ.setdefault("NX_INTERRUPT_RESTORE_WAIT", "0.05")
for k in ("NX_SOURCES", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_ACTIVE",
          "NX_MAX_CONTINUATIONS"):
    os.environ.pop(k, None)

import builtins

import nx as m  # noqa: E402  (needs sys.path set up first)
import nx._state as _st  # noqa: E402
import nx._turn as _turn  # noqa: E402  (patch points for model_turn's globals)


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


class _NoSpinner:
    def __init__(self, *a, **kw):
        self._t = None

    def start(self):
        pass

    def stop(self):
        pass


def _save() -> dict:
    """Capture the attributes each test monkeypatches, for _restore().

    model_turn() resolves LifeSpinner / _ensure_interrupt_watcher in
    nx._turn's module globals, so those are patched there — patching them on
    the nx package module would be a silent no-op."""
    return {"open_stream": m.open_stream,
            "spinner": _turn.LifeSpinner,
            "ensure": _turn._ensure_interrupt_watcher,
            "print": builtins.print}


def _patch_harness() -> None:
    _turn._ensure_interrupt_watcher = lambda: None
    _turn.LifeSpinner = _NoSpinner
    builtins.print = lambda *a, **k: None


def _restore(saved: dict) -> None:
    m.open_stream = saved["open_stream"]
    _turn.LifeSpinner = saved["spinner"]
    _turn._ensure_interrupt_watcher = saved["ensure"]
    builtins.print = saved["print"]


# --- 1. a normal text turn cut by length returns TURN_LENGTH_CUT --------------
def test_length_cut_returns_length_cut_status():
    saved = _save()

    def fake_open_stream(messages, **kw):
        return iter([_FakeChunk(_FakeDelta(content="The answer starts here and then"),
                                finish_reason="length")])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages)
        assert status == m.TURN_LENGTH_CUT, f"expected TURN_LENGTH_CUT, got {status!r}"
        # model_turn appended NOTHING: the partial is handed to the loop via
        # st._LAST_PARTIAL_TEXT and the loop owns both continue and close-out.
        assert not any(msg.get("role") == "assistant" for msg in messages)
        assert _st._LAST_PARTIAL_TEXT == "The answer starts here and then"
    finally:
        _st._LAST_PARTIAL_TEXT = ""
        _restore(saved)
    print("PASS — length-cut text turn returns TURN_LENGTH_CUT with nothing appended")


# --- 2. the loop continues: partial kept, resume note appended, TURN_DONE -----
def test_loop_continues_and_finishes():
    saved = _save()
    calls = {"n": 0}

    def fake_open_stream(messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return iter([_FakeChunk(_FakeDelta(content="Part one, cut mid-"),
                                    finish_reason="length")])
        return iter([_FakeChunk(_FakeDelta(content="sentence. All done."),
                                finish_reason="stop")])

    m.open_stream = fake_open_stream
    _patch_harness()
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        summary = m._run_model_turn_loop(messages)
        assert summary["status"] == "done", summary
        assert calls["n"] == 2, "the stream should re-open exactly once"
        roles = [msg["role"] for msg in messages]
        # The partial becomes a real assistant turn and the resume note is its
        # OWN trailing user message (0.40, append-only): the user's prompt is
        # never rewritten, so everything the model already saw — including the
        # server's prefix cache — survives a continuation.
        assert roles == ["system", "user", "assistant", "user", "assistant"], roles
        assert messages[1] == {"role": "user", "content": "answer me"}
        assert messages[2]["content"] == "Part one, cut mid-"
        assert messages[3]["content"].startswith("[Runtime note: ")
        assert "token limit" in messages[3]["content"], messages[3]["content"]
        assert messages[-1]["content"] == "sentence. All done."
    finally:
        _restore(saved)
    print("PASS — loop continues a length-cut answer and finishes seamlessly")


# --- 3. exhausted continuation budget closes out with the truncation marker ---
def test_continuation_budget_close_out():
    saved = _save()
    calls = {"n": 0}

    def fake_open_stream(messages, **kw):
        calls["n"] += 1
        # every turn is cut by the cap — the budget must exhaust, not spin
        return iter([_FakeChunk(_FakeDelta(content=f"chunk {calls['n']} "),
                                finish_reason="length")])

    m.open_stream = fake_open_stream
    _patch_harness()
    saved_limit = m.MAX_CONTINUATIONS
    m.MAX_CONTINUATIONS = 2
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        summary = m._run_model_turn_loop(messages)
        assert summary["status"] == "done", summary
        # initial cut + 2 continuations, then the close-out (no extra open)
        assert calls["n"] == 3, calls
        roles = [msg["role"] for msg in messages]
        assert roles[-1] == "assistant"
        assert "chunk 3" in messages[-1]["content"]
        assert "[Truncated by token limit before completion.]" in messages[-1]["content"]
        # each continuation appends a partial assistant turn plus its own resume
        # note (0.40), and the close-out lands after the last note — nothing
        # ends on a dangling note waiting for a reply that never comes
        assert roles == ["system", "user"] + ["assistant", "user"] * 2 + ["assistant"], roles
    finally:
        m.MAX_CONTINUATIONS = saved_limit
        _restore(saved)
    print("PASS — exhausted continuation budget closes out with truncation marker")


# --- 4. MAX_CONTINUATIONS=0 disables the auto-continue -------------------------
def test_zero_budget_disables_continuation():
    saved = _save()
    calls = {"n": 0}

    def fake_open_stream(messages, **kw):
        calls["n"] += 1
        return iter([_FakeChunk(_FakeDelta(content="cut answer "),
                                finish_reason="length")])

    m.open_stream = fake_open_stream
    _patch_harness()
    saved_limit = m.MAX_CONTINUATIONS
    m.MAX_CONTINUATIONS = 0
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        summary = m._run_model_turn_loop(messages)
        assert summary["status"] == "done", summary
        assert calls["n"] == 1, "no continuation should be attempted"
        assert "[Truncated by token limit before completion.]" in messages[-1]["content"]
    finally:
        m.MAX_CONTINUATIONS = saved_limit
        _restore(saved)
    print("PASS — MAX_CONTINUATIONS=0 keeps the old single-turn behavior")


# --- 5. a forced-final cut still saves the partial with the marker -------------
def test_forced_final_cut_unchanged():
    saved = _save()
    args_seen = {}

    def fake_open_stream(messages, **kw):
        args_seen.update(kw)
        return iter([_FakeChunk(_FakeDelta(content="partial forced answer "),
                                finish_reason="length")])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "answer me"}]
        status = m.model_turn(messages, forced_final=True)
        assert status == m.TURN_DONE, status
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].startswith("partial forced answer")
        assert "[Truncated by token limit before completion.]" in messages[-1]["content"]
    finally:
        _restore(saved)
    print("PASS — forced-final cut still saves the partial with the truncation marker")


# --- 6. a length-cut turn WITH a (parseable) native tool call still runs it ----
def test_tool_calls_win_over_length_cut():
    saved = _save()

    def fake_open_stream(messages, **kw):
        tc = type("TC", (), {"index": 0, "id": "call_1",
                             "function": type("F", (), {
                                 "name": "read_file",
                                 "arguments": '{"path": "x.py"}'})()})()
        return iter([_FakeChunk(_FakeDelta(tool_calls=[tc]), finish_reason="length")])

    m.open_stream = fake_open_stream
    try:
        messages = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "read it"}]
        status = m.model_turn(messages)
        assert status == m.TURN_TOOL, f"expected TURN_TOOL, got {status!r}"
        # the tool ran: an assistant(tool_calls) turn plus a tool result landed
        assert any(msg.get("role") == "tool" for msg in messages)
    finally:
        _restore(saved)
    print("PASS — native tool call with finish_reason=length still executes")


if __name__ == "__main__":
    test_length_cut_returns_length_cut_status()
    test_loop_continues_and_finishes()
    test_continuation_budget_close_out()
    test_zero_budget_disables_continuation()
    test_forced_final_cut_unchanged()
    test_tool_calls_win_over_length_cut()
    print("ALL PASS")
