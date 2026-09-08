#!/usr/bin/env python3
"""0.40 (ROADMAP 1.1): recovery nudges are prompt-cache-safe.

`_nudge_current_user_turn` used to inject every recovery nudge by REWRITING the
last user turn in place (strip the old `[Runtime note: …]`, re-attach a new
one). That message is one the model has already been sent, so each of the ten
recovery call sites invalidated the server's KV/prefix cache from that point on
— on precisely the turns with the largest prompt, in the exact sessions that
retry the most. This file pins the replacement contract:

  • a nudge only ever APPENDS a trailing user message;
  • no message older than the append is ever mutated;
  • a nudge is recognized as an injected note, not a conversation turn
    (`_last_is_dangling_tool`, and compression's verbatim keep-window).

Standalone-runnable (python3 tests/test_recovery_cache_safety.py) as well as
pytest-collectable; harness style mirrors test_length_continuation.py.
"""
import builtins
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("NX_SKIP_STARTUP_PROBES", "1")
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
os.environ.setdefault("NX_INTERRUPT_RESTORE_WAIT", "0.05")
for k in ("NX_SOURCES", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_ACTIVE"):
    os.environ.pop(k, None)

import copy  # noqa: E402

import nx as m  # noqa: E402  (needs sys.path set up first)
import nx._turn as _turn  # noqa: E402  (real patch points for model_turn's globals)


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


class _FakeTC:
    def __init__(self, index, call_id, name, args):
        self.index = index
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": args})()
        self.model_extra = {}


class _NoSpinner:
    def __init__(self, *a, **kw):
        self._t = None

    def start(self):
        pass

    def stop(self):
        pass


def _save() -> dict:
    return {"open_stream": m.open_stream,
            "run_tool": m.run_tool,
            "watcher": _turn._ensure_interrupt_watcher,
            "spinner": _turn.LifeSpinner,
            "retry": m.REASONING_ONLY_RETRY_LIMIT,
            "print": builtins.print}


def _patch_harness() -> None:
    _turn._ensure_interrupt_watcher = lambda: None
    _turn.LifeSpinner = _NoSpinner
    builtins.print = lambda *a, **k: None


def _restore(saved: dict) -> None:
    m.open_stream = saved["open_stream"]
    m.run_tool = saved["run_tool"]
    _turn._ensure_interrupt_watcher = saved["watcher"]
    _turn.LifeSpinner = saved["spinner"]
    m.REASONING_ONLY_RETRY_LIMIT = saved["retry"]
    builtins.print = saved["print"]


def _snapshot(messages):
    """Deep copy so a later mutation of any earlier message is detectable."""
    return copy.deepcopy(messages)


# --- the primitive -------------------------------------------------------------
def test_nudge_appends_and_mutates_nothing_older():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "the real prompt"},
            {"role": "assistant", "content": "a turn"},
            {"role": "tool", "tool_call_id": "c1", "content": "observation"}]
    before = _snapshot(msgs)
    m._nudge_current_user_turn(msgs, "first note")
    assert msgs[:4] == before[:4], "an older message was mutated"
    assert msgs[-1] == {"role": "user", "content": "[Runtime note: first note]"}
    # the note goes AFTER the tool result (the layout the model must see next),
    # not smeared onto the user's prompt three messages back
    m._nudge_current_user_turn(msgs, "second note")
    assert msgs[-2]["content"] == "[Runtime note: first note]"
    assert msgs[-1]["content"] == "[Runtime note: second note]"
    assert msgs[1] == {"role": "user", "content": "the real prompt"}


def test_identical_pending_note_is_not_doubled():
    """A loop that re-enters the same rung with the same text must not stack a
    byte-identical note directly on top of itself."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "p"}]
    m._nudge_current_user_turn(msgs, "same")
    n = len(msgs)
    m._nudge_current_user_turn(msgs, "same")
    assert len(msgs) == n
    m._nudge_current_user_turn(msgs, "different")
    assert len(msgs) == n + 1


# --- the whole recovery stack, driven through the real turn loop --------------
_RECOVERY_STREAMS = {
    # family            stream                                          status
    "malformed_args": [(_FakeChunk(_FakeDelta(tool_calls=[
        _FakeTC(0, "c1", "write_file", '{"path":"a","content":"unterminated')]),
        finish_reason="length"), "cut")],
    "empty_turn":     [(_FakeChunk(_FakeDelta(content=""), None), "empty")],
    "reasoning_stall": [(_FakeChunk(_FakeDelta(reasoning="thinking hard about it"),
                                    None), "stall")],
}


def test_every_recovery_family_only_appends():
    """The regression ROADMAP 1.1 asks for: a nudge never mutates a message
    older than the turn boundary, for ANY recovery rung, not just the one whose
    test happened to be written."""
    for family, stream in _RECOVERY_STREAMS.items():
        saved = _save()
        _patch_harness()
        m.run_tool = lambda *a, **k: "ok"
        chunks = [c for c, _ in stream]
        m.open_stream = lambda messages, **kw: iter(list(chunks))
        try:
            msgs = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": "the user's prompt"},
                    {"role": "assistant", "content": "earlier assistant turn"},
                    {"role": "user", "content": "follow-up question"}]
            before = _snapshot(msgs)
            m.model_turn(msgs, reasoning_loop_cut_count=1,
                         malformed_stream_cut_count=0, empty_turn_count=0)
            assert msgs[:4] == before, f"{family}: recovery rewrote an older message"
            tail = [x for x in msgs[4:]]
            assert tail, f"{family}: expected the nudge to be appended"
            notes = [x for x in tail
                     if x.get("role") == "user"
                     and str(x.get("content", "")).startswith("[Runtime note: ")]
            assert notes, f"{family}: expected a [Runtime note:] tail: {tail!r}"
            assert msgs[-1] in tail
        finally:
            _restore(saved)


def test_dangling_tool_detection_survives_a_trailing_note():
    """0.40 layout detail: a flushed notice or nudge now sits AFTER a tool
    result, so 'does a tool result await a follow-up' has to look past it."""
    base = [{"role": "system", "content": "s"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "output"}]
    assert m._last_is_dangling_tool(base)
    with_note = base + [{"role": "user", "content": "[Runtime note: backgrounded job finished]"}]
    assert m._last_is_dangling_tool(with_note), "a note must not mask a dangling tool result"
    answered = with_note + [{"role": "assistant", "content": "here is the answer"}]
    assert not m._last_is_dangling_tool(answered)
    # and a note is only a note when it IS one: a real user prompt still ends
    # the dangling state
    real_user = base + [{"role": "user", "content": "thanks, that's enough"}]
    assert not m._last_is_dangling_tool(real_user)


# --- compression keeps its real turns ----------------------------------------
def test_compression_keep_window_ignores_trailing_notes(monkeypatch, capsys):
    """`keep` counts MESSAGES. With nudges as their own user messages, two
    trailing notes would otherwise consume the entire verbatim window of a
    manual /compress and fold away the last real tool result."""
    from nx import _compress

    calls = {}

    class _FakeCompletions:
        def __init__(self):
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            return type("R", (), {"choices": [type("C", (), {
                "message": type("M", (), {"content": "fake summary"})()})()]})()

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

    def fake_open_stream(messages, **kw):
        calls["messages"] = list(messages)
        return iter([_FakeChunk(_FakeDelta(content="SUMMARY"), finish_reason="stop")])

    monkeypatch.setattr(_compress, "open_stream", fake_open_stream)
    monkeypatch.setattr(m._state, "client", _FakeClient())
    monkeypatch.setattr(m._state, "MODEL", "fake-model")
    monkeypatch.setattr(m._state, "COMPRESS_KEEP", 2)

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(4):
        msgs.append({"role": "user", "content": f"prompt {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    msgs.append({"role": "user", "content": "[Runtime note: nudge one]"})
    msgs.append({"role": "user", "content": "[Runtime note: nudge two]"})
    before = _snapshot(msgs)

    out = _compress.compress(msgs)
    assert out is not None
    kept_n, summarized_n, _ = out
    contents = [str(x.get("content", "")) for x in msgs]
    # The keep=2 window must hold two REAL turns. Without the note-aware keep,
    # the tail is exactly the two notes and `answer 3` — the model's last real
    # output — gets folded into the summary.
    assert kept_n == 4, f"keep should have widened past the 2 notes: {kept_n} / {contents}"
    assert summarized_n == 6, summarized_n
    assert "answer 3" in contents, f"last real answer folded away: {contents}"
    assert contents[-2:] == ["[Runtime note: nudge one]", "[Runtime note: nudge two]"]


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
