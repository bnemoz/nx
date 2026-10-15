#!/usr/bin/env python3
"""Two-tier overflow recovery + window-aware max_tokens (0.41, roadmap 1.8).

Three behaviors pinned here that 0.40 and earlier got wrong:

1. The fold ladder escalates: refusal #1 folds ~1/3 (tier 0), refusal #2
   halves the kept tail (tier 1, explicit keep), refusal #3 halves it again
   AND spill-evicts the largest tool results (tier 2) — instead of dying on
   the second refusal because the conservative keep survived the fold.
2. A fold whose kept tail STILL overflows no longer burns the whole run:
   the next tier can fold what the previous tier couldn't (compress returns
   None at a big keep, then succeeds at a smaller one).
3. Request max_tokens is clamped to window - prompt_estimate - margin when
   the source's context window is known, instead of a flat 16000 that both
   manufactures finish_reason=length cuts on small-window models and
   triggers the very 400 the fold machinery recovers from.
"""
import io
import os
import sys
import tempfile
import urllib.error  # noqa: F401  (import guard parity with sibling tests)


_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _compress as cp  # noqa: E402


def _msgs(n_tool=6, tool_chars=6000):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_tool):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {
                "name": "run_bash", "arguments": f'{{"command": "ls {i}"}}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "x" * tool_chars})
    return msgs


# --- fold tier escalation ---------------------------------------------------------

def test_tier0_keeps_auto_third(monkeypatch):
    calls = []

    def fake_compress(msgs, keep=None, auto=False):
        calls.append((keep, auto))
        return (2, 4, 30)

    monkeypatch.setattr(m, "compress", fake_compress)
    assert cp._compress_for_overrun(_msgs(), tier=0) is True
    assert calls == [(None, True)], "tier 0 = the conservative auto fold"


def test_tier1_pins_halved_keep(monkeypatch):
    calls = []

    def fake_compress(msgs, keep=None, auto=False):
        calls.append((keep, auto))
        return (2, 4, 30)

    monkeypatch.setattr(m, "compress", fake_compress)
    msgs = _msgs(n_tool=9)
    assert cp._compress_for_overrun(msgs, tier=1) is True
    keep, auto = calls[0]
    assert auto is True
    # 27 body messages → tier 1 keeps 27//6 = 4, not the auto body//3 = 9
    body_len = len(msgs) - 1
    assert keep == max(m.st.OVERFLOW_FOLD_KEEP_FLOOR,
                       body_len // (3 * (2 ** 1))), keep


def test_tier2_spill_evicts_largest_tool_results(monkeypatch):
    monkeypatch.setattr(m, "compress", lambda *a, **k: None)  # nothing foldable
    msgs = _msgs(n_tool=4, tool_chars=9000)
    assert cp._compress_for_overrun(msgs, tier=2) is True
    evicted = [mm for mm in msgs
               if mm.get("role") == "tool" and "saved to:" in mm["content"]]
    assert evicted, "tier 2 must spill the big tool results"
    assert len(evicted) <= m.st.OVERFLOW_EVICT_MAX
    # replace-in-place: the tool messages must still sit beside their calls
    roles = [mm.get("role") for mm in msgs]
    assert roles.count("tool") == 4, "messages are replaced, never deleted"
    for mm in evicted:
        path = mm["content"].rsplit("saved to: ", 1)[1].split(" —", 1)[0].strip()
        assert os.path.exists(path)
        with open(path) as f:
            assert "x" * 100 in f.read(), "spill holds the FULL text"


def test_eviction_skips_pinned_and_already_spilled(monkeypatch):
    monkeypatch.setattr(m, "compress", lambda *a, **k: None)
    msgs = _msgs(n_tool=3, tool_chars=9000)
    m.st._CTX_PINNED = [3]  # first tool result's index is pinned
    try:
        cp._evict_largest_tool_results(msgs)
        assert "saved to:" not in msgs[3]["content"], "pinned turn survives"
        assert "saved to:" in msgs[6]["content"]
        # already-spilled bodies are not re-evicted
        before = msgs[6]["content"]
        cp._evict_largest_tool_results(msgs)
        assert msgs[6]["content"] == before
    finally:
        m.st._CTX_PINNED = []


def test_eviction_respects_min_chars_floor():
    msgs = _msgs(n_tool=2, tool_chars=200)  # below OVERFLOW_EVICT_MIN_CHARS
    assert cp._evict_largest_tool_results(msgs) == 0
    assert all("saved to:" not in (mm.get("content") or "") for mm in msgs)


def test_second_refusal_folds_at_smaller_keep(monkeypatch):
    """The exact one-shot death 0.40 had: fold succeeds, retry 400s again,
    run dies. Now tier 1 (halved keep) folds what tier 0 couldn't."""
    nx, _ = _fresh(m)
    _install_source(m)
    _quiet(monkeypatch)
    tiers = []

    def fake_compress(msgs, keep=None, auto=False):
        tiers.append(keep)
        if keep is None:  # tier 0: conservative keep — "nothing to fold"
            return None
        msgs[:] = [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "[Compressed context — summary]"}]
        return (2, 4, 30)

    monkeypatch.setattr(m, "compress", fake_compress)
    calls = {"i": 0}

    def overrun_then_success(messages, *a, **k):
        calls["i"] += 1
        if calls["i"] <= 2:
            m.st._LAST_CONTEXT_OVERRUN = "maximum context length is 8192 tokens"
            return m._turn.TURN_CONTEXT_OVERRUN
        messages.append({"role": "assistant", "content": "finally."})
        return m._turn.TURN_DONE

    monkeypatch.setattr(m, "model_turn", overrun_then_success)
    messages = m._initial_messages() + [
        {"role": "user", "content": f"turn {i}"} for i in range(10)]
    summary = m._run_model_turn_loop(messages, max_turns_override=8)
    assert summary["status"] == "done", summary
    assert None in tiers and any(k is not None for k in tiers), \
        "tier 1 must run with an explicit smaller keep after the tier-0 miss"


# --- max_tokens clamp ---------------------------------------------------------------

class _WindowSrc:
    def __init__(self, window):
        self._window = window

    def resolve_context_window(self, model=None, force=False):
        return self._window

    def extra_request_kwargs(self):
        return {}


def test_clamp_caps_to_window_minus_prompt_minus_margin(monkeypatch):
    m.st.ACTIVE = _WindowSrc(8000)
    msgs = [{"role": "user", "content": "x" * 8000}]  # nx/_tokens est
    monkeypatch.setattr(m.st, "TOKENIZE_PREFLIGHT", 0)
    from nx import _tokens as _tk
    est = _tk.estimate_tokens_from_chars(8000)
    clamped = cp._clamp_max_tokens(msgs, 16000)
    assert clamped == 8000 - est - m.st.TOKENS_MARGIN, clamped


def test_clamp_floor_keeps_request_well_formed(monkeypatch):
    m.st.ACTIVE = _WindowSrc(8000)
    msgs = [{"role": "user", "content": "x" * 40000}]  # ~10k tok est: room < 0
    monkeypatch.setattr(m.st, "TOKENIZE_PREFLIGHT", 0)
    assert cp._clamp_max_tokens(msgs, 16000) == max(1, m.st.MIN_COMPLETION_TOKENS)


def test_clamp_never_raises_above_requested(monkeypatch):
    m.st.ACTIVE = _WindowSrc(40000)
    msgs = [{"role": "user", "content": "hi"}]
    monkeypatch.setattr(m.st, "TOKENIZE_PREFLIGHT", 0)
    assert cp._clamp_max_tokens(msgs, 2048) == 2048


def test_clamp_noop_without_window(monkeypatch):
    m.st.ACTIVE = _WindowSrc(None)
    msgs = [{"role": "user", "content": "hi"}]
    assert cp._clamp_max_tokens(msgs, 16000) == 16000


def test_clamp_noop_without_source():
    m.st.ACTIVE = None
    assert cp._clamp_max_tokens([{"role": "user", "content": "hi"}], 16000) == 16000


def test_open_stream_applies_the_clamp(monkeypatch):
    m.st.ACTIVE = _WindowSrc(8000)
    m.st.MODEL = "fake-model"
    monkeypatch.setattr(m.st, "TOKENIZE_PREFLIGHT", 0)
    captured = {}

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter([])

    class _Resp:
        usage = None

    def fake_create(**k):
        captured.update(k)
        return _Stream()

    class _Cp:
        create = staticmethod(fake_create)

    class _Ch:
        completions = _Cp()

    class _Client:
        chat = _Ch()

    m.st.client = _Client()
    msgs = [{"role": "user", "content": "x" * 8000}]
    from nx import _tokens as _tk
    m.open_stream(msgs)
    assert captured["max_tokens"] == 8000 - _tk.estimate_tokens_from_chars(8000) - m.st.TOKENS_MARGIN


# --- helpers ------------------------------------------------------------------------

def _fresh(_m):
    return _m, None


def _install_source(m):
    from nx import _sources as _sources_mod
    src = _sources_mod.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    m.st.SOURCES["fake"] = src
    m.st.SOURCE_ORDER.append("fake")
    m.st.ACTIVE = src
    return src


def _quiet(monkeypatch):
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err
