#!/usr/bin/env python3
"""Context-window overflow is recovered inside the turn loop, not mislabeled "done".

The gap this pins down: auto-compression used to run only between user turns
(nx/_repl.py), but _run_model_turn_loop() can append up to MAX_MODEL_TURNS
assistant+tool messages inside ONE turn loop. The first request that no longer
fits returned a context-length 400, open_stream() returned None, and
model_turn() scored it TURN_DONE — headless/batch runs reported a long task as
successfully completed with (near-)nothing done.

Contract under test:
- _is_context_overflow_error(): only an HTTP-400 exception whose message/body
  matches the over-limit wording; anything else (500, connection error,
  context-ish wording on another status) is NOT an overrun.
- open_stream() latches st._LAST_CONTEXT_OVERRUN (verbatim server text) and
  returns None WITHOUT latching st._LAST_API_ERROR for such a 400.
- model_turn() maps that to TURN_CONTEXT_OVERRUN — never TURN_DONE — and the
  latch is cleared before every open (a stale overrun can't poison the next).
- _run_model_turn_loop() folds the older turns (nx._compress_for_overrun →
  compress(auto=True)) and re-runs the turn; a summary turn replaces the
  folded history on the same `messages` list; a second refusal reports
  {"status": "error"} with a context-overflow message, not "done"/"cap".
- Cheap in-loop guard: a fresh above-threshold prompt-token reading triggers
  _maybe_autocompress before the next request; a stale (unchanged) reading
  does not fire fold after fold.
- The turn-cap rescue turn gets the same one fold + retry; double refusal is
  an error, not a cap.
- Sub-agent quiet posture: no banner lines (assertions run against stdout).

Offline: the OpenAI client is faked; compress() is stubbed.
"""
import importlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)


class _Overrun400(Exception):
    """Shape of the APIError nx's own client raises for a context-length 400:
    non-2xx, .status_code=400, .message/.body carry the server's wording."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message
        self.body = {"error": {"message": message}}
        self.status_code = 400


class _PlainError(Exception):
    pass


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    for k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(k, None)
    nx.st._LAST_API_ERROR = None
    nx.st._LAST_CONTEXT_OVERRUN = None
    importlib.reload(nx)


def _install_source(nx):
    from nx import _sources as _sources_mod
    src = _sources_mod.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    nx.st.SOURCES["fake"] = src
    nx.st.SOURCE_ORDER.append("fake")
    nx.st.ACTIVE = src
    return src


def _quiet(monkeypatch):
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def _overrun_client(monkeypatch, message):
    """A client whose chat.completions.create always raises a context-400."""
    def boom(*a, **k):
        raise _Overrun400(message)
    cp = type("Cp", (), {"create": staticmethod(boom)})
    ch = type("Ch", (), {"completions": cp()})
    return type("C", (), {"chat": ch()})(), "fake-model"


# --- overrun detection ---------------------------------------------------------

def test_overrun_detector_matches_400_wording_only(nx_fresh):
    nx, _ = nx_fresh
    from nx._compress import _is_context_overflow_error
    yes = [
        "Error code: 400 - This model's maximum context length is 8192 tokens, "
        "however you requested 9000 tokens",
        "Error code: 400 - {'error': {'message': \"This model's maximum context "
        "length is 4096 tokens\"}}",
        "Error code: 400 - context length exceeded",
        "Error code: 400 - Please reduce the length of the messages",
    ]
    for m in yes:
        assert _is_context_overflow_error(_Overrun400(m)), m
    no = [
        _PlainError("context length exceeded"),          # no status_code
        _Overrun400("Error code: 400 - invalid api key"),  # 400, wrong wording
        type("E500", (Exception,), {})("server on fire"),  # no attrs at all
    ]
    class _E500(_Overrun400):
        def __init__(self):
            super().__init__("maximum context length is 8 tokens")
            self.status_code = 500
    no.append(_E500())  # right wording, wrong status
    for e in no:
        assert not _is_context_overflow_error(e), e


# --- open_stream latch -----------------------------------------------------------

def test_open_stream_latches_overrun_not_api_error(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    client, _model = _overrun_client(
        monkeypatch, "This model's maximum context length is 8192 tokens")
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    _quiet(monkeypatch)
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is None
    assert nx.st._LAST_CONTEXT_OVERRUN, "overrun must be latched verbatim"
    assert "maximum context length" in nx.st._LAST_CONTEXT_OVERRUN
    assert nx.st._LAST_API_ERROR is None, \
        "an over-length prompt is recoverable — not the terminal API-failure path"


def test_model_turn_maps_overrun_never_to_done(nx_fresh, monkeypatch):
    """The core mislabel: an over-length prompt must not be scored TURN_DONE."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    client, _model = _overrun_client(monkeypatch, "maximum context length is 8192 tokens")
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    assert nx.model_turn(messages) == nx._turn.TURN_CONTEXT_OVERRUN
    assert nx._turn.TURN_CONTEXT_OVERRUN not in (nx._turn.TURN_DONE, nx._turn.TURN_EMPTY)


def test_stale_overrun_latch_cleared_before_every_open(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    nx.st._LAST_CONTEXT_OVERRUN = "stale from an earlier turn"
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: iter([]))  # opens fine, yields nothing
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    status = nx.model_turn(messages)
    assert status != nx._turn.TURN_CONTEXT_OVERRUN
    assert status in (nx._turn.TURN_DONE, nx._turn.TURN_EMPTY)


# --- loop recovery: fold + retry once --------------------------------------------

def test_loop_folds_and_retries_after_overrun(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    n = {"i": 0}
    compress_calls = []

    def fake_compress(msgs, keep=None, auto=False):
        compress_calls.append(auto)
        assert auto is True, "overrun recovery must use the conservative auto fold"
        msgs[:] = [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "[Compressed context — summary]"}]
        return (2, 10, 42)

    monkeypatch.setattr(nx, "compress", fake_compress)

    def overrun_then_success(messages, *a, **k):
        n["i"] += 1
        if n["i"] == 1:
            nx.st._LAST_CONTEXT_OVERRUN = "maximum context length is 8192 tokens"
            return nx._turn.TURN_CONTEXT_OVERRUN
        # retry turn sees the folded history
        assert any("[Compressed context" in (m.get("content") or "") for m in messages), \
            "the retry must run against the folded messages"
        messages.append({"role": "assistant", "content": "All done."})
        return nx._turn.TURN_DONE

    monkeypatch.setattr(nx, "model_turn", overrun_then_success)
    messages = nx._initial_messages() + [
        {"role": "user", "content": f"turn {i}"} for i in range(10)]
    summary = nx._run_model_turn_loop(messages, max_turns_override=5)
    assert summary["status"] == "done", summary
    assert summary["turns"] == 2
    assert compress_calls == [True]


def test_loop_second_refusal_is_error_not_done(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)

    def always_overrun(messages, *a, **k):
        nx.st._LAST_CONTEXT_OVERRUN = "maximum context length is 8192 tokens"
        return nx._turn.TURN_CONTEXT_OVERRUN

    monkeypatch.setattr(nx, "model_turn", always_overrun)
    # compress cannot help (nothing foldable) → no retry, straight to error
    monkeypatch.setattr(nx, "compress", lambda *a, **k: None)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=5)
    assert summary["status"] == "error", summary
    assert "context window overflow" in summary["error"]
    assert "maximum context length" in summary["error"]
    assert summary["turns"] == 1  # terminal: no burn of the remaining cap


def test_loop_overrun_compresses_only_once(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    calls = {"n": 0}

    def fake_compress(msgs, keep=None, auto=False):
        calls["n"] += 1
        return None  # always refuses to fold

    monkeypatch.setattr(nx, "compress", fake_compress)
    monkeypatch.setattr(nx, "model_turn", lambda *a, **k: nx._turn.TURN_CONTEXT_OVERRUN)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=99)
    assert summary["status"] == "error"
    assert calls["n"] == 1, "exactly one fold attempt, then give up"


# --- cheap in-loop guard -----------------------------------------------------------

def test_in_loop_guard_compresses_when_above_threshold(nx_fresh, monkeypatch):
    """A fresh above-threshold reading mid-loop folds BEFORE the next request 400s."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    nx.st.AUTOCOMPRESS_PERCENT = 85

    class _Src:
        _context_window = 10000
        name = "fake"
        def resolve_context_window(self, model=None, force=False):
            return self._context_window
        def extra_request_kwargs(self):
            return {}

    nx.st.ACTIVE = _Src()
    n = {"i": 0}
    compress_calls = []

    def fake_compress(msgs, keep=None, auto=False):
        compress_calls.append(msgs and tuple(m.get("role") for m in msgs))
        msgs[:] = [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "[Compressed context — summary]"},
                   {"role": "assistant", "content": "ok"}]
        return (1, 2, 40)

    monkeypatch.setattr(nx, "compress", fake_compress)

    def tool_turns_growing_context(messages, *a, **k):
        n["i"] += 1
        if n["i"] == 1:
            nx.st._LAST_PROMPT_TOKENS = 8600  # 86% ≥ 85% — over threshold
        if n["i"] >= 4:
            messages.append({"role": "assistant", "content": "All done."})
            return nx._turn.TURN_DONE
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call-{n['i']}", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path": "x.py"}'}}]})
        messages.append({"role": "tool", "tool_call_id": f"call-{n['i']}",
                         "content": "x" * 1000})
        return nx._turn.TURN_TOOL

    monkeypatch.setattr(nx, "model_turn", tool_turns_growing_context)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=10)
    assert summary["status"] == "done"
    assert len(compress_calls) == 1, \
        f"guard should fold exactly once on the fresh over-threshold reading: {compress_calls}"


def test_in_loop_guard_does_not_fire_repeatedly_on_stale_reading(nx_fresh, monkeypatch):
    """A server reporting no usage leaves one stale number cached: it must not
    trigger fold after fold on every loop iteration."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    nx.st.AUTOCOMPRESS_PERCENT = 85
    nx.st._LAST_PROMPT_TOKENS = 99999  # over threshold, and stays there

    calls = {"n": 0}
    monkeypatch.setattr(nx, "compress",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or None)

    def tool_then_finish(messages, *a, **k):
        messages.append({"role": "tool", "tool_call_id": "c", "content": "y"})
        if len([m for m in messages if m.get("role") == "tool"]) >= 4:
            messages.append({"role": "assistant", "content": "done"})
            return nx._turn.TURN_DONE
        return nx._turn.TURN_TOOL

    monkeypatch.setattr(nx, "model_turn", tool_then_finish)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=6)
    assert summary["status"] == "done"
    assert calls["n"] <= 1, f"stale reading must not refire every iteration ({calls['n']} folds)"


# --- turn-cap rescue turn -----------------------------------------------------------

def test_cap_rescue_turn_overrun_is_error_not_cap(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx, "compress", lambda *a, **k: None)
    monkeypatch.setattr(nx, "model_turn", lambda *a, **k: nx._turn.TURN_CONTEXT_OVERRUN)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=1)
    assert summary["status"] == "error", \
        "a refused rescue answer must not be reported as a completed capped run"
    assert "context window overflow" in summary["error"]


def test_cap_rescue_turn_folds_and_answers(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    n = {"i": 0}

    def rescue_flow(messages, *a, **k):
        n["i"] += 1
        if n["i"] == 1:
            nx.st._LAST_CONTEXT_OVERRUN = "maximum context length is 8192 tokens"
            return nx._turn.TURN_CONTEXT_OVERRUN
        messages.append({"role": "assistant", "content": "final answer"})
        return nx._turn.TURN_DONE

    monkeypatch.setattr(nx, "model_turn", rescue_flow)
    monkeypatch.setattr(nx, "compress",
                        lambda msgs, keep=None, auto=False:
                        (msgs.__setitem__(slice(None), [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "[Compressed context]"}]),
                         (2, 4, 30))[1])
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=1)
    assert summary["status"] == "cap", summary
    assert n["i"] == 2


# --- user-visible quiet contract ------------------------------------------------

def test_quiet_posture_prints_no_overrun_banner(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    out, err = _quiet(monkeypatch)
    monkeypatch.setattr(nx, "compress", lambda *a, **k: None)
    monkeypatch.setattr(nx, "model_turn", lambda *a, **k: nx._turn.TURN_CONTEXT_OVERRUN)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    nx._run_model_turn_loop(messages, max_turns_override=3)
    assert "✂" not in out.getvalue(), "batch/sub-agent stdout must stay clean"
    assert out.getvalue() == ""
