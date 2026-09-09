"""ROADMAP P3-2 (0.51.0): a mid-stream timeout is an honest failure.

Contract under test:
- model_turn() no longer returns TURN_DONE on a mid-stream httpx
  TimeoutException: with retry ladder left it returns TURN_TIMED_OUT_RETRY,
  and with the ladder exhausted it returns TURN_TIMED_OUT.
- _run_model_turn_loop() applies the step-level retry at the step boundary:
  bounded count (NX_TURN_RETRIES, default = the HTTP layer's), re-entering
  the SAME request (nothing appended to `messages`), with backoff between
  attempts, and a success on retry is a normal done turn.
- Ladder exhaustion maps to a status-"error" summary — headless callers
  (batch/one-shot) exit 4, never a clean pass on truncated output.
- NX_TURN_RETRIES=0 disables the ladder (pre-0.51 behavior, but honest).
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NX_TURN_RETRIES", "2")
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    for k in ("NX_HOME", "NX_SESSIONS_DIR"):
        os.environ.pop(k, None)


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
    import io
    import sys as _sys
    monkeypatch.setattr(_sys, "stdout", io.StringIO())
    monkeypatch.setattr(_sys, "stderr", io.StringIO())


class _DyingStream:
    """A stream object that opens and then raises TimeoutException mid-stream
    (the shape `_ChatStream`'s un-retried read errors produce)."""
    dropped_lines = 0

    def __init__(self, times: int):
        self.times = times

    def __iter__(self):
        for _ in range(self.times):
            yield object()  # a chunk that the parser tolerates/drops
        import httpx
        raise httpx.ReadTimeout("timed out")

    def close(self):
        pass


def test_midstream_timeout_is_not_turn_done(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: _DyingStream(0))
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    status = nx.model_turn(messages)
    assert status == nx._turn.TURN_TIMED_OUT_RETRY


def test_ladder_exhausted_returns_timed_out_not_done(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: _DyingStream(0))
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    status = nx.model_turn(messages, step_retries=2)
    assert status == nx._turn.TURN_TIMED_OUT


def test_loop_retries_same_request_then_succeeds(nx_fresh, monkeypatch):
    """A timeout mid-task retries the SAME request (nothing appended) and a
    success on a retry is a normal done turn."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx._turn, "_step_backoff_delay", lambda a, r: 0.0)
    calls = {"n": 0}
    appends = {"n": 0}
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    base_len = len(messages)

    class _Counting(list):
        def append(self, item):
            appends["n"] += 1
            list.append(self, item)

    counting = _Counting(messages)

    def turn_fn(messages_arg, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return nx._turn.TURN_TIMED_OUT_RETRY
        return nx._turn.TURN_DONE  # second attempt: clean completion

    monkeypatch.setattr(nx, "model_turn", turn_fn)
    summary = nx._run_model_turn_loop(counting, max_turns_override=5)
    assert summary["status"] == "done"
    assert calls["n"] == 2
    # only the initial user message was appended — the retry replayed the
    # same request rather than adding recovery turns
    assert appends["n"] == 0
    assert len(counting) == base_len


def test_loop_exhaustion_is_error_summary(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx._turn, "_step_backoff_delay", lambda a, r: 0.0)
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: _DyingStream(0))
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=5)
    assert summary["status"] == "error"
    assert "timeout" in summary["error"]


def test_batch_scores_timeout_as_error_not_done(nx_fresh, monkeypatch, tmp_path):
    nx, _ = nx_fresh
    _install_source(nx)
    import io
    import json
    import sys as _sys
    out, err = io.StringIO(), io.StringIO()
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    monkeypatch.setattr(_sys, "stdout", out)
    monkeypatch.setattr(_sys, "stderr", err)
    monkeypatch.setattr(nx._turn, "_step_backoff_delay", lambda a, r: 0.0)
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: _DyingStream(0))
    f = tmp_path / "tasks.jsonl"
    f.write_text('{"instruction": "score me", "id": "t1"}\n', encoding="utf-8")
    from nx._batch import run_batch, EXIT_ERROR
    code = run_batch(["-f", str(f)])
    assert code == EXIT_ERROR, "a mid-stream timeout must not summarize as OK"
    line = json.loads(out.getvalue().strip())
    assert line["status"] == "error"
    assert "timeout" in line["error"]


def test_nx_turn_retries_zero_disables_ladder(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setenv("NX_TURN_RETRIES", "0")
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: _DyingStream(0))
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=5)
    assert summary["status"] == "error"
    assert "after 0 step retries" in summary["error"]


def test_step_ladder_defaults_track_http_layer(nx_fresh):
    nx, _ = nx_fresh
    from nx._openai_client import _HTTP_RETRIES_DEFAULT
    assert nx._turn._step_retry_limit() == _HTTP_RETRIES_DEFAULT
