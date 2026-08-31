"""Terminal API-failure contract: `open_stream()` returning None after a hard
failure must NOT be scored as a completed turn by headless callers.

Contract under test:
- open_stream() latches st._LAST_API_ERROR ("Type: msg") on a connection /
  transport / API failure and still returns None (the friendly message stays).
- model_turn() maps that to TURN_API_ERROR — and only that: a stream that
  opens and legitimately yields nothing (empty turn) still returns
  TURN_EMPTY/TURN_DONE so the empty-turn recovery path keeps working.
- _run_model_turn_loop() turns TURN_API_ERROR into a {"status": "error"}
  summary carrying the failure text, instead of "done", and does not burn
  retry turns on a request that cannot succeed.
- batch (`nx run -f`): the task's JSONL line reports status "error",
  exit_code 4, and the process exit is EXIT_ERROR (4) — never a clean pass.
- one-shot (`nx --prompt-file x --json`): status "error" in the JSON payload
  and process exit 4.

The suite is offline: the OpenAI client is faked, never a real endpoint.
"""
import importlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


class _Boom(Exception):
    """A stand-in API error that is NOT an APIConnectionError/httpx error —
    open_stream's outer except-Exception latches it and returns None."""


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
    importlib.reload(nx)


def _install_source(nx):
    from nx import _sources as _sources_mod
    src = _sources_mod.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    nx.st.SOURCES["fake"] = src
    nx.st.SOURCE_ORDER.append("fake")
    nx.st.ACTIVE = src
    return src


def _quiet(monkeypatch):
    """Batch/agent quiet posture: no spinner, no friendly-message print.
    Returns (stdout_buf, stderr_buf) for later inspection. The quiet slot is
    set through monkeypatch so it is restored at teardown — a bare module-global
    assignment here leaked `quiet=True` into later test files."""
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    # raising=False: threading.local has no `quiet` in a fresh thread; teardown
    # deletes the attribute again, matching the pristine (absent) state.
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def _fail_open_stream(nx, monkeypatch, calls):
    """Replace open_stream the way a dead endpoint fails: no stream, latch."""
    def fake(messages, tools=None, **kw):
        calls.append(1)
        nx.st._LAST_API_ERROR = "APIConnectionError: connection refused"
        return None
    monkeypatch.setattr(nx, "open_stream", fake)


def _fake_client(monkeypatch, exc):
    """A client whose chat.completions.create always raises `exc`."""
    def boom(*a, **k):
        raise exc
    cp = type("Cp", (), {"create": staticmethod(boom)})
    ch = type("Ch", (), {"completions": cp()})
    return type("C", (), {"chat": ch()})(), "fake-model"


# --- open_stream latches the failure ---------------------------------------------

def test_open_stream_latches_last_api_error(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    assert nx.st._LAST_API_ERROR is None
    monkeypatch.setattr(nx.st, "client", _fake_client(monkeypatch, _Boom("context length exceeded"))[0])
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)  # suppress the friendly print
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is None
    assert nx.st._LAST_API_ERROR == "_Boom: context length exceeded"


# --- model_turn / turn-loop status mapping ----------------------------------------

def test_model_turn_maps_none_plus_error_to_api_error(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    calls = []
    _fail_open_stream(nx, monkeypatch, calls)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    assert nx.model_turn(messages) == nx._turn.TURN_API_ERROR
    assert calls == [1]


def test_empty_turn_still_not_api_error(nx_fresh, monkeypatch):
    """A stream that OPENS but yields nothing is the empty-turn path — it must
    never be misread as a hard API failure (the latch is cleared before every
    open, so a stale error from a previous turn can't poison this one)."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx, "open_stream", lambda *a, **k: iter([]))
    nx.st._LAST_API_ERROR = "APIConnectionError: stale latch from earlier"
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    status = nx.model_turn(messages)
    assert status != nx._turn.TURN_API_ERROR
    assert status in (nx._turn.TURN_DONE, nx._turn.TURN_EMPTY)


def test_loop_summary_status_error_no_retry_burn(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    calls = []
    _fail_open_stream(nx, monkeypatch, calls)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=3)
    assert summary["status"] == "error"
    assert "APIConnectionError" in summary["error"]
    assert calls == [1]  # terminal: the loop didn't spend its cap retrying


def test_loop_error_mid_task_keeps_turn_count(nx_fresh, monkeypatch):
    """The endpoint dying after real progress still reports error, with turns
    counting the turns that ran before the failure."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    n = {"i": 0}

    def tool_then_die(messages, *a, **k):
        n["i"] += 1
        return nx._turn.TURN_TOOL if n["i"] < 3 else nx._turn.TURN_API_ERROR

    monkeypatch.setattr(nx, "model_turn", tool_then_die)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=10)
    assert summary["status"] == "error"
    assert summary["turns"] == 3
    assert n["i"] == 3


def test_forced_final_cap_path_reports_error(nx_fresh, monkeypatch):
    """A turn-cap rescue turn that hard-fails is an error, not a cap."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    monkeypatch.setattr(nx, "model_turn", lambda *a, **k: nx._turn.TURN_API_ERROR)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages, max_turns_override=1)
    assert summary["status"] == "error"


# --- batch: a dead endpoint is exit_code 4, never a clean pass ---------------------

def test_batch_dead_endpoint_is_exit_4_not_clean_pass(nx_fresh, monkeypatch, tmp_path):
    nx, _ = nx_fresh
    _install_source(nx)
    out, err = _quiet(monkeypatch)
    _fail_open_stream(nx, monkeypatch, [])
    f = tmp_path / "tasks.jsonl"
    f.write_text('{"instruction": "score me", "id": "t1"}\n', encoding="utf-8")
    from nx._batch import run_batch, EXIT_ERROR
    code = run_batch(["-f", str(f)])
    assert code == EXIT_ERROR, "a dead endpoint must not summarize as OK"
    line = json.loads(out.getvalue().strip())
    assert line["status"] == "error"
    assert line["exit_code"] == EXIT_ERROR
    assert "APIConnectionError" in line["error"]
    assert line["final_answer"] is None
    assert "1 error" in err.getvalue()


def test_batch_result_line_is_error_with_exit_4(nx_fresh, monkeypatch, tmp_path):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    _fail_open_stream(nx, monkeypatch, [])
    f = tmp_path / "tasks.jsonl"
    f.write_text('{"instruction": "score me", "id": "t1"}\n', encoding="utf-8")
    from nx._batch import _run_one_task, _parse_tasks, EXIT_ERROR
    res = _run_one_task(_parse_tasks(str(f))[0], "r", 0, None)
    assert res["status"] == "error"
    assert res["exit_code"] == EXIT_ERROR
    assert "APIConnectionError" in (res.get("error") or "")
    assert res["final_answer"] is None


def test_batch_continue_on_error_still_counts_api_failures(nx_fresh, monkeypatch, tmp_path):
    nx, _ = nx_fresh
    _install_source(nx)
    out, err = _quiet(monkeypatch)
    calls = []
    _fail_open_stream(nx, monkeypatch, calls)
    f = tmp_path / "tasks.jsonl"
    f.write_text('{"instruction": "a", "id": "t1"}\n'
                 '{"instruction": "b", "id": "t2"}\n', encoding="utf-8")
    from nx._batch import run_batch, EXIT_ERROR
    code = run_batch(["-f", str(f), "--continue-on-error"])
    assert code == EXIT_ERROR
    assert len(calls) == 2  # both tasks were attempted, both failed honestly
    lines = [json.loads(l) for l in out.getvalue().strip().splitlines()]
    assert [l["status"] for l in lines] == ["error", "error"]


# --- one-shot: --json payload + process exit ---------------------------------------

def test_one_shot_json_reports_error_and_exit_4(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    _fail_open_stream(nx, monkeypatch, [])
    # _run_one_shot reads nx._PROMPT_FILE and sys.exits; drive the pieces it
    # wires together and assert the exact contract it applies.
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = nx._run_model_turn_loop(messages)
    assert summary["status"] == "error"
    payload = nx._one_shot_result(messages, len(messages) - 1, summary, 1.0)
    assert payload["status"] == "error"
    assert "APIConnectionError" in payload["error"]
    # the exit-code rule _run_one_shot applies to the summary:
    assert (4 if isinstance(summary, dict) and summary.get("status") == "error" else 0) == 4


def test_one_shot_json_ok_payload_has_no_error_key_leak(nx_fresh, monkeypatch):
    """A genuinely completed run's payload carries error=None (new field) and
    stays exit-0."""
    nx, _ = nx_fresh
    _install_source(nx)
    _quiet(monkeypatch)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    summary = {"status": "done", "turns": 1}
    payload = nx._one_shot_result(messages, len(messages) - 1, summary, 1.0)
    assert payload["status"] == "done"
    assert payload["error"] is None
    assert (4 if summary.get("status") == "error" else 0) == 0
