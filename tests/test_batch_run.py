"""Headless batch mode (`nx run -f tasks.jsonl`, nx/_batch.py).

Contract under test:
- task-file parsing (strings, objects, comments, blanks, malformed lines)
- per-task result shape + exit codes (0 done / 3 cap / 4 error / 2 misuse)
- the turn loop is invoked once per task with fresh messages + fresh metrics,
  and the result line carries that task's own token totals
- --parallel runs tasks concurrently with per-task session ids
- --continue-on-error keeps scheduling; without it the first error stops
  scheduling new tasks
- process exit code summarizes the run (0/3/4, misuse=2)
- batch workers are "quiet" and _batch_active() during the loop (verified by
  a fake model_turn reading the TLS flag), and verify/bg-scan gating skips
  them

The suite is offline: _run_model_turn_loop is faked, never a real endpoint.
"""
import io
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    """Tier A: import nx once, no reload.

    The old importlib.reload(nx) rebuilt import-time state from this
    fixture's env. Both jobs it did are covered without a reload:
    - the per-test session store is read LIVE from NX_SESSIONS_DIR/NX_HOME
      on every access (nx/_sessions_fs._sessions_dir / _sessiondb), so the
      env vars set here already redirect every read/write;
    - the model source (st.ACTIVE/client/MODEL) is rebuilt with the proven
      re-wire: _discover_sources() + switch_source() bind st.ACTIVE/
      st.client/st.MODEL from a discovered source. resolve_model() reads the
      configured model (probes are skipped under NX_SKIP_STARTUP_PROBES), so
      no network round-trip happens.
    env rewind is universal (conftest), so the fixture needs no teardown."""
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    yield nx, tmp_path


class _FakeMetrics:
    """Duck-typed stand-in for _Metrics — only what _one_shot_result reads."""

    def __init__(self, session_id, tag="fake"):
        self.session_id = session_id
        self.tag = tag
        self.input_tokens = 11
        self.output_tokens = 7
        self.cache_read_tokens = 0
        self.reasoning_tokens = 0
        self.api_calls = 1

    def as_meta(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_read_tokens": 0, "reasoning_tokens": 0, "api_calls": 1,
                "session_id": self.session_id}


@pytest.fixture()
def fake_loop(nx_fresh, monkeypatch):
    """Replace _run_model_turn_loop with a recorder that returns a pluggable
    status and captures (messages, max_turns, metrics-at-call-time). Also
    installs a fake active source so the batch orchestrator gets past its
    source check."""
    nx, tmp = nx_fresh
    calls = []
    holder = {"status": "done", "fail_task_id": None}

    from nx import _sources as _sources_mod
    src = _sources_mod.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    # Register through monkeypatch on REPLACED copies of the shared
    # registry containers (never mutate the baseline dicts in place — Tier A
    # files get no module-dict rewind), so every swap auto-restores.
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)

    def fake_run_model_turn_loop(messages, force_final=False, recovery_sampling=False,
                                 on_step=None, max_turns_override=None):
        task = nx._batch._batch_task()
        # batch workers install their recorder on the per-thread slot; resolve
        # the ACTIVE recorder the way _record_usage does
        metrics = nx.st._active_metrics()
        sid = None
        mname = None
        if metrics is not None:
            sid = metrics.session_id
            mname = type(metrics).__name__
        calls.append({
            "messages": [dict(m) for m in messages],
            "max_turns": max_turns_override,
            "metrics": mname,
            "metrics_obj": metrics,
            "session_id": sid,
            "task_active": nx._batch._batch_active(),
            "quiet": __import__("nx._subagent", fromlist=["_subagent_quiet"])._subagent_quiet(),
            "instruction": messages[-1]["content"],
        })
        if holder["fail_task_id"] and task and task.get("id") == holder["fail_task_id"]:
            raise RuntimeError("boom")
        return {"status": holder["status"], "turns": 3}

    monkeypatch.setattr(nx, "_run_model_turn_loop", fake_run_model_turn_loop)
    return nx, calls, holder


def _write_tasks(tmp, lines):
    p = tmp / "tasks.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _run(nx, argv):
    from nx._batch import run_batch
    return run_batch(argv)


def _capture_stdout(monkeypatch):
    import io
    buf = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(sys, "stderr", err)
    return buf, err


# --- task-file parsing --------------------------------------------------------

def test_parse_tasks_strings_and_objects(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    f = _write_tasks(tmp, [
        '"do the thing"',                                   # bare string
        '{"instruction": "second task", "id": "t2"}',       # object
        "",                                                 # blank skipped
        "# a comment",                                      # comment skipped
        '{"instruction": "third", "id": "t3", "session": "mine"}',
    ])
    tasks = nx._batch._parse_tasks(f)
    assert [t["instruction"] for t in tasks] == ["do the thing", "second task", "third"]
    assert [t.get("id") for t in tasks] == [None, "t2", "t3"]
    assert tasks[2]["session"] == "mine"
    assert tasks[0]["line"] == 1 and tasks[2]["line"] == 5


def test_parse_tasks_rejects_bad_lines(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    with pytest.raises(nx._batch._BatchUsageError, match="not valid JSON"):
        nx._batch._parse_tasks(_write_tasks(tmp, ["{oops"]))
    with pytest.raises(nx._batch._BatchUsageError, match="instruction"):
        nx._batch._parse_tasks(_write_tasks(tmp, ['{"id": "x"}']))
    with pytest.raises(nx._batch._BatchUsageError, match="JSON string or object"):
        nx._batch._parse_tasks(_write_tasks(tmp, ["42"]))
    with pytest.raises(nx._batch._BatchUsageError, match="session id"):
        nx._batch._parse_tasks(_write_tasks(tmp, ['{"instruction": "x", "session": "../evil"}']))


def test_parse_tasks_rejects_path_session_ids(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    f = _write_tasks(tmp, ['{"instruction": "x", "session": "a/b"}'])
    with pytest.raises(nx._batch._BatchUsageError, match="session id"):
        nx._batch._parse_tasks(f)


# --- argument parsing ---------------------------------------------------------

def test_arg_parsing(nx_fresh):
    nx, _ = nx_fresh
    o = nx._batch._parse_batch_args(["-f", "t.jsonl", "--parallel", "4",
                                     "--continue-on-error", "--max-turns", "9"])
    assert o == {"file": "t.jsonl", "parallel": 4, "continue_on_error": True,
                 "max_turns": 9, "timeout": None, "model": None}
    assert nx._batch._parse_batch_args(["-f", "t", "--timeout", "900"])["timeout"] == 900.0
    assert nx._batch._parse_batch_args(["-f", "t", "--timeout", "2.5"])["timeout"] == 2.5
    with pytest.raises(nx._batch._BatchUsageError):
        nx._batch._parse_batch_args(["--frobnicate"])
    with pytest.raises(nx._batch._BatchUsageError, match="requires -f"):
        nx._batch._parse_batch_args([])
    with pytest.raises(nx._batch._BatchUsageError, match=">= 1"):
        nx._batch._parse_batch_args(["-f", "t", "--parallel", "0"])
    # --yolo/--approval/--source are consumed globally; they parse through
    assert nx._batch._parse_batch_args(
        ["-f", "t", "--yolo", "--approval", "high", "--source", "zai"])["file"] == "t"


# --- misuse / empty-file exits -----------------------------------------------

def test_zero_tasks_is_misuse(nx_fresh, tmp_path, monkeypatch):
    nx, tmp = nx_fresh
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", _write_tasks(tmp, ["# only a comment"])])
    assert code == nx._batch.EXIT_MISUSE
    assert "no tasks" in err.getvalue()


def test_unreadable_file_is_misuse(nx_fresh, tmp_path, monkeypatch):
    nx, tmp = nx_fresh
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", str(tmp / "nope.jsonl")])
    assert code == nx._batch.EXIT_MISUSE
    assert "cannot read" in err.getvalue()


def test_missing_file_flag_is_misuse(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, [])
    assert code == nx._batch.EXIT_MISUSE
    assert "usage: nx run" in err.getvalue()


def test_run_help_flag(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["--help"])
    assert code == 0
    assert "usage: nx run" in buf.getvalue()


# --- the happy path: one result line per task ---------------------------------

def test_batch_runs_each_task_and_emits_jsonl(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, [
        '"first task"',
        '{"instruction": "second task", "id": "t2"}',
    ])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f])
    out = buf.getvalue()
    assert code == 0, f"exit={code}, stderr={err.getvalue()!r}, stdout={out!r}"
    lines = [json.loads(l) for l in out.strip().splitlines()]
    assert len(lines) == 2
    assert [l["id"] for l in lines] == ["task-0", "t2"]
    assert all(l["exit_code"] == 0 and l["status"] == "done" for l in lines)
    assert calls[0]["instruction"] == "first task"
    assert calls[1]["instruction"] == "second task"
    # fresh per-task metrics (real _Metrics instance, fresh per task) +
    # per-task session ids
    assert calls[0]["metrics"] == "_Metrics"
    assert calls[0]["metrics_obj"] is not calls[1]["metrics_obj"]
    assert calls[0]["session_id"] != calls[1]["session_id"]
    assert lines[0]["session"] != lines[1]["session"]
    # per-task token totals from that task's own metrics (a fresh _Metrics
    # with no real API calls reports zeros — the shape is what's contract here)
    assert set(lines[0]["tokens"]) == {"input", "output", "cache_read", "reasoning", "api_calls"}
    assert lines[0]["tokens"]["api_calls"] == 0
    # summary goes to stderr, never stdout
    assert "2 done" in err.getvalue()
    assert "nx run:" not in out
    # a real, resumable session was written per task (0.101.0: in the store;
    # the legacy mirror files are no longer produced)
    sessions_dir = tmp_path / "nxhome" / "sessions"
    ids = [lines[0]["session"], lines[1]["session"]]
    loaded = [nx._load_session(sid) for sid in ids]
    assert all(s is not None for s in loaded)
    assert all(isinstance(s.get("messages"), list) for s in loaded)


def test_turn_cap_plumbing(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, ['"capped task"'])
    holder["status"] = "cap"
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--max-turns", "5"])
    assert code == nx._batch.EXIT_CAP
    line = json.loads(buf.getvalue().strip())
    assert line["exit_code"] == 3 and line["status"] == "cap"
    assert calls[0]["max_turns"] == 5


def test_max_turns_defaults_to_none(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, ['"t"'])
    buf, _ = _capture_stdout(monkeypatch)
    _run(nx, ["-f", f])
    assert calls[0]["max_turns"] is None  # → the global NX_MAX_MODEL_TURNS cap


# --- failures: task error + stop-scheduling + continue-on-error ----------------

def test_task_error_exit_code_4(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    holder["fail_task_id"] = "boom"
    f = _write_tasks(tmp_path, ['{"instruction": "x", "id": "boom"}'])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f])
    assert code == nx._batch.EXIT_ERROR
    line = json.loads(buf.getvalue().strip())
    assert line["exit_code"] == 4 and line["status"] == "error"
    assert "boom" in (line["error"] or "")


def test_error_stops_scheduling_without_flag(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    holder["fail_task_id"] = "bad"
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "bad"}',
        '{"instruction": "b", "id": "never"}',
        '{"instruction": "c", "id": "never2"}',
    ])
    buf, _ = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f])
    assert code == nx._batch.EXIT_ERROR
    ids = [c["instruction"] for c in calls]
    assert ids == ["a"]  # b and c were never scheduled


def test_continue_on_error_runs_remaining_tasks(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    holder["fail_task_id"] = "bad"
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "bad"}',
        '{"instruction": "b", "id": "next"}',
        '{"instruction": "c", "id": "next2"}',
    ])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--continue-on-error"])
    assert code == nx._batch.EXIT_ERROR
    assert [c["instruction"] for c in calls] == ["a", "b", "c"]
    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    assert [l["exit_code"] for l in lines] == [4, 0, 0]
    assert "1 error, 2 done" in err.getvalue() or "2 done, 1 error" in err.getvalue()


# --- parallelism ---------------------------------------------------------------

def test_parallel_runs_concurrently_with_per_task_state(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    n = 4
    started = threading.Barrier(n, timeout=10)

    real_loop = nx._run_model_turn_loop

    def slow_loop(messages, **kw):
        started.wait()  # all 4 tasks must be inside the loop simultaneously
        return real_loop(messages, **kw)

    monkeypatch.setattr(nx, "_run_model_turn_loop", slow_loop)
    f = _write_tasks(tmp_path, [json.dumps({"instruction": f"t{i}", "id": f"p{i}"})
                                for i in range(n)])
    buf, err = _capture_stdout(monkeypatch)
    t0 = time.time()
    code = _run(nx, ["-f", f, "--parallel", "4", "--continue-on-error"])
    elapsed = time.time() - t0
    assert code == 0
    assert len(calls) == 4
    assert elapsed < 9.0  # a barrier of 4 proves they overlapped; loose bound
    sessions = [json.loads(l)["session"] for l in buf.getvalue().strip().splitlines()]
    assert len(set(sessions)) == 4  # distinct per-task session ids


# --- quiet mode + _batch_active -------------------------------------------------

def test_worker_is_quiet_and_batch_active_inside_loop(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, ['"q"'])
    buf, _ = _capture_stdout(monkeypatch)
    _run(nx, ["-f", f])
    assert calls[0]["task_active"] is True
    assert calls[0]["quiet"] is True
    # and the TLS slot is cleared after the run
    assert nx._batch._batch_active() is False


def test_batch_active_false_outside_batch(nx_fresh):
    nx, _ = nx_fresh
    assert nx._batch_active() is False


def test_verify_and_bg_scan_gated_off_for_batch(fake_loop, tmp_path, monkeypatch):
    """A batch worker must not launch NX_VERIFY_CMD nor run the bg-completion
    scan (both are interactive-parent-only machinery)."""
    nx, calls, holder = fake_loop
    launched = []
    monkeypatch.setattr(nx, "_launch_verify_if_settled",
                        lambda *a, **k: launched.append(1))
    from nx._dispatch import run_tool
    scan_ran = []
    monkeypatch.setattr(nx._dispatch, "_scan_background_completions",
                        lambda: scan_ran.append(1) or "")
    # A non-batch thread still runs the scan (interactive behavior unchanged)…
    # monkeypatch registers the import-time value so the flag the scan sets
    # True auto-restores at teardown (no module-dict rewind in Tier A);
    # the mid-test reset below is covered by that same undo registration.
    monkeypatch.setattr(nx.st, "_BG_SCAN_DONE", False)
    run_tool("read_file", {"path": "README.md"})
    assert scan_ran == [1]
    # …but a batch-active thread skips it.
    import nx._batch as batch_mod
    batch_mod._TASK_TLS.task = {"instruction": "x"}
    try:
        nx.st._BG_SCAN_DONE = False
        run_tool("read_file", {"path": "README.md"})
        assert scan_ran == [1]  # no second scan
    finally:
        batch_mod._TASK_TLS.task = None
    assert not launched  # turn loop is faked, so verify launch is skipped


# --- exit code summary -----------------------------------------------------------

def test_mixed_run_exit_code_is_worst(fake_loop, tmp_path, monkeypatch):
    nx, calls, holder = fake_loop
    holder["fail_task_id"] = "bad"
    holder["status"] = "done"
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "ok1"}',
        '{"instruction": "b", "id": "bad"}',
        '{"instruction": "c", "id": "ok2"}',
    ])

    def loop(messages, **kw):
        task = nx._batch._batch_task()
        if task and task.get("id") == "bad":
            raise RuntimeError("kaboom")
        return {"status": holder["status"], "turns": 2}

    monkeypatch.setattr(nx, "_run_model_turn_loop", loop)
    buf, _ = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--continue-on-error"])
    assert code == nx._batch.EXIT_ERROR
    # a cap-status task alone exits 3 (error beats cap, cap beats ok)
    holder["fail_task_id"] = None
    holder["status"] = "cap"
    f2 = _write_tasks(tmp_path, ['{"instruction": "d", "id": "capped"}'])
    buf2, _ = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f2])
    assert code == nx._batch.EXIT_CAP


def test_result_exit_code_mapping(nx_fresh):
    nx, _ = nx_fresh
    f = nx._batch._result_exit_code
    assert f("done") == 0
    assert f("cap") == 3
    assert f("esc") == 4
    assert f(None) == 4
    assert f("error") == 4


# --- per-task metrics under --parallel > 1 ------------------------------------
# _record_usage() resolves the recorder from the worker thread's metrics slot
# (st._METRICS_TLS → st._active_metrics()), not the process-global st._METRICS:
# a swap-and-restore of the global races every sibling worker and silently
# misattributes (or loses) each task's token totals on the JSONL result lines.

def test_record_usage_lands_in_thread_metrics_not_global(fake_loop, monkeypatch):
    """A batch worker's usage goes to ITS metrics even though another thread
    (or the REPL parent) has installed a different recorder on st._METRICS."""
    nx, calls, holder = fake_loop
    from types import SimpleNamespace

    from nx._metrics import _record_usage
    global_metrics = nx._metrics._Metrics("repl-session")
    # monkeypatch, not a bare write + `= None` teardown: the manual teardown
    # would leak None past this file (whatever st._METRICS held at rewind
    # time is what the next Tier A file sees; there is no module-dict
    # rewind to put it back).
    monkeypatch.setattr(nx.st, "_METRICS", global_metrics)
    nx.st._METRICS_TLS.metrics = nx._metrics._Metrics("task-session")
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3,
                            prompt_tokens_details=None,
                            output_tokens_details=None)
    try:
        _record_usage(usage)
    finally:
        nx.st._METRICS_TLS.metrics = None  # thread-local, cleared by hand
    assert global_metrics.api_calls == 0
    assert global_metrics.input_tokens == 0
    assert global_metrics.output_tokens == 0


def test_parallel_task_token_totals_are_per_task(fake_loop, tmp_path, monkeypatch):
    """End-to-end: N overlapping tasks, each _record_usage'ing its own volume.
    Every result line's tokens must equal what that task recorded — under the
    old global swap-and-restore, task A's calls landed in whichever metrics
    task B had last installed on st._METRICS, so lines came back zeroed,
    swapped, or double-counted (nondeterministically)."""
    nx, calls, holder = fake_loop
    n = 6
    barrier = threading.Barrier(n, timeout=10)
    from types import SimpleNamespace

    from nx._metrics import _record_usage

    def loop_with_usage(messages, **kw):
        task = nx._batch._batch_task()
        tid = int(task["id"].lstrip("p"))  # "p3" -> 3
        barrier.wait()                     # force real overlap
        _record_usage(SimpleNamespace(prompt_tokens=100 + tid,
                                      completion_tokens=10 + tid,
                                      prompt_tokens_details=None,
                                      output_tokens_details=None))
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", loop_with_usage)
    f = _write_tasks(tmp_path, [json.dumps({"instruction": f"t{i}", "id": f"p{i}"})
                                for i in range(n)])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--parallel", str(n), "--continue-on-error"])
    assert code == 0, f"exit={code}, stderr={err.getvalue()!r}, lines={buf.getvalue()!r}"
    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    by_id = {l["id"]: l for l in lines}
    assert len(by_id) == n
    for i in range(n):
        line = by_id[f"p{i}"]
        # exact per-task attribution: no sibling's tokens, none lost
        assert line["tokens"] == {"input": 100 + i, "output": 10 + i,
                                  "cache_read": 0, "reasoning": 0,
                                  "api_calls": 1}
    # and the REPL's global recorder was never touched by any worker
    assert nx.st._METRICS is None or nx.st._METRICS.api_calls == 0


def test_parallel_task_sessions_carry_own_totals(fake_loop, tmp_path, monkeypatch):
    """The per-task session JSONs are what a dashboard/resume reads back: each
    must carry its own task's totals, not a sibling's (parallel workers used
    to merge whatever st._METRICS happened to hold at save time)."""
    nx, calls, holder = fake_loop
    n = 4
    barrier = threading.Barrier(n, timeout=10)
    from types import SimpleNamespace

    from nx._metrics import _record_usage

    def loop_with_usage(messages, **kw):
        task = nx._batch._batch_task()
        tid = int(task["id"].lstrip("p"))
        barrier.wait()
        _record_usage(SimpleNamespace(prompt_tokens=7,
                                      completion_tokens=tid,
                                      prompt_tokens_details=None,
                                      output_tokens_details=None))
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", loop_with_usage)
    f = _write_tasks(tmp_path, [json.dumps({"instruction": f"t{i}", "id": f"p{i}"})
                                for i in range(n)])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--parallel", str(n), "--continue-on-error"])
    assert code == 0, f"exit={code}, stderr={err.getvalue()!r}"
    # one saved session per task; each carries its own totals — match the
    # sessions back to tasks by their distinct output-token value (0.101.0:
    # sessions live in the store, read via _load_session)
    sessions_dir = tmp_path / "nxhome" / "sessions"

    sids = sorted(nx._sessiondb.all_ids())
    assert len(sids) == n
    totals = sorted(nx._load_session(sid)["output_tokens"] for sid in sids)
    assert totals == [0, 1, 2, 3]


# --- quiet render-path regressions ---------------------------------------------
# A batch worker's stdout must carry ONLY the JSONL result lines: every
# human-facing render on the shared turn path (stream-parser banners, the
# forced-final handlers, the turn-limit banner) is gated on the quiet posture.
# These pin the leak where a quiet StreamParser.finish() still printed the ANSI
# reset that polluted the first live batch run's stdout.

def test_quiet_stream_parser_finish_writes_nothing(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    from nx._stream_parse import StreamParser
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    p = StreamParser(buf, t0=0.0, quiet=True)
    p.mode = "think"      # close_think_block has a closer line pending
    p.think_start = 0.0
    p.finish()
    assert buf.getvalue() == ""
    # The interactive closer is intact — gated, not removed.
    buf2 = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf2)
    StreamParser(buf2, t0=0.0, quiet=False).finish()
    assert buf2.getvalue() == "\x1b[0m"  # inline reset — no self-owned blank line


def test_quiet_forced_final_handler_is_silent(nx_fresh):
    nx, _ = nx_fresh
    from nx._subagent import _ASYNC_QUIET_SLOTS
    messages = []
    tcs = {0: {"id": "1", "name": "final_answer",
               "args": json.dumps({"answer": "the answer", "status": "answered"})}}
    buf = io.StringIO()
    old = sys.stdout
    # A batch worker's TLS quiet slot is what makes the shared handler silent
    # (a bare main-thread call is an interactive turn and may print).
    _ASYNC_QUIET_SLOTS.quiet = True
    sys.stdout = buf
    try:
        status = nx._turn._handle_forced_final_tool_call(
            messages, tcs, "", [], 0)
    finally:
        sys.stdout = old
        _ASYNC_QUIET_SLOTS.quiet = False
    assert status == nx._turn.TURN_DONE
    assert buf.getvalue() == ""                  # answer text is stdout noise here
    assert messages[-1]["content"] == "the answer"  # transcript still complete


def test_quiet_loop_prints_nothing_even_at_turn_cap(nx_fresh, monkeypatch):
    """The cap path (MODEL TURN LIMIT banner + forced-final rescue) used to
    print unconditionally — polluting a batch consumer's stdout stream."""
    nx, _ = nx_fresh

    def tool_then_done(messages, *a, forced_final=False, **k):
        return nx._turn.TURN_DONE if forced_final else nx._turn.TURN_TOOL

    monkeypatch.setattr(nx, "model_turn", tool_then_done)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    buf, err = io.StringIO(), io.StringIO()
    from nx._subagent import _ASYNC_QUIET_SLOTS
    _ASYNC_QUIET_SLOTS.quiet = True  # the posture _run_one_task gives a worker
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(sys, "stderr", err)
    try:
        summary = nx._run_model_turn_loop(messages, max_turns_override=2)
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
    assert summary["status"] == "cap"
    assert buf.getvalue() == ""


# --- per-task wall-clock timeout -------------------------------------------------
# The batch docstring used to promise "the per-task timeout" without shipping
# one: --max-turns x (600s stream read + retries) was the only bound, so one
# slow/pathological task could hold a CI run open for hours and a --parallel 1
# batch serialized behind it. The deadline is armed per worker thread (a
# thread-local in _turn.py) and enforced at the loop's on_step seam.

def test_task_timeout_resolves_precedence_and_zero_off(nx_fresh):
    nx, _ = nx_fresh
    f = nx._batch._task_timeout
    # task field wins over the --timeout default; 0 anywhere = off
    assert f({"timeout_seconds": 30, "line": 1}, 900) == 30.0
    assert f({}, 900) == 900.0
    assert f({}, None) is None
    assert f({"timeout_seconds": 0}, 900) is None
    assert f({"timeout_seconds": 0, "line": 2}, None) is None
    assert f({}, 0) is None
    with pytest.raises(nx._batch._BatchUsageError, match="timeout_seconds"):
        f({"timeout_seconds": -5, "line": 3}, 900)
    with pytest.raises(nx._batch._BatchUsageError, match="timeout_seconds"):
        f({"timeout_seconds": "soon", "line": 4}, 900)


def test_task_timeout_fires_at_step_boundary(nx_fresh, tmp_path, monkeypatch):
    """A task whose deadline lapses mid-loop reports an error line with
    timed_out=true, and the process exit is 4."""
    nx, tmp = nx_fresh

    seen = {}

    def slow_loop(messages, on_step=None, max_turns_override=None, **kw):
        deadline = nx._turn._task_deadline()
        seen["deadline"] = deadline
        steps = 0
        while steps < (max_turns_override or 50):
            time.sleep(0.05)
            steps += 1
            if on_step is not None:
                on_step()
            if nx._turn._task_deadline_exceeded():
                return nx._turn._task_timeout_summary(steps)
        return {"status": "cap", "turns": steps}

    monkeypatch.setattr(nx, "_run_model_turn_loop", slow_loop)
    f = _write_tasks(tmp, ['{"instruction": "slog", "id": "slow"}'])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--timeout", "0.2"])
    assert code == nx._batch.EXIT_ERROR
    line = json.loads(buf.getvalue().strip())
    assert line["status"] == "error" and line["exit_code"] == 4
    assert line["timed_out"] is True
    assert "task timeout" in (line["error"] or "")
    assert seen["deadline"] is not None
    assert "timed out" in err.getvalue()
    # the worker's deadline slot is disarmed after the run
    assert nx._turn._task_deadline() is None


def test_task_timeout_beats_turn_cap_rescue(nx_fresh, tmp_path, monkeypatch):
    """At the turn cap the forced-final rescue turn is skipped when the
    deadline has lapsed — the timeout, not one more model turn, is the bound."""
    nx, tmp = nx_fresh
    rescue_calls = []

    def tool_then_cap(messages, *a, forced_final=False, **k):
        if forced_final:
            rescue_calls.append(1)
        return nx._turn.TURN_TOOL

    monkeypatch.setattr(nx, "model_turn", tool_then_cap)
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    from nx._subagent import _ASYNC_QUIET_SLOTS
    _ASYNC_QUIET_SLOTS.quiet = True
    nx._turn._task_deadline_at(time.time() - 1)  # already lapsed
    try:
        summary = nx._run_model_turn_loop(messages, max_turns_override=2)
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
        nx._turn._task_deadline_at(None)
    assert summary["status"] == "error" and summary.get("timed_out") is True
    assert rescue_calls == []  # no forced-final turn after the deadline


def test_task_timeout_field_per_task(nx_fresh, tmp_path, monkeypatch):
    """`timeout_seconds` on the task line wins over --timeout and only binds
    that one task."""
    nx, tmp = nx_fresh

    def loop(messages, on_step=None, max_turns_override=None, **kw):
        steps = 0
        while steps < (max_turns_override or 50):
            time.sleep(0.05)
            steps += 1
            if on_step is not None:
                on_step()
            if nx._turn._task_deadline_exceeded():
                return nx._turn._task_timeout_summary(steps)
        return {"status": "done", "turns": steps}

    monkeypatch.setattr(nx, "_run_model_turn_loop", loop)
    f = _write_tasks(tmp, [
        '{"instruction": "short", "id": "quick", "timeout_seconds": 0.15}',
        '{"instruction": "long", "id": "dur", "timeout_seconds": 30}',
    ])
    buf, err = _capture_stdout(monkeypatch)
    # --continue-on-error so the timed-out task doesn't stop "dur" from
    # being scheduled: the deadline binds only the task that declares it.
    code = _run(nx, ["-f", f, "--timeout", "60", "--continue-on-error"])
    lines = [json.loads(line) for line in buf.getvalue().strip().splitlines()]
    assert len(lines) == 2
    by_id = {line["id"]: line for line in lines}
    assert by_id["quick"]["timed_out"] is True
    assert by_id["quick"]["status"] == "error"
    assert by_id["dur"]["status"] == "done"
    assert code == nx._batch.EXIT_ERROR


def test_task_timeout_env_default(nx_fresh, tmp_path, monkeypatch):
    """NX_TASK_TIMEOUT_SECONDS is the flag-less default: run_batch reads it
    from _TASK_TIMEOUT_DEFAULT (the import-time _env_int read — patched here
    to the value the env var would have supplied). 0 / unset keeps the old
    unbounded behavior."""
    nx, tmp = nx_fresh

    def loop(messages, on_step=None, max_turns_override=None, **kw):
        # mirror the real loop: the deadline binds at the step boundary
        time.sleep(0.3)
        if nx._turn._task_deadline_exceeded():
            return nx._turn._task_timeout_summary(1)
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", loop)
    assert nx._batch._TASK_TIMEOUT_DEFAULT == 0  # env default: off
    # monkeypatch, not set/restore-by-hand: auto-restores the import-time 0
    monkeypatch.setattr(nx._batch, "_TASK_TIMEOUT_DEFAULT", 0.15)
    f = _write_tasks(tmp, ['{"instruction": "slog", "id": "slow"}'])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f])
    line = json.loads(buf.getvalue().strip())
    assert code == nx._batch.EXIT_ERROR
    assert line["timed_out"] is True


def test_no_timeout_by_default_is_unbounded(nx_fresh, tmp_path, monkeypatch):
    """Without the flag/env/field the deadline slot stays empty — the flagless
    run behaves exactly as before (the fix must not time out legacy runs)."""
    nx, tmp = nx_fresh

    def loop(messages, on_step=None, max_turns_override=None, **kw):
        seen = nx._turn._task_deadline()
        return {"status": "done", "turns": 1} if seen is None else {"status": "cap"}

    monkeypatch.setattr(nx, "_run_model_turn_loop", loop)
    f = _write_tasks(tmp, ['"t"'])
    buf, _ = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f])
    assert code == 0
    assert json.loads(buf.getvalue().strip())["status"] == "done"


def test_parallel_bad_timeout_field_is_misuse(nx_fresh, tmp_path, monkeypatch):
    """A non-numeric timeout_seconds is a harness misuse (exit 2), caught up
    front under --parallel so it can't fail lazily mid-run."""
    nx, tmp = nx_fresh
    f = _write_tasks(tmp, [
        '{"instruction": "a", "id": "ok"}',
        '{"instruction": "b", "id": "bad", "timeout_seconds": "soon"}',
    ])
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--parallel", "2"])
    assert code == nx._batch.EXIT_MISUSE
    assert "timeout_seconds" in err.getvalue()
    assert buf.getvalue() == ""  # nothing ran


# --- 0.156.0: per-task overrides (approval / model / approval_policy) ----------

def test_arg_parsing_model_flag(nx_fresh):
    nx, _ = nx_fresh
    o = nx._batch._parse_batch_args(["-f", "t", "--model", "glm-5.3"])
    assert o["model"] == "glm-5.3"
    with pytest.raises(nx._batch._BatchUsageError, match="requires a model id"):
        nx._batch._parse_batch_args(["-f", "t", "--model"])
    with pytest.raises(nx._batch._BatchUsageError, match="unknown nx run flag"):
        nx._batch._parse_batch_args(["-f", "t", "--modle", "x"])
    assert "usage: nx run" in nx._batch._USAGE_TEXT and "--model" in nx._batch._USAGE_TEXT


def test_parse_tasks_rejects_bad_override_keys(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    bad = [
        '{"instruction": "x", "approval": "nuke"}',                 # bad value
        '{"instruction": "x", "approval": 1}',                      # non-string
        '{"instruction": "x", "model": "nonesource:m"}',            # unknown prefix
        '{"instruction": "x", "model": ["m"]}',                     # non-string
        '{"instruction": "x", "approval_policy": "allow-all:bash"}',  # wrong prefix
        '{"instruction": "x", "approval_policy": "headless-allow:"}',  # empty globs
        '{"instruction": "x", "approval_policy": 7}',
    ]
    for line in bad:
        with pytest.raises(nx._batch._BatchUsageError, match="task line 1"):
            nx._batch._parse_tasks(_write_tasks(tmp, [line]))
    # valid values parse through (bare id + a registered source prefix)
    ok = nx._batch._parse_tasks(_write_tasks(tmp, [
        '{"instruction": "x", "approval": "yolo", "model": "local:whatever",'
        ' "approval_policy": "headless-allow: bash* , edit_file"}']))
    assert ok[0]["approval"] == "yolo" and ok[0]["model"] == "local:whatever"


def test_precedence_task_beats_flag_beats_env(nx_fresh, tmp_path, monkeypatch):
    """Per-key resolution precedence, parse/unit level (plan item 1): the
    `approval` task key is ABSOLUTE — "low" inside a --yolo run demotes
    ((False, "low"), not a merge) — while a key-less task renders the run's
    own posture from the globals."""
    nx, _ = nx_fresh
    tasks = nx._batch._parse_tasks(_write_tasks(tmp_path, [
        '{"instruction": "a", "approval": "low"}',
        '{"instruction": "b"}',
    ]))
    monkeypatch.setattr(nx.st, "YOLO", True)
    monkeypatch.setattr(nx.st, "APPROVE_LEVEL", None)
    # task key wins absolutely: "low" in a --yolo run demotes
    assert nx._batch._task_posture(tasks[0]) == ((False, "low"), "low")
    # absent key → the run's yolo posture (today's behavior)
    assert nx._batch._task_posture(tasks[1]) == (None, "yolo")


def test_task_approval_absolute_demotes_yolo(fake_loop, tmp_path, monkeypatch):
    """A task `approval: "low"` in a --yolo run is ABSOLUTE: the task's own
    posture scope installs (False, "low") for the body — while a sibling with
    no key keeps the run's yolo."""
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, [
        '{"instruction": "y", "id": "yolo-task", "approval": "yolo"}',
        '{"instruction": "d", "id": "low-task", "approval": "low"}',
        '{"instruction": "g", "id": "global"}',
    ])
    monkeypatch.setattr(nx.st, "YOLO", True)
    monkeypatch.setattr(nx.st, "APPROVE_LEVEL", None)

    def probe(messages, on_step=None, max_turns_override=None, **kw):
        yolo, level = nx.st._posture()
        task = nx._batch._batch_task()
        calls.append({"id": task.get("id"), "yolo": yolo, "level": level,
                      "task_level": nx._approval._task_level(),
                      "policy": nx._approval_policy._policy_globs()})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", probe)
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--continue-on-error"])
    assert code == 0, f"exit={code}, stderr={err.getvalue()!r}"
    by_id = {c["id"]: c for c in calls}
    assert by_id["yolo-task"] == {"id": "yolo-task", "yolo": True, "level": None,
                                  "task_level": None, "policy": ()}
    assert by_id["low-task"]["yolo"] is False and by_id["low-task"]["level"] == "low"
    assert by_id["low-task"]["task_level"] == "low"  # LEVEL opt-in installed
    # the key-less task rides the run's yolo globals untouched
    assert by_id["global"]["yolo"] is True and by_id["global"]["task_level"] is None
    # globals restored after the run
    assert nx.st._posture() == (True, None)
    lines = {json.loads(l)["id"]: json.loads(l) for l in buf.getvalue().strip().splitlines()}
    assert lines["yolo-task"]["approval"] == "yolo"
    assert lines["low-task"]["approval"] == "low"
    assert lines["global"]["approval"] == "yolo"


def test_per_task_model_reaches_call_target(fake_loop, tmp_path, monkeypatch):
    """Task `model` > `--model` flag > globals: the resolved target rides
    _CALL_TARGET_TLS during the task body and is gone after it."""
    nx, calls, holder = fake_loop
    from nx._sources import Source
    other = Source("other", "http://127.0.0.1:8/v1", "k", "other-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "other": other})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", list(nx.st.SOURCE_ORDER) + ["other"])
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "src", "model": "other:other-model"}',
        '{"instruction": "b", "id": "bare", "model": "bare-model"}',
        '{"instruction": "c", "id": "flag"}',
        '{"instruction": "d", "id": "none"}',
    ])
    seen = []

    def probe(messages, on_step=None, max_turns_override=None, **kw):
        t = nx._subagent_model.call_target()
        seen.append({"task": nx._batch._batch_task().get("id"), "target": t})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", probe)
    buf, _ = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--model", "flag-model", "--continue-on-error"])
    assert code == 0
    by_id = {s["task"]: s for s in seen}
    assert by_id["src"]["target"][0] is other.client
    assert by_id["src"]["target"][1] == "other-model"
    assert by_id["src"]["target"][3] is other  # pin_src
    assert by_id["bare"]["target"][1] == "bare-model"
    assert by_id["bare"]["target"][0] is nx.st.client  # active source's client
    assert by_id["flag"]["target"][1] == "flag-model"  # --model run default
    # a key-less task rides the RUN's --model default too (run-level flag)
    assert by_id["none"]["target"] is not None
    assert by_id["none"]["target"][1] == "flag-model"
    lines = {json.loads(l)["id"]: json.loads(l) for l in buf.getvalue().strip().splitlines()}
    assert lines["src"]["model"] == "other-model"
    assert lines["bare"]["model"] == "bare-model"
    assert lines["flag"]["model"] == "flag-model"
    # no scope left installed after the run
    assert nx._subagent_model.call_target() is None
    # and without --model a key-less task stays on the globals (legacy)
    f2 = _write_tasks(tmp_path, ['{"instruction": "legacy", "id": "legacy"}'])
    seen.clear()
    buf2, _ = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f2])
    assert code == 0 and seen[0]["target"] is None


def test_session_meta_stamps_effective_model(fake_loop, tmp_path, monkeypatch):
    """The saved session meta carries the task's EFFECTIVE model, not st.MODEL."""
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "routed", "model": "routed-model"}',
    ])
    buf, _ = _capture_stdout(monkeypatch)
    _run(nx, ["-f", f])
    sid = json.loads(buf.getvalue().strip())["session"]
    sess = nx._load_session(sid)
    assert sess["model"] == "routed-model"
    assert sess["model"] != nx.st.MODEL


def test_per_task_policy_scope_in_batch(fake_loop, tmp_path, monkeypatch):
    """`approval_policy` on a task policy-allows its tool headless; the run
    keeps the env-only behavior for tasks without the key."""
    nx, calls, holder = fake_loop
    monkeypatch.setenv("NX_APPROVAL_POLICY", "headless-allow:env_tool")
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "task-pol", "approval_policy": "headless-allow:task_tool"}',
        '{"instruction": "b", "id": "no-key"}',
    ])
    seen = []

    def probe(messages, on_step=None, max_turns_override=None, **kw):
        seen.append({"task": nx._batch._batch_task().get("id"),
                     "globs": nx._approval_policy._policy_globs()})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", probe)
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--continue-on-error"])
    assert code == 0, f"exit={code}, stderr={err.getvalue()!r}"
    by_id = {s["task"]: s for s in seen}
    assert by_id["task-pol"]["globs"] == ("task_tool",)
    assert by_id["no-key"]["globs"] == ("env_tool",)  # env governs without the key
    assert nx._approval_policy._policy_globs() == ("env_tool",)  # restored


def test_batch_never_sets_subagent_ctx(fake_loop, tmp_path, monkeypatch):
    """Guard: a batch task routes via _CALL_TARGET_TLS, never via
    _SUBAGENT_TLS.ctx (a batch task is a FULL agent — spec §Rejected)."""
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "model": "other-m", "approval": "low"}',
    ])

    def probe(messages, on_step=None, max_turns_override=None, **kw):
        from nx._subagent import _subagent_ctx
        calls.append(_subagent_ctx())
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", probe)
    buf, _ = _capture_stdout(monkeypatch)
    _run(nx, ["-f", f])
    assert calls == [None]


def test_parallel_sibling_divergence_yolo_vs_deny(fake_loop, tmp_path, monkeypatch):
    """Two parallel tasks: the yolo task's _confirm is True, the sibling's is
    'DENIED by user' — drive _confirm directly inside each task body."""
    nx, calls, holder = fake_loop
    f = _write_tasks(tmp_path, [
        '{"instruction": "a", "id": "loose", "approval": "yolo"}',
        '{"instruction": "b", "id": "tight"}',
    ])
    verdicts = {}

    def probe(messages, on_step=None, max_turns_override=None, **kw):
        task = nx._batch._batch_task()
        verdicts[task["id"]] = nx._approval._confirm("write: secret file")
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx, "_run_model_turn_loop", probe)
    barrier = threading.Barrier(2, timeout=10)

    def slow(messages, **kw):
        barrier.wait()
        return probe(messages)

    monkeypatch.setattr(nx, "_run_model_turn_loop", slow)
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", f, "--parallel", "2", "--continue-on-error"])
    assert code == 0, f"exit={code}, stderr={err.getvalue()!r}"
    assert verdicts == {"loose": True, "tight": False}


def test_unknown_flag_values_still_misuse(nx_fresh, tmp_path, monkeypatch):
    """--model with an unknown source prefix is a parse error before any run."""
    nx, tmp = nx_fresh
    buf, err = _capture_stdout(monkeypatch)
    code = _run(nx, ["-f", _write_tasks(tmp, ['"x"']), "--model", "ghost:m"])
    assert code == nx._batch.EXIT_MISUSE
    assert "ghost" in err.getvalue()
    assert buf.getvalue() == ""  # no API call, no JSONL line
