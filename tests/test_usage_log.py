#!/usr/bin/env python3
"""v1 usage-metrics event log: run_tool emits a tool_call event per dispatch
(successes too), NX_USAGE_LOG=0 disables emission at the flag, the extended
repair record carries model+raw, and a withheld-tool refusal emits a gate
event with the CTX_TAIL.

Capture pattern mirrors tests/test_subagent_error_status.py (patch the log
sink, read back what was written): here the sink is the module-level
_AsyncLogWriter, so each test swaps in a writer over its own tmp path and
flush()s before reading.
"""
import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = "/dev/null"
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    # The _logging module is cached across reloads, so its module-level
    # writers still point at the FIRST import's NX_HOME: swap in tmp writers
    # for both streams (usage + the legacy llamacpp.log the repair record
    # rides) so this test reads exactly what IT emitted.
    from nx import _logging
    usage_path = tmp_path / "usage.jsonl"
    req_path = tmp_path / "llamacpp.log"
    monkeypatch.setattr(_logging, "_ulog",
                        _logging._AsyncLogWriter(str(usage_path), 0))
    monkeypatch.setattr(_logging, "_llog",
                        _logging._AsyncLogWriter(str(req_path), 0))
    monkeypatch.setattr(_logging, "USAGE_LOG", True)
    _logging.st._TOOL_RESULT_CACHE.clear()
    yield nx, tmp_path, usage_path, req_path


def _read(path):
    from nx import _logging
    _logging._ulog.flush()
    _logging._llog.flush()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_tool_emits_tool_call_event(nx_fresh, capsys):
    nx, tmp, usage_path, _req = nx_fresh
    from nx._dispatch import run_tool
    out = run_tool("list_dir", {"path": "."})
    capsys.readouterr()
    assert not out.startswith("ERROR")
    events = [e for e in _read(usage_path) if e.get("event") == "tool_call"]
    assert len(events) == 1
    d = events[0]["data"]
    assert d["tool"] == "list_dir"
    assert d["outcome"] == "ok"
    assert d["error_kind"] == "ok"
    assert d["duration_ms"] >= 0.0
    assert d["args_bytes"] > 0
    assert '"path"' in d["args_raw"]
    assert d["result_bytes"] == len(d["result_head"]) or d["result_bytes"] >= len(d["result_head"])
    assert d["cache_hit"] is False
    # common fields
    assert d["model"] == "test-model"
    assert d["agent"] == "main"
    assert d["subagent"] is False
    assert "ts" in events[0]  # stamped by the writer


def test_run_tool_error_outcome_and_ctx_tail(nx_fresh, capsys):
    nx, tmp, usage_path, _req = nx_fresh
    from nx._dispatch import run_tool
    from nx._usage_ctx import remember
    remember("I will now read a file that does not exist")
    out = run_tool("read_file", {"path": str(tmp / "definitely-missing.txt")})
    capsys.readouterr()
    assert out.startswith("ERROR")
    events = [e for e in _read(usage_path) if e.get("event") == "tool_call"]
    assert events, "error calls must emit too"
    d = events[-1]["data"]
    assert d["outcome"] == "error"
    assert d["error_kind"] == "ERROR"
    # CTX_TAIL: the last assistant text before the call rides failures
    assert "does not exist" in d["ctx_tail"]


def test_usage_log_zero_disables_emission(nx_fresh, capsys, monkeypatch):
    nx, tmp, usage_path, _req = nx_fresh
    from nx import _logging
    monkeypatch.setattr(_logging, "USAGE_LOG", False)
    from nx._dispatch import run_tool
    run_tool("list_dir", {"path": "."})
    capsys.readouterr()
    assert _read(usage_path) == []


def test_repair_record_carries_model_and_raw(nx_fresh):
    nx, tmp, _usage, req_path = nx_fresh
    from nx._argsalvage import _repair_log
    from nx._usage_ctx import remember
    remember("let me batch these edits")
    raw = '{"edits": [{"old": "x", ' + ("y" * 9000)
    _repair_log("malformed_args", "multi_edit: truncated", raw=raw)
    records = [r for r in _read(req_path) if r.get("dir") == "repair"]
    assert len(records) == 1
    d = records[0]["data"]
    assert d["kind"] == "malformed_args"
    assert d["model"] == "test-model"
    assert d["agent"] == "main"
    assert d["raw"].startswith('{"edits"')
    assert len(d["raw"]) == 8192  # truncated
    assert d["ctx_tail"] == "let me batch these edits"


def test_gate_refusal_emits_event(nx_fresh, capsys):
    nx, tmp, usage_path, _req = nx_fresh
    from nx._tool_gates import _unknown_tool_error
    out = _unknown_tool_error("wait_background")
    capsys.readouterr()
    assert out.startswith("ERROR")
    events = [e for e in _read(usage_path) if e.get("event") == "gate"]
    assert events
    d = events[-1]["data"]
    assert d["tool"] == "wait_background"
    assert d["model"] == "test-model"


def test_breaker_event_on_trip(nx_fresh):
    nx, tmp, usage_path, _req = nx_fresh
    from nx import _breaker
    from nx import _state as st
    _breaker._TRIP_ANNOUNCED.clear()
    st._MALFORMED_TOOL_FAILURES.clear()
    st._MALFORMED_TOOL_BROKEN.clear()
    st._MALFORMED_TOOL_FAILURES["multi_edit"] = 3
    st._MALFORMED_TOOL_BROKEN.add("multi_edit")
    assert _breaker._breaker_broken("multi_edit") is True
    # second read of the same open must NOT re-announce
    assert _breaker._breaker_broken("multi_edit") is True
    events = [e for e in _read(usage_path) if e.get("event") == "breaker"]
    assert len(events) == 1
    d = events[0]["data"]
    assert d["tool"] == "multi_edit"
    assert d["trip_count"] == 3
    # cleanup so the trip state can't leak into sibling tests in this worker
    _breaker._breaker_decay("multi_edit")
    assert _breaker._breaker_broken("multi_edit") is False
    _read(usage_path)


def test_retry_event_per_ladder_step(nx_fresh):
    nx, tmp, usage_path, _req = nx_fresh
    from nx._turn_retry import _retry_log
    _retry_log(1, "ReadTimeout")
    events = [e for e in _read(usage_path) if e.get("event") == "retry"]
    assert len(events) == 1
    d = events[0]["data"]
    assert d["attempt"] == 1
    assert d["error_class"] == "ReadTimeout"
    assert d["model"] == "test-model"
