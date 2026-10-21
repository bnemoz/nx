#!/usr/bin/env python3
"""PTC Phase 0 baseline counters (docs/ptc-migration-spec.md Phase 0).

Asserts that the `ptc-baseline` footer counters are threaded from
_handle_native_tool_calls' repair loop / breaker block: malformed-args
splits (repaired-by-candidates / bracket-salvaged / normalized /
hard-failed / breaker-tripped), per-turn tool-call count, fan-out size,
and the envelope token estimate. No behavior change beyond the footer.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
    os.environ.pop(k, None)

import pytest  # noqa: E402


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    importlib.reload(nx)
    yield nx, tmp_path
    from nx import _compress
    _compress.set_toolcalls_mode("auto")


def _reset_ptc(nx):
    st = nx._state if hasattr(nx, "__version__") or hasattr(nx, "TOOLS") else nx
    for k in ("_PTC_TURN_TOOL_CALLS", "_PTC_FANOUT_SIZE", "_PTC_ENVELOPE_CHARS",
              "_PTC_REPAIRED_BY_CANDIDATES", "_PTC_BRACKET_SALVAGED",
              "_PTC_NORMALIZED", "_PTC_HARD_FAILED", "_PTC_BREAKER_TRIPPED"):
        setattr(st, k, 0)


def test_clean_calls_count_and_envelope(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    nx.YOLO = True
    target = tmp / "t.txt"
    target.write_text("hello")
    _reset_ptc(nx)
    tcs = {
        0: {"id": "c0", "name": "read_file", "args": json.dumps({"path": str(target)})},
        1: {"id": "c1", "name": "read_file", "args": json.dumps({"path": str(target)})},
    }
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls([{"role": "system", "content": "s"}], tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    assert nx._state._PTC_TURN_TOOL_CALLS == 2
    assert nx._state._PTC_FANOUT_SIZE == 2  # all-read-only -> parallel batch
    assert nx._state._PTC_ENVELOPE_CHARS == sum(len(c["args"]) for c in tcs.values())
    assert nx._state._PTC_REPAIRED_BY_CANDIDATES == 0
    assert nx._state._PTC_BRACKET_SALVAGED == 0
    assert nx._state._PTC_NORMALIZED == 0
    assert nx._state._PTC_HARD_FAILED == 0
    assert nx._state._PTC_BREAKER_TRIPPED == 0


def test_malformed_args_hard_fail(nx_fresh, tmp_path):
    nx, _tmp = nx_fresh
    nx.YOLO = True
    _reset_ptc(nx)
    tcs = {0: {"id": "c0", "name": "write_file", "args": '{"path": "nx/_tu'}}
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls([{"role": "system", "content": "s"}], tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_STREAM_CUT
    assert nx._state._PTC_TURN_TOOL_CALLS == 1
    assert nx._state._PTC_HARD_FAILED == 1
    assert nx._state._PTC_REPAIRED_BY_CANDIDATES == 0


def test_bracket_salvage_counter(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    nx.YOLO = True
    target = tmp / "t.txt"
    target.write_text("hello")
    _reset_ptc(nx)
    tcs = {0: {"id": "c0", "name": "read_file",
               "args": json.dumps({"path": str(target)})[:-1]}}
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls([{"role": "system", "content": "s"}], tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    assert nx._state._PTC_BRACKET_SALVAGED == 1
    assert nx._state._PTC_HARD_FAILED == 0


def test_normalized_counter(nx_fresh, tmp_path):
    nx, _tmp = nx_fresh
    nx.YOLO = True
    _reset_ptc(nx)
    # double-encoded args: normalize_tool_args folds it (repair candidates
    # don't parse it raw — the string wraps a JSON string)
    inner = json.dumps({"path": "x.txt"})
    tcs = {0: {"id": "c0", "name": "write_file", "args": json.dumps(inner)}}
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls([{"role": "system", "content": "s"}], tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    assert nx._state._PTC_NORMALIZED == 1


def test_breaker_tripped_counter(nx_fresh):
    nx, _tmp = nx_fresh
    nx.YOLO = True
    _reset_ptc(nx)
    nx._state.MALFORMED_TOOL_BREAKER_LIMIT = 1
    tcs = {0: {"id": "c0", "name": "write_file", "args": "not json at all"}}
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls([{"role": "system", "content": "s"}], tcs, "", ["tool_calls"], 5)
    assert status in (nx.TURN_STREAM_CUT, nx._turn._gave_up.__self__ if False else "gave_up")
    assert nx._state._PTC_HARD_FAILED == 1
    assert nx._state._PTC_BREAKER_TRIPPED == 1


def test_footer_field_renders_ptc_baseline():
    from nx import _state as st
    from nx._turn_footer import _ptc_baseline_field
    old = {k: getattr(st, k) for k in
           ("_PTC_TURN_TOOL_CALLS", "_PTC_FANOUT_SIZE", "_PTC_ENVELOPE_CHARS",
            "_PTC_REPAIRED_BY_CANDIDATES", "_PTC_BRACKET_SALVAGED",
            "_PTC_NORMALIZED", "_PTC_HARD_FAILED", "_PTC_BREAKER_TRIPPED")}
    try:
        _reset_ptc(st)
        assert _ptc_baseline_field() == ""
        st._PTC_TURN_TOOL_CALLS = 3
        st._PTC_FANOUT_SIZE = 3
        st._PTC_ENVELOPE_CHARS = 70
        line = _ptc_baseline_field()
        assert line.startswith("ptc-baseline: ")
        assert "3 calls" in line and "fan-out x3" in line
        assert "env tok" in line
        st._PTC_REPAIRED_BY_CANDIDATES = 1
        st._PTC_BRACKET_SALVAGED = 1
        st._PTC_NORMALIZED = 1
        st._PTC_HARD_FAILED = 1
        st._PTC_BREAKER_TRIPPED = 1
        line = _ptc_baseline_field()
        assert "cand-repaired 1" in line and "salvaged 1" in line
        assert "normalized 1" in line and "hard-failed 1" in line
        assert "breaker-tripped 1" in line
        st._PTC_TURN_TOOL_CALLS = 1
        assert "1 call ·" in _ptc_baseline_field()
    finally:
        for k, v in old.items():
            setattr(st, k, v)
