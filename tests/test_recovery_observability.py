#!/usr/bin/env python3
"""0.30.0: structural / observability.

- recovery tally: arg repairs + corrective retries surface as one footer
  field and one JSONL 'repair' record per applied/rejected repair
- /toolcalls text|native|auto forces the tool-calling mode (the rung past
  the breaker): 'text' re-attaches the grammar and restores tool_choice
  'none' even on a native-confirmed source
- truncated-args salvage: bracket-closure repair for the CACHEABLE READ
  tools only — the write tools keep discard-and-nudge (closure repair would
  invent file content)
- _schemas() caches per tool-list identity without breaking the dynamic seam
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

import nx as m  # noqa: E402
from nx import _argrepair  # noqa: E402


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


# --- salvage ------------------------------------------------------------------

def test_salvage_closes_trailing_brackets():
    from nx._turn import _salvage_truncated_args
    assert _salvage_truncated_args('{"path": "nx", "limit": 4') == {"path": "nx", "limit": 4}
    assert _salvage_truncated_args('{"path": "a", "offsets": [1, 2') == \
        {"path": "a", "offsets": [1, 2]}


def test_salvage_refuses_a_cut_inside_a_string():
    from nx._turn import _salvage_truncated_args
    # the cut landed mid-value: closing would invent the string's tail
    assert _salvage_truncated_args('{"path": "nx/_tu') is None


def test_salvage_refuses_garbage_and_non_dicts():
    from nx._turn import _salvage_truncated_args
    assert _salvage_truncated_args('{"path": "a",,,') is None
    assert _salvage_truncated_args('[1, 2') is None
    assert _salvage_truncated_args('') is None
    assert _salvage_truncated_args(None) is None


def test_native_read_tool_salvages_but_write_tool_does_not(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    nx.YOLO = True
    target = tmp / "t.txt"
    target.write_text("hello")
    messages = [{"role": "system", "content": "s"}]
    # read_file with a truncated tail: salvage closes it and the call runs
    tcs = {0: {"id": "c0", "name": "read_file",
               "args": json.dumps({"path": str(target)})[:-1]}}
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == nx.TURN_TOOL
    assert "hello" in [x for x in messages if x.get("role") == "tool"][0]["content"]
    # write_file with the same shape of truncation: no salvage, straight nudge
    messages2 = [{"role": "system", "content": "s"}]
    tcs2 = {0: {"id": "c1", "name": "write_file",
                "args": json.dumps({"path": str(tmp / "o.txt"), "content": "x"})[:-1]}}
    status2 = _handle_native_tool_calls(messages2, tcs2, "", ["tool_calls"], 0)
    assert status2 == nx.TURN_STREAM_CUT
    assert not (tmp / "o.txt").exists()


# --- /toolcalls mode rung ------------------------------------------------------

def test_toolcalls_text_mode_restores_text_protocol(nx_fresh):
    nx, _tmp = nx_fresh
    from nx import _compress
    nx._state._NATIVE_TOOL_CALLS_CONFIRMED = True
    assert _compress.toolcalls_mode() == "auto"
    # with native confirmed, no grammar attaches
    assert _compress._grammar_extra_body(list(nx.TOOLS)) is None
    assert _compress.set_toolcalls_mode("text") is True
    assert _compress.toolcalls_mode() == "text"
    assert nx._state._NATIVE_TOOL_CALLS_CONFIRMED is False
    # and now the grammar attaches again despite the CAPABILITY_NATIVE probe
    from nx._capability import CAPABILITY_NATIVE
    if nx._state.ACTIVE is not None:
        nx._state.ACTIVE._capability = CAPABILITY_NATIVE
    assert _compress._grammar_extra_body(list(nx.TOOLS)) is not None
    _compress.set_toolcalls_mode("auto")
    assert _compress.toolcalls_mode() == "auto"


def test_toolcalls_native_and_unknown_modes(nx_fresh):
    nx, _tmp = nx_fresh
    from nx import _compress
    nx._state._NATIVE_TOOL_CALLS_CONFIRMED = False
    assert _compress.set_toolcalls_mode("native") is True
    assert nx._state._NATIVE_TOOL_CALLS_CONFIRMED is True
    assert _compress.set_toolcalls_mode("wat") is False
    assert _compress.toolcalls_mode() == "native"  # unchanged by the rejection


def test_toolcalls_cmd_parses_modes(capsys):
    from nx._repl_settings_cmds import _cmd_toolcalls

    class _S:
        messages = []

    from nx import _compress
    assert _compress.set_toolcalls_mode("text")
    _cmd_toolcalls("", _S())
    assert "tool-calling mode: text" in capsys.readouterr().out
    _cmd_toolcalls("auto", _S())
    capsys.readouterr()
    assert _compress.toolcalls_mode() == "auto"
    _cmd_toolcalls("banana", _S())
    assert "text|native|auto" in capsys.readouterr().out


# --- recovery tally ------------------------------------------------------------

def test_repair_args_increments_the_tally():
    st = m._state
    before = st._TALLY_ARG_REPAIRS
    _args, notes = _argrepair.repair_args("read_file", {"path": "a", "limit": "5"})
    assert notes
    assert st._TALLY_ARG_REPAIRS == before + len(notes)


def test_footer_shows_the_tally(capsys):
    st = m._state
    st._TALLY_ARG_REPAIRS = 2
    st._TALLY_RETRIES = 1
    from nx._turn import _print_stats_footer

    class _U:  # minimal usage object
        prompt_tokens = 10
        completion_tokens = 20

    _print_stats_footer(_U(), None, 0, None, 1.0, "", {})
    out = capsys.readouterr().out
    assert "2 args repaired" in out and "1 retry" in out
    st._TALLY_ARG_REPAIRS = 0
    st._TALLY_RETRIES = 0
    _print_stats_footer(_U(), None, 0, None, 1.0, "", {})
    out2 = capsys.readouterr().out
    assert "repaired" not in out2 and "retry" not in out2


def test_model_turn_resets_the_tally(nx_fresh, monkeypatch):
    nx, _tmp = nx_fresh
    st = nx._state
    st._TALLY_ARG_REPAIRS = 9
    st._TALLY_RETRIES = 9

    class _FakeParser:
        usage = None
        timings = None
        streamed_chars = 0
        t_first = None
        tcs = {}
        finish_reasons = ["stop"]
        reasoning_only_chars = 0
        leaked_token_run = 0
        repeat_chunk_run = 0
        reasoning_only_start = None
        reasoning_parts = []

        def gen_window(self, elapsed):
            return elapsed, False

    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: None
    saved = nx.open_stream
    nx.open_stream = lambda messages, **kw: iter([])
    try:
        nx.model_turn([{"role": "system", "content": "s"}])
    finally:
        builtins.print = real_print
        nx.open_stream = saved
    assert st._TALLY_ARG_REPAIRS == 0
    assert st._TALLY_RETRIES == 0


# --- _schemas() cache -----------------------------------------------------------

def test_schemas_cache_follows_tool_list_identity(monkeypatch):
    _argrepair._SCHEMAS_CACHE = None
    base = list(m.TOOLS)
    first = _argrepair._schemas()
    assert "read_file" in first
    # same list object identity -> cache hit (no rebuild)
    assert _argrepair._schemas() is first
    # a whole new list must invalidate: [0] is a different object
    monkeypatch.setattr(m, "TOOLS", base + [{"function": {
        "name": "cachetest", "parameters": {"properties": {"x": {"type": "string"}}}}}])
    second = _argrepair._schemas()
    assert "cachetest" in second
    assert second is not first
    _argrepair._SCHEMAS_CACHE = None
    assert set(_argrepair._schemas()) == {t["function"]["name"] for t in m.TOOLS} | {"final_answer"}
