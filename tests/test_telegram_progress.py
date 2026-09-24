"""0.95.0: first-and-last status rendering + live sub-agent progress sink."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nx._telegram as tg
import nx._state as st
from nx._progress import note_progress


def test_describe_turn_collapses_duplicate_calls():
    import json
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "list_dir",
             "arguments": json.dumps({"path": "."})}}]},
        {"role": "tool", "tool_call_id": "1", "content": "x" * 300},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "2", "function": {"name": "list_dir",
             "arguments": json.dumps({"path": "."})}}]},
        {"role": "tool", "tool_call_id": "2", "content": "x" * 300},
    ]
    lines, n = tg._describe_turn(msgs, 0)
    assert lines == ["\U0001f4c2 list_dir: . \u00d72"]
    assert n == 2
    # later slices see nothing
    assert tg._describe_turn(msgs, len(msgs)) == ([], 0)


def test_describe_turn_home_shortening_and_bad_args():
    import json
    import os
    home = os.path.expanduser("~")
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "function": {"name": "read_file",
         "arguments": json.dumps({"path": home + "/Documents/Projects/x.rs"})}},
        {"id": "2", "function": {"name": "edit_file", "arguments": "not json"}},
    ]}]
    lines, _ = tg._describe_turn(msgs, 0)
    assert home not in lines[0] and "~/Documents/Projects/x.rs" in lines[0]
    assert "edit_file" in lines[1]          # salvage line, no crash


def test_compose_status_first_and_last():
    lines = ["line%d" % i for i in range(20)]
    out = tg._compose_status("fix the bug", lines, 7, None, 65.0)
    assert "\u2753 fix the bug" in out
    assert "earlier calls" in out
    assert "line19" in out and "line5" not in out
    assert "7 tool calls" in out and "65s" in out
    short = tg._compose_status("q", ["only"], 1, "thinking...", None,
                               "agent turn 3/100")
    assert "total: 1 tool calls" in short
    assert "\U0001f916 agent turn 3/100" in short
    assert "\U0001f4ac thinking..." in short


def test_progress_sink_receives_notes():
    got = []
    st._PROGRESS_SINK = got.append
    try:
        note_progress("agent turn 2/100")
        assert got == ["agent turn 2/100"]
        note_progress("")          # empty clears, sink sees nothing new
        assert got == ["agent turn 2/100", ""]

        def boom(_t):
            raise RuntimeError("sink must not break the caller")
        st._PROGRESS_SINK = boom
        note_progress("still fine")   # a raising sink is swallowed
    finally:
        st._PROGRESS_SINK = None
