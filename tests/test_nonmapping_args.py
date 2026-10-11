#!/usr/bin/env python3
"""0.29.5: degenerate `arguments` payloads and the pre-hook ordering.

Native and text-protocol parse paths both fold double-encoded JSON strings and
[{name,value}] pair-lists into a plain mapping; an unrecoverable scalar routes
to the malformed-call nudge (TURN_STREAM_CUT / TURN_MALFORMED_CALL) instead of
executing a doomed call. run_tool()'s pre-hook sees the REPAIRED args.
"""
import importlib
import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_PRE_TOOL_HOOK", "NX_POST_TOOL_HOOK"):
    os.environ.pop(k, None)

import pytest  # noqa: E402

import nx as m  # noqa: E402
from nx._text_fallback import normalize_tool_args  # noqa: E402


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
    os.environ.pop("NX_PRE_TOOL_HOOK", None)
    os.environ.pop("NX_POST_TOOL_HOOK", None)


# --- normalize_tool_args unit -------------------------------------------------

def test_double_encoded_string_folds():
    assert normalize_tool_args('{"path": "a.py"}') == {"path": "a.py"}


def test_pair_list_folds():
    assert normalize_tool_args([{"name": "path", "value": "a.py"}]) == {"path": "a.py"}
    assert normalize_tool_args([{"key": "path", "val": 1}]) == {"path": 1}


def test_unrecoverable_shapes_pass_through():
    for bad in ("ab", "12", None, [{"x": 1}], [{"name": "a"}], [1, 2], "[]"):
        assert normalize_tool_args(bad) == bad or normalize_tool_args(bad) is bad


def test_plain_dict_and_empty_list_unchanged():
    assert normalize_tool_args({"path": "a"}) == {"path": "a"}
    assert normalize_tool_args([]) == {}
    # a real string arg is NOT JSON, so it stays a string
    assert normalize_tool_args("some plain text") == "some plain text"


# --- native path --------------------------------------------------------------

def test_native_double_encoded_args_execute(nx_fresh, tmp_path, capsys):
    nx, _tmp = nx_fresh
    target = tmp_path / "t.txt"
    target.write_text("hi")
    m.YOLO = True
    tcs = {0: {"id": "c0", "name": "read_file",
               "args": json.dumps(json.dumps({"path": str(target), "limit": 1}))}}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == m.TURN_TOOL
    tool_msg = [x for x in messages if x.get("role") == "tool"][0]
    assert "hi" in tool_msg["content"]


def test_native_pair_list_args_execute(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    target = tmp_path / "t.txt"
    target.write_text("hi")
    m.YOLO = True
    tcs = {0: {"id": "c0", "name": "read_file",
               "args": json.dumps([{"name": "path", "value": str(target)}])}}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == m.TURN_TOOL
    tool_msg = [x for x in messages if x.get("role") == "tool"][0]
    assert "hi" in tool_msg["content"]


def test_native_scalar_args_route_to_nudge(nx_fresh, capsys):
    nx, _tmp = nx_fresh
    tcs = {0: {"id": "c0", "name": "read_file", "args": json.dumps("ab")}}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == m.TURN_STREAM_CUT
    out = capsys.readouterr().out
    assert "MALFORMED TOOL CALL" in out
    assert "must be a JSON object, got str" in out


# --- text-protocol path -------------------------------------------------------

def test_text_double_encoded_args_execute(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    target = tmp_path / "t.txt"
    target.write_text("hi")
    m.YOLO = True
    inner = json.dumps({"path": str(target), "limit": 1})
    text = m.__dict__.get("_TC", "")
    text = '<tool_call>{"name": "read_file", "arguments": ' + json.dumps(inner) + '}</tool_call>'
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_text_tool_calls
    status = _handle_text_tool_calls(messages, text, malformed_call_count=0)
    assert status == m.TURN_TOOL
    obs = messages[-1]["content"]
    assert "hi" in obs


def test_text_pair_list_args_execute(nx_fresh, tmp_path):
    nx, tmp = nx_fresh
    target = tmp_path / "t.txt"
    target.write_text("hi")
    m.YOLO = True
    text = ('<tool_call>{"name": "read_file", "arguments": '
            + json.dumps([{"name": "path", "value": str(target)}]) + '}</tool_call>')
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_text_tool_calls
    status = _handle_text_tool_calls(messages, text, malformed_call_count=0)
    assert status == m.TURN_TOOL
    assert "hi" in messages[-1]["content"]


# --- pre-hook sees repaired args ----------------------------------------------

def _hook_capturing(tmp, name="hook.sh"):
    out = tmp / (name + ".json")
    p = tmp / name
    p.write_text('#!/bin/sh\ncat > ' + str(out) + '\nexit 0\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p), out


def test_pre_hook_sees_repaired_args(nx_fresh):
    nx, tmp = nx_fresh
    hook, out = _hook_capturing(tmp)
    os.environ["NX_PRE_TOOL_HOOK"] = hook
    from nx._dispatch import run_tool
    run_tool("read_file", {"pth": "/etc/hostname", "limit": "5"})
    payload = json.loads(out.read_text())
    # repaired key + coerced type, exactly what the body would receive
    assert payload["tool"] == "read_file"
    assert payload["args"] == {"path": "/etc/hostname", "limit": 5}


def test_pre_hook_sees_resolved_tool_name(nx_fresh):
    nx, tmp = nx_fresh
    hook, out = _hook_capturing(tmp, name="hook2.sh")
    os.environ["NX_PRE_TOOL_HOOK"] = hook
    from nx._dispatch import run_tool
    run_tool("grep", {"pattern": "x", "path": "."})  # alias -> search_files
    payload = json.loads(out.read_text())
    assert payload["tool"] == "search_files"


def test_pre_hook_veto_still_blocks_after_repair(nx_fresh):
    nx, tmp = nx_fresh
    p = tmp / "veto.sh"
    p.write_text('#!/bin/sh\ncat >/dev/null\necho "no reads on /etc" >&2\nexit 3\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    os.environ["NX_PRE_TOOL_HOOK"] = str(p)
    from nx._dispatch import run_tool
    out = run_tool("read_file", {"pth": "/etc/hostname"})
    assert out.startswith("ERROR: blocked by NX_PRE_TOOL_HOOK")
    assert "no reads on /etc" in out
