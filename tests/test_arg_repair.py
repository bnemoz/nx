#!/usr/bin/env python3
"""Schema-level argument repair (nx/_argrepair.py, wired into run_tool).

The layer below JSON repair: args that PARSE but don't match the tool's
declared schema. Coercions and renames must be unambiguous to apply, every
applied fix is reported in the note list (and, through run_tool, in the tool
result), and a disabled gate / unknown tool / non-dict args passes through
untouched. No model or network round-trip.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
from nx import _argrepair  # noqa: E402


@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv("NX_ARG_REPAIR", raising=False)
    yield
    monkeypatch.delenv("NX_ARG_REPAIR", raising=False)


# --- coercion ---------------------------------------------------------------

def test_string_integer_coerced_with_notes(clean_env):
    args, notes = _argrepair.repair_args(
        "read_file", {"path": "nx/_turn.py", "offset": "10", "limit": "5"})
    assert args == {"path": "nx/_turn.py", "offset": 10, "limit": 5}
    assert notes == ['offset: "10" -> 10', 'limit: "5" -> 5']


def test_string_boolean_coerced(clean_env):
    args, notes = _argrepair.repair_args("git_diff", {"path": ".", "staged": "true"})
    assert args == {"path": ".", "staged": True}
    assert notes == ['staged: "true" -> True']


def test_integer_to_string_coerced(clean_env):
    args, notes = _argrepair.repair_args("write_file", {"path": 42, "content": "x"})
    assert args == {"path": "42", "content": "x"}
    assert notes == ['path: 42 -> "42"']


def test_negative_and_signed_integers(clean_env):
    args, _n = _argrepair.repair_args("read_file", {"path": "a", "offset": "-3"})
    assert args["offset"] == -3
    args, _n = _argrepair.repair_args("read_file", {"path": "a", "offset": "+7"})
    assert args["offset"] == 7


def test_nonnumeric_string_left_alone(clean_env):
    args, notes = _argrepair.repair_args("read_file", {"path": "a", "offset": "1e3"})
    assert args["offset"] == "1e3"  # "1e3" names line 1e3, not line 1000
    assert notes == []


def test_garbage_string_untouched(clean_env):
    args, notes = _argrepair.repair_args("run_bash", {"command": "ls", "timeout": "soon"})
    assert args["timeout"] == "soon"
    assert notes == []


# --- enum case --------------------------------------------------------------

def test_enum_case_folded_when_unique(clean_env):
    args, notes = _argrepair.repair_args(
        "todo_write", {"todos": [{"text": "a", "status": "In_Progress"}]})
    assert args["todos"][0]["status"] == "in_progress"
    assert notes == ['todos[0].status: "In_Progress" -> "in_progress"']


def test_enum_ambiguous_case_left_alone(clean_env, monkeypatch):
    monkeypatch.setattr(m, "TOOLS", [{"function": {
        "name": "ambig_enum",
        "parameters": {"properties": {"mode": {"enum": ["read", "READ"]}}},
    }}])
    args, notes = _argrepair.repair_args("ambig_enum", {"mode": "Read"})
    assert args == {"mode": "Read"}
    assert notes == []


# --- key rename -------------------------------------------------------------

def test_near_miss_key_renamed(clean_env):
    args, notes = _argrepair.repair_args("search_files", {"maxresults": 3, "pattern": "x"})
    assert args == {"pattern": "x", "max_results": 3}
    assert notes == ["maxresults -> max_results (renamed)"]


def test_rename_never_clobbers_present_key(clean_env):
    args, notes = _argrepair.repair_args("edit_file", {"path": "p", "old": "a", "Old": "b"})
    assert args == {"path": "p", "old": "a", "Old": "b"}
    assert notes == []


def test_rename_ambiguous_between_two_properties_skipped(clean_env, monkeypatch):
    monkeypatch.setattr(m, "TOOLS", [{"function": {
        "name": "ambig",
        "parameters": {"properties": {"tooll": {"type": "string"},
                                      "toolm": {"type": "string"}}},
    }}])
    args, notes = _argrepair.repair_args("ambig", {"toolx": "v"})
    assert args == {"toolx": "v"}
    assert notes == []


def test_unrelated_key_left_alone(clean_env):
    args, notes = _argrepair.repair_args("read_file", {"path": "a", "zzzzzzzz": 1})
    assert args == {"path": "a", "zzzzzzzz": 1}
    assert notes == []


# --- JSON-encoded array ------------------------------------------------------

def test_json_encoded_array_parsed(clean_env):
    args, notes = _argrepair.repair_args(
        "todo_write", {"todos": '[{"text": "a", "status": "pending"}]'})
    assert args == {"todos": [{"text": "a", "status": "pending"}]}
    assert notes


def test_non_array_json_string_left_alone(clean_env):
    args, notes = _argrepair.repair_args("todo_write", {"todos": '{"text": "a"}'})
    assert args == {"todos": '{"text": "a"}'}
    assert notes == []


# --- pass-through cases ------------------------------------------------------

def test_missing_required_not_invented(clean_env):
    args, notes = _argrepair.repair_args("read_file", {"limit": "5"})
    assert args == {"limit": 5}
    assert notes == ['limit: "5" -> 5']  # coerced, but `path` never fabricated


def test_unknown_tool_untouched(clean_env):
    assert _argrepair.repair_args("zzz", {"a": "1"}) == ({"a": "1"}, [])


def test_non_dict_and_empty_args_untouched(clean_env):
    assert _argrepair.repair_args("read_file", None) == (None, [])
    assert _argrepair.repair_args("read_file", "x=1") == ("x=1", [])
    assert _argrepair.repair_args("read_file", {}) == ({}, [])


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("NX_ARG_REPAIR", "0")
    assert _argrepair.enabled() is False
    assert _argrepair.repair_args("read_file", {"offset": "10"}) == ({"offset": "10"}, [])


def test_schemas_cover_all_tools_and_final_answer(clean_env):
    """Every dispatchable tool (and final_answer) must be repairable: a tool
    whose schema _schemas() can't read gets NO repair — that silence is a bug."""
    names = {t["function"]["name"] for t in m.TOOLS} | {"final_answer"}
    assert set(_argrepair._schemas()) == names


# --- tool-name resolution (0.29.0) -------------------------------------------

def test_resolve_exact_name_is_passthrough():
    assert _argrepair.resolve_tool_name("read_file") == ("read_file", None)


def test_resolve_similar_name_renamed():
    got, note = _argrepair.resolve_tool_name("readfile")
    assert got == "read_file"
    assert note and "renamed" in note
    got, note = _argrepair.resolve_tool_name("ReadFile")
    assert got == "read_file"


def test_resolve_alias_table():
    got, note = _argrepair.resolve_tool_name("bash")
    assert got == "run_bash"
    assert note and "aliased" in note
    assert _argrepair.resolve_tool_name("grep")[0] == "search_files"
    assert _argrepair.resolve_tool_name("str_replace_editor")[0] == "edit_file"
    assert _argrepair.resolve_tool_name("cat")[0] == "read_file"
    assert _argrepair.resolve_tool_name("ls")[0] == "list_dir"


def test_resolve_ambiguous_never_guesses(monkeypatch):
    """Two equally-close schema names -> None (the unknown-tool error lists
    the candidates instead); a single near-miss still resolves."""
    monkeypatch.setattr(m, "TOOLS", [
        {"function": {"name": "tool_aaa", "parameters": {"properties": {"a": {"type": "string"}}}}},
        {"function": {"name": "tool_aab", "parameters": {"properties": {"a": {"type": "string"}}}}},
    ])
    assert _argrepair.resolve_tool_name("tool_aax") == (None, None)
    assert _argrepair.resolve_tool_name("zzzzzzzzzz") == (None, None)


def test_alias_never_shadows_a_real_tool_name(monkeypatch):
    """If a future tool is really named `bash`, the alias must not steal it."""
    monkeypatch.setattr(m, "TOOLS", [
        {"function": {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}}}},
        {"function": {"name": "run_bash", "parameters": {"properties": {"command": {"type": "string"}}}}},
    ])
    assert _argrepair.resolve_tool_name("bash") == ("bash", None)


# --- arg alias map (0.29.0) ---------------------------------------------------

def test_arg_alias_cmd_to_command():
    args, notes = _argrepair.repair_args("run_bash", {"cmd": "ls"})
    assert args == {"command": "ls"}
    assert notes == ["cmd -> command (renamed)"]


def test_arg_alias_old_str_to_old():
    args, notes = _argrepair.repair_args(
        "edit_file", {"path": "p", "old_str": "a", "new_str": "b"})
    assert args == {"path": "p", "old": "a", "new": "b"}


def test_arg_alias_never_clobbers_a_real_key():
    args, notes = _argrepair.repair_args(
        "edit_file", {"path": "p", "old": "keep", "old_str": "drop"})
    assert args["old"] == "keep"
    assert args["old_str"] == "drop"  # unknown key, no clobber, no rename
    assert notes == []


def test_arg_alias_respects_declared_properties_only(clean_env, monkeypatch):
    """An alias pointing at a key the tool doesn't declare must not fire."""
    monkeypatch.setattr(m, "TOOLS", [
        {"function": {"name": "weird", "parameters": {"properties": {"x": {"type": "string"}}}}},
    ])
    args, notes = _argrepair.repair_args("weird", {"file": "p"})  # "file" aliases to "path" elsewhere
    assert args == {"file": "p"}
    assert notes == []


# --- required-args pre-check (0.29.0) -----------------------------------------

def test_check_required_message():
    err = _argrepair.check_required("read_file", {"limit": 5})
    assert err is not None
    assert err.startswith('ERROR: read_file missing required "path"')
    assert "schema: path:" in err
    assert "Re-send with all required arguments" in err


def test_check_required_passes_when_present():
    assert _argrepair.check_required("read_file", {"path": "x"}) is None
    # a None value counts as missing (the body can't use it either)
    assert _argrepair.check_required("read_file", {"path": None}) is not None
    assert _argrepair.check_required("list_dir", {}) is None  # nothing required


def test_check_required_after_rename_counts_repaired_key():
    """The pre-check runs on the REPAIRED args, so an aliased key satisfies it."""
    err = _argrepair.check_required("run_bash", {"command": "ls"})
    assert err is None


def test_run_tool_missing_required_reports_not_executes(clean_env, monkeypatch, capsys):
    ran = []

    def fake_tool(path, limit=0):
        ran.append(path)
        return "ok"

    monkeypatch.setattr(m, "TOOLS", m.TOOLS + [{"function": {
        "name": "faketool3",
        "parameters": {"properties": {"path": {"type": "string"},
                                      "limit": {"type": "integer"}},
                       "required": ["path"]}}},
    ])
    import nx._dispatch as dispatch
    monkeypatch.setattr(dispatch, "DISPATCH", dict(dispatch.DISPATCH, faketool3=fake_tool))
    m.YOLO = True
    out = dispatch.run_tool("faketool3", {"limit": "5"})
    assert ran == []  # the body never ran
    assert 'missing required "path"' in out
    assert "schema: path: string, limit: integer" in out


def test_run_tool_name_repair_executes_real_tool_with_note(clean_env, monkeypatch, capsys):
    seen = {}

    def fake_tool(path):
        seen["path"] = path
        return "ok"

    monkeypatch.setattr(m, "TOOLS", m.TOOLS + [{"function": {
        "name": "faketool4",
        "parameters": {"properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    ])
    import nx._dispatch as dispatch
    monkeypatch.setattr(dispatch, "DISPATCH", dict(dispatch.DISPATCH, faketool4=fake_tool))
    m.YOLO = True
    out = dispatch.run_tool("faketool4x", {"path": "p"})  # one near-miss away
    assert seen == {"path": "p"}
    assert "[arg repair: tool faketool4x -> faketool4" in out
    # ambiguity still reports instead of guessing (nothing close: plain list)
    out2 = dispatch.run_tool("zzzzzzzzzz", {"path": "p"})
    assert out2.startswith("ERROR: unknown tool zzzzzzzzzz")
    assert "Available tools:" in out2 and "faketool4" in out2


# --- run_tool integration ----------------------------------------------------

def test_run_tool_repairs_args_and_notes_result(clean_env, monkeypatch, capsys):
    seen = {}

    def fake_tool(path, limit=0):
        seen.update(path=path, limit=limit)
        return "ok"

    monkeypatch.setattr(m, "TOOLS", m.TOOLS + [{"function": {
        "name": "faketool",
        "parameters": {"properties": {"path": {"type": "string"},
                                      "limit": {"type": "integer"}},
                       "required": ["path"]}}},
    ])
    import nx._dispatch as dispatch
    monkeypatch.setattr(dispatch, "DISPATCH", dict(dispatch.DISPATCH, faketool=fake_tool))
    m.YOLO = True
    result = dispatch.run_tool("faketool", {"limit": "5", "pth": "x.txt"})
    assert seen == {"path": "x.txt", "limit": 5}
    assert result.startswith("[arg repair: limit: \"5\" -> 5, pth -> path (renamed)] ok")


def test_run_tool_no_note_when_args_already_valid(clean_env, monkeypatch, capsys):
    monkeypatch.setattr(m, "TOOLS", m.TOOLS + [{"function": {
        "name": "faketool2",
        "parameters": {"properties": {"path": {"type": "string"}}}}},
    ])
    import nx._dispatch as dispatch
    monkeypatch.setattr(dispatch, "DISPATCH", dict(
        dispatch.DISPATCH, faketool2=lambda path: "ok"))
    m.YOLO = True
    assert dispatch.run_tool("faketool2", {"path": "x"}) == "ok"


def test_native_turn_repairs_before_execution(clean_env, monkeypatch, capsys, tmp_path):
    """End-to-end: a streamed native tool_call whose args are JSON but
    wrong-typed still executes, with the note riding the result."""
    target = tmp_path / "t.txt"
    target.write_text("hello")
    tcs = {0: {"id": "c0", "name": "read_file",
               "args": json.dumps({"path": str(target), "limit": "2"})}}
    messages = [{"role": "system", "content": "s"}]
    from nx._turn import _handle_native_tool_calls
    status = _handle_native_tool_calls(messages, tcs, "", ["tool_calls"], 0)
    assert status == m.TURN_TOOL
    tool_msg = [x for x in messages if x.get("role") == "tool"][0]
    assert tool_msg["content"].startswith("[arg repair: limit: \"2\" -> 2]")
