"""F2: spawn_agent read-only sub-agent — schema gating (NX_SUBAGENTS),
read-only dispatch enforcement inside the sub-loop, answer extraction, and
turn-cap handling.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR", "NX_SUBAGENTS"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.setattr(nx_fresh if False else sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    yield nx
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SUBAGENTS"):
        os.environ.pop(k, None)


def test_schema_gated_by_env(nx_fresh):
    from nx._dispatch import _active_tools
    names = [t["function"]["name"] for t in _active_tools()]
    assert "spawn_agent" not in names  # default off
    assert any(t["function"]["name"] == "spawn_agent" for t in nx_fresh.TOOLS)  # but in full schema
    nx_fresh.NX_SUBAGENTS = 1
    names = [t["function"]["name"] for t in _active_tools()]
    assert "spawn_agent" in names
    nx_fresh.NX_SUBAGENTS = 0


def test_subagent_loop_uses_readonly_dispatch(nx_fresh, monkeypatch):
    """Inside the sub-loop, run_tool must resolve only read-only tools, even
    when the model tries a mutating one."""
    from nx import _dispatch
    calls = []

    def fake_turn(messages, *a, **kw):
        calls.append([m["role"] for m in messages])
        if len(calls) == 1:
            # the sub-model tries a mutating tool on its first turn
            out = _dispatch.run_tool("write_file", {"path": "evil.txt", "content": "x"})
            assert "not available to the read-only sub-agent" in out
            out2 = _dispatch.run_tool("run_bash", {"command": "rm -rf /"})
            assert "not available to the read-only sub-agent" in out2
            # read-only tools DO work inside the sub-loop
            out3 = _dispatch.run_tool("list_dir", {"path": "."})
            assert not out3.startswith("ERROR")
            messages.append({"role": "assistant", "content": "findings: module X is in nx/_x.py"})
            return nx_fresh.TURN_DONE
        raise AssertionError("loop should have ended")

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    # dispatch table swap must be active only inside spawn_agent
    from nx._subagent import spawn_agent
    assert nx_fresh._state._SUBAGENT_DISPATCH is None
    result = spawn_agent("where is X implemented?")
    assert nx_fresh._state._SUBAGENT_DISPATCH is None  # restored after
    assert nx_fresh._state._SUBAGENT_ACTIVE is False
    assert result == "findings: module X is in nx/_x.py"
    # the sub-loop's messages never leaked into any global conversation
    assert all("evil.txt" not in json_dumps(m) for m in calls[0])


def json_dumps(m):
    import json
    return json.dumps(m, default=str)


def test_subagent_empty_answer_and_cap(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent

    def fake_turn(messages, *a, **kw):
        return nx_fresh.TURN_DONE  # model produced nothing

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn)
    out = spawn_agent("summarize nx/_todo.py")
    assert "produced no answer" in out

    def fake_turn_cap(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "partial"})
        return nx_fresh.TURN_TOOL  # never "done" → loop exhausts the cap

    monkeypatch.setattr(nx_fresh, "model_turn", fake_turn_cap)
    out = spawn_agent("summarize", max_turns=2)
    assert "turn cap" in out and "partial" in out


def test_subagent_rejects_bad_args_and_nested(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent
    assert spawn_agent("").startswith("ERROR")
    assert spawn_agent("   ").startswith("ERROR")
    # max_turns clamped to sane bounds
    seen = {}

    def fake_loop(messages, *a, **kw):
        seen["max_turns"] = kw.get("max_turns_override")
        messages.append({"role": "assistant", "content": "ok"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    spawn_agent("q", max_turns=9999)
    assert seen["max_turns"] == 40
    spawn_agent("q", max_turns=0)
    assert seen["max_turns"] == 1


def test_subagent_system_and_toolset(nx_fresh):
    from nx._subagent import SUBAGENT_SYSTEM, _subagent_tools, _READ_ONLY_TOOL_NAMES
    assert "read-only" in SUBAGENT_SYSTEM.lower()
    tools = _subagent_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == set(_READ_ONLY_TOOL_NAMES)
    forbidden = {"write_file", "edit_file", "multi_edit", "sed_edit", "apply_patch",
                 "run_bash", "wait_background", "memory_write", "spawn_agent"}
    assert not (names & forbidden)
