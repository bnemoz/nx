"""0.132.0 feature #3 — the critique tool: a tier-2 scoped minimal reviewer.

The design contract this file pins:

1. TIER-2, NOT ALWAYS-ON. The `critique` schema is registered in TOOLS but
   gated on NX_SUBAGENTS, so it is withheld from the per-turn floor unless the
   session turned sub-agents on. A reviewer that rides every request is rent on
   every turn; the always-on budget in tests/test_initial_context.py must not
   move for it (that guard's ledger is the single place a new always-on cost is
   justified, and this tool is not one).
2. READ-ONLY. The reviewer's dispatch table has no write tool, no run_tests,
   no consult_advisor, no wait_background: a reviewer that can fix what it finds
   stops being a reviewer, and the parent owns the approval gate.
3. SCOPED. `focus` is required and the read scope is optional but explicit —
   a scope-free review returns noise, which is the failure mode this tool exists
   to avoid.
4. NO WRITE OPT-IN. critique does NOT require NX_SUBAGENT_WRITE=1 (the write
   tiers do): there is nothing to approve.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import nx
import nx._subagent as S
from nx._subagent_tiers import (
    _SubagentContext, _SUBAGENT_TLS, _critique_severity, _critique_tier_dispatch,
    _normalize_tier, _subagent_tools, _READ_ONLY_TOOL_NAMES,
)


def _capture_spawn(monkeypatch):
    """Capture the (instruction, kwargs) critique() hands to spawn_agent."""
    seen = {}

    def fake_spawn(instruction, **kwargs):
        seen["instruction"] = instruction
        seen["kwargs"] = kwargs
        return "spawned"

    monkeypatch.setattr(S, "spawn_agent", fake_spawn)
    return seen


# --- schema, gate, floor ------------------------------------------------------

def test_critique_schema_is_registered_and_gated():
    names = {t["function"]["name"] for t in nx.TOOLS}
    assert "critique" in names
    props = next(t["function"]["parameters"]
                 for t in nx.TOOLS if t["function"]["name"] == "critique")
    # focus is required (the scope is the feature); the severity floor is an
    # enum so a model cannot invent a level the prompt does not understand.
    assert props["required"] == ["focus"]
    assert props["properties"]["severity"]["enum"] == [
        "critical", "major", "minor", "info"]


def test_critique_is_withheld_unless_subagents_are_on(monkeypatch):
    from nx._tool_gates import _active_tools, _gate_closed, _unknown_tool_error

    saved = nx.NX_SUBAGENTS
    try:
        nx.NX_SUBAGENTS = False
        assert _gate_closed("critique") is True
        assert "critique" not in {t["function"]["name"] for t in _active_tools()}
        refusal = _unknown_tool_error("critique")
        assert "NX_SUBAGENTS" in refusal and "reviewer" in refusal
        nx.NX_SUBAGENTS = True
        assert _gate_closed("critique") is False
        assert "critique" in {t["function"]["name"] for t in _active_tools()}
    finally:
        nx.NX_SUBAGENTS = saved


def test_critique_does_not_move_the_always_on_floor():
    """The reviewer must cost 0 tokens on the measured floor. Both halves:
    (a) the initial-context guard measures with NX_SUBAGENTS off, and the
    critique schema is gated out; (b) with it on it rides the existing
    sub-agent opt-in the budget already documents as out of scope."""
    from nx._tool_gates import _active_tools, _invalidate_tool_list_cache

    saved = nx.NX_SUBAGENTS
    try:
        nx.NX_SUBAGENTS = False
        _invalidate_tool_list_cache()
        assert "critique" not in {t["function"]["name"] for t in _active_tools()}
    finally:
        nx.NX_SUBAGENTS = saved
        _invalidate_tool_list_cache()


# --- tier mechanics -----------------------------------------------------------

def test_normalize_tier_accepts_critique():
    assert _normalize_tier("critique") == "critique"
    assert _normalize_tier(" CRITIQUE ") == "critique"
    assert _normalize_tier("bogus") == ""


def test_critique_dispatch_is_read_only(tmp_path, monkeypatch):
    table = _critique_tier_dispatch()
    assert set(table) == set(_READ_ONLY_TOOL_NAMES)
    for forbidden in ("write_file", "edit_file", "multi_edit", "sed_edit",
                      "run_tests", "git_commit", "consult_advisor",
                      "wait_background", "spawn_agent"):
        assert forbidden not in table


def test_critique_tier_offers_only_read_only_schemas(tmp_path, monkeypatch):
    nx.NX_SUBAGENT_WRITE = 1  # even with the write grant, the reviewer cannot write
    ctx = _SubagentContext(_critique_tier_dispatch(), write=False)
    ctx.tier = "critique"
    prev = getattr(_SUBAGENT_TLS, "ctx", None)
    _SUBAGENT_TLS.ctx = ctx
    try:
        offered = {t["function"]["name"] for t in _subagent_tools()}
    finally:
        _SUBAGENT_TLS.ctx = prev
    assert offered == set(_READ_ONLY_TOOL_NAMES)
    assert "critique" not in offered and "consult_advisor" not in offered


def test_critique_severity_floor_defaults_to_major():
    assert _critique_severity(None) == "major"
    assert _critique_severity("") == "major"
    assert _critique_severity("critical") == "critical"
    assert _critique_severity("nonsense") == "major"


# --- the tool body ------------------------------------------------------------

def test_critique_requires_focus(tmp_path, monkeypatch):
    out = S.critique("")
    assert out.startswith("ERROR:") and "focus" in out
    out = S.critique(None)  # ty: ignore[invalid-argument-type]
    assert out.startswith("ERROR:") and "focus" in out


def test_critique_refused_when_subagents_off(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", False)
    out = S.critique("the diff")
    assert out.startswith("ERROR:") and "NX_SUBAGENTS" in out


def test_critique_refuses_nested_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True)
    ctx = _SubagentContext(_critique_tier_dispatch())
    ctx.tier = "critique"
    prev = getattr(_SUBAGENT_TLS, "ctx", None)
    _SUBAGENT_TLS.ctx = ctx
    try:
        out = S.critique("the diff")
    finally:
        _SUBAGENT_TLS.ctx = prev
    assert out.startswith("ERROR:") and "nested" in out


def test_critique_instruction_carries_focus_scope_and_floor(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True)
    seen = _capture_spawn(monkeypatch)
    out = S.critique("the retry loop never exits early", paths="nx/_turn.py",
                     severity="critical")
    assert out == "spawned"
    assert seen["kwargs"]["tier"] == "critique"
    # The focus, the read scope and the severity floor all ride the instruction
    # (no extra spawn signature — tests monkeypatch _run_sync_agent_body).
    text = seen["instruction"]
    assert "the retry loop never exits early" in text
    assert "nx/_turn.py" in text
    assert "critical" in text
    assert "No findings above the floor." in text


def test_critique_accepts_a_list_of_paths_too(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True)
    seen = _capture_spawn(monkeypatch)
    S.critique("the diff", paths=["nx/_turn.py", "nx/_subagent.py"])
    assert "nx/_turn.py, nx/_subagent.py" in seen["instruction"]


def test_critique_does_not_require_the_write_grant(monkeypatch):
    """Only the write tiers need NX_SUBAGENT_WRITE=1; a reviewer cannot write."""
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True)
    monkeypatch.setattr(nx, "NX_SUBAGENT_WRITE", 0)
    seen = _capture_spawn(monkeypatch)
    out = S.critique("the diff")
    assert out == "spawned"
    assert seen["kwargs"]["tier"] == "critique"


def test_critique_severity_default_reaches_the_instruction(monkeypatch):
    monkeypatch.setattr(nx, "NX_SUBAGENTS", True)
    seen = _capture_spawn(monkeypatch)
    S.critique("the diff")
    assert "major" in seen["instruction"]


# --- the reviewer prompt ------------------------------------------------------

def test_critique_system_prompt_pins_the_contract():
    from nx._prompts import CRITIQUE_SYSTEM

    # The three things the prompt must not lose: the no-fix rule, the ranked
    # finding format, and the empty-review escape hatch (a reviewer that pads
    # to look useful is worse than one that reports nothing).
    assert "never fix" in CRITIQUE_SYSTEM
    assert "SEVERITY" in CRITIQUE_SYSTEM and "file:line" in CRITIQUE_SYSTEM
    assert "No findings above the floor." in CRITIQUE_SYSTEM
    assert "severity floor" in CRITIQUE_SYSTEM
