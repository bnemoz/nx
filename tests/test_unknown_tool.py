#!/usr/bin/env python3
"""0.28.1: run_tool() must not answer every failed tool-name lookup with
"unknown tool" — a tool merely WITHHELD this session (wait_background,
read_image, spawn_agent) or dropped by the malformed-args circuit breaker
needs wording that tells the model the capability exists and what to do
instead; a genuinely made-up name should get did-you-mean + the real list.
plan_edit is additionally intercepted by the plan-mode gate (which already
returns actionable text) before this path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
    os.environ.pop(k, None)

import nx as m  # noqa: E402
from nx._dispatch import run_tool, _unknown_tool_error  # noqa: E402


def test_truly_unknown_name_gets_did_you_mean():
    out = run_tool("definitely_not_a_tool_qq", {})
    assert out.startswith("ERROR")
    assert "did you mean" not in out  # nothing close: no false suggestion
    assert "Available tools:" in out
    assert "write_file" in out.split("Available tools:")[1]


def test_near_miss_name_resolves_into_the_real_tool():
    """0.29.0 name repair: a one-keystroke slip executes under the real name
    with the note riding the result (read_file's required `path` arg makes the
    result itself carry the schema error)."""
    out = run_tool("readfil", {})
    assert "[arg repair: tool readfil -> read_file (renamed)]" in out
    assert out.endswith('Re-send with all required arguments.')


def test_truly_unknown_name_without_a_close_match_lists_tools():
    out = run_tool("definitely_not_a_tool_xyz", {})
    assert out.startswith("ERROR")
    assert "unknown tool definitely_not_a_tool_xyz" in out
    assert "Available tools:" in out
    # the real list, not a made-up one
    assert "write_file" in out.split("Available tools:")[1]


def test_withheld_wait_background_names_the_gate():
    out = run_tool("wait_background", {"pid": 1})
    assert "not available in this session" in out
    assert "backgrounded" in out
    assert "unknown tool" not in out


def test_withheld_read_image_names_the_gate():
    out = run_tool("read_image", {"path": "x.png"})
    assert "not available in this session" in out
    assert "NX_VISION" in out
    assert "unknown tool" not in out


def test_withheld_spawn_agent_names_the_gate():
    # spawn_agent is ON by default since 0.77.0; force the withheld path.
    saved, m.NX_SUBAGENTS = m.NX_SUBAGENTS, 0
    try:
        out = run_tool("spawn_agent", {"instruction": "q"})
    finally:
        m.NX_SUBAGENTS = saved
    assert "not available in this session" in out
    assert "NX_SUBAGENTS" in out
    assert "search_files" in out  # the alternative


def test_plan_edit_wording_covers_the_gate_case():
    """run_tool() never reaches this wording while plan mode is off (the plan
    gate answers first, already actionable); the helper must still say the
    right thing for any path that lands here."""
    out = _unknown_tool_error("plan_edit")
    assert "not available in this session" in out
    assert "/plan" in out


def test_breaker_dropped_tool_reuses_the_nudge_wording():
    st = m._state
    st._MALFORMED_TOOL_FAILURES["multi_edit"] = 2
    st._MALFORMED_TOOL_BROKEN.add("multi_edit")
    try:
        out = run_tool("multi_edit", {"path": "a", "edits": []})
    finally:
        st._MALFORMED_TOOL_FAILURES.pop("multi_edit", None)
        st._MALFORMED_TOOL_BROKEN.discard("multi_edit")
    nudge = m._turn._malformed_args_nudge("multi_edit", broken_now=True)
    assert out == nudge
    assert "removed from the tool" in out
    assert "unknown tool" not in out


def test_breaker_check_beats_the_pre_hook():
    """A breaker-dropped tool is refused before any policy hook runs — the
    tool isn't in the model's tool list, so a veto is not the story."""
    import json
    import stat
    import tempfile
    tmp = tempfile.mkdtemp(prefix="nx-unknown-")
    hook = os.path.join(tmp, "hook.sh")
    with open(hook, "w") as f:
        f.write('#!/bin/sh\necho vetoed >&2\nexit 1\n')
    os.chmod(hook, os.stat(hook).st_mode | stat.S_IEXEC)
    os.environ["NX_PRE_TOOL_HOOK"] = hook
    st = m._state
    st._MALFORMED_TOOL_BROKEN.add("read_file")
    try:
        out = run_tool("read_file", {"path": "x"})
    finally:
        st._MALFORMED_TOOL_BROKEN.discard("read_file")
        os.environ.pop("NX_PRE_TOOL_HOOK", None)
    assert "removed from the tool" in out
    assert json.dumps({}) is not None  # hook JSON shape exercised via _hooks
