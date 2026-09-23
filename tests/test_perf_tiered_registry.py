#!/usr/bin/env python3
"""Tiered tool registry + search_tools meta-tool (0.62.0, PERF §6 items 1-2).

_active_tools() is a declarative gate table (_TOOL_GATES) memoized on the
gate-vector instead of five sequential O(N) filter passes; search_tools lifts
a withheld schema into context on demand so a future always-on tool can ship
tier-2 (0 tokens until used).
"""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _dispatch as d  # noqa: E402


def _names():
    return [t["function"]["name"] for t in d._active_tools()]


def test_gate_table_covers_the_old_withheld_set():
    """Every tool the old hand-written filters withheld has a declared gate —
    a gate added in code without a _TOOL_GATES entry silently becomes
    always-on (~150 tok of permanent rent)."""
    for name in ("wait_background", "spawn_agent", "read_image", "plan_edit"):
        assert name in d._TOOL_GATES, f"{name} lost its gate entry"


def test_default_withholds_the_gated_four():
    saved = (m.NX_SUBAGENTS, m.NX_VISION)
    m.NX_SUBAGENTS = m.NX_VISION = False
    try:
        names = _names()
    finally:
        m.NX_SUBAGENTS, m.NX_VISION = saved
    assert "wait_background" not in names
    assert "spawn_agent" not in names
    assert "read_image" not in names
    assert "plan_edit" not in names
    assert "search_tools" in names  # the meta-tool itself is always on
    # and the always-on set is unchanged: one entry per non-gated schema
    all_names = [t["function"]["name"] for t in m.TOOLS]
    assert sorted(names) == sorted(
        n for n in all_names if n not in ("wait_background", "spawn_agent",
                                          "read_image", "plan_edit",
                                          "obs_recall",
                                          "lint_check"))  # web_fetch is ON by default (0.71.0), so it stays in names; lint_check is OFF by default (0.94.0)


def test_memoized_on_the_gate_vector():
    nx_st = m.st
    nx_st._SESSION_HAS_BACKGROUNDED = False
    a = d._active_tools()
    assert d._active_tools() is a, "same gate-vector must return the cached list"
    nx_st._SESSION_HAS_BACKGROUNDED = True
    b = d._active_tools()
    assert b is not a
    assert "wait_background" in _names()
    assert "wait_background" not in [t["function"]["name"] for t in a]
    nx_st._SESSION_HAS_BACKGROUNDED = False


def test_breaker_veto_still_drops_a_tool():
    saved = dict(m.st._MALFORMED_TOOL_BROKEN)
    m.st._MALFORMED_TOOL_BROKEN = {"sed_edit": 3}
    try:
        assert "sed_edit" not in _names()
    finally:
        m.st._MALFORMED_TOOL_BROKEN.clear()
        m.st._MALFORMED_TOOL_BROKEN.update(saved)


def test_search_tools_lists_withheld_and_not_active():
    m.st._SESSION_HAS_BACKGROUNDED = False
    out = d.search_tools()
    assert "wait_background" in out
    active = {t["function"]["name"] for t in d._active_tools()}
    listed_names = {line.split(": ", 1)[0] for line in out.splitlines()
                    if ": {" in line}
    assert not (active & listed_names), f"active tools listed: {active & listed_names}"
    hit = d.search_tools("wait_background")
    assert "wait_background" in hit
    # the schema it surfaces must be the real one, usable to call it
    assert json.loads(hit.split("wait_background: ", 1)[1].split("\n")[0])[
        "function"]["name"] == "wait_background"
    hit_lines = d.search_tools("wait_background").splitlines()
    assert any(line.startswith("wait_background: ") for line in hit_lines)
    assert not any(line.startswith("spawn_agent: ") for line in hit_lines)
    assert "no withheld tools matched" in d.search_tools("zzz_no_such_tool").lower()


def test_dispatch_knows_search_tools():
    assert "search_tools" in d.DISPATCH
    out = d.run_tool("search_tools", {"pattern": "plan_edit"})
    assert "plan_edit" in out
