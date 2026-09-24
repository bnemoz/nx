#!/usr/bin/env python3
"""Swarm P13: the tier-2 tool-floor slimming knob (NX_TOOL_FLOOR_SLIM=1).

Withholds the swarm-irrelevant always-on schemas (duckdb, read_image)
behind search_tools so a swarm of N agents doesn't pay ~400 tokens of
schema rent ×N for tools a member rarely calls on turn 1.

Invariants:
- default (env unset): the active set is EXACTLY the pre-P13 always-on set
  (baselines.json's active_tools_floor=20 unchanged);
- with =1: duckdb and read_image are withheld, discoverable via
  search_tools, and dispatch refuses with the gate wording naming the knob;
- the memoized _active_tools() gate-vector reflects the toggle.
"""
import json
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-p13-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import nx as m  # noqa: E402
from nx import _dispatch as d  # noqa: E402


def _names():
    return [t["function"]["name"] for t in d._active_tools()]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("NX_TOOL_FLOOR_SLIM", raising=False)
    m.st._SESSION_HAS_BACKGROUNDED = False
    d._TOOL_LIST_CACHE.clear()
    yield
    d._TOOL_LIST_CACHE.clear()


def test_default_floor_unchanged():
    names = _names()
    assert "duckdb" in names
    assert "search_tools" in names
    assert "read_image" not in names  # NX_VISION off by default anyway


def test_slim_withholds_duckdb_and_read_image(monkeypatch):
    monkeypatch.setenv("NX_TOOL_FLOOR_SLIM", "1")
    m.NX_VISION = True
    m.st._NATIVE_TOOL_CALLS_CONFIRMED = True
    try:
        names = _names()
        assert "duckdb" not in names
        assert "read_image" not in names
        assert "search_tools" in names  # the lifter stays
    finally:
        m.NX_VISION = False
        m.st._NATIVE_TOOL_CALLS_CONFIRMED = False


def test_slimmed_tools_surface_via_search_tools(monkeypatch):
    monkeypatch.setenv("NX_TOOL_FLOOR_SLIM", "1")
    out = d.search_tools()
    assert "duckdb" in out
    hit = d.search_tools("duckdb")
    assert json.loads(hit.split("duckdb: ", 1)[1].split("\n")[0])[
        "function"]["name"] == "duckdb"


def test_slimmed_duckdb_dispatch_refuses_with_gate_wording(monkeypatch):
    monkeypatch.setenv("NX_TOOL_FLOOR_SLIM", "1")
    out = d.run_tool("duckdb", {"sql": "SELECT 1"})
    assert "NX_TOOL_FLOOR_SLIM" in out
    assert "ERROR" in out


def test_gate_vector_reflects_toggle(monkeypatch):
    a = d._active_tools()
    monkeypatch.setenv("NX_TOOL_FLOOR_SLIM", "1")
    b = d._active_tools()
    assert b is not a
    assert "duckdb" not in [t["function"]["name"] for t in b]
    monkeypatch.delenv("NX_TOOL_FLOOR_SLIM", raising=False)
    c = d._active_tools()
    assert "duckdb" in [t["function"]["name"] for t in c]


def test_slim_actually_shrinks_the_floor(monkeypatch):
    def floor():
        return len(json.dumps(d._active_tools()))
    before = floor()
    monkeypatch.setenv("NX_TOOL_FLOOR_SLIM", "1")
    after = floor()
    assert after < before
    # the slimmed set is at least ~250 chars smaller (duckdb's schema alone)
    assert before - after > 250
