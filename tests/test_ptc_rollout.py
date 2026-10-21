#!/usr/bin/env python3
"""PTC Phase 4 rollout policy (docs/ptc-migration-spec.md §Phase 4).

Coverage:
  1. Per-model `NX_PTC=auto` heuristic matrix (claude/gpt-5/o3/o4 on;
     gpt-4o/gpt-4.1/unknown off), including negative-list precedence and
     provider-qualified ids.
  2. Resolver precedence: NX_PTC=1 forces on (even for a regressor class and
     over an engaged fallback ladder), NX_PTC=0 forces off (even for claude),
     unset/auto defers to the heuristic answer baked onto nx.NX_PTC.
  3. Fallback ladder: two run_tools breaker TRIPS (open edges) drop the tool
     from the gate for the session; NX_PTC=1 forces it back on; /toolcalls-
     style clear_ptc_ladder() re-arms.
  4. Env docs presence (README/REFERENCE name the knob — the parity test is
     the real gate; this is the fast local tripwire).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
import nx._state as st  # noqa: E402
from nx import _breaker, _ptc_exec  # noqa: E402
from nx._ptc_rollout import (  # noqa: E402
    PTC_TRIP_LIMIT,
    _ptc_enabled,
    clear_ptc_ladder,
    ptc_rollout_enabled,
    resolve_ptc_setting,
)
from nx._tool_gates import _TOOL_GATES, _active_tools  # noqa: E402


@pytest.fixture()
def rollout(monkeypatch):
    """Clean slate: no env knob, no ladder state, watcher re-armed, the
    nx.NX_PTC attribute pinned OFF so a test sees exactly what it installs —
    and the breaker's live containers (identity-preserved, so conftest's
    Tier-A rewind does NOT clear their contents) snapshotted and restored:
    without that the trip this file provokes leaks `run_tools` into the
    broken set for every later test in the process."""
    failures, broken = _breaker._breaker_views()
    saved_failures = dict(failures)
    saved_broken = set(broken)
    from nx import _breaker as _br
    saved_announced = set(_br._TRIP_ANNOUNCED)
    monkeypatch.delenv("NX_PTC", raising=False)
    monkeypatch.delenv("NX_PTC_TIMEOUT", raising=False)
    monkeypatch.delenv("NX_PTC_TIMEOUT_MAX", raising=False)
    monkeypatch.setattr(m, "NX_PTC", 0)
    monkeypatch.setattr(st, "_PTC_DISABLED", False)
    monkeypatch.setattr(st, "_PTC_BREAKER_TRIPS", 0)
    _breaker.rearm_run_tools_trip_watch()
    yield monkeypatch
    failures, broken = _breaker._breaker_views()
    failures.clear()
    failures.update(saved_failures)
    broken.clear()
    broken.update(saved_broken)
    _br._TRIP_ANNOUNCED.clear()
    _br._TRIP_ANNOUNCED.update(saved_announced)
    _breaker.rearm_run_tools_trip_watch()


# --- 1. per-model heuristic matrix ---------------------------------------------

@pytest.mark.parametrize("model_id,expected", [
    # current Anthropic generations: on
    ("claude-sonnet-4-5", True),
    ("claude-opus-4-1", True),
    ("CLAUDE-HAIKU-4", True),           # case-insensitive
    # recent OpenAI reasoning generations: on
    ("gpt-5", True),
    ("gpt-5-mini", True),
    ("o3", True),
    ("o3-mini", True),
    ("o4-mini", True),
    # gpt-4o / gpt-4.1 class regress per arXiv 2608.06370: off
    ("gpt-4o", False),
    ("gpt-4o-mini", False),
    ("gpt-4.1", False),
    ("gpt-4.1-nano", False),
    ("gpt-4-turbo", False),
    ("o1", False),
    # unknown generations stay OFF (JSON stays the default-off-ramp)
    ("glm-5.3", False),
    ("deepseek-v4-pro", False),
    ("qwen3.8-flash-next", False),
    ("", False),
    (None, False),                       # falls to st.MODEL, empty in tests
    # provider-qualified ids classify on the final segment
    ("together:claude-sonnet-4-5", True),
    ("openrouter/o3", True),
    ("zai-org/GLM-5.3", False),
])
def test_auto_heuristic_matrix(rollout, model_id, expected):
    assert ptc_rollout_enabled(model_id) is expected


def test_negative_list_wins_over_positive(rollout):
    # a hypothetical id that matches a positive prefix AND a negative one:
    # gpt-4o* also starts with "gpt-" but never with a positive prefix;
    # the real precedence check is the 4-class family + unknown fallback.
    assert ptc_rollout_enabled("gpt-4o5") is False
    assert ptc_rollout_enabled("gpt-45") is False


def test_resolve_setting_reads_env_first(rollout):
    monkeypatch = rollout
    monkeypatch.setenv("NX_PTC", "1")
    assert resolve_ptc_setting("gpt-4o") == 1       # override beats heuristic
    monkeypatch.setenv("NX_PTC", "0")
    assert resolve_ptc_setting("claude-sonnet-4-5") == 0
    monkeypatch.delenv("NX_PTC")
    assert resolve_ptc_setting("claude-sonnet-4-5") == 1
    assert resolve_ptc_setting("gpt-4o") == 0
    assert resolve_ptc_setting("not-a-model") == 0
    monkeypatch.setenv("NX_PTC", "auto")            # explicit auto == unset
    assert resolve_ptc_setting("o3") == 1


# --- 2. resolver precedence ------------------------------------------------------

def test_env_1_forces_on_even_for_a_regressor(rollout):
    monkeypatch = rollout
    monkeypatch.setenv("NX_PTC", "1")
    monkeypatch.setattr(m, "NX_PTC", 0)             # attribute says off
    assert _ptc_enabled() is True
    assert _TOOL_GATES["run_tools"]() is True


def test_env_0_forces_off_even_for_claude(rollout):
    monkeypatch = rollout
    monkeypatch.setenv("NX_PTC", "0")
    monkeypatch.setattr(m, "NX_PTC", 1)             # attribute says on
    assert _ptc_enabled() is False
    assert _TOOL_GATES["run_tools"]() is False


def test_auto_follows_the_baked_attribute(rollout):
    # unset env: the resolver mirrors the heuristic's answer as baked onto
    # nx.NX_PTC (at import by __init__ / on a model switch by switch_source)
    monkeypatch = rollout
    monkeypatch.setattr(m, "NX_PTC", 1)
    assert _ptc_enabled() is True
    monkeypatch.setattr(m, "NX_PTC", 0)
    assert _ptc_enabled() is False


def test_ptc_exec_reader_sees_the_attribute(rollout):
    # _ptc_exec._ptc_enabled (note loading, stub staging, _dispatch wiring)
    # and the gate agree on the attribute state — the Phase 4 mirror keeps
    # both readers consistent through a ladder drop.
    monkeypatch = rollout
    monkeypatch.setattr(m, "NX_PTC", 1)
    assert _ptc_exec._ptc_enabled() is True
    assert _TOOL_GATES["run_tools"]() is True


# --- 3. fallback ladder (double trip) ---------------------------------------------

def _trip_once(monkeypatch):
    """One run_tools breaker OPEN via the real PTC trip path
    (_ptc_exec._breaker_increment with limit 1), then let the gate vector
    re-read the breaker so the ladder's edge detector sees the open."""
    monkeypatch.setattr(st, "MALFORMED_TOOL_BREAKER_LIMIT", 1)
    _ptc_exec._breaker_increment()
    _breaker._breaker_views()          # the gate path's read = the edge check


def test_double_trip_drops_run_tools_for_the_session(rollout):
    monkeypatch = rollout
    monkeypatch.setattr(m, "NX_PTC", 1)
    _trip_once(monkeypatch)
    # first trip: tool stays offered (one trip is evidence, not a drop)
    assert st._PTC_BREAKER_TRIPS == 1
    assert st._PTC_DISABLED is False
    assert _TOOL_GATES["run_tools"]() is True
    # a decay/reset re-arms the breaker, then a SECOND open trips the ladder
    _breaker._breaker_views()[0].clear()
    _breaker._breaker_views()[1].discard("run_tools")
    _trip_once(monkeypatch)
    assert st._PTC_BREAKER_TRIPS == PTC_TRIP_LIMIT
    assert st._PTC_DISABLED is True
    # the gate now refuses run_tools even though the attribute still says on
    assert _TOOL_GATES["run_tools"]() is False
    assert _ptc_enabled() is False
    # run_tools left the offered list (cache was invalidated by the hook)
    assert "run_tools" not in {t["function"]["name"] for t in _active_tools()}


def test_env_1_re_enables_after_the_drop(rollout):
    monkeypatch = rollout
    monkeypatch.setattr(m, "NX_PTC", 1)
    _trip_once(monkeypatch)
    _breaker._breaker_views()[0].clear()
    _breaker._breaker_views()[1].discard("run_tools")
    _trip_once(monkeypatch)
    assert st._PTC_DISABLED is True
    assert _TOOL_GATES["run_tools"]() is False
    # the documented reversible knob: NX_PTC=1 forces it back on (the drop
    # itself stays recorded — a model switch or /toolcalls reset clears it)
    monkeypatch.setenv("NX_PTC", "1")
    assert _ptc_enabled() is True
    assert _TOOL_GATES["run_tools"]() is True


def test_ladder_clear_rearms(rollout):
    monkeypatch = rollout
    monkeypatch.setattr(m, "NX_PTC", 1)
    _trip_once(monkeypatch)
    _breaker._breaker_views()[0].clear()
    _breaker._breaker_views()[1].discard("run_tools")
    _trip_once(monkeypatch)
    assert st._PTC_DISABLED is True
    # what /toolcalls reset does (the nx.NX_PTC mirror made it 0; restore the
    # heuristic's answer the way a model switch would):
    clear_ptc_ladder()
    monkeypatch.setattr(m, "NX_PTC", 1)
    assert st._PTC_DISABLED is False and st._PTC_BREAKER_TRIPS == 0
    assert _TOOL_GATES["run_tools"]() is True


def test_single_open_with_repeated_failures_does_not_drop(rollout):
    # repeated failures against ONE open breaker are one trip, not two
    # (the spec counts TRIPS — breaker opens — not failures):
    monkeypatch = rollout
    monkeypatch.setattr(m, "NX_PTC", 1)
    monkeypatch.setattr(st, "MALFORMED_TOOL_BREAKER_LIMIT", 1)
    _ptc_exec._breaker_increment()
    _breaker._breaker_views()
    _ptc_exec._breaker_increment()   # same open streak, no new edge
    _breaker._breaker_views()
    assert st._PTC_BREAKER_TRIPS == 1
    assert st._PTC_DISABLED is False


# --- 4. env docs presence (fast local tripwire) -----------------------------------

def test_ptc_env_vars_are_documented():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = open(os.path.join(root, "docs", "REFERENCE.md"), encoding="utf-8").read()
    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    example = open(os.path.join(root, "sources.example.env"),
                   encoding="utf-8").read()
    for var in ("NX_PTC", "NX_PTC_TIMEOUT", "NX_PTC_TIMEOUT_MAX"):
        assert var in ref, f"{var} missing from the REFERENCE env table"
    assert "NX_PTC" in readme, "NX_PTC missing from the README env table row"
    for var in ("NX_PTC", "NX_PTC_TIMEOUT", "NX_PTC_TIMEOUT_MAX"):
        assert var in example, f"{var} missing from sources.example.env"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
