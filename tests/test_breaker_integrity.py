#!/usr/bin/env python3
"""0.40 (ROADMAP 1.3): breaker integrity.

The malformed-args breaker is the recovery stack's only tool-level circuit
breaker, and three of its edges were wrong:

- run_tool() checked `_MALFORMED_TOOL_BROKEN` against the EMITTED name, while
  alias/similarity resolution happened after it — so a dropped `multi_edit`
  stayed reachable as `multi_edits`.
- Name repair resolved against the full `TOOLS` list, so a typo could land on a
  tool that was never offered this turn (a withheld `spawn_agent`).
- The breaker was process-global: switching to a healthy endpoint left the
  model permanently missing tools that endpoint handles fine.

Plus the config landmine: `_state.MALFORMED_TOOL_BREAKER_LIMIT` defaulted to 2
while `__init__.py` assigns 3, so anything reading it before `__init__` ran got
a different number than the one the README documents.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SUBAGENTS", "NX_VISION",
          "NX_MALFORMED_TOOL_BREAKER"):
    os.environ.pop(k, None)

import nx as m  # noqa: E402
import nx._state as st  # noqa: E402
from nx._dispatch import run_tool, _breaker_views, _active_tools  # noqa: E402
from nx import _argrepair  # noqa: E402


def _tool_names():
    return [t["function"]["name"] for t in _active_tools()]


# --- the limit has exactly one owner ------------------------------------------
def test_state_no_longer_carries_a_conflicting_default():
    """The old `= 2` in _state.py was a live landmine: test_breaker_ergonomics
    only passed because __init__ happens to run first."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "nx", "_state.py")).read()
    assert "MALFORMED_TOOL_BREAKER_LIMIT: int = " not in src
    # The annotation-only declaration survives, so a read before __init__
    # assigns is an AttributeError, not a silently different limit.
    assert "MALFORMED_TOOL_BREAKER_LIMIT: int" in src
    assert m.MALFORMED_TOOL_BREAKER_LIMIT == 3


# --- breaker check covers the RESOLVED name ----------------------------------
def test_broken_tool_is_unreachable_under_a_near_miss_spelling():
    """The bug: `_MALFORMED_TOOL_BROKEN` was checked against the emitted name
    only, so a broken multi_edit ran when the model typed `multi_edits`."""
    st._MALFORMED_TOOL_BROKEN.add("multi_edit")
    try:
        assert _argrepair.resolve_tool_name("multi_edits")[0] == "multi_edit"
        out = run_tool("multi_edits", {"path": "a", "edits": []})
        assert "removed from the tool" in out
        assert "multi_edit" in out
    finally:
        st._MALFORMED_TOOL_BROKEN.discard("multi_edit")


def test_broken_tool_is_unreachable_through_its_alias():
    st._MALFORMED_TOOL_BROKEN.add("run_bash")
    try:
        out = run_tool("bash", {"command": "true"})
        assert "removed from the tool" in out
    finally:
        st._MALFORMED_TOOL_BROKEN.discard("run_bash")


def test_exact_name_still_refused_before_resolution():
    """The pre-existing edge stays: an exact hit is refused without ever
    reaching the dispatch table."""
    st._MALFORMED_TOOL_BROKEN.add("list_dir")
    try:
        out = run_tool("list_dir", {"path": "."})
        assert "removed from the tool" in out
    finally:
        st._MALFORMED_TOOL_BROKEN.discard("list_dir")


# --- name repair is scoped to what was offered -------------------------------
def test_repaired_name_cannot_land_on_a_withheld_tool():
    """The worse half of the same bug: the session gates at the top of
    run_tool() match on the EMITTED name, so `spawn_agnet` repaired onto the
    real spawn_agent and EXECUTED it with sub-agents switched off."""
    saved = m.NX_SUBAGENTS
    m.NX_SUBAGENTS = 0  # the gate reads this at call time
    try:
        assert "spawn_agent" not in _tool_names()
        assert "spawn_agent" in _argrepair._tool_names()  # in the full schema set
        assert _argrepair.resolve_tool_name("spawn_agnet")[0] == "spawn_agent"
        out = run_tool("spawn_agnet", {"instruction": "q"})
        assert "NX_SUBAGENTS" in out  # the gate's own wording, and nothing ran
    finally:
        m.NX_SUBAGENTS = saved


def test_exact_withheld_name_still_reaches_its_gate_message():
    """The pre-existing exact-name gate wording is unchanged."""
    saved = m.NX_SUBAGENTS
    m.NX_SUBAGENTS = 0
    try:
        out = run_tool("spawn_agent", {"instruction": "q"})
        assert "NX_SUBAGENTS" in out
    finally:
        m.NX_SUBAGENTS = saved


def test_repaired_name_still_works_for_an_offered_tool():
    """The 0.29.0 repair path must keep working: a real near-miss executes the
    real tool and the note rides the result."""
    out = run_tool("readflie", {"path": "README.md"})
    assert "renamed" in out or "# " in out  # note, or the file body itself


# --- breaker state is per (source, model) ------------------------------------
class _FakeSource:
    def __init__(self, name, model):
        self.name = name
        self.model = model


def test_breaker_clears_when_the_endpoint_changes():
    saved = (st.ACTIVE, st.MODEL, st._BREAKER_ENDPOINT_KEY,
             dict(st._MALFORMED_TOOL_FAILURES), set(st._MALFORMED_TOOL_BROKEN))
    try:
        st.ACTIVE = _FakeSource("flaky", "glm-5.3")
        st.MODEL = "glm-5.3"
        failures, broken = _breaker_views()
        failures["multi_edit"] = 3
        broken.add("multi_edit")
        st._BREAKER_ENDPOINT_KEY = ("flaky", "glm-5.3")
        assert "multi_edit" in _breaker_views()[1]
        assert "multi_edit" not in _tool_names()

        # Switch to a healthy source: the schema comes back with no nudge.
        st.ACTIVE = _FakeSource("healthy", "qwen3-coder")
        st.MODEL = "qwen3-coder"
        assert _breaker_views() == ({}, set())
        assert "multi_edit" in _tool_names()
        out = run_tool("multi_edit", {"path": "nope", "edits": []})
        assert "removed from the tool" not in out
    finally:
        st.ACTIVE, st.MODEL, st._BREAKER_ENDPOINT_KEY = saved[0], saved[1], saved[2]
        st._MALFORMED_TOOL_FAILURES.clear()
        st._MALFORMED_TOOL_FAILURES.update(saved[3])
        st._MALFORMED_TOOL_BROKEN.clear()
        st._MALFORMED_TOOL_BROKEN.update(saved[4])


def test_model_override_on_one_source_is_a_new_endpoint():
    """/source together <other-model> points one Source at different weights —
    truncation evidence is about the (model, server) pair, not the Source row."""
    saved = (st.ACTIVE, st.MODEL, st._BREAKER_ENDPOINT_KEY,
             dict(st._MALFORMED_TOOL_FAILURES), set(st._MALFORMED_TOOL_BROKEN))
    try:
        src = _FakeSource("together", "glm-5.3")
        st.ACTIVE = src
        st.MODEL = "glm-5.3"
        failures, broken = _breaker_views()
        failures["multi_edit"] = 3
        broken.add("multi_edit")
        st._BREAKER_ENDPOINT_KEY = ("together", "glm-5.3")
        assert "multi_edit" in _breaker_views()[1]

        st.MODEL = "glm-5.3-flash"
        assert _breaker_views() == ({}, set())
    finally:
        st.ACTIVE, st.MODEL, st._BREAKER_ENDPOINT_KEY = saved[0], saved[1], saved[2]
        st._MALFORMED_TOOL_FAILURES.clear()
        st._MALFORMED_TOOL_FAILURES.update(saved[3])
        st._MALFORMED_TOOL_BROKEN.clear()
        st._MALFORMED_TOOL_BROKEN.update(saved[4])


def test_caller_installed_streak_survives_the_first_observation():
    """The sentinel must not eat state a caller installed directly (a test, or
    a hypothetical future /toolcalls inject): only evidence gathered on a
    DIFFERENT observed endpoint gets dropped."""
    saved = (st._BREAKER_ENDPOINT_KEY, dict(st._MALFORMED_TOOL_FAILURES))
    try:
        st._BREAKER_ENDPOINT_KEY = st.BREAKER_UNOBSERVED
        st._MALFORMED_TOOL_FAILURES["read_file"] = 2
        assert _breaker_views()[0].get("read_file") == 2
    finally:
        st._BREAKER_ENDPOINT_KEY = saved[0]
        st._MALFORMED_TOOL_FAILURES.clear()
        st._MALFORMED_TOOL_FAILURES.update(saved[1])


def test_toolcalls_reset_rearms_the_endpoint_key():
    from nx._repl_settings_cmds import _cmd_toolcalls

    class _S:
        messages = []

    saved = (st._BREAKER_ENDPOINT_KEY, dict(st._MALFORMED_TOOL_FAILURES),
             set(st._MALFORMED_TOOL_BROKEN))
    try:
        _breaker_views()[0]["multi_edit"] = 3
        st._MALFORMED_TOOL_BROKEN.add("multi_edit")
        _cmd_toolcalls("reset", _S())
        assert st._BREAKER_ENDPOINT_KEY == st.BREAKER_UNOBSERVED
        assert st._MALFORMED_TOOL_BROKEN == set()
    finally:
        st._BREAKER_ENDPOINT_KEY, = saved[:1]
        st._MALFORMED_TOOL_FAILURES.clear()
        st._MALFORMED_TOOL_FAILURES.update(saved[1])
        st._MALFORMED_TOOL_BROKEN.clear()
        st._MALFORMED_TOOL_BROKEN.update(saved[2])


if __name__ == "__main__":
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_"):
            _fn()
            print(f"PASS {_name}")
    print("ALL PASS")
