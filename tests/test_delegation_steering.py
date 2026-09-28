#!/usr/bin/env python3
"""Delegation steering: models left spawn_agent unused (grinding through
dozens of sequential reads themselves) unless the user explicitly asked for a
fan-out. Three prompt layers fix that, and this file guards each:

  1. the spawn_agent schema description carries PROACTIVELY + the trigger list
     (it's the only text sent every turn when NX_SUBAGENTS=1),
  2. a session-start [Delegation note:] paid ONLY in sub-agent-enabled sessions
     (_initial_messages / _add_delegation_note_if_absent),
  3. a runtime streak nudge staged by run_tool() after enough solo exploration
     calls without a spawn (re-arming: the fire resets the streak, so a later
     full run stages it again; silent when the tool isn't offered, flushed by
     _flush_pending_background_notice — see test_delegation_nudge.py).

No model/network round-trip.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in [k for k in os.environ if k.startswith("NX_")]:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    # Keep any spawn synchronous: under the default background mode a
    # spawn_agent call here would register a real worker in the process-global
    # async registry and leak it into later test files (test_subagent_async
    # counts that registry).
    os.environ["NX_SUBAGENT_ASYNC"] = "0"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    yield nx
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()


# --- layer 1: schema description ---------------------------------------------

def test_schema_description_carries_proactive_triggers(nx_fresh):
    schema = next(t for t in nx_fresh.TOOLS
                  if t["function"]["name"] == "spawn_agent")
    desc = schema["function"]["description"]
    assert "PROACTIVELY" in desc, desc
    # The triggers that were missing before: multi-file/multi-module work and
    # the one-call-per-question fan-out instruction.
    assert "one call per question" in desc or "ONE CALL PER QUESTION" in desc, desc
    assert "callers" in desc, desc


# --- layer 2: session-start note ----------------------------------------------

def test_delegation_note_injected_only_when_subagents_on(nx_fresh):
    nx_fresh.NX_SUBAGENTS = 1
    msgs = nx_fresh._initial_messages()
    notes = [m for m in msgs if "[Delegation note:" in (m.get("content") or "")]
    assert len(notes) == 1, "fresh NX_SUBAGENTS=1 session must carry the note"
    assert notes[0]["role"] == "user"

    nx_fresh.NX_SUBAGENTS = 0
    msgs = nx_fresh._initial_messages()
    assert not [m for m in msgs if "[Delegation note:" in (m.get("content") or "")], \
        "sub-agents-disabled session must not pay for the note"


def test_delegation_note_not_duplicated_on_resume(nx_fresh):
    nx_fresh.NX_SUBAGENTS = 1
    from nx._subagent import _add_delegation_note_if_absent
    msgs = [{"role": "system", "content": "sys"}]
    _add_delegation_note_if_absent(msgs)
    _add_delegation_note_if_absent(msgs)
    notes = [m for m in msgs if "[Delegation note:" in (m.get("content") or "")]
    assert len(notes) == 1, "note must be inserted at most once"
    # Inserted after the system message, not appended at the tail.
    assert msgs.index(notes[0]) == 1
    # Disabled: nothing added.
    nx_fresh.NX_SUBAGENTS = 0
    bare = [{"role": "system", "content": "sys"}]
    _add_delegation_note_if_absent(bare)
    assert len(bare) == 1


# --- layer 3: runtime streak nudge --------------------------------------------

def _streak_reset(m):
    m.st._READ_STREAK = 0
    m.st._DELEGATION_NUDGE_COUNT = 0
    m.st._PENDING_DELEGATION_NOTICE = None


def test_streak_nudge_fires_after_threshold_reads(nx_fresh, tmp_path, monkeypatch):
    from nx._dispatch import _DELEGATION_NUDGE_THRESHOLD
    nx_fresh.NX_SUBAGENTS = 1
    _streak_reset(nx_fresh)
    f = tmp_path / "probe.txt"
    f.write_text("x\n")
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)  # skip bg sweep
    for i in range(_DELEGATION_NUDGE_THRESHOLD - 1):
        nx_fresh.run_tool("read_file", {"path": str(f), "offset": i + 1, "limit": 1})
        assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None
    nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._PENDING_DELEGATION_NOTICE is not None, \
        "streak nudge should fire once the threshold is crossed"
    assert "spawn_agent" in nx_fresh.st._PENDING_DELEGATION_NOTICE
    # the fire resets the streak, so a single further read does NOT re-stage it
    # (re-fire needs a full 2x-threshold run — see test_delegation_nudge.py)
    assert nx_fresh.st._READ_STREAK == 0
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    nx_fresh.st._PENDING_DELEGATION_NOTICE = None
    nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None, \
        "a lone read after a fire must not re-stage the nudge"


def test_spawn_agent_resets_streak(nx_fresh, tmp_path, monkeypatch):
    from nx._dispatch import _DELEGATION_NUDGE_THRESHOLD
    nx_fresh.NX_SUBAGENTS = 1
    _streak_reset(nx_fresh)
    f = tmp_path / "probe.txt"
    f.write_text("x\n")
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for _ in range(_DELEGATION_NUDGE_THRESHOLD - 1):
        nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._READ_STREAK == _DELEGATION_NUDGE_THRESHOLD - 1
    # A spawn resets the counter. No client is configured, so the real
    # spawn_agent returns an ERROR string — the reset is bookkeeping and
    # happens regardless of the result.
    nx_fresh.run_tool("spawn_agent", {"instruction": "x"})
    assert nx_fresh.st._READ_STREAK == 0
    nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None


def test_streak_silent_when_subagents_disabled(nx_fresh, tmp_path, monkeypatch):
    from nx._dispatch import _DELEGATION_NUDGE_THRESHOLD
    nx_fresh.NX_SUBAGENTS = 0
    _streak_reset(nx_fresh)
    f = tmp_path / "probe.txt"
    f.write_text("x\n")
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for _ in range(_DELEGATION_NUDGE_THRESHOLD + 2):
        nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None, \
        "no nudge toward a tool the session doesn't offer"
    assert not nx_fresh.st._DELEGATION_NUDGE_COUNT


def test_threshold_zero_disables_nudge(nx_fresh, tmp_path, monkeypatch):
    from nx import _dispatch
    nx_fresh.NX_SUBAGENTS = 1
    _streak_reset(nx_fresh)
    monkeypatch.setattr(_dispatch, "_DELEGATION_NUDGE_THRESHOLD", 0)
    f = tmp_path / "probe.txt"
    f.write_text("x\n")
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for _ in range(3):
        nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None


def test_nudge_flushed_into_messages_once(nx_fresh):
    from nx._compress import _flush_pending_background_notice
    nx_fresh.st._PENDING_DELEGATION_NOTICE = "DELEG-NOTE"
    msgs: list = []
    _flush_pending_background_notice(msgs)
    assert msgs and msgs[-1]["content"] == "DELEG-NOTE"
    msgs2: list = []
    _flush_pending_background_notice(msgs2)
    assert msgs2 == [], "staged notice must be consumed exactly once"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
