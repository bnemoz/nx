#!/usr/bin/env python3
"""The delegation streak nudge's RE-FIRE mechanics (guards the aggressive-
steering change; the three prompt layers themselves are guarded by
test_delegation_steering.py).

The nudge is no longer once-per-session: each fire resets st._READ_STREAK, so
the first fire lands after `threshold` consecutive exploration reads and every
later one after 2x `threshold` MORE (cumulative 5, 15, 25 at the default 5).
Covered here: the lowered default threshold, the re-fire cadence, duckdb
counting as an exploration read, and a non-read tool breaking the streak. Plus
the resume seed: a transcript that already saw a nudge counts as one fire.

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
    os.environ["NX_SUBAGENT_ASYNC"] = "0"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    importlib.reload(nx)
    yield nx
    from nx import _subagent
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS.clear()


def _reset(m):
    m.st._READ_STREAK = 0
    m.st._DELEGATION_NUDGE_COUNT = 0
    m.st._PENDING_DELEGATION_NOTICE = None


def _probe(tmp_path):
    f = tmp_path / "probe.txt"
    f.write_text("x\n")
    return f


# --- the lowered default -------------------------------------------------------

def test_default_threshold_is_five():
    import nx
    from nx import _session_preamble
    importlib.reload(_session_preamble)
    assert _session_preamble._DELEGATION_NUDGE_THRESHOLD == 5, \
        "default nudge threshold lowered 8 -> 5 (env scrubbed, so the default applies)"
    importlib.reload(nx)


# --- fire at the threshold, re-fire at 15 --------------------------------------

def test_nudge_refires_every_2x_threshold(nx_fresh, tmp_path, monkeypatch):
    """Fires at 5 reads, then needs 2*5=10 MORE consecutive reads each time:
    cumulative 5, 15, 25. The streak reset on fire is what buys the cadence."""
    from nx._dispatch import _DELEGATION_NUDGE_THRESHOLD
    assert _DELEGATION_NUDGE_THRESHOLD == 5
    nx_fresh.NX_SUBAGENTS = 1
    _reset(nx_fresh)
    f = _probe(tmp_path)
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)

    fires_at = []
    seen = 0
    for i in range(1, 26):  # cumulative reads
        nx_fresh.run_tool("read_file", {"path": str(f), "offset": 1, "limit": 1})
        seen += 1
        if nx_fresh.st._PENDING_DELEGATION_NOTICE is not None:
            fires_at.append(seen)
            assert nx_fresh.st._READ_STREAK == 0, "fire must reset the streak"
            nx_fresh.st._PENDING_DELEGATION_NOTICE = None

    assert fires_at == [5, 15, 25], \
        f"nudge should fire at cumulative 5/15/25, got {fires_at}"
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 3


def test_no_refire_before_full_run(nx_fresh, tmp_path, monkeypatch):
    """Between fires (reads 6..14) the nudge must stay silent — one straggler
    read after a fire never re-stages it."""
    nx_fresh.NX_SUBAGENTS = 1
    _reset(nx_fresh)
    f = _probe(tmp_path)
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for i in range(5):
        nx_fresh.run_tool("read_file", {"path": str(f), "offset": 1, "limit": 1})
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    nx_fresh.st._PENDING_DELEGATION_NOTICE = None
    for i in range(9):  # reads 6..14 cumulative, streak 1..9 (< 10)
        nx_fresh.run_tool("read_file", {"path": str(f), "offset": 1, "limit": 1})
        assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    # the 10th read after the fire (cumulative 15) re-fires
    nx_fresh.run_tool("read_file", {"path": str(f), "offset": 1, "limit": 1})
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 2


# --- duckdb counts as an exploration read --------------------------------------

def test_duckdb_counts_toward_streak(nx_fresh, tmp_path, monkeypatch):
    """duckdb is analytical READ work the model should be delegating, so it
    feeds the streak exactly like read_file does. The counter runs before the
    body, so the (absent duckdb CLI) ERROR result is irrelevant here."""
    nx_fresh.NX_SUBAGENTS = 1
    _reset(nx_fresh)
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for i in range(4):
        nx_fresh.run_tool("duckdb", {"sql": f"SELECT {i}"})
        assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None
    assert nx_fresh.st._READ_STREAK == 4
    nx_fresh.run_tool("duckdb", {"sql": "SELECT 4"})
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1, \
        "duckdb must count toward the delegation streak"
    assert nx_fresh.st._PENDING_DELEGATION_NOTICE is not None


# --- non-read tools break the streak -------------------------------------------

def test_non_read_tool_resets_streak(nx_fresh, tmp_path, monkeypatch):
    """An edit/bash/todo means the model is acting on what it read, not
    surveying — it must zero the run so a long read/edit ping-pong never
    accumulates to a nudge."""
    nx_fresh.NX_SUBAGENTS = 1
    _reset(nx_fresh)
    f = _probe(tmp_path)
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for _ in range(4):
        nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._READ_STREAK == 4
    nx_fresh.run_tool("git_status", {})  # not a streak-counted read tool
    assert nx_fresh.st._READ_STREAK == 0
    # and it takes a FULL threshold again from here, not one more read
    for _ in range(4):
        nx_fresh.run_tool("read_file", {"path": str(f)})
        assert nx_fresh.st._PENDING_DELEGATION_NOTICE is None
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 0
    nx_fresh.run_tool("read_file", {"path": str(f)})
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1


# --- resume seeds the fired count ----------------------------------------------

def test_resume_counts_prior_nudge_as_one_fire(nx_fresh):
    """A resumed transcript that already saw a nudge (or already spawned) seeds
    the count to 1, so its next fire needs 2*threshold reads, not threshold."""
    from nx._resume import _apply_resumed_nudge_flags
    nx_fresh.NX_SUBAGENTS = 1
    _reset(nx_fresh)
    _apply_resumed_nudge_flags(
        [{"role": "user", "content": "[Runtime note: without using spawn_agent"}])
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    _reset(nx_fresh)
    _apply_resumed_nudge_flags(
        [{"role": "assistant", "tool_calls": [
            {"function": {"name": "spawn_agent", "arguments": "{}"}}]}])
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    _reset(nx_fresh)
    _apply_resumed_nudge_flags([{"role": "user", "content": "nothing here"}])
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
