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
    # The old `importlib.reload(nx)` rebuilt nx's import-time state from this
    # fixture's env — most importantly the model source: conftest imports nx
    # once under a scrubbed env, so st.client/st.MODEL stay None (no active
    # source). Instead of a reload, rebuild just that wiring: re-discover
    # sources from the env we just set and switch to the first one (binds
    # st.client + st.MODEL). resolve_model() reads NX_MODEL ("test-model"), so
    # no network round-trip happens; the dead 127.0.0.1:9 endpoint is never
    # contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # nx reads these knobs through the MODULE ATTR first, falling back to the
    # env (see _subagent._is_async_default / _delegation_note.
    # _delegation_enabled), and the attr holds the import-time default. The
    # reload used to rebuild the attrs from this fixture's env; with the reload
    # gone, state the behavior-changing knobs the env alone can no longer reach.
    # NX_SUBAGENT_ASYNC=0 = foreground sub-loop; NX_SUBAGENTS=1 = the
    # spawn_agent schema (and thus the delegation nudge) is offered.
    monkeypatch.setattr(nx, "NX_SUBAGENT_ASYNC", 0, raising=False)
    monkeypatch.setattr(nx, "NX_SUBAGENTS", 1, raising=False)
    # Hidden module state the old module-dict rewind used to wipe: a note left
    # pending by an earlier test/file reads as THIS test's fire, and a warm read
    # cache changes what run_tool() returns. Pin both per-test.
    from nx import _state as st
    monkeypatch.setattr(st, "_PENDING_DELEGATION_NOTICE", None, raising=False)
    monkeypatch.setattr(st, "_READ_STREAK", 0, raising=False)
    monkeypatch.setattr(st, "_DELEGATION_NUDGE_COUNT", 0, raising=False)
    monkeypatch.setattr(st, "_TOOL_RESULT_CACHE", {}, raising=False)
    yield nx
    # env rewind is universal (conftest), async-agent state (_ASYNC_AGENTS,
    # quiet slots, TLS) is reset by the universal nx_state_reset autouse
    # fixture, so no manual teardown here.


def _reset(m, monkeypatch):
    monkeypatch.setattr(m.st, "_READ_STREAK", 0)
    monkeypatch.setattr(m.st, "_DELEGATION_NUDGE_COUNT", 0)
    monkeypatch.setattr(m.st, "_PENDING_DELEGATION_NOTICE", None)


def _probe(tmp_path):
    f = tmp_path / "probe.txt"
    f.write_text("x\n")
    return f


# --- the lowered default -------------------------------------------------------

def test_default_threshold_is_five(nx_fresh, tmp_path, monkeypatch):
    """The shipped default is 5 (lowered from 8). conftest imports nx under a
    scrubbed NX_* env, so the live import-time value IS the default — the old
    guard got there by reloading _session_preamble; Tier A imports once. The
    counter reads the threshold off _dispatch's module attr, so pin it there
    (the same seam test_delegation_steering uses to disable the nudge) and show
    the fire line moves with it."""
    from nx import _dispatch, _session_preamble
    assert _session_preamble._DELEGATION_NUDGE_THRESHOLD == 5, \
        "default nudge threshold lowered 8 -> 5 (env scrubbed, so the default applies)"
    assert _dispatch._DELEGATION_NUDGE_THRESHOLD == 5, \
        "the streak counter must read the same default"
    monkeypatch.setattr(_dispatch, "_DELEGATION_NUDGE_THRESHOLD", 7)
    f = _probe(tmp_path)
    monkeypatch.setattr(nx_fresh.st, "_BG_SCAN_DONE", True)
    for _ in range(6):
        nx_fresh.run_tool("read_file", {"path": str(f), "offset": 1, "limit": 1})
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 0, "6 reads is under a threshold of 7"
    nx_fresh.run_tool("read_file", {"path": str(f), "offset": 1, "limit": 1})
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1, \
        "the fire line must be the _dispatch threshold attr"


# --- fire at the threshold, re-fire at 15 --------------------------------------

def test_nudge_refires_every_2x_threshold(nx_fresh, tmp_path, monkeypatch):
    """Fires at 5 reads, then needs 2*5=10 MORE consecutive reads each time:
    cumulative 5, 15, 25. The streak reset on fire is what buys the cadence."""
    from nx._dispatch import _DELEGATION_NUDGE_THRESHOLD
    assert _DELEGATION_NUDGE_THRESHOLD == 5
    nx_fresh.NX_SUBAGENTS = 1
    _reset(nx_fresh, monkeypatch)
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
    _reset(nx_fresh, monkeypatch)
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
    _reset(nx_fresh, monkeypatch)
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
    _reset(nx_fresh, monkeypatch)
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

def test_resume_counts_prior_nudge_as_one_fire(nx_fresh, monkeypatch):
    """A resumed transcript that already saw a nudge (or already spawned) seeds
    the count to 1, so its next fire needs 2*threshold reads, not threshold."""
    from nx._resume import _apply_resumed_nudge_flags
    _reset(nx_fresh, monkeypatch)
    _apply_resumed_nudge_flags(
        [{"role": "user", "content": "[Runtime note: without using spawn_agent"}])
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    _reset(nx_fresh, monkeypatch)
    _apply_resumed_nudge_flags(
        [{"role": "assistant", "tool_calls": [
            {"function": {"name": "spawn_agent", "arguments": "{}"}}]}])
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 1
    _reset(nx_fresh, monkeypatch)
    _apply_resumed_nudge_flags([{"role": "user", "content": "nothing here"}])
    assert nx_fresh.st._DELEGATION_NUDGE_COUNT == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
