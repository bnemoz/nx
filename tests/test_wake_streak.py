#!/usr/bin/env python3
"""Unattended multi-phase campaigns (0.121.0): the campaign wake instruction.

Before this, the only wake instruction was WAKE_INSTRUCTION, whose "stop when
done" is right for a wake that answers a question but reads as "wrap up after
this delivery" mid-campaign — so a phased plan stalled until the user typed
the next "continue". Now, when a todo plan is mid-flight AND this session has
spawned a write-tier agent, the wake turn gets CAMPAIGN_WAKE_INSTRUCTION
("you are the coordinator, spawn the next phase"), and consecutive unattended
wakes are capped by NX_WAKE_STREAK_LIMIT. A typed turn resets the streak.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx as m  # noqa: E402
from nx import _autowake  # noqa: E402
import nx._delegation_note as d  # noqa: E402


def setup_function(_function):
    """Fresh wake state per test: a leak from the previous case would either
    hide the campaign variant (streak at the cap) or trip it where the plain
    instruction is meant (todos/flag left set)."""
    m.st._WAKE_ARMED = False
    m.st._WAKE_DUE = False
    m.st._WAKE_PREFILL = ""
    m.st._TURN_ACTIVE = False
    m.st._WAKE_STREAK = 0
    m.st._WAKE_CHAIN_SPEND = 0
    m.st._WAKE_CHAIN_BASE = 0
    m.st._SPAWNED_WRITE_TIER = False
    m.st._TODOS = []
    os.environ["NX_WAKE_PROMPT"] = "1"
    os.environ.pop("NX_WAKE_STREAK_LIMIT", None)
    os.environ.pop("NX_WAKE_CHAIN_TOKEN_BUDGET", None)
    _poller = getattr(_autowake, "_POLLER", None)
    if _poller is not None:
        try:
            _poller.stop()
        except Exception:
            pass
    _autowake._POLLER = None


def _pending_plan():
    return [{"content": "phase 1", "status": "done"},
            {"content": "phase 2", "status": "pending"}]


# ---------------------------------------------------------------------------
# 1. the selector picks the campaign variant on a mid-flight plan
# ---------------------------------------------------------------------------

def test_campaign_variant_when_plan_midflight_and_write_tier_spawned():
    m.st._TODOS = _pending_plan()
    m.st._SPAWNED_WRITE_TIER = True
    m.st._WAKE_STREAK = 1
    out = _autowake.wake_instruction()
    assert out == _autowake.CAMPAIGN_WAKE_INSTRUCTION.format(n=1, m=25)
    assert "coordinator" in out
    # {n}/{m} were FORMATTED, not passed through as literal braces.
    assert "wake 1 of 25" in out and "{" not in out and "}" not in out
    assert out != _autowake.WAKE_INSTRUCTION


def test_campaign_variant_stamps_the_live_streak_and_limit():
    m.st._TODOS = _pending_plan()
    m.st._SPAWNED_WRITE_TIER = True
    m.st._WAKE_STREAK = 4
    os.environ["NX_WAKE_STREAK_LIMIT"] = "7"
    out = _autowake.wake_instruction()
    assert "wake 4 of 7" in out


# ---------------------------------------------------------------------------
# 2. the plain instruction everywhere else
# ---------------------------------------------------------------------------

def test_plain_when_todos_all_done():
    m.st._TODOS = [{"content": "phase 1", "status": "done"}]
    m.st._SPAWNED_WRITE_TIER = True
    assert _autowake.wake_instruction() == _autowake.WAKE_INSTRUCTION


def test_plain_when_no_write_tier_spawned():
    m.st._TODOS = _pending_plan()
    m.st._SPAWNED_WRITE_TIER = False
    assert _autowake.wake_instruction() == _autowake.WAKE_INSTRUCTION


def test_plain_when_todos_empty():
    m.st._TODOS = []
    m.st._SPAWNED_WRITE_TIER = True
    assert _autowake.wake_instruction() == _autowake.WAKE_INSTRUCTION
    m.st._TODOS = None
    assert _autowake.wake_instruction() == _autowake.WAKE_INSTRUCTION


def test_campaign_pending_predicate():
    m.st._TODOS = _pending_plan()
    m.st._SPAWNED_WRITE_TIER = True
    assert _autowake._campaign_pending() is True
    m.st._SPAWNED_WRITE_TIER = False
    assert _autowake._campaign_pending() is False
    m.st._SPAWNED_WRITE_TIER = True
    m.st._TODOS = [{"content": "x", "status": "done"}]
    assert _autowake._campaign_pending() is False


# ---------------------------------------------------------------------------
# 3. the chain guard in wake_due()
# ---------------------------------------------------------------------------

def test_wake_due_false_at_streak_limit():
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        m.st._WAKE_STREAK = 25
        assert _autowake.wake_due() is False
        # ...and the guard is not merely shadowing the constant: at limit-1
        # with the same staged notice a wake IS due.
        m.st._WAKE_STREAK = 24
        assert _autowake.wake_due() is True
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_chain_guard_honors_configured_limit():
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    os.environ["NX_WAKE_STREAK_LIMIT"] = "3"
    try:
        m.st._WAKE_STREAK = 3
        assert _autowake.wake_due() is False
        m.st._WAKE_STREAK = 2
        assert _autowake.wake_due() is True
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_wake_due_ignores_streak_for_a_fresh_session():
    # Streak 0 (the reset state a typed turn leaves) never blocks.
    m.st._WAKE_ARMED = True
    m.st._WAKE_STREAK = 0
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        assert _autowake.wake_due() is True
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


# ---------------------------------------------------------------------------
# 4. the limit knob
# ---------------------------------------------------------------------------

def test_wake_streak_limit_default_and_override():
    assert _autowake._wake_streak_limit() == 25
    os.environ["NX_WAKE_STREAK_LIMIT"] = "40"
    assert _autowake._wake_streak_limit() == 40


def test_wake_streak_limit_clamps_zero_to_one():
    os.environ["NX_WAKE_STREAK_LIMIT"] = "0"
    assert _autowake._wake_streak_limit() == 1
    os.environ["NX_WAKE_STREAK_LIMIT"] = "-5"
    assert _autowake._wake_streak_limit() == 1
    # a garbage value falls back to the default (then the >=1 clamp)
    os.environ["NX_WAKE_STREAK_LIMIT"] = "not-an-int"
    assert _autowake._wake_streak_limit() == 25


# ---------------------------------------------------------------------------
# 5. the streak reset contract the REPL implements
#    (_repl.py sets st._WAKE_STREAK = 0 on every non-wake input path, and
#    increments it on a wake turn before calling wake_instruction(); this
#    pins the STATE CONTRACT those lines rely on, not a fake REPL.)
# ---------------------------------------------------------------------------

def test_streak_reset_reopens_the_chain():
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        m.st._WAKE_STREAK = 25
        assert _autowake.wake_due() is False, "chain capped"
        m.st._WAKE_STREAK = 0  # what a typed turn (incl. a slash command) does
        assert _autowake.wake_due() is True, "typed turn restarts the chain"
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_streak_increment_is_visible_to_the_instruction():
    # The REPL increments BEFORE wake_instruction(), so n is 1-based.
    m.st._TODOS = _pending_plan()
    m.st._SPAWNED_WRITE_TIER = True
    m.st._WAKE_STREAK = getattr(m.st, "_WAKE_STREAK", 0) + 1
    assert "wake 1 of 25" in _autowake.wake_instruction()


# ---------------------------------------------------------------------------
# 6. the delegation note carries the campaign bullet
# ---------------------------------------------------------------------------

def test_delegation_note_documents_the_campaign_pattern(monkeypatch):
    monkeypatch.setattr(m, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(m, "NX_SUBAGENT_WRITE", 1)
    content = d._load_delegation_note()["content"]
    assert "driven through" in content
    assert "NX_WAKE_STREAK_LIMIT" in content
    assert "BACKGROUND write/implement-" in content


def test_campaign_bullet_absent_without_write_authorization(monkeypatch):
    monkeypatch.setattr(m, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(m, "NX_SUBAGENT_WRITE", 0)
    content = d._load_delegation_note()["content"]
    assert "driven through" not in content


# ---------------------------------------------------------------------------
# 7. the write-tier flag is set at spawn time
# ---------------------------------------------------------------------------

def test_spawn_agent_marks_the_write_tier_flag(monkeypatch):
    m.st._SPAWNED_WRITE_TIER = False
    monkeypatch.setattr(m, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(m, "NX_SUBAGENT_WRITE", 1)
    monkeypatch.setattr(m.st, "YOLO", True)
    monkeypatch.setattr(m.st, "client", None, raising=False)
    out = m._subagent.spawn_agent("test", tier="write")
    assert "no active model source" in out  # reached the spawn path
    assert m.st._SPAWNED_WRITE_TIER is True


def test_spawn_agent_read_tier_leaves_the_flag_alone(monkeypatch):
    m.st._SPAWNED_WRITE_TIER = False
    monkeypatch.setattr(m, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(m.st, "client", None, raising=False)
    m._subagent.spawn_agent("test", tier="read")
    assert m.st._SPAWNED_WRITE_TIER is False


def test_spawn_agent_refused_write_tier_does_not_mark_the_flag(monkeypatch):
    # A refused spawn (no NX_SUBAGENT_WRITE) delegated nothing: the campaign
    # gate must stay closed.
    m.st._SPAWNED_WRITE_TIER = False
    monkeypatch.setattr(m, "NX_SUBAGENTS", 1)
    monkeypatch.setattr(m, "NX_SUBAGENT_WRITE", 0)
    out = m._subagent.spawn_agent("test", tier="write")
    assert "requires NX_SUBAGENT_WRITE=1" in out
    assert m.st._SPAWNED_WRITE_TIER is False


# ---------------------------------------------------------------------------
# 8. chain cost bound (0.162.0, AUTOWAKE_SPEC Phase D): the streak cap counts
#    WAKES while cost per wake varies ~100x; NX_WAKE_CHAIN_TOKEN_BUDGET counts
#    context tokens instead. Default 0 = off; negatives clamp to 0 (off).
# ---------------------------------------------------------------------------

_HOLD_SLOT = "_PENDING_WAKE_HOLD_NOTICE"


def _chain_notice_staged():
    from nx import _notices
    return _notices.get(_HOLD_SLOT) or getattr(m.st, _HOLD_SLOT, None)


def _clear_chain_hold():
    from nx import _notices
    _notices.set_(_HOLD_SLOT, None)
    m.st._PENDING_WAKE_HOLD_NOTICE = None
    _autowake._HOLD_STAGED = None


def test_chain_budget_default_and_clamp():
    # 0 = the documented default (off); a negative clamps to off, it cannot
    # arm the guard by accident.
    assert _autowake._wake_chain_budget() == 0
    os.environ["NX_WAKE_CHAIN_TOKEN_BUDGET"] = "150000"
    assert _autowake._wake_chain_budget() == 150000
    os.environ["NX_WAKE_CHAIN_TOKEN_BUDGET"] = "-5"
    assert _autowake._wake_chain_budget() == 0


def test_budget_zero_never_trips():
    # The no-behavior-change contract for existing users: an enormous spend
    # with the budget unset (0) must not stop a productive chain.
    m.st._WAKE_ARMED = True
    m.st._WAKE_CHAIN_SPEND = 10 ** 12
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        assert _autowake.wake_due() is True
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_budget_guard_trips_with_hold_notice_exactly_once():
    _clear_chain_hold()
    os.environ["NX_WAKE_CHAIN_TOKEN_BUDGET"] = "50000"
    m.st._WAKE_ARMED = True
    m.st._WAKE_CHAIN_SPEND = 50000  # exactly at budget -> trips (>=)
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        assert _autowake.wake_due() is False
        note = _chain_notice_staged()
        assert note is not None
        assert "chain budget spent (50000/50000 tokens)" in note
        # poll ticks re-trip the same guard every 2s; the (guard, streak) key
        # stages the explanation exactly once (Phase B idempotency).
        for _ in range(5):
            assert _autowake.wake_due() is False
        assert _autowake._HOLD_STAGED == (_autowake._HOLD_BUDGET, 0)
        assert _chain_notice_staged() == note
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_budget_below_trip_point_still_wakes():
    os.environ["NX_WAKE_CHAIN_TOKEN_BUDGET"] = "50000"
    m.st._WAKE_ARMED = True
    m.st._WAKE_CHAIN_SPEND = 49999
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        assert _autowake.wake_due() is True
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_typed_input_reset_reopens_a_budget_parked_chain():
    # The contract the REPL's typed-input lines implement: the streak AND
    # both chain accumulators reset together (_autowake.reset_chain_cost),
    # and the trip key releases so a later chain explains itself again.
    _clear_chain_hold()
    os.environ["NX_WAKE_CHAIN_TOKEN_BUDGET"] = "50000"
    m.st._WAKE_ARMED = True
    m.st._WAKE_CHAIN_SPEND = 60000
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        assert _autowake.wake_due() is False
        _autowake.reset_chain_cost()  # what the typed path calls
        assert m.st._WAKE_CHAIN_SPEND == 0 and m.st._WAKE_CHAIN_BASE == 0
        assert _autowake.wake_due() is True, "typed turn restarts the chain"
        assert _autowake._HOLD_STAGED is None, "trip key released with the chain"
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_note_chain_spend_adds_only_this_turns_delta():
    # Wake start snapshots the baseline; the post-turn finally adds the delta
    # once and RE-snapshots, so a multi-step turn's growth (already embedded
    # in the final prompt-token reading) is attributed exactly once, and the
    # next turn in the chain adds only its own growth.
    m.st._WAKE_CHAIN_BASE = 10000
    m.st._LAST_PROMPT_TOKENS = 14000
    _autowake.note_chain_spend()
    assert m.st._WAKE_CHAIN_SPEND == 4000
    assert m.st._WAKE_CHAIN_BASE == 14000
    m.st._LAST_PROMPT_TOKENS = 15000  # next wake turn of the chain
    _autowake.note_chain_spend()
    assert m.st._WAKE_CHAIN_SPEND == 5000


def test_note_chain_spend_ignores_unknown_usage_and_shrink():
    # No usage reported -> nothing to attribute. A context fold SHRINKS the
    # reading -> max(0, delta) keeps it from paying spend back. A base of 0
    # (the typed-input reset) never books the absolute reading.
    m.st._WAKE_CHAIN_BASE = 0
    m.st._LAST_PROMPT_TOKENS = 0
    _autowake.note_chain_spend()
    assert m.st._WAKE_CHAIN_SPEND == 0
    m.st._WAKE_CHAIN_BASE = 10000
    m.st._LAST_PROMPT_TOKENS = 8000
    _autowake.note_chain_spend()
    assert m.st._WAKE_CHAIN_SPEND == 0
    assert m.st._WAKE_CHAIN_BASE == 8000
    m.st._WAKE_CHAIN_BASE = 0
    m.st._LAST_PROMPT_TOKENS = 12000
    _autowake.note_chain_spend()
    assert m.st._WAKE_CHAIN_SPEND == 0
    assert m.st._WAKE_CHAIN_BASE == 12000


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
