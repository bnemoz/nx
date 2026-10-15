#!/usr/bin/env python3
"""Plan-stall wake (0.161.0, AUTOWAKE_SPEC Phase C).

The trigger that fires on the deliberate ABSENCE of evidence: a campaign with
pending `todo_write` items whose last done turn left NOTHING in flight (an
agent FAILED and its note was already delivered; the model flipped an item to
`in_progress` and answered in prose). Every existing wake guard in that state
correctly concludes "nothing to say", which is exactly why such a campaign
never resumed.

These tests pin the two things that make the trigger safe rather than a nag
loop: the grace period, and the terminator — at most ONE stall wake per plan
revision, plus the deliberate-stop hold that parks it until the user types.
"""
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"
# Persistence off: these tests set the plan directly, never through the file.
os.environ["NX_TODO_FILE"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nx as m  # noqa: E402
from nx import _autowake  # noqa: E402
from nx import _state as st  # noqa: E402
from nx._todo import _plan_revision  # noqa: E402

GRACE = "NX_WAKE_STALL_GRACE_SECONDS"


def setup_function(_function):
    """A clean campaign per case: the stall trigger's whole contract is which
    combination of (clock, revision, hold, in-flight, cap) is set, so residue
    from a previous case would flip a False case to True or vice versa."""
    st._WAKE_ARMED = True
    st._WAKE_DUE = False
    st._WAKE_STREAK = 0
    st._TURN_ACTIVE = False
    st._SPAWNED_WRITE_TIER = True
    st._BG_NOTICE_SEEN = set()
    st._WAKE_STALL_AT = 0.0
    st._WAKE_STALL_REVISION = -1
    st._WAKE_STALL_HOLD = False
    st._WAKE_STALL_PENDING = False
    os.environ["NX_WAKE_PROMPT"] = "1"
    os.environ.pop("NX_WAKE_STREAK_LIMIT", None)
    os.environ[GRACE] = "20"
    _plan([{"text": "phase 1", "status": "done"},
           {"text": "phase 2", "status": "in_progress"}])
    poller = getattr(_autowake, "_POLLER", None)
    if poller is not None:
        try:
            poller.stop()
        except Exception:
            pass
    _autowake._POLLER = None


def _plan(todos):
    """Set the plan of record AND refresh todo_write's stashed revision, which
    is what the poller compares (todo_write is the only production writer)."""
    st._TODOS = todos
    st._PLAN_REVISION = _plan_revision(todos)
    return todos


def _arm(grace_ago=60.0, stamp=None):
    """Armed stall clock (a done turn left the campaign with nothing in
    flight) `grace_ago` seconds ago; optionally stamp the revision a previous
    stall wake fired for."""
    st._WAKE_STALL_AT = time.monotonic() - grace_ago
    st._WAKE_STALL_REVISION = (_plan_revision() if stamp is None
                               else stamp)


def _no_bg_jobs(monkeypatch):
    """This process owns no bg-bash job maps at all (the shared SESSION_HOME of
    the suite may hold other tests' jobs)."""
    monkeypatch.setattr(m._bash_jobs, "list_bg_jobs", lambda **kw: [])


def _live_bg_job(monkeypatch):
    monkeypatch.setattr(m._bash_jobs, "list_bg_jobs", lambda **kw: [
        {"pid": "999991", "alive": True, "owner_pid": os.getpid(),
         "log_path": "/nonexistent/nx-stall-test.log"}])


# ---------------------------------------------------------------------------
# 1. the trigger itself: the inverse of every existing test_autowake case
# ---------------------------------------------------------------------------

def test_stall_wake_due_with_nothing_staged(monkeypatch):
    """campaign pending + grace elapsed + revision changed ⇒ wake_due() True
    with NO notice staged (every existing wake test asserts False here)."""
    _no_bg_jobs(monkeypatch)
    _arm(grace_ago=60.0, stamp=-1)  # never woken for this revision
    assert not _autowake._any_pending_notice()
    assert _autowake.wake_due() is True
    assert not _autowake._any_pending_notice()  # a stall wake stages nothing
    assert st._WAKE_STALL_PENDING is True  # the instruction selector saw it


def test_grace_not_elapsed_is_false(monkeypatch):
    _no_bg_jobs(monkeypatch)
    _arm(grace_ago=1.0, stamp=-1)
    assert _autowake.wake_due() is False


def test_revision_unchanged_is_false(monkeypatch):
    """ONE stall wake per plan revision — the terminator that replaces a cap."""
    _no_bg_jobs(monkeypatch)
    _arm(grace_ago=60.0)  # stamp == the current revision: already woken for it
    assert _autowake.wake_due() is False
    # ...and it is the terminator, not something else: a plan change re-arms it.
    _plan([{"text": "phase 1", "status": "done"},
           {"text": "phase 2", "status": "done"},
           {"text": "phase 3", "status": "pending"}])
    assert _autowake.wake_due() is True


def test_grace_zero_disables_the_trigger(monkeypatch):
    """The documented off switch: floor 0 = disabled, everything else equal."""
    _no_bg_jobs(monkeypatch)
    os.environ[GRACE] = "0"
    _arm(grace_ago=1e6, stamp=-1)
    assert _autowake.wake_due() is False


def test_negative_grace_clamps_to_off(monkeypatch):
    _no_bg_jobs(monkeypatch)
    os.environ[GRACE] = "-5"
    _arm(grace_ago=1e6, stamp=-1)
    assert _autowake.wake_due() is False


# ---------------------------------------------------------------------------
# 2. what is NOT a campaign (the _campaign_pending contract)
# ---------------------------------------------------------------------------

def test_no_campaign_when_todos_all_done(monkeypatch):
    _no_bg_jobs(monkeypatch)
    _plan([{"text": "phase 1", "status": "done"}])
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


def test_no_campaign_without_a_write_tier_spawn(monkeypatch):
    """Pending todos with no delegated implementation are an interactive
    working list — nagging those is the failure mode Phase C must not add."""
    _no_bg_jobs(monkeypatch)
    st._SPAWNED_WRITE_TIER = False
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


# ---------------------------------------------------------------------------
# 3. the stall branch sits BEHIND every liveness guard and the streak cap
# ---------------------------------------------------------------------------

def test_live_bg_job_wins_over_the_stall_branch(monkeypatch):
    _live_bg_job(monkeypatch)
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


def test_running_async_agent_wins_over_the_stall_branch(monkeypatch):
    _no_bg_jobs(monkeypatch)
    monkeypatch.setattr(m._subagent, "_async_running", lambda: True)
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


def test_queued_async_agent_wins_over_the_stall_branch(monkeypatch):
    """Queued (spawn admitted, worker not started yet) still finishes and
    announces itself, so it is not a stall."""
    _no_bg_jobs(monkeypatch)
    monkeypatch.setattr(m._subagent, "_ASYNC_QUEUED", ["job-queued-stall-test"])
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


def test_streak_cap_cannot_be_bypassed(monkeypatch):
    """The stall branch sits after the cap's early-return: pin it (a stall wake
    is still an unattended wake and spends the chain's budget)."""
    _no_bg_jobs(monkeypatch)
    os.environ["NX_WAKE_STREAK_LIMIT"] = "2"
    _arm(grace_ago=60.0, stamp=-1)
    st._WAKE_STREAK = 2
    assert _autowake.wake_due() is False
    st._WAKE_STREAK = 1
    assert _autowake.wake_due() is True


def test_mid_turn_never_wakes(monkeypatch):
    _no_bg_jobs(monkeypatch)
    st._TURN_ACTIVE = True
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


# ---------------------------------------------------------------------------
# 4. the deliberate-stop hold
# ---------------------------------------------------------------------------

def test_hold_blocks_the_trigger(monkeypatch):
    _no_bg_jobs(monkeypatch)
    st._WAKE_STALL_HOLD = True
    _arm(grace_ago=60.0, stamp=-1)
    assert _autowake.wake_due() is False


def test_no_change_stall_turn_trips_the_hold():
    """A stall wake that changed nothing (no todo edit, nothing newly armed)
    parked the trigger: the clock was armed coming in and the revision still
    equals the stamp the wake fired for."""
    _arm()  # clock armed, stamp == current revision == "this wake's" revision
    _autowake.note_turn_end(True)
    assert st._WAKE_STALL_HOLD is True
    assert st._WAKE_STALL_AT == 0.0  # disarmed as well


def test_a_plan_edit_keeps_the_trigger_alive():
    """The other side of the same seam: the wake advanced the plan, so this is
    progress, not a deliberate stop — no hold, and the clock re-arms for the
    new revision."""
    _arm()
    st._WAKE_STALL_REVISION = _plan_revision()
    _plan([{"text": "phase 1", "status": "done"},
           {"text": "phase 2", "status": "done"},
           {"text": "phase 3", "status": "pending"}])
    before = st._WAKE_STALL_AT
    _autowake.note_turn_end(True)
    assert st._WAKE_STALL_HOLD is False
    assert st._WAKE_STALL_AT != before  # re-armed on the new revision


def test_turn_with_work_in_flight_clears_the_clock():
    """A campaign whose turn ends WITH a background job running has work that
    will wake us for real: no stall clock, and no hold (the wake did not end
    with nothing changed — it ended with work outstanding)."""
    _arm()
    real = m._autowake._work_in_flight

    def _in_flight():
        real  # keep the reference obvious to a reader
        return True
    m._autowake._work_in_flight = _in_flight
    try:
        _autowake.note_turn_end(True)
    finally:
        m._autowake._work_in_flight = real
    assert st._WAKE_STALL_AT == 0.0
    assert st._WAKE_STALL_HOLD is False


def test_note_turn_end_arms_the_clock_for_a_stalled_campaign(monkeypatch):
    """The REPL's post-turn finally, in isolation: done + campaign + nothing
    in flight ⇒ armed; anything else ⇒ 0.0."""
    _no_bg_jobs(monkeypatch)
    st._WAKE_STALL_AT = 0.0
    _autowake.note_turn_end(True)
    assert st._WAKE_STALL_AT > 0.0
    _autowake.note_turn_end(False)  # turn failed / escaped
    assert st._WAKE_STALL_AT == 0.0
    st._SPAWNED_WRITE_TIER = False
    _autowake.note_turn_end(True)
    assert st._WAKE_STALL_AT == 0.0


def test_typed_input_clears_clock_and_hold(monkeypatch):
    """The _repl.py typed-input reset (_WAKE_STREAK = 0 next to
    clear_stall_state()), pinned at the contract level."""
    _no_bg_jobs(monkeypatch)
    _arm(grace_ago=60.0, stamp=-1)
    st._WAKE_STALL_HOLD = True
    assert _autowake.wake_due() is False  # parked
    _autowake.clear_stall_state()
    assert st._WAKE_STALL_AT == 0.0
    assert st._WAKE_STALL_HOLD is False
    assert st._WAKE_STALL_PENDING is False
    # The user speaking is not itself a wake: the clock starts over, so the
    # next stall needs a fresh grace period before it can fire again.
    assert _autowake.wake_due() is False


# ---------------------------------------------------------------------------
# 5. the instruction: names the open items, carries the escape hatch
# ---------------------------------------------------------------------------

_ESCAP = ("if you stopped because you need the user, or because the plan is "
          "complete, say so in one line and take no action")


def test_stall_instruction_names_the_pending_items():
    _plan([{"text": "merge phase 1 report", "status": "done"},
           {"text": "write the phase 2 gate tests", "status": "pending"},
           {"text": "document the stall wake", "status": "in_progress"}])
    st._WAKE_STALL_PENDING = True
    out = _autowake.wake_instruction()
    assert out == _autowake.STALL_WAKE_INSTRUCTION.format(
        items="- write the phase 2 gate tests\n- document the stall wake")
    assert "write the phase 2 gate tests" in out
    assert "document the stall wake" in out
    assert "merge phase 1 report" not in out  # done items are not "open"
    assert _ESCAP in out
    assert "{" not in out and "}" not in out  # {items} was FORMATTED


def test_stall_instruction_is_one_shot():
    """Consumed by the first read: the next (non-stall) wake must not inherit
    the stall wording."""
    st._WAKE_STALL_PENDING = True
    assert "stalled" in _autowake.wake_instruction()
    assert st._WAKE_STALL_PENDING is False
    assert _autowake.wake_instruction() == _autowake.CAMPAIGN_WAKE_INSTRUCTION.format(
        n=1, m=25)  # still a campaign, but a plain completion wake now


def test_wake_due_decides_the_stall_marker_per_evaluation(monkeypatch):
    """A staged notice wins (a real completion note beats a stall hypothesis),
    and then the marker must be False — no stall wording for that wake."""
    monkeypatch.setattr(m._bash_jobs, "list_bg_jobs", lambda **kw: [])
    _arm(grace_ago=60.0, stamp=-1)
    st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    try:
        assert _autowake.wake_due() is True
        assert st._WAKE_STALL_PENDING is False
    finally:
        st._PENDING_SUBAGENT_NOTICE = None


def test_stall_grace_default_and_clamp():
    os.environ.pop(GRACE, None)
    assert _autowake._stall_grace_seconds() == 20.0
    os.environ[GRACE] = "45.5"
    assert _autowake._stall_grace_seconds() == 45.5
    os.environ[GRACE] = "junk"
    assert _autowake._stall_grace_seconds() == 20.0


# ---------------------------------------------------------------------------
# 6. the revision helper
# ---------------------------------------------------------------------------

def test_plan_revision_is_stable_and_plan_sensitive():
    a = [{"text": "x", "status": "pending"}, {"text": "y", "status": "done"}]
    b = [{"text": "x", "status": "pending"}, {"text": "y", "status": "done"}]
    assert _plan_revision(a) == _plan_revision(b)          # stable
    assert _plan_revision(a) != _plan_revision(
        [{"text": "x", "status": "done"}, {"text": "y", "status": "done"}])
    assert _plan_revision(a) != _plan_revision(            # order matters
        [{"text": "y", "status": "done"}, {"text": "x", "status": "pending"}])
    assert _plan_revision([]) != -1                        # never the sentinel
    # None means "hash the live plan" (the poller-side call form): same value
    # as the current st._TODOS, whatever the caller set up.
    assert _plan_revision(None) == _plan_revision(st._TODOS)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
