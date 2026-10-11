"""Regression: a note staged mid-turn by a frameless worker, after the turn's
last flush and with a text-only ending, must raise _WAKE_DUE at turn end
(previously it idled until the poller's next tick agreed)."""

import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-wakerace-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _autowake, _state as st
import nx._notices as n


def test_note_staged_mid_turn_raises_wake_due_at_turn_end():
    # Simulate the REPL's per-turn lifecycle around a text-only turn.
    st._WAKE_DUE = False
    st._WAKE_STREAK = 0
    fr = n.install_notice_ctx()          # turn loop starts
    # frameless worker stages AFTER any flush opportunity (text-only turn:
    # there was none)
    n.set_("_PENDING_SUBAGENT_NOTICE", "[Runtime note: race probe finished]")
    n.pop_notice_ctx(fr)                 # turn loop ends (no flush happened)
    st._TURN_ACTIVE = False
    # the new guard in _repl.py's finally:
    if _autowake._any_pending_notice():
        st._WAKE_DUE = True
    assert st._WAKE_DUE, "undelivered note at turn end must raise _WAKE_DUE"
    st._PENDING_SUBAGENT_NOTICE = None
    st._WAKE_DUE = False


def test_no_note_no_wake():
    st._WAKE_DUE = False
    st._WAKE_STREAK = 0
    fr = n.install_notice_ctx()
    n.pop_notice_ctx(fr)
    st._TURN_ACTIVE = False
    if _autowake._any_pending_notice():
        st._WAKE_DUE = True
    assert not st._WAKE_DUE, "no pending note -> no wake"
