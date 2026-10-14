"""0.156.1 regression: a sub-agent sub-loop must never DELIVER the parent's
pending runtime notes. run_tool refuses to STAGE them inside a sub-agent loop
(the _dispatch.py first-tool-of-turn gate); before this fix the matching
FLUSH was ungated, so a sibling async agent's (or a sync agent's) tool-call
flush drained the parent's staged notes — including worker-staged module-store
notes — into the sub-loop's doomed messages list. The note's `announced` mark
was already burned at staging, so the parent never re-received it and
wake_due() saw nothing pending: the /tabs dashboard showed every agent done
while the main loop sat un-woken (the "all agents finished, never woken"
report).

The gate: _flush_pending_background_notice is a no-op while
_subagent_ctx() is set. Notes the sub-loop itself stages fold back into the
parent frame at pop (_notices), so nothing is lost either way."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

from nx import _notices as nz


class _Ctx:
    """Minimal stand-in for a _SubagentContext: presence is the whole test."""


def _in_subloop(fn):
    """Run fn with a sub-agent context installed on THIS thread (the flag
    _subagent_ctx() reads), restoring after."""
    from nx import _subagent_tls
    prev = getattr(_subagent_tls._SUBAGENT_TLS, "ctx", None)
    _subagent_tls._SUBAGENT_TLS.ctx = _Ctx()
    try:
        return fn()
    finally:
        _subagent_tls._SUBAGENT_TLS.ctx = prev


def test_subloop_flush_delivers_nothing():
    """Both channels — the frame slot and the module store — stay untouched
    for a sub-loop flush, and the parent's own flush still delivers them."""
    import nx as m
    st = m.st
    slots = m._runtime_notes._MODULE_NOTE_SLOTS
    saved = {s: getattr(st, s, None) for s in slots}
    nz._CURRENT.set(None)
    # A note staged BEFORE the loop (frame-seeded) and one staged into the
    # module store by a frameless worker mid-loop: the two shapes the theft
    # bug consumed.
    st._PENDING_SUBAGENT_NOTICE = "[Runtime note: agent A finished]"
    fr = nz.install_notice_ctx()
    try:
        nz.set_("_PENDING_VERIFY_NOTICE", "[Runtime note: verify passed]")

        def flush_in_subloop():
            msgs: list = []
            m._runtime_notes._flush_pending_background_notice(msgs)
            return msgs

        assert _in_subloop(flush_in_subloop) == [], \
            "a sub-agent sub-loop must not deliver the parent's notes"
        # Both channels intact for the parent.
        assert nz.get("_PENDING_SUBAGENT_NOTICE") == "[Runtime note: agent A finished]"
        assert nz.get("_PENDING_VERIFY_NOTICE") == "[Runtime note: verify passed]"
        # The parent's own flush still delivers everything, once.
        parent_msgs: list = []
        m._runtime_notes._flush_pending_background_notice(parent_msgs)
        assert set(mm["content"] for mm in parent_msgs) == {
            "[Runtime note: agent A finished]", "[Runtime note: verify passed]"}, \
            parent_msgs
    finally:
        nz._CURRENT.set(fr)
        nz.pop_notice_ctx(fr)
        for s, v in saved.items():
            setattr(st, s, v)


def test_parent_flush_untouched_outside_subloop():
    """Guard on the guard: the same call outside a sub-agent loop still
    delivers (this is the 0.144.x wake-misfire contract)."""
    import nx as m
    st = m.st
    saved = st._PENDING_SUBAGENT_NOTICE
    nz._CURRENT.set(None)
    st._PENDING_SUBAGENT_NOTICE = "[Runtime note: agent B finished]"
    try:
        msgs: list = []
        m._runtime_notes._flush_pending_background_notice(msgs)
        assert [mm["content"] for mm in msgs] == ["[Runtime note: agent B finished]"]
        assert st._PENDING_SUBAGENT_NOTICE is None
    finally:
        st._PENDING_SUBAGENT_NOTICE = saved
