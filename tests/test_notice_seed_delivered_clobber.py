"""0.152.6 regression: a turn loop's frame seeds a pre-existing note from the
module store, DELIVERS it mid-turn (frame slot -> None), a frameless agent
worker stages a NEW note into the module store, and pop_notice_ctx must not
clobber it. The plain overwrite fired whenever the seed was non-empty (the
concat guard only covers an empty seed), destroying a sub-agent's already-
announced completion note: `announced` is burned, the sweep never re-stages,
wake_due() stays False forever — the tabs UI shows all agents done but the
main loop is never woken."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

from nx import _notices as nz


def test_delivered_seed_does_not_clobber_worker_note():
    nz._CURRENT.set(None)
    st = nz.st
    saved = st._PENDING_SUBAGENT_NOTICE
    try:
        st._PENDING_SUBAGENT_NOTICE = "[Runtime note: first agent done]"
        fr = nz.install_notice_ctx()
        # Frame's seed for this slot is the pre-existing note. Mid-turn, the
        # flush delivers it and clears the FRAME slot only.
        assert nz.get("_PENDING_SUBAGENT_NOTICE") == "[Runtime note: first agent done]"
        nz.set_("_PENDING_SUBAGENT_NOTICE", None)
        # A frameless worker stages a new note into the module store.
        nz._CURRENT.set(None)
        nz.set_("_PENDING_SUBAGENT_NOTICE", "[Runtime note: second agent done]")
        nz._CURRENT.set(fr)
        nz.pop_notice_ctx(fr)
        # The worker's note must survive; the frame held nothing new (its seed
        # was already delivered), so the module copy is kept.
        assert nz.get("_PENDING_SUBAGENT_NOTICE") == "[Runtime note: second agent done]"
    finally:
        nz._CURRENT.set(None)
        st._PENDING_SUBAGENT_NOTICE = saved
