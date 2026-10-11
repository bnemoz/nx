import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

from nx import _notices as nz


def test_worker_note_survives_parent_pop():
    """A note staged into the module store by a worker thread (no frame)
    while a parent turn loop is running must survive pop_notice_ctx."""
    nz._CURRENT.set(None)
    fr = nz.install_notice_ctx()
    try:
        # Worker (no frame in its thread) stages a note mid-turn.
        nz._CURRENT.set(None)
        nz.set_("_PENDING_SUBAGENT_NOTICE", "worker finished")
        # Back in the parent frame: it stages nothing itself.
        nz._CURRENT.set(fr)
    finally:
        nz.pop_notice_ctx(fr)
    assert nz.get("_PENDING_SUBAGENT_NOTICE") == "worker finished"


def test_frame_note_still_folds_back():
    nz._CURRENT.set(None)
    fr = nz.install_notice_ctx()
    try:
        nz.set_("_PENDING_VERIFY_NOTICE", "frame note")
    finally:
        nz.pop_notice_ctx(fr)
    assert nz.get("_PENDING_VERIFY_NOTICE") == "frame note"


def test_nested_seed_and_fold():
    nz._CURRENT.set(None)
    parent = nz.install_notice_ctx()
    nz.set_("_PENDING_BG_COMPLETION_NOTICE", "p")
    child = nz.install_notice_ctx()
    assert nz.get("_PENDING_BG_COMPLETION_NOTICE") == "p"
    nz.set_("_PENDING_BG_COMPLETION_NOTICE", "p-c")
    nz.pop_notice_ctx(child)
    assert nz.get("_PENDING_BG_COMPLETION_NOTICE") == "p-c"
    nz.pop_notice_ctx(parent)
    assert nz.get("_PENDING_BG_COMPLETION_NOTICE") == "p-c"


def test_string_notes_merge_when_both_stage():
    nz._CURRENT.set(None)
    fr = nz.install_notice_ctx()
    try:
        nz._CURRENT.set(None)
        nz.set_("_PENDING_SUBAGENT_NOTICE", "worker A finished. ")
        nz._CURRENT.set(fr)
        nz.set_("_PENDING_SUBAGENT_NOTICE", "frame also staged.")
    finally:
        nz.pop_notice_ctx(fr)
    v = nz.get("_PENDING_SUBAGENT_NOTICE")
    assert "worker A" in v and "frame also" in v


def test_flush_delivers_worker_staged_module_note_inside_frame():
    """The 0.144.x wake-misfire fix: a worker thread stages its completion
    note into the module store while the parent turn loop's frame is
    installed — the frame-scoped drain can't see it, so _flush_pending_
    background_notice must also sweep the module store. Without this the
    parent's turn ends text-only with the note undelivered and the wake
    poller fires a pointless auto-continuation ("Nothing needs doing") at
    the end of every chat that used an async agent."""
    import nx as m
    st = m.st
    saved = {s: getattr(st, s, None) for s in m._runtime_notes._MODULE_NOTE_SLOTS}
    nz._CURRENT.set(None)
    fr = nz.install_notice_ctx()
    try:
        # Worker (frameless thread) stages mid-parent-turn.
        nz._CURRENT.set(None)
        nz.set_("_PENDING_SUBAGENT_NOTICE", "[Runtime note: sub-agent finished]")
        nz._CURRENT.set(fr)
        msgs: list = []
        m._runtime_notes._flush_pending_background_notice(msgs)
        assert msgs == [{"role": "user",
                         "content": "[Runtime note: sub-agent finished]"}], \
            "worker-staged note must be delivered inside the parent's frame"
        assert nz.get("_PENDING_SUBAGENT_NOTICE") is None
        # Delivered once, not twice (module slot cleared, frame slot stays None).
        msgs2: list = []
        m._runtime_notes._flush_pending_background_notice(msgs2)
        assert msgs2 == []
    finally:
        nz._CURRENT.set(fr)
        nz.pop_notice_ctx(fr)
        for s, v in saved.items():
            setattr(st, s, v)


def test_flush_does_not_double_deliver_seeded_frame_note():
    """A note staged BEFORE the frame was installed is seeded into the frame
    by reference; the frame drain delivers it while clearing only the frame
    slot — the module-store sweep must not append the same string again."""
    import nx as m
    st = m.st
    nz._CURRENT.set(None)
    st._PENDING_VERIFY_NOTICE = "[Runtime note: verify passed]"
    fr = nz.install_notice_ctx()
    try:
        msgs: list = []
        m._runtime_notes._flush_pending_background_notice(msgs)
        assert [mm["content"] for mm in msgs] == ["[Runtime note: verify passed]"]
        assert getattr(st, "_PENDING_VERIFY_NOTICE") is None
    finally:
        nz._CURRENT.set(fr)
        nz.pop_notice_ctx(fr)
        st._PENDING_VERIFY_NOTICE = None
