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
