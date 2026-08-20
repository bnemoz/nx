#!/usr/bin/env python3
"""Session-start orientation note: the one-time user-role message appended to
a fresh session that tells the model its real working directory (SYSTEM only
says "cwd = user's current directory" without the path) and points it at the
approval-free read-only tools.

Covers the loader itself, its injection into _initial_messages(), and the
NX_ORIENTATION=0 suppression knob. No model/network round-trip.
"""
import os
import sys
import tempfile

# Same isolation convention as the other tests: NX_HOME + NX_SESSIONS_DIR at
# a temp dir, NX_ENV_FILE=/dev/null so local credentials can't affect results.
# NX_ORIENTATION unset here; we set/clear it per test below and restore after.
_tmp = tempfile.mkdtemp(prefix="nx-test-orient-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _orientation  # noqa: E402


def _orientation_notes(msgs):
    return [x for x in msgs
            if x.get("role") == "user"
            and "Orientation note" in x.get("content", "")]


def test_load_orientation_note_includes_real_cwd():
    # Default on: note gives the model its actual cwd and tool hint.
    old = os.environ.get("NX_ORIENTATION")
    os.environ.pop("NX_ORIENTATION", None)
    try:
        note = _orientation._load_orientation_note()
        assert note is not None, "orientation note missing by default"
        assert note["role"] == "user"
        cwd = os.path.realpath(os.getcwd())
        assert cwd in note["content"], note["content"]
        # Points at the approval-free read-only tool, not a shell recipe.
        assert "list_dir" in note["content"], note["content"]
        assert "pwd" not in note["content"].lower(), note["content"]
    finally:
        if old is not None:
            os.environ["NX_ORIENTATION"] = old


def test_load_orientation_note_disabled_by_env():
    # NX_ORIENTATION=0 suppresses the note (token-floor opt-out).
    old = os.environ.get("NX_ORIENTATION")
    os.environ["NX_ORIENTATION"] = "0"
    try:
        assert _orientation._load_orientation_note() is None
    finally:
        if old is None:
            os.environ.pop("NX_ORIENTATION", None)
        else:
            os.environ["NX_ORIENTATION"] = old


def test_initial_messages_includes_orientation_note_by_default():
    old = os.environ.get("NX_ORIENTATION")
    os.environ.pop("NX_ORIENTATION", None)
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = None
    try:
        msgs = m._initial_messages()
        assert msgs[0]["role"] == "system"
        notes = _orientation_notes(msgs)
        assert notes, "orientation note not injected into initial messages"
        assert os.path.realpath(os.getcwd()) in notes[0]["content"]
    finally:
        if old is not None:
            os.environ["NX_ORIENTATION"] = old
        m.st.PROJECT_ROOT = old_root


def test_initial_messages_omits_orientation_note_when_disabled():
    old = os.environ.get("NX_ORIENTATION")
    os.environ["NX_ORIENTATION"] = "0"
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = None
    try:
        msgs = m._initial_messages()
        assert not _orientation_notes(msgs), (
            "orientation note injected despite NX_ORIENTATION=0"
        )
    finally:
        if old is None:
            os.environ.pop("NX_ORIENTATION", None)
        else:
            os.environ["NX_ORIENTATION"] = old
        m.st.PROJECT_ROOT = old_root


def test_orientation_note_reports_git_project_root_when_distinct():
    # When PROJECT_ROOT is a real git root distinct from cwd, the note should
    # surface it; the cwd-fallback (PROJECT_ROOT_IS_GIT False) must not.
    root = tempfile.mkdtemp(prefix="nx-proj-orient-")
    old_root = m.st.PROJECT_ROOT
    old_git = m.st.PROJECT_ROOT_IS_GIT
    old = os.environ.get("NX_ORIENTATION")
    os.environ.pop("NX_ORIENTATION", None)
    m.st.PROJECT_ROOT = root
    m.st.PROJECT_ROOT_IS_GIT = True
    try:
        note = _orientation._load_orientation_note()
        assert note is not None
        assert root in note["content"], note["content"]
        assert "Project root" in note["content"], note["content"]
    finally:
        m.st.PROJECT_ROOT = old_root
        m.st.PROJECT_ROOT_IS_GIT = old_git
        if old is not None:
            os.environ["NX_ORIENTATION"] = old


def test_orientation_note_omits_root_on_cwd_fallback():
    # Running from $HOME (no git root): PROJECT_ROOT == cwd, is_git False.
    # The note must report only cwd, not a misleading "Project root" line.
    old_root = m.st.PROJECT_ROOT
    old_git = m.st.PROJECT_ROOT_IS_GIT
    old = os.environ.get("NX_ORIENTATION")
    os.environ.pop("NX_ORIENTATION", None)
    m.st.PROJECT_ROOT = os.path.realpath(os.getcwd())
    m.st.PROJECT_ROOT_IS_GIT = False
    try:
        note = _orientation._load_orientation_note()
        assert note is not None
        assert "Project root" not in note["content"], note["content"]
    finally:
        m.st.PROJECT_ROOT = old_root
        m.st.PROJECT_ROOT_IS_GIT = old_git
        if old is not None:
            os.environ["NX_ORIENTATION"] = old


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))