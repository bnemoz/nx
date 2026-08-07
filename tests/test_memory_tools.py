#!/usr/bin/env python3
"""Model-driven memory: the `memory_write` / `memory_search` tools and the
session-start memory-awareness note.

Covers the two scopes (project under <root>/.nx/memories, global under
NX_HOME/memories), the approval gate on writes, the read-only search, and the
one-time note appended to a fresh session's message list. No model/network
round-trip — `_confirm` is stubbed and the tool bodies touch only temp dirs.
"""
import os
import sys
import tempfile

# Isolate ~/.nx so a real memory store can't leak in (or out). Same convention
# as the other tests: NX_HOME + NX_SESSIONS_DIR both point at a temp dir, and
# NX_ENV_FILE=/dev/null so local credentials can't affect results.
#
# nx is imported once per pytest process and st.SESSION_HOME is a shared global
# that other test files reassign; setting NX_HOME here would only matter on the
# *first* import. So we also pin st.SESSION_HOME to our _tmp in setup and
# restore the prior value in teardown, so a global-write test always lands in
# *our* dir regardless of which file ran last.
_tmp = tempfile.mkdtemp(prefix="nx-test-mem-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _memory_cmd  # noqa: E402


# --- stubs -------------------------------------------------------------------
# memory_write goes through the approval gate (nx._confirm). The real _confirm
# prompts on stdin, which explodes under pytest's captured stdin — and because
# `m._confirm` is a shared module global that other test files save/restore,
# we save/restore on both sides of every test here too (same pattern as
# test_edit_file.py) rather than relying on a module-level assignment.
_confirm_calls = []


def _fake_confirm(action, preview=None):
    _confirm_calls.append(action)
    return True


_saved_confirm = None
_saved_home = None


def _setup_function():
    global _saved_confirm, _saved_home
    _confirm_calls.clear()
    _saved_confirm = m._confirm
    _saved_home = m.st.SESSION_HOME
    m._confirm = _fake_confirm
    # Pin our isolated home so global-scope writes land in _tmp, not whatever
    # dir the previously-run test file left in the shared st.SESSION_HOME.
    m.st.SESSION_HOME = _tmp


def _teardown_function():
    global _saved_confirm, _saved_home
    # restore the exact gate reference so we don't leak the stub into other
    # test files (m._confirm is a shared module global — see test_edit_file.py).
    if _saved_confirm is not None:
        m._confirm = _saved_confirm
    if _saved_home is not None:
        m.st.SESSION_HOME = _saved_home
    _saved_confirm = None
    _saved_home = None


setup_function = lambda _function: _setup_function()
teardown_function = lambda _function: _teardown_function()


def _write_memory(title, body, scope="project", project_root=None):
    """Call memory_write with a controlled PROJECT_ROOT (the tool reads it
    from st.PROJECT_ROOT at call time)."""
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = project_root
    try:
        return _memory_cmd.memory_write(title=title, body=body, scope=scope)
    finally:
        m.st.PROJECT_ROOT = old


# --- tests -------------------------------------------------------------------

def test_memory_write_project_scope_writes_under_project_root():
    root = tempfile.mkdtemp(prefix="nx-proj-")
    res = _write_memory("Build uses uv", "Run `uv run pytest` to test.", scope="project", project_root=root)
    assert res.startswith("saved project memory: Build uses uv"), res
    pdir = os.path.join(root, ".nx", "memories")
    files = [f for f in os.listdir(pdir) if f.endswith(".md")]
    assert len(files) == 1, files
    with open(os.path.join(pdir, files[0]), encoding="utf-8") as f:
        content = f.read()
    assert content.startswith("# Build uses uv\n\n")
    assert "Run `uv run pytest` to test." in content
    assert "_scope: project" in content


def test_memory_write_global_scope_writes_under_nx_home():
    res = _write_memory("Global fact", "Cross-project learning.", scope="global", project_root=None)
    assert res.startswith("saved global memory: Global fact"), res
    gdir = m._memories_dir()
    # The global dir is shared across this file's tests; assert on the memory
    # just written (by title) rather than on a count.
    written = [f for f in os.listdir(gdir) if f.endswith(".md")
               and open(os.path.join(gdir, f), encoding="utf-8").read().startswith("# Global fact\n")]
    assert len(written) == 1, written
    with open(os.path.join(gdir, written[0]), encoding="utf-8") as f:
        content = f.read()
    assert "Cross-project learning." in content
    assert "_scope: global" in content


def test_memory_write_project_falls_back_to_global_without_root():
    # scope=project but no PROJECT_ROOT -> falls back to global so the memory
    # isn't lost (the tool rewrites scope to global and writes under NX_HOME).
    res = _write_memory("No root fallback", "Saved globally.", scope="project", project_root=None)
    assert res.startswith("saved global memory: No root fallback"), res
    gdir = m._memories_dir()
    written = [f for f in os.listdir(gdir) if f.endswith(".md")
               and open(os.path.join(gdir, f), encoding="utf-8").read().startswith("# No root fallback\n")]
    assert len(written) == 1, written


def test_memory_write_requires_title_and_body():
    assert "ERROR" in _write_memory("", "body", project_root=None)
    assert "ERROR" in _write_memory("title", "", project_root=None)


def test_memory_write_goes_through_approval_gate():
    _confirm_calls.clear()
    _write_memory("Gated", "approved body", scope="global", project_root=None)
    assert _confirm_calls, "memory_write did not call _confirm (approval gate)"
    assert any("write memory" in a for a in _confirm_calls), _confirm_calls


def test_memory_write_denied_returns_denied_string():
    m._confirm = lambda action, preview=None: False
    res = _write_memory("Denied", "nope", scope="global", project_root=None)
    assert res == "DENIED by user", res
    # teardown restores the real gate


def test_memory_search_finds_across_both_scopes():
    root = tempfile.mkdtemp(prefix="nx-proj2-")
    _write_memory("Project quirk", "The API needs X-Header: 1.", scope="project", project_root=root)
    _write_memory("Global tip", "Use rg over grep for speed.", scope="global", project_root=None)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        res = _memory_cmd.memory_search(query="X-Header")
        assert "Project quirk" in res, res
        assert "[project]" in res, res
        res2 = _memory_cmd.memory_search(query="grep")
        assert "Global tip" in res2, res2
        assert "[global]" in res2, res2
    finally:
        m.st.PROJECT_ROOT = old


def test_memory_search_global_only_skips_project():
    root = tempfile.mkdtemp(prefix="nx-proj3-")
    _write_memory("Proj-only", "project specific detail", scope="project", project_root=root)
    _write_memory("Glob", "global specific detail", scope="global", project_root=None)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        res = _memory_cmd.memory_search(query="detail", scope="global")
        assert "Glob" in res, res
        assert "Proj-only" not in res, res
    finally:
        m.st.PROJECT_ROOT = old


def test_memory_search_empty_query_lists_newest():
    _write_memory("A memory", "body a", scope="global", project_root=None)
    res = _memory_cmd.memory_search(query="")
    assert "memories (" in res, res


def test_memory_search_no_match_message():
    res = _memory_cmd.memory_search(query="zzz-nonexistent-term-xyz")
    assert "no memories matching" in res, res


def test_memory_search_is_cacheable_read_tool():
    # memory_search is read-only; it should be in the cacheable set so repeat
    # searches in a turn don't re-scan the filesystem.
    assert "memory_search" in m._CACHEABLE_READ_TOOLS


def test_load_memory_note_none_when_empty():
    # A fresh NX_HOME with no memories -> no note (token floor stays flat).
    fresh = tempfile.mkdtemp(prefix="nx-fresh-")
    old_home = os.environ.get("NX_HOME")
    os.environ["NX_HOME"] = fresh
    m.st.SESSION_HOME = fresh
    try:
        assert m._load_memory_note(None) is None
        assert m._load_memory_note(fresh) is None
    finally:
        if old_home is not None:
            os.environ["NX_HOME"] = old_home
        m.st.SESSION_HOME = old_home


def test_load_memory_note_lists_titles_with_scope():
    root = tempfile.mkdtemp(prefix="nx-proj5-")
    # Seed both scopes.
    _write_memory("Project memo", "p-body", scope="project", project_root=root)
    _write_memory("Global memo", "g-body", scope="global", project_root=None)
    note = m._load_memory_note(root)
    assert note is not None, "expected a memory note when memories exist"
    assert note["role"] == "user"
    assert "[project] Project memo" in note["content"], note["content"]
    assert "[global] Global memo" in note["content"], note["content"]
    assert "memory_search" in note["content"]  # tells the model how to read them
    assert "memory_write" in note["content"]   # and how to add more


def test_initial_messages_includes_memory_note_when_memories_exist():
    root = tempfile.mkdtemp(prefix="nx-proj6-")
    _write_memory("Seed memo", "seed body", scope="project", project_root=root)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        msgs = m._initial_messages()
        # System first; memory note should appear as a user-role message.
        assert msgs[0]["role"] == "system"
        mem_notes = [x for x in msgs if x.get("role") == "user"
                     and "Saved memories" in x.get("content", "")]
        assert mem_notes, "memory note not injected into initial messages"
    finally:
        m.st.PROJECT_ROOT = old


def test_initial_messages_omits_memory_note_when_none_exist():
    fresh = tempfile.mkdtemp(prefix="nx-fresh2-")
    old_home = os.environ.get("NX_HOME")
    os.environ["NX_HOME"] = fresh
    m.st.SESSION_HOME = fresh
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = None
    try:
        msgs = m._initial_messages()
        mem_notes = [x for x in msgs if x.get("role") == "user"
                     and "Saved memories" in x.get("content", "")]
        assert not mem_notes, "memory note injected when no memories exist"
    finally:
        if old_home is not None:
            os.environ["NX_HOME"] = old_home
        m.st.SESSION_HOME = old_home
        m.st.PROJECT_ROOT = old


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))