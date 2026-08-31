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


# --- 0.7.3: hardening, ranking, lifecycle ------------------------------------

def test_memory_write_files_are_private_and_never_collide():
    a = _write_memory("Dup Title", "first body", scope="global", project_root=None)
    b = _write_memory("Dup Title", "second body", scope="global", project_root=None)
    fa = os.path.abspath(a.split("file: ", 1)[1].strip())
    fb = os.path.abspath(b.split("file: ", 1)[1].strip())
    assert fa != fb, (a, b)
    # The second write must not have clobbered the first.
    with open(fa, encoding="utf-8") as f:
        assert "first body" in f.read()
    mode = oct(os.stat(fb).st_mode & 0o777)
    assert mode == "0o600", mode


def test_memory_write_is_atomic_no_tmp_left_behind():
    a = _write_memory("Atomic", "body", scope="global", project_root=None)
    p = os.path.abspath(a.split("file: ", 1)[1].strip())
    leftovers = [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")]
    assert not leftovers, leftovers


def test_memory_search_ranks_title_above_body_and_requires_all_terms():
    root = tempfile.mkdtemp(prefix="nx-rank-")
    _write_memory("Quirk about the renderer",
                  "unrelated body words here", scope="project", project_root=root)
    _write_memory("Unrelated note",
                  "something about renderer internals", scope="project", project_root=root)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        res = _memory_cmd.memory_search(query="renderer")
        assert res.index("Quirk about the renderer") < res.index("Unrelated note"), res
        both = _memory_cmd.memory_search(query="renderer internals")
        assert "Unrelated note" in both, both
        assert "Quirk about the renderer" not in both, both
    finally:
        m.st.PROJECT_ROOT = old


def test_memory_search_caps_results_with_more_note():
    root = tempfile.mkdtemp(prefix="nx-cap-")
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    oldenv = os.environ.get("NX_MEMORY_SEARCH_LIMIT")
    os.environ["NX_MEMORY_SEARCH_LIMIT"] = "3"
    try:
        for i in range(5):
            _write_memory(f"Cap note {i}", f"capfill {i}", scope="project", project_root=root)
        res = _memory_cmd.memory_search(query="capfill")
        assert "Cap note" in res, res
        assert "more not shown" in res, res
        assert res.count("[project]") == 3, res
    finally:
        m.st.PROJECT_ROOT = old
        if oldenv is None:
            os.environ.pop("NX_MEMORY_SEARCH_LIMIT", None)
        else:
            os.environ["NX_MEMORY_SEARCH_LIMIT"] = oldenv


def test_memory_delete_exact_ambiguous_missing_and_denied():
    root = tempfile.mkdtemp(prefix="nx-del-")
    pdir = os.path.join(root, ".nx", "memories")
    _write_memory("Alpha Note", "alpha body", scope="project", project_root=root)
    _write_memory("Alpha Note Two", "alpha2 body", scope="project", project_root=root)
    _write_memory("Beta Note", "beta body", scope="project", project_root=root)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        res = _memory_cmd.memory_delete(target="Alpha")
        assert "matches 2 memories" in res, res
        res = _memory_cmd.memory_delete(target="zzz")
        assert "no memory matching" in res and "Beta Note" in res, res
        m._confirm = lambda action, preview=None: True
        res = _memory_cmd.memory_delete(target="Beta")
        assert res.startswith("deleted project memory: Beta Note"), res
        assert not [f for f in os.listdir(pdir) if f.endswith(".md") and "Beta" in f]
        res = _memory_cmd.memory_delete(target="Alpha Note Two")
        assert res.startswith("deleted project memory: Alpha Note Two"), res
        m._confirm = lambda action, preview=None: False
        res = _memory_cmd.memory_delete(target="Alpha Note")
        assert res == "DENIED by user", res
    finally:
        m.st.PROJECT_ROOT = old


def test_memory_delete_goes_through_approval_gate():
    _confirm_calls.clear()
    root = tempfile.mkdtemp(prefix="nx-delgate-")
    _write_memory("Gate Note", "gate body", scope="project", project_root=root)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        _memory_cmd.memory_delete(target="Gate Note")
    finally:
        m.st.PROJECT_ROOT = old
    assert _confirm_calls, "memory_delete did not call _confirm (approval gate)"
    assert any(a.startswith("delete memory ") for a in _confirm_calls), _confirm_calls


def test_memory_delete_drops_cached_search_results():
    root = tempfile.mkdtemp(prefix="nx-delcache-")
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    _write_memory("Cacheable Note", "cacheable body", scope="project", project_root=root)
    key = ("memory_search", '{"query": "cacheable"}')
    m.st._TOOL_RESULT_CACHE[key] = m._tool_cache_store_entry(
        "stale", os.path.join(root, "x"))
    try:
        m._confirm = lambda action, preview=None: True
        res = m.run_tool("memory_delete", {"target": "Cacheable"})
        assert res.startswith("deleted project memory"), res
        assert key not in m.st._TOOL_RESULT_CACHE
    finally:
        m.st.PROJECT_ROOT = old


def test_load_memory_note_overflow_hint():
    root = tempfile.mkdtemp(prefix="nx-overflow-")
    for i in range(7):
        _write_memory(f"Overflow note {i}", f"body {i}", scope="project", project_root=root)
    note = m._load_memory_note(root)
    assert note is not None
    assert "more saved" in note["content"], note["content"]
    assert "memory_search" in note["content"]
    lines = [ln for ln in note["content"].split("\n") if ln.startswith("[project]")]
    assert len(lines) == 5, note["content"]


def test_add_memory_note_if_absent_inserts_once_and_skips_existing():
    root = tempfile.mkdtemp(prefix="nx-note-")
    _write_memory("Resume memo", "resume body", scope="project", project_root=root)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        # Resumed transcript predating the feature -> note inserted after system.
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
        _memory_cmd._add_memory_note_if_absent(msgs)
        assert msgs[1]["role"] == "user" and "[Saved memories" in msgs[1]["content"]
        # Transcript already carrying one -> untouched.
        before = list(msgs)
        _memory_cmd._add_memory_note_if_absent(msgs)
        assert msgs == before
        # No memories anywhere -> no note.
        fresh = tempfile.mkdtemp(prefix="nx-note-fresh-")
        old_home = m.st.SESSION_HOME
        m.st.SESSION_HOME = fresh
        try:
            m.st.PROJECT_ROOT = None
            msgs2 = [{"role": "system", "content": "S"}]
            _memory_cmd._add_memory_note_if_absent(msgs2)
            assert msgs2 == [{"role": "system", "content": "S"}]
        finally:
            m.st.SESSION_HOME = old_home
    finally:
        m.st.PROJECT_ROOT = old


class _FakeSaveMsg:
    content = "# Distilled Title\n\nDistilled body about this project."


class _FakeSaveChoice:
    message = _FakeSaveMsg()


class _FakeSaveResp:
    choices = [_FakeSaveChoice()]
    usage = None

    def model_dump(self):
        return {}


class _FakeSaveCompletions:
    def create(self, **kwargs):
        return _FakeSaveResp()


class _FakeSaveChat:
    completions = _FakeSaveCompletions()


class _FakeSaveClient:
    chat = _FakeSaveChat()
    base_url = "http://localhost:9999/v1"


def test_memory_save_distills_into_project_scope_with_unified_footer():
    root = tempfile.mkdtemp(prefix="nx-save-")
    old = m.st.PROJECT_ROOT
    old_client, old_model, old_active = m.st.client, m.st.MODEL, m.st.ACTIVE
    old_metrics = m.st._METRICS
    m.st.PROJECT_ROOT = root
    m.st.client = _FakeSaveClient()
    m.st.MODEL = "test-model"
    m.st.ACTIVE = None
    m.st._METRICS = None  # usage recording is best-effort; keep it inert
    try:
        _memory_cmd._cmd_memory_save([{"role": "user", "content": "learned something"}])
        pdir = os.path.join(root, ".nx", "memories")
        files = [f for f in os.listdir(pdir) if f.endswith(".md")]
        assert len(files) == 1, files
        with open(os.path.join(pdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("# Distilled Title\n\n"), content
        assert "Distilled body" in content
        assert "_scope: project" in content, content  # unified footer, project scope
    finally:
        m.st.PROJECT_ROOT = old
        m.st.client, m.st.MODEL, m.st.ACTIVE = old_client, old_model, old_active
        m.st._METRICS = old_metrics


def test_memory_write_accepts_title_in_body_without_doubling_h1():
    a = _write_memory("Plain Title", "body only", scope="global", project_root=None)
    fa = os.path.abspath(a.split("file: ", 1)[1].strip())
    with open(fa, encoding="utf-8") as f:
        content = f.read()
    assert content.count("# Plain Title") == 1, content

    b = _memory_cmd._write_memory_file("Body Title", "# Body Title\n\nbody", "global")
    assert b is not None
    with open(b[0], encoding="utf-8") as f:
        content2 = f.read()
    assert content2.count("# Body Title") == 1, content2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))