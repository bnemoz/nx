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
        # Resumed transcript predating the feature -> note appended at the TAIL
        # (PERF.md PF-12: never inserted mid-history, which would bust the
        # prompt-prefix cache).
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
        _memory_cmd._add_memory_note_if_absent(msgs)
        assert msgs[3]["role"] == "user" and "[Saved memories" in msgs[3]["content"]
        assert msgs[1] == {"role": "user", "content": "hi"}  # prefix untouched
        # Transcript already carrying an up-to-date one -> untouched.
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


class _DegenerateSaveResp:
    choices = []           # empty: nothing to extract a body from
    usage = None

    def model_dump(self):
        return {}


class _DegenerateSaveCompletions:
    def create(self, **kwargs):
        return _DegenerateSaveResp()


class _DegenerateSaveChat:
    completions = _DegenerateSaveCompletions()


class _DegenerateSaveClient:
    chat = _DegenerateSaveChat()
    base_url = "http://localhost:9999/v1"


def test_memory_save_degenerate_response_reports_error_instead_of_crashing():
    """Empty `choices` used to raise IndexError out of /memory save into the
    REPL; it must print the failure and save nothing."""
    root = tempfile.mkdtemp(prefix="nx-save-degen-")
    old = m.st.PROJECT_ROOT
    old_client, old_model, old_active = m.st.client, m.st.MODEL, m.st.ACTIVE
    old_metrics = m.st._METRICS
    m.st.PROJECT_ROOT = root
    m.st.client = _DegenerateSaveClient()
    m.st.MODEL = "test-model"
    m.st.ACTIVE = None
    m.st._METRICS = None
    try:
        _memory_cmd._cmd_memory_save([{"role": "user", "content": "learned"}])  # no raise
        assert not os.path.exists(os.path.join(root, ".nx", "memories"))
    finally:
        m.st.PROJECT_ROOT = old
        m.st.client, m.st.MODEL, m.st.ACTIVE = old_client, old_model, old_active
        m.st._METRICS = old_metrics


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


# --- memory_search paths + resume note staleness (0.37.0) ---------------------

def _archive_all_notes(pdir, stamp):
    """Move every .md in a memories dir into <pdir>/archive/<stamp>/ — what a
    /memory optimize run does to retired notes (reversible, scanner-invisible)."""
    dst_dir = os.path.join(pdir, "archive", stamp)
    os.makedirs(dst_dir)
    for n in [n for n in os.listdir(pdir) if n.endswith(".md")]:
        os.replace(os.path.join(pdir, n), os.path.join(dst_dir, n))
    return dst_dir


def test_memory_search_reports_absolute_dirs_for_read_file():
    """memory_search prints bare filenames — a model that treats one as a path
    guesses a plausible dir (e.g. ~/.nx/memory/) and read_file 404s. The result
    now ends with the absolute dirs actually holding the shown hits."""
    root = tempfile.mkdtemp(prefix="nx-mem-paths-")
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        res = _write_memory("Path hint memo", "uniquepathmarker body", scope="project", project_root=root)
        assert res.startswith("saved project memory"), res
        out = _memory_cmd.memory_search(query="uniquepathmarker", scope="project")
        assert os.path.join(root, ".nx", "memories") in out, out
        assert "read_file" in out, out
    finally:
        m.st.PROJECT_ROOT = old


def test_memory_search_dir_hint_tracks_only_live_hit_dirs():
    """A dir line is only printed for dirs still holding a shown hit: after a
    note is archived (optimize ran), its dir must vanish from the hint instead
    of pointing at a file that is no longer there."""
    root = tempfile.mkdtemp(prefix="nx-mem-dirs-")
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        _write_memory("Dirs project memo", "dirsmarker037 body", scope="project", project_root=root)
        _write_memory("Dirs global memo", "dirsmarker037 body", scope="global", project_root=None)
        out = _memory_cmd.memory_search(query="dirsmarker037", scope="project")
        assert os.path.join(root, ".nx", "memories") in out, out
        assert m._memories_dir() in out, out
        _archive_all_notes(os.path.join(root, ".nx", "memories"), "20260906-000002")
        out2 = _memory_cmd.memory_search(query="dirsmarker037", scope="project")
        assert m._memories_dir() in out2, out2
        assert os.path.join(root, ".nx", "memories") not in out2, out2
    finally:
        m.st.PROJECT_ROOT = old


def test_refresh_memory_note_at_resume_updates_stale_titles():
    """A resumed transcript's [Saved memories] note lists titles as of the
    session's last run; /memory optimize in another session may have retired
    or merged notes since. --resume and /resume both rebuild it (and must be
    a no-op when the store still matches, so the prompt cache survives)."""
    root = tempfile.mkdtemp(prefix="nx-mem-resume-")
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        # A global note must exist for the whole test: when the ONLY note is
        # archived, _load_memory_note() returns None and the transcript note
        # is DELETED (not replaced) — the idx lookup below needs the marker
        # to survive via the global scope. (Writing it here, not relying on
        # another test in this file to have left one: test order varies.)
        _write_memory("Resume global anchor", "resumeanchor body", scope="global", project_root=None)
        _write_memory("Resume memo one", "resumeone body", scope="project", project_root=root)
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
        _memory_cmd._add_memory_note_if_absent(msgs)
        assert "[project] Resume memo one" in str(msgs[-1].get("content")), str(msgs[-1])
        # the store moved on: the only note is archived by an optimize elsewhere
        _archive_all_notes(os.path.join(root, ".nx", "memories"), "20260906-000003")
        assert _memory_cmd._refresh_memory_note(msgs) is True
        # Append-only (PF-12): the stale session-start note stays where it was,
        # and a fresh tail note supersedes it.
        assert "Resume memo one" not in str(msgs[-1].get("content"))
        assert "[global] Resume global anchor" in str(msgs[-1].get("content"))
        # refreshing again with an unchanged store must not touch the transcript
        assert _memory_cmd._refresh_memory_note(msgs) is False
        # a new note lands in the store: a fresh note is APPENDED at the tail
        # (the earlier bytes are left untouched — prompt-cache safe).
        _write_memory("Resume memo two", "resumetwo body", scope="project", project_root=root)
        before = len(msgs)
        assert _memory_cmd._refresh_memory_note(msgs) is True
        assert len(msgs) == before + 1
        assert "[project] Resume memo two" in str(msgs[-1].get("content"))
        # refreshing again with the unchanged store is still a no-op
        assert _memory_cmd._refresh_memory_note(msgs) is False
    finally:
        m.st.PROJECT_ROOT = old


def test_iter_memory_dirs_skips_project_dir_that_is_the_global_dir():
    """A bare-cwd project root equal to NX_HOME makes <root>/.nx/memories the
    global dir itself; it must not be scanned twice (every global note used to
    appear as both [global] and [project] in /memory list and search)."""
    root = tempfile.mkdtemp(prefix="nx-proj-home-")
    _write_memory("Dup probe", "dircollision body", scope="global", project_root=None)
    old = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        # Collision: global = SESSION_HOME/memories must equal <root>/.nx/memories.
        old_home = m.st.SESSION_HOME
        m.st.SESSION_HOME = os.path.join(root, ".nx")
        try:
            pairs = _memory_cmd._iter_memory_dirs("project")
            assert [label for _d, label in pairs] == ["global"], pairs
            assert [d for d, _label in pairs] == [_memory_cmd._memories_dir()]
        finally:
            m.st.SESSION_HOME = old_home
        # A genuinely distinct project dir is still returned.
        real_root = tempfile.mkdtemp(prefix="nx-proj-real-")
        m.st.PROJECT_ROOT = real_root
        pairs = _memory_cmd._iter_memory_dirs("project")
        assert [label for _d, label in pairs] == ["global", "project"], pairs
    finally:
        m.st.PROJECT_ROOT = old


# --- self-ingestion filter + `_derived:` provenance (0.158.19) ----------------

def test_is_satellite_message_covers_every_injected_block_not_bracketed_user_text():
    """The shared predicate: every marker nx injects must be recognised, and a
    genuine user message that merely OPENS with a bracket must not be — the bug
    in the old `startswith("[")` heuristic."""
    from nx._prefix import SATELLITE_MARKERS, is_satellite_message as sat
    nl = chr(10)
    injected = [
        "[Project instructions from AGENTS.md:]" + nl + "RULE",  # nx/_agents.py:42
        "[Saved memories — use memory_search to read any:]" + nl + "[project] t",  # _memory_note.py:31
        "[Saved memories: the store is now empty — the earlier list is stale;]",  # _memory_note.py:36
        "[Orientation note:]" + nl + "Working directory: /x",  # _orientation.py:51
        "[Delegation note:]" + nl + "spawn agents",           # _delegation_note.py:108
        "[Runtime note: a command you ran just backgrounded]",  # _runtime_notes.py:24
        "[Steering note: the user typed this while your turn was running]",  # _surfaces.py:125
        "[Plan mode active — goal: g" + nl + "Plan file: p]",  # _plan.py:369
        "[User pressed Esc at a tool approval prompt and returned to chat]",  # _turn.py:771
        "[User hard-cancelled this turn (double Esc / Ctrl+C).]",  # _turn.py:781
        "[User interrupted your previous response with Esc.]",   # _turn.py:771
    ]
    for text in injected:
        assert sat({"role": "user", "content": text}), text
    for text in ["[bug] the parser drops the second arg", "[1/3] step one",
                 "plain user text", "[not a marker]"]:
        assert not sat({"role": "user", "content": text}), text
    # role and shape guards
    assert not sat({"role": "assistant", "content": "[Saved memories"})
    assert not sat({"role": "user", "content": [{"type": "text"}]})
    assert not sat("not a dict")
    assert len(SATELLITE_MARKERS) == 6


def test_render_transcript_for_memory_drops_satellite_blocks_keeps_bracketed_user():
    """The /memory save transcript must not carry nx's own injected blocks back
    into the distiller (the self-ingestion loop), while a GENUINE user turn that
    merely opens with a bracket survives — the old `startswith("[")` bluntness
    dropped exactly those."""
    msgs = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "[Saved memories — use memory_search to read any:]\n[project] A stale title"},
        {"role": "user", "content": "[Orientation note:]\nWorking directory: /x"},
        {"role": "user", "content": "[Delegation note:]\nspawn agents"},
        {"role": "user", "content": "[Project instructions from AGENTS.md:]\nRULE"},
        {"role": "user", "content": "[Runtime note: a command you ran just backgrounded]"},
        {"role": "user", "content": "[Steering note: the user typed this while your turn was running]"},
        {"role": "user", "content": "[bug] the parser drops the second arg"},
        {"role": "assistant", "content": "looking into it"},
    ]
    out = _memory_cmd._render_transcript_for_memory(msgs)
    assert "[bug] the parser drops the second arg" in out, out
    assert "[user] [bug] the parser drops the second arg" in out, out
    assert "Saved memories" not in out, out
    assert "Orientation note" not in out, out
    assert "Delegation note" not in out, out
    assert "Project instructions" not in out, out
    assert "Runtime note" not in out, out
    assert "Steering note" not in out, out
    assert "SYSTEM" not in out, out


def test_extract_derived_marker_pops_the_line_and_strips_a_leftover_fence():
    body = "# T\n\nlearned\n\n```\n_derived: memory\n```\n"
    cleaned, derived = _memory_cmd._extract_derived_marker(body)
    assert derived == "memory"
    assert "_derived" not in cleaned
    assert "```" not in cleaned, cleaned
    assert cleaned.endswith("learned"), cleaned
    # A first-hand note is untouched, byte for byte.
    plain = "# T\n\nfirst-hand body"
    assert _memory_cmd._extract_derived_marker(plain) == (plain, None)


class _DerivedSaveMsg:
    content = ("# Recycled Title\n\nRestated prior art about this project."
               "\n\n_derived: memory")


class _DerivedSaveResp:
    choices = [type("C", (), {"message": _DerivedSaveMsg()})()]
    usage = None

    def model_dump(self):
        return {}


class _DerivedSaveClient:
    class _Chat:
        class completions:  # noqa: N801 (mirrors the other fakes here)
            @staticmethod
            def create(**kwargs):
                return _DerivedSaveResp()

    chat = _Chat()
    base_url = "http://localhost:9999/v1"


def test_memory_save_stamps_derived_and_keeps_the_marker_out_of_the_body():
    """The distiller's `_derived: memory` marker is provenance, not prose: it
    must land in the footer and never in the body."""
    root = tempfile.mkdtemp(prefix="nx-save-derived-")
    old = m.st.PROJECT_ROOT
    old_client, old_model, old_active = m.st.client, m.st.MODEL, m.st.ACTIVE
    old_metrics = m.st._METRICS
    m.st.PROJECT_ROOT = root
    m.st.client = _DerivedSaveClient()
    m.st.MODEL = "test-model"
    m.st.ACTIVE = None
    m.st._METRICS = None
    try:
        _memory_cmd._cmd_memory_save([{"role": "user", "content": "learned"}])
        pdir = os.path.join(root, ".nx", "memories")
        files = [f for f in os.listdir(pdir) if f.endswith(".md")]
        assert len(files) == 1, files
        with open(os.path.join(pdir, files[0]), encoding="utf-8") as f:
            content = f.read()
        assert "_derived: memory" in content, content
        # Exactly one footer line, and the marker is NOT in the body.
        assert content.count("_derived:") == 1, content
        body = content.split("\n---\n", 1)[0]
        assert "_derived" not in body, body
        assert _memory_cmd._note_derived(content) == "memory"
    finally:
        m.st.PROJECT_ROOT = old
        m.st.client, m.st.MODEL, m.st.ACTIVE = old_client, old_model, old_active
        m.st._METRICS = old_metrics


def test_write_memory_file_derived_provenance_round_trips_through_archive():
    """archive -> restore of a derived note yields EXACTLY ONE footer, still
    derived; a pre-`_derived` note restores with an unchanged (derived-free)
    footer, so existing footer parsing is unaffected."""
    root = tempfile.mkdtemp(prefix="nx-derived-arch-")
    old_root, old_home = m.st.PROJECT_ROOT, m.st.SESSION_HOME
    m.st.PROJECT_ROOT = root
    m.st.SESSION_HOME = os.path.join(root, "home")
    try:
        derived = _memory_cmd._write_memory_file(
            "Derived note", "body of the derived note", "project", derived="memory")
        plain = _memory_cmd._write_memory_file(
            "Plain note", "body of the plain note", "project")
        assert derived is not None and plain is not None
        dpath = derived[0]
        with open(dpath, encoding="utf-8") as f:
            dcontent = f.read()
        assert dcontent.count("_derived: memory") == 1, dcontent
        assert "_derived" not in dcontent.split("\n---\n", 1)[0]
        with open(plain[0], encoding="utf-8") as f:
            pcontent = f.read()
        assert "_derived" not in pcontent, pcontent  # untouched default path

        # Both note files are already moved down into the run dir by helper;
        # only the restore side is exercised here.
        _archive_all_notes(os.path.join(root, ".nx", "memories"), "run")
        _memory_cmd._cmd_memory_archived("Derived note")
        _memory_cmd._cmd_memory_archived("Plain note")
        restored = [f for f in os.listdir(os.path.join(root, ".nx", "memories"))
                    if f.endswith(".md")]
        assert len(restored) == 2, restored
        texts = {}
        for f in restored:
            with open(os.path.join(root, ".nx", "memories", f), encoding="utf-8") as fh:
                texts[f] = fh.read()
        dtext = [t for t in texts.values() if t.startswith("# Derived note")]
        ptext = [t for t in texts.values() if t.startswith("# Plain note")]
        assert len(dtext) == 1 and len(ptext) == 1
        assert dtext[0].count("---") == 1, dtext[0]        # no doubled footer
        assert dtext[0].count("_derived: memory") == 1, dtext[0]
        assert _memory_cmd._note_derived(dtext[0]) == "memory"
        assert ptext[0].count("---") == 1, ptext[0]
        assert "_derived" not in ptext[0], ptext[0]
        assert "body of the derived note" in dtext[0]
    finally:
        m.st.PROJECT_ROOT, m.st.SESSION_HOME = old_root, old_home


# --- secret redaction at the one writer (0.158.20) -----------------------------
#
# Notes are RE-INJECTED into every future session (session-start note, preference
# kinds every session, the FTS index, any backup), so a credential that merely
# appeared in tool output must not be distilled into one and then live in every
# prompt forever. The mitigation is at the single writer: REDACT, never block.
# Full pattern-by-pattern coverage lives here; the two traps (the drift stamp's
# sha, and a legitimate `[REDACTED:` string) are the ones a "helpful" extra hex
# rule would break.

# Every token line below carries a trailing `# nx-allow-secret`: the commit
# guard honours the marker ONLY on that same added line, ONLY in a file that
# already exists at HEAD, and ONLY on a test-shaped path (nx/_gitcommit.py).
# The literals stay literal and real-shaped -- the test's point is that each
# pattern matches the format a real key has.
SECRET_CASES = [
    ("anthropic_key", "sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"),  # nx-allow-secret
    ("openai_key", "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"),  # nx-allow-secret
    ("openai_key", "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"),  # nx-allow-secret
    ("stripe_key", "sk_live_abcdefghijklmnop1234567890"),
    ("stripe_key", "pk_live_abcdefghijklmnop1234567890"),
    ("github_token", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"),  # nx-allow-secret
    ("github_token", "gho_abcdefghijklmnopqrstuvwxyz0123456789"),  # nx-allow-secret
    ("github_token", "github_pat_11ABCDEFG0abcdefghijklmn_9xYzabcdefghijklmnop"),
    ("aws_key", "AKIAIOSFODNN7EXAMPLE1"),  # nx-allow-secret
    ("slack_token", "xoxb-1234567890-abcdefghij"),  # nx-allow-secret
    ("google_key", "AIzaSyabcdefghijklmnopqrstuvwxyz0123456"),
    ("jwt", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
    ("db_url_password", "postgres://app:S3cr3tPw@db.internal:5432/prod"),
    ("private_key_block",
     "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA\n"  # nx-allow-secret
     "-----END OPENSSH PRIVATE KEY-----"),
    ("private_key_marker", "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA"),  # nx-allow-secret
]


def _red_root():
    """A fresh project root for one redaction test (restored by teardown)."""
    root = tempfile.mkdtemp(prefix="nx-red-")
    m.st.PROJECT_ROOT = root
    return root


def _red_written(res):
    """Content of the note a successful memory_write result reports."""
    for ln in res.split("\n"):
        if ln.startswith("file: "):
            with open(os.path.abspath(ln[6:]), encoding="utf-8") as f:
                return f.read()
    raise AssertionError(f"no file line in {res!r}")


def test_each_secret_class_is_redacted_from_the_body():
    for label, secret in SECRET_CASES:
        _red_root()
        res = _memory_cmd.memory_write(
            "Widget notes", f"The deploy reads {secret} from env, so rotate it.",
            scope="project")
        assert res.startswith("saved "), f"{label}: {res}"
        content = _red_written(res)
        assert secret not in content, f"{label} survived:\n{content}"
        assert f"[REDACTED:{label}]" in content, f"{label} marker missing:\n{content}"
        # the lesson survives — only the credential is gone
        assert "The deploy reads" in content and "from env, so rotate it." in content
        # the caller learns it happened, without the matched text coming back
        assert "secret(s) redacted" in res, res
        assert secret not in res


def test_secret_in_the_title_is_redacted_and_never_echoed_back():
    _red_root()
    secret = "sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"
    res = _memory_cmd.memory_write(f"Rotating {secret} today", "body text",
                                   scope="project")
    content = _red_written(res)
    assert secret not in content
    assert content.startswith("# Rotating [REDACTED:anthropic_key] today"), content
    # the filename slug comes from the redacted title too
    assert secret not in os.path.basename(content and _red_fn(res))
    # the tool result reports the title AS WRITTEN, not the raw credential
    assert secret not in res
    assert "[REDACTED:anthropic_key]" in res


def _red_fn(res):
    for ln in res.split("\n"):
        if ln.startswith("file: "):
            return os.path.abspath(ln[6:])
    return ""


def test_bearer_header_and_generic_key_value_are_redacted():
    _red_root()
    res = _memory_cmd.memory_write(
        "Auth quirks",
        "curl -H 'Authorization: Bearer abcdef1234567890abcdef1234567890' api/x\n"
        "APP_SECRET=hunter2hunter2\n"
        "api_key: 'abcdef123456'\n",
        scope="project")
    content = _red_written(res)
    assert "hunter2" not in content, content
    assert "abcdef1234567890" not in content, content
    # the KEY NAME survives, so the note still says WHICH variable held it
    assert "APP_SECRET=" in content and "api_key:" in content
    assert "[REDACTED:secret_value]" in content
    assert "secret(s) redacted" in res


def test_redaction_counts_and_never_repeats_the_matched_text():
    _red_root()
    a = "sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"
    b = "AKIAIOSFODNN7EXAMPLE1"
    res = _memory_cmd.memory_write("Two leaks", f"{a} and {b} both appeared",
                                   scope="project")
    assert "2 secret(s) redacted" in res, res
    assert a not in res and b not in res
    assert "anthropic_key" in res and "aws_key" in res


def test_clean_note_writes_unchanged_no_false_positives():
    """Ordinary note content — shas, versions, URLs, UUIDs, hyphenated prose —
    must be byte-identical, with nothing reported."""
    _red_root()
    body = ("Bump to 0.158.19 (see `nx/_memory_store.py`). The task-tracking "
            "queue stays sk-shaped in prose. Docs at https://example.com/a/b and "
            "git@github.com:acme/widgets.git. Row uuid "
            "1f0e2d3c-4b5a-4978-8665-4321fedcba98; the fix landed in commit "
            "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433.")
    res = _memory_cmd.memory_write("Bump notes", body, scope="project")
    assert "redacted" not in res, res
    assert body in _red_written(res)


def test_hex_sha_in_body_survives_and_the_drift_stamp_stays_intact():
    """THE TRAP. A bare 'long hex' rule would shred the `_drift:` stamp's sha
    (breaking drift demotion + the write-time challenge) and every quoted commit
    sha. Redaction runs on title+body only; the footer is assembled after it."""
    root = _red_root()
    sha = "a3f5b7c9d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6"
    res = _memory_cmd.memory_write(
        "Bisect result",
        f"The regression is at {sha}; also `git show a3f5b7c` in "
        f"nx/_memory_store.py.", scope="project")
    assert "redacted" not in res, res
    content = _red_written(res)
    assert sha in content and "a3f5b7c" in content     # body hex survives
    footer = content.split("\n---\n", 1)[1]
    from nx import _memory_store
    stamp = _memory_store._DRIFT_FIELD_RE.search(footer)
    assert stamp is not None, footer          # the stamp token still parses
    assert stamp.group(1).count("|") == 4, footer   # unscanned: 5 parts intact
    assert "[REDACTED" not in footer, footer  # a bare-hex rule would shred it
    assert len(footer.split("\n")[0]) > 0 and "_drift:" in footer
    # outside a git repo the sha is recorded as '-' (never invented); either way
    # it must be the UNscanned stamp token, not a [REDACTED:…] marker
    assert "[REDACTED" not in footer, footer


def test_legitimate_redacted_string_round_trips_unchanged():
    """A note ABOUT redaction (or copied from a redacted tool result) is stored
    verbatim and reports zero redactions."""
    _red_root()
    body = ("The pass replaces each hit with a `[REDACTED:secret_value]` marker so "
            "a human can see what was there and why it went.")
    res = _memory_cmd.memory_write("Redaction design", body, scope="project")
    assert "redacted" not in res, res
    assert body in _red_written(res)


def test_redaction_never_blocks_a_write_and_covers_the_gate_preview():
    """REDACT, never refuse: a body that is ONLY a secret still writes (an auto
    write that failed would silently lose the lesson)."""
    _red_root()
    res = _memory_cmd.memory_write(
        "All secrets", "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Jn\n"
                       "-----END OPENSSH PRIVATE KEY-----", scope="project")
    assert res.startswith("saved "), res
    content = _red_written(res)
    assert "b3Jn" not in content
    assert "[REDACTED:private_key_block]" in content


def test_redaction_is_at_the_one_writer_so_every_caller_is_covered():
    """The single-choke-point property: `/memory save`, `/retro`, the optimize
    merge and the archive restore all re-enter `_write_memory_file`."""
    labels = []
    written = _memory_cmd._write_memory_file(
        "Retro path", "uses ghp_abcdefghijklmnopqrstuvwxyz0123456789", "global",
        gate=False, kind="gotcha", redactions=labels)
    assert written is not None
    assert labels == ["github_token"], labels
    with open(written[0], encoding="utf-8") as f:
        content = f.read()
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in content
    assert "[REDACTED:github_token]" in content
    note = _memory_cmd._redaction_note(labels)
    assert "1 secret(s) redacted" in note and "ghp_" not in note
    assert _memory_cmd._redaction_note([]) == ""
    # footer fields a caller passes ride the SAME line convention
    assert "_kind: gotcha" in content.split("\n---\n")[1]


def test_memory_save_reports_redaction_to_the_user():
    """`/memory save` distils 48K chars of tool output — the likeliest way a
    credential reaches a note — so its output must say a redaction happened."""
    import types
    root = _red_root()
    secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"

    class _Msg:
        content = f"# Leaky note\n\nexport GITHUB_TOKEN={secret}\n"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = None

        def model_dump(self):
            return {}

    old_client, old_model = m.st.client, m.st.MODEL
    m.st.client = types.SimpleNamespace(
        base_url="http://offline",
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kw: _Resp())))
    m.st.MODEL = "test-model"
    out = []
    real = sys.stdout
    try:
        sys.stdout = types.SimpleNamespace(
            write=lambda s: out.append(s), flush=lambda: None)
        _memory_cmd._cmd_memory_save([{"role": "user", "content": "hi"}])
    finally:
        sys.stdout, m.st.client, m.st.MODEL = real, old_client, old_model
    text = "".join(out)
    assert "secret(s) redacted" in text, text
    assert secret not in text
    notes = [f for f in os.listdir(os.path.join(root, ".nx", "memories"))
             if f.endswith(".md")]
    assert len(notes) == 1, notes                      # the note still got written
    with open(os.path.join(root, ".nx", "memories", notes[0]),
              encoding="utf-8") as f:
        stored = f.read()
    assert secret not in stored and "[REDACTED:github_token]" in stored


# --- supersession as footer data (0.158.20) ------------------------------------

def _sw(title, body, **kw):
    """memory_write in a caller-pinned project root; returns the note's path."""
    res = _memory_cmd.memory_write(title=title, body=body, scope="project", **kw)
    assert res.startswith("saved "), res
    return _red_fn(res)


def _sw_read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_supersedes_by_unique_prefix_retires_and_stamps_both_footers():
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        old = _sw("Redis pooling", "one pool per process")
        mid = _sw("Redis pooling v2", "one pool per namespace")
        res = _memory_cmd.memory_write(title="Redis pooling v3",
                                       body="one pool per worker",
                                       scope="project", supersedes="Redis pooling")
        assert res.startswith("saved "), res
        new = _red_fn(res)
        old_c, mid_c, new_c = _sw_read(old), _sw_read(mid), _sw_read(new)
        # retired note: the pair, pointing at the NEW note's filename
        assert _memory_cmd._note_superseded_by(old_c) == os.path.basename(new)
        assert "_valid_until: 20" in old_c
        # new note: points back at the old one
        assert _memory_cmd._note_supersedes(new_c) == os.path.basename(old)
        # an unrelated note is untouched
        assert _memory_cmd._note_superseded_by(mid_c) == ""
        # same footer LINE as `_scope:` (the strip-regex convention), one footer
        assert "_superseded_by:" in old_c.split("\n---\n")[1].split("\n")[0]
        assert "_supersedes:" in new_c.split("\n---\n")[1].split("\n")[0]
        assert len(new_c.split("\n---\n")) == 2
        assert os.path.exists(old)          # stays on disk: reversible by design
    finally:
        m.st.PROJECT_ROOT = old_root


def test_ambiguous_supersedes_writes_nothing_and_lists_candidates():
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root, old_calls = m.st.PROJECT_ROOT, list(_confirm_calls)
    m.st.PROJECT_ROOT = root
    try:
        a = _sw("Cache layer notes", "redis")
        b = _sw("Cache layer tuning", "memcached")
        pdir = os.path.join(root, ".nx", "memories")
        before = set(os.listdir(pdir))
        _confirm_calls.clear()
        res = _memory_cmd.memory_write(title="Cache merged", body="both",
                                       scope="project", supersedes="Cache layer")
        assert res.startswith("ERROR"), res
        assert "nothing was written" in res
        assert "Cache layer notes" in res and "Cache layer tuning" in res
        assert set(os.listdir(pdir)) == before          # no new note at all
        assert "_superseded_by" not in _sw_read(a)      # and nothing stamped
        assert "_superseded_by" not in _sw_read(b)
        assert not _confirm_calls                       # refused before the gate
    finally:
        m.st.PROJECT_ROOT = old_root
        _confirm_calls[:] = old_calls


def test_unknown_supersedes_writes_nothing():
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        _sw("Real note", "body")
        pdir = os.path.join(root, ".nx", "memories")
        before = set(os.listdir(pdir))
        res = _memory_cmd.memory_write(title="Ghost", body="body",
                                       scope="project", supersedes="no such note")
        assert res.startswith("ERROR") and "nothing was written" in res
        assert set(os.listdir(pdir)) == before
    finally:
        m.st.PROJECT_ROOT = old_root


def test_superseded_note_absent_from_search_the_session_note_and_optimize():
    """THE INVARIANT: one authoritative layer (`_iter_notes`), so every consumer
    inherits it — memory_search, the [Saved memories] note, /memory optimize's
    index, and the delete resolver."""
    from nx import _memory_optimize as opt
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        old = _sw("Widget claim", "widget uniqmarkerbody uses redis")
        _sw("Widget claim", "widget uniqmarkerbody uses memcached",
            supersedes="Widget claim")
        base = os.path.basename(old)
        hits, _ov, total = _memory_cmd._memory_scan("widget uniqmarkerbody",
                                                    "project", 0)
        assert base not in [h.name for h in hits], [h.name for h in hits]
        assert total == 1
        assert base not in {h.name for h in
                            _memory_cmd._scan_memory_entries("project")}
        note = _memory_cmd._load_memory_note(root)
        assert note is not None
        assert note["content"].count("Widget claim") == 1   # only the live one
        assert base not in [n.name for n in opt._collect_notes("project")]
    finally:
        m.st.PROJECT_ROOT = old_root


def test_memory_list_still_shows_superseded_notes():
    """Discoverability is what makes the filter safe: a hidden note you cannot
    find is silent data loss if supersession ever misfires."""
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        _sw("Listable retired", "body one")
        _sw("Listable retired", "body two", supersedes="Listable retired")
        text = _capture(_memory_cmd._cmd_memory_list)
        assert "Listable retired" in text
        assert "superseded (1)" in text, text
        assert "unsupersede" in text, text
    finally:
        m.st.PROJECT_ROOT = old_root


def test_unsupersede_restores_the_note_everywhere():
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        old = _sw("Unsup markerbody", "unsup one")
        _sw("Unsup markerbody", "unsup two", supersedes="Unsup markerbody")
        assert _memory_cmd._memory_scan("unsup markerbody", "project",
                                        0)[2] == 1
        _confirm_calls.clear()
        # target by FILENAME: the live and retired notes share a title, so a
        # title target would be (correctly) refused as ambiguous by the resolver.
        text = _capture(lambda: _memory_cmd._cmd_memory_unsupersede(
            os.path.basename(old)))
        assert "unsuperseded" in text.lower(), text
        content = _sw_read(old)
        assert _memory_cmd._note_superseded_by(content) == ""
        assert "_valid_until" not in content
        assert len(content.split("\n---\n")) == 2         # exactly one footer
        assert _memory_cmd._memory_scan("unsup markerbody", "project", 0)[2] == 2
        assert any("unsupersede" in c for c in _confirm_calls), _confirm_calls
    finally:
        m.st.PROJECT_ROOT = old_root


def test_superseding_an_already_retired_note_redirects_not_chains():
    """A→B→C would leave a dangling middle link; pointing at a retired note
    REDIRECTS to the note that retired it (and restamps, never stacks)."""
    root = tempfile.mkdtemp(prefix="nx-sup-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        a = _sw("Chain alpha", "chainmarker one")
        b = _sw("Chain beta", "chainmarker two", supersedes="Chain alpha")
        res = _memory_cmd.memory_write(title="Chain gamma",
                                       body="chainmarker three", scope="project",
                                       supersedes="Chain beta")
        assert res.startswith("saved "), res
        c = _red_fn(res)
        assert _memory_cmd._note_superseded_by(_sw_read(a)) == os.path.basename(c)
        assert _memory_cmd._note_superseded_by(_sw_read(b)) == os.path.basename(c)
        assert _memory_cmd._note_supersedes(_sw_read(c)) == os.path.basename(b)
        hits, _ov, total = _memory_cmd._memory_scan("chainmarker", "project", 0)
        assert total == 1 and hits[0].name == os.path.basename(c)
        # restamped, not stacked
        assert _sw_read(a).count("_superseded_by:") == 1
        assert _sw_read(a).count("_valid_until:") == 1
    finally:
        m.st.PROJECT_ROOT = old_root


def test_repoint_finds_pointers_beyond_the_head_window():
    """The re-point scan is titles-only, so it must reach the footer through the
    TAIL probe — a note whose body pushes `_superseded_by:` past the 256-byte
    head read still has to be redirected (otherwise chains survive silently)."""
    root = tempfile.mkdtemp(prefix="nx-sup-long-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        long_body = "filler line to push the footer past the head window.\n" * 20
        a = _sw("Long alpha", long_body + "alpha tail marker")
        _sw("Long beta", long_body + "beta tail marker", supersedes="Long alpha")
        assert _memory_cmd._note_superseded_by(_sw_read(a)), "A must start retired"
        res = _memory_cmd.memory_write(title="Long gamma",
                                       body=long_body + "gamma tail marker",
                                       scope="project", supersedes="Long beta")
        assert res.startswith("saved "), res
        c = _red_fn(res)
        assert "_superseded_by: " + os.path.basename(c) in _sw_read(a), \
            "A's pointer (beyond the head window) must follow the chain tip"
        assert _sw_read(a).count("_superseded_by:") == 1
    finally:
        m.st.PROJECT_ROOT = old_root


def _capture(fn):
    import io
    buf = io.StringIO()
    real = sys.stdout
    try:
        sys.stdout = buf
        fn()
    finally:
        sys.stdout = real
    return buf.getvalue()


LEGACY_NO_NEW_FIELDS = (
    "# Legacy note\n\nthe widget is defined in `src/app.py`\n\n"
    "---\n_scope: project  _saved: 2025-01-01 00:00:00  _kind: gotcha  "
    "_derived: memory\n")
LEGACY_WITH_DRIFT = (
    "# Legacy drift\n\nbody in nx/_memory_store.py\n\n"
    "---\n_scope: project  _saved: 2025-01-01 00:00:00  _kind: fact  "
    "_drift: 0123456789abcdef0123456789abcdef01234567|-|nx/_memory_store.py|-|"
    "2025-01-01T00:00:00|/some/repo|-\n")


def test_pre_existing_footers_parse_and_stay_byte_identical():
    """Notes written before the new fields must scan, parse and stay untouched
    (no field invented, no rewrite, still live)."""
    root = tempfile.mkdtemp(prefix="nx-legacy-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        d = os.path.join(root, ".nx", "memories")
        os.makedirs(d)
        bodies = [LEGACY_NO_NEW_FIELDS, LEGACY_WITH_DRIFT]
        paths = []
        for i, body in enumerate(bodies):
            p = os.path.join(d, f"2025010{i}-000000_legacy{i}.md")
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
            paths.append(p)
        notes = _memory_cmd._iter_notes(d)
        assert [n[0] for n in notes] == [os.path.basename(p) for p in
                                         paths][::-1], notes
        for p, body in zip(paths, bodies):
            assert _sw_read(p) == body, p               # byte-identical
            assert "_superseded_by" not in _sw_read(p)
        assert _memory_cmd._note_superseded_by(LEGACY_NO_NEW_FIELDS) == ""
        assert _memory_cmd._note_summary_of(LEGACY_WITH_DRIFT) == ""
        assert _memory_cmd._note_derived(LEGACY_NO_NEW_FIELDS) == "memory"
        assert _memory_cmd._note_kind(LEGACY_NO_NEW_FIELDS) == "gotcha"
        stamp = _memory_cmd._parse_drift(LEGACY_WITH_DRIFT)
        assert stamp is not None and stamp.sha == "0123456789abcdef" \
            "0123456789abcdef01234567"
    finally:
        m.st.PROJECT_ROOT = old_root


def test_unsupersede_on_a_note_that_was_never_retired_is_a_no_op():
    root = tempfile.mkdtemp(prefix="nx-legacy-")
    old_root = m.st.PROJECT_ROOT
    m.st.PROJECT_ROOT = root
    try:
        d = os.path.join(root, ".nx", "memories")
        os.makedirs(d)
        p = os.path.join(d, "20250101-000000_legacy.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(LEGACY_NO_NEW_FIELDS)
        text = _capture(lambda: _memory_cmd._cmd_memory_unsupersede(
            "Legacy note"))
        assert "not superseded" in text, text
        assert _sw_read(p) == LEGACY_NO_NEW_FIELDS
    finally:
        m.st.PROJECT_ROOT = old_root


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))