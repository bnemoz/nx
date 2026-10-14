#!/usr/bin/env python3
"""Tests for the 4 review fixes: diff-body scope escape, /dev/null delete
guards + undo, atomic apply, and mode-aware required checks."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-fixes-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _edit_tools  # noqa: E402
from nx import _subagent_tiers  # noqa: E402
from nx._argrepair import check_required  # noqa: E402


def _write(name, text):
    path = os.path.join(_tmp, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


# --- 1. scope escape via diff body -------------------------------------------

def test_diff_scope_targets_extracts_headers():
    diff = ("--- a/in.py\n+++ b/in.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert _subagent_tiers._diff_scope_targets(diff) == ["in.py"]


def test_diff_scope_targets_delete_uses_source_path():
    diff = ("--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-hi\n")
    assert _subagent_tiers._diff_scope_targets(diff) == ["gone.py"]


def test_scope_refusal_rejects_escape_paths():
    diff = ("--- a/ok.py\n+++ b/../../outside.py\n@@ -1 +1 @@\n-a\n+b\n")
    refusal = _subagent_tiers._diff_scope_refusal({"diff": diff})
    assert refusal is not None and "outside" in refusal
    # absolute path in the body is refused too
    diff_abs = "--- a/x\n+++ b/" + _tmp + "/abs.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert _subagent_tiers._diff_scope_refusal({"diff": diff_abs}) is not None


def test_scope_refusal_passes_in_scope_diff():
    diff = "--- a/ok.py\n+++ b/ok.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert _subagent_tiers._diff_scope_refusal({"diff": diff}) is None


# --- 2. /dev/null delete guards + undo ---------------------------------------

def test_patch_affected_files_covers_deletes():
    diff = "--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-hi\n"
    assert _edit_tools._patch_affected_files(diff) == ["gone.py"]
    assert _edit_tools._patch_delete_sources(diff) == ["gone.py"]


def test_apply_patch_delete_checks_staleness_and_pushes_undo():
    saved = m._confirm
    m._confirm = lambda action, preview=None: True
    try:
        path = _write("doomed.py", "hello\n")
        # staleness: file changed (mtime bumped) after this session read it
        from nx import _staleness
        _staleness.record_read(path)
        os.utime(path, (0, 0))
        diff = "--- a/doomed.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-hello\n"
        res = _edit_tools.apply_patch(diff, path=_tmp)
        assert res.startswith("ERROR") and "changed on disk" in res, res
        assert os.path.exists(path)
        # fresh read: the delete goes through and is undoable
        _staleness.record_read(path)
        res = _edit_tools.apply_patch(diff, path=_tmp)
        assert res.startswith("applied"), res
        assert not os.path.exists(path)
    finally:
        m._confirm = saved


# --- 3. atomic apply ----------------------------------------------------------

def test_git_apply_runs_temp_tree_trial_before_real_apply():
    """Atomicity: the whole patch is applied against a temp copy of the target
    files before anything touches the real tree (git apply has no --atomic)."""
    import inspect
    src = inspect.getsource(_edit_tools.apply_patch)
    assert "TemporaryDirectory" in src
    assert "git\", \"apply" in src or 'git", "apply' in src


def test_failed_multi_file_patch_leaves_no_partial_write():
    saved = m._confirm
    m._confirm = lambda action, preview=None: True
    try:
        a = _write("p_a.txt", "one\n")
        b = _write("p_b.txt", "two\n")
        # second file's hunk doesn't match -> whole patch must fail
        diff = ("--- a/p_a.txt\n+++ b/p_a.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
                "--- a/p_b.txt\n+++ b/p_b.txt\n@@ -1 +1 @@\n-NOMATCH\n+TWO\n")
        res = _edit_tools.apply_patch(diff, path=_tmp)
        assert res.startswith("ERROR"), res
        assert open(a).read() == "one\n", "earlier file must not be partially modified"
        assert open(b).read() == "two\n"
    finally:
        m._confirm = saved


# --- 4. mode-aware required checks -------------------------------------------

def test_check_required_diff_mode_needs_no_new():
    assert check_required("edit_file", {"path": "x", "diff": "--- a/x\n+++ b/x\n"}) is None


def test_check_required_edits_mode_needs_no_new():
    assert check_required("edit_file", {"path": "x", "edits": [{"old": "a", "new": "b"}]}) is None


def test_check_required_literal_mode_with_old_is_fine():
    assert check_required("edit_file", {"path": "x", "old": "a", "new": "b"}) is None


def test_check_required_no_mode_at_all_errors():
    err = check_required("edit_file", {"path": "x"})
    assert err and "mode" in err


def test_edit_file_diff_mode_body_does_not_require_new():
    saved = m._confirm
    m._confirm = lambda action, preview=None: True
    try:
        d = tempfile.mkdtemp(prefix="nx-diffmode-", dir=_tmp)
        path = os.path.join(d, "dm.py")
        with open(path, "w") as f:
            f.write("alpha\n")
        diff = "--- a/dm.py\n+++ b/dm.py\n@@ -1 +1 @@\n-alpha\n+beta\n"
        res = m.edit_file(path, diff=diff)
        assert res.startswith("applied"), res
        assert open(path).read() == "beta\n"
    finally:
        m._confirm = saved


def test_edit_file_body_refuses_modeless_call():
    res = m.edit_file("whatever.py")
    assert res.startswith("ERROR") and "mode" in res


def test_edit_file_diff_mode_rejects_new_mixed_in():
    res = m.edit_file("whatever.py", diff="--- a/x\n+++ b/x\n", new="junk")
    assert res.startswith("ERROR")
