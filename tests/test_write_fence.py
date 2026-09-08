"""The hard write fence (ROADMAP 2.1): deterministic, approval-independent.

Every test here sets NX_WRITE_ROOTS/PROJECT_ROOT explicitly — the fence reads
env and st at call time, so no xdist order dependence (tests/conftest.py
rewinds both anyway).
"""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import nx as _nx
from nx import _fence
from nx import _state as st


@pytest.fixture(autouse=True)
def _fence_env(tmp_path, monkeypatch):
    """A project root per test + clean fence env; removes the real ~/.nx from
    the picture by pointing NX_HOME into tmp_path. Sets NX_WRITE_FENCE=1
    explicitly: conftest's baseline has it 0 (most files test unfenced tool
    bodies), so this file is where the shipped default-on behavior is pinned."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(st, "PROJECT_ROOT", str(root))
    monkeypatch.setattr(st, "SESSION_HOME", str(tmp_path / ".nx"))
    monkeypatch.delenv("NX_WRITE_ROOTS", raising=False)
    monkeypatch.setenv("NX_WRITE_FENCE", "1")
    yield root


def _roots() -> list[str]:
    return _fence._writable_roots()


# --- roots -------------------------------------------------------------------

def test_roots_include_project_and_session_home(_fence_env, tmp_path):
    roots = _roots()
    assert roots[0] == os.path.realpath(str(_fence_env))
    assert os.path.realpath(str(tmp_path / ".nx")) in roots[1:]
    # +1 for the session home; >2 only when NX_WRITE_ROOTS adds more
    assert len(roots) == 2


def test_roots_extra_and_relative_expansion(_fence_env, monkeypatch, tmp_path):
    monkeypatch.setenv("NX_WRITE_ROOTS", os.pathsep.join(
        [str(tmp_path / "extra"), "rel/inside"]))
    roots = _roots()
    assert os.path.realpath(str(tmp_path / "extra")) in roots
    assert os.path.realpath(str(_fence_env / "rel" / "inside")) in roots


# --- the path tools -----------------------------------------------------------

def test_write_inside_project_allowed(_fence_env):
    assert _fence.check_write_fence("write_file",
                                    {"path": str(_fence_env / "a.py"), "content": "x"}) == ""


def test_write_outside_project_refused(_fence_env, tmp_path):
    r = _fence.check_write_fence("write_file", {"path": str(tmp_path / "elsewhere.py"), "content": "x"})
    assert r.startswith("ERROR: write fence:")
    assert "NX_WRITE_ROOTS" in r and "NX_WRITE_FENCE" in r


def test_relative_path_resolved_against_cwd(_fence_env, monkeypatch):
    # The model passes relative paths; the fence judges where the BODY would
    # write (the body resolves against os.getcwd(), so cwd is the truth).
    monkeypatch.chdir(_fence_env)
    assert _fence.check_write_fence("write_file", {"path": "relative.py", "content": ""}) == ""
    monkeypatch.chdir(os.path.dirname(str(_fence_env)))
    assert _fence.check_write_fence("write_file", {"path": "relative.py", "content": ""}).startswith(
        "ERROR: write fence:")


def test_delete_outside_refused_even_basename_clean(_fence_env, tmp_path):
    r = _fence.check_write_fence("delete_file", {"path": str(tmp_path / "victim.txt")})
    assert r.startswith("ERROR: write fence:")


def test_multi_edit_single_path(_fence_env):
    ok = _fence.check_write_fence("multi_edit", {"path": str(_fence_env / "x.py"), "edits": []})
    bad = _fence.check_write_fence("multi_edit", {"path": "/etc/cron.d/nx", "edits": []})
    assert ok == "" and bad.startswith("ERROR: write fence:")


# --- sed_edit glob expansion ---------------------------------------------------

def test_sed_edit_literal_glob_outside_refused(_fence_env, tmp_path):
    r = _fence.check_write_fence("sed_edit", {"path": str(tmp_path / "**" / "*.py"),
                                              "pattern": "a", "replacement": "b"})
    assert r.startswith("ERROR: write fence:")


def test_sed_edit_matched_file_outside_refused(_fence_env, tmp_path, monkeypatch):
    outside = tmp_path / "elsewhere"
    (outside / "sub").mkdir(parents=True)
    (outside / "sub" / "leak.py").write_text("x = 1\n")
    # glob is matched against the CWD, so chdir for the duration
    monkeypatch.chdir(_fence_env)
    r = _fence.check_write_fence("sed_edit", {"path": "../elsewhere/sub/*.py",
                                              "pattern": "a", "replacement": "b"})
    assert r.startswith("ERROR: write fence:")


def test_sed_edit_inside_glob_allowed(_fence_env, monkeypatch):
    monkeypatch.chdir(_fence_env)
    (Path(_fence_env) / "pkg").mkdir(exist_ok=True)
    (Path(_fence_env) / "pkg" / "m.py").write_text("x = 1\n")
    assert _fence.check_write_fence("sed_edit", {"path": "pkg/*.py",
                                                 "pattern": "a", "replacement": "b"}) == ""


# --- apply_patch targets --------------------------------------------------------

def test_apply_patch_target_outside_refused(_fence_env):
    diff = ("--- a/ok.py\n+++ b/ok.py\n@@ -1 +1 @@\n-a\n+b\n"
            "--- a/../../evil.py\n+++ b/../../evil.py\n@@ -1 +1 @@\n-a\n+b\n")
    r = _fence.check_write_fence("apply_patch", {"diff": diff, "path": "."})
    assert r.startswith("ERROR: write fence:")
    assert "evil.py" in r


def test_apply_patch_clean_allowed(_fence_env, monkeypatch):
    monkeypatch.chdir(_fence_env)  # patch paths are relative to the cwd like the body's git apply
    diff = "--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert _fence.check_write_fence("apply_patch", {"diff": diff, "path": "."}) == ""


# --- symlink / TOCTOU ------------------------------------------------------------

def test_symlink_out_of_fence_refused(_fence_env, tmp_path):
    """The TOCTOU the roadmap item exists for: a committed in-project symlink
    whose target is outside the fence must be refused — _atomic_write writes
    THROUGH the link, so the target is the real write destination."""
    target = tmp_path / "target.txt"
    target.write_text("original\n")
    link = _fence_env / "config"
    os.symlink(str(target), str(link))
    r = _fence.check_write_fence("write_file", {"path": str(link), "content": "x"})
    assert r.startswith("ERROR: write fence:")


def test_symlink_into_denylist_refused_even_inside(_fence_env):
    """An in-root link to ~/.ssh/authorized_keys: the write lands in ~/.ssh,
    so the denylist must win over root membership."""
    ssh_dir = Path(os.path.expanduser("~/.ssh"))
    if not (os.path.isdir(ssh_dir) and ssh_dir.is_symlink() is False):
        pytest.skip("no real ~/.ssh to demonstrate with")
    link = _fence_env / "authorized_keys"
    os.symlink(os.path.join(str(ssh_dir), "authorized_keys"), str(link))
    r = _fence.check_write_fence("write_file", {"path": str(link), "content": "x"})
    assert r.startswith("ERROR: write fence:")


def test_link_farm_inside_fence_still_works(_fence_env, tmp_path):
    """The use case the fence must NOT break: a dotfile farm where the repo
    holds committed links pointing at targets that also live inside the fence
    (e.g. a second tree registered in NX_WRITE_ROOTS)."""
    farm_target = tmp_path / "farm" / "gitconfig"
    farm_target.parent.mkdir()
    farm_target.write_text("[user]\n")
    monkeypatch_env = os.environ.copy()
    os.environ["NX_WRITE_ROOTS"] = str(tmp_path / "farm")
    try:
        link = _fence_env / "dot-gitconfig-named"  # basename NOT in denylist
        os.symlink(str(farm_target), str(link))
        r = _fence.check_write_fence("write_file", {"path": str(link), "content": "x"})
        assert r == ""
    finally:
        os.environ.clear()
        os.environ.update(monkeypatch_env)


def test_denied_target_direct_write_refused(_fence_env):
    r = _fence.check_write_fence("write_file", {"path": os.path.expanduser("~/.gitconfig"), "content": "x"})
    assert r.startswith("ERROR: write fence:")


def test_denied_private_key_suffix(_fence_env, tmp_path):
    r = _fence.check_write_fence("write_file", {"path": str(_fence_env / "host.pem"), "content": "x"})
    assert r.startswith("ERROR: write fence:")


# --- passthrough / opt-out --------------------------------------------------------

def test_read_tools_not_fenced(_fence_env):
    for name in ("read_file", "list_dir", "search_files", "git_status", "todo_write"):
        assert _fence.check_write_fence(name, {"path": "/etc/passwd"}) == ""


def test_fence_disabled(_fence_env, monkeypatch):
    monkeypatch.setenv("NX_WRITE_FENCE", "0")
    assert _fence.check_write_fence("write_file", {"path": "/etc/passwd", "content": "x"}) == ""


def test_run_tool_dispatch_refuses_outside(tmp_path, monkeypatch, _fence_env):
    """End-to-end through run_tool: the fence fires before the body would run."""
    called = []
    monkeypatch.setitem(_nx.DISPATCH, "write_file",
                        lambda **a: called.append(a) or "wrote")
    monkeypatch.setattr(_nx, "_confirm", lambda *a, **k: True, raising=False)
    out = _nx.run_tool("write_file", {"path": str(tmp_path / "out.py"), "content": "x"})
    assert out.startswith("ERROR: write fence:")
    assert not called


def test_run_tool_dispatch_allows_inside(tmp_path, monkeypatch, _fence_env):
    called = []
    monkeypatch.setitem(_nx.DISPATCH, "write_file",
                        lambda **a: called.append(a) or "wrote")
    monkeypatch.setattr(_nx, "_confirm", lambda *a, **k: True, raising=False)
    out = _nx.run_tool("write_file", {"path": str(_fence_env / "in.py"), "content": "x"})
    assert out.startswith("wrote")  # run_tool appends its py_compile verdict
    assert called
