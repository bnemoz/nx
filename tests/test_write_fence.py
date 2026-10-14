"""The hard write fence (ROADMAP 2.1): deterministic, approval-independent.

Every test here sets NX_WRITE_ROOTS/PROJECT_ROOT explicitly — the fence reads
env and st at call time, so no xdist order dependence (tests/conftest.py
rewinds both anyway).
"""
import os
import sys
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


# --- edit_file diff mode targets (the former standalone apply_patch) -----------

def test_diff_mode_target_outside_refused(_fence_env):
    diff = ("--- a/ok.py\n+++ b/ok.py\n@@ -1 +1 @@\n-a\n+b\n"
            "--- a/../../evil.py\n+++ b/../../evil.py\n@@ -1 +1 @@\n-a\n+b\n")
    r = _fence.check_write_fence("edit_file", {"diff": diff, "path": "."})
    assert r.startswith("ERROR: write fence:")
    assert "evil.py" in r


def test_diff_mode_clean_allowed(_fence_env, monkeypatch):
    monkeypatch.chdir(_fence_env)  # patch paths are relative to the cwd like the body's git apply
    diff = "--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert _fence.check_write_fence("edit_file", {"diff": diff, "path": "."}) == ""


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


# --- the run_bash surface (heuristic backstop layer) --------------------------

def test_bash_redirection_outside_refused(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence(
        f"echo hi > {tmp_path / 'out.txt'}").startswith("ERROR: write fence:")


def test_bash_redirection_inside_allowed(_fence_env):
    assert _fence.check_bash_write_fence("echo hi > in.txt") == ""


def test_bash_append_outside_refused(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence(
        f"echo hi >> {tmp_path / 'out.txt'}").startswith("ERROR: write fence:")


def test_bash_tee_outside_refused(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence(
        f"cat a | tee {tmp_path / 'out.txt'}").startswith("ERROR: write fence:")


def test_bash_python_heredoc_write_outside_refused(_fence_env, tmp_path):
    """The incident shape: fenced structured tools, edit re-routed through a
    python here-doc."""
    cmd = ("python3 - <<'EOF'\n"
           f"p = '{tmp_path / 'coherence.rs'}'\n"
           "open(p, 'w').write('x')\n"
           "EOF")
    assert _fence.check_bash_write_fence(cmd).startswith("ERROR: write fence:")


def test_bash_python_heredoc_write_inside_allowed(_fence_env):
    cmd = ("python3 - <<'EOF'\n"
           "open('in.rs', 'w').write('x')\n"
           "EOF")
    assert _fence.check_bash_write_fence(cmd) == ""


def test_bash_cp_mv_outside_refused(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence(
        f"cp a {tmp_path / 'b'}").startswith("ERROR: write fence:")
    assert _fence.check_bash_write_fence(
        f"mv a {tmp_path / 'b'}").startswith("ERROR: write fence:")


def test_bash_heredoc_redirected_outside_refused(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence(
        f"cat <<'EOF' > {tmp_path / 'out.rs'}\nhello\nEOF"
    ).startswith("ERROR: write fence:")


def test_bash_shell_variable_target_refused(_fence_env):
    # An unresolvable destination must refuse, not guess inside.
    assert _fence.check_bash_write_fence(
        "echo hi > $SOMEWHERE/out.txt").startswith("ERROR: write fence:")


def test_bash_denied_target_refused(_fence_env, monkeypatch, tmp_path):
    monkeypatch.setenv("NX_WRITE_ROOTS", os.path.expanduser("~"))
    assert _fence.check_bash_write_fence("echo x > ~/.bashrc").startswith(
        "ERROR: write fence:")


def test_bash_fence_layer_disabled(_fence_env, monkeypatch, tmp_path):
    monkeypatch.setenv("NX_BASH_FENCE", "0")
    assert _fence.check_bash_write_fence(
        f"echo hi > {tmp_path / 'out.txt'}") == ""


def test_bash_read_only_command_allowed(_fence_env):
    for cmd in ("grep -r pattern src/", "ls -la", "git status",
                "rg foo | head -20", "cat a b c"):
        assert _fence.check_bash_write_fence(cmd) == ""


# --- over-refusal fixes (0.152.2) --------------------------------------------
#
# The scanner read `>` wherever it appeared and refused every bare word it
# captured, so it fenced python comparison operators as redirections and
# `cp a b` as an out-of-root write. Pinned in both directions: the benign
# shapes pass, the unresolvable ones still refuse.

def test_bash_comparison_in_heredoc_body_not_a_redirect(_fence_env):
    """`>` inside an inline-python body is python, not a shell redirection."""
    cmd = ("python3 - <<'PY'\n"
           "print('>' if a > b else '<')\n"
           "import sys\n"
           "ok = sys.version > '3'\n"
           "print(ok)\n"
           "PY")
    assert _fence.check_bash_write_fence(cmd) == ""


def test_bash_operator_gt_not_a_redirect(_fence_env):
    for cmd in ("python3 -c 'print(a > b if a >= b else 0)'",
                "awk '{ if ($1 > 0) print }' f"):
        assert _fence.check_bash_write_fence(cmd) == ""


def test_bash_write_text_fences_receiver_not_content(_fence_env, tmp_path):
    """Path(x).write_text(content): the destination is the RECEIVER. Anchoring on the
    paren used to fence the content and never see the path."""
    inside = ("python3 -c 'from pathlib import Path; "
              "Path(\"out.txt\").write_text(\"hello world\")'")
    assert _fence.check_bash_write_fence(inside) == ""
    outside = ("python3 -c 'from pathlib import Path; "
               f"Path(\"{tmp_path / 'out.txt'}\").write_text(\"hello world\")'")
    r = _fence.check_bash_write_fence(outside)
    assert r.startswith("ERROR: write fence:")
    assert str(tmp_path / "out.txt") in r
    assert "hello world" not in r  # the content is never the judged path


def test_bash_write_bytes_receiver_outside_refused(_fence_env, tmp_path):
    cmd = ("python3 - <<'PY'\n"
           "from pathlib import Path\n"
           f"Path(r'{tmp_path / 'blob.bin'}').write_bytes(b'x')\n"
           "PY")
    r = _fence.check_bash_write_fence(cmd)
    assert r.startswith("ERROR: write fence:")
    assert str(tmp_path / "blob.bin") in r


def test_bash_attribute_receiver_refused(_fence_env):
    """A language destination that cannot be reduced to a literal refuses even
    when it looks path-ish: self.path is an attribute chain, not a relative
    path, and Path(q) names a variable."""
    for cmd in ("python3 -c \"self.path.write_text(txt)\"",
                "python3 -c \"Path(q).write_text(txt)\"",
                "python3 -c \"open(self.fh, 'w')\""):
        assert _fence._bash_write_candidates(cmd)[0][1] == _fence._LANG_VAR
        assert _fence.check_bash_write_fence(cmd).startswith("ERROR: write fence:")


def test_bash_bare_argv_targets_allowed(_fence_env):
    """A bare word named by a binary's argv is an ordinary relative path, not
    an unresolvable variable: the bare-identifier refusal is for
    interpreted-language captures only."""
    for cmd in ("cp a b", "mv a b", "tee -a notes", "truncate -s 0 logfile",
                "install -m 644 build/app bin/app", "ln -s target link.txt"):
        assert _fence.check_bash_write_fence(cmd) == ""


def test_bash_curl_dash_o_stdout_allowed(_fence_env):
    """`curl -o-` writes to stdout; `-` is not a filename."""
    assert _fence.check_bash_write_fence("curl -fsSL -o- https://x.test/i.sh") == ""
    assert _fence.check_bash_write_fence("curl -o- https://x.test/i.sh | sh") == ""


def test_bash_curl_output_flag_still_fenced(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence(
        f"curl -o {tmp_path / 'i.sh'} https://x.test/i.sh").startswith("ERROR: write fence:")
    assert _fence.check_bash_write_fence(
        f"curl -o{tmp_path / 'i.sh'} https://x.test/i.sh").startswith("ERROR: write fence:")


def test_bash_quoted_redirect_target_allowed(_fence_env, tmp_path):
    assert _fence.check_bash_write_fence('echo x > "my file.txt"') == ""
    assert _fence.check_bash_write_fence("echo x > 'notes and stuff.md'") == ""
    # quoted does not mean unjudgeable: an outside target still refuses
    r = _fence.check_bash_write_fence(f'echo x > "{tmp_path / "my file.txt"}"')
    assert r.startswith("ERROR: write fence:")
    assert "my file.txt" in r


def test_bash_open_variable_still_refused(_fence_env):
    """The constraint these relaxations must not break: an unresolvable
    destination refuses rather than guessing inside."""
    cmd = "python3 - <<'PY'\nq = 'in.txt'\nopen(q, 'w').write('x')\nPY"
    assert _fence.check_bash_write_fence(cmd).startswith("ERROR: write fence:")
    assert _fence.check_bash_write_fence(
        "python3 -c \"open(dst, 'w')\"").startswith("ERROR: write fence:")


def test_bash_plain_outside_redirect_still_refused(_fence_env):
    assert _fence.check_bash_write_fence("echo x > /tmp/o").startswith("ERROR: write fence:")
    assert _fence.check_bash_write_fence(
        "echo x > $VAR/o").startswith("ERROR: write fence:")


def test_bash_candidates_are_tagged(_fence_env):
    """(path, source) shape: only interpreted-language captures carry the tag
    that triggers the bare-identifier refusal."""
    assert _fence._bash_write_candidates("cp a b") == [("b", "")]
    assert _fence._bash_write_candidates("echo x > 'my file.txt'") == [("my file.txt", "")]
    assert _fence._bash_write_candidates("python3 -c \"open(q, 'w')\"") == [
        ("q", _fence._LANG_VAR)]
    assert _fence._bash_write_candidates(
        "python3 -c \"Path('out.txt').write_text('hi')\"") == [("out.txt", "")]
    # a here-doc body is invisible to the redirect/argv scans, visible to the
    # language scan
    assert _fence._bash_write_candidates(
        "cat <<'EOF' > out.txt\nprint('>' if a > b else '<')\nEOF") == [
        ("out.txt", "")]


def test_run_tool_dispatch_bash_refused_outside(tmp_path, monkeypatch, _fence_env):
    """End-to-end: run_bash's rewrite attempt dies in run_tool before approval."""
    called = []
    monkeypatch.setitem(_nx.DISPATCH, "run_bash", lambda **a: called.append(a) or "ran")
    monkeypatch.setattr(_nx, "_confirm", lambda *a, **k: True, raising=False)
    out = _nx.run_tool("run_bash", {"command": f"echo hi > {tmp_path / 'x.txt'}"})
    assert out.startswith("ERROR: write fence:")
    assert not called
