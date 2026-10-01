#!/usr/bin/env python3
"""Regression tests for the clobber guard (0.104.2).

run_bash had no static write coverage (the write fence's own docstring
records the gap): a YOLO-mode session destroyed a populated ~/.env with
`cat > ~/.env` — a truncation no tool showed a diff for and nothing
backed up. The guard forces an approval prompt (even in YOLO) when a
command would TRUNCATE an existing, non-empty file. Appends, FD
duplication, new files and empty targets stay prompt-free.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-clobber-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for k in list(os.environ):
    if k.startswith("NX_") and k not in ("NX_HOME", "NX_SESSIONS_DIR"):
        del os.environ[k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx._clobber import clobber_targets, _redirect_targets  # noqa: E402


def _seed(name="f.txt", content="content\n"):
    p = os.path.join(_tmp, name)
    with open(p, "w") as f:
        f.write(content)
    return p


def test_simple_truncate_matches():
    p = _seed()
    assert clobber_targets("cat > " + p) == [p]


def test_heredoc_truncate_matches():
    p = _seed()
    cmd = "cat > " + p + " <<'EOF'\nnew body\nEOF"
    assert clobber_targets(cmd) == [p]


def test_append_is_not_a_truncation():
    p = _seed()
    assert clobber_targets("echo hi >> " + p) == []


def test_fd_duplication_is_not_a_file_truncation():
    p = _seed()
    assert clobber_targets("cmd 2>&1 | tee " + p) == []
    assert clobber_targets("cmd >" + p + " 2>&1") == [p]  # the > still truncates


def test_stderr_truncate_counts():
    p = _seed()
    assert clobber_targets("cmd 2> " + p) == [p]


def test_new_file_is_free():
    new = os.path.join(_tmp, "never-existed.txt")
    assert clobber_targets("cmd > " + new) == []


def test_empty_target_is_free():
    p = _seed("empty.txt", "")
    assert clobber_targets("cmd > " + p) == []


def test_quoted_target():
    p = _seed("has space.txt")
    assert clobber_targets("cmd > '" + p + "'") == [p]
    assert clobber_targets('cmd > "' + p + '"') == [p]


def test_unexpanded_variable_is_skipped_not_guessed():
    assert clobber_targets("cmd > $HOME/whatever") == []


def test_no_redirection_no_match():
    p = _seed()
    assert clobber_targets("grep x " + p) == []
    assert _redirect_targets("echo done") == []


def test_kill_switch():
    p = _seed()
    os.environ["NX_CLOBBER_GUARD"] = "0"
    try:
        assert clobber_targets("cat > " + p) == []
    finally:
        del os.environ["NX_CLOBBER_GUARD"]


def test_log_and_temp_extensions_are_exempt():
    p = _seed("run.log")
    assert clobber_targets("cmd > " + p) == []
    assert _redirect_targets("cmd > " + p) == [p]  # still a target, just exempt


def test_log_extension_opt_out():
    p = _seed("run.log")
    os.environ["NX_CLOBBER_LOG_EXTENSIONS"] = "0"
    try:
        assert clobber_targets("cmd > " + p) == [p]
    finally:
        del os.environ["NX_CLOBBER_LOG_EXTENSIONS"]


def test_lock_and_dotfiles_are_never_exempt():
    p = _seed("bench.lock")
    d = _seed(".myapp.env")
    assert clobber_targets("cmd > " + p) == [p]
    assert clobber_targets("cmd > " + d) == [p and d]
    k = _seed("server.pem")
    assert clobber_targets("cmd > " + k) == [k]


def test_dedup():
    p = _seed()
    assert clobber_targets("cat > " + p + " && cat > " + p) == [p]
