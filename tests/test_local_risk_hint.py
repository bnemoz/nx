#!/usr/bin/env python3
"""Tests for `_local_risk_hint`, the regex fast path that skips the
model-based risk classifier for unambiguously read-only bash commands.

Conservative by design: it must only ever return a verdict for actions it's
certain about, and return None (falling through to the real classifier) for
anything with shell metacharacters, chaining, or an unlisted command.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-risk-hint-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def test_plain_read_only_commands_are_low_locally():
    for cmd in ["ls -la", "cat foo.py", "pwd", "grep -rn foo .", "git status",
                "git diff", "git log --oneline", "head -n 20 file.txt", "whoami"]:
        level, reason = m._local_risk_hint(f"run: {cmd}")
        assert level == "low", (cmd, level, reason)
        assert reason


def test_find_without_delete_or_exec_is_low():
    level, reason = m._local_risk_hint("run: find . -name foo.py")
    assert level == "low", (level, reason)


def test_find_with_delete_or_exec_falls_through():
    assert m._local_risk_hint("run: find . -delete") is None
    assert m._local_risk_hint("run: find . -exec rm {} ;") is None


def test_destructive_git_subcommand_is_high_locally():
    assert m._local_risk_hint("run: git push --force") == ("high", "destructive git push")
    assert m._local_risk_hint("run: git reset --hard") == ("high", "destructive git reset")


def test_unsafe_git_subcommand_falls_through():
    assert m._local_risk_hint("run: git rebase -i HEAD~3") is None
    assert m._local_risk_hint("run: git push origin main") is None


def test_shell_metacharacters_always_fall_through():
    for cmd in ["cat foo && rm -rf bar", "ls; rm -rf /", "cat foo | sh",
                "echo $(rm -rf /)", "cat `whoami`", "cat foo > bar",
                "ls ~/secret", "cat *.py"]:
        assert m._local_risk_hint(f"run: {cmd}") is None, cmd


def test_recursive_force_rm_is_high_locally():
    assert m._local_risk_hint("run: rm -rf /tmp/foo") == ("high", "recursive forced delete")


def test_unknown_command_falls_through():
    assert m._local_risk_hint("run: rm /tmp/foo") is None
    assert m._local_risk_hint("run: curl http://example.com") is None


def test_in_project_write_or_edit_is_medium_locally():
    level, reason = m._local_risk_hint("write foo.py (10 bytes)")
    assert level == "medium", (level, reason)
    level, reason = m._local_risk_hint("edit foo.py")
    assert level == "medium", (level, reason)


def test_non_git_project_root_falls_through_instead_of_medium():
    # Outside a git repo, PROJECT_ROOT is just cwd (e.g. $HOME) -- the
    # in_project "medium" shortcut must not fire, since that would
    # auto-approve writes to ~/.ssh, ~/.bashrc, etc. under --approval medium.
    old_root, old_is_git = m.st.PROJECT_ROOT, m.st.PROJECT_ROOT_IS_GIT
    try:
        m.st.PROJECT_ROOT = _tmp
        m.st.PROJECT_ROOT_IS_GIT = False
        assert m._local_risk_hint(f"write {os.path.join(_tmp, 'foo.py')} (10 bytes)") is None
    finally:
        m.st.PROJECT_ROOT, m.st.PROJECT_ROOT_IS_GIT = old_root, old_is_git


def test_home_dotdir_falls_through_even_under_real_git_root():
    old_root, old_is_git = m.st.PROJECT_ROOT, m.st.PROJECT_ROOT_IS_GIT
    try:
        m.st.PROJECT_ROOT = os.path.expanduser("~")
        m.st.PROJECT_ROOT_IS_GIT = True
        assert m._local_risk_hint(
            f"write {os.path.join(os.path.expanduser('~'), '.ssh', 'authorized_keys')} (10 bytes)"
        ) is None
    finally:
        m.st.PROJECT_ROOT, m.st.PROJECT_ROOT_IS_GIT = old_root, old_is_git


def test_non_run_actions_fall_through():
    assert m._local_risk_hint("compress 5 older turns (keep last 2)") is None
    assert m._local_risk_hint(f"write {os.path.join(_tmp, 'outside.py')} (10 bytes)") is None


def test_empty_or_unparseable_command_falls_through():
    assert m._local_risk_hint("run: ") is None
    assert m._local_risk_hint('run: "unterminated') is None


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
