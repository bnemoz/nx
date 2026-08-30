#!/usr/bin/env python3
"""Tests for the background-command gating added for improvement-plan items
#5 ("make the docker/CUDA guidance in SYSTEM conditional") and #6 ("expose
wait_background conditionally"): the detailed background-operating guidance
and the wait_background tool schema are only added to context once a
run_bash call has actually backgrounded this session — a session where
nothing ever backgrounds shouldn't pay tokens for either.

Also covers #8 (`nx --tokens` / `--bench`), the flag that prints the
mechanically-counted prompt-token floor instead of an eyeballed README figure.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-bg-gating-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402


def _reset_background_state():
    m._SESSION_HAS_BACKGROUNDED = False
    m._PENDING_BACKGROUND_NOTICE = None


# _SESSION_HAS_BACKGROUNDED / _PENDING_BACKGROUND_NOTICE are module-level
# globals on `nx` itself, shared across every test file pytest imports in
# one run (not just within this file) — reset on both sides of every test so
# a test here can never leak state into a test in another file that doesn't
# know to reset it (e.g. test_initial_context.py's fresh-session assumption).
setup_function = lambda _function: _reset_background_state()
teardown_function = lambda _function: _reset_background_state()


# ---------------------------------------------------------------------------
# _active_tools()
# ---------------------------------------------------------------------------
def test_active_tools_excludes_wait_background_by_default():
    names = [t["function"]["name"] for t in m._active_tools()]
    assert "wait_background" not in names
    # multi_edit's schema is sent from the start (unlike wait_background) —
    # only a usage nudge is deferred, not availability (see _active_tools()).
    assert "multi_edit" in names
    assert set(names) == {"read_file", "write_file", "edit_file", "delete_file",
                          "list_dir", "run_bash", "search_files", "git_status",
                          "git_diff", "git_log", "sed_edit", "multi_edit",
                          "apply_patch", "todo_write", "memory_write",
                          "memory_search"}


def test_active_tools_includes_wait_background_once_flag_set():
    m._SESSION_HAS_BACKGROUNDED = True
    names = [t["function"]["name"] for t in m._active_tools()]
    assert "wait_background" in names


def test_full_tools_list_still_has_every_tool_for_callers_that_want_the_whole_schema():
    # TOOLS itself (as opposed to _active_tools()) stays the complete list —
    # e.g. --tokens's "full schema" framing and DISPATCH both rely on it.
    names = [t["function"]["name"] for t in m.TOOLS]
    assert names.count("wait_background") == 1
    assert len(m.TOOLS) == 19  # 15 base + wait_background + spawn_agent + read_image + delete_file


# ---------------------------------------------------------------------------
# run_tool() detecting the first backgrounded run_bash result
# ---------------------------------------------------------------------------
def test_run_tool_flips_flag_on_first_backgrounded_result(monkeypatch):
    monkeypatch.setitem(
        m.DISPATCH, "run_bash",
        lambda **kw: "[background] PID 123 | log: /tmp/x.log\nstill running",
    )
    assert m._SESSION_HAS_BACKGROUNDED is False
    result = m.run_tool("run_bash", {"command": "sleep 100"})
    assert result.startswith("[background]")
    assert m._SESSION_HAS_BACKGROUNDED is True
    assert m._PENDING_BACKGROUND_NOTICE == m.BACKGROUND_GUIDANCE


def test_run_tool_only_sets_pending_notice_on_the_first_transition(monkeypatch):
    m._SESSION_HAS_BACKGROUNDED = True  # already flipped by an earlier call
    monkeypatch.setitem(
        m.DISPATCH, "run_bash",
        lambda **kw: "[background] PID 456 | log: /tmp/y.log\nstill running",
    )
    m.run_tool("run_bash", {"command": "sleep 200"})
    assert m._PENDING_BACKGROUND_NOTICE is None


def test_run_tool_ignores_non_backgrounded_run_bash_results(monkeypatch):
    monkeypatch.setitem(m.DISPATCH, "run_bash", lambda **kw: "[exit 0]\nok")
    m.run_tool("run_bash", {"command": "echo hi"})
    assert m._SESSION_HAS_BACKGROUNDED is False
    assert m._PENDING_BACKGROUND_NOTICE is None


def test_run_tool_ignores_other_tools_even_if_result_looks_backgrounded(monkeypatch):
    # Only run_bash's own result should ever trip the flag.
    monkeypatch.setitem(m.DISPATCH, "read_file", lambda **kw: "[background] not really")
    m.run_tool("read_file", {"path": "x"})
    assert m._SESSION_HAS_BACKGROUNDED is False


# ---------------------------------------------------------------------------
# _flush_pending_background_notice()
# ---------------------------------------------------------------------------
def test_flush_pending_background_notice_appends_once():
    m._PENDING_BACKGROUND_NOTICE = m.BACKGROUND_GUIDANCE
    messages = [{"role": "system", "content": "sys"}]
    m._flush_pending_background_notice(messages)
    assert messages[-1] == {"role": "user", "content": m.BACKGROUND_GUIDANCE}
    assert m._PENDING_BACKGROUND_NOTICE is None
    m._flush_pending_background_notice(messages)  # no-op the second time
    assert len(messages) == 2


def test_flush_pending_background_notice_noop_when_nothing_pending():
    messages = [{"role": "system", "content": "sys"}]
    m._flush_pending_background_notice(messages)
    assert len(messages) == 1


# ---------------------------------------------------------------------------
# _messages_show_prior_backgrounding() — used on session resume
# ---------------------------------------------------------------------------
def test_messages_show_prior_backgrounding_false_for_plain_history():
    assert m._messages_show_prior_backgrounding(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    ) is False


def test_messages_show_prior_backgrounding_true_when_marker_present():
    assert m._messages_show_prior_backgrounding([
        {"role": "user", "content": "run something"},
        {"role": "tool", "content": "[background] PID 1 | log: /tmp/x.log"},
    ]) is True


# ---------------------------------------------------------------------------
# --tokens / --bench (#8)
# ---------------------------------------------------------------------------
def test_cli_tokens_flag_recognized_and_prints_floor(capsys):
    assert m._cli_tokens(["--tokens"]) is True
    out = capsys.readouterr().out
    assert "prompt-token floor" in out
    assert "SYSTEM" in out and "TOOLS" in out


def test_cli_bench_alias_recognized(capsys):
    assert m._cli_tokens(["--bench"]) is True
    capsys.readouterr()


def test_cli_tokens_ignores_unrelated_args():
    assert m._cli_tokens(["--yolo"]) is False
    assert m._cli_tokens([]) is False


def test_cli_tokens_reflects_backgrounded_state(capsys):
    m._SESSION_HAS_BACKGROUNDED = True
    m._cli_tokens(["--tokens"])
    out = capsys.readouterr().out
    assert "wait_background excluded" not in out


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
