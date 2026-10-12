"""Regression: a command with no trailing newline must survive the exit-code
wrapper (0.36.2).

`_run_bash` wraps the model's command as `( <cmd> ); echo "[exit: $?]"`. When
the closing `)` was glued straight onto the command's last line, every command
ending in something that must sit alone on its own line broke at *parse* time —
above all a heredoc, whose terminator has to be alone on a line. Models emit
heredocs constantly and omit the final newline with similar frequency, so this
surfaced as a random `sh: Syntax error: end of file unexpected (expecting ")")`
across every class of model.

The wrapper-level tests are pure-string (fast, no subprocess). The end-to-end
tests drive the real `_run_bash` launch path so the fix cannot silently regress
into "helper is right, call site is wrong".
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR",
          "NX_BASH_MAX_SECONDS"):
    os.environ.pop(k, None)

import pytest  # noqa: E402


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    """Tier A: nx is imported ONCE (module scope, below); no reload. The env set
    here is rewound universally by conftest. The old reload rebuilt exactly two
    things — the model source wiring and the session-background flags — both
    handled explicitly below without re-executing ~100 modules."""
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    # Rebuild just the model-source wiring the reload used to redo: re-discover
    # sources from the env above and switch to the first one (binds st.client +
    # st.MODEL). resolve_model() reads NX_MODEL ("test-model"), so no network
    # round-trip happens; the dead 127.0.0.1:9 endpoint is never contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(m._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # These tests launch real jobs — pin (not just reset) the session flags they
    # flip, so monkeypatch rewinds them at teardown and later files don't see a
    # session that "already backgrounded something".
    st = m._state
    monkeypatch.setattr(st, "_SESSION_HAS_BACKGROUNDED", False, raising=False)
    monkeypatch.setattr(st, "_PENDING_BACKGROUND_NOTICE", None, raising=False)
    monkeypatch.setattr(st, "_BG_SCAN_DONE", False, raising=False)
    monkeypatch.setattr(st, "_PENDING_BG_COMPLETION_NOTICE", None, raising=False)
    monkeypatch.setattr(st, "_BG_NOTICE_SEEN", None, raising=False)
    yield m, tmp_path


# Tier A: import nx once, at module scope, under the scrubbed env above.
import nx as m  # noqa: E402,F811


# --- wrapper shape (pure string) ---------------------------------------------

def test_close_paren_never_glued_to_last_line():
    from nx._bash_jobs import _wrap_exit_trailer
    for cmd in ("printf hi", "printf hi\n", "cat <<'EOF'\nhi\nEOF", "echo x # c"):
        wrapped = _wrap_exit_trailer(cmd)
        # the `)` must always open a line by itself, whatever the input ended with
        assert "\n);" in wrapped, cmd


def test_trailing_comment_survives():
    from nx._bash_jobs import _wrap_exit_trailer
    # glued on, `)` would be swallowed by the comment and the subshell never closes
    assert _wrap_exit_trailer("true # note").startswith("(true # note\n)")


# --- end-to-end through the real launch path ---------------------------------

def _drain(run_bash, command, timeout=5, want_code=True):
    """Run via _run_bash and return (log_content, exit_code).

    exit_code is None when the trailer never appears — which is what happens
    when `sh` fails to *parse* the script at all: the trailing
    `echo "[exit: $?]"` never runs either. Pass want_code=False for that case.
    """
    pid, log_path = run_bash(command)
    assert pid.isdigit(), f"launcher returned no pid: {pid!r}"
    read_log = run_bash.__globals__["_read_log"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        content, code = read_log(log_path)
        if code is not None:
            return content, code
        # a parse error exits immediately and writes no trailer: the process is
        # gone, so stop waiting instead of burning the whole window
        if not want_code and not _alive(pid):
            return read_log(log_path)[0], None
        time.sleep(0.05)
    pytest.fail(f"command never finished: {command!r}")


def _alive(pid):
    """True unless the pid is provably gone. `os.kill(pid, 0)` returns None on
    success, so the live case must be returned explicitly, not fallen through."""
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but not signalable by us: still counts as alive


def test_heredoc_without_trailing_newline(nx_fresh):
    """The exact reported failure: heredoc, no final newline."""
    nx, _ = nx_fresh
    from nx._bash_jobs import _run_bash
    cmd = "python3 - <<'PY'\nprint('hello from heredoc')\nPY"  # no trailing \n
    out, code = _drain(_run_bash, cmd)
    assert "hello from heredoc" in out, out
    assert code == 0, out


def test_heredoc_with_trailing_newline_still_works(nx_fresh):
    nx, _ = nx_fresh
    from nx._bash_jobs import _run_bash
    out, code = _drain(_run_bash, "cat <<'EOF'\nbody\nEOF\n")
    assert out.strip() == "body", out
    assert code == 0, out


def test_trailing_comment_and_case_block(nx_fresh):
    nx, _ = nx_fresh
    from nx._bash_jobs import _run_bash
    out, code = _drain(_run_bash, "echo ok  # trailing comment")
    assert out.strip() == "ok" and code == 0, out
    case = "case a in\na) echo matched;;\nesac"  # `esac` must be alone on its line
    out, code = _drain(_run_bash, case)
    assert out.strip() == "matched" and code == 0, out


def test_exit_code_still_captured(nx_fresh):
    """The newline must not eat the trailer `_read_log` parses."""
    nx, _ = nx_fresh
    from nx._bash_jobs import _run_bash
    out, code = _drain(_run_bash, "false")
    assert code == 1, out
    out, code = _drain(_run_bash, "true")
    assert code == 0, out
    # multi-line + nonzero: exit status is the *last* command's, unchanged
    out, code = _drain(_run_bash, "echo one\necho two\nexit 7")
    assert "one" in out and "two" in out and code == 7, out


# --- fused-idiom delimiter trap (pre-flight diagnosis) ------------------------

from nx._bash_jobs import _heredoc_quote_trap


def _fused(core: str = "EOF") -> str:
    """The ssh quote-embedding idiom wrapped around a delimiter word: every
    single quote respelled as an alternating double/single-quote pair."""
    sq, dq = chr(39), chr(34)
    return sq + dq + sq + sq + core + sq + sq + dq + sq


def test_trap_fires_on_reported_ssh_shape():
    """The exact field failure: ssh + fused-idiom delimiter + bare EOF close."""
    cmd = "ssh sunk 'cat > /tmp/x.sh' << " + _fused() + "\n#!/bin/bash\necho body\nEOF"
    msg = _heredoc_quote_trap(cmd)
    assert msg is not None
    assert "quoting idiom" in msg
    assert "ssh quote-embedding" in msg
    assert "<< 'EOF'" in msg  # the correction is spelled out


def test_trap_fires_for_any_bare_terminator_core():
    cmd = "ssh h 'cat > /tmp/x.sh' << " + _fused("PY") + "\nbody\nPY"
    assert _heredoc_quote_trap(cmd) is not None


def test_legal_delimiter_spellings_pass():
    sq, dq = chr(39), chr(34)
    ok = [
        "cat <<'EOF'\nhi\nEOF",
        "cat <<" + sq + sq + "EOF" + sq + sq + "\nhi\nEOF",
        "cat <<-'EOF'\nhi\n   EOF",
        "python3 - <<'PY'\nprint(1)\nPY",
        "cat <<EOF\nhi\nEOF",
    ]
    for cmd in ok:
        assert _heredoc_quote_trap(cmd) is None, cmd


def test_backslash_escaped_delimiter_passes():
    # built from chr() so the literal never reads like quote-soup to reviewers
    bs = chr(92)
    assert _heredoc_quote_trap("cat <<" + bs + "EOF\nhi\nEOF") is None


def test_unterminated_or_selfconsistent_body_not_flagged():
    sq, dq = chr(39), chr(34)
    # no bare terminator anywhere: broken, but not by THIS trap — stay silent
    assert _heredoc_quote_trap("cat << " + _fused() + "\nbody") is None
    # body closes with the token the delimiter really decodes to: consistent
    exotic = dq + sq + "EOF" + sq + dq
    assert _heredoc_quote_trap("cat << " + _fused() + "\nbody\n" + exotic) is None


def test_run_bash_refuses_trapped_command_without_launching(nx_fresh, monkeypatch):
    """The diagnosis comes back as the tool result and nothing is executed —
    not even an approval prompt for a command that cannot run."""
    nx, _ = nx_fresh
    from nx import _bash_jobs
    launched: list[str] = []
    monkeypatch.setattr(_bash_jobs, "_run_bash",
                        lambda c: launched.append(c) or ("0", "/dev/null"))
    msg = _bash_jobs.run_bash("ssh h 'cat > /tmp/x.sh' << " + _fused() + "\nbody\nEOF")
    assert "quoting idiom" in msg
    assert not launched


# --- suffixed terminator trap (pre-flight diagnosis) --------------------------

from nx._bash_jobs import _heredoc_terminator_suffix_trap


def test_suffix_trap_fires_on_reported_shape():
    """The exact field failure: '2>&1' glued onto the terminator line."""
    cmd = "cat > /tmp/x <<EOF\nbody\nEOF 2>&1"
    msg = _heredoc_terminator_suffix_trap(cmd)
    assert msg is not None
    assert "2>&1" in msg
    assert "nothing was garbled" in msg  # names the misdiagnosis
    assert "EOF 2>&1" in repr(msg) or "EOF 2>&1" in msg


def test_suffix_trap_fires_for_chain_and_redirection():
    for tail in ("EOF && echo done", "EOF > /tmp/out"):
        cmd = "cat <<EOF\nbody\n" + tail
        assert _heredoc_terminator_suffix_trap(cmd) is not None, tail


def test_wellformed_heredocs_not_flagged():
    ok = [
        "cat <<EOF\nbody\nEOF",
        "cat <<EOF\nbody\nEOF\n",
        "cat <<EOF 2>&1\nbody\nEOF",        # redirection on the OPERATOR line: legal
        "cat <<-'EOF'\n\tbody\n\tEOF",
        "cat <<EOF\nbody\nEOF\necho next",  # bare terminator present somewhere
        "cat <<EOF\nEOF is a word in the body\nx\nEOF",  # legal body line w/ delim
    ]
    for cmd in ok:
        assert _heredoc_terminator_suffix_trap(cmd) is None, cmd


def test_quoted_delimiter_suffix_trap():
    """Quoted delimiter decodes; trap fires on the decoded terminator."""
    cmd = "cat <<'EOF'\nbody\nEOF 2>&1"
    assert _heredoc_terminator_suffix_trap(cmd) is not None


def test_run_bash_refuses_suffixed_terminator(nx_fresh, monkeypatch):
    from nx import _bash_jobs
    launched: list[str] = []
    monkeypatch.setattr(_bash_jobs, "_run_bash",
                        lambda c: launched.append(c) or ("0", "/dev/null"))
    msg = _bash_jobs.run_bash("cat <<EOF\nbody\nEOF 2>&1")
    assert "nothing was garbled" in msg
    assert not launched


def test_run_bash_still_runs_good_heredoc(nx_fresh, monkeypatch):
    """The pre-flight check must not scare off legitimate heredocs."""
    nx, _ = nx_fresh
    from nx import _bash_jobs
    monkeypatch.setattr(nx, "_confirm", lambda *a, **k: True)
    out = _bash_jobs.run_bash("cat <<'EOF'\nbody\nEOF\n", timeout=10)
    assert "body" in out, out
    assert "[exit:" not in out  # the trailer is still stripped from results

