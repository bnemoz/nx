"""`nx -v` must print the version and exit, exactly like `nx --version` —
it used to fall through to the interactive REPL because only the long flag
was recognized (both in main()'s short-circuit and in nx/__init__.py's
import-time guard that skips switch_source()'s network probes)."""
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)



# --- _cli_help_or_version recognizes -v --------------------------------------

def test_version_flag_aliases():
    from nx import _cli
    for argv in (["--version"], ["-v"]):
        assert _cli._cli_help_or_version(argv) is True


def test_bare_v_flag_is_not_version():
    # -v inside a larger token (or unknown flags) must not trigger it
    from nx import _cli
    assert _cli._cli_help_or_version(["-vv"]) is False
    assert _cli._cli_help_or_version(["--session", "-v-thing"]) is False


def test_typo_verb_is_rejected_not_repl(monkeypatch, capsys):
    # `nx v` (forgotten dash) must not silently fall through to the REPL
    import pytest
    from nx import _cli
    for argv in (["v"], ["version"], ["sesions"]):
        with pytest.raises(SystemExit) as e:
            _cli._cli_help_or_version(argv)
        assert e.value.code == 2
        assert "unknown command" in capsys.readouterr().err
    # known verbs and flags still fall through / short-circuit as before
    assert _cli._cli_help_or_version(["run"]) is False
    assert _cli._cli_help_or_version(["--tokens"]) is False
    assert _cli._cli_help_or_version([]) is False


def test_main_prints_version_for_v(capsys, monkeypatch):
    from nx import _cli
    monkeypatch.setattr(sys, "argv", ["nx", "-v"])
    _cli.main()
    out = capsys.readouterr().out
    assert out == f"nx {_cli._cli_version()}\n"


def test_prompt_flag_uses_inline_text(monkeypatch, capsys):
    # -p/--prompt runs one-shot mode with the inline text, no REPL
    from nx import _cli
    import nx
    monkeypatch.setattr(sys, "argv", ["nx", "-p", "hello world"])
    monkeypatch.setattr(nx, "_PROMPT_TEXT", "hello world")
    monkeypatch.setattr(nx, "_PROMPT_FILE", None)
    monkeypatch.setattr(_cli, "_PROMPT_TEXT", "hello world")
    monkeypatch.setattr(_cli, "_PROMPT_FILE", None)
    monkeypatch.setattr(_cli, "_initial_messages", lambda: [])
    captured = {}
    def fake_loop(messages, on_step=None):
        captured["content"] = messages[-1]["content"]
        return {"status": "ok"}
    monkeypatch.setattr(_cli, "_run_model_turn_loop", fake_loop)
    monkeypatch.setattr(_cli, "_ensure_ctx_probe", lambda a: None)
    monkeypatch.setattr(_cli, "_fresh_metrics", lambda session_id=None: object())
    with pytest.raises(SystemExit) as e:
        _cli._run_one_shot()
    assert e.value.code == 0
    assert captured["content"] == "hello world"
    assert capsys.readouterr().out == ""  # no REPL banner, no extra output


def test_prompt_and_prompt_file_fall_through_to_repl(monkeypatch):
    from nx import _cli
    monkeypatch.setattr(_cli, "_PROMPT_TEXT", None)
    monkeypatch.setattr(_cli, "_PROMPT_FILE", None)
    assert _cli._run_one_shot() is False


# --- _help_colored: uv/rustup-style blocks, TTY-gated -------------------------

def test_help_colored_marks_entries_and_headers(monkeypatch):
    import io
    from nx import _cli
    fake_tty = io.StringIO()
    fake_tty.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", fake_tty)
    from nx._ansi import RESET
    out = _cli._help_colored(_cli._HELP_TEXT)
    # plain text input (no ANSI in the source string)
    assert RESET not in _cli._HELP_TEXT
    # token column entries and section headers are wrapped in escapes
    assert "\x1b[38;2;" in out
    assert "\x1b[1mOptions:\x1b[0m" in out
    # entry tokens colored, descriptions stay plain after the gap
    line = next(l for l in out.splitlines() if l.rstrip().endswith(
        "Start in never-prompt mode (auto-approve everything)"))
    assert "--yolo" in line
    # long-only Options entries (indent 6) are entry lines too
    assert "\x1b[0m  ".startswith("\x1b[0m")  # sanity: RESET imported/used
    src_line = next(l for l in out.splitlines() if "--source <name>" in l)
    assert src_line.startswith("      \x1b[38;2;") and src_line.rstrip().endswith(
        "Start on a specific source")
    # continuation and prose lines stay entirely plain
    assert "line per task on stdout. Flags: --parallel N" in out
    assert "See docs/REFERENCE.md for the full NX_* env" in out


def test_help_colored_falls_back_without_tty(monkeypatch):
    import io
    from nx import _cli
    monkeypatch.setattr(sys, "stdout", io.StringIO())  # isatty() -> False
    assert _cli._help_colored(_cli._HELP_TEXT) == _cli._HELP_TEXT


def test_help_colored_no_color_env(monkeypatch):
    import io
    from nx import _cli
    fake_tty = io.StringIO()
    fake_tty.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", fake_tty)
    monkeypatch.setenv("NO_COLOR", "1")
    assert _cli._help_colored(_cli._HELP_TEXT) == _cli._HELP_TEXT


# --- import-time guard skips source setup for -v -----------------------------

def test_import_time_guard_skips_switch_source_for_v(tmp_path):
    """With a source configured, importing nx under `-v` must not call
    switch_source() (which would hit the network before main() runs).

    Tier A: this asserts import-time behavior, so it runs in a subprocess
    instead of `importlib.reload(nx)` — the parent `nx` module is never
    touched and no module-dict rewind is needed. The guard leaves
    st.ACTIVE/st.MODEL unwired (switch_source() is what binds them); if it
    regressed, the dummy source would be discovered AND switched to, so the
    printed flags flip to False/False. NX_SKIP_STARTUP_PROBES=1 (inherited
    from conftest, pinned here) keeps the subprocess offline either way —
    the wiring, not a network timeout, is what we assert, so a regression
    fails instantly instead of after TCP timeouts. (The old test patched
    nx._sources.switch_source and relied on reload re-executing nx's
    `from ._sources import switch_source` to pick the patch up; with no
    reload there is no re-import to intercept, hence the subprocess.)
    """
    env = dict(os.environ)
    for k in ("NX_SOURCES", "NX_ACTIVE"):
        env.pop(k, None)  # exercise only the legacy NX_* single-source fallback
    env.update({
        "NX_ENV_FILE": "/dev/null",
        "NX_HOME": str(tmp_path / "nxhome"),
        # a dummy, unreachable source: if the guard regresses, switch_source()
        # would wire it in (and, in production, probe it for real)
        "NX_BASE_URL": "http://127.0.0.1:9",
        "NX_MODEL": "dummy",
        "NX_API_KEY": "dummy",
        "NX_SKIP_STARTUP_PROBES": "1",
    })
    # `python -c CODE -v` lands "-v" in sys.argv[1:] — exactly the argv the
    # import-time guard in nx/__init__.py short-circuits on.
    r = subprocess.run(
        [sys.executable, "-c",
         "import nx; print(nx._state.ACTIVE is None, nx._state.MODEL is None)",
         "-v"],
        capture_output=True, text=True, env=env, cwd=_ROOT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True True"
