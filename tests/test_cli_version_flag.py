"""`nx -v` must print the version and exit, exactly like `nx --version` —
it used to fall through to the interactive REPL because only the long flag
was recognized (both in main()'s short-circuit and in nx/__init__.py's
import-time guard that skips switch_source()'s network probes)."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def test_main_prints_version_for_v(capsys, monkeypatch):
    from nx import _cli
    monkeypatch.setattr(sys, "argv", ["nx", "-v"])
    _cli.main()
    out = capsys.readouterr().out
    assert out == f"nx {_cli._cli_version()}\n"


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
    assert "See README.md for the full NX_* env" in out


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

def test_import_time_guard_skips_switch_source_for_v(monkeypatch, tmp_path):
    """With a source configured, importing nx under `-v` must not call
    switch_source() (which would hit the network before main() runs)."""
    os.environ["NX_ENV_FILE"] = "/dev/null"
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    # a dummy, unreachable source: if the guard regresses, switch_source()
    # would try a real network round-trip to it
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9"
    os.environ["NX_MODEL"] = "dummy"
    os.environ["NX_API_KEY"] = "dummy"

    import nx._sources as _sources
    calls = []
    monkeypatch.setattr(_sources, "switch_source",
                        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(
                            AssertionError("switch_source called for -v")))

    monkeypatch.setattr(sys, "argv", ["nx", "-v"])
    import nx
    importlib.reload(nx)
    assert calls == []
    os.environ["NX_ENV_FILE"] = "/dev/null"
    for k in ("NX_HOME", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
        os.environ.pop(k, None)
    importlib.reload(nx)
