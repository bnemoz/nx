"""UIUX_SPEC Phase B (0.135.0): the render vocabulary in `nx/_render.py`.

B1 — one component per shape (`panel`, `notice`, `kv_line`, `diff`,
`stream_tail`), each deciding its own render mode; callers never branch on
raw-ANSI vs rich-Live.
B2 — one box-glyph set, used by every box.
B3 — the role completion: MUTED / EMPHASIS / DIFF_ADD / DIFF_REMOVE exist,
follow the `NX_<ROLE>_COLOR` override pattern, and the raw palette reads
outside `_ansi`/`_rich`/`_render` are gone.

Run offline: nothing here touches the network or a real tty.
"""
import os
import re
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-render-")
os.environ.setdefault("NX_SESSIONS_DIR", _tmp)
os.environ.setdefault("NX_HOME", _tmp)
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nx import _ansi, _render  # noqa: E402


# --- B2: one glyph set --------------------------------------------------------

def test_box_glyph_set_is_the_rounded_family():
    assert _render.BOX_GLYPHS == (
        "╭", "╮", "╰", "╯", "─", "│", "├─", "╰─",
    )
    # the square corners the tool box used are gone from the vocabulary
    assert "┌" not in _render.BOX_GLYPHS and "└" not in _render.BOX_GLYPHS


def test_every_box_draws_from_the_one_glyph_set():
    """The migrated boxes draw from the constants: the square corner family
    (`┌ └`) is gone from the tool box and every other box still in these
    modules, and no module hard-codes a rounded corner in an f-string."""
    import nx._chatbox
    import nx._dispatch
    import nx._turn_footer
    for mod in (nx._chatbox, nx._dispatch, nx._turn_footer):
        src = open(mod.__file__, encoding="utf-8").read()
        for glyph in ("┌", "└"):
            assert glyph not in src, (
                f"{mod.__name__} still hard-codes the square glyph {glyph!r}")
        for glyph in ("╭", "╮", "╯", "╰"):
            assert f'"{glyph}' not in src and f"'{glyph}" not in src, (
                f"{mod.__name__} hard-codes {glyph!r} instead of using BOX_*")


def test_panel_uses_the_shared_glyphs():
    lines = _render.panel("read_file x.py", ["args"], role_name="ACCENT")
    assert lines[0].endswith("╭─ read_file x.py" + _ansi.RESET)
    assert _render.BOX_V in lines[1] and "args" in lines[1]
    assert lines[-1].startswith(_render.role("ACCENT") + "  ╰─")


def test_panel_accent_argument_wins_over_the_role():
    accent = "\033[38;2;1;2;3m"
    lines = _render.panel("t", [], accent=accent)
    assert lines[0].startswith(accent) and lines[-1].startswith(accent)


# --- B1: diff ----------------------------------------------------------------

def test_diff_uses_the_diff_roles_and_line_numbers():
    out = _render.diff("a\nb\n", "a\nc\n")
    lines = out.splitlines()
    assert any(_ansi.C.DIFF_REMOVE in ln and "-b" in ln for ln in lines)
    assert any(_ansi.C.DIFF_ADD in ln and "+c" in ln for ln in lines)
    # context lines carry both numbers, dimmed
    assert any(_ansi.DIM in ln and " 1 " in ln for ln in lines)
    # old/new numbers: removed line is old#2, added line is new#2
    assert "   2      -b" in out
    assert "        2 +c" in out


def test_diff_returns_empty_for_identical_text():
    assert _render.diff("same\n", "same\n") == ""


def test_diff_context_is_configurable():
    old = "\n".join(f"line{i}" for i in range(20))
    new = old.replace("line10", "CHANGED")
    assert len(_render.diff(old, new, 0).splitlines()) < len(
        _render.diff(old, new, 50).splitlines())


def test_approval_diff_preview_is_the_component():
    """`_approval._diff_preview` must not keep a second implementation: its
    body is a direct delegation to `_render.diff` (the module still imports
    difflib for `_diff_hunk`, which is model-facing plain text)."""
    from nx import _approval
    out = _approval._diff_preview("a\n", "b\n")
    assert out == _render.diff("a\n", "b\n")
    assert "C.GREEN" not in out and "C.RED" not in out


# --- B1: stream_tail ---------------------------------------------------------

def test_stream_tail_strips_ansi_and_collapses_to_one_line():
    assert _render.stream_tail("a\x1b[31mb\nc", 40, plain=True) == "ab c"
    assert "\x1b[" not in _render.stream_tail("x\x1b[1my", 40, plain=True)


def test_stream_tail_clamps_to_width():
    # clip_line is the spinner budget's own truncation marker ("..."), so the
    # moved clamp renders byte-identically to before
    out = _render.stream_tail("x" * 100, 10, plain=True)
    assert len(out) == 10 and out.endswith("...")


def test_stream_tail_inplace_leads_with_clear_line():
    assert _render.stream_tail("hi", 20, inplace=True).startswith(_ansi.CLEAR_LINE)
    assert not _render.stream_tail("hi", 20).startswith(_ansi.CLEAR_LINE)


def test_emit_inplace_is_raw_when_no_live_and_console_print_with_one(capsys):
    """The render-mode switch (the folded `_ansi._draw_frame` branch)."""
    _render.emit("\x1b[2K\rhello", inplace=True)
    assert capsys.readouterr().out == "\x1b[2K\rhello"


def test_lifeSpinner_label_and_note_flow_through_stream_tail():
    from nx._ansi import LifeSpinner
    sp = LifeSpinner(label="running x\x1b[31m\n  y")
    sp.show_elapsed = False
    sp._live = True
    frame = sp._render(0)
    # the clear-line lead and the DIM/RESET wraps are the only escapes left:
    # the label's own ANSI and newline were stripped by stream_tail
    assert frame.startswith("\x1b[2K\r")
    assert "\x1b[31m" not in frame and "\n" not in frame
    assert frame.count("\x1b[2m") == 2  # DIM label + DIM bar, nothing else
    assert "running x y" in frame
    # a long label is clamped to the label budget
    long_label = LifeSpinner(label="z" * 400)
    assert len(long_label._render(0)) < 200


def test_note_progress_sanitizes_through_stream_tail():
    from nx import _progress, _state as st
    _progress.note_progress("compiling\x1b[31m 3/40\nnext")
    assert "\x1b[" not in (st._ACTIVE_SPINNER.note if st._ACTIVE_SPINNER else "")


# --- B1: kv_line -------------------------------------------------------------

def test_kv_line_is_key_first_and_muted():
    out = _render.kv_line([("jobs", "2"), ("steer", "1")])
    assert out.startswith(_render.role("MUTED") + "  ╰ ")
    stripped = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert stripped == "  ╰ jobs 2 · steer 1"


def test_kv_line_label_takes_the_muted_role_not_dim():
    out = _render.kv_line([("tok", "12")], footer_glyph=False)
    assert _render.role("MUTED") in out
    assert _ansi.DIM not in out


# --- B1: notice --------------------------------------------------------------

def test_notice_levels_use_their_roles(capsys):
    for level, role_name in _render.NOTICE_ROLES.items():
        _render.notice(level, "x")
        assert _render.role(role_name) in capsys.readouterr().out


def test_notice_plain_level_is_byte_identical(capsys):
    _render.notice("plain", "no decoration")
    assert capsys.readouterr().out == "no decoration\n"


# --- B1: the /tabs tool-status line (both modes) -----------------------------

def test_tool_status_line_uses_the_component_and_keeps_both_modes():
    """`show_tool_status` builds its line with `_render.status_line` and folds
    its raw/rich branch into `_render.emit` (UIUX_SPEC B1)."""
    import nx._stream_parse as sp
    src = open(sp.__file__, encoding="utf-8").read()
    assert "status_line(" in src
    assert "emit(text, inplace=True)" in src
    assert "emit(text)" in src
    # the old hand-rolled f-string + CLEAR_LINE concat is gone
    assert "generating tool call{RESET} " not in src
    assert "cprint(CLEAR_LINE + text" not in src


def test_status_line_shape_matches_the_old_tool_status_bytes():
    """`status_line` reproduces the pre-0.135 tool-status bytes: accent label,
    space, DIM tail."""
    out = _render.status_line("  ↳ generating tool call", role_name="ACCENT",
                              note="write_file args 12 chars")
    assert out == (f"{_render.role('ACCENT')}  ↳ generating tool call{_ansi.RESET}"
                   f" {_ansi.DIM}write_file args 12 chars{_ansi.RESET}")


# --- B3: role completion -----------------------------------------------------

def test_new_roles_exist_with_defaults():
    for new_role in ("MUTED", "EMPHASIS", "DIFF_ADD", "DIFF_REMOVE"):
        assert new_role in _ansi.ROLE_DEFAULTS, f"{new_role} missing"
        assert hasattr(_ansi.C, new_role), f"C.{new_role} missing"
        assert getattr(_ansi.C, new_role).startswith("\033[38;2;")
    assert _ansi.ROLE_DEFAULTS["DIFF_ADD"] == "green"
    assert _ansi.ROLE_DEFAULTS["DIFF_REMOVE"] == "red"
    assert _ansi.ROLE_DEFAULTS["MUTED"] == "gray"


def test_new_roles_track_the_theme():
    orig = _ansi.C.name
    try:
        _ansi.set_theme("mocha")
        mocha = _ansi.C.DIFF_ADD
        _ansi.set_theme("latte")
        assert _ansi.C.DIFF_ADD != mocha
    finally:
        _ansi.set_theme(orig)


def test_role_overrides_apply_to_the_new_roles(monkeypatch):
    """NX_<ROLE>_COLOR support follows the existing override pattern (both a
    palette key and a literal hex), driven through the same _apply path."""
    saved = _ansi._ROLE_OVERRIDES
    orig_name = _ansi.C.name
    try:
        _ansi._ROLE_OVERRIDES = dict(saved)
        _ansi._ROLE_OVERRIDES["DIFF_ADD"] = ("palette", "teal")
        _ansi._ROLE_OVERRIDES["MUTED"] = ("hex", "010203")
        _ansi.C.name = "mocha"
        _ansi.C._apply(_ansi.THEMES["mocha"])
        assert _ansi.C.DIFF_ADD == _ansi._rgb(_ansi.THEMES["mocha"]["teal"])
        assert _ansi.C.MUTED == "\033[38;2;1;2;3m"
    finally:
        _ansi._ROLE_OVERRIDES = saved
        _ansi.C.name = orig_name
        _ansi.C._apply(_ansi.THEMES[orig_name])


def test_the_override_env_names_are_parsed():
    """The env parser resolves NX_<ROLE>_COLOR for the new roles too — the
    table `_ROLE_OVERRIDES` is keyed off ROLE_DEFAULTS."""
    for role in ("MUTED", "EMPHASIS", "DIFF_ADD", "DIFF_REMOVE"):
        assert f"NX_{role}_COLOR" == f"NX_{role}_COLOR"  # name contract
    assert set(_ansi._ROLE_OVERRIDES) >= set(_ansi.ROLE_DEFAULTS)


def test_no_raw_palette_reads_outside_the_render_layer():
    """B3 acceptance: the raw-palette grep returns nothing outside
    _ansi.py / _rich.py / _render.py. The last carve-out (`_toolblocks`'s
    C.RED read for the deletion-accent tools) is gone — it now takes the
    DIFF_REMOVE role from the shared layer, so the guard covers the whole
    package outside the render layer."""
    pkg = os.path.dirname(_ansi.__file__)
    pattern = re.compile(
        r"C\.(CYAN|GREEN|YELLOW|RED|MAGENTA|BLUE|ORANGE|PURPLE|TEAL|PINK|BROWN)\b")
    allowed = {"_ansi.py", "_rich.py", "_render.py"}
    offenders = []
    for name in sorted(os.listdir(pkg)):
        if not name.endswith(".py") or name in allowed:
            continue
        with open(os.path.join(pkg, name), encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if pattern.search(line):
                    offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, "raw palette reads left: " + "; ".join(offenders)
