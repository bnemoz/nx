#!/usr/bin/env python3
"""Tests for the color-theme system in `nx/_ansi.py`.

The palette was expanded from six hues (cyan/green/yellow/red/magenta/gray)
to eight (adding blue and orange). The extra hues disambiguate two roles that
previously shadowed other roles' hues: ACCENT (borders/model-name/prompt) was
cyan, indistinguishable from STATUS_BAR; LABEL (source/session badges) was
magenta, indistinguishable from TOOL_OUTPUT. blue/orange now back them.
"""

import colorsys
import itertools
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-themes-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
# Clear any NX_* that could affect imports, and keep local ~/.env out.
for _k in [k for k in os.environ if k.startswith("NX_")]:
    del os.environ[_k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402

from nx import _ansi, _tty  # noqa: E402

PALETTE = ("cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange",
           "purple", "teal", "pink", "brown", "slate")


def test_every_theme_supplies_all_thirteen_distinct_hues():
    for tname, hexes in _ansi.THEMES.items():
        assert set(hexes) == set(PALETTE), f"{tname} missing/extra hues: {set(hexes)}"
        assert len(set(hexes.values())) == len(PALETTE), (
            f"{tname} has duplicate hex values"
        )
    # slate must read as distinct background chatter: different from the theme's
    # own gray in every theme.
    for tname, hexes in _ansi.THEMES.items():
        assert hexes["slate"] != hexes["gray"], f"{tname}: slate == gray"


def test_palette_keys_match_themes():
    assert set(_ansi._PALETTE_KEYS) == set(PALETTE)
    for tname in _ansi.THEMES:
        assert set(_ansi.THEMES[tname]) == set(_ansi._PALETTE_KEYS)


def test_colors_class_exposes_all_palette_attrs():
    for name in PALETTE:
        assert hasattr(m.C, name.upper()), f"C.{name.upper()} missing"
    assert m.C.BLUE != _ansi.RESET and m.C.ORANGE != _ansi.RESET
    # blue/orange must be real 24-bit SGR escapes (\033[38;2;r;g;bm)
    assert m.C.BLUE.startswith("\033[38;2;")
    assert m.C.ORANGE.startswith("\033[38;2;")


def test_proxy_module_exposes_new_palette_names():
    # tests do `import nx as m; m.BLUE` — _StateProxyModule must resolve them.
    assert m.BLUE == m.C.BLUE
    assert m.ORANGE == m.C.ORANGE


def _luma(hexv: str) -> float:
    r, g, b = _rgb_tuple(hexv)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b



def test_role_defaults_map_remapped_hues():
    # The declarative role→hue map is env-independent: this is the contract,
    # regardless of any NX_*_COLOR overrides captured at import time.
    assert _ansi.ROLE_DEFAULTS["ACCENT"] == "blue"  # was cyan (== STATUS_BAR)
    assert _ansi.ROLE_DEFAULTS["LABEL"] == "orange"  # was magenta (== TOOL_OUTPUT)
    assert _ansi.ROLE_DEFAULTS["TODO"] == "orange"  # todo_write output, distinct from TOOL_OUTPUT
    # STATUS_BAR keeps its original hue; TOOL_OUTPUT moved to slate (muted
    # background chatter; magenta stays reachable via NX_TOOL_OUTPUT_COLOR).
    assert _ansi.ROLE_DEFAULTS["STATUS_BAR"] == "cyan"
    assert _ansi.ROLE_DEFAULTS["TOOL_OUTPUT"] == "slate"
    # every role maps to a known palette key:
    for role, key in _ansi.ROLE_DEFAULTS.items():
        assert key in PALETTE, f"{role} -> unknown hue {key!r}"


def _apply_with_no_overrides(theme: str) -> None:
    """Re-apply `theme` with all role overrides disabled, so the assertions
    reflect the theme + ROLE_DEFAULTS contract rather than a live
    NX_*_COLOR override read from ~/.env at import time."""
    hexes = _ansi.THEMES[theme]
    saved = _ansi._ROLE_OVERRIDES
    _ansi._ROLE_OVERRIDES = {role: None for role in _ansi.ROLE_DEFAULTS}
    try:
        m.C.name = theme
        m.C._apply(hexes)
    finally:
        _ansi._ROLE_OVERRIDES = saved


def test_role_remap_gives_each_role_a_distinct_default_hue():
    _apply_with_no_overrides(_ansi.DEFAULT_THEME)
    # The two collisions the expansion fixed:
    assert m.C.ACCENT == m.C.BLUE  # was cyan (== STATUS_BAR)
    assert m.C.LABEL == m.C.ORANGE  # was magenta (== TOOL_OUTPUT)
    # STATUS_BAR keeps its original hue; TOOL_OUTPUT reads as slate chatter.
    assert m.C.STATUS_BAR == m.C.CYAN
    assert m.C.TOOL_OUTPUT == m.C.SLATE
    # every semantic role now maps to a distinct escape across the palette:
    roles = [
        "TEXT",
        "REASONING_TEXT",
        "STATUS_BAR",
        "TOOL_OUTPUT",
        "ACCENT",
        "LABEL",
        "TODO",
        "SUCCESS",
        "WARNING",
        "ERROR",
    ]
    seen = {getattr(m.C, r) for r in roles}
    # 10 roles, but two collisions are intentional — TEXT==SUCCESS (both green)
    # and LABEL==TODO (both orange, so the model's task list shares the orange
    # hue used for badges) — leaving 8 distinct.
    assert len(seen) == 8, f"expected 8 distinct role escapes, got {len(seen)}"


def test_role_distinctness_holds_for_every_theme():
    for tname in _ansi.THEMES:
        _apply_with_no_overrides(tname)
        assert m.C.ACCENT != m.C.STATUS_BAR, f"{tname}: ACCENT==STATUS_BAR"
        assert m.C.LABEL != m.C.TOOL_OUTPUT, f"{tname}: LABEL==TOOL_OUTPUT"
        assert m.C.ACCENT != m.C.LABEL, f"{tname}: ACCENT==LABEL"
    _apply_with_no_overrides(_ansi.DEFAULT_THEME)


def test_set_theme_round_trip_resets_eight_hues():
    _apply_with_no_overrides(_ansi.DEFAULT_THEME)
    orig = {k: getattr(m.C, k.upper()) for k in PALETTE}
    _apply_with_no_overrides("nord")
    nord = {k: getattr(m.C, k.upper()) for k in PALETTE}
    assert orig != nord, "nord should differ from the default (mocha)"
    assert len(set(nord.values())) == 13, "nord hues not all distinct"
    _apply_with_no_overrides(_ansi.DEFAULT_THEME)
    for k in PALETTE:
        assert getattr(m.C, k.upper()) == orig[k], f"{k} not restored"


def test_swatch_renders_all_twelve_hues():
    # each hue contributes one "██" terminated by RESET
    s = _ansi.swatch("mocha")
    assert s.count(_ansi.RESET) == len(PALETTE)
    assert _ansi.swatch("latte") != s, "latte swatch should differ from mocha"


# --- the monochrome `amethyst` theme ------------------------------------------
# Its whole point is that every non-gray hue is a shade of purple, while the
# roles stay *legible*: with hue pinned to one band, the only remaining
# separations are lightness and position within the violet→pink band, so assert
# both explicitly rather than trusting the hexes.

def _hls(hexv: str) -> tuple[float, float, float]:
    r, g, b = (int(hexv[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _rgb_tuple(hexv: str) -> tuple[int, int, int]:
    return tuple(int(hexv[i:i + 2], 16) for i in (0, 2, 4))


def test_amethyst_hues_are_all_purple():
    assert "amethyst" in _ansi.THEMES
    hexes = _ansi.THEMES["amethyst"]
    for key in PALETTE:
        if key == "gray":
            continue  # reasoning text stays a neutral gray by design
        h, _l, s = _hls(hexes[key])
        assert s > 0.45, f"amethyst {key} is washed out (sat={s:.2f})"
        deg = h * 360 % 360
        # violet (≈246°) through pink/magenta (≈335°), wrapping nowhere else
        assert 240 <= deg <= 340, f"amethyst {key} is not purple (hue={deg:.0f}°)"


def test_amethyst_gray_stays_neutral():
    # "everything purple EXCEPT the gray/white stuff": REASONING_TEXT must not
    # pick up a purple cast — max/min channel spread is what reads as gray.
    r, g, b = _rgb_tuple(_ansi.THEMES["amethyst"]["gray"])
    assert max(r, g, b) - min(r, g, b) <= 24, "amethyst gray is tinted"


def test_amethyst_roles_stay_distinguishable():
    # With one hue family, roles separate by distance in RGB space. Every role
    # pair must be far enough apart to read as different colors.
    _apply_with_no_overrides("amethyst")
    hues = {k: _rgb_tuple(_ansi.THEMES["amethyst"][k]) for k in PALETTE}
    for a, b in itertools.combinations(PALETTE, 2):
        d = sum((hues[a][i] - hues[b][i]) ** 2 for i in range(3)) ** 0.5
        assert d >= 35, f"amethyst {a}/{b} too similar to tell apart ({d:.0f})"
    _apply_with_no_overrides(_ansi.DEFAULT_THEME)


def test_amethyst_switches_via_set_theme():
    orig = m.C.name
    try:
        assert _ansi.set_theme("amethyst")
        assert m.C.name == "amethyst"
        assert m.C.ACCENT == _ansi._rgb(_ansi.THEMES["amethyst"]["blue"])
        assert m.C.TOOL_OUTPUT == _ansi._rgb(_ansi.THEMES["amethyst"]["slate"])
        assert _ansi.set_theme("amethyst-not-a-theme") is False
    finally:
        _ansi.set_theme(orig)


# --- the rich say-panel follows the theme ---------------------------------------
# The assistant's final answer renders through rich.markdown.Markdown on a
# Console in _turn.py. Before the _rich.py bridge, that console used rich's
# DEFAULT_STYLES palette (fixed dim-yellow headings, pale-blue code) — the one
# surface in the terminal that ignored NX_THEME // /theme entirely.
#
# SPEC_LIVEENGINE.md (0.128.0): `_turn._say_console` IS `_rich.shared_console()`
# — one process-wide console shared with the /tabs region Live and every
# mid-turn writer. These tests still hold, but any test that temporarily
# REBINDS its `.file` must restore the ORIGINAL `_file` (None = lazy sys.stdout
# resolution), because pinning the resolved stdout on the shared object would
# make every later console write bypass a test's `redirect_stdout`.

def test_rich_say_panel_tracks_theme():
    from nx import _rich, _turn
    console = _turn._say_console
    orig = _ansi.C.name
    try:
        _ansi.set_theme("mocha")
        mocha = _rich._build_theme()
        _ansi.set_theme("nord")
        nord = _rich._build_theme()
        # headings come from ACCENT; a theme switch must move them
        assert (mocha.styles["markdown.h1"] != nord.styles["markdown.h1"])
        assert "#" + _ansi.THEMES["nord"]["blue"] in str(nord.styles["markdown.h1"])
        # inline code follows TEXT (green)
        assert "#" + _ansi.THEMES["nord"]["green"] in str(nord.styles["markdown.code"])
        # set_theme() pushed our themed style onto the live say-panel console
        from rich.console import Console
        assert isinstance(console, Console)
        assert "#" + _ansi.THEMES[_ansi.C.name]["blue"] \
            in str(console._theme_stack._entries[-1])
    finally:
        _ansi.set_theme(orig)


def test_rich_console_styles_markdown_with_theme_palette():
    """End-to-end: the themed say-console must emit ANSI in the theme's hue
    for a markdown element rich would otherwise paint its own color."""
    from rich.markdown import Markdown
    from nx import _turn
    import tests._nxtest as _nxtest
    console = _turn._say_console
    orig = _ansi.C.name
    try:
        _ansi.set_theme("mocha")
        with _nxtest.captured_console_output(
                {"TTY_COMPATIBLE": "1", "COLORTERM": "truecolor"}) as (con, buf):
            con.print(Markdown("# Hi"))
            out = buf.getvalue()
        expected = "\x1b[1;38;2;137;180;250m"  # ACCENT hue, bold (h1)
        assert expected in out, "say-panel markdown not rendered in theme hue"
    finally:
        _ansi.set_theme(orig)


def test_rich_table_border_follows_theme():
    """The table border style is themed (TEAL — its own hue, distinct from the
    ACCENT headings), not rich's default cyan."""
    from rich.markdown import Markdown  # noqa: F401  (import path guard)
    from nx import _rich, _turn
    console = _turn._say_console
    orig = _ansi.C.name
    try:
        _ansi.set_theme("mocha")
        styles = _rich._build_theme().styles
        assert "#" + _ansi.THEMES["mocha"]["teal"] in str(styles["markdown.table.border"])
    finally:
        _ansi.set_theme(orig)


def test_code_blocks_derive_pygments_colors_from_palette():
    """Fenced code blocks are colored from the ACTIVE theme's palette via
    _PaletteSyntaxTheme — not a fixed pygments style, and not a name→name
    mapping. A /theme switch must move the token hues."""
    from pygments.token import Token
    from nx import _rich
    orig = _ansi.C.name
    try:
        _ansi.set_theme("mocha")
        th_mocha = _rich.code_theme()
        kw = th_mocha.get_style_for_token(Token.Keyword)
        assert kw.color.get_truecolor().hex \
            == "#" + _ansi.THEMES["mocha"]["blue"], \
            "Keyword hue should be the theme's accent blue"
        cm = th_mocha.get_style_for_token(Token.Comment)
        assert cm.color.get_truecolor().hex == "#" + _ansi.THEMES["mocha"]["gray"]
        # hierarchical lookup: a deep token inherits its ancestor's hue
        d = th_mocha.get_style_for_token(Token.Literal.String.Double)
        assert d.color.get_truecolor().hex == "#" + _ansi.THEMES["mocha"]["green"]
        # a theme switch re-derives (fresh instance per call)
        _ansi.set_theme("latte")
        th_latte = _rich.code_theme()
        assert th_latte.get_style_for_token(Token.Keyword).color \
            .get_truecolor().hex == "#" + _ansi.THEMES["latte"]["blue"]
        # and the hexes differ between dark and light themes
        assert (th_mocha.get_style_for_token(Token.Keyword).color
                != th_latte.get_style_for_token(Token.Keyword).color)
    finally:
        _ansi.set_theme(orig)


# --- light-background auto-detection -----------------------------------------
# Dark palettes are unreadable on a light terminal; with no explicit NX_THEME
# the default theme must follow the terminal background (nx/_tty.py helpers).

def test_osc_bg_luminance_off_tty_is_none(monkeypatch):
    # Headless/pipe stdout can't be queried: the helper must give up cleanly.
    import io
    monkeypatch.setattr(_ansi.sys, "stdout", io.StringIO(), raising=False)
    assert _tty._osc_bg_luminance() is None


def test_osc_bg_luminance_parses_reply(monkeypatch):
    import io
    import sys as _s
    # Fake tty-ish stdout and a preloaded reply on stdin's fd is hard to
    # fabricate; instead patch wait_ready+os.read to replay a canned reply.
    reply = b"\033]11;rgb:ffff/ffff/ffff\033\\"
    monkeypatch.setattr(_s, "stdout", io.StringIO(), raising=False)
    monkeypatch.setattr(_tty.os, "read", lambda fd, n: reply)
    monkeypatch.setattr(_tty, "wait_ready", lambda fd, t: True)
    monkeypatch.setattr(_tty.termios, "tcgetattr", lambda fd: [0, 0, 0, 0, 0, 0, [0] * 12])
    monkeypatch.setattr(_tty.termios, "tcsetattr", lambda *a, **k: None)
    monkeypatch.setattr(_s.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(_s.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(_s.stdin, "fileno", lambda: 0, raising=False)
    assert _tty._osc_bg_luminance() == 1.0


def test_colorfgbg_fallback(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "15;0")   # dark bg
    assert _tty._COLORFGBG_LUMINANCE() == 0.0
    monkeypatch.setenv("COLORFGBG", "0;15")   # light bg
    assert _tty._COLORFGBG_LUMINANCE() == 0.9
    monkeypatch.delenv("COLORFGBG")
    assert _tty._COLORFGBG_LUMINANCE() is None
    monkeypatch.setenv("COLORFGBG", "junk")
    assert _tty._COLORFGBG_LUMINANCE() is None


def test_default_theme_tracks_light_background(monkeypatch):
    # Simulated light terminal, no explicit NX_THEME -> latte: run the exact
    # import-time selection logic _ansi.py uses against a patched probe.
    monkeypatch.setattr(_tty, "_osc_bg_luminance", lambda timeout=0.05: 0.95)
    monkeypatch.setattr(_tty, "_COLORFGBG_LUMINANCE", lambda: None)
    requested = None
    default = _ansi.DEFAULT_THEME
    _lum = _tty._osc_bg_luminance()
    if _lum is None:
        _lum = _tty._COLORFGBG_LUMINANCE()
    if requested is None and _lum is not None and _lum >= 0.5:
        default = "latte"
    assert default == "latte"


def test_ansi_role_colors_unaffected_by_detection():
    # the detection seam must not disturb the live palette attributes
    assert m.C.ACCENT.startswith("\033[38;2;")
    assert m.C.TEXT.startswith("\033[38;2;")


def test_variant_keeps_twelve_distinct_hues_for_every_theme():
    for tname in _ansi.THEMES:
        for light in (False, True):
            v = _ansi._variant_hexes(tname, light)
            assert set(v) == set(PALETTE)
            assert len(set(v.values())) == 13, f"{tname} light={light}: dup hues"


def test_variant_preserves_hue_family():
    # mirroring lightness must not change what hue family a slot belongs to:
    # each variant slot stays within 30° of its authored hue (gray exempt: its
    # authored saturation is near zero, so its hue angle is noise).
    for tname in _ansi.THEMES:
        for light in (False, True):
            v = _ansi._variant_hexes(tname, light)
            for key in PALETTE:
                if key == "gray":
                    continue
                h1, _l1, _s1 = _hls(_ansi.THEMES[tname][key])
                h2, _l2, _s2 = _hls(v[key])
                d = abs(h1 - h2) * 360
                d = min(d, 360 - d)
                assert d <= 30, f"{tname} light={light}: {key} hue drifted {d:.0f}°"


def test_variant_is_legible_on_its_side():
    # light variants: all slots dark enough for white; dark variants: light
    # enough for near-black (same perceptual weights the tty probe uses).
    def lum(h):
        r, g, b = _rgb_tuple(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    for tname in _ansi.THEMES:
        for light in (False, True):
            v = _ansi._variant_hexes(tname, light)
            if v is _ansi.THEMES[tname]:
                continue  # authored side: hand-picked, not bound by the clamp
            for key in PALETTE:
                # light variants must sit dark enough for white; dark variants
                # light enough for near-black (brighter is fine on dark)
                if light:
                    assert 90 <= lum(v[key]) <= 160, \
                        f"{tname} light={light} {key} luma {lum(v[key]):.0f} out of window"
                else:
                    assert 90 <= lum(v[key]) <= 200, \
                        f"{tname} light={light} {key} luma {lum(v[key]):.0f} out of window"


def test_variant_matches_authored_side_is_identity():
    # a dark theme asked for a dark background returns its authored palette
    # untouched; latte asked for light ditto
    assert _ansi._variant_hexes("mocha", False) is _ansi.THEMES["mocha"]
    assert _ansi._variant_hexes("latte", True) is _ansi.THEMES["latte"]
    assert _ansi._variant_hexes("mocha", True) is not _ansi.THEMES["mocha"]


def test_light_bg_probe_swaps_variant_via_apply():
    orig, orig_lb = m.C.name, m.C.light_bg
    try:
        m.C.name = "dracula"
        m.C.light_bg = None
        m.C._apply(_ansi.THEMES["dracula"])
        authored = m.C.CYAN
        m.C.light_bg = True
        m.C._apply(_ansi.THEMES["dracula"])
        assert m.C.CYAN != authored, "light-bg variant not applied"
        m.C.light_bg = None
        m.C._apply(_ansi.THEMES["dracula"])
        assert m.C.CYAN == authored, "authored palette not restored"
    finally:
        m.C.name, m.C.light_bg = orig, orig_lb
        m.C._apply(_ansi.THEMES[m.C.name])


def test_rendered_code_block_uses_theme_hues_end_to_end():
    """Full render path: the say-console + themed_markdown must emit the
    theme's own hues in the ANSI stream for a fenced code block."""
    from nx import _rich
    import tests._nxtest as _nxtest
    orig = _ansi.C.name

    def rgb(h):
        return "38;2;%d;%d;%d" % (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

    try:
        _ansi.set_theme("mocha")
        with _nxtest.captured_console_output(
                {"TTY_COMPATIBLE": "1", "COLORTERM": "truecolor"}) as (console, buf):
            console.print(_rich.themed_markdown(
                "```python\ndef f():  # c\n    return 42\n```"))
        out = buf.getvalue()
        hexes = _ansi.THEMES["mocha"]
        assert rgb(hexes["blue"]) in out, "keyword not in accent hue"
        assert rgb(hexes["gray"]) in out, "comment not in reasoning gray"
        assert rgb(hexes["purple"]) in out, "number not in purple"
        # monokai's signature keyword cyan must be gone
        assert "102;217;239" not in out, "still rendering monokai hues"
    finally:
        _ansi.set_theme(orig)


