#!/usr/bin/env python3
"""Tests for the color-theme system in `nx/_ansi.py`.

The palette was expanded from six hues (cyan/green/yellow/red/magenta/gray)
to eight (adding blue and orange). The extra hues disambiguate two roles that
previously shadowed other roles' hues: ACCENT (borders/model-name/prompt) was
cyan, indistinguishable from STATUS_BAR; LABEL (source/session badges) was
magenta, indistinguishable from TOOL_OUTPUT. blue/orange now back them.
"""

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

from nx import _ansi  # noqa: E402

PALETTE = ("cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange")


def test_every_theme_supplies_all_eight_distinct_hues():
    for tname, hexes in _ansi.THEMES.items():
        assert set(hexes) == set(PALETTE), f"{tname} missing/extra hues: {set(hexes)}"
        assert len(set(hexes.values())) == len(PALETTE), (
            f"{tname} has duplicate hex values"
        )


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


def test_role_defaults_map_remapped_hues():
    # The declarative role→hue map is env-independent: this is the contract,
    # regardless of any NX_*_COLOR overrides captured at import time.
    assert _ansi.ROLE_DEFAULTS["ACCENT"] == "blue"  # was cyan (== STATUS_BAR)
    assert _ansi.ROLE_DEFAULTS["LABEL"] == "orange"  # was magenta (== TOOL_OUTPUT)
    # STATUS_BAR and TOOL_OUTPUT keep their original hues:
    assert _ansi.ROLE_DEFAULTS["STATUS_BAR"] == "cyan"
    assert _ansi.ROLE_DEFAULTS["TOOL_OUTPUT"] == "magenta"
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
    # STATUS_BAR and TOOL_OUTPUT keep their original hues:
    assert m.C.STATUS_BAR == m.C.CYAN
    assert m.C.TOOL_OUTPUT == m.C.MAGENTA
    # every semantic role now maps to a distinct escape across the palette:
    roles = [
        "TEXT",
        "REASONING_TEXT",
        "STATUS_BAR",
        "TOOL_OUTPUT",
        "ACCENT",
        "LABEL",
        "SUCCESS",
        "WARNING",
        "ERROR",
    ]
    seen = {getattr(m.C, r) for r in roles}
    # 9 roles, but TEXT==SUCCESS (both green) is intentional → 8 distinct.
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
    assert len(set(nord.values())) == 8, "nord hues not all distinct"
    _apply_with_no_overrides(_ansi.DEFAULT_THEME)
    for k in PALETTE:
        assert getattr(m.C, k.upper()) == orig[k], f"{k} not restored"


def test_swatch_renders_all_eight_hues():
    # each hue contributes one "██" terminated by RESET
    s = _ansi.swatch("mocha")
    assert s.count(_ansi.RESET) == len(PALETTE)
    assert _ansi.swatch("latte") != s, "latte swatch should differ from mocha"
