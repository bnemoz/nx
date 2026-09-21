"""Bridge between nx's ANSI theme (`_ansi.py`) and the one place a rich
Console renders user-facing output: the assistant's live markdown panel.

Everything else nx prints goes through raw ANSI from `C` (semantic roles on
the shared `_Colors` instance), so it already tracks `/theme` switches. The
say-panel's `rich.markdown.Markdown` render, however, resolves its own
`markdown.*` styles from rich's DEFAULT_STYLES — a fixed palette (dim yellow
headers, pale blue code) that ignored the active theme entirely. This module
maps nx's eight theme hues onto those style names and rebuilds the console's
theme on every `set_theme()`, so the markdown panel and the rest of the
terminal speak the same theme.

Kept in its own module rather than `_ansi.py` so the core ANSI layer stays
rich-free (it is imported before anything may need to render markdown) and
so the markdown style mapping has one obvious home.
"""
from __future__ import annotations

from rich import theme as _rich_theme
from rich.console import Console

from . import _ansi

# rich markdown style name -> nx palette hue. Only the styles whose defaults
# carry a *color* are remapped; structure-only styles (bold/italic/strike on
# strong, em, s) are left at rich's defaults so emphasis stays emphasis in
# every theme. Heading levels all use ACCENT (borders/model name share it, so
# headings, borders, and prompt read as one accent color); code uses the
# palette's green — the same hue as TEXT, since inline code is answer text.
_MD_STYLE_MAP: dict[str, str] = {
    "markdown.h1": "bold {ACCENT_HEX}",
    "markdown.h2": "bold {ACCENT_HEX}",
    "markdown.h3": "bold {ACCENT_HEX}",
    "markdown.h4": "bold {ACCENT_HEX}",
    "markdown.h5": "bold {ACCENT_HEX}",
    "markdown.h6": "bold {ACCENT_HEX}",
    "markdown.h1.border": "{ACCENT_HEX}",
    "markdown.h2.border": "{ACCENT_HEX}",
    "markdown.h3.border": "{ACCENT_HEX}",
    "markdown.hr": "{GRAY_HEX}",
    "markdown.code": "{GREEN_HEX}",
    "markdown.code_block": "{GREEN_HEX}",
    "markdown.link": "underline {ACCENT_HEX}",
    "markdown.link_url": "{GRAY_HEX}",
    "markdown.item.bullet": "{ORANGE_HEX}",
    "markdown.item.number": "{ORANGE_HEX}",
    "markdown.block_quote": "{GRAY_HEX} italic",
    "markdown.table.header": "bold",
}


def _build_theme() -> _rich_theme.Theme:
    """A rich Theme reflecting the *current* state of `C` (palette + role
    overrides), read at call time so a `set_theme()` is picked up."""
    hexes = _ansi.THEMES[_ansi.C.name]
    roles = {role: hexes[key] for role, key in _ansi.ROLE_DEFAULTS.items()}
    # role overrides win, same precedence _Colors._apply uses
    for role, override in _ansi._ROLE_OVERRIDES.items():
        if override is None:
            continue
        roles[role] = (hexes[override[1]] if override[0] == "palette"
                       else override[1])
    styles = {}
    for style_name, pattern in _MD_STYLE_MAP.items():
        styles[style_name] = pattern.format(
            ACCENT_HEX="#" + roles["ACCENT"],
            GRAY_HEX="#" + roles.get("REASONING_TEXT", hexes["gray"]),
            GREEN_HEX="#" + roles["TEXT"],
            ORANGE_HEX="#" + roles["LABEL"],
        )
    return _rich_theme.Theme(styles)


def themed_console(**kwargs) -> Console:
    """A rich Console whose markdown styles follow nx's active theme."""
    return Console(theme=_build_theme(), **kwargs)


def retheme_console(console: Console) -> None:
    """Restyle an existing Console in place to the current theme, so a live
    `/theme` switch reaches the say-panel without constructing a new console
    (which would drop its captured file/width state)."""
    console.push_theme(_build_theme())
