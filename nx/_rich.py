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
from rich.style import Style
from rich.syntax import SyntaxTheme

from . import _ansi

# Pygments token class -> nx palette hue. A fenced code block is rendered by
# Pygments, which colors tokens from its own fixed styles — no console style
# can reach them. Instead of mapping nx theme names onto shipped Pygments
# styles (feel-matched but hue-alien), we GENERATE the style from the active
# theme: every entry here resolves to a palette hue (via the role map below,
# so NX_*_COLOR overrides apply) at build time, and a live /theme switch
# re-derives it. Keys are pygments token classes; unlisted tokens inherit from
# their nearest listed ancestor (pygments resolves styles hierarchically).
# Names here are role keys in _ansi.ROLE_DEFAULTS, resolved through
# _syntax_role_hues() so an NX_<ROLE>_COLOR override tracks too.
_TOKEN_HUES: dict[str, str] = {
    "Keyword": "ACCENT",          # def, return, import, if — the accent family
    "Keyword.Type": "TEXT",       # int, str, bool — the type itself is data
    "Name.Builtin": "TEXT",       # print, len — same family as ordinary text
    "Name.Function": "ACCENT",    # the thing being defined — accent like headings
    "Name.Class": "ACCENT",
    "Name.Decorator": "PINK",
    "Name.Tag": "ACCENT",         # HTML/XML tags
    "Name.Attribute": "PINK",
    "Name.Exception": "ERROR",
    "String": "TEXT",             # strings are answer text — stay legible, calm
    "Literal.String": "TEXT",
    "String.Doc": "BROWN italic",         # docstrings: warm, muted, their own slot
    "Literal.String.Doc": "BROWN italic",
    "Number": "PURPLE",
    "Literal.Number": "PURPLE",
    "Comment": "REASONING_TEXT italic",   # comments read as reasoning: muted gray
    "Operator": "TEXT",
    "Punctuation": "REASONING_TEXT",
    "Generic.Deleted": "ERROR",   # diff +/- — match the approval-prompt hues
    "Generic.Inserted": "SUCCESS",
    "Generic.Heading": "ACCENT bold",
    "Generic.Subheading": "ACCENT",
    "Error": "ERROR bold",
}

# rich markdown style name -> nx palette hue. Only the styles whose defaults
# carry a *color* are remapped; structure-only styles (bold/italic/strike on
# strong, em, s) are left at rich's defaults so emphasis stays emphasis in
# every theme. Heading levels all use ACCENT (borders/model name share it, so
# headings, borders, and prompt read as one accent color); code uses the
# palette's green — the same hue as TEXT, since inline code is answer text.
# Bullets/numbers get PURPLE (frees orange = badges only); link URLs BROWN
# (frees gray = reasoning only); table borders TEAL (the one surface teal owns).
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
    "markdown.link_url": "{BROWN_HEX}",
    "markdown.item.bullet": "{PURPLE_HEX}",
    "markdown.item.number": "{PURPLE_HEX}",
    "markdown.block_quote": "{GRAY_HEX} italic",
    "markdown.table.header": "bold",
    "markdown.table.border": "{TEAL_HEX}",
}


def _syntax_role_hues() -> dict[str, str]:
    """Name -> #hex for every name _TOKEN_HUES may reference: the semantic
    roles (override-aware, same resolution _build_theme() uses) PLUS the raw
    palette keys (purple/teal/pink/brown), so code tokens can target the
    widened palette directly."""
    hexes = _ansi.THEMES[_ansi.C.name]
    named = {key.upper(): hexes[key] for key in _ansi._PALETTE_KEYS}
    named.update({role: hexes[key] for role, key in _ansi.ROLE_DEFAULTS.items()})
    for role, override in _ansi._ROLE_OVERRIDES.items():
        if override is None:
            continue
        named[role] = (hexes[override[1]] if override[0] == "palette"
                       else override[1])
    return named


class _PaletteSyntaxTheme(SyntaxTheme):
    """rich SyntaxTheme resolving pygments token colors from the ACTIVE nx
    theme, via _TOKEN_HUES. Rebuilt per Markdown construction, so a /theme
    switch is picked up on the next render without any registration."""

    def __init__(self) -> None:
        self._cache: dict[tuple, Style] = {}

    def get_style_for_token(self, token_type) -> Style:
        key = tuple(token_type)
        style = self._cache.get(key)
        if style is not None:
            return style
        # walk from the most specific token class up to Token, so e.g.
        # Token.Literal.String.Double inherits String's hue
        roles = _syntax_role_hues()
        parts = key
        while parts:
            hue = _TOKEN_HUES.get(".".join(parts))
            if hue is not None:
                break
            parts = parts[:-1]
        else:
            hue = "TEXT"  # plain identifiers and unclassified tokens
        text = hue.split()
        color = roles[text[0]]
        style = Style(color="#" + color,
                      italic="italic" in text[1:],
                      bold="bold" in text[1:])
        self._cache[key] = style
        return style

    def get_background_style(self) -> Style:
        return Style()  # transparent — the code block sits on the panel, no slab


def code_theme() -> _PaletteSyntaxTheme:
    """The syntax theme for the active nx theme (fresh instance per call, so
    a /theme switch is always reflected)."""
    return _PaletteSyntaxTheme()


def themed_markdown(markup: str, **kwargs) -> "object":
    """A rich.markdown.Markdown whose code-block (and inline-code) pygments
    theme follows nx's active theme, on top of the console's themed styles.
    Built here rather than at each call site so the theme mapping has one
    home and a live `/theme` switch is picked up per render."""
    from rich.markdown import Markdown
    kwargs.setdefault("code_theme", code_theme())
    return Markdown(markup, **kwargs)


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
            PURPLE_HEX="#" + hexes["purple"],
            TEAL_HEX="#" + hexes["teal"],
            BROWN_HEX="#" + hexes["brown"],
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
