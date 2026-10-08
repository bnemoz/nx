"""ANSI color/cursor constants and the terminal waiting animation (LifeSpinner)."""
from __future__ import annotations

import itertools
import os
import re
import sys
import time
import threading
from typing import Literal

from ._progress import (LABEL_CELLS, MAX_NOTE_SHOWN, MAX_TAIL, MIN_TAIL,
                        TAIL_BUDGET, clip_line, fmt_secs)
from ._tty import raw_mode  # noqa: F401  re-exported for back-compat (callers do `from ._ansi import raw_mode`)

def _rgb(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CLEAR_LINE = "\033[2K\r"   # erase entire line, return cursor to col 0

# Strips CSI (\033[...<final byte>), OSC (\033]...BEL or \033\\), and other C0
# control chars (except \t/\n) before untrusted content (tool output, resumed
# transcripts, streamed model text) hits the real terminal. This is a
# display-only boundary — the model-facing copy of the same text is left
# untouched (see _sanitize_tool_result's separate, size-focused contract).
_CSI_RE = re.compile(r"\033\[[0-?]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"\033\](?:[^\033\a]|\033(?!\\))*(?:\a|\033\\)?")
_OTHER_ESC_RE = re.compile(r"\033[@-Z\\-_]")
_C0_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ansi_for_display(text: str) -> str:
    """Strip terminal control sequences from `text` before it's printed.

    Untrusted bytes (file/command output, resumed session transcripts, the
    model's own streamed text) can otherwise smuggle an OSC-52 clipboard
    write, a cursor-hide, or other terminal escape into the real terminal.
    SGR color codes (\\033[...m) written by nx itself are applied *after*
    this strip, so nx's own coloring is unaffected."""
    text = _OSC_RE.sub("", text)
    text = _CSI_RE.sub("", text)
    text = _OTHER_ESC_RE.sub("", text)
    return _C0_RE.sub("", text)

# Twelve hex values per theme: cyan/green/yellow/red/magenta + gray, blue,
# orange (the original eight), plus purple, teal, pink, and brown. Every slot
# is expected to be *used* somewhere: the ten semantic roles draw from eight of
# them (ROLE_DEFAULTS below), and _rich.py maps markdown styles and code-block
# syntax tokens onto the rest (purple bullets/numbers, pink decorators/links,
# brown docstrings/link URLs, teal table borders). "gray" is reasoning text,
# where plain SGR-2 dim or ANSI bright-black (90) is unreadable on many
# terminals, so it needs its own legible-but-muted hue per theme rather than a
# fixed gray. A new theme must be designed as a whole: all twelve slots
# distinct in RGB space, and the amethyst monochrome rule (below) is the
# exception that proves it.
THEMES: dict[str, dict[str, str]] = {
    "mocha": {   # Catppuccin Mocha (default) — dark
        "cyan": "89dceb", "green": "a6e3a1", "yellow": "f9e2af",
        "red": "f38ba8", "magenta": "cba6f7", "gray": "7f849c",
        "blue": "89b4fa", "orange": "fab387",
        "purple": "b4befe", "teal": "94e2d5", "pink": "f5c2e7",
        "brown": "e0b089",
    },
    "latte": {   # Catppuccin Latte — light
        "cyan": "04a5e5", "green": "40a02b", "yellow": "df8e1d",
        "red": "d20f39", "magenta": "8839ef", "gray": "6c6f85",
        "blue": "1e66f5", "orange": "fe640b",
        "purple": "7287fd", "teal": "179299", "pink": "ea76cb",
        "brown": "9a6b4f",
    },
    "dracula": {
        "cyan": "8be9fd", "green": "50fa7b", "yellow": "f1fa8c",
        "red": "ff5555", "magenta": "bd93f9", "gray": "6272a4",
        "blue": "7aa6da", "orange": "ffb86c",
        "purple": "aa87f0", "teal": "5af0c8", "pink": "ff79c6",
        "brown": "c9a56e",
    },
    "nord": {
        "cyan": "88c0d0", "green": "a3be8c", "yellow": "ebcb8b",
        "red": "bf616a", "magenta": "b48ead", "gray": "4c566a",
        "blue": "81a1c1", "orange": "d08770",
        "purple": "9d7fc9", "teal": "8fbcbb", "pink": "d8a1c0",
        "brown": "a97c5e",
    },
    "gruvbox": {   # Gruvbox dark
        "cyan": "8ec07c", "green": "b8bb26", "yellow": "fabd2f",
        "red": "fb4934", "magenta": "d3869b", "gray": "928374",
        "blue": "83a598", "orange": "fe8019",
        "purple": "b16286", "teal": "689d6a", "pink": "e78a4e",
        "brown": "af3a03",
    },
    "vaporwave": {   # neon pastel — fluorescent but soft
        "cyan": "5ffbf1", "green": "8aff80", "yellow": "fdfd96",
        "red": "ff6ec7", "magenta": "d183ff", "gray": "9a86b3",
        "blue": "7aa2f7", "orange": "ff9e64",
        "purple": "a44dff", "teal": "3dfcd2", "pink": "ff9ecb",
        "brown": "d9a05b",
    },
    "frappe": {   # Catppuccin Frappé — medium-dark
        "cyan": "81c8be", "green": "a6d189", "yellow": "e5c890",
        "red": "e78284", "magenta": "ca9ee6", "gray": "838ba7",
        "blue": "8caaee", "orange": "ef9f76",
        "purple": "babbf1", "teal": "99d1db", "pink": "f4b8e4",
        "brown": "d699b6",
    },
    "opencode": {   # OpenCode default theme (opencode.ai) — dark, warm orange
        # primary #fab283 as orange; accent #9d7cd8 as magenta; secondary
        # #5c9cf5 as blue; warning #f5a742 as yellow; error #e06c75,
        # success #7fd88f, info #56b6c2; text #eeeeee, muted #808080.
        "cyan": "56b6c2", "green": "7fd88f", "yellow": "f5a742",
        "red": "e06c75", "magenta": "9d7cd8", "gray": "808080",
        "blue": "5c9cf5", "orange": "fab283",
        "purple": "ffc09f", "teal": "94e2d5", "pink": "f5c2e7",
        "brown": "e0af79",
    },
    "amethyst": {   # monochrome purple — every hue is a violet/orchid/pink
        # shade, so roles stay distinguishable by lightness and by position in
        # the violet→pink band rather than by hue. "gray" stays a true neutral
        # gray: reasoning text is the one thing that shouldn't go purple.
        "cyan": "b581f8", "green": "dfccff", "yellow": "b433ff",
        "red": "f65aa3", "magenta": "ec8fff", "gray": "9e9eaa",
        "blue": "6655fc", "orange": "fa47cd",
        "purple": "8a5cf6", "teal": "c4a8ff", "pink": "ff8ad8",
        "brown": "b092d9",
    },
}
DEFAULT_THEME = "opencode"
_PALETTE_KEYS: tuple[str, ...] = (
    "cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange",
    "purple", "teal", "pink", "brown",
)

# --- light/dark variants ------------------------------------------------------
# Every theme gains an automatic opposite-background variant: a dark theme
# (mocha, dracula, ...) gets a light variant for pale terminals, and the one
# authored light theme (latte) gets a dark one. The variant is DERIVED from
# the authored palette, not hand-tuned: mirror each hue's lightness into the
# band that's legible on the opposite background while keeping its hue family
# (so "mocha on white" still reads as mocha), nudge saturation up slightly on
# light backgrounds (white compresses perceived saturation), then push any
# near-duplicate pair apart in lightness until the twelve slots are as
# distinguishable as an authored palette's. A user who prefers a hand-tuned
# look can still pick the authored side explicitly via /theme.

def _variant_derive(hexes: dict[str, str], *, to_light: bool) -> dict[str, str]:
    """Derive the opposite-background variant of an authored palette."""
    import colorsys
    hls: dict[str, list[float]] = {}
    for k, h in hexes.items():
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        hue, lum_, sat = colorsys.rgb_to_hls(r, g, b)
        if to_light:
            # mirror lightness into the 0.22-0.52 band (legible on white);
            # the 0.92 damp keeps extreme highlights from over-darkening
            nl = min(0.52, max(0.22, 1.0 - lum_ * 0.92))
            ns = min(1.0, sat * 1.12)   # white compresses perceived saturation
        else:
            nl = min(0.82, max(0.62, 1.0 - lum_ * 0.95))  # legible on near-black
            ns = sat
        hls[k] = [hue, nl, ns]

    def rgb(k: str) -> str:
        r, g, b = colorsys.hls_to_rgb(*hls[k])
        return "%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))

    # Legibility clamp on PERCEIVED luma, not HLS lightness: HLS lightness
    # doesn't control luma uniformly across hues (a saturated yellow-green at
    # HLS 0.5 is far brighter than a blue at the same lightness). Walk each
    # slot into its side's luma window directly.
    def luma_of(k: str) -> float:
        h = rgb(k)
        return sum(w * int(h[i:i + 2], 16) for w, i in
                   ((0.2126, 0), (0.7152, 2), (0.0722, 4)))
    for k in hls:
        if to_light:
            # walk into the luma window from BOTH directions: mirroring a
            # very light authored hue lands too dark (mocha yellow), a very
            # saturated one too bright (gruvbox green)
            while luma_of(k) < 100 and hls[k][1] < 0.90:
                hls[k][1] += 0.02
            while luma_of(k) > 160 and hls[k][1] > 0.10:
                hls[k][1] -= 0.02
        else:
            while luma_of(k) < 100 and hls[k][1] < 0.95:
                hls[k][1] += 0.02
            while luma_of(k) > 185 and hls[k][1] > 0.10:
                hls[k][1] -= 0.02
    # The lightness walk above can stall below the luma floor for hues whose
    # luma is dominated by a channel their hue pins low (saturated violet on
    # the light side, gruvbox's red-dominant brown on the dark side);
    # desaturating toward gray covers the full luma range as a fixup.
    for k in hls:
        while luma_of(k) < 100 and hls[k][2] > 0.05:
            hls[k][2] -= 0.05
        while luma_of(k) > 160 and hls[k][2] > 0.05 and to_light:
            hls[k][2] -= 0.05

    def dist(a: str, b: str) -> float:
        return sum((int(rgb(a)[i:i + 2], 16) - int(rgb(b)[i:i + 2], 16)) ** 2
                   for i in (0, 2, 4)) ** 0.5

    # Deterministic separability pass: repeatedly take the closest hue pair
    # and split them in lightness until the twelve slots are distinct enough
    # (the authored amethyst palette's own min pair distance is the bar).
    for _ in range(20):
        pair = min(itertools.combinations(hls, 2), key=lambda p: dist(*p))
        if dist(*pair) >= 30:
            break
        a, b = pair
        # split the pair in lightness, clamped to the side's legibility band
        lo, hi = (0.22, 0.52) if to_light else (0.62, 0.82)
        if hls[a][1] <= hls[b][1]:
            hls[a][1] = max(lo, hls[a][1] - 0.045)
            hls[b][1] = min(hi, hls[b][1] + 0.045)
        else:
            hls[a][1] = min(hi, hls[a][1] + 0.045)
            hls[b][1] = max(lo, hls[b][1] - 0.045)
        # re-clamp luma: the band clamp alone can still leave a slot outside
        # its luma window. Prefer walking lightness (usually room on both
        # sides of the pair), then desaturate as the last resort.
        for k in (a, b):
            other = b if k == a else a
            if luma_of(k) < 100 and hls[k][1] < 0.90 and hls[k][1] <= hls[other][1] + 0.02:
                while luma_of(k) < 100 and hls[k][1] < 0.90:
                    hls[k][1] += 0.02
            while luma_of(k) < 100 and hls[k][2] > 0.05:
                hls[k][2] -= 0.05
            while luma_of(k) > (160 if to_light else 185) and hls[k][2] > 0.05:
                hls[k][2] -= 0.05
    return {k: rgb(k) for k in hexes}


# AUTHORED_THEME_LIGHTNESS: True for themes designed for a light terminal.
# The natural variant of theme T for a given background is T itself when the
# background matches T's authored side, else the derived opposite.
AUTHORED_THEME_LIGHTNESS: dict[str, bool] = {t: False for t in THEMES}
AUTHORED_THEME_LIGHTNESS["latte"] = True


def variant_for(name: str, *, light_bg: bool) -> dict[str, str]:
    """The palette hexes for `name` rendered for `light_bg` (True = white-ish
    terminal background): the authored palette itself when that already matches
    the background, else the derived variant."""
    hexes = THEMES[name]
    if AUTHORED_THEME_LIGHTNESS[name] == light_bg:
        return hexes
    return _variant_derive(hexes, to_light=light_bg)


# Semantic roles used across the codebase, each backed by one of the twelve
# palette hues above by default. Centralizing the role→hue mapping here means
# a new theme only has to supply the twelve palette hexes (above) — it never
# needs to know about roles (several roles may share a hue). Each role can
# also be overridden independently
# via `NX_<ROLE>_COLOR` (see _parse_color_override), e.g. NX_TEXT_COLOR=green,
# NX_STATUS_BAR_COLOR=ffaa00, NX_REASONING_TEXT_COLOR=gray.
ROLE_DEFAULTS: dict[str, str] = {
    "TEXT": "green",           # the model's final answer text
    "REASONING_TEXT": "gray",  # reasoning/thinking block text
    "STATUS_BAR": "cyan",      # per-turn stats footer (tok/s, ctx, ttft, wall)
    "TOOL_OUTPUT": "magenta",  # tool/bash result body lines
    "ACCENT": "blue",          # borders, model name, chatbox prompt
    "MEMORY": "purple",        # memory tool calls (memory_search/write/delete)
    "SUBAGENT": "teal",        # spawn_agent tool call boxes (delegated work)
    "LABEL": "orange",         # source name / session id badges
    "TODO": "orange",          # todo_write tool output (distinct from TOOL_OUTPUT)
    "SUCCESS": "green",        # yolo/auto-approve/allow indicators
    "WARNING": "yellow",       # medium-risk / approval-prompt / recoverable-cut notices
    "ERROR": "red",            # hard failures
}


_ColorOverride = tuple[Literal["palette", "hex"], str]


def _parse_color_override(env_name: str) -> _ColorOverride | None:
    """Resolve `NX_<ROLE>_COLOR` to either a palette key (resolved dynamically
    against the active theme, so it tracks theme switches) or a literal 6-digit
    hex (theme-independent). Returns None if unset or invalid."""
    val = os.environ.get(env_name)
    if not val:
        return None
    v = val.strip()
    if v.lower() in _PALETTE_KEYS:
        return ("palette", v.lower())
    h = v.lstrip("#")
    if len(h) == 6 and all(c in "0123456789abcdefABCDEF" for c in h):
        return ("hex", h)
    print(f"nx: ignoring invalid {env_name}={val!r} — want one of "
          f"{', '.join(_PALETTE_KEYS)} or a 6-digit hex color", file=sys.stderr)
    return None


_ROLE_OVERRIDES: dict[str, _ColorOverride | None] = {
    role: _parse_color_override(f"NX_{role}_COLOR") for role in ROLE_DEFAULTS
}


class _Colors:
    """Mutable color palette. Other modules do `from ._ansi import C` and read
    `C.ACCENT` / `C.TEXT` / etc. (semantic roles) or the raw `C.CYAN` etc. at
    call time (not via plain name imports), so `set_theme()` mutating this
    single shared instance is immediately visible everywhere."""

    CYAN: str
    GREEN: str
    YELLOW: str
    RED: str
    MAGENTA: str
    GRAY: str
    BLUE: str
    ORANGE: str
    PURPLE: str
    TEAL: str
    PINK: str
    BROWN: str
    TEXT: str
    REASONING_TEXT: str
    STATUS_BAR: str
    TOOL_OUTPUT: str
    ACCENT: str
    MEMORY: str
    SUBAGENT: str
    LABEL: str
    TODO: str
    SUCCESS: str
    WARNING: str
    ERROR: str

    def __init__(self, name: str) -> None:
        self.name = name
        # None = the terminal background side is unknown; use the authored
        # palette. True/False (light/dark, set by the luminance probe below or
        # by a future re-probe) swaps in the derived opposite-side variant.
        self.light_bg: bool | None = None
        self._apply(THEMES[name])

    def _apply(self, hexes: dict[str, str]) -> None:
        if self.light_bg is not None:
            hexes = _variant_hexes(self.name, self.light_bg)
        self.CYAN = _rgb(hexes["cyan"])
        self.GREEN = _rgb(hexes["green"])
        self.YELLOW = _rgb(hexes["yellow"])
        self.RED = _rgb(hexes["red"])
        self.MAGENTA = _rgb(hexes["magenta"])
        self.GRAY = _rgb(hexes["gray"])
        self.BLUE = _rgb(hexes["blue"])
        self.ORANGE = _rgb(hexes["orange"])
        self.PURPLE = _rgb(hexes["purple"])
        self.TEAL = _rgb(hexes["teal"])
        self.PINK = _rgb(hexes["pink"])
        self.BROWN = _rgb(hexes["brown"])
        for role, palette_key in ROLE_DEFAULTS.items():
            override = _ROLE_OVERRIDES.get(role)
            if override is None:
                hexv = hexes[palette_key]
            elif override[0] == "palette":
                hexv = hexes[override[1]]
            else:
                hexv = override[1]
            setattr(self, role, _rgb(hexv))


_requested_theme = os.environ.get("NX_THEME")
if _requested_theme and _requested_theme not in THEMES:
    avail = ", ".join(sorted(THEMES))
    print(f"nx: unknown NX_THEME {_requested_theme!r} — falling back to {DEFAULT_THEME!r}. "
          f"Available: {avail}", file=sys.stderr)

# Auto-detect the terminal background's lightness and, when the active theme's
# authored side doesn't match it, apply that theme's derived opposite-side
# variant (the probes live in nx/_tty.py; explicit NX_THEME always wins; a
# failed probe keeps the authored palette). C.light_bg also drives _apply().


def _variant_hexes(name: str, light_bg: bool) -> dict[str, str]:
    """Palette hexes for `name` adapted to the terminal background side:
    the authored palette when its side already matches, else the derived
    opposite-side variant. One resolution shared by _Colors._apply()."""
    hexes = THEMES[name]
    if AUTHORED_THEME_LIGHTNESS[name] == light_bg:
        return hexes
    return _variant_derive(hexes, to_light=light_bg)


C = _Colors(_requested_theme if _requested_theme in THEMES else DEFAULT_THEME)
if not _requested_theme:
    from ._tty import _osc_bg_luminance, _COLORFGBG_LUMINANCE
    _lum = _osc_bg_luminance()
    if _lum is None:
        _lum = _COLORFGBG_LUMINANCE()
    if _lum is not None:
        C.light_bg = _lum >= 0.5
        C._apply(THEMES[C.name])
        if C.light_bg:
            DEFAULT_THEME = "latte"  # informational: the light-suited default


def set_theme(name: str) -> bool:
    """Switch the active theme in place. Returns True if `name` is a known theme."""
    if name not in THEMES:
        return False
    C.name = name
    C._apply(THEMES[name])
    # Keep the one rich-rendered surface (the say-panel's markdown, see
    # _rich.py) on the same palette — cheap no-op when rich isn't in play.
    try:
        from ._rich import retheme_console
        from ._turn import _say_console
        retheme_console(_say_console)
    except Exception:
        pass  # never let a theme switch break the ANSI path
    return True


def swatch(theme_name: str) -> str:
    """Return a one-line rendered sample of `theme_name`'s palette (the twelve hue slots)."""
    hexes = THEMES[theme_name]
    order = ("cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange",
             "purple", "teal", "pink", "brown")
    return " ".join(f"{_rgb(hexes[k])}██{RESET}" for k in order)


# --- waiting animation (calm sweep) ------------------------------------------
# Replaces the old 24-cell Game-of-Life row: that animation redrew EVERY cell
# at ~11 Hz in a high-contrast glyph pattern, which is both visually noisy and
# a genuine photosensitive-epilepsy risk (large flashing contrast areas are
# the classic trigger — generalized flashes above ~3 Hz with >10% screen area
# are what safety guidelines flag). This one is designed to be safe:
#   * a static-width bar — the SAME cells are "on" every frame except one
#     head and one tail cell, so at most 2 of ~12 glyphs change per frame
#     (the rest never blink);
#   * slow (default 4 fps) — well under any photosensitivity threshold;
#   * low contrast (dim gray track, one highlighted runner) and no color
#     cycling;
#   * NX_SPINNER=0 disables the animation entirely (label only).
class LifeSpinner:
    """A quiet, static-width activity line: `label [▪▪▪▪···]`.

    Kept for API compat with the old name so call sites/tests stay as-is
    (`start()`/`stop()`/`label`). The label is clamped and control-char
    stripped on render — `_bash_jobs.py` streams raw log tails into it, which
    previously injected ANSI/OSC straight into the terminal and broke the
    line width, smearing the spinner over the tool box.

    Three optional live segments can ride the line, all off by default so the
    classic render is byte-identical to before. They exist because a tool that
    took 40 seconds looked exactly like one that took 0.1 seconds, leaving the
    user in the dark about what was happening (0.35.0, see nx/_progress.py):
      * `show_elapsed=True` appends the seconds since `start()`.
      * `counter` (a `_progress.ProgressCounter`, set by a parallel fan-out)
        appends how many of the group's calls have landed, as `2/3`.
      * a tool body calling `note_progress()` appends its note.
    """

    _WIDTH = 12       # total cells; runner covers TRACK+1 while sweeping
    _TRACK = 4        # runner length in cells
    _MIN_TICK = 0.12  # seconds per step — hard floor for a gentle pace

    def __init__(self, label: str = "thinking", tick: float = 0.25) -> None:
        self.label = label
        self.tick = max(self._MIN_TICK, tick)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._t: threading.Thread | None = None
        self._pos = 0
        # Live tail of the line (see the class docstring). At their off-values,
        # every pre-existing call site renders exactly as before.
        self.show_elapsed = False
        self.counter = None
        self.note = ""
        self._live = True
        self._t0: float | None = None

    def _tail(self, width: int = MAX_TAIL) -> str:
        """The live `· 4.2s · 2/3 · note` segment (DIM), or "" when this spinner
        tracks nothing. Safe to call from any thread: every input is read once
        and each read is a GIL-atomic reference load, so a frame sees a
        consistent before-or-after value, never a torn one.

        Returns "" when nothing is animating (NX_SPINNER=0 draws its line once,
        so a frozen clock or a stale count would be a lie).

        `width` covers the WHOLE segment including its ` · ` lead, which is what
        _render() passes (whatever the label left of the row's budget), keeping
        its arithmetic plain addition: len(label) + len(this) <= TAIL_BUDGET."""
        if not self._live:
            return ""
        parts = []
        if self.show_elapsed and self._t0 is not None:
            parts.append(fmt_secs(time.monotonic() - self._t0) + "s")
        if self.counter is not None:
            parts.append(str(self.counter))
        note = " ".join(strip_ansi_for_display(self.note or "").split())
        if note:
            parts.append(clip_line(note, MAX_NOTE_SHOWN))
        if not parts:
            return ""
        # Clip per PART, then the whole: clipping only the joined string would
        # spend the budget on the elapsed count and amputate the note, which is
        # the one segment a long tool chose to say something with.
        return " " + DIM + "· " + clip_line(" · ".join(parts), width - 3) + RESET

    def _render(self, pos: int) -> str:
        # Clamp + sanitize the label: it may come from a live command's log
        # tail (st._ACTIVE_SPINNER.label = ...), i.e. untrusted bytes. \n is
        # NOT stripped by strip_ansi_for_display, so collapse whitespace
        # explicitly — the spinner must stay exactly one line.
        label = " ".join(strip_ansi_for_display(self.label or "").split())
        # label + tail share one TAIL_BUDGET-cell row, and the LABEL is served
        # first: `running search_files` is the most useful thing on the line, so
        # it keeps up to LABEL_CELLS and the tail clips into the remainder (never
        # below MIN_TAIL, so the clock still reads). Clamping only the label let a
        # long note squeeze the tool name to five cells; the tail is redrawn every
        # frame and unbounded by construction, so both ends need a bound or the
        # row wraps and leaves droppings over the tool box.
        label = clip_line(label, LABEL_CELLS)
        tail_w = TAIL_BUDGET - len(label)
        tail = self._tail(tail_w if tail_w > MIN_TAIL else MIN_TAIL)
        cells = ["·"] * self._WIDTH
        for i in range(self._TRACK):
            cells[(pos + i) % self._WIDTH] = "▪"
        bar = "".join(cells)
        return CLEAR_LINE + "  " + DIM + label + tail + RESET + " " + DIM + "[" + bar + "]" + RESET

    def _run(self) -> None:
        try:
            sys.stdout.write(self._render(self._pos))
            sys.stdout.flush()
            while not self._stop.wait(self.tick):
                self._pos = (self._pos + 1) % self._WIDTH
                sys.stdout.write(self._render(self._pos))
                sys.stdout.flush()
        finally:
            # wipe the spinner line deterministically before returning —
            # _run always exits through here, so the caller's next print
            # never lands on a half-drawn bar
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()
            self._done.set()

    def start(self) -> None:
        if not sys.stdout.isatty():
            return  # headless (piped/redirected) — no animation
        self._stop.clear()
        self._done.clear()
        # Stamped once per spinner and deliberately NOT re-stamped by a later
        # start(): _confirm() restarts this same spinner around an approval
        # prompt, and the number must keep reading as how long the tool has been
        # in flight rather than resetting to 0.0s the moment the Y/n goes away.
        if self._t0 is None:
            self._t0 = time.monotonic()
        if os.environ.get("NX_SPINNER", "1") == "0":
            # animation disabled: draw the label ONCE, statically, and remember
            # it so stop() can clear the same line. No thread, no redraws.
            self._live = False
            self._static_line = self._render(0)
            sys.stdout.write(self._static_line)
            sys.stdout.flush()
            return
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            # join WITHOUT a timeout: the render loop is bounded (one write,
            # one flush per tick) so it always finishes promptly, and an
            # abandoned live thread (old join(timeout=0.5)) is exactly what
            # smeared the bar over the tool box.
            self._t.join()
            self._t = None
            return
        if getattr(self, "_static_line", None) is not None:
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()
            self._static_line = None


