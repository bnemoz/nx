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


def _no_color() -> bool:
    """True when the user or the environment has asked for plain output.
    Matches rich's own predicate (0.128.x/NO_COLOR gap fix): `NO_COLOR` counts
    only when set to a non-empty value (rich >=14.0.0 semantics), and
    `TERM=dumb` means no usable color. One shared predicate for --help and the
    raw-ANSI core, so the two can't disagree."""
    if os.environ.get("NO_COLOR"):
        return True
    return os.environ.get("TERM") == "dumb"


DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CLEAR_LINE = "\033[2K\r"   # erase entire line, return cursor to col 0


def notice(text: str) -> None:
    """One human-facing line to stdout, flushed (UIUX_SPEC A3 writer
discipline, 0.134.0). The single replacement for ad-hoc `print()` calls in
REPL feedback paths: same bytes as `print(text)` (caller keeps its own
DIM/RESET decoration) but flushed like the live-render path expects, so a
notice never interleaves torn with the spinner/live region. Phase B moves
this to `_render.py` as `notice(level, text)` with role colors."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def warn(text: str) -> None:
    """One human-facing warning line to stderr, flushed (UIUX_SPEC A3).
The single stderr writer for configuration warnings (theme/role-color
messages) — keeps those off the stdout render stream without each site
hand-rolling `file=sys.stderr`."""
    sys.stderr.write(text + "\n")
    sys.stderr.flush()

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

# --- display width / ANSI-aware clip (one measure/pad/clip trio) --------------
# The rail composite used to slice styled lines with a raw codepoint slice
# (`ln[:lw]`), which cut escape sequences in half (leaking a literal `m`) and
# counted codepoints where the terminal counts CELLS — a CJK/emoji cell then
# shifted the `│` separator right, wrapped the row, and desynced the live
# region's erase arithmetic. One implementation lives here so `_panel._clip` and
# the `_surfaces` rail composite can never disagree again.
#
# The width table comes from `rich.cells` (already an nx dependency, and the
# same table the Live console uses to lay the frame out — so nx's idea of a
# cell width always matches rich's). `wcwidth` is deliberately NOT a dependency:
# it isn't installed, and rich needs the table anyway.


def _char_cells(ch: str) -> int:
    """Display cells one character occupies (0 combining, 1 narrow, 2 wide)."""
    from rich.cells import get_character_cell_size
    return get_character_cell_size(ch)


def sgr_span_end(s: str, i: int) -> int:
    """End index (exclusive) of the SGR sequence starting at `s[i] == '\\x1b'`,
    or -1 when there is no complete sequence here. Only `\\033[...m` is treated as
    zero-width: any other escape is skipped whole so it can never leak."""
    if s[i] != "\x1b":
        return -1
    j = i + 1
    if j < len(s) and s[j] == "[":
        j += 1
        while j < len(s) and (s[j].isdigit() or s[j] in ";:"):
            j += 1
        if j < len(s) and s[j] == "m":
            return j + 1
        return -1
    for j in range(i + 1, len(s)):
        if "@" <= s[j] <= "~":
            return j + 1
    return -1


def display_width(s: str) -> int:
    """Visible columns of `s`: ANSI escapes count 0, wide chars count 2."""
    from rich.cells import cell_len
    return cell_len(strip_ansi_for_display(s))


def clip_display(s: str, w: int) -> str:
    """Truncate `s` to at most `w` VISIBLE columns.

    SGR sequences are copied through uncounted and are never cut mid-sequence
    (the raw `[:w]` slice this replaces could emit a bare `m`); a wide char that
    would straddle the limit is dropped rather than half-drawn."""
    if w <= 0:
        return ""
    out: list[str] = []
    n = 0
    i = 0
    ln = len(s)
    while i < ln:
        if s[i] == "\x1b":
            j = sgr_span_end(s, i)
            if j == -1:
                i += 1          # malformed: drop the lone ESC, keep scanning
                continue
            out.append(s[i:j])
            i = j
            continue
        if s[i] == "\n":
            break
        cw = _char_cells(s[i])
        if cw == 0:
            if n < w:
                out.append(s[i])
            i += 1
            continue
        if n + cw > w:
            break
        out.append(s[i])
        n += cw
        i += 1
    return "".join(out)


def pad_display(s: str, w: int) -> str:
    """`s` padded with spaces to exactly `w` visible columns (already-wider
    input is clipped). Callers appending a separator after the pad must leave a
    column free themselves — see `_surfaces._live_frame`'s cols-1 reserve."""
    cur = display_width(s)
    if cur >= w:
        return clip_display(s, w)
    return s + " " * (w - cur)


# Thirteen hex values per theme: cyan/green/yellow/red/magenta + gray, blue,
# orange (the original eight), plus purple, teal, pink, brown, and slate. Every slot
# is expected to be *used* somewhere: the ten semantic roles draw from eight of
# them (ROLE_DEFAULTS below), and _rich.py maps markdown styles and code-block
# syntax tokens onto the rest (purple bullets/numbers, pink decorators/links,
# brown docstrings/link URLs, teal table borders). "gray" is reasoning text,
# where plain SGR-2 dim or ANSI bright-black (90) is unreadable on many
# terminals, so it needs its own legible-but-muted hue per theme rather than a
# fixed gray. A new theme must be designed as a whole: all thirteen slots
# distinct in RGB space, and the amethyst monochrome rule (below) is the
# exception that proves it.
THEMES: dict[str, dict[str, str]] = {
    "mocha": {   # Catppuccin Mocha (default) — dark
        "cyan": "89dceb", "green": "a6e3a1", "yellow": "f9e2af",
        "red": "f38ba8", "magenta": "cba6f7", "gray": "7f849c",
        "blue": "89b4fa", "orange": "fab387",
        "purple": "b4befe", "teal": "94e2d5", "pink": "f5c2e7",
        "brown": "e0b089", "slate": "9399b2",
    },
    "latte": {   # Catppuccin Latte — light
        "cyan": "04a5e5", "green": "40a02b", "yellow": "df8e1d",
        "red": "d20f39", "magenta": "8839ef", "gray": "6c6f85",
        "blue": "1e66f5", "orange": "fe640b",
        "purple": "7287fd", "teal": "179299", "pink": "ea76cb",
        "brown": "9a6b4f", "slate": "8c8fa1",
    },
    "dracula": {
        # softened from upstream: every slot re-derived in OKLCH at uniform
        # pastel lightness (0.74-0.82) and low chroma (0.09-0.12); hues keep
        # dracula's cool violet/pink lean. Contrast 4.7:1+ on dark backgrounds.
        "cyan": "72cede", "green": "90d192", "yellow": "d8c477",
        "red": "eb9097", "magenta": "c8a2e4", "gray": "868991",
        "blue": "89a9f1", "orange": "e7ad74",
        "purple": "b5a4ea", "teal": "7cccc1", "pink": "eda4cc",
        "brown": "bd9670", "slate": "829cb3",
    },
    "nord": {
        "cyan": "88c0d0", "green": "a3be8c", "yellow": "ebcb8b",
        "red": "bf616a", "magenta": "b48ead", "gray": "4c566a",
        "blue": "81a1c1", "orange": "d08770",
        "purple": "9d7fc9", "teal": "8fbcbb", "pink": "d8a1c0",
        "brown": "a97c5e", "slate": "7b88a1",
    },
    "gruvbox": {   # Gruvbox dark — softened
        # warm-hue retune in OKLCH: lower chroma (0.07-0.11), raised luma, warm
        # gray/slate; the old af3a03 brown was invisible on dark backgrounds.
        "cyan": "72cede", "green": "afc177", "yellow": "d7bb70",
        "red": "e1897b", "magenta": "d295bd", "gray": "857f76",
        "blue": "71afc3", "orange": "e09865",
        "purple": "b5a4ea", "teal": "87c0a6", "pink": "eba6c6",
        "brown": "aa7b54", "slate": "9c917e",
    },
    "vaporwave": {   # dusk pastel — the neon softened to an evening-fog palette
        # same OKLCH architecture as mist with violet/teal-leaning hues
        # (magenta 315, teal 175, pink 340); no slot above chroma 0.12.
        "cyan": "83d5d4", "green": "90d192", "yellow": "d8c477",
        "red": "f4928a", "magenta": "d4a0ec", "gray": "868991",
        "blue": "8bb1f7", "orange": "e7ad74",
        "purple": "ba9cef", "teal": "7bd1ba", "pink": "eda8d5",
        "brown": "bd9670", "slate": "829cb3",
    },
    "ayu-light": {   # ayu light (ayu-theme) — light
        # authored: yellow #eca944, orange #f2ae49, teal #4cbf99 sit above the
        # luma-160 legibility band on white; nudged darker in-hue (orange
        # #d78c16) to keep all twelve inside the band.
        "cyan": "2ba587", "green": "6d9c00", "yellow": "d18910",
        "red": "ea6c66", "magenta": "a37acc", "gray": "5c6166",
        "blue": "399ee6", "orange": "d78c16",
        "purple": "9e75c7", "teal": "2aa886", "pink": "ed6ea0",
        "brown": "99703f", "slate": "8a9199",
    },
    "bluloco-light": {   # bluloco light (uloco) — light
        # authored: yellow #c5a332 / teal #27618d outside the band; nudged
        # (yellow #b9992f, teal #296695, red #d52753 -> #db3a63 for 90+).
        "cyan": "0099e1", "green": "3cbc66", "yellow": "b9992f",
        "red": "db3a63", "magenta": "ce33c0", "gray": "676a77",
        "blue": "275fe4", "orange": "df631c",
        "purple": "823ff1", "teal": "296695", "pink": "ff6480",
        "brown": "946b3d", "slate": "8f95a5",
    },
    "frappe": {   # Catppuccin Frappé — medium-dark
        "cyan": "81c8be", "green": "a6d189", "yellow": "e5c890",
        "red": "e78284", "magenta": "ca9ee6", "gray": "838ba7",
        "blue": "8caaee", "orange": "ef9f76",
        "purple": "babbf1", "teal": "99d1db", "pink": "f4b8e4",
        "brown": "d699b6", "slate": "949cbb",
    },
    "opencode": {   # OpenCode default theme (opencode.ai) — dark, warm
        # primary #fab283 as orange; accent #9d7cd8 as magenta; secondary
        # #5c9cf5 as blue; warning #f5a742 as yellow; error #e06c75,
        # success #7fd88f, info #56b6c2. 0.149.0: softened in OKLCH — warm
        # slate-gray reasoning text (the old #808080 pure gray had no family
        # tie), the #ffc09f purple slot re-hued into the warm band, every
        # slot 4.5:1+ on dark backgrounds.
        "cyan": "72cede", "green": "90d192", "yellow": "e5bf6d",
        "red": "f4928a", "magenta": "b7a0e4", "gray": "8c857f",
        "blue": "85b4f0", "orange": "f2ad73",
        "purple": "e0b79c", "teal": "7cccc1", "pink": "eba6c6",
        "brown": "ca9971", "slate": "a88f82",
    },
    "amethyst": {   # monochrome purple — every hue is a violet/orchid/pink
        # shade, so roles stay distinguishable by lightness and by position in
        # the violet→pink band rather than by hue. "gray" stays a true neutral
        # gray: reasoning text is the one thing that shouldn't go purple.
        # 0.149.0: saturation softened toward 0.62 (HLS) with the lightness
        # pyramid widened — same separability bar, calmer read.
        "cyan": "b58fe5", "green": "e1d6f5", "yellow": "a856d7",
        "red": "df76a7", "magenta": "d998e7", "gray": "9e9eaa",
        "blue": "6154d4", "orange": "d755b7",
        "purple": "845fd9", "teal": "cab9ee", "pink": "e48cc6",
        "brown": "9d78d0", "slate": "623dae",
    },
    "mist": {   # calm pastel dark ("evening fog") — the house style
        # generated in OKLCH: uniform lightness bands (0.60-0.82), low chroma
        # (0.012-0.12), slate-tinted neutrals. Every slot >= 4.69:1 on dark
        # backgrounds; closest hue pair still 39/255 apart in RGB space.
        "cyan": "72cede", "green": "90d192", "yellow": "d8c477",
        "red": "f4928a", "magenta": "df9ed9", "gray": "868991",
        "blue": "85b4f0", "orange": "e7ad74",
        "purple": "b5a4ea", "teal": "7cccc1", "pink": "eba6c6",
        "brown": "bd9670", "slate": "829cb3",
    },
}
DEFAULT_THEME = "mist"
_PALETTE_KEYS: tuple[str, ...] = (
    "cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange",
    "purple", "teal", "pink", "brown", "slate",
)

# --- light/dark variants ------------------------------------------------------
# Every theme gains an automatic opposite-background variant: a dark theme
# (mocha, dracula, ...) gets a light variant for pale terminals, and the one
# authored light theme (latte) gets a dark one. The variant is DERIVED from
# the authored palette, not hand-tuned: mirror each hue's lightness into the
# band that's legible on the opposite background while keeping its hue family
# (so "mocha on white" still reads as mocha), nudge saturation up slightly on
# light backgrounds (white compresses perceived saturation), then push any
# near-duplicate pair apart in lightness until the thirteen slots are as
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
            # saturated one too bright (gruvbox green). The 118 top (not just
            # "not too dark") is a contrast requirement: luma 160 is only
            # ~2.6:1 against white; 118 holds ~4.7:1 (WCAG AA large-text).
            while luma_of(k) < 100 and hls[k][1] < 0.90:
                hls[k][1] += 0.02
            while luma_of(k) > 118 and hls[k][1] > 0.10:
                hls[k][1] -= 0.02
        else:
            while luma_of(k) < 100 and hls[k][1] < 0.95:
                hls[k][1] += 0.02
            while luma_of(k) > 185 and hls[k][1] > 0.10:
                hls[k][1] -= 0.02
            # a fully desaturated slot stalls in the loop above (its guard
            # requires sat > 0.05), so finish the dark-side luma ceiling on
            # lightness alone — hue is preserved where desaturation lost it
            while luma_of(k) > 200 and hls[k][1] > 0.10:
                hls[k][1] -= 0.02
    # The lightness walk above can stall below the luma floor for hues whose
    # luma is dominated by a channel their hue pins low (saturated violet on
    # the light side, gruvbox's red-dominant brown on the dark side);
    # desaturating toward gray covers the full luma range as a fixup.
    for k in hls:
        while luma_of(k) < 100 and hls[k][2] > 0.05:
            hls[k][2] -= 0.05
        while luma_of(k) > 118 and hls[k][2] > 0.05 and to_light:
            hls[k][2] -= 0.05

    def dist(a: str, b: str) -> float:
        return sum((int(rgb(a)[i:i + 2], 16) - int(rgb(b)[i:i + 2], 16)) ** 2
                   for i in (0, 2, 4)) ** 0.5

    # Deterministic separability pass: repeatedly take the closest hue pair
    # and split them in lightness until the thirteen slots are distinct enough
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
            while luma_of(k) > (118 if to_light else 200) and hls[k][2] > 0.05:
                hls[k][2] -= 0.05
            while luma_of(k) > (118 if to_light else 200) and hls[k][1] > 0.10:
                hls[k][1] -= 0.02
    return {k: rgb(k) for k in hexes}


# AUTHORED_THEME_LIGHTNESS: True for themes designed for a light terminal.
# The natural variant of theme T for a given background is T itself when the
# background matches T's authored side, else the derived opposite.
AUTHORED_THEME_LIGHTNESS: dict[str, bool] = {t: False for t in THEMES}
AUTHORED_THEME_LIGHTNESS["latte"] = True
AUTHORED_THEME_LIGHTNESS["ayu-light"] = True
AUTHORED_THEME_LIGHTNESS["bluloco-light"] = True


def variant_for(name: str, *, light_bg: bool) -> dict[str, str]:
    """The palette hexes for `name` rendered for `light_bg` (True = white-ish
    terminal background): the authored palette itself when that already matches
    the background, else the derived variant."""
    hexes = THEMES[name]
    if AUTHORED_THEME_LIGHTNESS[name] == light_bg:
        return hexes
    return _variant_derive(hexes, to_light=light_bg)


# Semantic roles used across the codebase, each backed by one of the thirteen
# palette hues above by default. Centralizing the role→hue mapping here means
# a new theme only has to supply the thirteen palette hexes (above) — it never
# needs to know about roles (several roles may share a hue). Each role can
# also be overridden independently
# via `NX_<ROLE>_COLOR` (see _parse_color_override), e.g. NX_TEXT_COLOR=green,
# NX_STATUS_BAR_COLOR=ffaa00, NX_REASONING_TEXT_COLOR=gray.
ROLE_DEFAULTS: dict[str, str] = {
    "TEXT": "green",           # the model's final answer text
    "REASONING_TEXT": "gray",  # reasoning/thinking block text
    "STATUS_BAR": "cyan",      # per-turn stats footer (tok/s, ctx, ttft, wall)
    "TOOL_OUTPUT": "slate",    # tool/bash result body lines (muted background
                               # chatter; magenta via NX_TOOL_OUTPUT_COLOR=magenta)
    "ACCENT": "blue",          # borders, model name, chatbox prompt
    "MEMORY": "purple",        # memory tool calls (memory_search/write/delete)
    "SUBAGENT": "teal",        # spawn_agent tool call boxes (delegated work)
    "LABEL": "orange",         # source name / session id badges
    "TODO": "orange",          # todo_write tool output (distinct from TOOL_OUTPUT)
    "SUCCESS": "green",        # yolo/auto-approve/allow indicators
    "WARNING": "yellow",       # medium-risk / approval-prompt / recoverable-cut notices
    "ERROR": "red",            # hard failures
    # Phase B role completion (0.135.0). MUTED/EMPHASIS are *text* roles
    # replacing raw `DIM`/`BOLD` on body text (the constants stay, for mode
    # control — spinners and hints). DIFF_ADD/DIFF_REMOVE default to the
    # green/red palette keys and exist so the approval/git diff surfaces can be
    # recolored independently of SUCCESS/ERROR (e.g. a theme where "added"
    # should read teal, not the same green as "ok").
    "MUTED": "gray",           # secondary/metadata text (was raw DIM on text)
    "EMPHASIS": "cyan",        # emphasized body text (was raw BOLD on text)
    "DIFF_ADD": "green",       # added lines in a diff preview
    "DIFF_REMOVE": "red",      # removed lines in a diff preview
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
    warn(f"nx: ignoring invalid {env_name}={val!r} — want one of "
         f"{', '.join(_PALETTE_KEYS)} or a 6-digit hex color")
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
    SLATE: str
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
    MUTED: str
    EMPHASIS: str
    DIFF_ADD: str
    DIFF_REMOVE: str

    def __init__(self, name: str) -> None:
        self.name = name
        # None = the terminal background side is unknown; use the authored
        # palette. True/False (light/dark, set by the luminance probe below or
        # by a future re-probe) swaps in the derived opposite-side variant.
        self.light_bg: bool | None = None
        self._apply(THEMES[name])

    def _apply(self, hexes: dict[str, str]) -> None:
        if _no_color():
            # Plain output requested: emit no SGR at all, so every styled
            # string round-trips to the empty decoration (ROADMAP gap fix).
            self.CYAN = self.GREEN = self.YELLOW = self.RED = ""
            self.MAGENTA = self.GRAY = self.BLUE = self.ORANGE = ""
            self.PURPLE = self.TEAL = self.PINK = self.BROWN = self.SLATE = ""
            for role in ROLE_DEFAULTS:
                setattr(self, role, "")
            return
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
        self.SLATE = _rgb(hexes["slate"])
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
    warn(f"nx: unknown NX_THEME {_requested_theme!r} — falling back to {DEFAULT_THEME!r}. "
         f"Available: {avail}")

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


def hex_of(sgr: str) -> str:
    """The 6-digit hex a `_rgb()`-built SGR foreground string encodes
    (e.g. '\\033[38;2;147;153;178m' -> '9399b2'). Round-trips the palette
    escapes so call sites can re-derive ramps from the live C.* values."""
    parts = sgr[len("\033[38;2;"):sgr.rindex("m")].split(";")
    return "%02x%02x%02x" % tuple(int(p) for p in parts[:3])


def swatch(theme_name: str) -> str:
    """Return a one-line rendered sample of `theme_name`'s palette (the thirteen hue slots)."""
    hexes = THEMES[theme_name]
    order = ("cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange",
             "purple", "teal", "pink", "brown", "slate")
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
def _region_live_open() -> bool:
    """Is the /tabs region Live currently owning the terminal line?

    SPEC_LIVEENGINE.md. Imported lazily (and defensively) because `_surfaces`
    imports `_ansi`, so a module-level import here would be a cycle."""
    try:
        from ._surfaces import dashboard_open
        return dashboard_open()
    except Exception:
        return False


def _draw_frame(text: str) -> None:
    """Emit one spinner frame (SPEC_LIVEENGINE.md §3, LifeSpinner).

    The raw-ANSI vs rich-Live branch that used to live here is now the
    `_render` component's (`emit(inplace=True)`, UIUX_SPEC B1: callers never
    branch on the render mode):

    * region Live OPEN — the Live owns line discipline, so a `\x1b[2K\r`
      rewrite would erase the Live's own frame (that is the tear this spec
      fixes). The frame becomes ONE committed console line via `cprint`, which
      rich composes above the frame. The `_WRITE_LOCK` is taken first so a
      4 fps repaint can never interleave with the frame write (no deadlock:
      the repaint path takes the same lock and only then the console lock).
    * region CLOSED — the classic in-place repaint, byte-identical to before
      (the spinner draws over itself on one line; committing a line per tick
      would scroll the terminal for every tool call).
    """
    from ._render import emit
    try:
        emit(text, inplace=True)
    except Exception:
        pass  # never let a spinner frame break the tool call (as before)


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
        from ._render import stream_tail
        parts = []
        if self.show_elapsed and self._t0 is not None:
            parts.append(fmt_secs(time.monotonic() - self._t0) + "s")
        if self.counter is not None:
            parts.append(str(self.counter))
        note = stream_tail(self.note or "", MAX_NOTE_SHOWN, plain=True)
        if note:
            parts.append(note)
        if not parts:
            return ""
        # Clip per PART, then the whole: clipping only the joined string would
        # spend the budget on the elapsed count and amputate the note, which is
        # the one segment a long tool chose to say something with.
        return " " + DIM + "· " + clip_line(" · ".join(parts), width - 3) + RESET

    def _render(self, pos: int) -> str:
        from ._render import stream_tail
        # Clamp + sanitize the label via the shared `stream_tail` component: it
        # may come from a live command's log tail (st._ACTIVE_SPINNER.label = ...),
        # i.e. untrusted bytes, and \n is NOT stripped by strip_ansi_for_display,
        # so whitespace is collapsed there — the spinner must stay exactly one
        # line. `plain=True` because the label is re-styled right here.
        label = stream_tail(self.label or "", LABEL_CELLS, plain=True)
        # label + tail share one TAIL_BUDGET-cell row, and the LABEL is served
        # first: `running search_files` is the most useful thing on the line, so
        # it keeps up to LABEL_CELLS and the tail clips into the remainder (never
        # below MIN_TAIL, so the clock still reads). Clamping only the label let a
        # long note squeeze the tool name to five cells; the tail is redrawn every
        # frame and unbounded by construction, so both ends need a bound or the
        # row wraps and leaves droppings over the tool box. `stream_tail` above
        # already applied the LABEL_CELLS clamp.
        tail_w = TAIL_BUDGET - len(label)
        tail = self._tail(tail_w if tail_w > MIN_TAIL else MIN_TAIL)
        cells = ["·"] * self._WIDTH
        for i in range(self._TRACK):
            cells[(pos + i) % self._WIDTH] = "▪"
        bar = "".join(cells)
        return CLEAR_LINE + "  " + DIM + label + tail + RESET + " " + DIM + "[" + bar + "]" + RESET

    def _run(self) -> None:
        try:
            _draw_frame(self._render(self._pos))
            while not self._stop.wait(self.tick):
                self._pos = (self._pos + 1) % self._WIDTH
                _draw_frame(self._render(self._pos))
        finally:
            # wipe the spinner line deterministically before returning —
            # _run always exits through here, so the caller's next print
            # never lands on a half-drawn bar. Under an open region Live there
            # is nothing to wipe: every frame is a committed console line.
            if not _region_live_open():
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
            _draw_frame(self._static_line)
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
            self._static_line = None
            if not _region_live_open():
                sys.stdout.write(CLEAR_LINE)
                sys.stdout.flush()


