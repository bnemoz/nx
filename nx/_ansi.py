"""ANSI color/cursor constants and the terminal waiting animation (LifeSpinner)."""
from __future__ import annotations

import os
import re
import sys
import threading
from typing import Literal

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

# Eight hex values per theme: cyan/green/yellow/red/magenta, plus "gray",
# "blue", and "orange". blue backs the ACCENT role (borders, model name, the
# chatbox prompt) so it no longer collides with STATUS_BAR's cyan; orange backs
# LABEL (source/session badges) so it no longer collides with TOOL_OUTPUT's
# magenta. "gray" is used for reasoning-block text, where plain SGR-2 dim or
# ANSI bright-black (90) is unreadable on many terminals, so it needs its own
# legible-but-muted hue per theme rather than a fixed gray.
THEMES: dict[str, dict[str, str]] = {
    "mocha": {   # Catppuccin Mocha (default) — dark
        "cyan": "89dceb", "green": "a6e3a1", "yellow": "f9e2af",
        "red": "f38ba8", "magenta": "cba6f7", "gray": "7f849c",
        "blue": "89b4fa", "orange": "fab387",
    },
    "latte": {   # Catppuccin Latte — light
        "cyan": "04a5e5", "green": "40a02b", "yellow": "df8e1d",
        "red": "d20f39", "magenta": "8839ef", "gray": "6c6f85",
        "blue": "1e66f5", "orange": "fe640b",
    },
    "dracula": {
        "cyan": "8be9fd", "green": "50fa7b", "yellow": "f1fa8c",
        "red": "ff5555", "magenta": "bd93f9", "gray": "6272a4",
        "blue": "7aa6da", "orange": "ffb86c",
    },
    "nord": {
        "cyan": "88c0d0", "green": "a3be8c", "yellow": "ebcb8b",
        "red": "bf616a", "magenta": "b48ead", "gray": "4c566a",
        "blue": "81a1c1", "orange": "d08770",
    },
    "gruvbox": {   # Gruvbox dark
        "cyan": "8ec07c", "green": "b8bb26", "yellow": "fabd2f",
        "red": "fb4934", "magenta": "d3869b", "gray": "928374",
        "blue": "83a598", "orange": "fe8019",
    },
    "vaporwave": {   # neon pastel — fluorescent but soft
        "cyan": "5ffbf1", "green": "8aff80", "yellow": "fdfd96",
        "red": "ff6ec7", "magenta": "d183ff", "gray": "9a86b3",
        "blue": "7aa2f7", "orange": "ff9e64",
    },
    "amethyst": {   # monochrome purple — every hue is a violet/orchid/pink
        # shade, so roles stay distinguishable by lightness and by position in
        # the violet→pink band rather than by hue. "gray" stays a true neutral
        # gray: reasoning text is the one thing that shouldn't go purple.
        "cyan": "b581f8", "green": "dfccff", "yellow": "b433ff",
        "red": "f65aa3", "magenta": "ec8fff", "gray": "9e9eaa",
        "blue": "6655fc", "orange": "fa47cd",
    },
}
DEFAULT_THEME = "mocha"
_PALETTE_KEYS: tuple[str, ...] = ("cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange")

# Semantic roles used across the codebase, each backed by one of the eight
# palette hues above by default. Centralizing the role→hue mapping here means
# a new theme only has to supply the eight palette hexes (above) — it never
# needs to know about roles. Each role can also be overridden independently
# via `NX_<ROLE>_COLOR` (see _parse_color_override), e.g. NX_TEXT_COLOR=green,
# NX_STATUS_BAR_COLOR=ffaa00, NX_REASONING_TEXT_COLOR=gray.
ROLE_DEFAULTS: dict[str, str] = {
    "TEXT": "green",           # the model's final answer text
    "REASONING_TEXT": "gray",  # reasoning/thinking block text
    "STATUS_BAR": "cyan",      # per-turn stats footer (tok/s, ctx, ttft, wall)
    "TOOL_OUTPUT": "magenta",  # tool/bash result body lines
    "ACCENT": "blue",          # borders, model name, chatbox prompt
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
    TEXT: str
    REASONING_TEXT: str
    STATUS_BAR: str
    TOOL_OUTPUT: str
    ACCENT: str
    LABEL: str
    TODO: str
    SUCCESS: str
    WARNING: str
    ERROR: str

    def __init__(self, name: str) -> None:
        self.name = name
        self._apply(THEMES[name])

    def _apply(self, hexes: dict[str, str]) -> None:
        self.CYAN = _rgb(hexes["cyan"])
        self.GREEN = _rgb(hexes["green"])
        self.YELLOW = _rgb(hexes["yellow"])
        self.RED = _rgb(hexes["red"])
        self.MAGENTA = _rgb(hexes["magenta"])
        self.GRAY = _rgb(hexes["gray"])
        self.BLUE = _rgb(hexes["blue"])
        self.ORANGE = _rgb(hexes["orange"])
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

C = _Colors(_requested_theme if _requested_theme in THEMES else DEFAULT_THEME)


def set_theme(name: str) -> bool:
    """Switch the active theme in place. Returns True if `name` is a known theme."""
    if name not in THEMES:
        return False
    C.name = name
    C._apply(THEMES[name])
    return True


def swatch(theme_name: str) -> str:
    """Return a one-line rendered sample of `theme_name`'s palette (the eight hue slots)."""
    hexes = THEMES[theme_name]
    order = ("cyan", "green", "yellow", "red", "magenta", "gray", "blue", "orange")
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

    def _render(self, pos: int) -> str:
        # Clamp + sanitize the label: it may come from a live command's log
        # tail (st._ACTIVE_SPINNER.label = ...), i.e. untrusted bytes. \n is
        # NOT stripped by strip_ansi_for_display, so collapse whitespace
        # explicitly — the spinner must stay exactly one line.
        label = " ".join(strip_ansi_for_display(self.label or "").split())
        if len(label) > 48:
            label = label[:45] + "..."
        cells = ["·"] * self._WIDTH
        for i in range(self._TRACK):
            cells[(pos + i) % self._WIDTH] = "▪"
        bar = "".join(cells)
        return f"{CLEAR_LINE}  {DIM}{label}{RESET} {DIM}[{bar}]{RESET}"

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
        if os.environ.get("NX_SPINNER", "1") == "0":
            # animation disabled: draw the label ONCE, statically, and remember
            # it so stop() can clear the same line. No thread, no redraws.
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


