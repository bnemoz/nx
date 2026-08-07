"""ANSI color/cursor constants and the terminal waiting animation (LifeSpinner)."""
from __future__ import annotations

import contextlib
import os
import random
import re
import signal
import sys
import termios
import threading
import time
from collections.abc import Iterator
from typing import Literal


@contextlib.contextmanager
def raw_mode(fd: int, *, icrnl: bool = False) -> Iterator[list]:
    """Put fd into raw (non-canonical, no-echo) mode for the duration of the
    block, restoring the original attributes on exit. Centralizes the
    termios save/restore/flag-twiddle shared by _chatbox.py's editor and
    _approval.py's Y/n/esc prompt, so their terminal-mode handling remains
    consistent."""
    old = termios.tcgetattr(fd)
    new = old[:]
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[0] &= ~termios.IXON
    if icrnl:
        new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0

    def _on_sigterm(signum, frame):
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    prev_handler = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _on_sigterm)
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        yield old
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, prev_handler)

def _rgb(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CLEAR_LINE = "\033[2K\r"   # erase entire line, return cursor to col 0
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

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


# --- waiting animation (tiny Conway's Game of Life) -------------------------
# A spinner is boring. A 1-row toroidal Game of Life is the same shape on screen
# (one line of cells) but actually does something — patterns glide, blinkers
# flash, gliders crawl. Runs in a background thread; the main loop kills it
# the instant the first token arrives.
_GOL_W = 24
_GOL_ALIVE = "█"
_GOL_DEAD = "·"
_GOL_GLIDER: set[tuple[int, int]] = {(0, 0), (1, 1), (2, 1), (0, 2), (1, 0)}  # 5-cell, period-4


class LifeSpinner:
    def __init__(self, width: int = _GOL_W, tick_ms: int = 90, label: str = "thinking") -> None:
        self.w = width
        self.tick = tick_ms / 1000
        self.label = label  # shown before the cells ("thinking" / "running" / ...)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _seed(self) -> list[int]:
        row = [0] * self.w
        x = random.randrange(self.w)
        for dx, _ in _GOL_GLIDER:
            row[(x + dx) % self.w] = 1
        for _ in range(2):
            x = random.randrange(self.w)
            row[x] = row[(x + 1) % self.w] = row[(x + 2) % self.w] = 1
        for _ in range(self.w // 6):
            row[random.randrange(self.w)] = 1
        return row

    def _step(self, row: list[int]) -> list[int]:
        # A 1-row GoL is degenerate (cells have only 2 neighbors). Cheat: treat
        # the row as the middle of a 3-row toroidal world where the rows above
        # and below mirror the current one. Gives every cell the standard 8
        # neighbors, so gliders/blinkers/etc. actually work.
        w, above, below, nxt = self.w, row, row, [0] * self.w
        for x in range(w):
            n = (above[(x - 1) % w] + above[x] + above[(x + 1) % w] +
                 row[(x - 1) % w]                   + row[(x + 1) % w] +
                 below[(x - 1) % w] + below[x] + below[(x + 1) % w])
            cur = row[x]
            nxt[x] = 1 if (cur and n in (2, 3)) or (not cur and n == 3) else 0
        return nxt

    def _render(self, row: list[int]) -> str:
        return (CLEAR_LINE + "  " + DIM + f"{self.label} " + RESET +
                "".join(_GOL_ALIVE if c else _GOL_DEAD for c in row))

    def _run(self) -> None:
        sys.stdout.write(HIDE_CURSOR)
        try:
            row = self._seed()
            # initial render — also reserve the line so subsequent prints don't shift things
            sys.stdout.write(self._render(row))
            sys.stdout.flush()
            while not self._stop.is_set():
                time.sleep(self.tick)
                if self._stop.is_set():
                    break
                row = self._step(row)
                sys.stdout.write(self._render(row))
                sys.stdout.flush()
        finally:
            # wipe the spinner line and restore cursor
            sys.stdout.write(CLEAR_LINE + SHOW_CURSOR)
            sys.stdout.flush()

    def start(self) -> None:
        if not sys.stdout.isatty():
            return  # headless (piped/redirected) — no spinner animation
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.5)
            self._t = None


