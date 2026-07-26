"""ANSI color/cursor constants and the terminal waiting animation (LifeSpinner)."""
import contextlib
import os
import random
import sys
import termios
import threading
import time


@contextlib.contextmanager
def raw_mode(fd, *, icrnl=False):
    """Put fd into raw (non-canonical, no-echo) mode for the duration of the
    block, restoring the original attributes on exit. Centralizes the
    termios save/restore/flag-twiddle that used to be copy-pasted across
    _chatbox.py's editor and _approval.py's Y/n/esc prompt — see
    changelog's "terminal breaks on resize" entry for why duplicating this
    is a liability."""
    old = termios.tcgetattr(fd)
    new = old[:]
    new[3] &= ~(termios.ECHO | termios.ICANON)
    if icrnl:
        new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        yield old
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass

def _rgb(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CLEAR_LINE = "\033[2K\r"   # erase entire line, return cursor to col 0
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

# Five hex values per theme: cyan/green/yellow/red/magenta, plus "gray" —
# used for reasoning-block text, where plain SGR-2 dim or ANSI bright-black
# (90) is unreadable on many terminals, so it needs its own legible-but-muted
# hue per theme rather than a fixed gray.
THEMES = {
    "mocha": {   # Catppuccin Mocha (default) — dark
        "cyan": "89dceb", "green": "a6e3a1", "yellow": "f9e2af",
        "red": "f38ba8", "magenta": "cba6f7", "gray": "7f849c",
    },
    "latte": {   # Catppuccin Latte — light
        "cyan": "04a5e5", "green": "40a02b", "yellow": "df8e1d",
        "red": "d20f39", "magenta": "8839ef", "gray": "6c6f85",
    },
    "dracula": {
        "cyan": "8be9fd", "green": "50fa7b", "yellow": "f1fa8c",
        "red": "ff5555", "magenta": "bd93f9", "gray": "6272a4",
    },
    "nord": {
        "cyan": "88c0d0", "green": "a3be8c", "yellow": "ebcb8b",
        "red": "bf616a", "magenta": "b48ead", "gray": "4c566a",
    },
    "gruvbox": {   # Gruvbox dark
        "cyan": "8ec07c", "green": "b8bb26", "yellow": "fabd2f",
        "red": "fb4934", "magenta": "d3869b", "gray": "928374",
    },
    "vaporwave": {   # neon pastel — fluorescent but soft
        "cyan": "5ffbf1", "green": "8aff80", "yellow": "fdfd96",
        "red": "ff6ec7", "magenta": "d183ff", "gray": "9a86b3",
    },
}
DEFAULT_THEME = "mocha"
_PALETTE_KEYS = ("cyan", "green", "yellow", "red", "magenta", "gray")

# Semantic roles used across the codebase, each backed by one of the six
# palette hues above by default. Centralizing the role→hue mapping here means
# a new theme only has to supply the six palette hexes (above) — it never
# needs to know about roles. Each role can also be overridden independently
# via `NX_<ROLE>_COLOR` (see _parse_color_override), e.g. NX_TEXT_COLOR=green,
# NX_STATUS_BAR_COLOR=ffaa00, NX_REASONING_TEXT_COLOR=gray.
ROLE_DEFAULTS = {
    "TEXT": "green",           # the model's final answer text
    "REASONING_TEXT": "gray",  # reasoning/thinking block text
    "STATUS_BAR": "cyan",      # per-turn stats footer (tok/s, ctx, ttft, wall)
    "TOOL_OUTPUT": "magenta",  # tool/bash result body lines
    "ACCENT": "cyan",          # borders, model name, chatbox prompt
    "LABEL": "magenta",        # source name / session id badges
    "SUCCESS": "green",        # yolo/auto-approve/allow indicators
    "WARNING": "yellow",       # medium-risk / approval-prompt / recoverable-cut notices
    "ERROR": "red",            # hard failures
}


def _parse_color_override(env_name):
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


_ROLE_OVERRIDES = {role: _parse_color_override(f"NX_{role}_COLOR") for role in ROLE_DEFAULTS}


class _Colors:
    """Mutable color palette. Other modules do `from ._ansi import C` and read
    `C.ACCENT` / `C.TEXT` / etc. (semantic roles) or the raw `C.CYAN` etc. at
    call time (not via plain name imports), so `set_theme()` mutating this
    single shared instance is immediately visible everywhere."""

    def __init__(self, name):
        self.name = name
        self._apply(THEMES[name])

    def _apply(self, hexes):
        self.CYAN = _rgb(hexes["cyan"])
        self.GREEN = _rgb(hexes["green"])
        self.YELLOW = _rgb(hexes["yellow"])
        self.RED = _rgb(hexes["red"])
        self.MAGENTA = _rgb(hexes["magenta"])
        self.GRAY = _rgb(hexes["gray"])
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


def set_theme(name):
    """Switch the active theme in place. Returns True if `name` is a known theme."""
    if name not in THEMES:
        return False
    C.name = name
    C._apply(THEMES[name])
    return True


def swatch(theme_name):
    """Return a one-line rendered sample of `theme_name`'s colors (cyan/green/yellow/red/magenta/gray)."""
    hexes = THEMES[theme_name]
    order = ("cyan", "green", "yellow", "red", "magenta", "gray")
    return " ".join(f"{_rgb(hexes[k])}██{RESET}" for k in order)


# --- waiting animation (tiny Conway's Game of Life) -------------------------
# A spinner is boring. A 1-row toroidal Game of Life is the same shape on screen
# (one line of cells) but actually does something — patterns glide, blinkers
# flash, gliders crawl. Runs in a background thread; the main loop kills it
# the instant the first token arrives.
_GOL_W = 24
_GOL_ALIVE = "█"
_GOL_DEAD = "·"
_GOL_GLIDER = {(0, 0), (1, 1), (2, 1), (0, 2), (1, 0)}  # 5-cell, period-4


class LifeSpinner:
    def __init__(self, width=_GOL_W, tick_ms=90, label="thinking"):
        self.w = width
        self.tick = tick_ms / 1000
        self.label = label  # shown before the cells ("thinking" / "running" / ...)
        self._stop = threading.Event()
        self._t = None

    def _seed(self):
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

    def _step(self, row):
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

    def _run(self):
        sys.stdout.write(HIDE_CURSOR)
        try:
            row = self._seed()
            # initial render — also reserve the line so subsequent prints don't shift things
            sys.stdout.write(CLEAR_LINE + "  " + DIM + f"{self.label} " + RESET +
                             "".join(_GOL_ALIVE if c else _GOL_DEAD for c in row))
            sys.stdout.flush()
            while not self._stop.is_set():
                time.sleep(self.tick)
                if self._stop.is_set():
                    break
                row = self._step(row)
                sys.stdout.write(CLEAR_LINE + "  " + DIM + f"{self.label} " + RESET +
                                 "".join(_GOL_ALIVE if c else _GOL_DEAD for c in row))
                sys.stdout.flush()
        finally:
            # wipe the spinner line and restore cursor
            sys.stdout.write(CLEAR_LINE + SHOW_CURSOR)
            sys.stdout.flush()

    def start(self):
        if not sys.stdout.isatty():
            return  # headless (piped/redirected) — no spinner animation
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.5)
            self._t = None


