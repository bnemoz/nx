"""Shared render vocabulary (UIUX_SPEC Phase B, 0.135.0).

Every user-facing *shape* — box, notice, key/value row, diff, stream tail —
has exactly one implementation here. Each one resolves its own colors from the
semantic roles on `_ansi.C` (never from a raw palette hue) and its own glyphs
from the ONE box-glyph set below (B2: the rounded `╭│╰╮╯` family everywhere).
Callers pass plain text and a role; they never pick a color, a glyph, or a
render mode.

Render mode is decided ONCE per component:

* a rich `Live` owns the shared console (`_rich.shared_console()` — the
  SPEC_LIVEENGINE single-console contract) — the line must go through
  `_rich.cprint`, which composes it above the live frame instead of tearing
  through it, and an in-place `\\x1b[2K\\r` rewrite must NOT be emitted;
* no Live — the classic byte-identical raw write (in-place repaint for tails).

`emit()` is that one switch, so callers never branch on raw-ANSI vs rich-Live.
The previously duplicated branches (`_ansi._draw_frame` and `_stream_parse`'s
tool-status write) were folded into it.

Everything here is display-only: no component builds model-facing text, mutates
state, or starts a thread.
"""
from __future__ import annotations

from typing import Iterable, Optional

from . import _ansi as _A
from ._layout import DIFF_CONTEXT_LINES

# Bound from the module object (not `from ._ansi import ...`) so this module can
# be imported FIRST: `_ansi` imports `_render` only lazily inside its render
# helpers, so importing this module triggers a complete `_ansi` execution and
# these names are then real. `C` is a single instance mutated in place by
# `set_theme()`, so binding the attribute (not its value) stays live.
DIM, RESET, CLEAR_LINE, C = _A.DIM, _A.RESET, _A.CLEAR_LINE, _A.C

# --- B2: one box-glyph set ----------------------------------------------------
# The rounded family, matching the chatbox frame. Every box in nx (tool box,
# chatbox frame, status rail, footer prefix, parallel-group + rail boxes) draws
# from these constants, so the terminal reads as one vocabulary rather than
# three. `BOX_GLYPHS` is the single exported tuple a test can pin.
BOX_TL = "╭"   # top-left corner
BOX_TR = "╮"   # top-right corner
BOX_BL = "╰"   # bottom-left corner
BOX_BR = "╯"   # bottom-right corner
BOX_H = "─"    # horizontal rule
BOX_V = "│"    # vertical rail
BOX_BRANCH_MID = "├─"   # non-final entry inside a box
BOX_BRANCH_END = "╰─"   # final entry inside a box

BOX_GLYPHS: tuple[str, ...] = (
    BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_H, BOX_V, BOX_BRANCH_MID, BOX_BRANCH_END,
)

# --- glyph vocabulary (spinners / hints / tails) ------------------------------
TAIL_BULLET = "·"   # the ` · ` separator inside a tail segment
BAR_FULL = "▪"      # spinner runner cell
BAR_EMPTY = "·"     # spinner track cell
FAIL_MARK = "✗"

# Notice levels (B1 `notice(level, text)`) -> semantic role name. `muted` is
# the low-key paper-trail level: an auto-allow line is secondary text next to
# the tool box it explains, so it takes MUTED (B3's replacement for the raw
# `DIM` those lines used to carry), not a bright accent.
NOTICE_ROLES: dict[str, str] = {
    "info": "ACCENT",
    "muted": "MUTED",
    "warning": "WARNING",
    "error": "ERROR",
    "success": "SUCCESS",
}

_PREFIX = "  "   # every box in nx is inset two columns


def role(role_name: str) -> str:
    """The active theme's SGR color for a semantic role name (e.g. "ACCENT").

    A thin indirection so components can name roles as *data* (a level -> role
    table) instead of attribute lookups at every call site. An unknown name
    degrades to ACCENT rather than raising: display must never break a turn.
    """
    return getattr(C, role_name, C.ACCENT)


# --- render mode: the one switch ---------------------------------------------
def live_open() -> bool:
    """Is a rich Live draining the shared console right now?"""
    try:
        from ._rich import live_active
        return live_active()
    except Exception:
        return False


def emit(text: str = "", *, inplace: bool = False, end: str = "\n") -> None:
    """Write ONE already-rendered line through the right channel (the mode
    branch, in one place).

    * Live up — `_rich.cprint`, whichever shape (rich composes the line above
      the frame; an in-place erase would wipe the Live's own frame, so the
      `CLEAR_LINE` lead is dropped);
    * no Live — an in-place line is a raw `sys.stdout.write` + flush (the
      classic byte-identical repaint), anything else is a plain `print`.

    Both branches take `_rich.cprint`'s `write_lock()` on the Live path; the
    raw path needs no lock (it is the pre-0.128 behaviour).
    """
    import sys
    if inplace and not live_open():
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    from ._rich import cprint
    cprint(text.replace(CLEAR_LINE, "") if inplace else text, end=end)


def commit_line(text: str = "", *, end: str = "\n") -> None:
    """`emit()` for the common case: one whole line, never an in-place erase."""
    emit(text, inplace=False, end=end)


# --- panel --------------------------------------------------------------------
def panel_header(title: str, accent: Optional[str] = None, *,
                 prefix: str = _PREFIX, role_name: str = "ACCENT") -> str:
    """A box's opening line: `╭─ title` in the accent color."""
    accent = accent if accent is not None else role(role_name)
    return f"{accent}{prefix}{BOX_TL}{BOX_H} {title}{RESET}"


def panel_row(text: str, accent: Optional[str] = None, *,
              prefix: str = _PREFIX, role_name: str = "ACCENT",
              body_role: Optional[str] = None) -> str:
    """One body line of a box: `│ text`. `body_role` (a semantic role name)
    colors `text` for callers whose body is not already styled."""
    accent = accent if accent is not None else role(role_name)
    if body_role:
        text = f"{role(body_role)}{text}{RESET}"
    return f"{accent}{prefix}{BOX_V} {RESET}{text}"


def panel_footer(accent: Optional[str] = None, *, prefix: str = _PREFIX,
                 text: str = "", role_name: str = "ACCENT") -> str:
    """A box's closing line: `╰─` plus an optional trailer.

    The trailer is metadata about the box (a collapse count, `(escaped)`), so it
    takes the MUTED role (B3's replacement for the raw DIM these trailers used).
    """
    accent = accent if accent is not None else role(role_name)
    tail = f" {role('MUTED')}{text}{RESET}" if text else ""
    return f"{accent}{prefix}{BOX_BL}{BOX_H}{RESET}{tail}"


def panel(title: str, lines: Iterable[str], accent: Optional[str] = None, *,
          prefix: str = _PREFIX, role_name: str = "ACCENT",
          body_role: Optional[str] = None, footer: str = "") -> list[str]:
    """A whole box as a list of ready-to-`emit` lines.

    `accent` (an SGR string) wins when a caller needs a per-call hue — the tool
    box colors memory / sub-agent / deletion calls differently — otherwise the
    `role_name` role is used. Body lines are passed through verbatim unless
    `body_role` is given, in which case each is wrapped in that role's color
    (pre-styled bodies such as todo_write pass `body_role=None`).
    """
    out = [panel_header(title, accent, prefix=prefix, role_name=role_name)]
    for line in lines:
        out.append(panel_row(line, accent, prefix=prefix, role_name=role_name,
                             body_role=body_role))
    out.append(panel_footer(accent, prefix=prefix, text=footer,
                            role_name=role_name))
    return out


# --- notice -------------------------------------------------------------------
def notice(level: str, text: str, *, end: str = "\n") -> None:
    """One human-facing line, colored by `level` (a NOTICE_ROLES key:
    info/warning/error/success). `level="plain"` (or any unknown level) emits
    `text` verbatim, for messages that already carry their own decoration —
    that is the A3 `_ansi.notice` behaviour, byte for byte.

    This is the single emission point for the ad-hoc `print()` feedback calls
    in the REPL paths (UIUX_SPEC A3/B1). stdout, flushed.
    """
    if level not in NOTICE_ROLES:
        _A.notice(text) if end == "\n" else commit_line(text, end=end)
        return
    commit_line(f"{role(NOTICE_ROLES[level])}{text}{RESET}", end=end)


# --- key/value rows -----------------------------------------------------------
def kv_line(pairs: Iterable[tuple[str, str]], *, sep: str = " · ",
            prefix: str = _PREFIX, footer_glyph: bool = True) -> str:
    """A footer-style row: `╰ 12 tok · 4.2s wall · 8k/32k ctx`.

    `pairs` is (key, value), key first — the shape every nx status row uses
    (`jobs 1`, `steer 2`, `12 tok`). The key takes the MUTED role (UIUX_SPEC
    B3: the semantic role, not a raw `DIM` wrap — so a theme can recolor
    metadata independently of the spinners/hints that legitimately use DIM as
    *mode*); the value is emitted verbatim, because a value's own coloring is a
    per-value decision the caller keeps (the ctx field's utilization hue, a
    status badge). An empty key renders the value alone. `footer_glyph=False`
    drops the `╰ ` lead for a bare row (the status rail's cells)."""
    muted = role("MUTED")
    body = sep.join(f"{muted}{k} {RESET}{v}" if k else v for k, v in pairs)
    lead = f"{muted}{prefix}{BOX_BL} {RESET}" if footer_glyph else ""
    return f"{lead}{body}"


# --- diff ---------------------------------------------------------------------
def diff(old: str, new: str, context: int = DIFF_CONTEXT_LINES) -> str:
    """The approval-prompt colored unified diff of `old` -> `new`.

    Added lines take the DIFF_ADD role (green by default), removed lines
    DIFF_REMOVE (red), hunk headers and context DIM, with old/new line numbers
    prefixed. Returns "" when the two texts are identical. Display-only: this
    never reaches the model (`_approval._diff_hunk` is that path).
    """
    import difflib
    import re
    if old == new:
        return ""
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    lines = difflib.unified_diff(old.splitlines(), new.splitlines(),
                                 n=context, lineterm="")
    out: list[str] = []
    old_no = new_no = 0
    for line in lines:
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("@@"):
            m = hunk_re.match(line)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(2))
            out.append(f"{DIM}{line}{RESET}")
        elif line.startswith("+"):
            out.append(f"{C.DIFF_ADD}{'':>4} {new_no:>4} {line}{RESET}")
            new_no += 1
        elif line.startswith("-"):
            out.append(f"{C.DIFF_REMOVE}{old_no:>4} {'':>4} {line}{RESET}")
            old_no += 1
        else:
            out.append(f"{DIM}{old_no:>4} {new_no:>4} {line}{RESET}")
            old_no += 1
            new_no += 1
    return "\n".join(out)


# --- stream tails -------------------------------------------------------------
def stream_tail(text: str, width: int, *, inplace: bool = False,
                plain: bool = False) -> str:
    """A spinner/log tail clamped to `width` VISIBLE columns, sanitized.

    ANSI and control bytes are stripped and whitespace collapsed first: tails
    carry untrusted bytes (command output, tool notes) and one raw ESC or
    newline smears the row it lands on.

    * `plain=True` — return the sanitized, clipped PLAIN text with no color
      and no erase prefix. This is the shape a spinner LABEL needs (the label
      is re-styled by its own render, and the tail may be a whole command's
      log line, not a full status row).
    * `inplace=True` — lead with `CLEAR_LINE` for the spinner's self-repaint;
      callers hand the result to `emit()`.

    The clamp is `_progress.clip_line` — the one truncation marker the spinner
    budget has always used (`...`), so moving the clamp here does not change a
    single rendered line.
    """
    from ._ansi import strip_ansi_for_display
    from ._progress import clip_line
    clean = " ".join(strip_ansi_for_display(text or "").split())
    clean = clip_line(clean, width) if width > 0 else clean
    if plain:
        return clean
    base = f"{DIM}{clean}{RESET}"
    return f"{CLEAR_LINE}{base}" if inplace else base


def status_line(label: str, *, elapsed: Optional[str] = None,
                counter: Optional[str] = None, note: str = "",
                role_name: str = "ACCENT") -> str:
    """A one-line tool/step status: `label · 4.2s · 2/3 · note` (label in the
    role color, tail DIM). The segment vocabulary lives here so the spinner
    tail and the tool-status line cannot drift apart.
    """
    parts = [f"{role(role_name)}{label}{RESET}"]
    tail = [p for p in (elapsed, counter, note) if p]
    if tail:
        parts.append(f"{DIM}{(' ' + TAIL_BULLET + ' ').join(tail)}{RESET}")
    return " ".join(parts)


__all__ = [
    "BOX_TL", "BOX_TR", "BOX_BL", "BOX_BR", "BOX_H", "BOX_V",
    "BOX_BRANCH_MID", "BOX_BRANCH_END", "BOX_GLYPHS",
    "TAIL_BULLET", "BAR_FULL", "BAR_EMPTY", "FAIL_MARK",
    "NOTICE_ROLES",
    "role", "live_open", "emit", "commit_line",
    "panel", "panel_header", "panel_row", "panel_footer",
    "notice", "kv_line", "diff", "stream_tail", "status_line",
]
