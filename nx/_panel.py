"""Right-hand status rail for the /tabs live region (SPEC_PANEL.md).

This module owns the panel's *content* only — which cells exist, from which
snapshot data, and how they render. Painting stays in `_surfaces._live_frame`,
so the live region keeps one region, one erase arithmetic, and one
`write_lock()` acquisition per tick.

SEVEN fixed slots, in this order, each hidden when it has nothing to say
(0.140.0; the rail is a passive read surface — no forks per tick, no probes, no
network, and the live aggregating state lives here while per-turn facts stay in
the turn footer):

  1. ctx      utilization + `▰▱` bar (rail narrower than 20 cells) or `cur/max`
  2. mode     permission/approval mode (yolo / prompt-all / auto:<level>)
  3. branch   git branch + dirty count, from a ~2s TTL-cached poll
  4. jobs     alive background jobs
  5. agents   alive async sub-agents
  6. steer    queued steering lines
  7. cost     session $ — only when at least one source is actually priced
     claims   active cross-process file claims (_fileclaim.active_claims)

Every cell is measured and clipped in display COLUMNS by the one ANSI-aware
trio in `_ansi` (clip_display/display_width), so a CJK instruction or an SGR
sequence can never shift the rail's `│` separator or cut an escape in half.

Gated by NX_PANEL / NX_PANEL_WIDTH and an in-process flag `/panel` toggles; on a
terminal too narrow for the rail the same cells degrade to ONE bottom line
(`degraded_line()`, rail-off / compact-on).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

from ._ansi import RESET, C, clip_display, display_width
from ._render import role as _role
from . import _state as st

# In-process override for the env default (NX_PANEL). `/panel` flips this via
# set_panel_enabled(); None = "no toggle yet, read the env".
_OVERRIDE: "bool | None" = None

_TRUTHY = ("1", "true", "yes")

# Below this rail width the ctx cell renders as a `▰▱` bar instead of cur/max.
BAR_MODE_WIDTH = 20

# Degradation ladder (0.140.0): when the rail can't fit (cols - PW <
# PANEL_MIN_CHAT_COLS, nx/_layout.py) but the user wants it on AND the terminal
# is at least this wide, the same cells render as ONE bottom status line
# (rail-off, compact-on). Narrower than this, one line of status is itself
# noise and the frame stays byte-identical to panel-off.
DEGRADED_MIN_COLS = 46

# The polled-cell TTLs: the rail repaints up to ~10 Hz (owner tick), so a git
# fork or a sqlite open per repaint is out of the question. Values are refreshed
# at most this often; tests monkeypatch `_panel.time`.
BRANCH_TTL = 2.0
CLAIMS_TTL = 2.0

_branch_cache: "tuple[float, str | None]" = (0.0, None)
_claims_cache: "tuple[float, int]" = (0.0, 0)


def ensure_region() -> None:
    """Open the live region at TURN START when the rail is enabled (0.127.1):
    the region is a mid-turn construct owned by whoever holds stdin
    (EVENTSURFACE_SPEC §2/§3), so it is opened here — never at the prompt.
    Idempotent; refuses nothing if _surfaces refuses."""
    if not panel_enabled():
        return
    from . import _surfaces
    if not _surfaces.dashboard_open():
        _surfaces.open_dashboard()


def panel_width() -> int:
    """NX_PANEL_WIDTH: the rail's column budget, default 24, clamped
    PANEL_MIN..PANEL_MAX (nx/_layout.py)."""
    from ._env import _env_int
    from ._layout import PANEL_MAX_COLS, PANEL_MIN_COLS
    return max(PANEL_MIN_COLS, min(PANEL_MAX_COLS, _env_int("NX_PANEL_WIDTH", 24)))


def _term_cols() -> int | None:
    """Terminal width, or None where it can't be read (no tty)."""
    try:
        return shutil.get_terminal_size().columns
    except OSError:
        return None


def panel_enabled() -> bool:
    """Is the status rail on? NX_PANEL=1/true/yes is the default; `/panel`
    (set_panel_enabled) overrides in-process. Silently off when the terminal
    is too narrow for chat + rail: cols - PW must stay >=
    PANEL_MIN_CHAT_COLS (nx/_layout.py)."""
    from ._layout import PANEL_MIN_CHAT_COLS
    if not panel_wants_on():
        return False
    cols = _term_cols()
    if cols is None:
        return False
    return cols - panel_width() >= PANEL_MIN_CHAT_COLS


def panel_wants_on() -> bool:
    """The user's intent flag (override or NX_PANEL env), WITHOUT the
    terminal-width guard. The `/panel` toggle keys off this, not
    panel_enabled(): on a narrow terminal the guard keeps panel_enabled()
    False even after an explicit on, which would make /panel print 'on'
    forever instead of toggling."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    return os.environ.get("NX_PANEL", "").strip().lower() in _TRUTHY


def set_panel_enabled(on: bool) -> None:
    """Set the in-process rail flag (the `/panel` toggle)."""
    global _OVERRIDE
    _OVERRIDE = bool(on)


def _abbr(n: int) -> str:
    """Compact token count: 832 -> '832', 1500 -> '1.5k', 78825 -> '78k',
    1234567 -> '1.2M'. Re-implemented from _metrics._abbr (lowercase k) so the
    rail stays cheap: importing _turn_footer would drag in _turn for one
    formatting helper."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n // 1000}k"
    if n < 10_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n // 1_000_000}M"


def _clip(s: str, w: int) -> str:
    """Truncate to `w` VISIBLE columns. Kept as the module-local name the rail
    has always used; the implementation is the shared `_ansi.clip_display`
    (ANSI-aware, wcwidth-accurate: SGR never counts and is never cut in half,
    a wide char counts 2)."""
    return clip_display(s, w)


def _muted() -> str:
    """The MUTED role's SGR (UIUX_SPEC B3), read at call time so a `/theme`
    switch lands on the next repaint."""
    return _role("MUTED")


def _elide_left(s: str, w: int) -> str:
    """Left-elide to `w` cells as `…tail` — a branch name's tail is the
    informative end (`…/feature/fix-rail` beats `feature/fix…`)."""
    if display_width(s) <= w:
        return s
    if w <= 1:
        return "…"[:w]
    # Drop from the front until it fits behind the ellipsis. One pass over the
    # characters is fine: a branch name is short and this runs at most once
    # per TTL refresh (the elided value is cached, not recomputed per tick).
    keep: list[str] = []
    n = 0
    for ch in reversed(s):
        cw = display_width(ch)
        if n + cw > w - 1:
            break
        keep.append(ch)
        n += cw
    return "…" + "".join(reversed(keep))


# --- 1. ctx ------------------------------------------------------------------

def _utilization() -> "tuple[int, int, float] | None":
    """(used tokens, window, ratio), or None when the window is unknown — the
    rail is passive, so an unknown max degrades instead of kicking a probe."""
    used = getattr(st, "_LAST_PROMPT_TOKENS", 0) or 0
    src = st.ACTIVE
    mx = getattr(src, "_context_window", None) if src is not None else None
    if not isinstance(mx, int) or mx <= 0:
        return None
    return used, mx, used / mx


def _util_color(ratio: float) -> str:
    """Same rule as the turn footer's ctx field: green <30%, yellow <60%,
    red at or above."""
    if ratio < 0.30:
        return C.SUCCESS
    if ratio < 0.60:
        return C.WARNING
    return C.ERROR


def _ctx_cell(width: int) -> str:
    """`<cur>/<max> ctx`, utilization-colored; rail narrower than
    BAR_MODE_WIDTH renders the `▰▱` bar + % instead (bar + `cur/max` is more
    than a 12-cell rail can say, and the bar reads at a glance). At the red
    threshold a `/compact` hint is appended when the cell has room for it.
    Unknown max degrades to `<cur> ctx` with no color and no probe kick."""
    used = getattr(st, "_LAST_PROMPT_TOKENS", 0) or 0
    cur = _abbr(used)
    u = _utilization()
    if u is None:
        return f"{cur} ctx"
    _, mx, ratio = u
    color = _util_color(ratio)
    red = ratio >= 0.60
    if width < BAR_MODE_WIDTH:
        n = max(3, min(8, width - 5))
        filled = max(0, min(n, round(ratio * n)))
        bar = "▰" * filled + "▱" * (n - filled)
        cell = f"{color}{bar} {round(ratio * 100):>3d}%{RESET}"
        hint = f"{_muted()} /compact{RESET}"
        if red and display_width(cell) + display_width(" /compact") <= width:
            cell += hint
        return cell
    head = (f"{color}{cur}{RESET}{_muted()}/{C.STATUS_BAR}{_abbr(mx)}"
            f"{RESET}{_muted()} ctx{RESET}")
    if red:
        hint = f"{_muted()} /compact{RESET}"
        if display_width(head) + display_width(" /compact") <= width:
            head += hint
    return head


# --- 2. mode -----------------------------------------------------------------

def _mode_cell() -> str | None:
    """The permission/approval mode: `yolo`, `prompt-all`, or `auto:<level>`
    (st.YOLO / st.APPROVE_LEVEL — the two knobs `_approval` prompts on and
    /approval + /yolo set). Hidden when the state is a type this build doesn't
    understand, so a future mode surface degrades to "no cell" rather than a
    wrong label. Read-only: a plain attribute read, never a config-file read."""
    yolo = getattr(st, "YOLO", False)
    lvl = getattr(st, "APPROVE_LEVEL", None)
    if yolo:
        value = "yolo"
    elif lvl is None:
        value = "prompt-all"
    elif isinstance(lvl, str):
        value = f"auto:{lvl}"
    else:
        return None
    return kv_cell("mode", value)


# --- 3. branch ---------------------------------------------------------------

def _poll_branch() -> str | None:
    """Fork-scope poll: `name[ +dirty]` for the current repo, None when there
    isn't one. ONE `git status --porcelain --branch` (branch line + dirty count
    together) plus a `rev-parse` only for a detached HEAD. Never called from a
    repaint: `_branch_cell` serves the TTL cache."""
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        pr = subprocess.run(
            ["git", "status", "--porcelain", "--branch",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=2.0, env=env)
        if pr.returncode != 0:
            return None
        head = ""
        dirty = 0
        for line in pr.stdout.splitlines():
            if line.startswith("## "):
                head = line[3:].strip()
            elif line.strip():
                dirty += 1
        if not head:
            return None
        name = head.split("...")[0].split(" ")[0]
        if name in ("HEAD", "(no", "No"):
            rp = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=2.0,
                                env=env)
            if rp.returncode != 0 or not rp.stdout.strip():
                return None
            name = rp.stdout.strip()
        return f"{name} +{dirty}" if dirty else name
    except Exception:
        return None  # not a repo / no git / slow: the cell just stays hidden


def _branch_value() -> str | None:
    """The TTL-cached branch poll (module cache, monkeypatchable clock)."""
    global _branch_cache
    now = time.monotonic()
    stamp, val = _branch_cache
    if now - stamp >= BRANCH_TTL:
        val = _poll_branch()
        _branch_cache = (now, val)
    return val


def _branch_cell(width: int) -> str | None:
    """`branch <name> [+dirty]`, left-elided to the rail width; hidden outside
    a git repo."""
    val = _branch_value()
    if not val:
        return None
    name, _, dirty = val.partition(" +")
    # budget: "branch " + name + " +<dirty>" (the key/badge are fixed here so
    # the elide target is exact rather than a guess at kv_line's layout)
    tail = f" +{dirty}" if dirty else ""
    room = width - display_width("branch ") - display_width(tail)
    value = _elide_left(name, max(1, room)) + tail
    return kv_cell("branch", value)


# --- 4/5/6. jobs · agents · steer -------------------------------------------

def _count_cell(key: str, n: int) -> str | None:
    """A `key N` cell, hidden at 0 — an empty queue is not a status."""
    return kv_cell(key, str(n)) if n else None


def _steer_count() -> int:
    try:
        from ._surfaces import steer_pending
        return steer_pending()
    except Exception:
        return 0


# --- 7. cost -----------------------------------------------------------------

def _cost_cell() -> str | None:
    """Session $ via `nx/_cost.py`'s own pricing — hidden entirely unless at
    least one used source is priced (an unpriced session must not read as
    `$0.00`: blank ≠ 0). Read-only over the in-process metrics; no env write,
    no network."""
    metrics = getattr(st, "_METRICS", None)
    buckets = getattr(metrics, "by_source", None) or {}
    if not buckets:
        return None
    from ._cost import _fmt_cost, _source_cost, st_active_model
    total = 0.0
    priced = False
    try:
        for name, bucket in buckets.items():
            c = _source_cost(bucket, name, st_active_model(name))
            if c is not None:
                priced = True
                total += c
    except Exception:
        return None
    return kv_cell("cost", _fmt_cost(total)) if priced else None


# --- claims ------------------------------------------------------------------

def _claims_store_count() -> int:
    """One read of the claim store (the seam the TTL cache wraps: kept a named
    function so the poll is patchable/mockable without a live session DB)."""
    from . import _fileclaim
    return int(_fileclaim.active_claims())


def _claims_count() -> int:
    """TTL-cached `_fileclaim.active_claims()` (the store is SQLite; the rail
    must not open it per repaint)."""
    global _claims_cache
    now = time.monotonic()
    stamp, n = _claims_cache
    if now - stamp >= CLAIMS_TTL:
        try:
            n = _claims_store_count()
        except Exception:
            n = 0
        _claims_cache = (now, n)
    return n


def kv_cell(key: str, value: str) -> str:
    """One rail cell: `key value` through the shared `kv_line` component
    (UIUX_SPEC B1), so the rail's rows and the turn footer's rows share one
    implementation and one glyph/role vocabulary. Lazily imported: `_render`
    imports `_ansi`, and `_panel` is loaded by `_surfaces` mid-import."""
    from ._render import kv_line
    return kv_line([(key, value)], footer_glyph=False)


def panel_cells(width: int, rows: list[dict]) -> list[str]:
    """The rail's cells (<=7, top-aligned with the frame's lines), styled.

    `rows` is the `_surfaces_snapshot()` row list (kind main|bg|agent, alive,
    status, id). Pure w.r.t. its argument; reads ctx/mode/steering/cost/claim
    state from their owning modules and the TTL-cached git poll. Each cell is
    truncated to `width` display columns; a slot with nothing to say is
    dropped, so the ORDER of the surviving cells is the slot order above."""
    alive = [r for r in rows if r.get("alive")]
    cells = [
        _ctx_cell(width),
        _mode_cell(),
        _branch_cell(width),
        _count_cell("jobs", sum(1 for r in alive if r.get("kind") == "bg")),
        _count_cell("agents", sum(1 for r in alive if r.get("kind") == "agent")),
        _count_cell("steer", _steer_count()),
        _cost_cell(),
        _count_cell("claims", _claims_count()),
    ]
    return [_clip(c, width) for c in cells if c][:7]


# --- degradation ladder ------------------------------------------------------

def degraded_line(rows: list[dict], cols: int | None = None) -> str | None:
    """The compact bottom status line for a terminal that can't fit the rail:
    None when the rail is off, when the user never asked for it, when the rail
    actually FITS (`panel_enabled()`), or when the terminal is narrower than
    DEGRADED_MIN_COLS — below that one line of status is itself noise.

    Only reached when the rail can't fit but the user wants it on
    (`panel_wants_on()` and not `panel_enabled()`), so `panel off` stays
    byte-identical: `_live_frame` returns before ever calling this."""
    from ._layout import PANEL_MIN_CHAT_COLS
    if not panel_wants_on() or panel_enabled():
        return None
    if cols is None:
        cols = _term_cols()
    if cols is None or cols < DEGRADED_MIN_COLS:
        return None
    if cols - panel_width() >= PANEL_MIN_CHAT_COLS:
        return None  # the full rail fits; _live_frame already drew it
    cells = panel_cells(max(DEGRADED_MIN_COLS, cols - 4), rows)
    if not cells:
        return None
    body = clip_display(f"{_muted()} · {RESET}".join(cells), max(1, cols - 1))
    return body if body else None
