"""Right-hand status rail for the /tabs live region (SPEC_PANEL.md, 0.127.0).

Phase 1 of the staged status-panel path: this module owns the panel's
*content* only — which cells exist, from which snapshot data, and how they
render. Painting stays in `_surfaces._live_frame`, so the live region keeps
one region, one erase arithmetic, and one `write_lock()` acquisition per tick.

Cells (phase 1, all data already collected): ctx utilization, alive bg-job
count, alive async-agent count, pending steering lines, and (best-effort)
active cross-process file claims. Gated by NX_PANEL / NX_PANEL_WIDTH and an
in-process flag `/panel` toggles.
"""
from __future__ import annotations

import os
import shutil

from ._ansi import DIM, RESET, C
from . import _state as st

# In-process override for the env default (NX_PANEL). `/panel` flips this via
# set_panel_enabled(); None = "no toggle yet, read the env".
_OVERRIDE: "bool | None" = None

_TRUTHY = ("1", "true", "yes")


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


def panel_enabled() -> bool:
    """Is the status rail on? NX_PANEL=1/true/yes is the default; `/panel`
    (set_panel_enabled) overrides in-process. Silently off when the terminal
    is too narrow for chat + rail: cols - PW must stay >=
    PANEL_MIN_CHAT_COLS (nx/_layout.py)."""
    from ._layout import PANEL_MIN_CHAT_COLS
    if not panel_wants_on():
        return False
    try:
        cols = shutil.get_terminal_size().columns
    except OSError:
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
    """Truncate to `w` VISIBLE columns: ANSI SGR sequences don't count
    toward the width and are never cut mid-sequence (a raw [:w] slice on a
    styled cell would break the escape and leak a literal `m`)."""
    if len(s) <= w:
        return s
    out: list[str] = []
    n = 0
    i = 0
    while i < len(s):
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j == -1:
                break
            out.append(s[i:j + 1])
            i = j + 1
            continue
        if n >= w:
            break
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out)


def _ctx_cell() -> str:
    """`<cur>/<max> ctx`, utilization-colored (same rule as the per-turn
    footer's ctx field: green <30%, yellow <60%, red at or above). Unknown
    max degrades to `<cur> ctx` with no color and no probe kick — the rail
    is passive."""
    cur = _abbr(getattr(st, "_LAST_PROMPT_TOKENS", 0) or 0)
    src = st.ACTIVE
    mx = getattr(src, "_context_window", None) if src is not None else None
    if not isinstance(mx, int) or mx <= 0:
        return f"{cur} ctx"
    ratio = (getattr(st, "_LAST_PROMPT_TOKENS", 0) or 0) / mx
    if ratio < 0.30:
        color = C.SUCCESS
    elif ratio < 0.60:
        color = C.WARNING
    else:
        color = C.ERROR
    return (f"{color}{cur}{RESET}{DIM}/{C.STATUS_BAR}{_abbr(mx)}"
            f"{RESET}{DIM} ctx{RESET}")


def panel_cells(width: int, rows: list[dict]) -> list[str]:
    """The rail's cells (<=5, top-aligned with the frame's lines), styled.

    `rows` is the `_surfaces_snapshot()` row list (kind main|bg|agent,
    alive, status, id). Pure w.r.t. its argument; reads ctx/steering/claim
    state from their owning modules. Each cell is truncated to `width`."""
    alive = [r for r in rows if r.get("alive")]
    cells = [
        _ctx_cell(),
        f"jobs {sum(1 for r in alive if r.get('kind') == 'bg')}",
        f"agents {sum(1 for r in alive if r.get('kind') == 'agent')}",
    ]
    try:
        from ._surfaces import steer_pending
        n_steer = steer_pending()
        if n_steer:  # 0 hidden: an empty queue is not a status
            cells.append(f"steer {n_steer}")
    except Exception:
        pass
    try:
        # Best-effort claims count: _fileclaim has no public count getter
        # today, so probe for one and skip the cell when there isn't.
        from . import _fileclaim as _fc
        getter = getattr(_fc, "active_claims", None)
        if callable(getter):
            n_claims = int(getter())
            if n_claims:
                cells.append(f"claims {n_claims}")
    except Exception:
        pass
    return [_clip(c, width) for c in cells[:5]]
