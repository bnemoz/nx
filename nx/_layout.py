"""Layout constants (UIUX_SPEC Phase A2, 0.134.0).

One named home for the widths/limits that were scattered as bare literals
across the render modules. PURE MOVE: every value is the pre-existing one,
no retuning, no new env vars (env overrides only where one already exists,
e.g. NX_PANEL_WIDTH clamps against the min/max here).

The spinner budget constants (LABEL_CELLS / TAIL_BUDGET / MIN_TAIL /
MAX_TAIL / MAX_NOTE / MAX_NOTE_SHOWN) stay DEFINED in `_progress.py` —
other modules import them from there (moved-names rule) — and this module
re-exports them so layout reads are complete from one place.
"""
from __future__ import annotations

from ._progress import (LABEL_CELLS, MAX_NOTE, MAX_NOTE_SHOWN, MAX_TAIL,
                        MIN_TAIL, TAIL_BUDGET)

# Terminal fallback: columns used whenever get_terminal_size() can't see a
# tty (chatbox frame render/erase, surfaces' width probe). Pairs with the
# standard 24-row fallback already passed to get_terminal_size().
TERM_FALLBACK_COLS = 80
TERM_FALLBACK_ROWS = 24
# Surfaces' minimum usable terminal width (below it, clamp up).
TERM_MIN_COLS = 40

# Reasoning-only step divider in the turn footer (_turn_footer.py): the dim
# rule width above the "(step: reasoning ...)" hint.
REASONING_DIVIDER_CELLS = 46

# Status rail (_panel.py): NX_PANEL_WIDTH default 24 clamped to
# [PANEL_MIN_WIDTH, PANEL_MAX_WIDTH]; the rail only renders when the
# terminal leaves at least PANEL_MIN_CHAT_COLS for the chat itself.
PANEL_MIN_WIDTH = 12
PANEL_MAX_WIDTH = 40
PANEL_MIN_COLS = PANEL_MIN_WIDTH
PANEL_MAX_COLS = PANEL_MAX_WIDTH
PANEL_MIN_CHAT_COLS = 60

# Approval diff preview (_approval.py): context lines around each hunk.
DIFF_CONTEXT_LINES = 2

# Tool-result box preview (_dispatch.py): display-only line cap before the
# "... [n more lines]" notice.
TOOL_PREVIEW_LINES = 8

# Chatbox dropdown: rows rendered before the list scrolls (_chatbox.py).
DROPDOWN_MAX_ROWS = 8

__all__ = [
    "LABEL_CELLS", "MAX_NOTE", "MAX_NOTE_SHOWN", "MAX_TAIL", "MIN_TAIL",
    "TAIL_BUDGET",
    "TERM_FALLBACK_COLS", "TERM_FALLBACK_ROWS", "TERM_MIN_COLS",
    "REASONING_DIVIDER_CELLS",
    "PANEL_MIN_WIDTH", "PANEL_MAX_WIDTH", "PANEL_MIN_COLS", "PANEL_MAX_COLS",
    "PANEL_MIN_CHAT_COLS",
    "DIFF_CONTEXT_LINES",
    "TOOL_PREVIEW_LINES",
    "DROPDOWN_MAX_ROWS",
]
