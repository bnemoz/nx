"""Shared mutable process state.

Historically all of this lived as bare module globals in `nx/__init__.py`,
mutated via `global X` from functions defined in that same file. Once those
functions were split across several `nx/_*.py` submodules, `global X` in
one of those submodules no longer reaches back into `__init__.py`'s
namespace — each module has its own globals dict. So instead every module
that reads or writes one of these names imports this module (`from . import
_state as st`) and does `st.ACTIVE = ...` / `st.ACTIVE` — one shared home,
plain attribute access, no magic.

`nx/__init__.py` proxies attribute access for these names through to
this module (see the `_StateProxyModule` class at the bottom of
`__init__.py`), so `import nx as m; m.ACTIVE` / `m.ACTIVE = x` — which
the test suite relies on extensively — keeps working unchanged and stays in
sync with whatever any submodule does via `st.ACTIVE`.
"""

# --- sources / active model connection --------------------------------------
SOURCES = {}        # name → Source
SOURCE_ORDER = []   # preserve definition order for /source listing
ACTIVE = None       # current Source

# `client` and `MODEL` mirror the active source; switch_source() reassigns
# both. Every function that needs them does a call-time lookup (via this
# module), so a mid-session swap is picked up instantly.
client = None
MODEL = None

# Updated by model_turn() on every turn that reports a prompt-token count.
# Read by _maybe_autocompress() to decide whether to silently auto-compress.
_LAST_PROMPT_TOKENS = 0

# --- approval gating ---------------------------------------------------------
YOLO = False
APPROVE_LEVEL = None
DEFAULT_APPROVE_LEVEL = None


def toggle_yolo():
    """Flip YOLO mode, mirroring /yolo. Shared by the Shift+Tab handlers in
    _chatbox.py's editor and _approval.py's approval prompt so both stay in
    sync instead of each re-implementing the toggle independently."""
    global YOLO, APPROVE_LEVEL
    YOLO = not YOLO
    APPROVE_LEVEL = None if YOLO else DEFAULT_APPROVE_LEVEL
    return YOLO

# Project root used by the risk classifier's path-scoping. Set once at
# startup by _detect_project_root() in _risk.py / _approval.py.
PROJECT_ROOT = None

# Set by run_tool() while a tool body is executing; read by _confirm() to
# pause/resume the spinner around approval prompts / risk-classifier calls.
_ACTIVE_SPINNER = None

RISK_CONNECTION_RETRIES = None
RISK_CONNECTION_RETRY_SECONDS = None

# --- interrupt watcher --------------------------------------------------------
# (the Event objects themselves are created once in _interrupt.py; this
# module only holds the thread handle, which does get reassigned.)
_INTERRUPT_THREAD = None

# Text the user typed *after* pressing Esc, while the watcher was still armed
# and draining stdin during the disarm handshake. Written only by the watcher
# thread (in its inner `finally`, after _INTERRUPT_RESTORED is set — so by the
# time main reads these the write has already happened-before). If the user
# hit Enter before disarm completed, _INTERRUPT_REDIRECT_SUBMITTED is True and
# model_turn() injects the text directly as the next user turn instead of
# dropping to a blank prompt. Otherwise it's partial text, consumed as a
# prefill for the next read_multiline() call in the REPL.
_INTERRUPT_REDIRECT_TEXT = ""
_INTERRUPT_REDIRECT_SUBMITTED = False

# Set by model_turn() when an Esc-interrupt captured a partial (no Enter yet)
# redirect line. The REPL consumes-and-clears this as the `initial` prefill
# for its next read_multiline() call so the user's half-typed line reappears
# in the prompt instead of vanishing.
_INTERRUPT_REDIRECT_PREFILL = ""

# --- background run_bash notice ----------------------------------------------
_SESSION_HAS_BACKGROUNDED = False
_PENDING_BACKGROUND_NOTICE = None

# --- lazy multi_edit unlock ---------------------------------------------------
_CONSECUTIVE_EDITS = 0
_MULTI_EDIT_UNLOCKED = False

# --- recovery / turn-loop tuning (env-configurable, test-monkeypatchable) ----
# Real defaults (env-resolved) are assigned onto these names at import time
# by the module that owns each setting (_turn.py, _compress.py) — this is
# just the shared home so a cross-module `st.NAME = x` mutation (tests, or
# /autocompress etc.) is visible everywhere.
REASONING_ONLY_RETRY_LIMIT = None
MALFORMED_STREAM_RETRY_LIMIT = None
FORCED_FINAL_MAX_TOKENS = None
EMPTY_TURN_RETRY_LIMIT = None
REASONING_ONLY_CHAR_LIMIT = None
RECOVERY_TEMPERATURE = None
RECOVERY_TOP_P = None
MAX_MODEL_TURNS = None

# --- reasoning display ---------------------------------------------------
# Whether live reasoning_content is streamed to the terminal as it arrives.
# Toggled by /reasoning (persisted to ~/.env as NX_REASONING_VISIBLE, same
# mechanism as /theme); the full text of the most recent turn's reasoning
# is always captured into _LAST_REASONING regardless, so /reasoning show
# works even right after switching into collapsed mode.
import os as _os
REASONING_VISIBLE = _os.environ.get("NX_REASONING_VISIBLE", "1") not in ("0", "false", "off")
_LAST_REASONING = ""

# --- auto context-compression -------------------------------------------------
AUTOCOMPRESS_PERCENT = None
AUTOCOMPRESS_DEFAULT = None
COMPRESS_KEEP = None

# --- usage metrics -------------------------------------------------------------
_METRICS = None  # set to a _Metrics() instance in _metrics.py

# --- write/edit checkpoint stack (for /undo) -----------------------------------
# Populated by run_tool() after a successful write_file/edit_file/multi_edit;
# see _undo.py for the entry shape and undo_last()'s restore logic.
_UNDO_STACK = []

# --- read-only tool result cache ----------------------------------------------
# Keyed by (tool name, json.dumps(sorted args)) -> tool result string. Populated
# and consulted only by run_tool() for idempotent reads (read_file/list_dir);
# invalidated per-path on write_file/edit_file/multi_edit and cleared entirely
# on run_bash (which can touch anything). See run_tool()'s docstring note.
_TOOL_RESULT_CACHE = {}
