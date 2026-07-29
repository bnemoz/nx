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

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._metrics import _Metrics
    from ._openai_client import OpenAI
    from ._sources import Source
    from ._undo import Checkpoint

# --- sessions filesystem root -------------------------------------------------
# Set (and re-set on every /reload) by __init__.py's top-level assignment from
# NX_HOME. Kept here so _bash_jobs.py's background-log paths (which live under
# SESSION_HOME/bg-logs) stay in sync with a mid-test reload without a circular
# import back into __init__.py.
SESSION_HOME: str = os.path.expanduser("~/.nx")

# --- sources / active model connection --------------------------------------
SOURCES: dict[str, "Source"] = {}
SOURCE_ORDER: list[str] = []   # preserve definition order for /source listing
ACTIVE: "Source | None" = None

# `client` and `MODEL` mirror the active source; switch_source() reassigns
# both. Every function that needs them does a call-time lookup (via this
# module), so a mid-session swap is picked up instantly.
client: "OpenAI | None" = None
MODEL: str | None = None

# Updated by model_turn() on every turn that reports a prompt-token count.
# Read by _maybe_autocompress() to decide whether to silently auto-compress.
_LAST_PROMPT_TOKENS: int = 0

# --- approval gating ---------------------------------------------------------
YOLO: bool = False
APPROVE_LEVEL: str | None = None
DEFAULT_APPROVE_LEVEL: str | None = None


def toggle_yolo() -> bool:
    """Flip YOLO mode, mirroring /yolo. Shared by the Shift+Tab handlers in
    _chatbox.py's editor and _approval.py's approval prompt so both stay in
    sync instead of each re-implementing the toggle independently."""
    global YOLO, APPROVE_LEVEL
    YOLO = not YOLO
    APPROVE_LEVEL = None if YOLO else DEFAULT_APPROVE_LEVEL
    return YOLO


# Project root used by the risk classifier's path-scoping. Set once at
# startup by _detect_project_root() in _risk.py / _approval.py.
PROJECT_ROOT: str | None = None

# Set by run_tool() while a tool body is executing; read by _confirm() to
# pause/resume the spinner around approval prompts / risk-classifier calls.
_ACTIVE_SPINNER: Any | None = None

RISK_CONNECTION_RETRIES: int = 3
RISK_CONNECTION_RETRY_SECONDS: float = 1.0

# --- interrupt watcher --------------------------------------------------------
# (the Event objects themselves are created once in _interrupt.py; this
# module only holds the thread handle, which does get reassigned.)
_INTERRUPT_THREAD: threading.Thread | None = None

# Text the user typed *after* pressing Esc, while the watcher was still armed
# and draining stdin during the disarm handshake. Written only by the watcher
# thread (in its inner `finally`, after _INTERRUPT_RESTORED is set — so by the
# time main reads these the write has already happened-before). If the user
# hit Enter before disarm completed, _INTERRUPT_REDIRECT_SUBMITTED is True and
# model_turn() injects the text directly as the next user turn instead of
# dropping to a blank prompt. Otherwise it's partial text, consumed as a
# prefill for the next read_multiline() call in the REPL.
_INTERRUPT_REDIRECT_TEXT: str = ""
_INTERRUPT_REDIRECT_SUBMITTED: bool = False

# Set by model_turn() when an Esc-interrupt captured a partial (no Enter yet)
# redirect line. The REPL consumes-and-clears this as the `initial` prefill
# for its next read_multiline() call so the user's half-typed line reappears
# in the prompt instead of vanishing.
_INTERRUPT_REDIRECT_PREFILL: str = ""

# --- background run_bash notice ----------------------------------------------
_SESSION_HAS_BACKGROUNDED: bool = False
_PENDING_BACKGROUND_NOTICE: str | None = None

# --- multi_edit batching nudge ------------------------------------------------
_CONSECUTIVE_EDITS: int = 0
_MULTI_EDIT_NUDGED: bool = False
_PENDING_MULTI_EDIT_NOTICE: str | None = None

# --- recovery / turn-loop tuning (env-configurable, test-monkeypatchable) ----
# Real defaults (env-resolved) are assigned onto these names at import time
# by the module that owns each setting (_turn.py, _compress.py) — this is
# just the shared home so a cross-module `st.NAME = x` mutation (tests, or
# /autocompress etc.) is visible everywhere.
REASONING_ONLY_RETRY_LIMIT: int = 3
MALFORMED_STREAM_RETRY_LIMIT: int = 2
FORCED_FINAL_MAX_TOKENS: int = 2048
EMPTY_TURN_RETRY_LIMIT: int = 3
REASONING_ONLY_CHAR_LIMIT: int = 36000
RECOVERY_TEMPERATURE: float = 1.0
RECOVERY_TOP_P: float = 0.95
MAX_MODEL_TURNS: int = 200

# --- reasoning display ---------------------------------------------------
# Whether live reasoning_content is streamed to the terminal as it arrives.
# Toggled by /reasoning (persisted to ~/.env as NX_REASONING_VISIBLE, same
# mechanism as /theme); the full text of the most recent turn's reasoning
# is always captured into _LAST_REASONING regardless, so /reasoning show
# works even right after switching into collapsed mode.
REASONING_VISIBLE: bool = os.environ.get("NX_REASONING_VISIBLE", "1") not in ("0", "false", "off")
_LAST_REASONING: str = ""

# --- auto context-compression -------------------------------------------------
AUTOCOMPRESS_PERCENT: float = 85
AUTOCOMPRESS_DEFAULT: float = 85
COMPRESS_KEEP: int = 2

# --- usage metrics -------------------------------------------------------------
_METRICS: "_Metrics | None" = None  # set to a _Metrics() instance in _metrics.py

# --- native tool-calling confirmation ------------------------------------------
# Set True by _handle_native_tool_calls() the first time a turn actually comes
# back with structured `tool_calls` deltas (native mode). Read by
# _compress.py's _grammar_extra_body() to skip attaching our own <tool_call>
# text-tag GBNF grammar once native mode is confirmed: llama.cpp already
# derives its own argument-constraining grammar internally from `tools=`
# for templates it recognizes as native-tool-call-capable, and there is no
# client-observable trigger string to hook a *second*, argument-level grammar
# off of in that mode — sending our own `grammar` field would at best be
# redundant and at worst fight/override llama.cpp's own tool-calling grammar.
# Reset per source switch (see switch_source()) since a different endpoint
# may behave differently.
_NATIVE_TOOL_CALLS_CONFIRMED: bool = False

# --- write/edit checkpoint stack (for /undo) -----------------------------------
# Populated by run_tool() after a successful write_file/edit_file/multi_edit;
# see _undo.py for the entry shape and undo_last()'s restore logic.
_UNDO_STACK: list["Checkpoint"] = []

# --- read-only tool result cache ----------------------------------------------
# Keyed by (tool name, json.dumps(sorted args)) -> tool result string. Populated
# and consulted only by run_tool() for idempotent reads (read_file/list_dir);
# invalidated per-path on write_file/edit_file/multi_edit and cleared entirely
# on run_bash (which can touch anything). See run_tool()'s docstring note.
_TOOL_RESULT_CACHE: dict[tuple[str, str], tuple[str, str]] = {}

# --- todo list -----------------------------------------------------------------
# Whole-list state for the todo_write tool (see _todo.py). Rewritten wholesale
# on each call (no per-item id/merge logic — same convention as Claude Code's
# own TodoWrite), persisted into the session JSON's "todos" key so /resume
# restores it (see _repl_state.py's _save_current/_reload_metrics_totals).
_TODOS: list[dict[str, str]] = []
