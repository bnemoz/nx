#!/usr/bin/env python3
"""nx — a deliberately tiny coding agent for self-hosted or remote models.

A small package, no TUI framework, and no `openai` SDK dependency — nx talks
to `/chat/completions` and `/models` directly over `httpx` (see
`nx/_openai_client.py`). Points at any OpenAI-compatible endpoint (vLLM /
llama.cpp / SGLang / Z.ai / OpenAI itself). Survives models whose native
tool-calling isn't wired up yet by falling back to parsing
<tool_call>...</tool_call> tags out of the text — the convention most open
models (Hermes/Qwen/Nemotron) emit.

  pip install httpx rich
  export NX_BASE_URL=http://localhost:8000/v1   # your served endpoint
  export NX_MODEL=your-model-name
  export NX_API_KEY=sk-noop                    # any string; local servers ignore it
  python nx.py

Multiple sources — define named endpoints and switch between them at runtime:

  NX_SOURCES=local,zai
  NX_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
  NX_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
  NX_SOURCE_ZAI_API_KEY=***                   # $name = key from env / ~/.env
  NX_SOURCE_ZAI_MODEL=glm-x-preview

  python nx.py --source zai                     # start on Z.ai

Sessions — every chat is auto-saved to ~/.nx/sessions/ and resumable:

  python nx.py sessions           # list saved sessions (prints + exits)
  python nx.py sessions --page 2  # next page of saved sessions
  python nx.py sessions refactor  # …filtered by a substring query
  python nx.py --resume <id|short-id|prefix|title>  # resume a past session
  python nx.py --resume 1                        # resume the most recent
  /sessions                       # list recent sessions (with short ids)
  /sessions --page 2              # next page of recent sessions
  /resume <n|short-id|title>      # switch to another session mid-chat
  /save [title]                   # save the current session (title optional)

Toggles in-session: /source [name] [model]  /provider [source] [a,b,…|off]  /yolo  /approval [level]  /compress  /compact  /autocompress [pct|off|on]  /memory save|remember|list  /reset  /clear  /new  /recover  /undo  /sessions  /resume  /save  /delete  /todo  /quit
Flags: --yolo  --approval <all|low|medium|high|yolo>  --source <name>  --resume <target>  --session <id>
Env:   NX_APPROVAL=<all|low|medium|high|yolo>  (persistent default approval mode; ~/.env or shell)
       NX_AUTOCOMPRESS_PERCENT=<0-100>  (auto-compress threshold; 0=off; default 85)
       NX_VERIFY_CMD=<cmd>  (post-edit verify command, e.g. `uv run pytest -q`;
                             auto-runs ~20s after the last .py edit settles —
                             set with /verify, force one run with /verify run)
       NX_BACKEND=vllm  (disables llama.cpp-only recovery knobs like min_p /
                             repeat_penalty / DRY that vLLM's speculative decoder
                             rejects; omits them from extra_body on retries)
       TOGETHER_API_KEY  (auto-registers a built-in `together` source; default model zai-org/GLM-5.3)
       OPENROUTER_API_KEY  (auto-registers a built-in `openrouter` source; default model z-ai/glm-5.3,
            routed to io-net/fp8 — override with /provider or NX_SOURCE_OPENROUTER_EXTRA_BODY)
"""
from __future__ import annotations

import os
import sys
import types as _types
from typing import Any

from ._openai_client import APIConnectionError, APIError

# --- env file ---------------------------------------------------------------
from ._env import _load_env_file

_load_env_file()

# --- shared state -------------------------------------------------------------
# See _state.py's module docstring. `st` is the single home for globals that
# get mutated from more than one module (sources, approval, turn loop,
# compression, metrics, …). `nx.<name>` access for these names is proxied
# through to `st` below (see _StateProxyModule at the bottom of this file) so
# `import nx as m; m.ACTIVE` (get OR set) stays in sync no matter which
# module actually performs the mutation.
from . import _state as st

_STATE_NAMES = frozenset((
    "SOURCES", "SOURCE_ORDER", "ACTIVE", "client", "MODEL", "_LAST_PROMPT_TOKENS",
    "YOLO", "APPROVE_LEVEL", "DEFAULT_APPROVE_LEVEL", "PROJECT_ROOT",
    "_ACTIVE_SPINNER", "RISK_CONNECTION_RETRIES", "RISK_CONNECTION_RETRY_SECONDS",
    "_INTERRUPT_THREAD", "_SESSION_HAS_BACKGROUNDED", "_PENDING_BACKGROUND_NOTICE",
    "_PENDING_MULTI_EDIT_NOTICE", "_PENDING_VERIFY_NOTICE", "_VERIFY_SEEN",
    "_PENDING_SUBAGENT_NOTICE",
    "_VERIFY_EDIT_LOG", "_VERIFY_INFLIGHT",
    "_CONSECUTIVE_EDITS", "_MULTI_EDIT_NUDGED",
    "_PLAN_ACTIVE", "_PLAN_GOAL", "_PLAN_SLUG", "_PLAN_STARTED",
    "REASONING_ONLY_RETRY_LIMIT", "MALFORMED_STREAM_RETRY_LIMIT",
    "MALFORMED_TOOL_BREAKER_LIMIT", "_MALFORMED_TOOL_FAILURES",
    "_MALFORMED_TOOL_BROKEN",
    "FORCED_FINAL_MAX_TOKENS", "REASONING_BUDGET_TOKENS", "EMPTY_TURN_RETRY_LIMIT",
    "MAX_CONTINUATIONS",
    "REASONING_ONLY_CHAR_LIMIT",
    "RECOVERY_TEMPERATURE", "RECOVERY_TOP_P", "MAX_MODEL_TURNS",
    "AUTOCOMPRESS_PERCENT", "AUTOCOMPRESS_DEFAULT", "COMPRESS_KEEP", "_METRICS",
    "_TOOL_RESULT_CACHE",
))


# --- sessions ---------------------------------------------------------------
# Chat history persistence. Each session is one JSON file under
# ~/.nx/sessions/ (override with NX_HOME / NX_SESSIONS_DIR).
# The file stores the exact `messages` array the model sees (system prompt
# + every turn + tool calls/results), plus a little metadata (id, title,
# created_at, updated_at, cwd, source). Greppable, human-readable, and it
# round-trips trivially — load it back in and you have a resumable chat.
#
# Design cribbed from Hermes (hermes_state.py / SessionDB), which uses a
# SQLite store + FTS5 search because it's a multi-platform gateway (web,
# CLI, Telegram, …) with billing and compression chains. nx is a single
# local agent, so a flat directory of JSON files gives the same UX
# (auto-save per turn, /sessions, /resume, /save) without the weight.
#
# The filesystem side (directory layout, index, read/write/delete, the
# description-refresh model call) lives in _sessions_fs.py; the pure-logic
# helpers (id generation, title sanitizing, message pruning) in
# _sessions_io.py — see their docstrings.

SESSION_HOME = os.path.expanduser(
    os.environ.get("NX_HOME", "~/.nx")
)
# Mirrored onto _state.py so _bash_jobs.py/_sessions_fs.py (which need this
# under a directory join) see the same value, including across a /reload
# in tests that reassigns this from a fresh NX_HOME.
st.SESSION_HOME = SESSION_HOME

from ._sessions_io import (
    _flatten_message_text, _new_session_id, _safe_title, _is_empty_assistant_message,
    _prune_empty_assistant_messages, _short_id, _summarize_session_data,
    _session_matches_query,
)
from ._sessions_fs import (
    SESSION_LIST_DEFAULT_LIMIT, SESSION_LIST_MAX_LIMIT,
    _sessions_dir, _memories_dir, _project_memories_dir, _session_path, _index_dir, _index_entry_path,
    _load_index, _write_index, _update_index, _write_session, _load_session,
    _session_files_newest, _session_summary_from_file, _list_sessions,
    _delete_session, _maybe_refresh_description, _resolve_session,
    _merge_session_messages, _is_compaction_rewrite,
)
from ._input_history import (
    _load_input_history, _record_input, INPUT_HISTORY_LINES_DEFAULT,
    INPUT_HISTORY_ENTRY_CHARS,
)

# After this many user turns, nx asks the model for a short (≤70 char)
# description of the whole conversation so far and stores it in the session
# file's `description` field. The description refreshes on every Nth turn
# thereafter, so it tracks what the chat is actually about as it evolves —
# far more useful in `nx sessions` than a static first-message title.
# 0 disables the refresh entirely (the auto-derived title is used as-is).
# The actual value is resolved after _env_int() is defined (below), so this
# is just a sentinel; see SESSION_DESC_REFRESH.
_DESC_REFRESH_DEFAULT = 6




# --- model sources ----------------------------------------------------------
from ._sources import (
    Source, _resolve_api_key, _parse_extra_body, _build_http_headers,
    _discover_sources, switch_source, restore_source, _set_provider_order,
    _provider_order, _startup_probes_skipped,
)

# Bootstrap the source registry. Deliberately kept here (not inside
# _sources.py) rather than as a _sources.py-import-time side effect: tests
# rely on `importlib.reload(m)` (this module) to re-provision sources from
# freshly-set env vars, and reload() only re-runs the reloaded module's own
# top-level code — a submodule that's already in sys.modules would NOT be
# re-executed, so this bootstrap would go stale on reload if it lived there.
st.SOURCES = {}        # name → Source
st.SOURCE_ORDER = []   # preserve definition order for /source listing
st.ACTIVE = None       # current Source

# `st.client` and `st.MODEL` are read throughout the codebase. They always
# mirror the active source; switch_source() reassigns both. Every function
# that needs them does a call-time lookup (via _state), so a mid-session
# swap is picked up instantly — same pattern /yolo already uses for its own
# globals.
st.client = None
st.MODEL = None

# Updated by model_turn() on every turn that reports a prompt-token count
# (llama.cpp timings.prompt_n or OpenAI/Z.ai usage.prompt_tokens). Read by
# _maybe_autocompress() after a turn settles to decide whether the context
# window is full enough to warrant a silent auto-compress.
st._LAST_PROMPT_TOKENS = 0

_discover_sources()

# Pick the starting source: --source flag, then NX_ACTIVE env, then first.
_start = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--source" and _i + 1 < len(sys.argv):
        _start = sys.argv[_i + 1]
        break
_start = _start or os.environ.get("NX_ACTIVE") or (st.SOURCE_ORDER[0] if st.SOURCE_ORDER else None)
if not _start or _start not in st.SOURCES:
    _start = st.SOURCE_ORDER[0] if st.SOURCE_ORDER else None
# --help/--version/-h exit before any source is ever used (see
# _cli_help_or_version in _cli.py) — skip switch_source()'s network calls
# (resolve_model()'s /v1/models round-trip, capability probe) so `nx --help`
# against an unreachable host stays instant instead of blocking ~10-15s.
# switch_source() itself must still run on every import: it wires
# st.ACTIVE/st.client/st.MODEL, and tests (test_sources.py) reload nx and
# assert that wiring. Per-probe skipping — resolve_model's network fallback,
# the capability/thinking background probes, the ctx overrun probe — is what
# NX_SKIP_STARTUP_PROBES (nx/_sources.py) gates instead.
if _start and not ({"-h", "--help", "--version", "-v"} & set(sys.argv[1:])):
    switch_source(_start)

# --- approval gating --------------------------------------------------------
# Three risk levels (low < medium < high) plus an explicit "prompt all" state
# and YOLO mode — see nx/_approval.py for _normalize_level/_normalize_approval/
# _approval_display/_apply_approval and the interactive Y/n/esc gate
# (_ask_approval/_confirm/_EscToChat), and nx/_risk.py for LEVEL_ORDER and
# the model-based classifier (_assess_risk et al).
from ._approval import (
    LEVEL_ORDER, _LEVEL_ALIASES, _normalize_level, _normalize_approval,
    _approval_display, _apply_approval, _EscToChat, _ask_approval, _confirm,
    _diff_preview,
    _diff_hunk,
)
from ._risk import (
    _detect_project_root, _is_under_path, _extract_action_path, RISK_SYSTEM,
    _SAFE_BASH_COMMANDS, _SAFE_GIT_SUBCOMMANDS, _SHELL_METACHARACTERS_RE,
    _local_risk_hint, _classify_action_path, _risk_user_message, _assess_risk,
)

st._ACTIVE_SPINNER = None  # set by run_tool() while a tool body is executing

# Approval default resolution (highest precedence wins):
#   1. --yolo / --approval CLI flags  (explicit, per-invocation)
#   2. NX_APPROVAL env var         (persistent, per-user — set in ~/.env)
#   3. prompt-all built-in default     (safest: prompts on every action)
#
# This lets each developer keep their own comfort level in ~/.env (e.g.
# NX_APPROVAL=medium) without typing --approval every time, while the
# repo ships with the safest default (prompt everything) so anyone
# cloning it starts in cautious mode and can raise it with /approval or the
# env var. Accepts all|low|medium|high|yolo.
st.YOLO = False
st.APPROVE_LEVEL = None
st.DEFAULT_APPROVE_LEVEL = None
_env_approval = os.environ.get("NX_APPROVAL", "").strip().lower()
if _env_approval:
    _kind, _lvl = _normalize_approval(_env_approval)
    if not _apply_approval(_kind, _lvl):
        print(f"  ✗ unknown NX_APPROVAL={_env_approval!r} "
              f"(want all|low|medium|high|yolo); prompting for all actions")
for _i, _arg in enumerate(sys.argv):
    if _arg == "--approval" and _i + 1 < len(sys.argv):
        _kind, _lvl = _normalize_approval(sys.argv[_i + 1])
        if not _apply_approval(_kind, _lvl):
            print(f"  ✗ unknown --approval level {sys.argv[_i + 1]!r} "
                  f"(want all|low|medium|high|yolo); keeping {_approval_display()}")
if "--yolo" in sys.argv:
    st.YOLO = True
    st.APPROVE_LEVEL = None  # --yolo overrides everything; never prompt

# --- --prompt-file: non-interactive single-shot mode ------------------------
# `--prompt-file <path>` (or `-f <path>`) reads the full task instruction from
# a file, runs exactly one model turn, and exits — no REPL, no banner, no
# session save. Designed for headless benchmark harnesses (Harbor/Terminal-Bench)
# that pipe a task in and wait for the agent to finish. When set, main() runs
# _run_one_shot() instead of the interactive loop. A value of "-" reads from
# stdin instead of a file (useful for `echo ... | nx -f -`).
_PROMPT_FILE = None
for _i, _arg in enumerate(sys.argv):
    if _arg in ("--prompt-file", "-f") and _i + 1 < len(sys.argv):
        _PROMPT_FILE = sys.argv[_i + 1]
        break

# `--json`: one-shot mode emits a single machine-readable JSON summary (final
# answer, status, token totals, ordered tool-call trace) to stdout instead of
# the normal streamed chat output — for benchmark harnesses that need to score
# a run programmatically rather than scrape terminal text.
_JSON_OUTPUT = "--json" in sys.argv

# --- base-level traffic log -------------------------------------------------
from ._logging import LOG_PATH, LOG_MAX_BYTES, _AsyncLogWriter, _llog, _log_event

# C is unused directly here but kept: _StateProxyModule.__getattr__ below
# proxies m.GREEN/m.CYAN/etc (the _COLOR_NAMES set) through to C's attributes.
from ._ansi import C

# --- interrupt watcher ------------------------------------------------------
from ._interrupt import (
    _INTERRUPT_ARMED, _INTERRUPT_EXIT, _USER_INTERRUPTED, _INTERRUPT_RESTORED,
    _interrupt_watcher, _ensure_interrupt_watcher, _watch_for_interrupts,
)


# --- tools ------------------------------------------------------------------
from ._tools import (
    read_file, _LINE_NUM_PREFIX, _strip_line_numbers, list_dir, search_files,
    git_status, git_diff,
)
from ._undo import undo_last, list_checkpoints

from ._edit_tools import write_file, edit_file, multi_edit, sed_edit, delete_file
from ._garble import garble_signature, garble_check, realign_new, swapped_args
from ._bash_jobs import (
    _run_bash, list_bg_jobs, _delete_old_bg_logs, _tail_last_line, _poll_pid,
    _read_log, _infer_timeout_from_sleep, run_bash, _finish_bg_job,
    wait_background, _find_bg_log_for_pid,
)

# The tool schema, sent on every turn. Each string is an instruction the model
# reads before picking a tool or filling an argument, so it states what the call
# RETURNS and, where tools overlap, which one is right (write_file vs edit_file,
# search_files vs paging read_file, edit_file vs multi_edit). Guidance sits on
# the PROPERTY it applies to -- a model choosing `old_lines` reads that property
# and nothing else. Approval prompting is a harness fact, stated once in SYSTEM.
#
# Structural constraints, guarded by tests/test_tool_schema_invariants.py:
#   - REQUIRED properties come BEFORE optional ones: nx/_grammar.py's GBNF
#     builder can only drop a trailing optional suffix, so a required property
#     after an optional one makes a legal call unrepresentable.
#   - `properties` stays FLAT (string/integer/boolean/enum values plus
#     todo_write/multi_edit's array-of-objects) -- the subset the grammar
#     understands. read_file/search_files use `oneOf` (str | [str]), which
#     deliberately falls through to the grammar's permissive `value` rule.
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file as numbered lines (`<n>\\t<text>`, 1-based). Long files return one window; page with offset/limit. To FIND a function or string, use search_files instead of paging.", "parameters": {"type": "object", "properties": {"path": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": "File, or a list of files (same window for each)"}, "offset": {"type": "integer", "description": "First line, 1-based (default 1)"}, "limit": {"type": "integer", "description": "Max lines (default 400)"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create a file, or replace one entirely: `content` becomes the WHOLE file. To change part of a file, use edit_file/multi_edit.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string", "description": "Complete new contents"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace literal `old` with `new` in an existing file. `old` must match EXACTLY ONCE: copy it from read_file, long enough to be unique, or set replace_all.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "new": {"type": "string", "description": "Replacement text ('' deletes the match)"}, "old": {"type": "string", "description": "Exact text to find; pasted line numbers and wrong indentation are tolerated"}, "replace_all": {"type": "boolean", "description": "Replace every occurrence"}, "old_lines": {"type": "string", "description": "Instead of `old`: line span \"N\" or \"N-M\" (1-based)"}}, "required": ["path", "new"]}}},
    {"type": "function", "function": {"name": "delete_file", "description": "Delete one file; restorable with /undo, unlike `rm` via run_bash. Files only (no directories, no dotfiles).", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File to delete"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "Names directly inside a directory; with `pattern`, a recursive glob search instead.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory (default \".\")"}, "pattern": {"type": "string", "description": "Glob, e.g. \"**/*.py\""}}}}},
    {"type": "function", "function": {"name": "search_files", "description": "Search file contents by regex (ripgrep) -> `path:line:match`, which feeds read_file(path, offset=line).", "parameters": {"type": "object", "properties": {"pattern": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": "Regex, not a glob. A list is OR'd"}, "path": {"type": "string", "description": "File or directory (default \".\")"}, "glob": {"type": "string", "description": "Only files whose name matches, e.g. \"*.py\""}, "max_results": {"type": "integer", "description": "Max hits (default 100)"}, "context": {"type": "integer", "description": "Lines shown around each hit"}, "ignore_case": {"type": "boolean", "description": "true = match case-insensitively"}, "output_mode": {"enum": ["content", "files_with_matches", "count"], "description": "content = path:line:match | files_with_matches = paths only | count = hits/file"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "git_status", "description": "`git status --short -b`: branch and changed files.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "git_diff", "description": "`git diff`: unstaged changes.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "staged": {"type": "boolean", "description": "true = staged changes"}, "file": {"type": "string", "description": "Restrict to this path"}}}}},
    {"type": "function", "function": {"name": "git_log", "description": "`git log --oneline`: recent commits, hash + subject.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_count": {"type": "integer", "description": "Max commits (default 20)"}, "file": {"type": "string", "description": "Restrict to this path"}}}}},
    {"type": "function", "function": {"name": "multi_edit", "description": "Several literal old->new edits to ONE file in one call: one approval, atomic (a failing edit writes nothing). Prefer over repeated edit_file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "description": "In order; each `old` must match once", "items": {"type": "object", "properties": {"old": {"type": "string", "description": "Exact text to find"}, "new": {"type": "string", "description": "Its replacement"}}, "required": ["old", "new"]}}}, "required": ["path", "edits"]}}},
    {"type": "function", "function": {"name": "sed_edit", "description": "Regex (Python `re`) find/replace over one file or a glob: for one change repeated across many files. Literal text in one file -> edit_file.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "replacement": {"type": "string", "description": "Literal text (no \\1 backreferences)"}, "path": {"type": "string", "description": "File, or glob like \"src/**/*.py\""}, "count": {"type": "integer", "description": "Max replacements per file (0 = all)"}, "flags": {"type": "string", "description": "Flag letters: i = ignore case, m = multiline, s = dot matches newline"}}, "required": ["pattern", "replacement", "path"]}}},
    {"type": "function", "function": {"name": "apply_patch", "description": "Apply a unified diff to every file it touches, atomically and only if it applies cleanly. For multi-file restructure.", "parameters": {"type": "object", "properties": {"diff": {"type": "string", "description": "Unified diff text; needs `--- a/x` / `+++ b/x` headers per file"}, "path": {"type": "string", "description": "Directory the diff's paths are relative to"}}, "required": ["diff"]}}},
    {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command and return its output (pipes, `&&` work). A slow one keeps running detached and the result reads `[background] PID <n> | log: <path>`.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Never prefix `sleep N &&` to make it wait -- that backgrounds too; use timeout"}, "timeout": {"type": "integer", "description": "Seconds to wait before backgrounding; 0 = until it exits"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "todo_write", "description": "Plan a multi-step task. Each call REPLACES the whole list, so resend every item with its status; exactly one item in_progress.", "parameters": {"type": "object", "properties": {"todos": {"type": "array", "description": "Complete list, in execution order", "items": {"type": "object", "properties": {"text": {"type": "string", "description": "Short imperative task"}, "status": {"enum": ["pending", "in_progress", "done"], "description": "pending = not started, done = finished"}}, "required": ["text"]}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "memory_write", "description": "Save a note for FUTURE sessions: a gotcha, a decision and why, a command worth keeping. Supersede stale notes instead of duplicating them.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Short topic title"}, "body": {"type": "string", "description": "Markdown: facts, exact paths/commands, why"}, "scope": {"enum": ["project", "global"], "description": "project = this repo, global = every project"}}, "required": ["title", "body"]}}},
    {"type": "function", "function": {"name": "memory_search", "description": "Find saved notes by keyword, title + snippet, best first.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "All terms must appear; \"\" lists newest titles"}, "scope": {"enum": ["project", "global"], "description": "project (default) = project + global notes"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "memory_delete", "description": "Delete one note. An ambiguous or unknown target deletes nothing and lists candidates.", "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "Exact title, filename, or unique title prefix"}, "scope": {"enum": ["project", "global"], "description": "project (default) = project + global notes"}}, "required": ["target"]}}},
]

# Only offered once a run_bash call has actually backgrounded this session
# (see _active_tools() / st._SESSION_HAS_BACKGROUNDED below) — the common short
# session where nothing ever backgrounds shouldn't pay tokens for a tool it
# will never call.

# Read-only sub-agent tool (spawn_agent): schema appended below, machinery in
# nx/_subagent.py. _active_tools() filters it out unless NX_SUBAGENTS=1.
from ._subagent import SUBAGENT_TOOL, spawn_agent

# Plan-mode machinery (schema/body): the plan_edit tool plus the gate the
# mutating tools cross while a plan is active. See nx/_plan.py.
from ._plan import (plan_edit, PLAN_GATE_TOOLS, PLAN_READONLY_TOOLS,
                    check_gate, gate_message, plan_active, plan_path, plan_dir,
                    render_plan_display, PLAN_EDIT_TOOL)

# Vision tool: same opt-in posture (NX_VISION=1), and additionally withheld
# until native tool_calls are confirmed this session (a text-protocol result
# can't carry a content-parts array).
from ._images import read_image, IMAGE_VISION_TOOL

_WAIT_BACKGROUND_TOOL = {"type": "function", "function": {"name": "wait_background", "description": "Wait for a command run_bash backgrounded and return its output when it exits. Waits as long as it takes; Esc cancels the WAIT, never the process. It cannot kill anything -- use it instead of reaching for pkill.",
    "parameters": {"type": "object", "properties": {"pid": {"type": "integer", "description": "The PID from the `[background]` result"}, "timeout": {"type": "integer", "description": "Give up after this many seconds (default 0 = wait until it exits)"}}, "required": ["pid"]}}}
TOOLS.append(_WAIT_BACKGROUND_TOOL)  # kept in TOOLS for callers that want the full schema (e.g. --tokens); _active_tools() is what's actually sent per-turn
TOOLS.append(SUBAGENT_TOOL)
TOOLS.append(IMAGE_VISION_TOOL)
# Plan-edit tool (schema here, body in nx/_plan.py): withheld per-turn by
# _active_tools() until a plan is active (/plan <goal>), like wait_background
# is withheld until something backgrounds.
TOOLS.append(PLAN_EDIT_TOOL)

FINAL_ANSWER_TOOL = {"type": "function", "function": {
    "name": "final_answer",
    "description": "End the turn with the answer the user sees; it must stand on its own, since the transcript before it is not re-shown. Call it when the work is done or when you need something from the user. Stop by calling this, not by replying with plain text.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The whole answer: what you did or found, plus anything left outstanding. Concise, and nothing after the substance.",
            },
            "status": {"type": "string", "enum": ["answered", "blocked"], "description": "\"blocked\" when you cannot proceed without something only the user can supply; otherwise omit it."},
        },
        "required": ["answer"],
    },
}}
FINAL_ANSWER_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "final_answer"},
}

SYSTEM = """Terminal coding agent, cwd = user's current directory.
Use tools to inspect/modify code. One concrete step at a time.
Locating something (a function, string, section) in a file you haven't
already read in full? search_files FIRST, always -- even on small files.
read_file is for reading a file you already need in full, never for finding
a thing inside one.
BE TERSE WITH YOUR REASONING TRACES

Every write/edit/bash call prompts the user for approval (reads never do), so batch
related edits into one multi_edit; "DENIED by user" means stop and ask, not retry.

No native tool calls? Emit text-protocol call exactly:
<tool_call><name>read_file</name><arguments><path>foo.py</path></arguments></tool_call>
Prefer the tag alone; wait for Observation. Tool-call examples in prose/code-block only (not standalone). Task done -> reply plain prose.

Shell cmds run detached (setsid) and never block you: a quick one returns its
output, a slow one backgrounds (result gives PID + log path -- read_file it).

3-4+ step task -> call todo_write FIRST, before any other tool (one
in_progress at a time). Re-issue todo_write as the work or its scope grows.

Memory: memory_search before re-deriving something already learned;
memory_write to persist a durable learning (scope=project for this repo,
global for cross-project)."""

from ._env import _env_int, _env_float


# --- usage metrics (default off) --------------------------------------------
from ._metrics import (
    _normalize_usage, _record_usage, _Metrics, _env_metrics_url, _abbr, _precise_abbr,
)

st.REASONING_ONLY_CHAR_LIMIT = _env_int("NX_REASONING_ONLY_CHARS", 36000)
# In addition to the char limit, cut a reasoning-only stream if it exceeds
# this many *seconds* of wall-clock time.  The char limit (36K) assumes a
# healthy generation speed; at slow speeds (6 tok/s on some local models) the
# model can burn the entire timeout in a single reasoning spiral without ever
# reaching the char limit.  120s gives ample room for genuine deep thinking
# while preventing a single reasoning block from eating more than ~2 minutes.
# Set to 0 to disable the time-based check (char limit only).
REASONING_ONLY_TIME_LIMIT = _env_int("NX_REASONING_ONLY_TIME", 120)
# When a reasoning-only turn stalls, capture the reasoning_content and feed it
# back to the model as an assistant message so it can *continue* from where it
# left off rather than starting from scratch.  This caps how many chars of
# reasoning we store (the rest is truncated to the tail, which usually contains
# the model's most recent conclusions / draft work).  0 disables capture
# (reverts to the old behavior: discard reasoning, nudge with no context).
REASONING_CAPTURE_CHARS = _env_int("NX_REASONING_CAPTURE_CHARS", 8000)
# Cap + dedup tool results before they enter the message history. Large,
# near-duplicate tool outputs (find/ls/grep path dumps) are the fuel for the
# context-copying repetition collapse some local models fall into — bounding
# them is the cheapest prevention. 0 disables the size cap (dedup still runs).
from ._text_fallback import TOOL_RESULT_CHARS
from ._overflow import (
    overflow_dir, spill_to_disk, build_overflow_result,
    OVERFLOW_TTL_DAYS, MAX_SPILL_ECHO,
)

# --- leaked special-token collapse ------------------------------------------
from ._special_tokens import (
    _LEAKED_SPECIAL_TOKEN_RE, LEAKED_TOKEN_RUN_LIMIT, REPEAT_CHUNK_LIMIT,
    _strip_leaked_tokens, _is_leak_chunk,
)

# Default sampling params for normal (non-recovery) completions.
# Set these to get deterministic/greedy output: NX_TEMPERATURE=0
# NX_TOP_P is only relevant when temperature > 0; set to 1 for full
# probability mass, or lower for nucleus sampling.
# Set either to a negative value to omit it from the request (let the
# server use its own default, typically temperature=1.0 / top_p=1.0).
st.RISK_CONNECTION_RETRIES = _env_int("NX_RISK_RETRIES", 3)
st.RISK_CONNECTION_RETRY_SECONDS = _env_int("NX_RISK_RETRY_SECONDS", 1)
# Resolved here (not at the sessions-section top) because _env_int() is
# defined just above. See the comment at _DESC_REFRESH_DEFAULT for behavior.
SESSION_DESC_REFRESH = _env_int("NX_SESSION_DESC_REFRESH", _DESC_REFRESH_DEFAULT)
DESC_SYSTEM = (
    "You summarize coding-agent conversations into one short line. "
    "Given the transcript, reply with ONLY a single line of at most 70 "
    "characters describing what this session is about / working on — the "
    "current task, key files, or goal. No preamble, no quotes, no trailing "
    "punctuation. Write in the same language as the conversation."
)
# Global cross-run input recall (Up/Down at the prompt): how many past
# submissions ~/.nx/input_history.json holds. 0 disables recall entirely.
# Resolved here (not in _input_history.py) because _env_int() is defined
# below this point; _input_history reads it lazily via `import nx as _m`.
HISTORY_LINES = _env_int("NX_HISTORY_LINES", INPUT_HISTORY_LINES_DEFAULT)

st.REASONING_ONLY_RETRY_LIMIT = _env_int("NX_REASONING_ONLY_RETRIES", 3)
st.MALFORMED_STREAM_RETRY_LIMIT = _env_int("NX_MALFORMED_STREAM_RETRIES", 2)
# Malformed-args circuit breaker: after this many malformed-args failures for
# the SAME tool name (native tool_calls path), that tool is dropped from
# _active_tools() for the rest of the session — an endpoint that reliably
# truncates one tool's JSON args stops being offered it (see nx/_turn.py).
# 0 disables the breaker.
st.MALFORMED_TOOL_BREAKER_LIMIT = _env_int("NX_MALFORMED_TOOL_BREAKER", 2)
st.FORCED_FINAL_MAX_TOKENS = _env_int("NX_FORCED_FINAL_MAX_TOKENS", 2048)
# Reasoning token budget: cap on the model's thinking phase, sent as
# {"reasoning": {"max_tokens": N}} in the request body. 0 = off (the default —
# let the model self-terminate its thinking). /budget changes it live.
st.REASONING_BUDGET_TOKENS = _env_int("NX_REASONING_BUDGET", 0) or None
# An empty assistant turn (no content, no tool calls) is a silent stall: the model
# produced nothing and the loop would otherwise return TURN_DONE, dropping the user
# to the chat prompt mid-task. That's confusing — it looks like the agent finished
# when it actually just went mute (common with quantized models on a short/empty
# tool result, where the model "knows" the answer but emits no tokens). Auto-nudge
# it back into producing either a tool call or a visible answer. 0 disables the
# auto-continue (an empty turn then drops to chat as before).
st.EMPTY_TURN_RETRY_LIMIT = _env_int("NX_EMPTY_TURN_RETRIES", 3)
# A turn cut off by finish_reason="length" (the per-request token cap hit —
# NX_MAX_TOKENS, or the server's own cap) ends mid-answer. Auto-continue it:
# keep the partial text as a real assistant turn, append a "continue exactly
# where you stopped" note, and re-open the stream. Budgeted by this many
# continuations per user request; beyond that the partial answer is closed out
# with a truncation marker (as before). 0 disables the auto-continue.
st.MAX_CONTINUATIONS = _env_int("NX_MAX_CONTINUATIONS", 2)

# --- auto context-compression -----------------------------------------------
# When the context window fills past this fraction (default 85%), the next
# settled model turn silently compresses the older history so the chat can
# keep going instead of running out of room. Unlike manual /compress (which
# keeps only st.COMPRESS_KEEP=2 turns), auto-compress is conservative: it keeps
# more recent turns so in-progress work isn't disturbed. Set to 0 to disable.
st.AUTOCOMPRESS_PERCENT = _env_int("NX_AUTOCOMPRESS_PERCENT", 85)
st.AUTOCOMPRESS_PERCENT = max(0, min(100, st.AUTOCOMPRESS_PERCENT))  # clamp 0–100
# What `/autocompress on` restores; falls back to 85 when configured off.
st.AUTOCOMPRESS_DEFAULT = st.AUTOCOMPRESS_PERCENT or 85

# When the model stalls in reasoning-only (thinks and thinks but never
# produces a tool call or visible text), the first nudge should push it to
# ACT — use tools, write code, run commands — rather than to produce a
# premature "final answer".  This is especially critical for coding-agent
# tasks where the model must manipulate the filesystem, not narrate.
# Only on the LAST retry do we ask for a final answer (forced_final=True).
REASONING_STALL_NUDGE = (
    "Your previous response was reasoning only — you produced no tool calls and "
    "no visible output. Stop planning in your head. ACT NOW, in this order:\n"
    "1. write_file — create the deliverable file the task asks for (a new file in "
    "this project, or the output file the user's task names). Put your best current approach "
    "into code, even if it's a first draft. An imperfect file you can iterate on beats a "
    "perfect plan that lives only in your head.\n"
    "2. run_bash — run it to see what happens (python3 <that file>, the project's "
    "test command, etc.).\n"
    "3. edit_file / write_file — fix what the run told you was wrong, then re-run.\n"
    "Do not produce a final_answer until the deliverable file exists and runs. Pick the "
    "most promising approach from your reasoning and execute it right now."
)
FORCED_FINAL_NUDGE = (
    "Your previous streamed response produced reasoning only. Do not continue "
    "private reasoning. Use the final_answer tool if available. Return a complete "
    "visible answer now in at most six short bullets or paragraphs. If you are "
    "blocked, say exactly what is blocking you and what input is needed."
)
EMPTY_TURN_NUDGE = (
    "Your previous response was empty — you produced no visible text and no tool "
    "call, so nothing happened. Do not stop. If the task is not finished, emit the "
    "next tool call now. If you have everything you need, use the final_answer tool "
    "(or reply with a concise visible answer). Do not repeat the empty turn."
)
MANUAL_RECOVERY_NUDGE = (
    "Manual recovery requested by the user because the previous response appeared "
    "off the rails or corrupted. Discard any corrupted reasoning/output and do not "
    "continue private reasoning. Use the final_answer tool if available. Return a "
    "bounded visible checkpoint with: (1) the last valid result you can rely on, "
    "(2) the next concrete action you would take, and (3) any blocker or "
    "uncertainty. Keep it concise."
)


# --- text-fallback parsing --------------------------------------------------
from ._text_fallback import (
    TOOL_TAGS, TOOL_PROTOCOL_TAG_RE, TOOL_RESULT_PROTOCOL_NOTE,
    parse_text_calls, parse_text_calls_with_errors,
    _escape_tool_protocol_delimiters, _sanitize_tool_result,
)


from ._dispatch import (
    DISPATCH, _active_tools, _messages_show_prior_backgrounding,
    _messages_show_prior_multi_edit_nudge, _messages_show_prior_delegation_nudge,
    _initial_messages,
    BACKGROUND_GUIDANCE, MULTI_EDIT_GUIDANCE, DELEGATION_STREAK_NOTICE,
    _MULTI_EDIT_NUDGE_THRESHOLD, _DELEGATION_NUDGE_THRESHOLD,
    _CACHEABLE_READ_TOOLS, _tool_cache_key, _invalidate_tool_cache_path,
    run_tool, _tool_args_parse_error,
)
from ._tool_cache import (  # noqa: F401 (re-export for tests)
    cache_store_entry as _tool_cache_store_entry,
    cache_valid as _tool_cache_valid,
    git_state_stamp as _tool_cache_git_state_stamp,
    drop_vcs_entries as _tool_cache_drop_vcs_entries,
    store_cacheable_result as _tool_cache_store_result,
)


# --- read-only sub-agent delegation -----------------------------------------
# spawn_agent runs a focused sub-loop with a read-only toolset so exploratory
# reads (find X, summarize module Y, list callers of Z) don't bloat the main
# context with intermediate tool output. Schema above; machinery in
# nx/_subagent.py. OFF by default (NX_SUBAGENTS=1 to enable) — it adds a tool
# schema to every turn's floor, so it's opt-in like NX_GRAMMAR/NX_NATIVE_TOOLS.
NX_SUBAGENTS = _env_int("NX_SUBAGENTS", 0)
# Sub-agent default execution mode (NX_SUBAGENT_ASYNC, default 1): a spawn runs
# in the background by default — quiet worker thread, answer lands as a short
# head + report-path [Runtime note] on a later turn (full answer on disk under
# SESSION_HOME/agents/) — so a fan-out no longer renders every sub-agent's
# spinner/footers in the parent TUI. NX_SUBAGENT_ASYNC=0 restores the old
# foreground (synchronous) default, which still shows the sub-agent's tool
# boxes and reasoning live.
NX_SUBAGENT_ASYNC = _env_int("NX_SUBAGENT_ASYNC", 1)
# Opt-in write tier for sub-agents (NX_SUBAGENT_WRITE=1): the four file-mutation
# tools (write_file/edit_file/multi_edit/sed_edit) are additionally offered
# inside a sub-agent, bound to wrappers that run the parent's real bodies — so
# every sub-agent write still crosses the parent's own _confirm() (risk gate +
# Y/n prompt with a diff preview), tagged (sub-agent) in the tool box.
NX_SUBAGENT_WRITE = _env_int("NX_SUBAGENT_WRITE", 0)
# Sub-agent budget knobs (read in nx/_subagent.py at spawn time): the default
# turn budget the schema advertises (NX_SUBAGENT_DEFAULT_TURNS, 100 — the old
# 12 left real investigations capping out with no answer) and the hard ceiling
# for a requested max_turns (NX_SUBAGENT_MAX_TURNS, 200). Kept here alongside
# the other NX_SUBAGENT_* resolution points so tests/REPL can flip them live.
NX_SUBAGENT_DEFAULT_TURNS = _env_int("NX_SUBAGENT_DEFAULT_TURNS", 100)
NX_SUBAGENT_MAX_TURNS = _env_int("NX_SUBAGENT_MAX_TURNS", 200)
NX_VISION = _env_int("NX_VISION", 0)
# Slug length cap for /plan <goal> → .nx/<slug>.md (see _plan.py).
NX_PLAN_MAX_WORDS = _env_int("NX_PLAN_MAX_WORDS", 8)


from ._compress import (
    RUNTIME_NOTE_RE, _nudge_current_user_turn, _flush_pending_background_notice,
    _last_is_dangling_tool, MAX_COMPLETION_TOKENS, RECOVERY_BACKEND,
    RECOVERY_MIN_P, RECOVERY_REPEAT_PENALTY, RECOVERY_REPEAT_LAST_N,
    RECOVERY_DRY_MULTIPLIER, RECOVERY_DRY_BASE, RECOVERY_DRY_ALLOWED_LENGTH,
    RECOVERY_DRY_SEQUENCE_BREAKERS, _recovery_sampling_opts, open_stream,
    compress, _maybe_autocompress, _compress_for_overrun,
)


# --- post-edit verification --------------------------------------------------
# (a) a free in-process ast.parse after every successful .py write/edit, whose
#     verdict rides on the edit's own tool result; (b) opt-in NX_VERIFY_CMD
#     (e.g. `uv run pytest -q`) launched detached ~NX_VERIFY_SETTLE_SECS after
#     the last edit settles, with pass/fail reported back as a [Runtime note].
#     See nx/_verify.py; user-facing control via /verify.
from ._verify import (
    _ast_check, _launch_verify_if_settled, record_py_edit, verify_enabled,
    _stage_verify_outcome, _is_verify_job,
)


# --- one model turn (streamed), returns TURN_* status -----------------------
from ._turn import (
    _LoggingStream, TURN_DONE, TURN_TOOL, TURN_STREAM_CUT,
    TURN_FORCE_FINAL, TURN_STALL, TURN_EMPTY, TURN_ESC, TURN_MALFORMED_CALL,
    TURN_API_ERROR, TURN_LENGTH_CUT, CONTINUATION_NUDGE, TRUNCATION_MARKER,
    _ctx_field,
    _ensure_ctx_probe, _print_stats_footer, _handle_forced_final_tool_call,
    _handle_native_tool_calls, _handle_text_tool_calls, _handle_loop_cut,
    model_turn, _run_model_turn_loop,
)


# --- multi-line chatbox input ---------------------------------------------
from ._chatbox import (
    _chatbox_fallback, _char_width, _str_width, _str_slice_by_width,
    _raw_read_key, _raw_read_available, _chatbox_raw, read_multiline,
)


# --- repl -------------------------------------------------------------------
from ._repl import (
    _banner, _repl, _session_id_from_args, _session_messages, _fmt_when,
    _print_transcript, _session_list_page_options, _print_session_list,
    _session_next_hint, _display_path, _cli_sessions, _cli_tokens,
    _cmd_sessions, _cmd_resume, _cmd_delete, _cmd_memory, _slugify,
    _cmd_memory_save, _cmd_memory_remember, _cmd_memory_delete, _cmd_memory_list,
    _render_transcript_for_memory, memory_write, memory_search, memory_delete,
    _load_memory_note, _write_memory_file,
)


# --- one-shot (non-interactive) / CLI entry ----------------------------------
from ._cli import _run_one_shot, _one_shot_result, main


# --- headless batch mode (nx run -f tasks.jsonl) -----------------------------
from ._batch import run_batch, _batch_active


# --- state proxy -------------------------------------------------------------
# Makes `nx.<name>` (read AND write) for every name in _STATE_NAMES
# transparently proxy to `_state` (`st`), so `import nx as m; m.ACTIVE = x`
# (used throughout the test suite) and any submodule's `st.ACTIVE = x` always
# see the same value — see _state.py's docstring for why this is needed once
# the code that mutates these globals lives in more than one module.
_COLOR_NAMES = {"CYAN", "GREEN", "YELLOW", "RED", "MAGENTA", "GRAY", "BLUE", "ORANGE"}


class _StateProxyModule(_types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name in _STATE_NAMES:
            return getattr(st, name)
        if name in _COLOR_NAMES:
            return getattr(C, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _STATE_NAMES:
            setattr(st, name, value)
        else:
            super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | _STATE_NAMES)


sys.modules[__name__].__class__ = _StateProxyModule


if __name__ == "__main__":
    main()
