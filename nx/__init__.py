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

Toggles in-session: /source [name] [model]  /provider [source] [a,b,…|off]  /yolo  /approval [level]  /compress  /compact  /autocompress [pct|off|on]  /memory save|remember|list  /reset  /clear  /new  /recover  /undo  /sessions  /resume  /save  /delete  /quit
Flags: --yolo  --approval <all|low|medium|high|yolo>  --source <name>  --resume <target>  --session <id>
Env:   NX_APPROVAL=<all|low|medium|high|yolo>  (persistent default approval mode; ~/.env or shell)
       NX_AUTOCOMPRESS_PERCENT=<0-100>  (auto-compress threshold; 0=off; default 85)
       NX_BACKEND=vllm  (disables llama.cpp-only recovery knobs like min_p /
                             repeat_penalty / DRY that vLLM's speculative decoder
                             rejects; omits them from extra_body on retries)
       TOGETHER_API_KEY  (auto-registers a built-in `together` source; default model zai-org/GLM-5.2)
       OPENROUTER_API_KEY  (auto-registers a built-in `openrouter` source; default model z-ai/glm-5.2,
            routed to parasail/fp8 — override with /provider or NX_SOURCE_OPENROUTER_EXTRA_BODY)
"""
from __future__ import annotations

import glob
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
import types as _types
import urllib.error
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
    "_PENDING_MULTI_EDIT_NOTICE",
    "_CONSECUTIVE_EDITS", "_MULTI_EDIT_NUDGED",
    "REASONING_ONLY_RETRY_LIMIT", "MALFORMED_STREAM_RETRY_LIMIT",
    "FORCED_FINAL_MAX_TOKENS", "EMPTY_TURN_RETRY_LIMIT", "REASONING_ONLY_CHAR_LIMIT",
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

SESSION_HOME = os.path.expanduser(
    os.environ.get("NX_HOME", "~/.nx")
)
# Mirrored onto _state.py so _bash_jobs.py's background-log paths (which live
# under SESSION_HOME/bg-logs) see the same value, including across a /reload
# in tests that reassigns this from a fresh NX_HOME.
st.SESSION_HOME = SESSION_HOME
SESSION_LIST_DEFAULT_LIMIT = 10
SESSION_LIST_MAX_LIMIT = 100


def _sessions_dir() -> str:
    """Where session files live. Honors NX_SESSIONS_DIR, then NX_HOME/sessions."""
    return os.path.expanduser(
        os.environ.get("NX_SESSIONS_DIR",
                       os.path.join(SESSION_HOME, "sessions"))
    )


def _memories_dir() -> str:
    """Where memory files live. Always under SESSION_HOME/memories."""
    return os.path.join(SESSION_HOME, "memories")


from ._sessions_io import (
    _flatten_message_text, _new_session_id, _safe_title, _is_empty_assistant_message,
    _prune_empty_assistant_messages, _short_id, _summarize_session_data,
    _session_matches_query,
)


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def _index_path() -> str:
    return os.path.join(_sessions_dir(), ".index.json")


def _load_index() -> dict[str, Any]:
    """The sidecar listing index: {session_id: summary dict}. Missing or
    corrupt just returns {}, so callers degrade to a full per-file parse —
    same as before this index existed."""
    try:
        with open(_index_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, IOError, json.JSONDecodeError):
        return {}


def _write_index(idx: dict[str, Any]) -> None:
    path = _index_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass  # best-effort — a stale/missing index just falls back to full parses


def _update_index(session_id: str, data: dict[str, Any]) -> None:
    """Refresh this session's entry in the sidecar listing index so
    `nx sessions [query]` can list/search without json.load-ing every
    full transcript on disk (some can be large — tool outputs, diffs, …).
    Called from _write_session on every save, so the index stays at least
    as fresh as the file it mirrors."""
    idx = _load_index()
    idx[session_id] = _summarize_session_data(session_id, data)
    _write_index(idx)


def _write_session(session_id: str, messages: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    """Persist `messages` to the session file. Creates the dir if needed.

    Writes atomically (temp file + rename) so a crash mid-write can't
    corrupt the existing session. `meta` (title, source, cwd, …) is
    merged into the stored metadata.
    """
    d = _sessions_dir()
    os.makedirs(d, exist_ok=True)
    _prune_empty_assistant_messages(messages)
    path = _session_path(session_id)
    now = time.time()
    existing = {}
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
    except (OSError, IOError, json.JSONDecodeError):
        pass
    # Start from the existing metadata so partial writes (e.g. the session
    # description refresh, which only passes {"description", "desc_turns"})
    # don't clobber fields a prior _save_current wrote — token totals,
    # started_at, etc.  Then let `meta` override on top.
    data = dict(existing)
    data["id"] = session_id
    data["messages"] = messages
    data["created_at"] = existing.get("created_at", now)
    data["updated_at"] = now
    if meta:
        for k, v in meta.items():
            if v is not None:
                data[k] = v
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    try:
        os.utime(path, (data["updated_at"], data["updated_at"]))
    except (OSError, TypeError, ValueError):
        pass
    _update_index(session_id, data)


def _load_session(session_id: str) -> dict[str, Any] | None:
    """Read a session file. Returns the dict (id, messages, meta…) or None."""
    path = _session_path(session_id)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "messages" in data:
            _prune_empty_assistant_messages(data.get("messages"))
            return data
    except (OSError, IOError, json.JSONDecodeError):
        pass
    return None


def _session_files_newest() -> list[str]:
    """Return session JSON filenames newest-first using filesystem mtimes.

    Listing many sessions should not require parsing every old transcript.
    The session writer keeps mtime aligned with updated_at, so this is a
    cheap index for recent-session browsing.
    """
    d = _sessions_dir()
    files = []
    try:
        with os.scandir(d) as entries:
            for entry in entries:
                # ".json" but excludes the sidecar index (a dotfile, .index.json)
                if not entry.name.endswith(".json") or entry.name.startswith("."):
                    continue
                try:
                    if not entry.is_file():
                        continue
                    files.append((entry.stat().st_mtime, entry.name))
                except OSError:
                    continue
    except OSError:
        return []
    files.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [name for _, name in files]


def _session_summary_from_file(fname: str, idx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the listing summary for one session file. When `idx` (the
    sidecar index — see _load_index) already has an entry for this session,
    return that directly instead of json.load-ing the full transcript;
    `idx=None` or a cache miss falls back to a full parse exactly as before
    the index existed (e.g. a session written by an older nx, or a
    missing/corrupt index file)."""
    sid = fname[:-5] if fname.endswith(".json") else fname
    if idx is not None:
        cached = idx.get(sid)
        if cached is not None:
            return cached
    d = _sessions_dir()
    path = os.path.join(d, fname)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "messages" not in data:
            return None
    except (OSError, IOError, json.JSONDecodeError):
        return None
    return _summarize_session_data(data.get("id") or sid, data)


def _list_sessions(limit: int | None = 20, offset: int = 0, query: str | None = None) -> list[dict[str, Any]]:
    """Return sessions newest-first as dicts: id, title, description, preview, n.

    `preview` is the first ~60 chars of the first user message; `description`
    is an optional model-generated one-liner refreshed every N turns (richer
    than the static first-message title). `n` is the turn count (non-system
    messages)."""
    limit = max(0, int(limit)) if limit is not None else None
    offset = max(0, int(offset or 0))
    files = _session_files_newest()
    idx = _load_index()
    out = []
    for fname in files:
        summary = _session_summary_from_file(fname, idx)
        if not summary or not _session_matches_query(summary, query):
            continue
        out.append(summary)
        if query:
            continue
        if limit is not None and len(out) >= offset + limit:
            break
    if query:
        out.sort(key=lambda s: s["updated_at"], reverse=True)
    end = None if limit is None else offset + limit
    return out[offset:end]


def _delete_session(session_id: str) -> bool:
    """Remove a session file. Returns True if something was deleted."""
    path = _session_path(session_id)
    try:
        os.remove(path)
    except OSError:
        return False
    idx = _load_index()
    if idx.pop(session_id, None) is not None:
        _write_index(idx)
    return True


# After this many user turns, nx asks the model for a short (≤70 char)
# description of the whole conversation so far and stores it in the session
# file's `description` field. The description refreshes on every Nth turn
# thereafter, so it tracks what the chat is actually about as it evolves —
# far more useful in `nx sessions` than a static first-message title.
# 0 disables the refresh entirely (the auto-derived title is used as-is).
# The actual value is resolved after _env_int() is defined (below), so this
# is just a sentinel; see SESSION_DESC_REFRESH.
_DESC_REFRESH_DEFAULT = 6


def _maybe_refresh_description(session_id: str, messages: list[dict[str, Any]]) -> str | None:
    """Ask the model for a one-line session description, if enough turns have
    passed since the last refresh.

    Refreshes at the configured interval (every SESSION_DESC_REFRESH user
    turns). The description is stored in the session file's `description`
    field and surfaced in `nx sessions` / `/sessions` listings. A failure
    (server down, empty response) leaves the existing description untouched.

    Returns the new description string, or None if no refresh happened.
    """
    if SESSION_DESC_REFRESH <= 0:
        return None
    if st.client is None or st.MODEL is None:
        return None
    # Count user turns (exclude synthetic runtime-note-only turns).
    user_turns = sum(
        1 for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and not m["content"].lstrip().startswith("[")
    )
    existing = _load_session(session_id) or {}
    last_desc_turns = existing.get("desc_turns", 0)
    # Refresh when we've crossed a multiple of SESSION_DESC_REFRESH since the
    # last recorded refresh. Don't refresh before the first threshold (so a
    # 1-turn "hello" session doesn't burn a model call).
    if user_turns < SESSION_DESC_REFRESH:
        return None
    if user_turns - last_desc_turns < SESSION_DESC_REFRESH:
        return None
    # Build a compact transcript for the summarizer — truncate tool outputs
    # and skip the system prompt to keep the call cheap.
    def _trim(msgs: list[dict[str, Any]], per_msg: int = 500, cap: int = 20) -> str:
        out: list[str] = []
        for m in msgs:
            if m.get("role") == "system":
                continue
            c = _flatten_message_text(m, per_msg=per_msg)
            out.append(f"[{m.get('role', '?')}] {c}")
            if len(out) >= cap:
                break
        return "\n".join(out)
    prompt = _trim(messages[-30:])  # last ~30 messages is plenty of context
    payload = [
        {"role": "system", "content": DESC_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        _log_event("req", {"model": st.MODEL, "messages": payload,
                           "stream": False, "_purpose": "session_desc"})
        resp = st.client.chat.completions.create(
            model=st.MODEL, messages=payload, stream=False, timeout=20,
            **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
        try:
            _log_event("resp", {"_purpose": "session_desc", "data": resp.model_dump()})
        except Exception:
            pass
    except APIConnectionError:
        return None  # server down — leave existing description as-is
    except Exception:
        return None
    desc = (resp.choices[0].message.content or "").strip().splitlines()
    desc = desc[0].strip() if desc else ""
    if not desc:
        return None
    desc = _safe_title(desc, maxlen=70) or desc[:70]
    # Persist the description + the turn count it was generated at so we know
    # when to refresh next. Merge into existing meta (don't clobber messages).
    _write_session(session_id, messages, {"description": desc,
                                          "desc_turns": user_turns})
    return desc


def _resolve_session(target: str | None, sessions: list[dict[str, Any]] | None = None) -> str | None:
    """Resolve a user-typed target to a session id.

    Accepts: a full id, a numeric index into the recent-sessions list,
    a unique id prefix, or an exact title. Returns the id or None.
    """
    if not target:
        return None
    target = target.strip()
    sessions = sessions if sessions is not None else _list_sessions(limit=50)
    ids = [s["id"] for s in sessions]
    # numeric index → recent-sessions slot
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]["id"]
    # exact id
    if target in ids:
        return target
    # unique prefix
    prefixed = [i for i in ids if i.startswith(target)]
    if len(prefixed) == 1:
        return prefixed[0]
    # short id (the 6-hex suffix shown in listings) — match the tail segment
    # since the date+time prefix is shared/redundant across sessions created
    # in the same minute.
    suffixed = [i for i in ids if i.endswith("-" + target) or _short_id(i) == target]
    if len(suffixed) == 1:
        return suffixed[0]
    # exact title
    titled = [s["id"] for s in sessions if s["title"] == target]
    if len(titled) == 1:
        return titled[0]
    return None


# --- model sources ----------------------------------------------------------
from ._sources import (
    Source, _resolve_api_key, _parse_extra_body, _build_http_headers,
    _discover_sources, switch_source, restore_source, _set_provider_order,
    _provider_order,
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
if _start:
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

# --- base-level traffic log -------------------------------------------------
from ._logging import LOG_PATH, LOG_MAX_BYTES, _AsyncLogWriter, _llog, _log_event

# C is unused directly here but kept: _StateProxyModule.__getattr__ below
# proxies m.GREEN/m.CYAN/etc (the _COLOR_NAMES set) through to C's attributes.
from ._ansi import C

# --- interrupt watcher ------------------------------------------------------
from ._interrupt import (
    _INTERRUPT_ARMED, _INTERRUPT_EXIT, _USER_INTERRUPTED, _INTERRUPT_RESTORED,
    _interrupt_watcher, _ensure_interrupt_watcher,
)


# --- tools ------------------------------------------------------------------
from ._tools import (
    read_file, _LINE_NUM_PREFIX, _strip_line_numbers, list_dir, search_files,
    git_status, git_diff,
)
from ._undo import undo_last, list_checkpoints

from ._edit_tools import write_file, edit_file, multi_edit, sed_edit
from ._bash_jobs import (
    _run_bash, list_bg_jobs, _delete_old_bg_logs, _tail_last_line, _poll_pid,
    _read_log, _infer_timeout_from_sleep, run_bash, _finish_bg_job,
    wait_background, _find_bg_log_for_pid,
)

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "read_file", "description": "Read file, cat -n style (1-based). Windowed: offset/limit (default 400 lines); header shows range+total. Do NOT use this to hunt for one function/section in a file you haven't seen yet -- use search_files first to find it, then read_file only if you still need more surrounding context.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer", "description": "1-based start line (default 1)"}, "limit": {"type": "integer", "description": "max lines (default 400; <=0 = to end)"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Overwrite file (new file, or full rewrite). Editing an existing file? Use edit_file/multi_edit instead.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of old->new in an EXISTING file (errors if missing -- use write_file for a new file). Use raw text (not regex); read_file's `<n>\\t` prefixes auto-stripped if pasted. Prefer this (or multi_edit) over sed_edit whenever old/new is literal text, even across multiple lines.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List directory",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_files", "description": "Regex content search (ripgrep/grep) -> path:line:match hits. Read-only, no approval. Pass context=N for N surrounding lines per hit (skip read_file for targeted look-ups); otherwise hits feed read_file(path, offset=line).",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "regex (not glob)"}, "path": {"type": "string", "description": "file/dir to search (default \".\")"}, "glob": {"type": "string", "description": "include filter e.g. \"*.py\""}, "max_results": {"type": "integer", "description": "cap (default 100; <=0 -> 1000)"}, "context": {"type": "integer", "description": "lines of context before/after each match (default 0)"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "git_status", "description": "git status --short -b. Read-only, no approval.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "repo/dir (default \".\")"}}}}},
    {"type": "function", "function": {"name": "git_diff", "description": "git diff (unstaged by default). Read-only, no approval.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "repo/dir (default \".\")"}, "staged": {"type": "boolean", "description": "true = git diff --staged"}, "file": {"type": "string", "description": "restrict to this file"}}}}},
    {"type": "function", "function": {"name": "multi_edit", "description": "Atomic multi find/replace on one file (literal old->new, not regex), all-or-nothing. edits=[{old,new},...], each old must match exactly once, applied in order. One approval for whole batch. Use for several literal changes at once instead of sed_edit.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "description": "list of {old, new} objects, e.g. [{\"old\": \"a\", \"new\": \"b\"}] -- an actual JSON array, not a string", "items": {"type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "string"}}, "required": ["old", "new"]}}}, "required": ["path", "edits"]}}},
    {"type": "function", "function": {"name": "sed_edit", "description": "Regex find/replace (Python re) over file or glob (e.g. \"src/**/*.py\"). Only use when the change actually needs regex/glob (e.g. a pattern across many files) -- for literal text in one known file, use edit_file/multi_edit instead, they need no escaping. flags: i/m/s chars = IGNORECASE/MULTILINE/DOTALL. count caps replacements/file (0=unlimited). One approval for all matched files.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "replacement": {"type": "string"}, "path": {"type": "string"}, "count": {"type": "integer", "description": "max replacements per file (default 0 = unlimited)"}, "flags": {"type": "string", "description": "i/m/s = IGNORECASE/MULTILINE/DOTALL (default \"\" = none)"}}, "required": ["pattern", "replacement", "path"]}}},
    {"type": "function", "function": {"name": "run_bash", "description": "Shell cmd, detached (setsid), non-blocking. Quick (~3s poll) returns output directly; long ones background -> PID+log path, check via read_file(log_path). Never use `sleep N` as chain trick (whole cmd backgrounds). timeout=N only if cmd KNOWN to finish in ~Ns (e.g. timeout=35 for `sleep 30`); leading `sleep N` auto-extends poll to N+10s. Logs at ~/.nx/bg-logs/, auto-cleaned (~50 files/7d).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "description": "sync wait secs (default unset=~3s poll; 0=wait indefinitely)"}}, "required": ["command"]}}},
]

# Only offered once a run_bash call has actually backgrounded this session
# (see _active_tools() / st._SESSION_HAS_BACKGROUNDED below) — the common short
# session where nothing ever backgrounds shouldn't pay tokens for a tool it
# will never call.
_WAIT_BACKGROUND_TOOL = {"type": "function", "function": {"name": "wait_background", "description": "Wait for backgrounded run_bash cmd, return its output. Waits indefinitely until done or Esc (Esc cancels WAIT only, not process). timeout=N to bail early. Locates log via pid (auto) or log_path. Polls only (kill -0) — never signals, cannot kill.",
    "parameters": {"type": "object", "properties": {"pid": {"type": "integer", "description": "PID to wait for"}, "log_path": {"type": "string", "description": "log path (optional, auto-located by pid)"}, "timeout": {"type": "integer", "description": "max wait secs (default 0=indefinite; rarely set — Esc cancels)"}}, "required": ["pid"]}}}
TOOLS.append(_WAIT_BACKGROUND_TOOL)  # kept in TOOLS for callers that want the full schema (e.g. --tokens); _active_tools() is what's actually sent per-turn

FINAL_ANSWER_TOOL = {"type": "function", "function": {
    "name": "final_answer",
    "description": "Return complete, concise visible answer. Use when no more tool calls needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Complete concise answer to show user. Short, no trail-off.",
            },
            "status": {"type": "string", "enum": ["answered", "blocked"], "description": "Optional, omit unless you cannot proceed without more info from the user -- then set \"blocked\"."},
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

No native tool calls? Emit text-protocol call exactly:
<tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>
Prefer the tag alone; wait for Observation. Tool-call examples in prose/code-block only (not standalone). Task done -> reply plain prose.

Shell cmds run detached (setsid); they never block you.
- Finishes quickly -> output returned directly.
- Takes long -> backgrounds; you get PID + log path, check via read_file(log_path)."""

from ._env import _env_int, _env_float


# --- usage metrics (default off) --------------------------------------------
from ._metrics import (
    _normalize_usage, _Metrics, _env_metrics_url, _abbr, _precise_abbr,
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
st.REASONING_ONLY_RETRY_LIMIT = _env_int("NX_REASONING_ONLY_RETRIES", 3)
st.MALFORMED_STREAM_RETRY_LIMIT = _env_int("NX_MALFORMED_STREAM_RETRIES", 2)
st.FORCED_FINAL_MAX_TOKENS = _env_int("NX_FORCED_FINAL_MAX_TOKENS", 2048)
# An empty assistant turn (no content, no tool calls) is a silent stall: the model
# produced nothing and the loop would otherwise return TURN_DONE, dropping the user
# to the chat prompt mid-task. That's confusing — it looks like the agent finished
# when it actually just went mute (common with quantized models on a short/empty
# tool result, where the model "knows" the answer but emits no tokens). Auto-nudge
# it back into producing either a tool call or a visible answer. 0 disables the
# auto-continue (an empty turn then drops to chat as before).
st.EMPTY_TURN_RETRY_LIMIT = _env_int("NX_EMPTY_TURN_RETRIES", 3)

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
    "1. write_file — create the deliverable file the task asks for (e.g. /app/solution.py, "
    "/app/main.rs, the output file the verifier checks). Put your best current approach "
    "into code, even if it's a first draft. An imperfect file you can iterate on beats a "
    "perfect plan that lives only in your head.\n"
    "2. run_bash — run it to see what happens (python3 /app/solution.py, cargo run, etc.).\n"
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
    TOOL_TAG_PATTERN, NX_TOOL_TAG_PATTERN, TOOL_TAG, NX_TOOL_TAG,
    TOOL_TAGS, TOOL_PROTOCOL_TAG_RE, TOOL_RESULT_PROTOCOL_NOTE,
    parse_text_calls, parse_text_calls_with_errors,
    _escape_tool_protocol_delimiters, _sanitize_tool_result,
)


from ._dispatch import (
    DISPATCH, _active_tools, _messages_show_prior_backgrounding,
    _messages_show_prior_multi_edit_nudge, _initial_messages,
    BACKGROUND_GUIDANCE, MULTI_EDIT_GUIDANCE, _MULTI_EDIT_NUDGE_THRESHOLD,
    _CACHEABLE_READ_TOOLS, _tool_cache_key, _invalidate_tool_cache_path,
    run_tool, _tool_args_parse_error,
)


from ._compress import (
    RUNTIME_NOTE_RE, _nudge_current_user_turn, _flush_pending_background_notice,
    _last_is_dangling_tool, MAX_COMPLETION_TOKENS, RECOVERY_BACKEND,
    RECOVERY_MIN_P, RECOVERY_REPEAT_PENALTY, RECOVERY_REPEAT_LAST_N,
    RECOVERY_DRY_MULTIPLIER, RECOVERY_DRY_BASE, RECOVERY_DRY_ALLOWED_LENGTH,
    RECOVERY_DRY_SEQUENCE_BREAKERS, _recovery_sampling_opts, open_stream,
    compress, _maybe_autocompress,
)


# --- one model turn (streamed), returns TURN_* status -----------------------
from ._turn import (
    _LoggingStream, TURN_DONE, TURN_TOOL, TURN_STREAM_CUT,
    TURN_FORCE_FINAL, TURN_STALL, TURN_EMPTY, TURN_ESC, TURN_MALFORMED_CALL, _ctx_field,
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
    _cmd_memory_save, _cmd_memory_remember, _cmd_memory_list,
    _render_transcript_for_memory,
)


# --- one-shot (non-interactive) / CLI entry ----------------------------------
from ._cli import _run_one_shot, main


# --- state proxy -------------------------------------------------------------
# Makes `nx.<name>` (read AND write) for every name in _STATE_NAMES
# transparently proxy to `_state` (`st`), so `import nx as m; m.ACTIVE = x`
# (used throughout the test suite) and any submodule's `st.ACTIVE = x` always
# see the same value — see _state.py's docstring for why this is needed once
# the code that mutates these globals lives in more than one module.
_COLOR_NAMES = {"CYAN", "GREEN", "YELLOW", "RED", "MAGENTA", "GRAY"}


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
