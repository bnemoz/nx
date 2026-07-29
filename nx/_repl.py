"""The interactive REPL: `_repl()` is the main input/dispatch loop (banner,
`read_multiline`, `/command` handling, auto-save/auto-compress after each
turn), plus `_banner()` and every `/sessions` `/resume` `/delete` `/memory`
command handler and their session-list/transcript-printing helpers.

Functions here that are monkeypatched by tests via `m.<name> = fake`
(`_banner`, `compress`, `_confirm`, `read_multiline`) are always called
through a lazy `import nx as _m; _m.<name>(...)` — including from other
functions defined in *this* file — since the patch only ever lands on the
top-level `nx` module's attribute, never on this submodule's own
globals. See _turn.py / _compress.py for the same pattern.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import sys
import time
from typing import Any, Callable

from ._openai_client import APIConnectionError

from . import _ansi
from ._ansi import DIM, BOLD, RESET, C
from ._approval import _EscToChat, _normalize_approval, _apply_approval, _approval_display
from ._env import _set_env_var
from ._logging import _log_event
from ._mentions import expand_at_mentions
from ._metrics import _abbr, _Metrics, _env_metrics_url
from ._resources import process_stats
from ._sessions_io import _flatten_message_text, _safe_title, _short_id
from ._sources import switch_source, restore_source, _set_provider_order, _provider_order
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from ._compress import _maybe_autocompress
from . import _state as st
from . import (
    SYSTEM, MANUAL_RECOVERY_NUDGE, _active_tools, _initial_messages,
    _sessions_dir, _memories_dir, SESSION_HOME,
    SESSION_LIST_DEFAULT_LIMIT, SESSION_LIST_MAX_LIMIT,
    _write_session, _load_session, _list_sessions, _delete_session,
    _resolve_session, _new_session_id, _messages_show_prior_backgrounding,
    _messages_show_prior_multi_edit_nudge,
    _maybe_refresh_description, list_bg_jobs,
)


_HELP_COMMANDS: list[tuple[str, str]] = [
    ("/help", "show this list of commands"),
    ("/quit, /exit", "save session (if dirty) and exit"),
    ("/source [name] [model]", "list or switch model source/backend"),
    ("/theme [name]", "list or switch color theme"),
    ("/reasoning [on|off|show|hide|last]", "toggle/show reasoning-content visibility"),
    ("/provider [source] [order|off]", "set/clear OpenRouter provider routing"),
    ("/resources, /resmon", "show CPU%/RSS process stats"),
    ("/bg [kill <pid>]", "list/kill background bash jobs"),
    ("/yolo", "toggle yolo mode (disable all approval prompts)"),
    ("/approval [all|low|medium|high|yolo]", "show/set approval level"),
    ("/reset, /clear, /new", "clear context, start a new session"),
    ("/init", "have the model write/update AGENTS.md"),
    ("/recover [note]", "force a visible recovery checkpoint turn"),
    ("/undo [--list]", "undo last file change / list undo checkpoints"),
    ("/compress, /compact", "fold older turns into a summary"),
    ("/autocompress [percent|on|off]", "configure automatic compression threshold"),
    ("/sessions [...]", "list/manage sessions"),
    ("/resume [target]", "switch to another session"),
    ("/save [title]", "persist current session, optionally with a title"),
    ("/memory [...]", "memory-related subcommands"),
    ("/delete [target]", "delete a session"),
]


# --- repl -------------------------------------------------------------------
def _banner() -> str:
    """Banner shown at startup and after source/approval changes."""
    sep = f"{DIM} · {RESET}"
    parts = [f"{BOLD}nx{RESET}", f"{C.ACCENT}{st.MODEL}{RESET}"]
    if len(st.SOURCES) > 1:
        parts.append(f"{C.LABEL}{st.ACTIVE.name}{RESET}")
    if st.YOLO:
        parts.append(f"{C.SUCCESS}yolo{RESET}")
    elif st.APPROVE_LEVEL is None:
        parts.append(f"{C.WARNING}prompt:all{RESET}")
    elif st.APPROVE_LEVEL == "high":
        parts.append(f"{C.SUCCESS}auto:high{RESET}")
    elif st.APPROVE_LEVEL == "medium":
        parts.append(f"{C.WARNING}auto:medium{RESET}")
    else:
        parts.append(f"{DIM}auto:low{RESET}")
    parts.append(f"{DIM}{st.client.base_url}{RESET}")
    return sep.join(parts)


class _ReplState:
    """Mutable REPL session state, threaded through command handlers instead
    of relying on closures/nonlocal — lets every slash command live as a
    plain module-level function dispatched from one table (see
    `_REPL_COMMANDS`), rather than some commands being nested if/elif
    branches and others being separately-extracted `_cmd_*` functions."""
    __slots__ = ("messages", "session_id", "session_dirty", "history")

    def __init__(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        session_dirty: bool,
        history: list[str],
    ) -> None:
        self.messages = messages
        self.session_id = session_id
        self.session_dirty = session_dirty
        self.history = history


def _save_current(state: _ReplState, meta: dict[str, Any] | None = None) -> None:
    """Persist `state.messages` to the current session file.

    Title is auto-derived from the first user message (unless the user
    has set one explicitly with /save <title>). Source and cwd are
    recorded so a resume can re-select the right endpoint.
    """
    if state.session_id is None:
        return
    m = dict(meta or {})
    if "title" not in m:
        first_user = next(
            (msg["content"] for msg in state.messages
             if msg.get("role") == "user" and isinstance(msg.get("content"), str)),
            "",
        )
        t = _safe_title(first_user)
        if t:
            m["title"] = t
    m.setdefault("source", st.ACTIVE.name if st.ACTIVE else None)
    m.setdefault("cwd", os.getcwd())
    m.setdefault("model", st.MODEL)
    # Record the active provider routing (if any) so a /resume re-applies
    # the same OpenRouter upstream choice. Stored as a bare `order` list
    # (not the whole extra_body) to keep the session JSON minimal and to
    # avoid persisting unrelated request-body keys a future feature might add.
    if st.ACTIVE:
        order = _provider_order(st.ACTIVE)
        if order:
            m["provider"] = order
    # Merge cumulative token totals into the session JSON so they persist
    # across restarts and are available to any metrics consumer that reads
    # the session files (e.g. a dashboard). This is always-on — the numbers
    # are already in hand from the stats footer, so storing them costs
    # nothing and a local session log is useful regardless of a push sink.
    if st._METRICS is not None:
        m.update(st._METRICS.as_meta())
    try:
        _write_session(state.session_id, state.messages, m)
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't save session: {type(e).__name__}: {e}{RESET}")
    # Fire-and-forget push to the optional metrics endpoint. Only does
    # anything if NX_METRICS_URL is set; self-disables on first
    # failure so a down/absent dashboard never slows the REPL.
    if st._METRICS is not None:
        st._METRICS.flush()


def _fresh_metrics(session_id: str) -> _Metrics:
    return _Metrics(
        session_id, model=st.MODEL,
        source=st.ACTIVE.name if st.ACTIVE else None,
        push_url=_env_metrics_url())


def _reload_metrics_totals(session_id: str) -> None:
    saved = _load_session(session_id) or {}
    if isinstance(saved.get("input_tokens"), int):
        st._METRICS.input_tokens = saved.get("input_tokens") or 0
        st._METRICS.output_tokens = saved.get("output_tokens") or 0
        st._METRICS.cache_read_tokens = saved.get("cache_read_tokens") or 0
        st._METRICS.reasoning_tokens = saved.get("reasoning_tokens") or 0
        st._METRICS.api_calls = saved.get("api_calls") or 0
        if saved.get("started_at"):
            st._METRICS.started_at = saved["started_at"]


# --- slash-command handlers --------------------------------------------------
# Each handler takes (user_input, state) and returns "break" to end the REPL
# loop, or None to continue it. Dispatched by first whitespace token via
# _REPL_COMMANDS (built below, once all handlers are defined).

def _cmd_quit(user: str, state: _ReplState) -> str | None:
    if state.session_dirty:
        _save_current(state)
    return "break"


def _cmd_help(user: str, state: _ReplState) -> None:
    print(f"{DIM}  commands:{RESET}")
    for cmd, desc in _HELP_COMMANDS:
        print(f"  {C.ACCENT}{cmd:<28}{RESET} {DIM}{desc}{RESET}")


def _cmd_source(user: str, state: _ReplState) -> None:
    import nx as _m
    sp = user.split()
    if len(sp) == 1:
        print(f"{DIM}  sources:{RESET}")
        for sname in st.SOURCE_ORDER:
            src = st.SOURCES[sname]
            mark = f"{C.SUCCESS}★{RESET}" if sname == st.ACTIVE.name else f"{DIM} ·{RESET}"
            m = src.display_model()
            ctx = src._context_window
            if isinstance(ctx, int) and ctx > 0:
                ctx_s = f" · {_abbr(ctx)} ctx"
            else:
                ctx_s = ""
                _ensure_ctx_probe(src)  # warm the cache for next time
            # OpenRouter provider routing, if pinned on this source.
            prov = _provider_order(src)
            prov_s = f" · via {','.join(prov)}" if prov else ""
            print(f"  {mark} {C.ACCENT}{sname:<12}{RESET} {DIM}{m} @ {src.base_url}{ctx_s}{prov_s}{RESET}")
        print(f"{DIM}  /source <name> [model] to switch · "
              "context preserved (use /reset to clear){RESET}")
        return
    target = sp[1]
    # Optional second token: a model id to pin for this switch.
    # Useful for multi-model hosts (e.g. Together) so you can do
    #   /source together                    → default (GLM-5.2)
    #   /source together zai-org/GLM-4.6     → that model, same host
    # The override is non-sticky: a later bare /source <name>
    # always falls back to the source's configured default.
    model_override = sp[2] if len(sp) >= 3 else None
    if target not in st.SOURCES:
        avail = ", ".join(st.SOURCE_ORDER)
        print(f"{C.ERROR}  ✗ unknown source {target!r} — available: {avail}{RESET}")
        return
    if target == st.ACTIVE.name and not model_override:
        print(f"{DIM}  already on {target}{RESET}")
        return
    switch_source(target, model_override=model_override)
    src = st.SOURCES[target]
    # An explicit /source switch is user-initiated, so a synchronous
    # probe here is the right trade: the user asked for this source
    # and will want to see its context limit. Caches after the first
    # switch, so later switches to a known source are instant.
    ctx = src.resolve_context_window()
    ctx_s = f" · {_abbr(ctx)} ctx" if isinstance(ctx, int) and ctx > 0 else ""
    print(f"{BOLD}{C.WARNING}  → switched to {target}{RESET} {DIM}({st.MODEL} @ {src.base_url}{ctx_s}){RESET}")
    print(_m._banner())
    state.session_dirty = True  # source/model change is worth persisting
    if st._METRICS is not None:
        st._METRICS.model = st.MODEL
        st._METRICS.source = target


def _cmd_theme(user: str, state: _ReplState) -> None:
    import nx as _m
    sp = user.split()
    if len(sp) == 1:
        print(f"{DIM}  themes:{RESET}")
        width = max(len(t) for t in _ansi.THEMES)
        for tname in sorted(_ansi.THEMES):
            mark = f"{C.SUCCESS}★{RESET}" if tname == C.name else f"{DIM} ·{RESET}"
            print(f"  {mark} {C.ACCENT}{tname:<{width}}{RESET}  {_ansi.swatch(tname)}")
        print(f"{DIM}  /theme <name> to switch{RESET}")
        return
    target = sp[1]
    if not _ansi.set_theme(target):
        avail = ", ".join(sorted(_ansi.THEMES))
        print(f"{C.ERROR}  ✗ unknown theme {target!r} — available: {avail}{RESET}")
        return
    _set_env_var("NX_THEME", target)
    os.environ["NX_THEME"] = target
    print(f"{BOLD}{C.WARNING}  → theme: {target}{RESET} {DIM}(saved to ~/.env){RESET}")
    print(_m._banner())


def _cmd_reasoning(user: str, state: _ReplState) -> None:
    sp = user.split()
    sub = sp[1].lower() if len(sp) > 1 else None
    if sub in ("on", "show", "expand"):
        st.REASONING_VISIBLE = True
        _set_env_var("NX_REASONING_VISIBLE", "1")
        os.environ["NX_REASONING_VISIBLE"] = "1"
        print(f"{BOLD}{C.WARNING}  → reasoning: visible{RESET} {DIM}(saved to ~/.env){RESET}")
        if sub == "show" and st._LAST_REASONING:
            print(f"{C.REASONING_TEXT}  ── last reasoning ──{RESET}")
            print(f"{C.REASONING_TEXT}{st._LAST_REASONING}{RESET}")
            print(f"{DIM}  ──────────────{RESET}")
    elif sub in ("off", "hide", "collapse"):
        st.REASONING_VISIBLE = False
        _set_env_var("NX_REASONING_VISIBLE", "0")
        os.environ["NX_REASONING_VISIBLE"] = "0"
        print(f"{BOLD}{C.WARNING}  → reasoning: collapsed{RESET} {DIM}(saved to ~/.env){RESET}")
    elif sub in ("last",):
        if st._LAST_REASONING:
            print(f"{C.REASONING_TEXT}  ── last reasoning ──{RESET}")
            print(f"{C.REASONING_TEXT}{st._LAST_REASONING}{RESET}")
            print(f"{DIM}  ──────────────{RESET}")
        else:
            print(f"{DIM}  no reasoning captured yet{RESET}")
    elif sub is None:
        state_str = "visible" if st.REASONING_VISIBLE else "collapsed"
        print(f"{DIM}  reasoning: {state_str}{RESET}")
        print(f"{DIM}  /reasoning on|off · /reasoning last (show most recent, "
              f"even if collapsed){RESET}")
    else:
        print(f"{C.ERROR}  ✗ unknown /reasoning arg {sub!r} — "
              f"use on, off, or last{RESET}")


def _cmd_provider(user: str, state: _ReplState) -> None:
    # /provider — control OpenRouter upstream-provider routing for the
    # active source. OpenRouter fronts many providers behind one model
    # id; routing picks which of them serves the request (different
    # providers back the same endpoint at different prices / precisions
    # / latencies). The `provider.order` list rides in the request body
    # via extra_body (see Source.extra_request_kwargs). This only
    # meaningfully routes on OpenRouter, but is safe to call on any
    # source — non-consumers ignore unknown body keys.
    #
    #   /provider                       show the active source's routing
    #   /provider openrouter           show routing for <source>
    #   /provider parasail/fp8         pin to that provider (active src)
    #   /provider Together,DeepInfra    ordered: try Together, then DeepInfra
    #   /provider off                   clear routing (let OpenRouter pick)
    #   /provider openrouter parasail   set routing on <source>, not active
    parts = user.split()
    if len(parts) == 1:
        # /provider with no args → show the active source's routing.
        cur = _provider_order(st.ACTIVE)
        if cur:
            print(f"{DIM}  provider order: {', '.join(cur)}"
                  f"{RESET} {DIM}(on {st.ACTIVE.name}){RESET}")
        else:
            print(f"{DIM}  no provider routing set on {st.ACTIVE.name}"
                  f"{RESET}")
        print(f"{DIM}  /provider <a,b,…> to pin · "
              "/provider <source> <a,b,…> · /provider off{RESET}")
        return
    arg = parts[1]
    # /provider <source> [order]  vs  /provider <order>
    if arg in st.SOURCES and len(parts) >= 3:
        target_src = st.SOURCES[arg]
        order_str = parts[2]
    else:
        target_src = st.ACTIVE
        order_str = arg
    if order_str.lower() in ("off", "none", "clear", ""):
        _set_provider_order(target_src, [])
        print(f"{DIM}  ✓ provider routing cleared on "
              f"{target_src.name}{RESET}")
    else:
        order = [p.strip() for p in order_str.split(",") if p.strip()]
        _set_provider_order(target_src, order)
        print(f"{DIM}  ✓ provider order: {', '.join(order)}"
              f"  (on {target_src.name}, no fallback){RESET}")
    state.session_dirty = True  # routing change is worth persisting


def _cmd_resources(user: str, state: _ReplState) -> None:
    print(f"{DIM}  sampling…{RESET}")
    stats = process_stats()
    rss = stats["rss_mb"]
    rss_s = f"{rss:.1f} MB" if rss is not None else "n/a"
    print(f"{DIM}  pid {stats['pid']} · cpu {stats['cpu_percent']:.1f}% · rss {rss_s}{RESET}")


def _cmd_bg(user: str, state: _ReplState) -> None:
    parts = user.split()
    if len(parts) >= 2 and parts[1] == "kill":
        if len(parts) != 3:
            print(f"{C.WARNING}  usage: /bg kill <pid>{RESET}")
            return
        target_pid = parts[2]
        try:
            os.kill(int(target_pid), signal.SIGTERM)
            print(f"{DIM}  ↳ sent SIGTERM to {target_pid}{RESET}")
        except ProcessLookupError:
            print(f"{C.WARNING}  no such process {target_pid}{RESET}")
        except (ValueError, PermissionError) as e:
            print(f"{C.ERROR}  ✗ {e}{RESET}")
        return
    jobs = list_bg_jobs()
    if not jobs:
        print(f"{DIM}  no background jobs{RESET}")
        return
    print(f"{DIM}  background jobs (newest first):{RESET}")
    for j in jobs:
        status = f"{C.SUCCESS}running{RESET}" if j["alive"] else f"{DIM}done{RESET}"
        age = _fmt_when(j["started"]) if j["started"] else "?"
        cmd = j["command"]
        cmd = cmd if len(cmd) <= 60 else cmd[:59] + "…"
        print(f"  {DIM}{j['pid']:>7}{RESET}  {status:<18}  {DIM}{age:<10}{RESET}  {cmd}")
        if j["tail"]:
            print(f"           {DIM}└ {j['tail']}{RESET}")
    print(f"{DIM}  /bg kill <pid> to terminate a running job{RESET}")


def _cmd_yolo(user: str, state: _ReplState) -> None:
    import nx as _m
    st.YOLO = not st.YOLO
    if st.YOLO:
        st.APPROVE_LEVEL = None  # never prompt
    else:
        st.APPROVE_LEVEL = st.DEFAULT_APPROVE_LEVEL
    print(f"{DIM}  yolo={st.YOLO}  approval={_approval_display()}{RESET}")
    print(_m._banner())


def _cmd_approval(user: str, state: _ReplState) -> None:
    import nx as _m
    parts = user.split()
    if len(parts) == 1:
        # /approval with no arg → show current setting
        print(f"{DIM}  approval={_approval_display()}  (all|low|medium|high|yolo){RESET}")
        return
    arg = parts[1]
    kind, resolved = _normalize_approval(arg)
    if kind:
        _apply_approval(kind, resolved, update_default=(kind != "yolo"))
        if kind == "prompt_all":
            print(f"{DIM}  approval=all (prompt every classified action){RESET}")
        elif kind == "level":
            print(f"{DIM}  approval={resolved} (auto-allow ≤ {resolved}){RESET}")
        else:
            print(f"{DIM}  approval=off (yolo — never prompt){RESET}")
        print(_m._banner())
    else:
        print(f"{C.WARNING}  unknown level {arg!r} — want all|low|medium|high|yolo{RESET}")


def _cmd_reset(user: str, state: _ReplState) -> None:
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()
    state.messages = _initial_messages()
    st._SESSION_HAS_BACKGROUNDED = False
    st._MULTI_EDIT_NUDGED = False
    st._CONSECUTIVE_EDITS = 0
    print(f"{DIM}  context cleared{RESET}")
    # A reset forks a new session id so the fresh chat isn't written
    # over the old one (mirrors Hermes's "new session on /new").
    state.session_id = _new_session_id()
    state.session_dirty = False
    # Fresh recorder for the fresh session.
    st._METRICS = _fresh_metrics(state.session_id)
    print(f"{DIM}  new session {state.session_id}{RESET}")


def _cmd_init(user: str, state: _ReplState) -> None:
    import nx as _m
    root = st.PROJECT_ROOT or os.getcwd()
    agents_path = os.path.join(root, "AGENTS.md")
    existing = os.path.isfile(agents_path)
    if existing:
        try:
            ok = _m._confirm(f"AGENTS.md already exists at {agents_path} — overwrite/update it?")
        except _EscToChat:
            print(f"{C.WARNING}  ↳ escaped /init approval — back to chat input{RESET}")
            return
        if not ok:
            print(f"{DIM}  cancelled{RESET}")
            return
    prompt = (
        "Explore this codebase (list_dir/read_file as needed — build config, "
        "entry points, module layout, test setup, existing README/AGENTS.md/CLAUDE.md) "
        "and then write_file an AGENTS.md at the project root "
        f"({agents_path}) documenting: what this project is, build/test/run commands, "
        "and architecture notes a coding agent would need working in this repo for the "
        "first time. Be concise and specific to what you actually find — no generic "
        "boilerplate. If AGENTS.md already exists, revise it rather than starting over."
    )
    state.messages.append({"role": "user", "content": prompt})
    print()
    _run_model_turn_loop(state.messages, on_step=lambda meta=None: _save_current(state, meta))
    state.session_dirty = True
    _save_current(state)


def _cmd_recover(user: str, state: _ReplState) -> None:
    note = user.split(None, 1)[1].strip() if " " in user else ""
    content = f"[Runtime note: {MANUAL_RECOVERY_NUDGE}"
    if note:
        content += f" User note: {note}"
    content += "]"
    state.messages.append({"role": "user", "content": content})
    print(f"{C.WARNING}  ↳ manual recovery requested — forcing visible checkpoint{RESET}")
    _run_model_turn_loop(state.messages, force_final=True, recovery_sampling=True,
                         on_step=lambda meta=None: _save_current(state, meta))
    state.session_dirty = True


def _cmd_undo(user: str, state: _ReplState) -> None:
    import nx as _m
    if user in ("/undo --list", "/undo -l", "/undo list"):
        checkpoints = _m.list_checkpoints()
        if not checkpoints:
            print(f"{DIM}  nothing to undo{RESET}")
            return
        print(f"{DIM}  pending undo checkpoints (most recent first):{RESET}")
        for i, c in enumerate(checkpoints, 1):
            files = c["files"]
            paths = files[0]["path"] if len(files) == 1 else f"{len(files)} files: " + ", ".join(f["path"] for f in files)
            print(f"  {DIM}{i:>2}{RESET}  {c['tool']:<11} {paths}  {DIM}({_fmt_when(c.get('ts'))}){RESET}")
        return
    entry, msg = _m.undo_last()
    color = C.ERROR if msg.startswith("ERROR") else DIM
    print(f"{color}  ↳ {msg}{RESET}")


def _cmd_compress(user: str, state: _ReplState) -> None:
    import nx as _m
    # nothing to compress if we're under (system + KEEP) turns
    body_len = len(state.messages) - (1 if state.messages and state.messages[0].get("role") == "system" else 0)
    if body_len <= st.COMPRESS_KEEP:
        print(f"{DIM}  nothing to compress ({body_len} turn{'s' if body_len != 1 else ''} in context){RESET}")
        return
    try:
        ok = _m._confirm(f"compress {body_len - st.COMPRESS_KEEP} older turns (keep last {st.COMPRESS_KEEP})")
    except _EscToChat:
        print(f"{C.WARNING}  ↳ escaped compress approval — back to chat input{RESET}")
        return
    if not ok:
        print(f"{DIM}  cancelled{RESET}")
        return
    print(f"{DIM}  compressing…{RESET}")
    result = _m.compress(state.messages)
    if result is None:
        return  # error already printed
    kept_n, summarized_n, summary_chars = result
    print(f"{DIM}  └ compressed {summarized_n} turns → 1 summary "
          f"({summary_chars} chars), kept last {kept_n} verbatim{RESET}")
    state.session_dirty = True


def _cmd_autocompress(user: str, state: _ReplState) -> None:
    arg = user[len("/autocompress"):].strip().lower()
    if not arg:
        if st.AUTOCOMPRESS_PERCENT <= 0:
            print(f"{DIM}  auto-compress: off{RESET}")
        else:
            print(f"{DIM}  auto-compress: {st.AUTOCOMPRESS_PERCENT}% "
                  f"(auto-compress when context ≥ {st.AUTOCOMPRESS_PERCENT}% full){RESET}")
        return
    if arg in ("off", "disable", "0"):
        st.AUTOCOMPRESS_PERCENT = 0
        print(f"{DIM}  auto-compress: off (disabled){RESET}")
        return
    if arg in ("on", "enable"):
        st.AUTOCOMPRESS_PERCENT = st.AUTOCOMPRESS_DEFAULT
        print(f"{DIM}  auto-compress: {st.AUTOCOMPRESS_PERCENT}% "
              f"(auto-compress when context ≥ {st.AUTOCOMPRESS_PERCENT}% full){RESET}")
        return
    try:
        val = int(arg)
    except ValueError:
        print(f"{C.WARNING}  usage: /autocompress [<1-100|off|on>] — "
              f"current: {st.AUTOCOMPRESS_PERCENT}%{RESET}")
        return
    if not (1 <= val <= 100):
        print(f"{C.WARNING}  {val} is out of range — want 1–100 (or off / on){RESET}")
        return
    st.AUTOCOMPRESS_PERCENT = val
    print(f"{DIM}  auto-compress: {val}% "
          f"(auto-compress when context ≥ {val}% full){RESET}")


def _cmd_sessions_handler(user: str, state: _ReplState) -> None:
    _cmd_sessions(user, state.session_id)


def _cmd_resume_handler(user: str, state: _ReplState) -> None:
    new_sid = _cmd_resume(user, state.session_id)
    if not new_sid:
        return
    # Persist the outgoing session before switching away.
    if state.session_dirty:
        _save_current(state)
    state.session_id = new_sid
    state.messages[:] = _session_messages(new_sid)
    st._SESSION_HAS_BACKGROUNDED = _messages_show_prior_backgrounding(state.messages)
    st._MULTI_EDIT_NUDGED = _messages_show_prior_multi_edit_nudge(state.messages)
    state.session_dirty = False
    # Reload the cumulative-token recorder from the resumed
    # session's saved totals so they keep climbing from here.
    st._METRICS = _fresh_metrics(new_sid)
    _reload_metrics_totals(new_sid)


def _cmd_save(user: str, state: _ReplState) -> None:
    title = user.split(None, 1)[1].strip() if " " in user else None
    _save_current(state, {"title": title} if title else {})
    state.session_dirty = False
    shown = f" as {title!r}" if title else ""
    print(f"{DIM}  ✓ saved session {state.session_id}{shown}{RESET}")


def _cmd_memory_handler(user: str, state: _ReplState) -> None:
    _cmd_memory(state.messages, user)


def _cmd_delete_handler(user: str, state: _ReplState) -> None:
    target = user.split(None, 1)[1].strip() if " " in user else None
    _cmd_delete(target, state.session_id)


CommandHandler = Callable[[str, _ReplState], "str | None"]

_REPL_COMMANDS: dict[str, CommandHandler] = {
    "/quit": _cmd_quit,
    "/exit": _cmd_quit,
    "/help": _cmd_help,
    "/source": _cmd_source,
    "/theme": _cmd_theme,
    "/reasoning": _cmd_reasoning,
    "/provider": _cmd_provider,
    "/resources": _cmd_resources,
    "/resmon": _cmd_resources,
    "/bg": _cmd_bg,
    "/yolo": _cmd_yolo,
    "/approval": _cmd_approval,
    "/reset": _cmd_reset,
    "/clear": _cmd_reset,
    "/new": _cmd_reset,
    "/init": _cmd_init,
    "/recover": _cmd_recover,
    "/undo": _cmd_undo,
    "/compress": _cmd_compress,
    "/compact": _cmd_compress,
    "/autocompress": _cmd_autocompress,
    "/sessions": _cmd_sessions_handler,
    "/resume": _cmd_resume_handler,
    "/save": _cmd_save,
    "/memory": _cmd_memory_handler,
    "/delete": _cmd_delete_handler,
}


def _repl() -> None:
    """The interactive loop: banner, ctx-probe warm-up, session state
    resolution/resume, then the read → dispatch → run-turn → autosave cycle
    until /quit or EOF. Slash commands are dispatched via `_REPL_COMMANDS`."""
    import nx as _m

    # The banner is printed at startup and after source/approval changes.
    print(_m._banner())
    print()
    # Warm the active source's max-context-window cache before the first
    # prompt: the probe (esp. on a still-loading local server, whose /v1/models
    # + /props 503 while mmapping) can take seconds-to-tens-of-seconds, but the
    # user typically spends that long typing their first message — so by the
    # time turn 1 finishes the cache is often already filled and the footer
    # shows the "/<max>" suffix immediately instead of a turn later. Background,
    # non-blocking; degrades to "no suffix" if it never resolves.
    if st.ACTIVE is not None:
        _ensure_ctx_probe(st.ACTIVE)
    messages = _initial_messages()
    history = []  # past user submissions, newest last; Up/Down navigates

    # --- session state ------------------------------------------------------
    # Every nx run has a session id. If the user passed --resume / -r
    # (with an id, prefix, index, or title) or --session <id>, we load that;
    # otherwise a fresh id is minted. The session is auto-saved after each
    # model turn (the messages array is the source of truth — we just write
    # it to ~/.nx/sessions/<id>.json atomically). A session that never
    # got any user input is never written to disk.
    session_id = _session_id_from_args()
    _resume_requested = any(a in ("--resume", "-r") for a in sys.argv[1:])
    if session_id is None:
        if _resume_requested:
            # bare `nx --resume` but no saved sessions exist — start fresh.
            print(f"{DIM}  no saved sessions to resume — starting fresh{RESET}")
        session_id = _new_session_id()
    session_dirty = False  # has the in-memory context diverged from disk?
    if _resume_requested or any(a == "--session" for a in sys.argv[1:]):
        # Only attempt to load when the user explicitly asked to resume a
        # specific (or most-recent) session. A plain `nx` run mints a
        # fresh id above and starts with an empty context.
        data = _load_session(session_id)
        if data and isinstance(data.get("messages"), list):
            messages = data["messages"]
            # Drop the old system prompt and inject the current one so a
            # resumed session picks up any SYSTEM edits / tool changes.
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": SYSTEM})
            st._SESSION_HAS_BACKGROUNDED = _messages_show_prior_backgrounding(messages)
            st._MULTI_EDIT_NUDGED = _messages_show_prior_multi_edit_nudge(messages)
            n_msgs = len([m for m in messages if m.get("role") != "system"])
            title = data.get("title") or "(untitled)"
            print(f"{DIM}  ↻ resumed session {session_id} ({title!r}, "
                  f"{n_msgs} messages){RESET}")
            print()
            # Show the conversation history so the user immediately remembers
            # what the session was about before the first prompt.
            if n_msgs:
                print(f"{DIM}  ── transcript ──{RESET}")
                _print_transcript(messages)
                print(f"{DIM}  ──────────────{RESET}")
                print()
            # Restore the active source (and any model override) the session
            # was started on, so a resume lands on the same endpoint + model.
            src_name = data.get("source")
            if src_name:
                restore_source(src_name, model=data.get("model"))
                # Re-apply provider routing that was in effect when the session
                # was saved, so an OpenRouter /provider choice survives a
                # resume. A saved "provider" list rebuilds the routing object.
                saved_prov = data.get("provider")
                if isinstance(saved_prov, list) and st.ACTIVE and src_name in st.SOURCES:
                    _set_provider_order(st.SOURCES[src_name], saved_prov)
            session_dirty = False  # just loaded — in sync with disk
        elif _resume_requested:
            print(f"{C.WARNING}  ✗ no saved session found for {session_id!r}; "
                  f"starting fresh{RESET}")

    # --- usage metrics -------------------------------------------------------
    # Build the cumulative-token recorder for this session. When resuming, we
    # reload its running totals from the saved session JSON so they keep
    # climbing across restarts instead of resetting to zero. The optional
    # HTTP push (NX_METRICS_URL) is off by default — see _Metrics.
    st._METRICS = _fresh_metrics(session_id)
    if _resume_requested or any(a == "--session" for a in sys.argv[1:]):
        _reload_metrics_totals(session_id)

    state = _ReplState(messages, session_id, session_dirty, history)

    while True:
        try:
            # Esc-then-type: if the user was mid-line typing a redirect when
            # the interrupt's disarm handshake completed, resume the prompt
            # with those characters already there instead of blank.
            prefill = st._INTERRUPT_REDIRECT_PREFILL
            st._INTERRUPT_REDIRECT_PREFILL = ""
            user = _m.read_multiline(initial=prefill, history=state.history)
        except (EOFError, KeyboardInterrupt):
            print()
            # Flush the current session on Ctrl-D / Ctrl-C exit so the last
            # turn isn't lost (matches Hermes's exit-path flush).
            if state.session_dirty:
                _save_current(state)
            # Show a grey resume hint so the user can pick up right where they
            # left off without having to run `nx sessions` first.
            if state.session_id:
                print(f"{DIM}  resume with: nx --resume {state.session_id}{RESET}")
            break
        user = user.strip()
        if not user:
            continue
        cmd = user.split(None, 1)[0]
        handler = _REPL_COMMANDS.get(cmd) if cmd.startswith("/") else None
        if handler is not None:
            if handler(user, state) == "break":
                break
            continue
        # record for history (skip duplicates of the very last entry so
        # Up doesn't immediately re-show what was just submitted)
        if not state.history or state.history[-1] != user:
            state.history.append(user)
        print()  # breathing room before the spinner/text starts
        state.messages.append({"role": "user", "content": expand_at_mentions(user)})
        _run_model_turn_loop(state.messages, on_step=lambda meta=None: _save_current(state, meta))
        # Auto-save after the turn settles (whether it ended cleanly, hit the
        # step cap, or was escaped). This is the Hermes pattern: persist every
        # turn so a crash / accidental close never loses work.
        state.session_dirty = True
        _save_current(state)
        # Maybe auto-compress: if the context window is over the configured
        # fill threshold, silently fold the older turns so the chat can keep
        # going. Done after the save so the pre-compression state is on disk;
        # the compressed result is persisted by the _save_current() below.
        if _maybe_autocompress(state.messages, st._LAST_PROMPT_TOKENS):
            _save_current(state)
        # Maybe refresh the model-generated description (a cheap non-streaming
        # call every SESSION_DESC_REFRESH turns). Done after the save so the
        # turn's messages are already on disk; the description lands as a
        # second write. Failures are silent — the title is always present.
        _maybe_refresh_description(state.session_id, state.messages)
        state.session_dirty = False



def _session_id_from_args() -> str | None:
    """Resolve a starting session id from CLI flags: --resume/-r and --session.

    `--resume` with no following target resumes the most recent session
    (the natural "pick up where I left off" intent). With a target it resolves
    by index/id/prefix/title. `--session <id>` forces an exact id (creates if
    absent — useful for scripts that want a stable id)."""
    args = sys.argv[1:]
    out = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--resume", "-r"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                # --resume <target>
                target = args[i + 1]
                sessions = _list_sessions(limit=50)
                out = _resolve_session(target, sessions) or target
                i += 2
            else:
                # bare --resume → most recent session, if any
                sessions = _list_sessions(limit=1)
                out = sessions[0]["id"] if sessions else None
                i += 1
            continue
        if a == "--session" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
            continue
        i += 1
    return out


def _session_messages(session_id: str) -> list[dict[str, Any]]:
    """Load just the messages list for a session, with the live SYSTEM prepended."""
    data = _load_session(session_id) or {}
    msgs = [m for m in data.get("messages", []) if m.get("role") != "system"]
    msgs.insert(0, {"role": "system", "content": SYSTEM})
    return msgs


def _fmt_when(ts: float | None) -> str:
    """Compact relative timestamp for the session list."""
    if not ts:
        return "?"
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _print_transcript(messages: list[dict[str, Any]], max_chars: int = 120) -> int:
    """Render a session's message history as a one-line-per-message recap.

    Used on resume so the user immediately sees what the conversation was
    about, and by `/sessions <id>` for its detail view. Each message is
    collapsed to a single line (newlines → spaces) and truncated so the whole
    history reads as a scannable timeline rather than a wall of text. Tool
    calls render as `→ name(...)` so you can see what ran. System messages
    are skipped. Returns the number of lines printed."""
    printed = 0
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue
        content = _flatten_message_text(m, per_msg=max_chars)
        color = {"user": C.ACCENT, "assistant": C.TEXT, "tool": DIM}.get(role, "")
        print(f"  {color}{role:>9}{RESET}  {content}")
        printed += 1
    return printed


def _session_list_page_options(
    args: list[str], default_limit: int = SESSION_LIST_DEFAULT_LIMIT,
) -> tuple[str | None, int, int, list[str]]:
    """Parse paging flags and leave everything else as the search query."""
    page = 1
    limit = default_limit
    query_parts = []
    warnings = []
    i = 0

    def positive_int(value, fallback, name):
        try:
            n = int(value)
        except (TypeError, ValueError):
            warnings.append(f"ignored invalid {name}: {value!r}")
            return fallback
        if n < 1:
            warnings.append(f"ignored invalid {name}: {value!r}")
            return fallback
        return n

    while i < len(args):
        a = args[i]
        if a in ("--page", "-p"):
            if i + 1 >= len(args):
                warnings.append(f"ignored missing value for {a}")
                i += 1
                continue
            page = positive_int(args[i + 1], page, "page")
            i += 2
            continue
        if a.startswith("--page="):
            page = positive_int(a.split("=", 1)[1], page, "page")
            i += 1
            continue
        if a in ("--limit", "-n"):
            if i + 1 >= len(args):
                warnings.append(f"ignored missing value for {a}")
                i += 1
                continue
            limit = positive_int(args[i + 1], limit, "limit")
            i += 2
            continue
        if a.startswith("--limit="):
            limit = positive_int(a.split("=", 1)[1], limit, "limit")
            i += 1
            continue
        query_parts.append(a)
        i += 1

    limit = max(1, min(limit, SESSION_LIST_MAX_LIMIT))
    query = " ".join(query_parts).strip() or None
    return query, page, limit, warnings


def _print_session_list(
    sessions: list[dict[str, Any]], start_index: int = 1, current_id: str | None = None,
) -> None:
    for offset, s in enumerate(sessions):
        i = start_index + offset
        title = s["title"] or "(empty)"
        if current_id is not None:
            mark = f"{C.SUCCESS}●{RESET}" if s["id"] == current_id else f"{DIM}{i:>3}{RESET}"
            prefix = f"  {mark} "
        else:
            prefix = f"  {DIM}{i:>3}{RESET}  "
        sid = s.get("id") or ""
        print(f"{prefix}{C.LABEL}{sid}{RESET}  {title}")

        meta = [f"{s['n']} msg", _fmt_when(s["updated_at"])]
        if s.get("source"):
            meta.append(f"source {s['source']}")
        if s.get("model"):
            meta.append(f"model {s['model']}")
        cwd = _display_path(s.get("cwd"))
        if cwd:
            meta.append(f"cwd {cwd}")
        print(f"       {DIM}{' · '.join(meta)}{RESET}")

        detail = s.get("description")
        label = "desc"
        if not detail and s.get("preview"):
            detail = s["preview"]
            label = "first"
        if detail:
            print(f"       {DIM}{label}: {detail}{RESET}")


def _session_next_hint(command: str, page: int, limit: int, query: str | None = None) -> str:
    quoted_query = f" {shlex.quote(query)}" if query else ""
    return f"{command}{quoted_query} --page {page + 1} --limit {limit}"


def _display_path(path: str | None) -> str | None:
    if not path:
        return None
    path = os.path.expanduser(str(path))
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _cli_sessions(args: list[str]) -> bool:
    """`nx sessions [query]` — list saved sessions and exit (no REPL).

    With no query: prints the first page of recent sessions. With a query:
    substring-filters (case-insensitive) across title, description, preview,
    and id, so once
    you have dozens of sessions you can narrow with e.g.
    `nx sessions refactor`. Add `--page/-p` and `--limit/-n` to page
    through the list. Returns True if it handled a listing request (caller
    should then exit); False if this wasn't a sessions invocation.

    Deliberately a top-level verb rather than `--resume list`: "list" looks
    like it could be a session title, and a "resume" flag doing two different
    things (list vs. switch) depending on whether it has an arg is the kind
    of inconsistency that confuses. `sessions` = discover, `--resume` = enter.
    """
    if not args:
        return False
    if args[0] not in ("sessions", "--sessions", "ls", "list"):
        return False
    query, page, limit, warnings = _session_list_page_options(args[1:])
    offset = (page - 1) * limit
    fetched = _list_sessions(limit=limit + 1, offset=offset, query=query)
    sessions = fetched[:limit]
    has_next = len(fetched) > limit
    for warning in warnings:
        print(f"{C.WARNING}  {warning}{RESET}")
    if not sessions:
        if query:
            print(f"  no sessions matching {query!r} on page {page}")
        else:
            if page > 1:
                print(f"  no sessions on page {page}  ({_sessions_dir()})")
            else:
                print(f"  no saved sessions yet  ({_sessions_dir()})")
        return True
    where = f" matching {query!r}" if query else ""
    print(f"{DIM}  sessions{where} · page {page} · {_sessions_dir()}{RESET}")
    _print_session_list(sessions, start_index=offset + 1)
    print(f"{DIM}  resume with: nx --resume <n|short-id|title>{RESET}")
    if has_next:
        print(f"{DIM}  next page: {_session_next_hint('nx sessions', page, limit, query)}{RESET}")
    return True


def _cli_tokens(args: list[str]) -> bool:
    """`nx --tokens` / `nx --bench` — print the exact prompt-token
    floor (SYSTEM + the active TOOLS schema + a 1-token "hey") for the active
    backend/model, then exit. Turns the README's "~625 tokens on a bare hey"
    claim into something mechanically checked per-backend/per-edit, instead
    of an eyeballed footer. Returns True if it handled the flag (caller
    should then exit); False otherwise.
    """
    if not any(a in ("--tokens", "--bench") for a in args):
        return False
    tools = _active_tools()
    if st.ACTIVE is not None:
        probe = st.ACTIVE.tokenize_count("hey")
    else:
        probe = None
    if probe is not None:
        def count(text):
            return st.ACTIVE.tokenize_count(text)
        method = f"server /tokenize ({st.ACTIVE.name})"
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")

            def count(text):
                return len(enc.encode(text))
            method = "tiktoken/cl100k_base (approximate for non-OpenAI models; no /tokenize endpoint found)"
        except ImportError:
            def count(text):
                return max(1, round(len(text) / 3.5))
            method = "chars/3.5 estimate (no /tokenize endpoint found; pip install tiktoken for a closer approximation)"
    system_tokens = count(SYSTEM)
    tools_tokens = count(json.dumps(tools))
    user_tokens = count("hey")
    total = system_tokens + tools_tokens + user_tokens
    backend = f"{st.ACTIVE.name} · {st.MODEL}" if st.ACTIVE and st.MODEL else (st.MODEL or "(no source configured)")
    bg_note = "" if st._SESSION_HAS_BACKGROUNDED else ", wait_background excluded — nothing has backgrounded yet"
    print(f"nx prompt-token floor — {method}")
    print(f"  backend   {backend}")
    print(f"  SYSTEM    {system_tokens:>5} tokens")
    print(f"  TOOLS     {tools_tokens:>5} tokens  ({len(tools)} functions{bg_note})")
    print(f"  'hey'     {user_tokens:>5} tokens")
    print(f"  {'-' * 28}")
    print(f"  total     {total:>5} tokens  (+ ~50 of server-side chat-template framing, not counted here)")
    return True


def _cmd_sessions(user: str, current_id: str | None) -> None:
    """Handle /sessions [n] — list recent sessions, or show one in detail."""
    parts = user.split()
    args = parts[1:]
    list_requested = not args or args[0] in ("--page", "-p", "--limit", "-n")
    list_requested = list_requested or (args and (
        args[0].startswith("--page=")
        or args[0].startswith("--limit=")
        or args[0] in ("page", "p")
    ))
    if list_requested:
        page_args = args
        if page_args and page_args[0] in ("page", "p"):
            page_args = ["--page", *page_args[1:]]
        query, page, limit, warnings = _session_list_page_options(page_args)
        offset = (page - 1) * limit
        fetched = _list_sessions(limit=limit + 1, offset=offset, query=query)
        sessions = fetched[:limit]
        has_next = len(fetched) > limit
        for warning in warnings:
            print(f"{C.WARNING}  {warning}{RESET}")
        if not sessions:
            if query:
                print(f"{DIM}  no sessions matching {query!r} on page {page}{RESET}")
            elif page > 1:
                print(f"{DIM}  no sessions on page {page} ({_sessions_dir()}){RESET}")
            else:
                print(f"{DIM}  no saved sessions yet (saved to {SESSION_HOME}/sessions){RESET}")
            return
        where = f" matching {query!r}" if query else ""
        print(f"{DIM}  recent sessions{where} · page {page} ({_sessions_dir()}):{RESET}")
        _print_session_list(sessions, start_index=offset + 1, current_id=current_id)
        print(f"{DIM}  /resume <n|short-id|title> to switch · /delete <n|id> to remove{RESET}")
        if has_next:
            query_tail = f" {shlex.quote(query)}" if query else ""
            print(f"{DIM}  next page: /sessions --page {page + 1} --limit {limit}{query_tail}{RESET}")
        return
    # /sessions <target> — show detail for one session
    arg = " ".join(args).strip()
    sessions = _list_sessions(limit=50)
    sid = _resolve_session(arg, sessions)
    if not sid:
        print(f"{C.WARNING}  no session matching {arg!r}{RESET}")
        return
    data = _load_session(sid)
    if not data:
        print(f"{C.WARNING}  ✗ couldn't read session {sid}{RESET}")
        return
    msgs = data.get("messages", [])
    title = data.get("title") or "(untitled)"
    print(f"{DIM}  session {sid}{RESET}")
    print(f"{DIM}  title: {title}{RESET}")
    print(f"{DIM}  messages: {len(msgs)} · updated {_fmt_when(data.get('updated_at', 0))}{RESET}")
    print(f"{DIM}  ── transcript ──{RESET}")
    _print_transcript(msgs)


def _cmd_resume(user: str, current_id: str | None) -> str | None:
    """Handle /resume [target] — pick a session to resume. Returns the new id
    or None if nothing changed."""
    parts = user.split(None, 1)
    target = parts[1].strip() if len(parts) > 1 else ""
    sessions = _list_sessions(limit=15)
    if not target:
        if not sessions:
            print(f"{DIM}  no saved sessions to resume{RESET}")
            return None
        print(f"{DIM}  recent sessions:{RESET}")
        for i, s in enumerate(sessions, 1):
            mark = f"{C.SUCCESS}●{RESET}" if s["id"] == current_id else f"{DIM}{i:>2}{RESET}"
            title = s["title"] or "(empty)"
            print(f"  {mark} {C.LABEL}{s.get('short', _short_id(s['id']))}{RESET}  "
                  f"{title}{DIM} · {s['n']} msg · {_fmt_when(s['updated_at'])}{RESET}")
            desc = s.get("description") or (s.get("preview") if s.get("preview") != title else None)
            if desc:
                print(f"        {DIM}{desc}{RESET}")
        print(f"{DIM}  /resume <n|short-id|prefix|title> to switch{RESET}")
        return None
    target = target.strip()
    # Let the user strip surrounding <> [] "" they may have copied from help.
    if len(target) >= 2 and target[0] in "<[\"'" and target[-1] in ">]\"'":
        target = target[1:-1].strip()
    sessions = _list_sessions(limit=50)
    sid = _resolve_session(target, sessions)
    if not sid:
        print(f"{C.WARNING}  ✗ no session matching {target!r}{RESET}")
        return None
    if sid == current_id:
        print(f"{DIM}  already on that session{RESET}")
        return None
    data = _load_session(sid)
    if not data:
        print(f"{C.WARNING}  ✗ couldn't read session {sid}{RESET}")
        return None
    # Reselect the recorded source (and model override) if configured, so a
    # resume lands on the same endpoint + model the session started on.
    src = data.get("source")
    if src and restore_source(src, model=data.get("model")):
        print(f"{BOLD}{C.WARNING}  → source {src}{RESET} {DIM}({st.MODEL}){RESET}")
    n_msgs = len([m for m in data.get("messages", []) if m.get("role") != "system"])
    title = data.get("title") or "(untitled)"
    print(f"{DIM}  ↻ resumed {sid} ({title!r}, {n_msgs} messages){RESET}")
    print()
    return sid


def _cmd_delete(target: str | None, current_id: str | None) -> None:
    """Handle /delete [target] — remove a session file."""
    if not target:
        print(f"{C.WARNING}  usage: /delete <n|id|prefix>  (use /sessions to list){RESET}")
        return
    sessions = _list_sessions(limit=50)
    sid = _resolve_session(target, sessions)
    if not sid:
        print(f"{C.WARNING}  ✗ no session matching {target!r}{RESET}")
        return
    if sid == current_id:
        print(f"{C.WARNING}  can't delete the session you're currently in — /reset first{RESET}")
        return
    if _delete_session(sid):
        print(f"{DIM}  ✓ deleted session {sid}{RESET}")
    else:
        print(f"{C.WARNING}  ✗ couldn't delete session {sid}{RESET}")


def _cmd_memory(messages: list[dict[str, Any]], user_input: str) -> None:
    """Handle /memory save|remember|list  — manual memories.

    /memory save [focus...]      — distill session into a memory (optional focus
                                   guidance for the model: e.g. "focus on config")
    /memory remember <query>     — keyword search through saved memories
    /memory list                 — list all saved memories
    """
    parts = user_input.split(None, 2)
    sub = parts[1].lower() if len(parts) >= 2 else ""

    if sub == "list":
        _cmd_memory_list()
        return

    if sub == "remember":
        query = parts[2] if len(parts) >= 3 else ""
        _cmd_memory_remember(query)
        return

    if sub == "save":
        notes = parts[2] if len(parts) >= 3 else None
        _cmd_memory_save(messages, notes)
        return

    # Bare /memory — show usage
    print(f"{DIM}  usage:{RESET}")
    print(f"{DIM}    /memory save [focus...]   save a memory (optional focus guidance){RESET}")
    print(f"{DIM}    /memory remember <query>   search memories{RESET}")
    print(f"{DIM}    /memory list               list all memories{RESET}")


def _slugify(text: str, maxlen: int = 60) -> str:
    """Turn a title into a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if len(s) > maxlen:
        s = s[:maxlen].rstrip("-")
    return s or "untitled"


def _cmd_memory_save(messages: list[dict[str, Any]], notes: str | None = None) -> None:
    """Save a memory to ~/.nx/memories/<slug>.md.

    The model always distills the session into a markdown memory note.
    ``notes``, if provided, serves as guidance: "focus on the build system",
    "capture the API endpoint details", etc.
    """
    memories_dir = _memories_dir()
    os.makedirs(memories_dir, exist_ok=True)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    fname_stamp = time.strftime("%Y%m%d-%H%M%S")

    print(f"{DIM}  distilling session into a memory…{RESET}")
    transcript = _render_transcript_for_memory(messages)

    focus = f"\nThe user specifically asks to focus on: {notes}\n" if notes else ""
    prompt = (
        "You are a developer's memory-keeper. Given the following coding session transcript, "
        "extract the key learnings, discoveries, and solutions that are worth remembering "
        "for future work on this project. Focus on:\n"
        "- What problem was being solved\n"
        "- What approach worked (and what didn't)\n"
        "- Specific file paths, commands, config values, or API quirks that were discovered\n"
        "- Why things work the way they do (the reasoning, not just the result)\n"
        f"{focus}"
        "Write this as a concise, knowledge-rich markdown note that another developer "
        "(or yourself weeks from now) can read and immediately understand. "
        "Use backticks for code, paths, and commands. "
        "Start with a `# Title` line that captures the topic.\n\n"
        f"---\n{transcript}\n---"
    )
    try:
        _log_event("req", {"model": st.MODEL, "messages": [{"role": "user", "content": prompt}],
                           "stream": False, "_purpose": "memory_save"})
        resp = st.client.chat.completions.create(
            model=st.MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False, timeout=30,
            **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
        try:
            _log_event("resp", {"_purpose": "memory_save", "data": resp.model_dump()})
        except Exception:
            pass
    except APIConnectionError:
        print(f"{C.ERROR}  ✗ can't reach {st.client.base_url} — memory not saved{RESET}")
        return
    except Exception as e:
        print(f"{C.ERROR}  ✗ memory save failed: {type(e).__name__}: {e}{RESET}")
        return

    body = (resp.choices[0].message.content or "").strip()
    if not body:
        print(f"{C.ERROR}  ✗ model returned empty — memory not saved{RESET}")
        return

    # Extract title from the markdown H1
    title = ""
    for line in body.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        title = body.split("\n")[0][:80].strip() or f"Memory {fname_stamp}"

    # Write the file — timestamp_slug.md so it's human-readable on the filesystem
    slug = _slugify(title)
    fname = f"{fname_stamp}_{slug}.md"
    content = f"{body}\n\n---\n_Saved on {stamp}_\n"
    path = os.path.join(memories_dir, fname)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"{DIM}  ✓ saved memory: {title}{RESET}")
        print(f"{DIM}    {path}{RESET}")
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't write memory: {e}{RESET}")


def _cmd_memory_remember(query: str) -> None:
    """Keyword-search through saved memories."""
    memories_dir = _memories_dir()
    if not os.path.isdir(memories_dir):
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    if not query:
        print(f"{C.WARNING}  usage: /memory remember <search terms>{RESET}")
        return

    q = query.lower()
    files = []
    try:
        for name in sorted(os.listdir(memories_dir), reverse=True):
            if not name.endswith(".md"):
                continue
            path = os.path.join(memories_dir, name)
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue
            if q in content.lower():
                # Extract title from first H1 or first line
                title = ""
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                if not title:
                    title = name
                files.append((name, title, content))
    except OSError:
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    if not files:
        print(f"{DIM}  no memories matching {query!r}{RESET}")
        return

    print(f"{DIM}  memories matching {query!r}:{RESET}")
    for name, title, content in files:
        print(f"\n{C.ACCENT}  ┌─ {title}{RESET}  {DIM}({name}){RESET}")
        # Show first ~10 lines as preview, strip the title line
        lines = content.split("\n")
        shown = 0
        for line in lines:
            if line.startswith("# ") and shown == 0:
                continue  # skip the title line in preview (already shown)
            if shown >= 10:
                print(f"{DIM}  │  …{RESET}")
                break
            print(f"{DIM}  │{RESET} {line[:100]}")
            shown += 1


def _cmd_memory_list() -> None:
    """List all saved memories with titles and dates."""
    memories_dir = _memories_dir()
    if not os.path.isdir(memories_dir):
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    files = []
    try:
        for name in os.listdir(memories_dir):
            if not name.endswith(".md"):
                continue
            path = os.path.join(memories_dir, name)
            try:
                mtime = os.path.getmtime(path)
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                title = first_line[2:] if first_line.startswith("# ") else name
                files.append((mtime, name, title))
            except OSError:
                continue
    except OSError:
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    if not files:
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    files.sort(key=lambda x: x[0], reverse=True)
    print(f"{DIM}  memories ({len(files)}):{RESET}")
    for i, (mtime, name, title) in enumerate(files, 1):
        date = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        print(f"  {C.SUCCESS}{i:>2}.{RESET} {title}  {DIM}({date}){RESET}")


def _render_transcript_for_memory(messages: list[dict[str, Any]]) -> str:
    """Render session messages into a compact text transcript for the model.

    Skips the system prompt, truncates long tool outputs, and shows tool calls.
    """
    out = []
    for m in messages:
        if m.get("role") == "system":
            continue
        role = m.get("role", "?")
        text = _flatten_message_text(m, per_msg=2000, tool_call_arg_chars=100)
        out.append(f"[{role}] {text}")
    return "\n\n".join(out)
