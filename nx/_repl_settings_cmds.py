"""Slash-command handlers for REPL settings/session-context toggles that
don't touch the saved-session list: `/source`, `/theme`, `/reasoning`,
`/provider`, `/resources`, `/bg`, `/yolo`, `/approval`, `/reset`, `/help`,
`/init`, `/recover`, `/undo`, `/compress`, `/autocompress`.

Each handler takes (user_input, state) and returns "break" to end the REPL
loop, or None to continue it — dispatched from `_REPL_COMMANDS` in
`_repl.py`.

Functions here that are monkeypatched by tests via `m.<name> = fake` are
always called through a lazy `import nx as _m; _m.<name>(...)` — the patch
only ever lands on the top-level `nx` module's attribute, never on this
submodule's own globals. See `_turn.py` / `_repl.py` for the same pattern.
"""
from __future__ import annotations

import os
import signal
import sys

from . import _ansi
from ._ansi import DIM, BOLD, RESET, C
from ._approval import _EscToChat, _normalize_approval, _apply_approval, _approval_display
from ._env import _set_env_var
from ._metrics import _abbr
from ._resources import process_stats
from ._sessions_cmd import _fmt_when
from ._sources import switch_source, _set_provider_order, _provider_order
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from . import _state as st
from . import MANUAL_RECOVERY_NUDGE, _initial_messages, _new_session_id, list_bg_jobs
from ._repl_state import _ReplState, _save_current, _fresh_metrics


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
    active_name = st.ACTIVE.name if st.ACTIVE is not None else None
    if len(sp) == 1:
        print(f"{DIM}  sources:{RESET}")
        for sname in st.SOURCE_ORDER:
            src = st.SOURCES[sname]
            mark = f"{C.SUCCESS}★{RESET}" if sname == active_name else f"{DIM} ·{RESET}"
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
    if target == active_name and not model_override:
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
    if st.ACTIVE is None:
        print(f"{C.ERROR}  ✗ no active source{RESET}")
        return
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
        known_pids = {j["pid"] for j in list_bg_jobs()}
        if target_pid not in known_pids:
            print(f"{C.WARNING}  {target_pid} is not a tracked background job (see /bg){RESET}")
            return
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
    _run_model_turn_loop(
        state.messages,
        on_step=lambda meta=None: _save_current(state, meta, update_index=False),
    )
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
    # Compression itself is a model call; save now so its usage is visible in
    # the session JSON even before another conversational turn or /quit.
    _save_current(state)


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
