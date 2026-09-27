"""Slash-command handlers for REPL settings/session-context toggles that
don't touch the saved-session list: `/source`, `/theme`, `/reasoning`,
`/tools` (legacy alias of `/tooloutput`), `/provider`, `/budget`, `/resources`,
`/bg`, `/yolo`, `/approval`, `/reset`,
`/help`, `/init`, `/recover`, `/undo`, `/compress`, `/autocompress`.

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
from ._subagent import list_async_agents
from ._repl_state import _ReplState, _save_current, _fresh_metrics
from ._todo import render_todos_display


_HELP_COMMANDS: list[tuple[str, str]] = [
    ("/help", "show this list of commands"),
    ("/quit, /exit", "save session (if dirty) and exit"),
    ("/source [name] [model]", "list or switch model source/backend"),
    ("/theme [name]", "list or switch color theme"),
    ("/reasoning [on|off|show|hide|last]", "toggle/show reasoning-content visibility"),
    ("/tooloutput [on|off|show|hide|last], /tools", "toggle/show tool-call result visibility"),
    ("/toolcalls [text|native|auto] [reset [tool]]", "show/set tool-calling mode, or clear the malformed-args breaker streaks"),
    ("/provider [source] [order|off]", "set/clear OpenRouter provider routing"),
    ("/resources, /resmon", "show CPU%/RSS process stats"),
    ("/bg [kill <pid>]", "list/kill background bash jobs"),
    ("/yolo", "toggle yolo mode (disable all approval prompts)"),
    ("/approval [level]", "show/set approval level (session-only)"),
    ("/approval set-default <level>", "persist the default approval mode to ~/.env"),
    ("/approval allow [list|forget <n>|<tool> <pattern>]", "manage the persistent approval allowlist"),
    ("/reset, /clear, /new", "clear context, start a new session"),
    ("/init", "have the model write/update AGENTS.md"),
    ("/recover [note]", "force a visible recovery checkpoint turn"),
    ("/undo [--list]", "undo last file change / list undo checkpoints"),
    ("/compress, /compact", "fold older turns into a summary"),
    ("/ctx [all|pin|unpin|clear]", "token breakdown by message; pin turns to survive compress"),
    ("/cost", "session usage + per-source cost scoreboard"),
    ("/status", "snapshot: source, model, caps, budgets, fence, approval"),
    ("/budget [<tokens|off>]", "cap the model's reasoning phase (reasoning token budget)"),
    ("/autocompress [percent|on|off]", "configure automatic compression threshold"),
    ("/sessions [...]", "list/manage sessions (non-title query → full-text body search)"),
    ("/export [md] [path]", "write the current session to a markdown file"),
    ("/resume [target]", "switch to another session"),
    ("/save [title]", "persist current session, optionally with a title"),
    ("/fork [title]", "branch the current conversation into a new session (original kept; prompt-cache-friendly)"),
    ("/memory [save|remember|delete|list|optimize]", "save a distilled memory, search/remove/list saved memories, optimize (merge duplicates, retire stale notes)"),
    ("/delete [target]", "delete a session"),
    ("/todo", "show the model's current todo_write list"),
    ("/wip [note]", "snapshot work state (branch, dirty files, todos, note) into the session; /resume shows it"),
    ("/plan [goal|done|off|discard]", "start/view/exit plan mode — while active, mutating tool calls are gated"),
    ("/verify [cmd|off|run]", "show/set/clear or force-run the post-edit verify command"),
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
    #   /source together                    → default (GLM-5.3)
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


def _cmd_tooloutput(user: str, state: _ReplState) -> None:
    sp = user.split()
    sub = sp[1].lower() if len(sp) > 1 else None
    if sub in ("on", "show", "expand"):
        st.TOOL_OUTPUT_VISIBLE = True
        _set_env_var("NX_TOOL_OUTPUT_VISIBLE", "1")
        os.environ["NX_TOOL_OUTPUT_VISIBLE"] = "1"
        print(f"{BOLD}{C.WARNING}  → tool output: visible{RESET} {DIM}(saved to ~/.env){RESET}")
        if sub == "show" and st._LAST_TOOL_OUTPUT:
            print(f"{C.ACCENT}  ┌─ {st._LAST_TOOL_NAME}{RESET}")
            print(f"{C.ACCENT}  └─{RESET} {C.TOOL_OUTPUT}{st._LAST_TOOL_OUTPUT}{RESET}")
    elif sub in ("off", "hide", "collapse"):
        st.TOOL_OUTPUT_VISIBLE = False
        _set_env_var("NX_TOOL_OUTPUT_VISIBLE", "0")
        os.environ["NX_TOOL_OUTPUT_VISIBLE"] = "0"
        print(f"{BOLD}{C.WARNING}  → tool output: collapsed{RESET} {DIM}(saved to ~/.env){RESET}")
    elif sub in ("last",):
        if st._LAST_TOOL_OUTPUT:
            print(f"{C.ACCENT}  ┌─ {st._LAST_TOOL_NAME}{RESET}")
            print(f"{C.ACCENT}  │ {RESET}{C.TOOL_OUTPUT}{st._LAST_TOOL_OUTPUT}{RESET}")
            print(f"{C.ACCENT}  └─{RESET}")
        else:
            print(f"{DIM}  no tool output captured yet{RESET}")
    elif sub is None:
        state_str = "visible" if st.TOOL_OUTPUT_VISIBLE else "collapsed"
        print(f"{DIM}  tool output: {state_str}{RESET}")
        print(f"{DIM}  /tooloutput on|off · /tooloutput last (show most recent, "
              f"even if collapsed){RESET}")
    else:
        print(f"{C.ERROR}  ✗ unknown /tooloutput arg {sub!r} — "
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
    #   /provider io-net/fp8           pin to that provider (active src)
    #   /provider Together,DeepInfra    ordered: try Together, then DeepInfra
    #   /provider off                   clear routing (let OpenRouter pick)
    #   /provider openrouter io-net    set routing on <source>, not active
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


def _cmd_ctx(user: str, state: _ReplState) -> None:
    """/ctx — per-message token report; /ctx pin|unpin <spec> — pin turns so
    compress() keeps them verbatim; /ctx clear — drop all pins."""
    from ._ctxreport import (_messages_token_report, _ctx_pin, _ctx_unpin,
                             _overhead_lines)
    from ._metrics import _abbr
    parts = user.split()
    sub = parts[1].lower() if len(parts) >= 2 else ""
    messages = state.messages
    if sub == "clear":
        removed, remaining = _ctx_unpin(None, messages)
        print(f"{DIM}  unpinned {len(removed)} turn(s){RESET}")
        return
    if sub in ("pin", "unpin"):
        spec = " ".join(parts[2:]) if len(parts) >= 3 else None
        if not spec:
            print(f"{C.WARNING}  usage: /ctx {sub} <n | n-m | n,m,n-m>  (1-based message numbers from /ctx){RESET}")
            return
        try:
            if sub == "pin":
                added, pinned = _ctx_pin(spec, messages)
                print(f"{DIM}  pinned {len(added)} turn(s) — compress will keep them verbatim ({len(pinned)} pinned){RESET}")
            else:
                removed, remaining = _ctx_unpin(spec, messages)
                print(f"{DIM}  unpinned {len(removed)} turn(s) ({len(remaining)} pinned remain){RESET}")
        except (ValueError, TypeError) as e:
            print(f"{C.WARNING}  {e} — want e.g. /ctx pin 12  or  /ctx pin 4,8,12-15{RESET}")
        return
    if sub == "all":
        # Every message row, none rolled up (0.42): the compact default hides
        # small turns behind a count; this is the full dump the hint names.
        report = _messages_token_report(messages)
        rows = report["rows"]
        cache = report.get("cache") or []
        print(f"{DIM}  context by message — ALL {len(rows)} messages, "
              f"~{report['total_est']} total est{RESET}")
        for i, role, t, preview, pinned in rows:
            mark = "📌" if pinned else " "
            hit = cache[i] if i < len(cache) else None
            hit_s = "—" if hit is None else f"{hit * 100:.0f}%"
            print(f"{DIM}  {mark}{i + 1:>4}  {role:<9} {t:>6}  {hit_s:>4}  {preview}{RESET}")
        for line in _overhead_lines(report["total_est"], report.get("overhead_est", 0)):
            print(f"{DIM}{line}{RESET}")
        return
    # default: the report
    report = _messages_token_report(messages)
    rows = report["rows"]
    cache = report.get("cache") or []
    print(f"{DIM}  context by message (est. tokens, {len(rows)} messages, "
          f"~{report['total_est']}{RESET}{DIM} total est · cache = that turn's"
          f" model call's prompt-cache hit){RESET}")
    # Compact: show the top consumers fully, roll small turns into one line
    top = sorted(rows, key=lambda r: -r[2])[:12]
    top_idx = {r[0] for r in top if r[2] >= max(32, report["total_est"] // 40)}
    for i, role, t, preview, pinned in rows:
        if i in top_idx or pinned:
            mark = "📌" if pinned else " "
            hit = cache[i] if i < len(cache) else None
            hit_s = "—" if hit is None else f"{hit * 100:.0f}%"
            print(f"{DIM}  {mark}{i + 1:>4}  {role:<9} {t:>6}  {hit_s:>4}  {preview}{RESET}")
    rolled = [r for r in rows if r[0] not in top_idx or r[2] < max(32, report["total_est"] // 40)]
    if rolled and len(rolled) < len(rows):
        print(f"{DIM}        … {len(rolled)} smaller turn(s), ~{sum(r[2] for r in rolled)} tokens "
              f"(all messages: /ctx all){RESET}")
    # Per-turn overhead (tools= schemas) is in total_est but has no message
    # row — name it here so "by role" summing to less than the header isn't
    # mistaken for lost tokens.
    for line in _overhead_lines(report["total_est"], report.get("overhead_est", 0)):
        print(f"{DIM}{line}{RESET}")
    by_role = report["by_role"]
    role_s = "  ".join(f"{r}:{_abbr(t)}" for r, t in sorted(by_role.items(), key=lambda kv: -kv[1]))
    print(f"{DIM}  by role: {role_s}{RESET}")
    mx = st.ACTIVE._context_window if st.ACTIVE else None
    if isinstance(mx, int) and mx > 0:
        pct = 100 * report["total_est"] // mx
        print(f"{DIM}  ≈{pct}% of {_abbr(mx)} window (estimate; the per-turn footer "
              f"uses the server's exact count){RESET}")
    print(f"{DIM}  pin turns to protect them from /compress: /ctx pin <n | n-m | n,m>{RESET}")


def _cmd_cost(user: str, state: _ReplState) -> None:
    """/cost — session usage totals + per-source scoreboard ($ when
    NX_PRICE_<SOURCE>_* is set)."""
    from ._cost import print_cost_report
    print_cost_report(st._METRICS)


def _cmd_status(user: str, state: _ReplState) -> None:
    """/status — P3-9 snapshot of the session posture a doctor would check:
    source, model, caps, budgets, fence, approval. No args, no state change
    (rendered by nx/_config_cmd.render_status)."""
    from ._config_cmd import render_status
    print(render_status())


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
    agents = list_async_agents()
    if not jobs and not agents:
        print(f"{DIM}  no background jobs or sub-agents{RESET}")
        return
    if agents:
        print(f"{DIM}  background sub-agents (newest first):{RESET}")
        for a in agents:
            if a["alive"]:
                status = f"{C.SUCCESS}running{RESET}"
            elif a["status"] == "queued":
                status = f"{DIM}queued{RESET}"
            elif a["status"] == "done":
                status = f"{C.SUCCESS}done{RESET}"
            else:
                status = f"{C.ERROR}{a['status']}{RESET}"
            instr = a["instruction"]
            instr = instr if len(instr) <= 60 else instr[:59] + "…"
            print(f"  {DIM}{a['id']:>12}{RESET}  {status:<18}  {instr}")
            if a["tail"]:
                print(f"           {DIM}└ {a['tail']}{RESET}")
            if a.get("report"):
                print(f"           {DIM}└ report: {a['report']}{RESET}")
        print(f"{DIM}  answers arrive as [Runtime note] on a later turn{RESET}")
    if not jobs:
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
        print(f"{DIM}  approval={_approval_display()}  (all|low|medium|high|yolo; set-default <level> persists){RESET}")
        return
    arg = parts[1]
    if arg == "allow":
        # /approval allow [list|forget N|forget-all|<tool> <pattern>]
        from . import _allowed as _allowed_mod
        sub = parts[2].lower() if len(parts) >= 3 else "list"
        if sub == "list":
            rules = _allowed_mod.list_rules()
            if not rules:
                print(f"{DIM}  no allow rules saved (press A at an approval prompt to save one){RESET}")
                return
            print(f"{DIM}  allow rules ({len(rules)}):{RESET}")
            for i, r in enumerate(rules, 1):
                print(f"{DIM}   {i:>2}. [{r['tool']}] {r['pattern']}  ({r['scope']}){RESET}")
            return
        if sub in ("forget", "rm", "remove"):
            if len(parts) < 4 or not parts[3].isdigit():
                print(f"{C.WARNING}  usage: /approval allow forget <n>  (see /approval allow list){RESET}")
                return
            removed = _allowed_mod.forget(int(parts[3]))
            if removed:
                print(f"{DIM}  forgot [{removed['tool']}] {removed['pattern']}{RESET}")
            else:
                print(f"{C.WARNING}  no rule #{parts[3]} (see /approval allow list){RESET}")
            return
        if sub == "forget-all":
            n = _allowed_mod.forget_all()
            print(f"{DIM}  forgot {n} rule(s){RESET}")
            return
        # /approval allow <tool> <pattern> — save a rule by hand
        if len(parts) >= 4:
            rule = _allowed_mod.add_rule(parts[2], " ".join(parts[3:]))
            print(f"{DIM}  saved [{rule['tool']}] {rule['pattern']} ({rule['scope']}){RESET}")
            return
        print(f"{C.WARNING}  usage: /approval allow [list | forget <n> | forget-all | <tool> <pattern>]{RESET}")
        return
    if arg in ("set-default", "default", "persist"):
        # /approval set-default <level> — persist the current (or given) mode to
        # ~/.env as NX_APPROVAL, the same way /theme persists NX_THEME. Bare
        # `/approval <level>` stays session-scoped so a one-off yolo doesn't
        # silently become your standing default.
        target = parts[2] if len(parts) >= 3 else _approval_display()
        kind, resolved = _normalize_approval(target)
        if not kind:
            print(f"{C.WARNING}  unknown level {target!r} — want all|low|medium|high|yolo{RESET}")
            return
        _apply_approval(kind, resolved, update_default=True)
        persisted = "yolo" if kind == "yolo" else (resolved or "all")
        _set_env_var("NX_APPROVAL", persisted)
        os.environ["NX_APPROVAL"] = persisted
        shown = "all" if kind == "prompt_all" else (resolved or "all")
        print(f"{BOLD}{C.WARNING}  → approval: {shown}{RESET} "
              f"{DIM}(saved to ~/.env — /approval set-default all to reset){RESET}")
        print(_m._banner())
        return
    kind, resolved = _normalize_approval(arg)
    if kind:
        _apply_approval(kind, resolved, update_default=(kind != "yolo"))
        if kind == "prompt_all":
            print(f"{DIM}  approval=all (prompt every classified action){RESET}")
        elif kind == "level":
            print(f"{DIM}  approval={resolved} (auto-allow ≤ {resolved}){RESET}")
        else:
            print(f"{DIM}  approval=off (yolo — never prompt){RESET}")
        print(f"{DIM}  (session only — /approval set-default <level> persists to ~/.env){RESET}")
        print(_m._banner())
    else:
        print(f"{C.WARNING}  unknown level {arg!r} — want all|low|medium|high|yolo "
              f"(or set-default <level> to persist){RESET}")


def _cmd_reset(user: str, state: _ReplState) -> None:
    import nx as _m
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()
    state.messages = _initial_messages()
    st._SESSION_HAS_BACKGROUNDED = False
    st._MULTI_EDIT_NUDGED = False
    st._CONSECUTIVE_EDITS = 0
    # A fresh conversation has nothing to un-write from the old one: drop both
    # the in-memory undo stack and its on-disk copy (0.47.0, ROADMAP 2.2).
    _m._undo.clear_persisted()
    # Fresh tool list too: a /reset implies a new conversation, and an
    # endpoint-level tool-args quirk may have been transient.
    st._MALFORMED_TOOL_FAILURES = {}
    st._MALFORMED_TOOL_BROKEN = set()
    # Re-arm the endpoint identity too (0.40) so the reset isn't undone by a
    # pending re-key, and a switch to another source is treated as fresh
    # evidence rather than a carry-over.
    st._BREAKER_ENDPOINT_KEY = st.BREAKER_UNOBSERVED
    print(f"{DIM}  context cleared{RESET}")
    # A reset forks a new session id so the fresh chat isn't written
    # over the old one (mirrors Hermes's "new session on /new").
    state.session_id = _new_session_id()
    st.SESSION_ID = state.session_id  # mirror: undo stack swaps with the fresh id
    state.session_dirty = False
    # Fresh recorder for the fresh session.
    st._METRICS = _fresh_metrics(state.session_id)
    print(f"{DIM}  new session {state.session_id}{RESET}")
    # The reset clears the screen, which wipes the startup banner — reprint it
    # so the user can see which model/source they're now chatting with.
    print(_m._banner())


def _cmd_toolcalls(user: str, state: _ReplState) -> None:
    """0.29.9 /toolcalls: inspect or reset the malformed-args circuit breaker.
    0.30.0: /toolcalls text|native|auto — force the tool-calling mode for this
    session, giving the escalation ladder a rung PAST the breaker: an endpoint
    whose NATIVE args keep truncating gets the grammar-constrained text
    protocol back without switching sources ('text'); 'native' re-arms the
    native flag; 'auto' returns to the reactive default.

    Bare: show each tool's failure streak and what's currently dropped.
    `reset [tool]`: clear the streak(s) and re-arm — a mid-session escape
    hatch for a flaky local endpoint whose truncations were transient, or for
    handing the tool back to the model after fixing the cause. (The new
    decay-on-success usually re-arms on its own; this forces it.)"""
    from ._compress import set_toolcalls_mode, toolcalls_mode
    from ._dispatch import _breaker_views
    arg = user.strip()
    if arg in ("text", "native", "auto"):
        if set_toolcalls_mode(arg):
            print(f"{DIM}  tool-calling mode: {arg} "
                  f"(grammar {'attached' if arg == 'text' else 'per source capability'}){RESET}")
        return
    if arg and arg.split()[0] not in ("reset",):
        print(f"{C.WARNING}  want /toolcalls, /toolcalls reset [tool], "
              f"or /toolcalls text|native|auto{RESET}")
        return
    # Read through _breaker_views() (0.40): the listing shows the ACTIVE
    # endpoint's streaks, not a previous source's leftovers.
    failures, broken = _breaker_views()
    if arg.startswith("reset"):
        target = arg[len("reset"):].strip()
        if target:
            st._MALFORMED_TOOL_FAILURES.pop(target, None)
            st._MALFORMED_TOOL_BROKEN.discard(target)
            print(f"{DIM}  {target}: streak cleared, tool re-offered{RESET}")
        else:
            st._MALFORMED_TOOL_FAILURES = {}
            st._MALFORMED_TOOL_BROKEN = set()
            st._BREAKER_ENDPOINT_KEY = st.BREAKER_UNOBSERVED
            print(f"{DIM}  malformed-args streaks cleared, all tools re-offered{RESET}")
        return
    print(f"{DIM}  tool-calling mode: {toolcalls_mode()}{RESET}")
    if failures or broken:
        print(f"{DIM}  malformed-args streaks (breaker opens at "
              f"{st.MALFORMED_TOOL_BREAKER_LIMIT}):{RESET}")
        for tool in sorted(set(failures) | set(broken)):
            mark = f"{C.WARNING} [broken — not offered]{RESET}" if tool in broken else ""
            print(f"    {tool}: {failures.get(tool, 0)}{mark}")
    else:
        print(f"{DIM}  no malformed-args failures recorded "
              f"(breaker opens at {st.MALFORMED_TOOL_BREAKER_LIMIT}){RESET}")


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


def _cmd_budget(user: str, state: _ReplState) -> None:
    # /budget — cap the model's reasoning (thinking) phase in tokens. Rides in
    # the request body as a nested "reasoning" object (see nx/_sources.py),
    # the same channel as /provider routing — so backends that ignore unknown
    # body keys are unaffected, and a source with its own "reasoning" object
    # in EXTRA_BODY keeps winning over this global cap.
    arg = user[len("/budget"):].strip().lower()
    if not arg:
        cur = st.REASONING_BUDGET_TOKENS or "off"
        print(f"{DIM}  reasoning budget: {cur}{RESET}")
        print(f"{DIM}  /budget <tokens> caps the thinking phase"
              f" (e.g. /budget 4096) · /budget off clears it{RESET}")
        return
    if arg in ("off", "none", "clear"):
        st.REASONING_BUDGET_TOKENS = None
        os.environ.pop("NX_REASONING_BUDGET", None)
        print(f"{DIM}  reasoning budget: off{RESET}")
        return
    try:
        val = int(arg)
    except ValueError:
        print(f"{C.WARNING}  usage: /budget [<tokens|off>] — "
              f"current: {st.REASONING_BUDGET_TOKENS or 'off'}{RESET}")
        return
    if val <= 0:
        print(f"{C.WARNING}  {val} is out of range — want a positive token"
              f" count (or off){RESET}")
        return
    st.REASONING_BUDGET_TOKENS = val
    os.environ["NX_REASONING_BUDGET"] = str(val)
    print(f"{DIM}  reasoning budget: {val} tokens{RESET}")


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


def _cmd_verify(user: str, state: _ReplState) -> None:
    """/verify — show / set / clear / force-run the post-edit verify command.

        /verify                  show the current NX_VERIFY_CMD (or "off")
        /verify <cmd...>         set it (e.g. /verify uv run pytest -q), saved to ~/.env
        /verify off              clear it
        /verify run              run it now, detached; the outcome arrives
                                 as a [Runtime note] when it finishes

    When set, nx also runs it automatically ~NX_VERIFY_SETTLE_SECS (default 20)
    after the last .py edit goes quiet, and an in-process ast.parse of every
    edited .py file reports 'py_compile OK' / 'SYNTAX ERROR at line N' on the
    edit's own tool result.
    """
    from ._verify import _launch_verify_if_settled, verify_enabled
    arg = user[len("/verify"):].strip()
    if not arg:
        cmd = os.environ.get("NX_VERIFY_CMD", "").strip()
        if cmd:
            print(f"{DIM}  verify: {cmd}{RESET}")
            print(f"{DIM}  (auto-runs ~NX_VERIFY_SETTLE_SECS after the last .py edit; "
                  f"/verify run to force, /verify off to disable){RESET}")
        else:
            print(f"{DIM}  verify: off — set one with /verify <command>, e.g. /verify uv run pytest -q{RESET}")
        return
    if arg in ("off", "disable", "0"):
        os.environ.pop("NX_VERIFY_CMD", None)
        _set_env_var("NX_VERIFY_CMD", "")
        st._VERIFY_EDIT_LOG.clear()
        print(f"{DIM}  verify: off{RESET}")
        return
    if arg in ("run", "force", "now"):
        if not verify_enabled():
            print(f"{C.WARNING}  no verify command set — /verify <command> first{RESET}")
            return
        st._VERIFY_EDIT_LOG.append(0.0)  # ancient timestamp → settled immediately
        _launch_verify_if_settled(force=True)
        print(f"{DIM}  ↳ verify command launched in background — its result will "
              f"surface as a runtime note when it finishes{RESET}")
        return
    # anything else: the whole remainder is the command
    _set_env_var("NX_VERIFY_CMD", arg)
    os.environ["NX_VERIFY_CMD"] = arg
    st._VERIFY_INFLIGHT = False  # a changed command must not stay "in flight"
    st._VERIFY_SEEN.clear()  # old command's finished jobs are no longer ours
    print(f"{BOLD}{C.WARNING}  → verify: {arg}{RESET} {DIM}(saved to ~/.env){RESET}")


def _cmd_todo(user: str, state: _ReplState) -> None:
    """Show the model's current todo_write list, colored by status.

    The list lives in ``st._TODOS`` and is rewritten wholesale by each
    ``todo_write`` tool call (see _todo.py); it is persisted into the session
    JSON so ``/resume`` restores it. ``/todo`` is read-only — it renders the
    current list with per-status colors via render_todos_display, the same
    coloring the todo_write tool output gets on screen (done=green,
    in_progress=yellow, pending=orange).
    """
    if not st._TODOS:
        print(f"{DIM}  (todo list empty){RESET}")
        return
    print(f"{DIM}  ── todos ──{RESET}")
    for line in render_todos_display().splitlines():
        print(f"  {line}")
    print(f"{DIM}  ──────────{RESET}")

