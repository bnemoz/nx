"""Session list/resume/delete/save UI: the `/sessions`, `/resume`, `/save`,
`/delete` REPL commands, the `nx sessions` / `nx --tokens` CLI entry points,
and their shared list-printing/paging helpers.

This is the REPL/CLI *presentation* layer for sessions — file I/O itself
(`_write_session`/`_load_session`/`_list_sessions`/`_delete_session`) lives in
`_sessions_io.py` / `nx/__init__.py`.

Functions here that are monkeypatched by tests via `m.<name> = fake` are
always called through a lazy `import nx as _m; _m.<name>(...)` — the patch
only ever lands on the top-level `nx` module's attribute, never on this
submodule's own globals. See `_turn.py` / `_repl.py` for the same pattern.
"""
from __future__ import annotations

import json
import os
import shlex
import time
from typing import Any

from ._ansi import DIM, BOLD, RESET, C, strip_ansi_for_display
from ._sessions_io import _flatten_message_text, _short_id
from ._sources import restore_source, _set_provider_order
from . import _state as st
from . import (
    SYSTEM, _active_tools, _sessions_dir, SESSION_HOME,
    SESSION_LIST_DEFAULT_LIMIT, SESSION_LIST_MAX_LIMIT,
    _load_session, _list_sessions, _delete_session, _resolve_session,
)
from ._repl_state import _ReplState, _save_current, _fresh_metrics, _reload_metrics_totals


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
        content = strip_ansi_for_display(_flatten_message_text(m, per_msg=max_chars))
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
    active = st.ACTIVE
    if active is not None:
        probe = active.tokenize_count("hey")
    else:
        probe = None
    if probe is not None and active is not None:
        def count(text):
            return active.tokenize_count(text)
        method = f"server /tokenize ({active.name})"
    else:
        try:
            import tiktoken  # ty: ignore[unresolved-import]
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
        # Not a title/id match — try full-text body search (FTS5) before
        # giving up. "which session touched _turn.py?" shouldn't need scrolling.
        _print_session_body_search(arg)
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


def _print_session_body_search(query: str, limit: int = 10) -> None:
    """Print FTS matches over session bodies for a /sessions <query> that
    didn't match any title/id. Degrades to the old 'no session matching'
    message when the index is unavailable or the query isn't valid FTS."""
    from ._sessions_search import search_sessions
    try:
        hits = search_sessions(query, limit=limit)
    except Exception:
        hits = []
    if not hits:
        print(f"{C.WARNING}  no session matching {query!r}{RESET}")
        return
    print(f"{DIM}  sessions containing {query!r}:{RESET}")
    for h in hits:
        title = h.get("title") or "(untitled)"
        print(f"{DIM}   · {h['id'][:12]}  {title}{RESET}")
        snippet = h.get("snippet") or ""
        if snippet:
            print(f"{DIM}     {snippet[:160]}{RESET}")
    print(f"{DIM}  /resume <n|short-id|title> to open one (body matches: /sessions <query>){RESET}")


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


# --- REPL command-table adapters --------------------------------------------
# Thin wrappers matching the (user, state) -> str | None handler signature
# used by `_REPL_COMMANDS` in `_repl.py`.

def _cmd_sessions_handler(user: str, state: _ReplState) -> None:
    _cmd_sessions(user, state.session_id)


def _cmd_export(user: str, state: _ReplState) -> None:
    """/export [md] [path] — write the current session to a markdown file.
    Default path: nx-export-<session-id>.md in the project root."""
    import nx as _m
    from ._sessions_search import export_markdown
    parts = user.split()
    args = [a for a in parts[1:] if a != "md"]  # 'md' is the (only) format today
    out_path = args[0] if args else f"nx-export-{(state.session_id or 'session')[:12]}.md"
    data = {
        "id": state.session_id,
        "title": getattr(state, "title", None) or None,
        "description": None,
        "source": st.ACTIVE.name if st.ACTIVE else None,
        "model": st.MODEL,
        "messages": state.messages,
        "usage": (st._METRICS.as_meta() if st._METRICS is not None else None),
    }
    md = export_markdown(data)
    try:
        parent = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    except OSError as e:
        print(f"{C.ERROR}  ✗ export failed: {e}{RESET}")
        return
    print(f"{DIM}  wrote {os.path.abspath(out_path)} ({len(md)} chars, "
          f"{len(state.messages)} messages){RESET}")


def _cmd_resume_handler(user: str, state: _ReplState) -> None:
    new_sid = _cmd_resume(user, state.session_id)
    if not new_sid:
        return
    # Persist the outgoing session before switching away.
    if state.session_dirty:
        _save_current(state)
    state.session_id = new_sid
    state.messages[:] = _session_messages(new_sid)
    from . import _messages_show_prior_backgrounding, _messages_show_prior_multi_edit_nudge, _messages_show_prior_delegation_nudge
    st._SESSION_HAS_BACKGROUNDED = _messages_show_prior_backgrounding(state.messages)
    st._MULTI_EDIT_NUDGED = _messages_show_prior_multi_edit_nudge(state.messages)
    from ._subagent import _add_delegation_note_if_absent
    _add_delegation_note_if_absent(state.messages)
    st._DELEGATION_NUDGED = _messages_show_prior_delegation_nudge(state.messages)
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


def _cmd_delete_handler(user: str, state: _ReplState) -> None:
    target = user.split(None, 1)[1].strip() if " " in user else None
    _cmd_delete(target, state.session_id)
