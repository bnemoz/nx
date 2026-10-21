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

from ._ansi import DIM, RESET, C, strip_ansi_for_display
from ._rich import grid_line
from . import _fullscreen
from ._sessions_io import _flatten_message_text, _short_id
from . import _state as st
from . import (
    SYSTEM, _active_tools, _sessions_dir, SESSION_HOME,
    SESSION_LIST_DEFAULT_LIMIT, SESSION_LIST_MAX_LIMIT,
    _load_session, _list_sessions, _delete_session, _resolve_session,
)
from ._repl_state import _ReplState, _save_current, _flush_save


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
        print(grid_line(color, 9, role, content, pad="  "))
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


_SESSIONS_USAGE = """\
usage: nx sessions [query] [--page N] [--limit N]
       nx sessions --prune [--reclaim]
       nx sessions --reindex

List saved sessions and exit. With a query: substring-filters (case-
insensitive) across title, description, preview and id.

  --page N / -p N    1-based page number (default 1)
  --limit N / -n N   rows per page (default 10)

  nx sessions --prune  run the store's GC by hand and exit: drop rows whose
                       session JSON vanished, collect unreferenced image
                       payloads immediately, enforce the NX_IMAGES_QUOTA_MB
                       cap, and sweep expired/over-quota objects from the
                       shared tool-result store (obsstore). One line per action.
                       Opt-in retention (off by default) also runs: see
                       NX_SESSION_RETAIN_SMALL_DAYS / _SMALL_MAX_MSGS and
                       NX_SESSION_RETAIN_FLOOR_MSGS / _FLOOR_DAYS.

  nx sessions --prune --reclaim
                       additionally hand back the space the 0.115.0
                       events-authoritative migration left behind: drop the
                       dead pre-0.115 sessions_fts/fts_body index, blank the
                       stale full-transcript session blobs whose events
                       already hold the text, then VACUUM. Takes an exclusive
                       lock on the store for the duration.

  nx sessions --reindex
                       rebuild the per-message search index NOW instead of
                       amortized: re-derive every session's indexed bodies
                       from its stored messages (the same builder an indexed
                       save uses) until the store carries the current FTS
                       body-schema version. One-line summary. Idempotent — a
                       store already at the current version does nothing.

Resume with: nx --resume <n|short-id|title>
"""

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
    if any(a in ("-h", "--help") for a in args):
        print(_SESSIONS_USAGE, end="")
        return True
    # `nx sessions --reindex` reaches its handler from HERE rather than from
    # the _cli.py verb chain (same ordering reason as --prune: the listing
    # path would otherwise swallow the flag as a query and print "no
    # sessions matching '--reindex'"). Verb position only, exactly like the
    # `nx sessions prune` spelling — a two-word query merely CONTAINING the
    # word still searches.
    if len(args) >= 2 and args[1] in ("--reindex", "reindex"):
        return _cli_reindex_sessions(args)
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


def _cli_prune_sessions(args: list[str]) -> bool:
    """`nx sessions --prune` — run the store's own GC by hand and exit.

    What it does, in order: (1) reconcile store rows (and any pre-2.3b
    legacy shards) whose session JSON vanished without a /delete (crash,
    manual rm) — they are removed so listings, search and the GC live-set
    stay truthful; (2) drop sidecar image payloads nothing references
    anymore (immediate, no age grace — this is an explicit user request,
    not a mid-save sweep); (3) enforce the NX_IMAGES_QUOTA_MB cap. Prints
    one line per action. Returns True if it handled the invocation (caller
    exits)."""
    # `nx sessions --prune` (verb-first) or a bare `nx --prune`/`prune`.
    # `--reclaim` is a modifier, not a verb: peel it off before the match so
    # every spelling (`nx sessions --prune --reclaim`, `nx prune --reclaim`)
    # reaches the same path.
    reclaim = "--reclaim" in args
    args = [a for a in args if a != "--reclaim"]
    if args[:2] not in (["--prune"], ["prune"], ["sessions", "--prune"],
                        ["sessions", "prune"]):
        return False
    if args[:2] == ["sessions", "prune"] and len(args) > 2:
        # `nx sessions prune` with a stray query is a typo — say so instead
        # of silently pruning against the wrong expectation.
        print(f"{C.WARNING}  usage: nx sessions --prune  (no other arguments){RESET}")
        return True
    from ._sessions_fs import _prune_sessions_store
    stats = _prune_sessions_store()
    if stats["rows_pruned"]:
        print(f"  pruned {stats['rows_pruned']} stale index row(s) with no session file")
    if stats["shards_dropped"]:
        print(f"  dropped {stats['shards_dropped']} legacy index shard(s) with no session file")
    print(f"  collected {stats['images_removed']} unreferenced image payload(s)"
          f" from {_sessions_dir()}/_images")
    if stats["quota_removed"]:
        print(f"  quota eviction removed {stats['quota_removed']} more payload(s)")
    if stats.get("expired_small", 0) + stats.get("expired_floor", 0) > 0:
        print(f"  retention removed {stats['expired_small'] + stats['expired_floor']} "
              f"session(s) ({stats['expired_small']} small-expired, "
              f"{stats['expired_floor']} floor-expired)")
    print(f"  {DIM}{stats['images_kept']} payload(s) still referenced by sessions{RESET}")
    if stats.get("obs_removed"):
        print(f"  collected {stats['obs_removed']} expired/unreferenced tool-result "
              f"object(s) from the shared obsstore")
    if stats.get("obs_kept"):
        print(f"  {DIM}{stats['obs_kept']} obsstore object(s) kept "
              f"({stats['obs_bytes'] // 1024} KiB){RESET}")
    if stats.get("undo_removed"):
        print(f"  collected {stats['undo_removed']} orphaned undo file(s) "
              f"(dead sessions; their checkpoints held file snapshots)")
    if reclaim:
        # Store-residue reclamation (0.115.0 migration leftovers). Deliberately
        # NOT part of plain --prune: reclaim_store VACUUMs, which takes an
        # exclusive lock on the whole store — a user-asked-for cost, not one
        # `--prune` should pay (or a lazy migration, ever).
        from ._sessiondb import reclaim_store
        rc = reclaim_store()
        if rc["fts_dropped"]:
            print("  dropped the dead pre-0.115 sessions_fts/fts_body index")
        print(f"  reclaimed {rc['blobs_emptied']} stale transcript blob(s)"
              f" and vacuumed the session store")
    return True


def _cli_reindex_sessions(args: list[str]) -> bool:
    """`nx sessions --reindex` — run the FTS body-schema backfill to
    completion by hand and exit (the manual escape hatch over the save
    path's bounded, amortized pass). One-line summary; idempotent.
    Returns True if it handled the invocation (caller exits)."""
    if args[:2] not in (["--reindex"], ["reindex"], ["sessions", "--reindex"],
                        ["sessions", "reindex"]):
        return False
    from ._sessiondb import fts_body_reindex_all
    rc = fts_body_reindex_all()
    if rc["sessions_reindexed"]:
        print(f"  reindexed {rc['sessions_reindexed']} session(s)"
              f" into body-schema v{rc['version']}")
    else:
        print("  nothing to reindex: the search index already carries the"
              f" current body schema (v{rc['version']})")
    return True


def _cli_migrate_sessions(args: list[str]) -> bool:
    """`nx sessions --migrate` — fold any legacy per-session JSON files
    into the session store by hand and exit.

    0.101.0 migration normally runs lazily (the first listing/load after an
    upgrade imports whatever mirror files exist); this runs it eagerly so
    an upgrade can be verified ("imported 47 sessions") before the next
    REPL start. Idempotent — already-imported files are skipped, and the
    files are left in place as a backup. Returns True if it handled the
    invocation (caller exits)."""
    if args[:2] not in (["--migrate"], ["migrate"], ["sessions", "--migrate"],
                        ["sessions", "migrate"]):
        return False
    from ._sessiondb import migrate_files, store_stats
    n = migrate_files(force=True)
    stats = store_stats()
    if n:
        print(f"  imported {n} legacy session file(s) into the store")
    else:
        print("  nothing to import: every legacy JSON file in the sessions"
              " dir is already in the store (idempotent; files are kept as"
              " backups and rescanned each run)")
    print(f"  {DIM}{stats['sessions_indexed']} session(s) now in "
          f"{_sessiondb_store_path()}{RESET}")
    return True


def _sessiondb_store_path() -> str:
    from ._sessiondb import _store_path
    return _store_path()


def _cli_tokens(args: list[str]) -> bool:
    """`nx --tokens` / `nx --bench` — print the exact prompt-token
    floor (SYSTEM + the active TOOLS schema + a 1-token "hey") for the active
    backend/model, then exit. Turns the README's "under ~4K tokens on a bare hey"
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
        def count(text: str) -> int:
            n = active.tokenize_count(text)
            assert n is not None  # probe above proved this source tokenizes; narrows for the checker
            return n
        method = f"server /tokenize ({active.name})"
    else:
        # Both branches count through the shared estimator (nx/_tokens.py).
        from . import _tokens as tokens
        count = tokens.estimate_tokens
        method = "chars/3.5 estimate (no /tokenize endpoint found)"
    system_tokens = count(SYSTEM)
    tools_tokens = count(json.dumps(tools))
    user_tokens = count("hey")
    total = system_tokens + tools_tokens + user_tokens
    backend = f"{st.ACTIVE.name} · {st.MODEL}" if st.ACTIVE and st.MODEL else (st.MODEL or "(no source configured)")
    bg_note = "" if st._SESSION_HAS_BACKGROUNDED else ", wait_background excluded — nothing has backgrounded yet"
    print(f"nx prompt-token floor — {method}")
    print(f"  backend   {backend}")
    print(grid_line("", 5, "SYSTEM", f"{system_tokens:>5} tokens", pad="  ").rstrip())
    print(grid_line("", 5, "TOOLS", f"{tools_tokens:>5} tokens  ({len(tools)} functions{bg_note})", pad="  ").rstrip())
    print(grid_line("", 5, "'hey'", f"{user_tokens:>5} tokens", pad="  ").rstrip())
    print(f"  {'-' * 28}")
    print(grid_line("", 5, "total", f"{total:>5} tokens  (+ ~50 of server-side chat-template framing, not counted here)", pad="  ").rstrip())
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
    # Selection only: loading, endpoint/cwd restore and the transcript banner
    # all happen in resume_session() (nx/_resume.py), which the handler calls —
    # the CLI and /resume must not own separate copies of that sequence.
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
    # Phase D: the tty path opens the full-screen browser; its fallback is
    # this exact inline listing, so off-tty output is unchanged.
    from . import _fullscreen
    if _fullscreen.wants_fullscreen() and not user.split()[1:]:
        cmd_sessions_view(user, state)
        return
    _cmd_sessions(user, state.session_id)


def _cmd_export(user: str, state: _ReplState) -> None:
    """/export [md] [path] — write the current session to a markdown file.
    Default path: nx-export-<session-id>.md in the project root."""
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
    # 0.103.4: drain before the id switches (and before resume_session reads
    # the target transcript back), so the outgoing save can never land under
    # the incoming session's id.
    _flush_save()
    # Release the outgoing session's file claims BEFORE resume_session swaps
    # the id (0.128.3; see _fileclaim.release_session_claims). The `--resume`
    # startup path skips this deliberately: at process start there are no
    # claims to release.
    from . import _fileclaim
    _fileclaim.release_session_claims(state.session_id)
    # Same function `nx --resume` runs: images expanded to live form, endpoint +
    # provider routing + cwd restored, nudge flags re-derived, metrics reloaded.
    from ._resume import resume_session
    resume_session(new_sid, state, header="resumed")


def _cmd_fork(user: str, state: _ReplState) -> None:
    """Handle /fork — branch the current conversation into a NEW session id,
    keeping every message so far (a prefix-identical copy, so the prompt
    cache stays warm) and leaving the original session untouched on disk.
    The classic use: explore an alternative approach without losing the
    main thread, and without a /compress-rebuild round trip."""
    if not state.messages or not any(m.get("role") == "user"
                                     for m in state.messages):
        print(f"{DIM}  nothing to fork yet — send a message first{RESET}")
        return
    from ._sessions_io import _new_session_id
    label = user.split(None, 1)[1].strip() if " " in user else ""
    # Persist the outgoing session first so the fork point is on disk under
    # the ORIGINAL id too (fork = copy, not move).
    if state.session_dirty:
        _save_current(state)
    # 0.103.4: the fork must not inherit a snapshot that still names the OLD
    # id — drain anything queued before the id below is reassigned.
    _flush_save()
    old_id = state.session_id
    # Release the outgoing session's file claims BEFORE the id is reassigned
    # (0.128.3; see _fileclaim.release_session_claims).
    from . import _fileclaim
    _fileclaim.release_session_claims(old_id)
    new_id = _new_session_id()
    # Carry the transcript + endpoint context into the new session file via
    # the normal save path, under the new id. _save_current merges
    # metadata (title/source/cwd/metrics) the same way /resume would see it.
    from ._repl_state import _fresh_metrics
    state.session_id = new_id
    st.SESSION_ID = new_id  # mirror: undo stack swaps with the forked id
    # Re-key the undo store on the switch (0.128.3): the parent's stack stays
    # persisted under the OLD id (fork = copy); the fork starts from whatever
    # its own undo file holds (typically empty).
    from . import _undo as _undo_store
    _undo_store.swap_session()
    # Fresh forked conversation: drop the outgoing session's read anchors
    # (0.128.3, same rationale as /reset).
    from . import _staleness
    _staleness.clear()
    state.session_dirty = True
    meta: dict = {"forked_from": old_id}
    if label:
        meta["title"] = label
    # Through the SIGINT-guarded save path (0.129.0): a second ^C landing
    # inside the store-body join used to crash the REPL mid-switch (the fork
    # row unwritten while the original was flushed — the fork silently lost).
    # _save_current_locked is otherwise internal to _save_current. The fork
    # row must be DURABLE before the "forked" report: flush the async queue
    # right after, so the queued snapshot under the new id lands now.
    _save_current(state, meta)
    _flush_save()
    state.session_dirty = False
    # Fresh cumulative-token recorder for the new id (metrics are per-session).
    st._METRICS = _fresh_metrics(new_id)
    shown = f" as {label!r}" if label else ""
    print(f"{DIM}  ⑂ forked to session {new_id}{shown} "
          f"({len(state.messages)} messages carried; original {old_id} "
          f"kept — /resume to switch back){RESET}")


def _cmd_save(user: str, state: _ReplState) -> None:
    title = user.split(None, 1)[1].strip() if " " in user else None
    _save_current(state, {"title": title} if title else {})
    # 0.103.4: "saved" is a promise the transcript is durable — with
    # NX_ASYNC_SAVE the call above only queued it, so drain before reporting.
    _flush_save()
    state.session_dirty = False
    shown = f" as {title!r}" if title else ""
    print(f"{DIM}  ✓ saved session {state.session_id}{shown}{RESET}")


def _cmd_delete_handler(user: str, state: _ReplState) -> None:
    target = user.split(None, 1)[1].strip() if " " in user else None
    _cmd_delete(target, state.session_id)


# ---------------------------------------------------------------------------
# /sessions — alternate-screen browser (UIUX_SPEC Phase D, 0.137.0)
# ---------------------------------------------------------------------------
# The list → preview → confirm drill-down the spec asks for, over the SAME
# `_list_sessions` / `_load_session` / `_print_transcript` data the inline
# command uses. Resuming is not a new mutation path: the view returns the id
# and the command hands it to the existing `_cmd_resume_handler`, which is the
# exact code path /resume takes (save outgoing, flush, release claims,
# `resume_session`).

class SessionsView(_fullscreen.View):
    """Full-screen session browser: FTS5 body search, one-session preview,
    resume on confirm. Read-only — nothing but `/resume` runs on confirmation.

    `state` is the `_ReplState` the REPL handler passes; the view keeps only
    the session id it is currently on (`state.session_id`) so the current row
    can be marked, exactly like the inline list does.
    """

    surface = "sessions"
    title = "sessions"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.state = state
        self.current_id = getattr(state, "session_id", None)
        self.query = ""
        self.mode = "list"       # "list" | "preview"
        self.sel = 0
        self.confirm_id: str | None = None  # session awaiting delete confirm
        self.sessions: list[dict[str, Any]] = []
        self.preview: list[str] = []
        self.notice = ""
        #: session id to resume — the command runs the existing /resume path
        self.resume_id: str | None = None
        self.refresh()

    def refresh(self) -> None:
        if self.query:
            self.sessions = _search_sessions(self.query, limit=SESSION_LIST_MAX_LIMIT)
        else:
            self.sessions = _list_sessions(limit=SESSION_LIST_DEFAULT_LIMIT)
        self.sel = min(self.sel, max(0, len(self.sessions) - 1))
        self.mode = "list"
        self.preview = []

    def _open_preview(self) -> None:
        if not self.sessions:
            return
        sid = self.sessions[self.sel]["id"]
        data = _load_session(sid) or {}
        head = [
            f"session {sid}",
            f"title: {data.get('title') or '(untitled)'}",
            f"messages: {len(data.get('messages', []))} · "
            f"updated {_fmt_when(data.get('updated_at', 0))}",
            "── transcript ──",
        ]
        self.preview = head + _transcript_lines(data.get("messages", []))
        self.mode = "preview"

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        head = (f"{role('ACCENT')}sessions{RESET}{role('MUTED')} — "
                f"{hint_line('sessions')}{RESET}")
        where = f" matching {self.query!r}" if self.query else ""
        lines = [head]
        if self.mode == "preview" and self.sessions:
            title = f"session · {self.sessions[self.sel]['id']}"
            body = [ln[:width] for ln in self.preview]
            footer = ("R resume · d delete · esc back to the list"
                      if self.confirm_id is None
                      else f"delete {self.confirm_id[:12]}? y confirm · n cancel")
        else:
            title = (f"sessions · {len(self.sessions)} row(s){where} · "
                     f"page 1")
            body = []
            for i, s in enumerate(self.sessions):
                title_txt = s["title"] or "(empty)"
                mark = ("●" if s["id"] == self.current_id
                        else ("›" if i == self.sel else " "))
                body.append(
                    f"{role('ACCENT')}{mark}{RESET} "
                    f"{role('LABEL')}{s.get('id', '')[:12]}{RESET}  "
                    f"{title_txt[:max(10, width - 30)]} "
                    f"{role('MUTED')}· {s['n']} msg · "
                    f"{_fmt_when(s['updated_at'])}{RESET}")
                desc = s.get("description") or s.get("preview")
                if desc:
                    body.append(f"{role('MUTED')}      {desc[:width - 8]}{RESET}")
            footer = ("enter preview · / search · R resume · d delete · q/esc back"
                      if self.confirm_id is None
                      else f"delete {self.confirm_id[:12]}? y confirm · n cancel")
            if not self.sessions:
                body = [f"{role('WARNING')}no sessions{where}{RESET}"]
        lines.extend(panel(title, body, footer=footer))
        if self.notice:
            lines.append(f"{role('WARNING')}{self.notice[:width]}{RESET}")
        return lines[:height]

    def on_key(self, key: str, shell) -> str:
        if key != "y" or self.confirm_id is None:
            self.notice = ""
        if key == "j/k":
            # `_keys` advertises the pair as ONE hint label ("j/k move"); the
            # label is not a key, and the two real keys are handled below.
            return "repaint"
        if key in ("j", "down"):
            self.sel = min(max(0, len(self.sessions) - 1), self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            self.sel = max(0, self.sel - 1)
            return "repaint"
        if key == "/":
            q = shell.ask("search sessions (FTS5 body + title) ›", self.query)
            if q is not None:
                self.query = q
            self.refresh()
            return "repaint"
        if key == "enter":
            if self.mode == "preview":
                self.mode = "list"
            else:
                self._open_preview()
            return "repaint"
        if key in ("R", "tab"):
            return self._resume()
        # "d" is advertised in both the list and the preview footer; the
        # armed row is always `self.sessions[self.sel]` in either mode.
        if key == "d" and self.confirm_id is None:
            return self._arm_delete()
        if key in ("y", "n") and self.confirm_id:
            return self._delete(shell)
        if key == "esc":
            if self.confirm_id:            # cancel the confirm, not the view
                self.confirm_id = None
                return "repaint"
            if self.mode == "preview":     # back out to the list first
                self.mode = "list"
                return "repaint"
            return "quit"
        if key in ("q", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"

    def _arm_delete(self) -> str:
        if not self.sessions:
            self.notice = "nothing to delete"
            return "repaint"
        sid = self.sessions[self.sel]["id"]
        if sid == self.current_id:
            self.notice = "can't delete the session you're currently in"
            return "repaint"
        self.confirm_id = sid
        return "repaint"

    def _delete(self, shell) -> str:
        sid, self.confirm_id = self.confirm_id, None
        if not sid:
            return "repaint"
        # The armed 'y' keypress IS the confirmation (no second typed y/n).

        def _act() -> None:
            ok = _delete_session(sid)
            self.notice = (f"✓ deleted session {sid[:12]}" if ok
                           else f"✗ could not delete session {sid[:12]}")
            self.refresh()

        self.pending_action = _act
        return "run_stay"

    def _resume(self) -> str:
        if not self.sessions:
            self.notice = "nothing to resume"
            return "repaint"
        sid = self.sessions[self.sel]["id"]
        if sid == self.current_id:
            self.notice = "already on that session"
            return "repaint"
        self.resume_id = sid
        return "quit"


def _search_sessions(query: str, limit: int) -> list[dict[str, Any]]:
    """FTS5 body hits, mapped to the same row shape `_list_sessions` returns
    (so the browser renders one kind of row). Lies about nothing: a hit whose
    session row is gone is dropped."""
    from ._sessions_search import search_sessions
    try:
        hits = search_sessions(query, limit=limit)
    except Exception:
        hits = []
    known = {s["id"]: s for s in _list_sessions(limit=SESSION_LIST_MAX_LIMIT)}
    rows: list[dict[str, Any]] = []
    for h in hits:
        sid = h.get("id")
        if sid not in known:
            continue
        row = dict(known[sid])
        row["preview"] = h.get("snippet") or row.get("preview")
        rows.append(row)
    return rows


def _transcript_lines(messages: list[dict[str, Any]],
                      max_chars: int = 160) -> list[str]:
    """`_print_transcript`'s one-line-per-message recap, returned as lines
    instead of printed — the inline path keeps printing it unchanged."""
    out: list[str] = []
    from ._render import role
    for m in messages:
        r = m.get("role", "?")
        if r == "system":
            continue
        content = strip_ansi_for_display(_flatten_message_text(m, per_msg=max_chars))
        role_name = {"user": "ACCENT", "assistant": "TEXT", "tool": "MUTED"}.get(r)
        label = f"{role(role_name)}{r:>9}{RESET}" if role_name else f"{r:>9}"
        out.append(f"  {label}  {content}")
    return out


def cmd_sessions_view(user: str, state: _ReplState) -> None:
    """The `/sessions` command's Phase D path: the alternate-screen browser,
    with the inline list as the fallback (off tty / NX_FULLSCREEN=0). Resuming
    from the browser re-enters `/resume` with the chosen target — the same
    handler, the same downstream path, no new mutation."""
    from . import _fullscreen
    view = SessionsView(state)
    fullscreen = _fullscreen.open_view(
        view, fallback=lambda: _cmd_sessions(user, state.session_id))
    if fullscreen and view.resume_id:
        _cmd_resume_handler(f"/resume {view.resume_id}", state)
