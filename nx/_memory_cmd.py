"""The `/memory` slash-command UI and the model-callable `memory_write` tool.

Split (0.88.0): the storage layer lives in `nx/_memory_store.py` (dirs, the
one scanner, the one writer, footer/kind conventions), the search layer in
`nx/_memory_search.py` (ranking, scan, `memory_search`/`memory_delete`), the
session-start note in `nx/_memory_note.py`, and the `archive/` read side in
`nx/_memory_archive.py`. This module keeps the interactive `/memory
save|remember|delete|list|archived|optimize` handlers, the transcript
distillation (`/memory save` round-trips the session through the model), and
`memory_write` — the one write tool, which goes through the store's writer
with gate=True.

Names moved to the sibling modules are re-exported below so existing
`from ._memory_cmd import …` call sites (and tests patching
`_memory_cmd._refresh_memory_note`) keep working unchanged."""
from __future__ import annotations

import os
import time
from typing import Any

from ._ansi import C, DIM, RESET
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from ._sessions_io import _flatten_message_text
from . import _state as st

# --- re-exports from the split-out layers (single import surface preserved) ---
from ._memory_store import (  # noqa: F401 (re-exports)
    MEMORY_KINDS, _MEMORY_FOOTER_RE, _confirm_memory, _first_title,
    _iter_memory_dirs, _iter_notes, _memories_dir, _memory_confirm_preview,
    _memory_dirs_for, _note_kind, _project_memories_dir, _publish_memory,
    _read_memory_file, _read_memory_title, _read_note_preview, _slugify,
    _write_memory_file,
)
from ._memory_search import (  # noqa: F401 (re-exports)
    MEMORY_SEARCH_DEFAULT_LIMIT, _MemoryHit, _memory_scan,
    _memory_search_limit, _memory_snippet, _rank_hit, _scan_memory_entries,
    memory_delete, memory_search,
)
from ._memory_note import (  # noqa: F401 (re-exports)
    MEMORY_NOTE_CLEARED, MEMORY_NOTE_MARKER, _add_memory_note_if_absent,
    _find_memory_note_index, _last_memory_note_index, _load_memory_note,
    _refresh_memory_note,
)
from ._memory_archive import (  # noqa: F401 (re-exports)
    _cmd_memory_archived, _iter_archive_notes,
)

# --- /memory slash commands ----------------------------------------------------

def _cmd_memory(messages: list[dict[str, Any]], user_input: str) -> None:
    """Handle /memory save|remember|delete|list  — manual memories.

    /memory save [focus...]      — distill session into a memory (optional focus
                                   guidance for the model: e.g. "focus on config")
    /memory remember <query>     — keyword search through saved memories
    /memory delete <target>      — remove one memory (title/file/prefix)
    /memory list                 — list all saved memories
    /memory archived [target]    — list notes /memory optimize archived, or
                                   restore one by title/file/prefix
    /memory optimize [dry] [focus]  — librarian pass over the store: fold
        duplicates together and archive notes a matured project outgrew
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

    if sub == "delete":
        target = parts[2] if len(parts) >= 3 else ""
        _cmd_memory_delete(target)
        return

    if sub == "save":
        notes = parts[2] if len(parts) >= 3 else None
        _cmd_memory_save(messages, notes)
        return

    if sub == "archived":
        target = parts[2] if len(parts) >= 3 else ""
        _cmd_memory_archived(target)
        return

    if sub == "optimize":
        # Imported lazily: _memory_optimize imports this module at module level.
        from ._memory_optimize import _cmd_memory_optimize
        _cmd_memory_optimize(messages, parts[2] if len(parts) >= 3 else "")
        return

    if sub == "reflect":
        # Imported lazily: _memory_reflect imports this module at module level.
        from ._memory_reflect import _cmd_memory_reflect
        _cmd_memory_reflect(parts[2] if len(parts) >= 3 else "")
        return

    # Bare /memory — show usage
    print(f"{DIM}  usage:{RESET}")
    print(f"{DIM}    /memory save [focus...]     save a memory (optional focus guidance){RESET}")
    print(f"{DIM}    /memory remember <query>    search memories{RESET}")
    print(f"{DIM}    /memory delete <target>     remove one memory (title/file/prefix){RESET}")
    print(f"{DIM}    /memory list                list all memories{RESET}")
    print(f"{DIM}    /memory archived [target]   list archived notes, or restore one{RESET}")
    print(f"{DIM}    /memory optimize [dry] [focus]  merge duplicates, retire stale notes "
          f"(dry prints the plan only){RESET}")
    print(f"{DIM}    /memory reflect [focus]     cross-project pass: promote recurring "
          f"preferences, surface shared patterns{RESET}")


def _cmd_memory_handler(user: str, state: Any) -> None:
    api_calls_before = st._METRICS.api_calls if st._METRICS is not None else 0
    _cmd_memory(state.messages, user)
    # `/memory save` is a model call but doesn't alter the conversation.
    # Persist its updated totals immediately instead of waiting for a later
    # conversational turn (or /quit) to happen to save the session.
    if st._METRICS is not None and st._METRICS.api_calls != api_calls_before:
        from ._repl_state import _save_current
        _save_current(state)


_MEMORY_TRANSCRIPT_CHARS = 48000  # hard cap on the /memory save distillation input


def _render_transcript_for_memory(messages: list[dict[str, Any]]) -> str:
    """Render session messages into a compact text transcript for the model.

    Skips the system prompt, truncates long tool outputs, and shows tool calls.
    The whole transcript is also capped (_MEMORY_TRANSCRIPT_CHARS): a long
    session's tail is dropped, and the cut is announced in-band so the
    distillation knows it saw a truncated session.
    """
    out = []
    total = 0
    truncated = False
    for m in messages:
        if m.get("role") == "system":
            continue
        if total >= _MEMORY_TRANSCRIPT_CHARS:
            truncated = True
            break
        role = m.get("role", "?")
        text = _flatten_message_text(m, per_msg=2000, tool_call_arg_chars=100)
        if total + len(text) > _MEMORY_TRANSCRIPT_CHARS:
            text = text[:_MEMORY_TRANSCRIPT_CHARS - total]
            truncated = True
        total += len(text)
        out.append(f"[{role}] {text}")
    if truncated:
        out.append("[transcript truncated — only the earlier part of the session appears above]")
    return "\n\n".join(out)


def _cmd_memory_save(messages: list[dict[str, Any]], notes: str | None = None) -> None:
    """Distill a session into a memory note.

    The model round-trips the transcript into a markdown memory note, saved to
    the project scope (<root>/.nx/memories/, falling back to global when there
    is no project root) — the same resolution as the `memory_write` tool, so
    both write paths share one convention (see _write_memory_file).
    ``notes``, if provided, serves as guidance: "focus on the build system",
    "capture the API endpoint details", etc.
    """
    if st.client is None or st.MODEL is None:
        print(f"{C.ERROR}  ✗ no active source — memory not saved{RESET}")
        return

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
        body = (resp.choices[0].message.content or "").strip()
        _record_usage(
            getattr(resp, "usage", None),
            fallback_chars=len(body),
            fallback_prompt_chars=len(prompt),
        )
    except APIConnectionError:
        print(f"{C.ERROR}  ✗ can't reach {st.client.base_url} — memory not saved{RESET}")
        return
    except Exception as e:
        print(f"{C.ERROR}  ✗ memory save failed: {type(e).__name__}: {e}{RESET}")
        return

    if not body:
        print(f"{C.ERROR}  ✗ model returned empty — memory not saved{RESET}")
        return

    title = _first_title(body, "")
    if not title:
        title = body.split("\n")[0][:80].strip() or "untitled"

    try:
        written = _write_memory_file(title, body, "project")
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't write memory: {e}{RESET}")
        return
    if written is None:  # gate denied — not possible with gate=False; kept safe
        return
    path, _scope = written
    print(f"{DIM}  ✓ saved memory: {title}{RESET}")
    print(f"{DIM}    {path}{RESET}")


def _cmd_memory_remember(query: str) -> None:
    """Keyword-search through saved memories — same scan/ranking as the
    memory_search tool, with a longer per-hit preview for the human."""
    query = (query or "").strip()
    if not query:
        print(f"{C.WARNING}  usage: /memory remember <search terms>{RESET}")
        return

    hits, overflow, total = _memory_scan(query, "project", 0)
    if not hits:
        print(f"{DIM}  no memories matching {query!r}{RESET}")
        return

    shown_n = f"{len(hits)} of {total}" if overflow else f"{total}"
    print(f"{DIM}  memories matching {query!r} ({shown_n}):{RESET}")
    for hit in hits:
        print(f"\n{C.ACCENT}  ┌─ {hit.title}{RESET}  {DIM}({hit.name}){RESET}")
        # Snippet preview (title line already shown separately)
        for line in hit.snippet.split(" / "):
            print(f"{DIM}  │{RESET} {line[:100]}")


def _cmd_memory_delete(target: str) -> None:
    """`/memory delete <title|file|prefix>` — same resolver and approval gate
    as the memory_delete tool."""
    if not target:
        print(f"{C.WARNING}  usage: /memory delete <title | filename | unique prefix>{RESET}")
        return
    res = memory_delete(target=target, scope="project")
    for line in res.split("\n"):
        print(f"{DIM}  {line}{RESET}")


def _cmd_memory_list() -> None:
    """List all saved memories with titles and dates — both scopes (global
    first, the same order every scanner uses), each row labelled. A
    project-only repo used to report "no memories saved yet" here."""
    files = []
    for d, label in _iter_memory_dirs("project"):
        for name, _path, mtime, title, _content in _iter_notes(d, titles_only=True):
            files.append((mtime, name, title, label))

    if not files:
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    files.sort(key=lambda x: x[0], reverse=True)
    print(f"{DIM}  memories ({len(files)}):{RESET}")
    for i, (mtime, name, title, label) in enumerate(files, 1):
        date = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        print(f"  {C.SUCCESS}{i:>2}.{RESET} [{label}] {title}  {DIM}({date}){RESET}")


def memory_write(
    title: str, body: str, scope: str = "project", kind: str = "", **_: Any
) -> str:
    """Persist a memory note the model can recall in future sessions.

    scope="project" writes to <project_root>/.nx/memories/ (falls back to
    global when there is no project root); scope="global" writes to
    ~/.nx/memories/ (cross-project). The note is a markdown file named
    <timestamp>_<slug>.md starting with a `# Title` line — written 0600 via
    tmp+rename, and never clobbering an existing note.

    kind: optional tag — "preference" (user's durable style/tooling choice),
    "fact", "gotcha", or "episodic" (one-off project event). Untagged is fine;
    the tag only helps ranking and injection treat the note appropriately.

    Search first: prefer updating/superseding an existing note (memory_delete
    the stale one) over accumulating near-duplicates.

    Goes through the approval gate (it writes a file) — same as write_file.
    """
    title = (title or "").strip()
    body = (body or "").strip()
    if not title:
        return "ERROR: memory_write requires a non-empty `title`"
    if not body:
        return "ERROR: memory_write requires a non-empty `body`"
    if scope not in ("project", "global"):
        scope = "project"
    if kind and kind not in MEMORY_KINDS:
        kind = ""  # unrecognized tag: store kind-free rather than refuse

    try:
        written = _write_memory_file(title, body, scope, gate=True, kind=kind)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    if written is None:
        return "DENIED by user"
    path, scope = written  # kind lives in the file footer, not the return
    rel = path
    try:
        rel = os.path.relpath(path)
    except ValueError:
        pass
    return f"saved {scope} memory: {title}\nfile: {rel}"
