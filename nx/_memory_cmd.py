"""The `/memory` subtree and the model-callable `memory_write` /
`memory_search` tools.

Two storage scopes:
  - global:  ~/.nx/memories/            (cross-project, via _memories_dir)
  - project: <project_root>/.nx/memories/ (via _project_memories_dir)

`/memory save|remember|list` are the manual slash commands (the model
round-trips the whole session into one distilled note). The two *tools*
are what the model calls itself mid-turn: `memory_write` to persist a
discrete learning, `memory_search` to recall prior notes. The session-start
memory-awareness note (`_load_memory_note`) lists saved titles so the model
knows memories exist without burning tokens on the full bodies.

Self-contained aside from a model round-trip (via `st.client`), the
transcript-flattening helper shared with `_sessions_cmd.py`, and
`_confirm()` (read via the top-level module, same pattern as _edit_tools).
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from ._ansi import DIM, RESET, C
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from ._sessions_io import _flatten_message_text
from . import _state as st
from . import _memories_dir, _project_memories_dir


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


def _cmd_memory_handler(user: str, state: Any) -> None:
    api_calls_before = st._METRICS.api_calls if st._METRICS is not None else 0
    _cmd_memory(state.messages, user)
    # `/memory save` is a model call but doesn't alter the conversation.
    # Persist its updated totals immediately instead of waiting for a later
    # conversational turn (or /quit) to happen to save the session.
    if st._METRICS is not None and st._METRICS.api_calls != api_calls_before:
        from ._repl_state import _save_current
        _save_current(state)


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
    if st.client is None or st.MODEL is None:
        print(f"{C.ERROR}  ✗ no active source — memory not saved{RESET}")
        return
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

    _record_usage(
        getattr(resp, "usage", None),
        fallback_chars=len(resp.choices[0].message.content or ""),
        fallback_prompt_chars=len(prompt),
    )

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


# --- model-callable tools + session-start injection ---------------------------

# Memory bodies are markdown; we reuse the H1 title convention the slash
# commands established. Keep these terse: they're sent every turn (SYSTEM
# mentions them) and the bodies run inside the tool-dispatch spinner.


def _memory_dirs_for(project_root: str | os.PathLike[str] | None) -> list[str]:
    """Resolved memory dirs that actually exist, global first then project.
    Project-scoped memory only applies when there is a real project root."""
    dirs = []
    g = _memories_dir()
    if os.path.isdir(g):
        dirs.append(g)
    p = _project_memories_dir(project_root)
    if p and os.path.isdir(p):
        dirs.append(p)
    return dirs


def _confirm_memory(action: str, preview: str | None = None) -> bool:
    """`_confirm` is monkeypatched by tests via the top-level module, so call
    it through `import nx as _m` — same pattern as _edit_tools."""
    import nx as _m
    return _m._confirm(action, preview=preview)


def memory_write(
    title: str, body: str, scope: str = "project", **_: Any
) -> str:
    """Persist a memory note the model can recall in future sessions.

    scope="project" writes to <project_root>/.nx/memories/ (falls back to
    global when there is no project root); scope="global" writes to
    ~/.nx/memories/ (cross-project). The note is a markdown file named
    <timestamp>_<slug>.md starting with a `# Title` line.

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

    if scope == "project":
        pdir = _project_memories_dir(getattr(st, "PROJECT_ROOT", None))
        if not pdir:
            # No project root (bare cwd / outside a repo): fall back to global
            # rather than refusing — a memory worth saving shouldn't be lost.
            scope = "global"
            pdir = None
    else:
        pdir = None

    memories_dir = pdir if scope == "project" and pdir else _memories_dir()
    os.makedirs(memories_dir, exist_ok=True)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    fname_stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = _slugify(title)
    fname = f"{fname_stamp}_{slug}.md"
    content = f"# {title}\n\n{body}\n\n---\n_scope: {scope}  _saved: {stamp}_\n"
    path = os.path.join(memories_dir, fname)

    action = f"write memory {path} ({len(content)} bytes)"
    if not _confirm_memory(action, preview=None):
        return "DENIED by user"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    rel = path
    try:
        rel = os.path.relpath(path)
    except ValueError:
        pass
    return f"saved {scope} memory: {title}\nfile: {rel}"


def _memory_snippet(content: str, query: str, max_lines: int = 4) -> str:
    """First `max_lines` of a memory body (title line stripped) as a preview."""
    lines = content.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("# ") and not out:
            continue  # skip the H1 title (already shown separately)
        if not ln.strip():
            continue
        out.append(ln.strip())
        if len(out) >= max_lines:
            break
    return " / ".join(out)


def memory_search(query: str, scope: str = "project", **_: Any) -> str:
    """Keyword-search saved memories (global + project) and return title +
    snippet for each match, newest first. Read-only; no approval. Use this to
    recall prior learnings before re-deriving them.

    scope="project" searches both project and global memories; scope="global"
    searches only global. An empty query returns the newest few titles as a
    quick index (so the model can discover what exists).
    """
    query = (query or "").strip()
    roots = _memory_dirs_for(
        getattr(st, "PROJECT_ROOT", None) if scope == "project" else None
    )
    if not roots:
        return "[no memories saved yet]"
    hits: list[tuple[float, str, str, str, str]] = []  # mtime, name, title, snippet, scope
    for d in roots:
        label = "project" if d == _project_memories_dir(
            getattr(st, "PROJECT_ROOT", None)) else "global"
        try:
            names = sorted(os.listdir(d), reverse=True)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(d, name)
            try:
                mtime = os.path.getmtime(path)
                with open(path, encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue
            title = ""
            for ln in content.splitlines():
                if ln.startswith("# "):
                    title = ln[2:].strip()
                    break
            if not title:
                title = name[:-3]
            if query and query.lower() not in content.lower():
                continue
            hits.append((mtime, name, title, _memory_snippet(content, query), label))
    if not hits:
        return f"[no memories matching {query!r}]" if query else "[no memories saved yet]"
    hits.sort(key=lambda x: x[0], reverse=True)
    out = [f"memories matching {query!r} ({len(hits)}):" if query else f"memories ({len(hits)}):"]
    for _mt, name, title, snippet, label in hits:
        out.append(f"[{label}] {title} ({name})")
        out.append(f"  {snippet}")
    return "\n".join(out)


def _load_memory_note(project_root: str | os.PathLike[str] | None) -> dict[str, str] | None:
    """One-time user-role note appended to a fresh session's message list so
    the model knows memories exist and which are already saved. Titles only
    (not bodies) and capped — keeps the per-turn token floor (see
    test_initial_context.py) from growing with how much you've remembered.
    Returns None when there are no memories in either scope.
    """
    roots = _memory_dirs_for(project_root)
    if not roots:
        return None
    titles: list[str] = []
    for d in roots:
        label = "project" if d == _project_memories_dir(project_root) else "global"
        try:
            names = sorted(os.listdir(d), reverse=True)
        except OSError:
            continue
        count = 0
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
            except OSError:
                continue
            t = first_line[2:].strip() if first_line.startswith("# ") else name[:-3]
            titles.append(f"[{label}] {t}")
            count += 1
            if count >= 5:
                break
    if not titles:
        return None
    body = "\n".join(titles)
    return {
        "role": "user",
        "content": (
            f"[Saved memories — use memory_search to read any, memory_write to add:]\n{body}"
        ),
    }
