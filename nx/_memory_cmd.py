"""The `/memory` subtree: `save`/`remember`/`list` subcommands for
distilling a session into a markdown memory note under `~/.nx/memories/`.

Self-contained aside from a model round-trip (via `st.client`) and the
transcript-flattening helper shared with `_sessions_cmd.py`.
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
from . import _memories_dir


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
