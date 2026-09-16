"""The `/memory` subtree and the model-callable `memory_write` /
`memory_search` / `memory_delete` tools.

Two storage scopes:
  - global:  ~/.nx/memories/            (cross-project, via _memories_dir)
  - project: <project_root>/.nx/memories/ (via _project_memories_dir)

`/memory save|remember|delete|list|optimize` are the manual slash commands
(`/memory save` round-trips the whole session into one distilled note;
`/memory optimize` is the consolidation pass in `_memory_optimize.py`).
The *tools* are
what the model calls itself mid-turn: `memory_write` to persist a discrete
learning, `memory_search` to recall prior notes, `memory_delete` to remove a
superseded one. The session-start memory-awareness note (`_load_memory_note`)
lists saved titles so the model knows memories exist without burning tokens
on the full bodies.

Every write — tool or slash command — goes through `_write_memory_file`: one
filename convention (`<timestamp>_<slug>.md`, `-2`/`-3`... on a same-second
same-title collision), one footer convention (`_scope:`/`_saved:`), atomic
tmp+rename, private 0600 permissions (memories are distilled from sessions
and may contain paths/commands/credentials; same posture as session files).
Publication is claim-based (`atomic_write_private_noreplace`): the filename
is not picked by an exists-then-write check that two concurrent writers (the
memory-merge workers) can both pass — the first to publish the bytes wins
and the loser takes the next suffix.

Every read goes through `_iter_notes` — the one scanner over a memories dir —
so title extraction, mtime handling and unreadable-file tolerance live in
exactly one loop. `memory_search` accelerates its per-dir scan with the FTS5
index in `nx/_memindex.py` (a self-healing cache; NX_MEMORY_FTS=0, or a
sqlite build without FTS5/trigram, falls back to the plain scan); hits from
either path are re-scored by the same `_rank_hit`, so ranking, the cap and
the overflow note are identical whichever path served the rows.

Searches keep the contract: every query term must appear, title hits rank
above body hits, newest wins ties, and results are capped at
NX_MEMORY_SEARCH_LIMIT so a big store can't dump an unbounded context blob
into a turn. `/memory remember` shares the scan (unlimited) for its previews.

Self-contained aside from a model round-trip (via `st.client`), the
transcript-flattening helper shared with `_sessions_cmd.py`, and `_confirm()`
(read via the top-level module, same pattern as _edit_tools). All writes go
through nx/_private.py (0600/0700, umask-proof)."""
from __future__ import annotations

import codecs
import os
import re
import time
from typing import Any, NamedTuple

from ._ansi import DIM, RESET, C
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from ._private import (atomic_write_private_noreplace,
                       private_dir)
from ._sessions_io import _flatten_message_text
from . import _state as st
from . import _memories_dir, _project_memories_dir

# Cap on memory_search results. A store that outgrows the cap says so
# in-band ("… N more not shown") instead of silently hiding or flooding.
MEMORY_SEARCH_DEFAULT_LIMIT = 8


def _memory_search_limit() -> int:
    """Result cap for memory_search (/memory remember stays unlimited).
    0 = no cap. Read via nx._env_int at call time so tests can set the env."""
    import nx as _m
    try:
        return max(0, _m._env_int("NX_MEMORY_SEARCH_LIMIT", MEMORY_SEARCH_DEFAULT_LIMIT))
    except Exception:
        return MEMORY_SEARCH_DEFAULT_LIMIT


def _first_title(content: str, fallback: str) -> str:
    """The `# Title` H1 of a memory note, else the filename stem."""
    for ln in content.splitlines():
        if ln.startswith("# "):
            t = ln[2:].strip()
            if t:
                return t
            break
    stem = fallback[:-3] if fallback.endswith(".md") else fallback
    return stem or fallback


def _slugify(text: str, maxlen: int = 60) -> str:
    """Turn a title into a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if len(s) > maxlen:
        s = s[:maxlen].rstrip("-")
    return s or "untitled"


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

    # Bare /memory — show usage
    print(f"{DIM}  usage:{RESET}")
    print(f"{DIM}    /memory save [focus...]     save a memory (optional focus guidance){RESET}")
    print(f"{DIM}    /memory remember <query>    search memories{RESET}")
    print(f"{DIM}    /memory delete <target>     remove one memory (title/file/prefix){RESET}")
    print(f"{DIM}    /memory list                list all memories{RESET}")
    print(f"{DIM}    /memory archived [target]   list archived notes, or restore one{RESET}")
    print(f"{DIM}    /memory optimize [dry] [focus]  merge duplicates, retire stale notes "
          f"(dry prints the plan only){RESET}")


def _cmd_memory_handler(user: str, state: Any) -> None:
    api_calls_before = st._METRICS.api_calls if st._METRICS is not None else 0
    _cmd_memory(state.messages, user)
    # `/memory save` is a model call but doesn't alter the conversation.
    # Persist its updated totals immediately instead of waiting for a later
    # conversational turn (or /quit) to happen to save the session.
    if st._METRICS is not None and st._METRICS.api_calls != api_calls_before:
        from ._repl_state import _save_current
        _save_current(state)


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


def _iter_archive_notes(d: str) -> list[tuple[str, str, float, str, str]]:
    """Every archived note under `<d>/archive/<run>/` as the same shape the live
    scanner yields, (run, name, mtime, title, path). Archived notes are exactly
    what `/memory optimize` moved there, so this is the read side of the
    reversible-by-`mv` promise the archive makes. Newest run first."""
    out: list[tuple[str, str, float, str, str]] = []
    archive = os.path.join(d, "archive")
    try:
        runs = sorted(os.listdir(archive), reverse=True)
    except OSError:
        return out
    for run in runs:
        run_dir = os.path.join(archive, run)
        if not os.path.isdir(run_dir):
            continue
        try:
            names = sorted(os.listdir(run_dir), reverse=True)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(run_dir, name)
            head = _read_memory_title(path, name)
            if head is None:
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            out.append((run, name, mtime, head[0], path))
    return out


def _cmd_memory_archived(target: str) -> None:
    """`/memory archived [target]` — list archived (retired) notes, or restore
    one by title / filename / unique prefix. Restoring moves the file back into
    its scope dir with a fresh name, so it re-enters the store (and the next
    search re-indexes it)."""
    entries = []
    for d, label in _iter_memory_dirs("project"):
        for run, name, mtime, title, path in _iter_archive_notes(d):
            entries.append((run, name, mtime, title, path, label, d))

    if not target:
        if not entries:
            print(f"{DIM}  no archived memories{RESET}")
            return
        print(f"{DIM}  archived memories ({len(entries)}):{RESET}")
        for run, name, _mtime, title, _path, label, _d in sorted(
                entries, key=lambda x: x[2], reverse=True):
            print(f"  {C.SUCCESS}·{RESET} [{label}] {title}  "
                  f"{DIM}({run} · {name}){RESET}")
        print(f"{DIM}  restore one with: /memory archived <title|filename|prefix>{RESET}")
        return

    low = target.lower()
    exact = [e for e in entries
             if e[1].lower() == low or e[1][:-3].lower() == low or e[3].lower() == low]
    if len(exact) == 1:
        chosen = exact[0]
    else:
        cands = exact if exact else [e for e in entries if e[3].lower().startswith(low)]
        if len(cands) > 1:
            listing = "\n".join(f"  - {e[3]} ({e[1]})" for e in cands[:10])
            print(f"{C.WARNING}  {target!r} matches {len(cands)} archived notes — "
                  f"use an exact title or filename:{RESET}\n{listing}")
            return
        if not cands:
            print(f"{DIM}  no archived memory matching {target!r}{RESET}")
            return
        chosen = cands[0]

    _run, name, _mtime, title, path, label, scope_dir = chosen
    if not _confirm_memory(f"restore memory {path} → {scope_dir}"):
        return
    # Publish with a fresh timestamped name via the one writer, then drop the
    # archived original — the restore goes through the same 0600/no-clobber path
    # as any other write, so a restored note is byte-conventional.
    try:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", "replace")
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't read {path}: {e}{RESET}")
        return
    # Strip the archived note's own footer — the writer appends a fresh one.
    body = _MEMORY_FOOTER_RE.sub("", content).rstrip()
    written = _write_memory_file(title, body, label, gate=False)
    if written is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass  # the restored copy exists; a stale archive file is harmless
    print(f"  {C.SUCCESS}restored{RESET} {title}  {DIM}→ {written[0]}{RESET}")


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


def _iter_memory_dirs(scope: str = "project") -> list[tuple[str, str]]:
    """Memory dirs to operate on as (dir, label) pairs — global first then
    project (scope="project"), or global only (scope="global"). Unlike
    _memory_dirs_for this includes not-yet-existing dirs; callers scan
    defensively."""
    if scope == "global":
        return [(_memories_dir(), "global")]
    pairs = [(_memories_dir(), "global")]
    p = _project_memories_dir(getattr(st, "PROJECT_ROOT", None))
    if p:
        pairs.append((p, "project"))
    return pairs


def _confirm_memory(action: str, preview: str | None = None) -> bool:
    """`_confirm` is monkeypatched by tests via the top-level module, so call
    it through `import nx as _m` — same pattern as _edit_tools."""
    import nx as _m
    return _m._confirm(action, preview=preview)


class _MemoryHit(NamedTuple):
    mtime: float
    name: str
    title: str
    snippet: str
    label: str
    score: float


def _read_memory_file(path: str, name: str) -> tuple[str, str] | None:
    """(title, content) for one memory file, or None when unreadable
    (missing, permissions, or a non-UTF-8 body)."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return None
    return _first_title(content, name), content


# The H1 of a memory note lives in its first line, so a title-only caller needs
# a bounded head read, not the whole body: _load_memory_note reads every note in
# both scopes at session start just for titles (a 56-note store is ~180 KB).
_TITLE_HEAD_BYTES = 256


def _read_memory_title(path: str, name: str) -> tuple[str, str] | None:
    """(title, head_text) for one note from a bounded head read, or None when
    the head is unreadable or not valid UTF-8 — the same skip a full read makes,
    so the scanner tolerates a broken note identically on both paths. The title
    falls back to the filename stem when no H1 appears in the head.

    A note whose first bytes are valid but whose *body* later is not would pass
    here and be skipped by a full read; that is the one (rare) divergence, and
    it only widens which titles a title-only caller lists.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(_TITLE_HEAD_BYTES)
    except OSError:
        return None
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        # final=False: a 256-byte cut may land mid-character; only genuinely
        # invalid bytes raise, matching the full read's UnicodeDecodeError skip.
        text = decoder.decode(head, final=False)
    except UnicodeDecodeError:
        return None
    for ln in text.splitlines():
        if ln.startswith("# "):
            t = ln[2:].strip()
            if t:
                return t, text
            break
    stem = name[:-3] if name.endswith(".md") else name
    return (stem or name), text


_MEMORY_PREVIEW_ELISION = "\n... (truncated)"


def _memory_confirm_preview(title, content, max_lines=12, max_chars=700):
    """Terminal-UX preview for a memory approval prompt: the title plus a
    bounded head of the body, so the user sees what is being written. Never
    sent to the model (does not touch the token floor). A body is unbounded, so
    the head is line- and char-capped with an elision marker."""
    raw = content or ""
    if not raw.strip():
        doc = "# " + title
    elif raw.lstrip().startswith("# "):
        doc = raw
    else:
        doc = "# " + title + "\n\n" + raw
    rows = doc.strip().splitlines()
    clipped = len(rows) > max_lines
    text = "\n".join(rows[:max_lines])
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        clipped = True
    if clipped:
        text = text + _MEMORY_PREVIEW_ELISION
    return text


def _read_note_preview(path: str, max_chars: int = 700) -> str:
    """A bounded head of a note's text for a delete-approval preview. The file
    was already resolved by the caller; an unreadable note yields '' so the
    prompt still shows the title alone."""
    try:
        with open(path, "rb") as f:
            raw = f.read(max_chars * 4)
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")[:max_chars]


_MEMORY_FOOTER_RE = re.compile(r"\n---\n_scope:.*$", re.DOTALL)


def _rank_hit(content: str, query: str) -> float:
    """Score one memory against a multi-term query (the caller has already
    verified every term appears somewhere): a term in the `# Title` outweighs
    body occurrences, and body occurrences are capped so a term repeated all
    over one long note can't dominate ranking."""
    low = content.lower()
    title_low = None
    for ln in content.splitlines():
        if ln.startswith("# "):
            title_low = ln[2:].strip().lower()
            break
    score = 0.0
    for term in query.lower().split():
        if title_low and term in title_low:
            score += 3.0
        else:
            score += min(2.0, 0.5 * low.count(term))
    return score


def _publish_memory(path: str, content: str) -> bool:
    """Publish one memory note 0600 and atomically, WITHOUT clobbering: the
    final step is a claim (os.link fails with FileExistsError when another
    writer just took this exact filename — the old exists-then-write race
    under the merge workers), not a replace that would silently drop the
    other writer's bytes. Returns False when the name was taken."""
    try:
        atomic_write_private_noreplace(path, content)
        return True
    except FileExistsError:
        return False


def _iter_notes(d: str, *, titles_only: bool = False
                ) -> list[tuple[str, str, float, str, str]]:
    """The one scanner over a memories dir: every readable note as
    (name, path, mtime, title, content), sorted newest-name first (the
    `<timestamp>_` prefix makes name order the creation order). Unreadable
    files (missing, permissions, non-UTF-8) are skipped — every other
    scanner builds on this, so they all tolerate a broken note identically.
    The `archive/` subdir is invisible by construction (a directory never
    ends in `.md`).

    `titles_only=True` keeps the exact same list shape but fills `content` with
    just the head it read for the title — for callers that use only (name,
    mtime, title) (memory_delete's resolver, `/memory list`, the session-start
    note) this avoids reading every body in the store. Any caller that uses the
    body must take the default.
    """
    out: list[tuple[str, str, float, str, str]] = []
    try:
        names = sorted(os.listdir(d), reverse=True)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".md"):
            continue
        path = os.path.join(d, name)
        if titles_only:
            head = _read_memory_title(path, name)
            if head is None:
                continue
            title, content = head[0], head[1]
        else:
            read = _read_memory_file(path, name)
            if read is None:
                continue
            title, content = read
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        out.append((name, path, mtime, title, content))
    return out


def _write_memory_file(title: str, body: str, scope: str, gate: bool = False) -> tuple[str, str] | None:
    """The single memory writer: shared by the memory_write tool and
    `/memory save` so both produce identical files. Returns (path, scope), or
    None when gate=True and the user denied the confirmation.

    Resolves project→global fallback (no project root), avoids a doubled H1
    when the body already starts with its own title, makedirs 0700, and never
    clobbers: a same-second same-title write gets a `-2`/`-3`… suffix instead
    of silently overwriting its predecessor — and because publication is a
    claim (see _publish_memory), two concurrent writers that pick the same
    name cannot erase each other's note.

    gate=True runs the approval gate (same action shape write_file uses — the
    risk classifier's `write memory <path> (N bytes)` regex) after the path is
    resolved but before anything is written. The user-invoked `/memory save`
    passes gate=False: the user asked for it directly.
    """
    title = (title or "").strip()
    body = (body or "").strip()
    if not title:
        title = _first_title(body, "") or (
            body.split("\n")[0][:80].strip() if body else "") or "untitled"
    # Avoid a doubled H1 when the distilled body already carries its title.
    lines = body.splitlines()
    if lines and lines[0].startswith("# ") and lines[0][2:].strip() == title:
        body = "\n".join(lines[1:]).strip()

    if scope == "global":
        memories_dir = _memories_dir()
    else:
        pdir = _project_memories_dir(getattr(st, "PROJECT_ROOT", None))
        if not pdir:
            # No project root (bare cwd / outside a repo): fall back to global
            # rather than refusing — a memory worth saving shouldn't be lost.
            scope = "global"
            memories_dir = _memories_dir()
        else:
            memories_dir = pdir
    private_dir(memories_dir)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    slug = _slugify(title)
    base = os.path.join(memories_dir, f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.md")
    content = f"# {title}\n\n{body}\n\n---\n_scope: {scope}  _saved: {stamp}\n"

    # Claim a name. The loop is only the suffix walk; publication itself is
    # atomic and refuses to clobber (see _publish_memory), so a concurrent
    # writer that picked the same name cannot erase our note — moving the
    # uniqueness check into the claim is what fixed the merge-workers TOCTOU.
    path, n = base, 1
    while True:
        if gate and not _confirm_memory(
                f"write memory {path} ({len(content)} bytes)",
                preview=_memory_confirm_preview(title, body)):
            return None
        if _publish_memory(path, content):
            return path, scope
        n += 1
        path = f"{base[:-3]}-{n}.md"


def memory_write(
    title: str, body: str, scope: str = "project", **_: Any
) -> str:
    """Persist a memory note the model can recall in future sessions.

    scope="project" writes to <project_root>/.nx/memories/ (falls back to
    global when there is no project root); scope="global" writes to
    ~/.nx/memories/ (cross-project). The note is a markdown file named
    <timestamp>_<slug>.md starting with a `# Title` line — written 0600 via
    tmp+rename, and never clobbering an existing note.

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

    try:
        written = _write_memory_file(title, body, scope, gate=True)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    if written is None:
        return "DENIED by user"
    path, scope = written
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


def _memory_scan(
    query: str, scope: str, limit: int
) -> tuple[list[_MemoryHit], int, int]:
    """Scan the memory dirs for `query`. Every term must appear; ranking is
    title-hits above body-hits, then newest first. Returns
    (hits, overflow, total): overflow = how many ranked hits the cap removed
    (0 when unlimited or nothing cut), total = matches before the cap.
    limit=0 → unlimited. Empty query → newest-first index of every note.

    Per dir, `nx._memindex.search_dir` (an mtime-reconciled FTS5 cache over
    the notes) supplies candidate rows without reading any file; the rows
    still pass the same lowercase-substring predicate and `_rank_hit`
    scoring, so results are identical to the plain scan up to whitespace
    folding between query characters (trigram matching is character-adjacent,
    not whitespace-exact) — the accelerator can only widen recall slightly.
    When it is off, unsupported, or returns no rows for a dir, the plain
    `_iter_notes` scan runs for that dir.
    """
    terms = query.lower().split()
    hits: list[_MemoryHit] = []
    for d, label in _iter_memory_dirs(scope):
        fts_rows = None
        if terms and all(len(t) >= 3 for t in terms):
            try:
                from ._memindex import search_dir
                fts_rows = search_dir(d, query)
            except Exception:
                fts_rows = None
        if fts_rows is not None:
            for name, mtime, title, body in sorted(
                    fts_rows, key=lambda r: r[0], reverse=True):
                low = body.lower()
                if not all(t in low for t in terms):
                    continue
                # Same preview builder as the plain scan (see _memindex: the
                # old FTS `snippet()` was token-bounded and truncated to ~12
                # chars, so the accelerator used to degrade every hit).
                hits.append(_MemoryHit(
                    mtime, name, title, _memory_snippet(body, query), label,
                    _rank_hit(body, query)))
            continue
        for name, _path, mtime, title, content in _iter_notes(d):
            if terms:
                low = content.lower()
                if not all(t in low for t in terms):
                    continue
                score = _rank_hit(content, query)
            else:
                score = 0.0
            hits.append(_MemoryHit(
                mtime, name, title, _memory_snippet(content, query), label, score))
    hits.sort(key=lambda h: (-h.score, -h.mtime))
    total = len(hits)
    if limit and total > limit:
        return hits[:limit], total - limit, total
    return hits, 0, total


def _scan_memory_entries(scope: str = "project") -> list[_MemoryHit]:
    """Every saved memory as a hit — no query, no snippet, no limit. The
    resolver behind memory_delete and `/memory delete`."""
    hits: list[_MemoryHit] = []
    for d, label in _iter_memory_dirs(scope):
        for name, _path, mtime, title, _content in _iter_notes(d, titles_only=True):
            hits.append(_MemoryHit(mtime, name, title, "", label, 0.0))
    hits.sort(key=lambda h: -h.mtime)  # newest first, same order memory_search uses
    return hits


def memory_search(query: str, scope: str = "project", **_: Any) -> str:
    """Keyword-search saved memories (global + project): title + snippet per
    match, best match first — title hits above body hits, newer above older.
    Read-only; no approval. Use this to recall prior learnings before
    re-deriving them.

    scope="project" searches both project and global memories; scope="global"
    searches only global. Every query term must appear. An empty query returns
    the newest titles as a quick index (so the model can discover what
    exists). Results are capped at NX_MEMORY_SEARCH_LIMIT (default 8); the
    response says how many were cut.
    """
    query = (query or "").strip()
    hits, overflow, total = _memory_scan(query, scope, _memory_search_limit())
    if not hits:
        return f"[no memories matching {query!r}]" if query else "[no memories saved yet]"
    if query:
        head = (f"memories matching {query!r} ({len(hits)} of {total}):" if overflow
                else f"memories matching {query!r} ({total}):")
    else:
        head = f"memories ({len(hits)} of {total}):" if overflow else f"memories ({total}):"
    out = [head]
    for hit in hits:
        out.append(f"[{hit.label}] {hit.title} ({hit.name})")
        out.append(f"  {hit.snippet}")
    if overflow:
        out.append(f"  … {overflow} more not shown — raise NX_MEMORY_SEARCH_LIMIT to see them")
    dirs: list[str] = []
    for d, _label in _iter_memory_dirs(scope):
        if d in dirs:
            continue
        if any(os.path.isfile(os.path.join(d, h.name)) for h in hits):
            dirs.append(d)
    if dirs:
        out.append("read a note with read_file <absolute path> — the (name) above is a bare filename; the files live in: "
                   + "; ".join(dirs))
    return "\n".join(out)


def memory_delete(target: str, scope: str = "project", **_: Any) -> str:
    """Delete one saved memory note by exact title, filename, or unique title
    prefix. An ambiguous or unknown target returns candidate titles instead of
    deleting. Goes through the approval gate like memory_write; on success it
    drops any cached memory_search results (see run_tool)."""
    target = (target or "").strip()
    if not target:
        return "ERROR: memory_delete requires a non-empty `target`"
    if scope not in ("project", "global"):
        scope = "project"
    entries = _scan_memory_entries(scope)
    if not entries:
        return "[no memories saved yet]"

    low = target.lower()
    exact = [h for h in entries
             if h.name.lower() == low or h.name[:-3].lower() == low
             or h.title.lower() == low]
    if len(exact) == 1:
        chosen = exact[0]
    else:
        cands = exact if exact else [
            h for h in entries if h.title.lower().startswith(low)]
        if len(cands) > 1:
            listing = "\n".join(f"  - {h.title} ({h.name})" for h in cands[:10])
            return (f"AMBIGUOUS: {target!r} matches {len(cands)} memories — "
                    f"delete by exact title or filename:\n{listing}")
        if not cands:
            recent = "\n".join(f"  - {h.title} ({h.name})" for h in entries[:5])
            return (f"no memory matching {target!r} — recent memories:\n{recent} "
                    f"(memory_search with an empty query lists all)")
        chosen = cands[0]

    path = None
    for d, _label in _iter_memory_dirs(scope):
        p = os.path.join(d, chosen.name)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return f"ERROR: memory file for {chosen.title!r} vanished"

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if not _confirm_memory(
            f"delete memory {path} ({size} bytes)",
            preview=_memory_confirm_preview(chosen.title, _read_note_preview(path))):
        return "DENIED by user"
    try:
        os.unlink(path)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    return f"deleted {chosen.label} memory: {chosen.title}\nfile: {path}"


def _add_memory_note_if_absent(messages: list[dict[str, Any]]) -> None:
    """Ensure the transcript carries the session-start memory note (see
    _load_memory_note). Fresh sessions get it from _initial_messages(); a
    resumed session replays a transcript that may predate the feature — or
    predate the memories it would list. A thin alias of _refresh_memory_note
    (appends at the TAIL, never inserts mid-history — PERF.md PF-12), kept
    under this name for its marker-scan contract (mirrored by
    _subagent._add_delegation_note_if_absent)."""
    _refresh_memory_note(messages)


MEMORY_NOTE_MARKER = "[Saved memories"

# Appended in place of the list when the store has gone empty, so the model
# stops reciting titles that no longer exist. Carries the marker so the next
# refresh recognises it as the current note.
MEMORY_NOTE_CLEARED = (
    "[Saved memories: the store is now empty — the earlier [Saved memories] "
    "list is stale; don't act on those titles.]"
)


def _find_memory_note_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the FIRST memory-awareness note, or None."""
    for i, m in enumerate(messages):
        if (m.get("role") == "user" and isinstance(m.get("content"), str)
                and MEMORY_NOTE_MARKER in m["content"]):
            return i
    return None


def _last_memory_note_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the LAST memory-awareness note, or None. Refresh compares
    against this one: it is the freshest statement of the store in the
    transcript."""
    found = None
    for i, m in enumerate(messages):
        if (m.get("role") == "user" and isinstance(m.get("content"), str)
                and MEMORY_NOTE_MARKER in m["content"]):
            found = i
    return found


def _refresh_memory_note(messages: list[dict[str, Any]]) -> bool:
    """Bring the transcript's view of the memory store up to date by APPENDING
    a fresh note at the TAIL when it drifted. Returns True when changed.

    APPEND-ONLY ON PURPOSE (PERF.md PF-12, invariant I-3). This used to rewrite
    the note at its original index — almost always position 1 — which busts the
    server prompt-prefix cache for the entire remaining prompt on every /memory
    optimize and every resume over a changed store. Appending leaves every
    earlier byte untouched, so the prefix survives; the tail note is the
    authoritative one (a model reading top-to-bottom sees the latest list
    last, superseding any stale note left mid-transcript). No-op when the last
    note already matches the store, so a resume over an untouched store stays
    prompt-cache-clean. When the store went empty a note is still appended
    saying so, so the model stops reciting archived titles."""
    at = _last_memory_note_index(messages)
    note = _load_memory_note(getattr(st, "PROJECT_ROOT", None))
    if at is None and note is None:
        return False
    fresh = note["content"] if note is not None else MEMORY_NOTE_CLEARED
    if at is not None and messages[at].get("content") == fresh:
        return False
    messages.append({"role": "user", "content": fresh})
    return True


def _load_memory_note(project_root: str | os.PathLike[str] | None) -> dict[str, str] | None:
    """One-time user-role note appended to a fresh session's message list so
    the model knows memories exist and which are already saved. Titles only
    (not bodies) and capped — keeps the per-turn token floor (see
    test_initial_context.py) from growing with how much you've remembered.
    When a scope holds more than the cap, a pointer to memory_search is added
    instead of silently hiding the rest. Returns None when there are no
    memories in either scope.
    """
    roots = _memory_dirs_for(project_root)
    if not roots:
        return None
    titles: list[str] = []
    overflow = False
    for d in roots:
        label = "project" if d == _project_memories_dir(project_root) else "global"
        notes = _iter_notes(d, titles_only=True)   # titles only — no body reads
        overflow = overflow or len(notes) > 5
        for _name, _path, _mtime, title, _content in notes[:5]:
            titles.append(f"[{label}] {title}")
    if not titles:
        return None
    lines = list(titles)
    if overflow:
        lines.append("… more saved — memory_search with an empty query lists the rest")
    body = "\n".join(lines)
    return {
        "role": "user",
        "content": (
            f"[Saved memories — use memory_search to read any, memory_write to add:]\n{body}"
        ),
    }
