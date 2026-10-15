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
import re
import time
from typing import Any

from ._ansi import C, DIM, RESET
from ._fullscreen import View
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from ._sessions_io import _flatten_message_text
from . import _state as st

# --- re-exports from the split-out layers (single import surface preserved) ---
from ._memory_store import (  # noqa: F401 (re-exports)
    MEMORY_KINDS, _MEMORY_FOOTER_RE, _annotate_drift, _clear_supersession,
    _confirm_memory, _drift_challenge, _drift_demote, _drift_freshness,
    _drift_line, _drift_root, _first_title, _iter_memory_dirs, _iter_notes,
    _memories_dir, _memory_confirm_preview, _memory_dirs_for, _note_derived,
    _note_kind, _note_summary_of, _note_superseded_by, _note_supersedes,
    _parse_drift, _redact_secrets, _redaction_note, _note_tail,
    _SUPERSESSION_MAX_HOPS,
    _project_memories_dir, _publish_memory, _read_memory_file,
    _read_memory_title, _read_note_preview, _slugify, _stamp_note_now,
    _write_memory_file, resolve_note_target, repoint_superseded,
    stamp_superseded,
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


def _memory_usage_inline() -> None:
    """The pre-view bare-`/memory` usage block (the fullscreen fallback)."""
    print(f"{DIM}  usage:{RESET}")
    print(f"{DIM}    /memory save [focus...]     save a memory (optional focus guidance){RESET}")
    print(f"{DIM}    /memory remember <query>    search memories{RESET}")
    print(f"{DIM}    /memory delete <target>     remove one memory (title/file/prefix){RESET}")
    print(f"{DIM}    /memory unsupersede <target>  bring a note a supersession retired back{RESET}")
    print(f"{DIM}    /memory list                list all memories{RESET}")
    print(f"{DIM}    /memory archived [target]   list archived notes, or restore one{RESET}")
    print(f"{DIM}    /memory optimize [dry] [focus]  merge duplicates, retire stale notes "
          f"(dry prints the plan only){RESET}")
    print(f"{DIM}    /memory reflect [focus]     cross-project pass: promote recurring "
          f"preferences, surface shared patterns{RESET}")

# --- /memory slash commands ----------------------------------------------------

def _cmd_memory(messages: list[dict[str, Any]], user_input: str) -> None:
    """Handle /memory save|remember|delete|list  — manual memories.

    /memory save [focus...]      — distill session into a memory (optional focus
                                   guidance for the model: e.g. "focus on config")
    /memory remember <query>     — keyword search through saved memories
    /memory delete <target>      — remove one memory (title/file/prefix)
    /memory unsupersede <target> — restore one a supersession retired
    /memory list                 — list all saved memories
    /memory archived [target]    — list notes /memory optimize archived, or
                                   restore one by title/file/prefix
    /memory optimize [dry] [focus]  — librarian pass over the store: fold
        duplicates together and archive notes a matured project outgrew
    """
    parts = user_input.split(None, 2)
    sub = parts[1].lower() if len(parts) >= 2 else ""

    if sub in ("", "list"):
        # Bare /memory and /memory list: the fullscreen MemoryView, with the
        # pre-view inline list (or usage block) as the off-tty fallback — the
        # same gate-then-open_view shape _cmd_ctx uses for /context.
        from . import _fullscreen
        if _fullscreen.wants_fullscreen():
            fallback = _cmd_memory_list if sub == "list" else _memory_usage_inline
            _fullscreen.open_view(MemoryView(), fallback=fallback)
            return

    if sub == "remember":
        query = parts[2] if len(parts) >= 3 else ""
        _cmd_memory_remember(query)
        return

    if sub == "delete":
        target = parts[2] if len(parts) >= 3 else ""
        _cmd_memory_delete(target)
        return

    if sub in ("unsupersede", "unretire"):
        target = parts[2] if len(parts) >= 3 else ""
        _cmd_memory_unsupersede(target)
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

    # Bare /memory — show usage (unreachable while the view is wired above;
    # kept for callers importing the handler directly with an unexpected sub)
    _memory_usage_inline()


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

# Provenance round-trip for `/memory save`. The distiller is told to end a note
# whose content was RECYCLED from memory the transcript already showed (the
# [Saved memories] titles, a note body pasted in as tool output) with this
# marker; the marker is popped out of the body here and handed to
# _write_memory_file as `derived=` so it lands in the footer as
# `_derived: memory` — a re-distilled note is then distinguishable from a
# first-hand one, and `/memory optimize` won't treat the pair as independent
# sources. Kept short: this prompt only runs on an explicit `/memory save`.
_DERIVED_MARKER_LINE_RE = re.compile(r"(?m)^[ \t]*_derived:[ \t]*memory[ \t]*$")
_FENCE_LINE_RE = re.compile(r"(?m)^[ \t]*```[a-zA-Z0-9_-]*[ \t]*$")


def _extract_derived_marker(body: str) -> tuple[str, str | None]:
    """Pop the distiller's `_derived: memory` marker out of a note body.

    Returns (body_without_marker, "memory" | None). The marker is the model's
    own provenance statement, so it must never survive into the stored body: a
    note body is prose, and `_derived:` there would be re-parsed as a footer
    field by `_note_derived`. Tolerates the model fending it (a content-free
    ``` block left behind is dropped)."""
    if not _DERIVED_MARKER_LINE_RE.search(body):
        return body, None
    body = _DERIVED_MARKER_LINE_RE.sub("", body)
    lines = body.splitlines()
    kept: list[str] = []
    i = 0
    while i < len(lines):
        # An empty fence block left by a fenced marker (a fence line, nothing but
        # whitespace, then the closing fence) is noise, not content.
        if _FENCE_LINE_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _FENCE_LINE_RE.match(lines[j]):
                i = j + 1
                continue
        kept.append(lines[i])
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), "memory"


def _render_transcript_for_memory(messages: list[dict[str, Any]]) -> str:
    """Render session messages into a compact text transcript for the model.

    Skips the system prompt and nx's own injected satellite blocks (see
    nx._prefix.is_satellite_message), truncates long tool outputs, and shows
    tool calls. The whole transcript is also capped
    (_MEMORY_TRANSCRIPT_CHARS): a long session's tail is dropped, and the cut
    is announced in-band so the distillation knows it saw a truncated session.
    """
    from ._prefix import is_satellite_message
    out = []
    total = 0
    truncated = False
    for m in messages:
        if m.get("role") == "system":
            continue
        # nx's own injected blocks (the [Saved memories] index note, the
        # orientation/delegation notes, [Runtime note: …] nudges, …) are NOT
        # session material: they are the harness's own prior memory output.
        # Feeding them back to the distiller is the self-ingestion loop this
        # filter closes — a re-distilled note would otherwise restate the note
        # it was distilled from, byte-indistinguishably from a first-hand one.
        if is_satellite_message(m):
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
        "Start with a `# Title` line that captures the topic.\n"
        "The transcript may show memory titles or note bodies nx already saved "
        "(a [Saved memories] list, a memory_search result). Those are PRIOR ART, "
        "not fresh findings: never restate them as your own learnings, and if a "
        "note you write does recycle them, end it with a line `_derived: memory` "
        "(the harness moves that marker into the note's footer).\n\n"
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
    # Provenance: the distiller flags a note recycled from memory the transcript
    # already showed (see _extract_derived_marker / the prompt's PRIOR ART line).
    body, derived = _extract_derived_marker(body)

    redactions: list = []
    try:
        written = _write_memory_file(title, body, "project", derived=derived,
                                     redactions=redactions)
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't write memory: {e}{RESET}")
        return
    if written is None:  # gate denied — not possible with gate=False; kept safe
        return
    path, _scope = written
    # The distiller reads 48K chars of tool output, so this is the path most
    # likely to have picked up a credential — say so rather than let the user
    # believe the note holds what the model wrote.
    note = _redaction_note(redactions)
    if note:
        print(f"{C.WARNING}  {note.strip()}{RESET}")
    saved = _read_memory_file(path, os.path.basename(path))
    print(f"{DIM}  ✓ saved memory: {_first_title(saved[1], title) if saved else title}{RESET}")
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
    project-only repo used to report "no memories saved yet" here.

    Superseded notes ARE shown (marked, after the live ones): the search/injection
    views hide them, so this is the surface that keeps a retired note findable.
    A note you can only find by hand is silent data-loss if supersession ever
    misfires, so the reversible path is advertised here too."""
    retired: list = []
    files: list = []
    for d, label in _iter_memory_dirs("project"):
        live = _iter_notes(d, titles_only=True)
        names = {r[0] for r in live}
        files.extend((r[2], r[0], r[3], label, r[4]) for r in live)
        # The retired set is the DIFFERENCE between a filtered and an unfiltered
        # scan, so the filter itself stays the single authority (a note counts as
        # retired exactly when the live scan refuses to serve it).
        for name, path, mtime, title, head in _iter_notes(
                d, titles_only=True, include_superseded=True):
            if name not in names:
                retired.append((mtime, name, title, label, path))

    live_n = len(files)
    if not files and not retired:
        print(f"{DIM}  no memories saved yet{RESET}")
        return

    files.sort(key=lambda x: x[0], reverse=True)
    print(f"{DIM}  memories ({live_n}):{RESET}")
    for i, (mtime, _name, title, label, _head) in enumerate(files, 1):
        date = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        print(f"  {C.SUCCESS}{i:>2}.{RESET} [{label}] {title}  {DIM}({date}){RESET}")
    if retired:
        retired.sort(key=lambda x: x[0], reverse=True)
        print(f"{DIM}  superseded ({len(retired)}) — hidden from memory_search and "
              f"the session-start note; `/memory unsupersede <title|file|prefix>` "
              f"brings one back:{RESET}")
        for mtime, _name, title, label, path in retired:
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            # The pointer lives in the footer, past the head window on any normal
            # note, so read it from the bounded tail (only for the few retired
            # rows this prints, never for the live ones).
            by = _note_superseded_by(_note_tail(path)) or "?"
            print(f"  {C.WARNING}×{RESET} [{label}] {title}  "
                  f"{DIM}({date} · superseded by {by}){RESET}")


def _retire_note(old_path: str, old_name: str, new_path: str) -> str:
    """Stamp the note the just-written note retires, then RE-POINT anything that
    already pointed at it. Returns '' on success, else a one-line note for the
    tool result.

    The re-point is what keeps the graph FLAT (no A→B→C chains): if an earlier
    note was retired into `old_name`, and `old_name` now retires into the new
    note, the earlier one is restamped to point straight at the new note. A chain
    would leave a middle link nothing resolves forward from — the live note at
    its tip would be found by walking forward from nothing.

    Supersession must never fail the write that caused it: the new note is
    already published and durable here, so a note we could not stamp just stays
    live (visible, harmless, pair simply not recorded)."""
    if not stamp_superseded(old_path, os.path.basename(new_path)):
        return (f"\nWARNING: wrote the new note but could not mark {old_name} as "
                f"superseded (unreadable, unwritable, or it has no footer); it "
                f"stays live.")
    repoint_superseded(old_name, os.path.basename(new_path))
    return ""


def _resolve_supersedes(target: str, scope: str) -> tuple:
    """Resolve a `supersedes` target to (entry, refusal). Exactly one of the two
    is non-empty: `entry` is the (name, path, mtime, title, label) note to
    retire, or `refusal` explains why NOTHING should be written.

    Ambiguity and unknown targets fail safe — never supersede-by-guess. A target
    that is ALREADY retired is REDIRECTED to the note that retired it (walk
    capped at _SUPERSESSION_MAX_HOPS, a hand-edited cycle bails out) rather than
    chained: A→B→C would leave a dangling middle link that nothing resolves
    forward from, and redirecting keeps one flat retired→live pair per note."""
    hops = 0
    cur = (target or "").strip()
    while True:
        entry, cands, all_entries = resolve_note_target(cur, scope)
        if entry is None:
            if not all_entries:
                return None, (f"ERROR: `supersedes` {cur!r} — no saved memories "
                              f"to supersede; nothing was written.")
            listing = "\n".join(f"  - {e[3]} ({e[0]})" for e in cands[:10])
            verb = ("matches" if len(cands) > 1 else "matches no saved memory — "
                    "recent")
            return None, (f"ERROR: `supersedes` {cur!r} {verb} {len(cands)} "
                          f"note(s); nothing was written. Name an exact title or "
                          f"filename:\n{listing}")
        by = ""
        try:
            read = _read_memory_file(entry[1], entry[0])
            by = _note_superseded_by(read[1]) if read is not None else ""
        except Exception:
            by = ""
        if not by:
            return entry, ""
        hops += 1
        if hops > _SUPERSESSION_MAX_HOPS or by == cur:
            return None, (f"ERROR: `supersedes` {target!r} is caught in a "
                          f"supersession chain ({cur} → {by}); nothing was "
                          f"written.")
        cur = by


def _cmd_memory_unsupersede(target: str) -> None:
    """`/memory unsupersede <title|file|prefix>` — undo a supersession: clear the
    target's `_superseded_by:`/`_valid_until:` footer fields so the note is live
    again. The file never left disk, so this is the whole restore."""
    if not target:
        print(f"{C.WARNING}  usage: /memory unsupersede <title | filename | "
              f"unique prefix>{RESET}")
        return
    entry, cands, _all = resolve_note_target(target, "project")
    if entry is None:
        listing = "\n".join(f"  - {e[3]} ({e[0]})" for e in cands[:10])
        print(f"{C.WARNING}  {target!r} is ambiguous or unknown — nothing "
              f"changed:{RESET}\n{listing}")
        return
    read = _read_memory_file(entry[1], entry[0])
    if read is None:
        print(f"{C.ERROR}  ✗ couldn't read {entry[1]}{RESET}")
        return
    if not _note_superseded_by(read[1]):
        print(f"{DIM}  {entry[3]} is not superseded — nothing to restore{RESET}")
        return
    if not _confirm_memory(f"unsupersede memory {entry[1]}"):
        return
    try:
        from ._private import atomic_write_private
        atomic_write_private(entry[1], _clear_supersession(read[1]))
    except OSError as e:
        print(f"{C.ERROR}  ✗ couldn't rewrite {entry[0]}: {e}{RESET}")
        return
    print(f"  {C.SUCCESS}unsuperseded{RESET} {entry[3]}  "
          f"{DIM}(live again: search + the session-start note see it){RESET}")


def _superseded_drift_challenge(title: str, scope: str) -> str:
    """The write-time challenge for a note this `memory_write` supersedes: when
    a note with the SAME scope+title already exists and is stale per the drift
    metric, return the challenge text, else ''.

    Called BEFORE the write (the fresh note would otherwise match its own
    title) and appended to the result AFTER it — the challenge is purely
    informational and must never block or veto the write. Kept cheap: the scan
    is titles-only (a bounded head read per note) and a body is read only on a
    title match — the stamp lives in the footer, so the head cannot answer it.
    """
    want = (title or "").strip().lower()
    if not want:
        return ""
    try:
        for d, _label in _iter_memory_dirs(scope):
            for _name, path, _mtime, t, _head in _iter_notes(d, titles_only=True):
                if t.strip().lower() != want:
                    continue
                read = _read_memory_file(path, _name)
                if read is None:
                    continue
                chal = _drift_challenge(_parse_drift(read[1]))
                if chal:
                    return chal
    except Exception:
        return ""  # a drift hiccup must not affect the write's outcome
    return ""


def memory_write(
    title: str, body: str, scope: str = "project", kind: str = "",
    supersedes: str = "", **_: Any
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

    supersedes: name (title / filename / unique title prefix) of the note this
    one replaces. Prefer it over memory_delete: the old note STAYS on disk, is
    stamped `_superseded_by:`, and stops appearing in memory_search, the
    session-start note and `/memory optimize` — reversibly, via
    `/memory unsupersede`. An ambiguous or unknown target writes NOTHING and
    returns the candidates.

    Anything credential-shaped in the title or body is replaced with
    `[REDACTED:<label>]` on the way to disk (notes are re-injected into every
    future session, so a token must not be archived); the result says how many.

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
    # Resolve the retirement FIRST: an ambiguous or unknown `supersedes` must
    # write nothing at all (never supersede-by-guess), so this precedes the gate.
    target, refusal = (None, "")
    if (supersedes or "").strip():
        target, refusal = _resolve_supersedes(supersedes, scope)
        if refusal:
            return refusal
    # Snapshot the superseded note's staleness BEFORE writing (afterward the
    # new note carries the same title and looks fresh by construction).
    challenge = _superseded_drift_challenge(title, scope)

    redactions: list = []
    try:
        written = _write_memory_file(
            title, body, scope, gate=True, kind=kind,
            supersedes=(target[0] if target else None), redactions=redactions)
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
    # The new note is durable now; retiring the old one cannot fail it.
    retired = _retire_note(target[1], target[0], path) if target else ""
    # Report the title as WRITTEN: a secret in the title is redacted on disk, and
    # echoing the caller's raw title back would re-state the credential we just
    # refused to store.
    read = _read_memory_file(path, os.path.basename(path))
    shown = _first_title(read[1], title) if read is not None else title
    return (f"saved {scope} memory: {shown}\nfile: {rel}" + retired
            + _redaction_note(redactions) + challenge)


# --- the /memory fullscreen view (0.151.0) -------------------------------------
# The same recipe as /diff's ReviewView and /context's ContextView: the view
# lives in the module that owns the data, renders exclusively through
# `_render.py`, and its ONLY mutation is the existing delete path handed back
# to the fullscreen shell as a deferred `pending_action` (run AFTER the shell
# leaves the alternate screen, so the gate's y/n prompt never paints into it).

def _memory_row(note: dict[str, Any], selected: bool, index: int,
                width: int) -> str:
    """One note row: selection marker, the same columns the inline list prints
    (index, scope label, title, date)."""
    from ._ansi import clip_display, display_width
    from ._render import RESET, hue, role
    date = time.strftime("%Y-%m-%d %H:%M", time.localtime(note["mtime"]))
    marker = "›" if selected else " "
    scope = f"[{note['label']}]"
    # Colors come from the render layer's `hue()` (B3: no raw `C.*` reads
    # outside _ansi/_rich/_render); same hues, resolved at call time.
    scope_color = hue("green") if note["label"] == "project" else hue("magenta")
    head = f"{index:>3}. {scope_color}{scope}{RESET} "
    tail = f"  {hue('yellow')}{date}{RESET})"
    # Clip the un-styled title first, then splice the styled scope/date in so
    # clip_display never has to reason about our escapes.
    visible = display_width(head) + display_width(tail)
    title = clip_display(note["title"], max(0, width - 3 - visible)).rstrip()
    return (f"{role('ACCENT')}{marker}{RESET} "
            f"{head}{role('MUTED')}{title}{tail}{RESET}")

class MemoryView(View):
    """The `/memory` view: note list → note body, `d` deletes (with confirm).

    Two modes like ReviewView: "list" over the newest-first note list the
    inline `/memory list` prints (title, scope label, date), `enter`/`tab`
    opens the body wrapped and scrollable; `d` resolves the selected note by
    filename (unambiguous — no prefix collision possible) and queues the
    EXISTING `memory_delete` machinery as the deferred action.
    """

    surface = "memory"
    title = "memory"

    def __init__(self) -> None:
        super().__init__()
        self.sel = 0
        self.offset = 0           # scroll offset inside a note body
        self.mode = "list"        # "list" | "note"
        self.notes: list[dict[str, Any]] = []
        self.body_lines: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-read the note list — newest first, both scopes, the exact rows
        `_cmd_memory_list` prints."""
        notes = []
        for d, label in _iter_memory_dirs("project"):
            for name, path, mtime, title, _content in _iter_notes(d, titles_only=True):
                notes.append({"mtime": mtime, "name": name, "path": path,
                              "title": title, "label": label})
        notes.sort(key=lambda n: n["mtime"], reverse=True)
        # global notes pinned above project ones, newest-first within each group
        notes.sort(key=lambda n: n["label"] != "global", reverse=False)
        self.notes = notes
        self.sel = min(self.sel, max(0, len(notes) - 1))
        self.mode = "list"
        self.body_lines = []
        self.offset = 0

    def _open_note(self) -> None:
        import textwrap
        if not self.notes:
            return
        note = self.notes[self.sel]
        read = _read_memory_file(note["path"], note["name"])
        content = read[1] if read is not None else (note["title"] + "\n")
        self.body_lines = []
        for para in content.split("\n"):
            self.body_lines.extend(textwrap.wrap(para, width=200,
                                                 replace_whitespace=False,
                                                 drop_whitespace=False) or [""])
        self.offset = 0
        self.mode = "note"

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        self.sel = min(self.sel, max(0, len(self.notes) - 1))
        lines = [f"{role('ACCENT')}memory{RESET}{role('MUTED')} — "
                 f"{hint_line('memory')}{RESET}"]
        budget = max(1, height - len(lines) - 2)   # header line + box borders
        body: list[str] = []
        if self.mode == "note":
            note = self.notes[self.sel]
            wrapped = [ln[:width] for ln in self.body_lines]
            n = len(wrapped)
            self.offset = max(0, min(self.offset, max(0, n - budget)))
            shown = wrapped[self.offset:self.offset + budget]
            body.extend(shown)
            lines.extend(panel(f"{note['title']}  ({note['label']})",
                               [ln[:width] for ln in body],
                               footer=f"line {self.offset + 1}"
                                      f"-{self.offset + len(shown)}/{n}"))
        else:
            n = len(self.notes)
            if not n:
                body.append(f"{role('MUTED')}  no memories saved yet — "
                            f"/memory list prints the same list inline{RESET}")
            else:
                shown = budget if n <= budget else max(1, budget - 1)
                start = max(0, min(self.sel - shown + 1, n - shown))
                stop = min(n, start + shown)
                for i in range(start, stop):
                    body.append(_memory_row(self.notes[i], i == self.sel,
                                            i + 1, width))
                hidden = n - (stop - start)
                if hidden > 0:
                    body.append(f"{role('MUTED')}  … {hidden} more note"
                                f"{'s' if hidden != 1 else ''} (j/k scroll){RESET}")
            lines.extend(panel(f"memories · {n}",
                               [ln[:width] for ln in body],
                               footer=f"{self.sel + 1}/{n}"))
        if self.notice:
            lines.append(f"{role('WARNING')}{self.notice[:width]}{RESET}")
        return lines[:height]

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        if key == "j/k":
            # The registry advertises the pair as ONE hint label ("j/k move");
            # the two real keys are handled below (the ReviewView convention).
            return "repaint"
        last = max(0, len(self.notes) - 1)
        if self.mode == "note":
            if key in ("j", "down"):
                self.offset += 1
                return "repaint"
            if key in ("k", "up"):
                self.offset = max(0, self.offset - 1)
                return "repaint"
            if key == "pagedown":
                self.offset += 10
                return "repaint"
            if key == "pageup":
                self.offset = max(0, self.offset - 10)
                return "repaint"
            if key in ("enter", "tab", "esc", "left"):
                self.mode = "list"
                return "repaint"
            if key == "d":
                return self._delete(shell)
            return "repaint"
        if key in ("j", "down"):
            self.sel = min(last, self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            self.sel = max(0, self.sel - 1)
            return "repaint"
        if key == "pagedown":
            self.sel = min(last, self.sel + 10)
            return "repaint"
        if key == "pageup":
            self.sel = max(0, self.sel - 10)
            return "repaint"
        if key == "enter":
            self._open_note()
            return "repaint"
        if key == "d":
            return self._delete(shell)
        if key == "r":
            self.refresh()
            return "repaint"
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"

    def _delete(self, shell: Any) -> str:
        """Queue the existing `memory_delete` for the selected note's FILENAME
        (an exact-name match can never be ambiguous) — deferred: the shell
        runs `pending_action` only after leaving the alternate screen."""
        if not self.notes:
            self.notice = "no notes to delete"
            return "repaint"
        note = self.notes[min(self.sel, len(self.notes) - 1)]
        if shell.ask(f"delete note '{note['title']}'? y/n ›") is None:
            self.notice = "delete cancelled"
            return "repaint"
        target, view = note["name"], self

        def _act() -> None:
            result = memory_delete(target, scope="project", confirm=False)
            view.notice = result.split("\n")[0]
            view.refresh()

        self.pending_action = _act
        return "run_stay"
