"""The session-start memory-awareness note: injection, refresh, and the
marker contract.

Split out of `_memory_cmd.py` (which keeps the `/memory` slash-command UI).
`_load_memory_note` builds the one-time user-role note that lets the model
know memories exist; `_refresh_memory_note` APPENDS a fresh note at the TAIL
when the store drifted (never rewrites mid-history — PERF.md PF-12), and
`_add_memory_note_if_absent` is the thin alias with the marker-scan contract
mirrored by `_subagent._add_delegation_note_if_absent`."""
from __future__ import annotations

import os
from typing import Any

from . import _state as st
from ._memory_store import _iter_notes, _memory_dirs_for, _note_kind
from . import _project_memories_dir


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
    # 0.68.0: preference-kind notes are injected EVERY session regardless of
    # recency (they are the user's durable style/tooling choices) and get a
    # deeper cap than ordinary titles; untagged/fact/episodic titles ride the
    # plain newest-first list. Preferences in either scope count.
    titles: list[str] = []
    prefs: list[tuple[float, str]] = []
    overflow = False
    plain_cap = 5
    pref_cap = 4
    for d in roots:
        label = "project" if d == _project_memories_dir(project_root) else "global"
        notes = _iter_notes(d, titles_only=True)   # titles only — no body reads
        overflow = overflow or len(notes) > plain_cap
        for _name, _path, _mtime, title, head in notes:
            if _note_kind(head) == "preference":
                prefs.append((_mtime, f"[{label}] {title}"))
        for _name, _path, _mtime, title, _content in notes[:plain_cap]:
            titles.append(f"[{label}] {title}")
    prefs.sort(key=lambda p: -p[0])  # newest preference first
    shown_prefs = [text for _mtime, text in prefs[:pref_cap]]
    if len(prefs) > pref_cap:
        overflow = True
    if not titles and not shown_prefs:
        return None
    lines = []
    if shown_prefs:
        lines.append("User preferences (always apply):")
        lines.extend(shown_prefs)
    if titles:
        if shown_prefs:
            lines.append("")
        lines.extend(titles)
    if overflow:
        lines.append("… more saved — memory_search with an empty query lists the rest")
    body = "\n".join(lines)
    return {
        "role": "user",
        "content": (
            f"[Saved memories — use memory_search to read any, memory_write to add:]\n{body}"
        ),
    }
