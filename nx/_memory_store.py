"""Memory store: dirs, the one scanner, the one writer, and the note-format
conventions (title H1, slug, footer, kinds).

Split out of `_memory_cmd.py` (which keeps the `/memory` slash-command UI and
`memory_write`); this module holds the *storage* layer every consumer —
`_memory_optimize`, `_memory_reflect`, `_memindex`, the session-start note —
builds on. Two storage scopes:
  - global:  ~/.nx/memories/            (cross-project, via _memories_dir)
  - project: <project_root>/.nx/memories/ (via _project_memories_dir)

Every write goes through `_write_memory_file`: one filename convention
(`<timestamp>_<slug>.md`, `-2`/`-3`... on a same-second same-title collision),
one footer convention (`_scope:`/`_saved:`/`_kind:`), atomic tmp+rename,
private 0600 permissions (memories are distilled from sessions and may contain
paths/commands/credentials; same posture as session files). Publication is
claim-based (`atomic_write_private_noreplace`): the filename is not picked by
an exists-then-write check that two concurrent writers (the memory-merge
workers) can both pass — the first to publish the bytes wins and the loser
takes the next suffix.

Every read goes through `_iter_notes` — the one scanner over a memories dir —
so title extraction, mtime handling and unreadable-file tolerance live in
exactly one loop. All writes go through nx/_private.py (0600/0700,
umask-proof)."""
from __future__ import annotations

import codecs
import os
import re
import time

from ._private import atomic_write_private_noreplace, private_dir
from ._sessions_fs import _memories_dir, _project_memories_dir  # noqa: F401 (re-exports)
from . import _state as st


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
        # A project root of the home dir (bare-cwd fallback in
        # _detect_project_root) makes <root>/.nx/memories the global dir
        # itself; scanning it twice would list every global note as both
        # [global] and [project].
        if os.path.realpath(p) != os.path.realpath(_memories_dir()):
            pairs.append((p, "project"))
    return pairs


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


_MEMORY_FOOTER_RE = re.compile(r"\n---\n_scope:.*$", re.DOTALL)

# Note kinds (0.68.0): a lightweight tag in the footer so retrieval and
# session-start injection can treat classes of knowledge differently (a
# preference is always relevant; an episodic project fact is not). Freeform
# notes stay first-class: an unrecognised or absent kind is stored as-is and
# behaves exactly like today. The recognized set is deliberately small.
MEMORY_KINDS = ("preference", "fact", "gotcha", "episodic")


def _note_kind(content: str) -> str:
    """The `_kind:` value of a note's footer, or '' when untagged.
    Tolerates footers written before kinds existed and hand-edited notes."""
    m = re.search(r"(?:^|\s)_kind:\s*(\S+)", content)
    return m.group(1) if m else ""


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


def _write_memory_file(title: str, body: str, scope: str, gate: bool = False,
                       kind: str = "") -> tuple[str, str] | None:
    """The single memory writer: shared by the memory_write tool and
    `/memory save` so both produce identical files. Returns (path, scope), or
    None when gate=True and the user denied the confirmation.

    kind: optional note kind tag (see MEMORY_KINDS) stored in the footer as
    `_kind:`; an empty string keeps the footer kind-free (old format).

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
    footer = f"_scope: {scope}  _saved: {stamp}"
    if kind:
        footer += f"  _kind: {kind}"
    content = f"# {title}\n\n{body}\n\n---\n{footer}\n"

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


def _confirm_memory(action: str, preview: str | None = None) -> bool:
    """`_confirm` is monkeypatched by tests via the top-level module, so call
    it through `import nx as _m` — same pattern as _edit_tools."""
    import nx as _m
    return _m._confirm(action, preview=preview)


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
