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
one footer convention (`_scope:`/`_saved:`/`_kind:`, plus the drift stamp
`_drift:` documented at the bottom of this module), atomic tmp+rename,
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
import subprocess
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

    One side effect (drift stamping): when the scan is running over *this
    project's* memory dir inside a git repo, a note that carries no `_drift:`
    field at all is stamped with the current HEAD and persisted — the
    retroactive backfill for notes written before stamping existed (they were
    written against some recent state, so HEAD is the honest guess). It happens
    here because this is the one scanner: every consumer backfills on its first
    read, once per note (see the drift section at the bottom of this module).
    """
    out: list[tuple[str, str, float, str, str]] = []
    try:
        names = sorted(os.listdir(d), reverse=True)
    except OSError:
        return out
    backfill_root = "" if titles_only else _backfill_root(d)
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
            if backfill_root:
                stamped = _stamp_note_now(content, backfill_root)
                if stamped is not None:
                    content = _backfill_persist(path, content, stamped)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        out.append((name, path, mtime, title, content))
    return out


def _backfill_root(d: str) -> str:
    """The repo to retroactively stamp `d`'s notes against, or '' when this
    scan must not write: only the *current project's* memory dir gets backfill
    (global notes belong to whichever repo wrote them, and stamping them with
    this repo's HEAD would be a lie)."""
    root = _drift_root()
    if not root:
        return ""
    pdir = _project_memories_dir(getattr(st, "PROJECT_ROOT", None))
    if not pdir:
        return ""
    try:
        if os.path.realpath(d) != os.path.realpath(pdir):
            return ""
    except OSError:
        return ""
    return root


def _backfill_persist(path: str, content: str, stamped: tuple) -> str:
    """Write a backfilled note back to disk and return its new content.

    The new mtime is DELIBERATELY left in place: in this store mtime is the
    cache-invalidation token (`_memindex.reconcile` re-indexes on it), so
    restoring it would leave the FTS index serving the old UNSTAMPED body
    forever — the accelerator would keep reporting a backfilled note as fresh
    however far the repo later moved, while the plain scan reported it stale.
    The one-time cost is that existing notes show a refreshed mtime in
    `/memory list` (the footer's `_saved:` still carries the real save date)
    and their index rows rebuild once.

    A failure to persist is fine either way: the in-memory stamp is still the
    honest answer for this scan, and the next one retries."""
    new_content, _field = stamped
    try:
        from ._private import atomic_write_private
        atomic_write_private(path, new_content)
    except OSError:
        return content
    return new_content


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
    # Drift stamp: which commit this note is verified against, plus the repo
    # paths its body mentions ('' outside a git repo — no stamp, not a lie).
    drift = _drift_line(body, _drift_root())
    if drift:
        footer += f"  {drift}"
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


# --- drift stamping: which commit a note was verified against ------------------
#
# A memory note is a claim about the repo, and repos move. Without a stamp a
# note ages silently: nothing records which commit it was true at, and nothing
# tells the model (or the ranking) that the code it describes changed 40
# commits later. Everything here is mechanical — no site relies on the model
# cooperating:
#
#   write time   `_write_memory_file` appends a `_drift:` footer field: the
#                repo's HEAD sha, the repo-relative paths the body mentions,
#                the stamp time, and the repo the stamp belongs to.
#   read time    `_memory_search` tags every hit with `_annotate_drift()` and
#                ranks on `score / (1 + _drift_demote(...))`, so stale notes sink.
#   write again  `memory_write` appends `_drift_challenge()` when the note it
#                just superseded was stale. Informational; never blocks.
#   retroactive  `_iter_notes` stamps notes that predate the field with the
#                current HEAD (they were written against some recent state).
#
# Git is optional everywhere: no git binary, no repo, no commits, or an sha the
# repo doesn't know all degrade to an unknown stamp, `[unverified freshness]`
# and an undemoted score. Every git call is memoized per process — HEAD per
# repo, `rev-list --count` / `diff --name-only` per (repo, sha) — so annotating
# a whole result set costs one subprocess pair per DISTINCT stamp, shared.
#
# STAMP FORMAT (one footer field; `-` = that part unknown; paths comma-joined):
#
#     _drift: <sha>|-|<path1,path2,...>|-|<YYYY-MM-DDTHH:MM:SS>|<repo-root>|-
#
# The sha is stored full and displayed abbreviated (7 chars). Spaces in the repo
# root are stored as `%20` so the field stays a single non-space token; a root
# containing a literal `|` is left unstampt (the separator is load-bearing).

# Only stamp paths that look like real source-ish files: a token needs a `/`
# AND a known extension (`nx/_tools.py`, `tests/test_mem.py`, `docs/arch.md`).
# Deliberately narrower than "any token with a slash" — URLs, prose (`and/or`)
# and shell fragments would stamp junk paths that make every note look stale.
_DRIFT_KNOWN_EXTS = frozenset("""
py pyi pyx ipy js ts tsx jsx mjs cjs json toml yaml yml md markdown rst txt
rs go c h cpp cc hh hpp java kt swift rb php sh bash zsh fish sql html css scss
vue lua pl ex exs erl tf cmake mk proto lock cfg ini
""".split())

# `(?<![\w./-])` stops a mid-word fragment (`foo/nx/_tools.py`) from matching
# from its inner slash onward.
_DRIFT_PATH_RE = re.compile(
    r"(?<![\w./-])[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*\.[A-Za-z0-9]{1,6}(?![\w/-])")

_DRIFT_MAX_PATHS = 12
_DRIFT_FIELD_RE = re.compile(r"(?:^|\s)_drift:\s*(\S+)")
_DRIFT_ANY_RE = re.compile(r"(?:^|\s)_drift:\s*\S*")
_DRIFT_SEP = "|"


def _extract_drift_paths(body: str) -> tuple:
    """Repo-relative paths the note body mentions, first-mention order, capped.
    Absolute, parent-escaping, URL and memory-store paths are dropped: the
    stamp feeds `git diff --name-only`, which only answers for repo-relative
    paths."""
    out: list = []
    for tok in _DRIFT_PATH_RE.findall(body or ""):
        if "://" in tok or tok.startswith("/") or ".." in tok.split("/"):
            continue
        p = tok[2:] if tok.startswith("./") else tok
        if p.rsplit(".", 1)[-1].lower() not in _DRIFT_KNOWN_EXTS:
            continue
        if "memories/" in p:
            continue  # a note pointing at another note proves nothing about code
        if p not in out:
            out.append(p)
        if len(out) >= _DRIFT_MAX_PATHS:
            break
    return tuple(out)


# Per-process memo tables (conftest rewinds nx.* module dicts between tests, so
# a test never inherits a cached HEAD from an earlier one).
_DRIFT_HEAD: dict = {}
_DRIFT_BEHIND: dict = {}
_DRIFT_TOUCHED: dict = {}


def _git_in(root: str, args: list, timeout: float = 2.0):
    """One git command in `root`: stdout on success, None on ANY failure (no
    git binary, not a repo, unknown sha, timeout). Never raises.

    Bounded at 2 s per call (was 5): drift stamping is informational, so a
    slow git (lock starvation, gc, saturated disk) degrades to an unknown
    stamp instead of stalling the synchronous memory_write path — worst case
    is now a few hundred ms per memoized call, not multiple seconds."""
    if not root:
        return None
    try:
        r = subprocess.run(["git", "-C", root] + list(args),
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip()


def _drift_head(root: str) -> str:
    """HEAD sha of `root`, or '' (no git / not a repo / no commits). Memoized."""
    key = root or ""
    if key not in _DRIFT_HEAD:
        _DRIFT_HEAD[key] = _git_in(root, ["rev-parse", "HEAD"]) or ""
    return _DRIFT_HEAD[key]


def _drift_commits_behind(root: str, sha: str):
    """Commits between `sha` and HEAD (`git rev-list --count sha..HEAD`), or
    None when unknowable (sha absent from the repo, no git). Memoized."""
    key = (root or "", sha)
    if key not in _DRIFT_BEHIND:
        out = _git_in(root, ["rev-list", "--count", sha + "..HEAD"])
        try:
            _DRIFT_BEHIND[key] = int(out) if out is not None else None
        except ValueError:
            _DRIFT_BEHIND[key] = None
    return _DRIFT_BEHIND[key]


def _drift_changed_paths(root: str, sha: str):
    """Paths changed between `sha` and HEAD (`git diff --name-only`), or None
    when unknowable. ONE call per (repo, sha) serves every stamped note in a
    search, so annotating 50 hits costs one subprocess."""
    key = (root or "", sha)
    if key not in _DRIFT_TOUCHED:
        out = _git_in(root, ["diff", "--name-only", sha + "..HEAD"])
        _DRIFT_TOUCHED[key] = None if out is None else frozenset(
            ln.strip() for ln in out.splitlines() if ln.strip())
    return _DRIFT_TOUCHED[key]


class _DriftStamp:
    """A parsed `_drift:` field: the full verified-at sha ('' = unknown), the
    repo-relative paths the body mentioned at write time, the stamp time, and
    the repo drift is computed against."""

    __slots__ = ("sha", "paths", "ts", "root")

    def __init__(self, sha: str, paths: tuple, ts: str, root: str):
        self.sha = sha
        self.paths = paths
        self.ts = ts
        self.root = root

    @property
    def known(self) -> bool:
        """True when the stamp carries an sha drift can be computed against."""
        return bool(self.sha)


def _drift_root() -> str:
    """The repo drift should be computed against right now: the git-detected
    project root, or '' when there is nothing to stamp against. Requires
    PROJECT_ROOT_IS_GIT (the cwd fallback is often $HOME, which would stamp
    unrelated paths and — worse — a real sha from some parent repo)."""
    if not getattr(st, "PROJECT_ROOT_IS_GIT", False):
        return ""
    root = getattr(st, "PROJECT_ROOT", None)
    if not root or "|" in str(root) or not os.path.isdir(str(root)):
        return ""
    return str(root)


def _drift_field(body: str, root: str) -> str:
    """The `_drift:` field value for a note written now in `root`. Outside a
    git repo the sha is still RECORDED (as `-`) with no paths: the write is on
    the record, nothing invents an sha, and nothing stamps paths there is no
    repo to diff against."""
    if not root:
        return _DRIFT_SEP.join(
            ["-", "-", time.strftime("%Y-%m-%dT%H:%M:%S"), "-", "-"])
    paths = _extract_drift_paths(body)
    return _DRIFT_SEP.join([
        _drift_head(root) or "-",
        ",".join(paths) or "-",
        time.strftime("%Y-%m-%dT%H:%M:%S"),
        root.replace(" ", "%20"),
        "-",
    ])


def _drift_line(body: str, root: str) -> str:
    """The footer fragment (`_drift: …`) for a note written now. Built AFTER
    the approval gate, so a denied write claims nothing."""
    return "_drift: " + _drift_field(body, root)


def _parse_drift(content: str):
    """The note's `_drift:` stamp, or None (no field, unknown sha, malformed).
    Body prose mentioning `_drift:` can't fake one by accident: the field is a
    single non-space token of `|`-separated parts and lives in the footer."""
    m = _DRIFT_FIELD_RE.search(content or "")
    if not m:
        return None
    parts = m.group(1).split(_DRIFT_SEP)
    if len(parts) < 4:
        return None
    sha = "" if parts[0] == "-" else parts[0]
    if not sha:
        return None
    paths = tuple(p for p in parts[1].split(",") if p and p != "-")
    ts = "" if parts[2] == "-" else parts[2]
    root = "" if parts[3] == "-" else parts[3].replace("%20", " ")
    return _DriftStamp(sha, paths, ts, root)


def _stamp_note_now(content: str, root: str):
    """Retroactive stamping: when `content` carries no `_drift:` field AT ALL
    AND `root` has a HEAD, return (new_content, line) stamped at that HEAD;
    else None. A note that already has the field — even an unknown-sha one — is
    left byte-identical, so the backfill runs once per note, not every scan;
    and with no HEAD to record there is nothing worth recording, so notes are
    left untouched rather than rewritten with a useless stamp (which would
    churn every mtime in the store and re-index the whole FTS cache)."""
    if not root or not _drift_head(root):
        return None
    if _DRIFT_ANY_RE.search(content or ""):
        return None
    line = _drift_line(content, root)
    stamped = _append_footer_field(content, line)
    if not stamped or stamped == content:
        return None  # nothing stampable — leave the note alone
    return stamped, line


def _append_footer_field(content: str, field: str) -> str:
    """Append `field` to the note's `_scope:` footer line, or `content`
    unchanged when it has no footer at all. Inventing a footer for a footer-less
    note is NOT done: `_MEMORY_FOOTER_RE` (which the archiver uses to strip the
    footer before re-saving) anchors on `_scope:`, so a footer-less note's new
    stamp line would never be stripped again and would leak into the body the
    optimizer/archiver rewrites. The original trailing newline is preserved."""
    lines = content.splitlines()
    for i, ln in enumerate(lines):
        if "_scope:" in ln:
            lines[i] = ln.rstrip() + "  " + field
            out = "\n".join(lines)
            if content.endswith("\n") and not out.endswith("\n"):
                out += "\n"
            return out
    return content


_DRIFT_FRESH = "[fresh: verified at {sha}, {behind} commits behind]"
_DRIFT_STALE = ("[stale: verified at {sha}, {behind} commits behind, "
                "{changed} of its files changed]")
_DRIFT_UNVERIFIED = "[unverified freshness]"


def _drift_freshness(stamp):
    """(commits_behind, stamped_paths_changed) for one stamp, against its own
    repo's HEAD. (0, 0) for unstamped/unknown stamps — those are annotated
    unverified and keep their raw score."""
    if stamp is None or not stamp.known:
        return 0, 0
    behind = _drift_commits_behind(stamp.root, stamp.sha)
    if behind is None:
        return 0, 0
    # Skip the diff when the note is not behind: the two calls stack (each
    # bounded at _git_in's timeout), and the challenge path runs synchronously
    # inside memory_write — 0 commits behind already means "fresh".
    if max(0, behind) == 0:
        return 0, 0
    touched = _drift_changed_paths(stamp.root, stamp.sha)
    changed = 0 if touched is None else sum(1 for p in stamp.paths if p in touched)
    return max(0, behind), changed


def _annotate_drift(stamp) -> str:
    """The SHORT freshness tag for one search hit — it lands on the hit line in
    the model's context, so it stays one short bracket."""
    if stamp is None or not stamp.known:
        return _DRIFT_UNVERIFIED
    behind = _drift_commits_behind(stamp.root, stamp.sha)
    if behind is None:
        return _DRIFT_UNVERIFIED
    behind = max(0, behind)
    if behind == 0:
        return _DRIFT_FRESH.format(sha=stamp.sha[:7], behind=0)
    touched = _drift_changed_paths(stamp.root, stamp.sha)
    changed = 0 if touched is None else sum(1 for p in stamp.paths if p in touched)
    return _DRIFT_STALE.format(sha=stamp.sha[:7], behind=behind, changed=changed)


def _drift_demote(stamp) -> float:
    """Rank demotion weight: 0.0 for unstamped/unknown/fresh notes (raw score),
    growing with commits-behind and with how many of the note's OWN files the
    drift touched. Callers rank on `score / (1 + _drift_demote(stamp))`."""
    behind, changed = _drift_freshness(stamp)
    if behind <= 0 and changed <= 0:
        return 0.0
    return min(10.0, 0.02 * behind + 0.5 * changed)


def _drift_challenge(stamp) -> str:
    """The write-time challenge for a note `memory_write` just superseded, or
    '' when it was fresh. Purely informational — appended after a successful
    write, never a refusal."""
    behind, changed = _drift_freshness(stamp)
    if behind <= 0 and changed <= 0:
        return ""
    return ("\nNote was " + str(behind) + " commits stale with " + str(changed)
            + " changed files; confirm its claims still hold or scope your update.")
