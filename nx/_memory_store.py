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
one footer convention (`_scope:`/`_saved:`/`_kind:`/`_derived:`, plus the drift stamp
`_drift:`, the supersession pair `_supersedes:` / `_superseded_by:` +
`_valid_until:`, and the consolidation list `_summary_of:` — all documented at
the bottom of this module), secret redaction of the title+body, atomic
tmp+rename, private 0600 permissions (memories are distilled from sessions and
may contain paths/commands/credentials; same posture as session files).
Publication is claim-based (`atomic_write_private_noreplace`): the filename is
not picked by an exists-then-write check that two concurrent writers (the
memory-merge workers) can both pass — the first to publish the bytes wins and
the loser takes the next suffix.

Every read goes through `_iter_notes` — the one scanner over a memories dir —
so title extraction, mtime handling, supersession filtering and unreadable-file
tolerance live in exactly one loop. All writes go through nx/_private.py
(0600/0700, umask-proof)."""
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
# The window a titles-only scan probes at EOF to answer one footer question
# ("is this note superseded?"). A footer line is one short line; 512 bytes
# covers it with room for hand-added fields, and stays far under a body read.
_NOTE_TAIL_BYTES = 512


def _read_memory_title(path: str, name: str,
                       tail: list | None = None) -> tuple[str, str] | None:
    """(title, head_text) for one note from a bounded head read, or None when
    the head is unreadable or not valid UTF-8 — the same skip a full read makes,
    so the scanner tolerates a broken note identically on both paths. The title
    falls back to the filename stem when no H1 appears in the head.

    A note whose first bytes are valid but whose *body* later is not would pass
    here and be skipped by a full read; that is the one (rare) divergence, and
    it only widens which titles a title-only caller lists.

    `tail` (0.158.20), when given, is a list a bounded TAIL read is appended to —
    for a title-only caller that also needs one footer field (`_superseded_by:`)
    without reading the body. Reading it from the SAME open is deliberate: the
    session-start note scans every note in both scopes, and a second open per
    note would double that cost for a field that lives in ~60 bytes at EOF.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(_TITLE_HEAD_BYTES)
            if tail is not None:
                try:
                    f.seek(-_NOTE_TAIL_BYTES, os.SEEK_END)
                except OSError:
                    f.seek(0)  # the note is shorter than the probe window
                tail.append(f.read(_NOTE_TAIL_BYTES).decode("utf-8", "replace"))
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


# --- secret redaction at the one writer ----------------------------------------
#
# nx's threat model (see nx/_private.py) protects `~/.nx` content with FILE MODES
# (0600/0700) only — there is no content scanner. For notes that is not enough,
# because a note is not inert storage: it is RE-INJECTED into every future
# session. `retro_on_exit` writes notes automatically at session end (gate=False,
# ON by default) and `/memory save` distils up to _MEMORY_TRANSCRIPT_CHARS of
# tool output, so a credential that merely APPEARED in one tool result can be
# distilled into a note, and from there into the session-start [Saved memories]
# note (preference-kind notes are injected every session), into the FTS index
# and into any backup. That promotes "one transcript contained a token" — which
# 0600 covers — to "every prompt forever, in every project", which it does not.
#
# So the one writer REDACTS. Two properties are load-bearing:
#   REDACT, never block. Dropping or failing an auto-write silently loses the
#   lesson; replacing the secret with `[REDACTED:<label>]` keeps the lesson and
#   kills the credential, and the label tells a human what was there and why it
#   went. The matched text is never echoed back to a caller.
#   ONE CHOKE POINT. It sits inside `_write_memory_file`, so `memory_write`,
#   `/memory save`, `/retro`, `/memory optimize` merges and `/memory reflect`
#   are all covered by one insertion point (archive restore re-enters through
#   the same writer).
#
# SCOPE BOUNDARY (deliberate): NOTES ONLY. Session transcripts (`_sessiondb`,
# `_sessions_fs`) are NOT scanned here — they are the raw record, and rewriting
# them would break replay/audit; they stay protected by file modes.
#
# THE FOOTER IS NEVER SCANNED. The `_drift:` stamp is
# `<sha>|-|<paths>|-|<ISO-time>|<repo-root>|-`, so ANY hex-shaped rule would
# shred that sha and silently break drift demotion plus the write-time staleness
# challenge. `_redact_secrets` is therefore called on the TITLE and BODY only;
# the footer block is assembled after redaction and passes through untouched.
# (A bare "long hex string" rule is deliberately ABSENT for the same reason:
# commit shas are legitimate note content and must round-trip.)

_REDACTION_MARK = "[REDACTED:%s]"

# Labelled pattern set, applied IN ORDER. Prefix-anchored on purpose: a rule
# keyed on "high entropy" shreds legitimate note content (shas, hashes, base64
# diffs, UUIDs), while a real credential almost always carries a stable prefix.
# Stdlib `re`, precompiled once at module scope.
_SECRET_PATTERNS: tuple = (
    # PEM / OpenSSH private key blocks (and the bare BEGIN marker, so a key
    # truncated by the transcript cap still can't leak its first material lines).
    ("private_key_block", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"
        r".*?-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----", re.DOTALL)),
    ("private_key_marker", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")),
    # Anthropic (sk-ant-…) / OpenAI (sk-, sk-proj-…). Body chars are pure
    # alphanumerics so prose like `task-tracking-…` cannot match: \b fails
    # before the "sk" inside "task", and a hyphen ends the run.
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9]{16,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|api-)?[A-Za-z0-9]{16,}")),
    # Stripe live/test publishable + secret keys.
    ("stripe_key", re.compile(r"\b[sp][ak]_(?:live|test|ramp)_[A-Za-z0-9]{8,}")),
    # GitHub personal/oauth/app/user/refresh/empirical + fine-grained PATs.
    ("github_token", re.compile(
        r"\bgh[pousrex]_[A-Za-z0-9]{16,}\b|\bgithub_pat_[A-Za-z0-9_]{16,}\b")),
    # AWS access key ids (AKIA/ASIA + the 16-char body; `{16,}` so a key that
    # runs long — or is followed by alnum padding — still gets caught).
    ("aws_key", re.compile(r"\bA[K-S]IA[0-9A-Z]{16,}\b")),
    # Slack bot/user/app/admin/server tokens.
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    # Google API keys.
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}")),
    # JWTs: three base64url segments, header always `eyJ…`.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}"
                       r"\.(?:[A-Za-z0-9_\-]{4,}|)\b")),
    # `Authorization: Bearer <token>` (the header name is optional; the long
    # opaque run is what makes it a credential rather than prose).
    ("bearer_token", re.compile(
        r"(?i)(?:\bauthorization\s*:\s*)?Bearer\s+(?!\[REDACTED:)"
        r"[A-Za-z0-9._~+/=\-]{16,}")),
    # DB URLs with an embedded password: scheme://user:***@host. The `:`
    # before the `@` is required, so `https://example.com/x` can't match.
    ("db_url_password", re.compile(
        r"\b[a-z][a-z0-9+.\-]{1,15}://[^\s:/@\\]*:(?!\[REDACTED:)"
        r"[^\s/@\\]+@[^\s'\"<>)\]]+")),
    # Generic `key = value` where the KEY names a secret. Group 1 (the key name)
    # is preserved so the note still says WHICH variable held the credential.
    # The `(?!\[REDACTED:)` guards are load-bearing: patterns apply IN ORDER, so
    # a later generic rule would otherwise swallow an earlier rule's marker
    # (`GITHUB_TOKEN=[REDACTED:github_token]` → `[REDACTED:secret_value]`),
    # destroying the more specific label and double-counting the redaction.
    ("secret_value", re.compile(
        r"(?i)(\b[A-Za-z0-9_]*(?:secret|token|password|passwd|pwd|credential"
        r"|api[_-]?key|access[_-]?key|private[_-]?key|auth[_-]?(?:key|token))"
        r"[A-Za-z0-9_]*\b\s*[:=]\s*)([\"']?)(?!\[REDACTED:)([^\"'\s,;]{6,})(\2)")),
)


def _redact_secrets(text: str, labels: list | None = None) -> str:
    """Replace every secret-shaped span in `text` with `[REDACTED:<label>]`.

    `labels` (optional, caller-owned) collects one label per replacement so a
    caller can report "N secret(s) redacted" WITHOUT echoing the matched text.
    Best-effort by contract: a pattern that raises is skipped, never propagated
    — a redaction bug must not be able to fail a write. The footer block is
    never passed through here (see the note above): only a note's title+body.
    """
    if not text:
        return text
    out = text

    def _keep_key(m):
        # secret_value: keep the key name and its quoting, replace the value.
        return m.group(1) + m.group(2) + _REDACTION_MARK % "secret_value" + m.group(4)

    for label, rx in _SECRET_PATTERNS:
        try:
            repl = _keep_key if label == "secret_value" else (_REDACTION_MARK % label)
            out, n = rx.subn(repl, out)
        except (re.error, RecursionError):
            continue
        if n and labels is not None:
            labels.extend([label] * n)
    return out


def _redaction_note(labels: list) -> str:
    """The caller-visible tail for a write that redacted something: a count plus
    the DISTINCT labels in first-seen order. Never the matched text."""
    if not labels:
        return ""
    seen: list = []
    for lab in labels:
        if lab not in seen:
            seen.append(lab)
    return ("\n%d secret(s) redacted (%s) — the lesson is kept, the credential "
            "is not stored." % (len(labels), ", ".join(seen[:6])))


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


def _note_derived(content: str) -> str:
    """The `_derived:` provenance value of a note's footer, or '' when the note
    is first-hand (or predates the field). Tolerates hand-edited notes the same
    way _note_kind does."""
    m = re.search(r"(?:^|\s)_derived:\s*(\S+)", content)
    return m.group(1) if m else ""


# --- supersession and consolidation, as FOOTER DATA ------------------------------
#
# Supersession used to be a docstring plea (`memory_write`'s description told the
# model to memory_delete the stale note). Deletion is irreversible and it throws
# away the note that still explains WHY the old answer was wrong, so it is now
# data: the retired note carries `_superseded_by: <new file>` + `_valid_until:
# <ISO>`, the new note carries `_supersedes: <old file>`. Both ride the `_scope:`
# footer line, exactly like `_derived:` does, so the footer-strip regex keeps
# working and an archive→restore round-trip neither doubles nor loses a field.
#
# WHERE THE INVARIANT IS ENFORCED (one authoritative layer): `_iter_notes`, the
# live scanner, drops superseded notes. Everything that must not see them —
# `memory_search`'s plain-scan arm, `_scan_memory_entries` (the delete resolver),
# the session-start [Saved memories] note, `/memory optimize`'s note index —
# builds on that scanner, so they inherit the filter with no code of their own.
# The one place that does NOT go through it is `nx._memindex.reconcile`, which
# builds FTS rows from a raw disk listing; it mirrors the skip (see there) so a
# superseded note's row is dropped on the next search instead of lingering.
#
# REVERSIBLE BY DESIGN, like the archive: the file always stays on disk, and
# `/memory unsupersede <target>` clears the two fields. A filtered note a human
# can only find by hand would be silent data-loss if supersession ever misfired,
# so `/memory list` keeps showing them (marked) and the resolver used to retire
# and to restore sees superseded notes too.
#
# `_summary_of:` (same line, same convention) is PROVENANCE ONLY — the source
# filenames of a consolidated note. It changes no ranking, no filtering, no
# visibility; it is parsed back out symmetrically with `_derived:`.

_SUPERSEDED_BY_RE = re.compile(r"(?:^|\s)_superseded_by:\s*(\S+)")
_SUPERSEDES_RE = re.compile(r"(?:^|\s)_supersedes:\s*(\S+)")
_VALID_UNTIL_RE = re.compile(r"(?:^|\s)_valid_until:\s*(\S+)")
_SUMMARY_OF_RE = re.compile(r"(?:^|\s)_summary_of:\s*(\S+)")
# Chain walk cap for a `supersedes` target that is itself already retired
# (A→B→C). Redirect to the terminal live note rather than build a chain; give up
# (refuse) past this depth instead of recursing on a hand-edited cycle.
_SUPERSESSION_MAX_HOPS = 8
# Cap on `_summary_of:` names, matching _extract_drift_paths' cap-at-12
# precedent: one giant merge must not produce an unbounded footer line.
_SUMMARY_OF_MAX = 12


def _note_superseded_by(content: str) -> str:
    """The `_superseded_by:` filename of a note's footer, or '' when the note is
    live. Tolerates hand-edited notes the way _note_kind does."""
    m = _SUPERSEDED_BY_RE.search(content or "")
    return m.group(1) if m else ""


def _note_supersedes(content: str) -> str:
    """The `_supersedes:` filename this note retired, or ''."""
    m = _SUPERSEDES_RE.search(content or "")
    return m.group(1) if m else ""


def _note_summary_of(content: str) -> str:
    """The comma-separated `_summary_of:` source list, or ''. Provenance only."""
    m = _SUMMARY_OF_RE.search(content or "")
    return m.group(1) if m else ""


def _summary_of_field(names) -> str:
    """The `_summary_of:` footer fragment for a consolidated note, capped at
    _SUMMARY_OF_MAX names (deduped, order-preserving). '' when nothing to say.
    Accepts a list of filenames or one comma-joined string (what the archive
    reads back), so a round-trip cannot iterate a string character by character."""
    if isinstance(names, str):
        names = names.split(",")
    seen: list = []
    for n in names or ():
        n = (n or "").strip()
        if n and n not in seen:
            seen.append(n)
    if not seen:
        return ""
    return "_summary_of: " + ",".join(seen[:_SUMMARY_OF_MAX])


def _clear_supersession(content: str) -> str:
    """`content` with the `_superseded_by:` and `_valid_until:` fields removed
    from its footer line — the unsupersede (restore-to-live) transform. A note
    with no footer, or without the fields, comes back byte-identical."""
    out = re.sub(r"[ \t]+_superseded_by:\s*\S+", "", content or "")
    return re.sub(r"[ \t]+_valid_until:\s*\S+", "", out)


def stamp_superseded(path: str, new_name: str) -> bool:
    """Stamp the retired note at `path` with the footer pair that makes
    supersession DATA instead of a docstring plea:

        _superseded_by: <new note's filename>  _valid_until: <ISO time>

    Both ride the `_scope:` footer line via `_append_footer_field` (the same
    route `_derived:` and `_drift:` take), so `_MEMORY_FOOTER_RE` still strips a
    whole footer and an archive→restore round-trip can neither double nor lose
    a field. IDEMPOTENT by construction: any existing pair is removed first, so
    re-superseding an already-retired note restamps it instead of stacking a
    second copy. The file STAYS ON DISK — that is the reversible-by-design part.
    An mtime bump is wanted here (not tolerated, as in `_backfill_persist`): the
    footer change must reach the FTS row, and mtime is `_memindex.reconcile`'s
    cache-invalidation token.

    Returns False (writing nothing) when the note has no footer to stamp, or is
    unreadable/unwritable — supersession is an optimisation on top of the write,
    so a caller must never fail the new note over it."""
    read = _read_memory_file(path, os.path.basename(path))
    if read is None:
        return False
    _title, content = read
    stamped = _append_footer_field(
        _clear_supersession(content),
        "_superseded_by: %s  _valid_until: %s" % (
            new_name, time.strftime("%Y-%m-%dT%H:%M:%S")))
    if not stamped or stamped == content:
        return False  # footer-less note: inventing one is not done (see _append_footer_field)
    try:
        from ._private import atomic_write_private
        atomic_write_private(path, stamped)
    except OSError:
        return False
    return True


def repoint_superseded(old_name: str, new_name: str) -> int:
    """Re-point every note whose `_superseded_by:` names `old_name` at
    `new_name` instead, over every memory dir. Returns how many were restamped.

    Keeps the supersession graph FLAT: retiring a note that is itself already a
    supersession target would otherwise create A→B→C, whose middle link is a
    retired note nothing resolves forward from. Flat means "find the live tip"
    is one lookup, and `/memory unsupersede` restores one pair, not a subtree.

    Best effort and cheap: a titles-only scan (the field is in the footer, which
    the tail probe reads) plus one rewrite per pointer we actually redirect."""
    moved = 0
    for d, _label in _iter_memory_dirs("project"):
        for name, path, _mtime, _title, head in _iter_notes(
                d, titles_only=True, include_superseded=True):
            if name == old_name or name == new_name:
                continue
            if _superseded_in_head(head, _note_tail(path)) != old_name:
                continue
            if stamp_superseded(path, new_name):
                moved += 1
    return moved


def resolve_note_target(target: str, scope: str,
                        include_superseded: bool = True):
    """(entry, candidates, entries) for `target` = an exact title, a filename
    (with or without `.md`), or a UNIQUE title prefix, over the dirs `scope`
    covers. `entry` is one (name, path, mtime, title, label) tuple when the
    target resolved, else None and `candidates` holds what made it ambiguous
    (or the recent notes, when it matched nothing).

    THE resolver `memory_delete` uses (its own copy in `_memory_search.py` is
    the same three-rule walk over `_scan_memory_entries`); supersession and
    `/memory unsupersede` go through HERE instead of writing a fourth variant,
    so "ambiguous ⇒ refuse and list" is one rule with one definition.

    `include_superseded=True` by default: retiring and un-retiring both have to
    be able to NAME a retired note (the default search view hides them), and
    redirecting A→B→C needs to see that B is already retired.
    """
    entries = []
    try:
        for d, label in _iter_memory_dirs(scope):
            for name, path, mtime, title, _c in _iter_notes(
                    d, titles_only=True, include_superseded=include_superseded):
                entries.append((name, path, mtime, title, label))
    except Exception:
        return None, [], []
    if not entries:
        return None, [], []
    low = (target or "").strip().lower()
    if not low:
        return None, entries[:10], entries
    exact = [e for e in entries
             if e[0].lower() == low or e[0][:-3].lower() == low
             or e[3].lower() == low]
    if len(exact) == 1:
        return exact[0], [], entries
    cands = exact if exact else [e for e in entries if e[3].lower().startswith(low)]
    if len(cands) == 1:
        return cands[0], [], entries
    return None, (cands if cands else entries[:10]), entries


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


def _iter_notes(d: str, *, titles_only: bool = False,
                include_superseded: bool = False
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

    Supersession filter (0.158.20): a note whose footer carries `_superseded_by:`
    is NOT returned — this scanner is the ONE authoritative layer for the rule
    (see the supersession section above), which is why search, the session-start
    note and `/memory optimize` all inherit it for free. `include_superseded=
    True` opts back in, for the surfaces that must keep retired notes reachable:
    `/memory list`, `/memory unsupersede`, and the `supersedes` resolver. In
    titles_only mode the field is read from a bounded TAIL read (the footer lives
    at EOF, the head cannot see it) off the same open file — one open per note
    either way.

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
        tail: list = [""]
        if titles_only:
            head = _read_memory_title(path, name, tail=tail)
            if head is None:
                continue
            title, content = head[0], head[1]
            if not include_superseded and _superseded_in_head(content, tail[0]):
                continue
        else:
            read = _read_memory_file(path, name)
            if read is None:
                continue
            title, content = read
            if not include_superseded and _note_superseded_by(content):
                continue
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


def _superseded_in_head(head: str, tail: str) -> str:
    """The `_superseded_by:` value of a note whose body we did NOT read, from the
    head and the bounded tail `_read_memory_title` collected in one open. A field
    already visible in the head (a tiny or hand-edited note) is trusted, else the
    tail answers it; an empty tail (unreadable EOF) means "live", because a note
    whose head read fine is not dropped over a footer probe that failed."""
    return _note_superseded_by(head) or _note_superseded_by(tail)


def _note_tail(path: str) -> str:
    """The footer window of one note (the last _NOTE_TAIL_BYTES), for a caller
    that needs a footer field without reading the body and does not have
    `_read_memory_title`'s tail at hand. '' on any failure."""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-_NOTE_TAIL_BYTES, os.SEEK_END)
            except OSError:
                f.seek(0)  # shorter than the probe window
            return f.read(_NOTE_TAIL_BYTES).decode("utf-8", "replace")
    except OSError:
        return ""


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
                       kind: str = "", derived: str | None = None,
                       supersedes: str | None = None, summary_of=None,
                       redactions: list | None = None) -> tuple[str, str] | None:
    """The single memory writer: shared by the memory_write tool and
    `/memory save` so both produce identical files. Returns (path, scope), or
    None when gate=True and the user denied the confirmation.

    kind: optional note kind tag (see MEMORY_KINDS) stored in the footer as
    `_kind:`; an empty string keeps the footer kind-free (old format).

    derived: optional PROVENANCE tag — set only when this note was distilled
    from nx's OWN memory output (the `[Saved memories]` titles / note bodies in
    a transcript, or an `/memory optimize` merge of existing notes) rather than
    from direct first-hand observation. Written as `_derived: <value>` on the
    `_scope:` footer line; None (the default) leaves every existing call site's
    bytes unchanged, so a pre-`_derived` footer round-trips identically.
    Provenance is RECORDED, never enforced: a derived note stays visible to
    search and to the session-start note.

    summary_of: optional list of SOURCE NOTE FILENAMES this note was
    consolidated from (`/memory optimize` merges, `/memory reflect` promotions).
    Written as `_summary_of: a.md,b.md` on the same footer line, capped at
    _SUMMARY_OF_MAX names. PROVENANCE ONLY — like `_derived:`, it changes no
    ranking, no filtering, no visibility.

    supersedes: optional filename of the note this one RETIRES, already
    resolved by the caller (see `_memory_cmd._resolve_note_target` — the
    resolver `memory_delete` uses, so an ambiguous target can refuse the whole
    write). Stamps `_supersedes: <file>` here; the CALLER stamps the matching
    `_superseded_by:`/`_valid_until:` pair onto the old note, because that needs
    THIS note's final filename.

    redactions: optional list the caller owns; one label per secret this write
    redacted is appended to it (never the matched text) so the caller can report
    the count. Redaction happens here, on the TITLE and BODY ONLY — see the
    redaction section above for why the footer is never scanned.

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

    # THE redaction choke point. Before the gate, so what the user approves is
    # what lands on disk; the title too, since a token can land in a title. Best
    # effort: `_redact_secrets` swallows its own regex failures, so a redaction
    # bug degrades to "unscanned", never to "write failed".
    labels: list = []
    title = _redact_secrets(title, labels)
    body = _redact_secrets(body, labels)
    if labels and redactions is not None:
        redactions.extend(labels)

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
    if derived:
        footer += f"  _derived: {derived}"
    if summary_of:
        field = _summary_of_field(summary_of)
        if field:
            footer += f"  {field}"
    if supersedes:
        footer += f"  _supersedes: {supersedes}"
    # Drift stamp: which commit this note is verified against, plus the repo
    # paths its body mentions ('' outside a git repo — no stamp, not a lie).
    # Computed from the ALREADY REDACTED body, and appended AFTER redaction: the
    # stamp's sha must never pass through the scanner (see the trap note above).
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
