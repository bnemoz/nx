"""Checkpoint/undo stack for file-touching tools.

`run_tool()` in `nx/_dispatch.py` snapshots a file's prior contents just
before `write_file`/`edit_file`/`multi_edit`/`delete_file` apply, and pushes
the snapshot here once the tool call actually succeeds (denied/errored calls
never touched the file, so they push nothing). `sed_edit` can touch a whole
glob of files in one call, so it snapshots each matched file itself and pushes
them as one entry. Either way, one stack entry = one tool call, and it may
cover several files: the `/undo` REPL command pops the most recent entry and
restores every file in it in one step, so undoing a `multi_edit` or
`sed_edit` call reverts all the files it touched together, not one at a time.

Since 0.47.0 the stack is also PERSISTENT and idempotent:

- `_UNDO_PATH` (per session id: `<NX_HOME>/undo/<session-id>.json`; the legacy
  shared `<NX_HOME>/undo.json` only when no session id is known) is written
  atomically on every push/pop, so a crash no longer loses the recovery story
  that YOLO/batch postures assume; `/reset` clears it. One pending-delete per
  path is kept:
  a re-delete of a file already pending undo updates the entry in place
  (one `/undo` still passes one delete) instead of stacking duplicates.
- `undo_last()` is atomic per file: each file's restore is written to a temp
  file and `os.replace`d, so a failure partway leaves earlier files restored
  and the entry retained (with the failed files dropped) for the next press.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Optional, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # 3.9/3.10 have no typing.NotRequired; typing_extensions if present,
    try:  # else a subscript shim — this is annotation-only, never enforced.
        from typing_extensions import NotRequired
    except ImportError:  # pragma: no cover
        class NotRequired:  # type: ignore[no-redef]
            def __class_getitem__(cls, item):
                return item

from ._env import _env_int
from ._fullscreen import View
from ._private import atomic_write_private
from . import _state as st

MAX_UNDO = _env_int("NX_MAX_UNDO", 20)

# Swarm P3: the stack holds whole file contents, so MAX_UNDO entries against
# big files is an unbounded RSS term ×N agents. A byte cap evicts oldest-first
# (evicting the OLDEST checkpoint is the safe direction: /undo works from the
# newest back). 0 disables the cap.
MAX_UNDO_BYTES = _env_int("NX_MAX_UNDO_BYTES", 32 * 1024 * 1024)

# PF-05 (0.61.0): persist_undo_stack() used to re-serialize the WHOLE stack —
# up to MAX_UNDO entries holding whole file contents — on every single edit,
# so twenty edits to a 200 KB file rewrote tens of MB of JSON. Persistence is
# now amortized: a push marks the stack dirty and the write happens once at
# the turn boundary (_flush_undo_stack); undo_last/clear_persisted still write
# inline, where the on-disk state must match the user's just-observed action.
_UNDO_DIRTY = False


class FileSnapshot(TypedDict):
    path: str
    existed: bool
    content: Optional[str]
    # Post-tool content hash, stamped by the push choke points (data-loss
    # guard — see undo_last/_content_sha). Optional so old persisted entries
    # (no field) still load.
    after_sha: NotRequired[str]


def _content_sha(path: str) -> str:
    """sha256 hex of a file's CURRENT bytes, or "" when it is absent (or
    unreadable) — the sentinel that also covers a pending delete.

    Why (data-loss guard): the undo stack is persistent and cross-session, so a
    checkpoint loaded at startup can be older than changes made since (another
    session, a commit, a hand edit). undo_last() used to rewrite a snapshot's
    OLD content unconditionally, reverting disk while the git index held the
    newer bytes — leaving the staged diff the exact inverse of the unstaged
    diff. Comparing hashes lets undo refuse a restore whose file no longer
    matches what the tool left."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _stamp_after_sha(files: list[FileSnapshot]) -> None:
    """Re-read each snapshot's path and record its post-tool content hash.
    Called from the two push choke points (all push sites go through them),
    right after the tool succeeded — one extra read per edited file per push.
    Must NOT be called from any hotter loop.

    If the file STILL holds the snapshot's own pre-tool content, this push is
    observing the state BEFORE the tool's write landed — sed_edit pushes its
    checkpoint ahead of its write loop (`nx/_edit_tools.py`), so at that
    instant disk shows the pre-state and we cannot see what the tool is about
    to leave. Recording `after_sha` there would make every `/undo` of a
    sed_edit refuse as "changed since checkpoint"; omit the field instead so
    that entry falls back to the legacy blind restore, exactly like an
    old-format persisted entry. Every other write tool pushes AFTER its write,
    so cur != pre and the guard is stamped as designed (delete_file stamps the
    "" sentinel: its push follows os.remove)."""
    for snap in files:
        cur = _content_sha(snap["path"])
        pre = ("" if not snap["existed"]
               else hashlib.sha256((snap["content"] or "").encode()).hexdigest())
        if cur == pre:
            continue
        snap["after_sha"] = cur


class Checkpoint(TypedDict):
    tool: str
    ts: float
    files: list[FileSnapshot]


def _read_prior(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def snapshot(path: Optional[str]) -> Optional[FileSnapshot]:
    """Capture a file's state before a write/edit tool touches it."""
    if not path:
        return None
    existed = os.path.exists(path)
    content = _read_prior(path) if existed else None
    # Unreadable-as-text existing file (binary, permissions) — content is None
    # but existed is True; skip so undo never overwrites it with garbage.
    if existed and content is None:
        return None
    return {"path": path, "existed": existed, "content": content}


def _pending_paths() -> set[str]:
    return {snap["path"] for entry in st._UNDO_STACK for snap in entry["files"]}


def _evict_over_bytes() -> None:
    if MAX_UNDO_BYTES <= 0:
        return
    total = sum(len(s.get("content") or "")
                for e in st._UNDO_STACK for s in e["files"])
    while st._UNDO_STACK and total > MAX_UNDO_BYTES and len(st._UNDO_STACK) > 1:
        dropped = st._UNDO_STACK.pop(0)  # oldest first — /undo consumes from the end
        total -= sum(len(s.get("content") or "") for s in dropped["files"])


def push_checkpoint(tool: str, snaps: list[Optional[FileSnapshot]]) -> None:
    """`snaps` is a list of per-file snapshot dicts (as from `snapshot()`);
    entries with a None snapshot are dropped. Pushes nothing if the list ends
    up empty (e.g. every file was unreadable).

    Idempotent per path (0.47.0): a file already pending in an earlier
    checkpoint of the same tool is UPDATED in place (its snapshot replaced
    with the fresh one) rather than stacked, so repeated writes to the same
    file and a re-delete of a deleted file still cost exactly one `/undo`
    press. The stack stays a faithful map from "press /undo" to "the state
    before that tool call last changed the file"."""
    files = [s for s in snaps if s is not None]
    if not files:
        return
    _stamp_after_sha(files)
    existing = set(_pending_paths())
    fresh: list[FileSnapshot] = []
    for snap in files:
        if snap["path"] in existing:
            for entry in st._UNDO_STACK:
                for old in entry["files"]:
                    if old["path"] == snap["path"]:
                        # Keep the tool label of the checkpoint that owns it.
                        old["existed"] = snap["existed"]
                        old["content"] = snap["content"]
                        # Carry the guard forward only when this push could
                        # observe the post-tool state; otherwise drop any stale
                        # field so the entry reverts to the legacy blind path.
                        if "after_sha" in snap:
                            old["after_sha"] = snap["after_sha"]
                        else:
                            old.pop("after_sha", None)
                        break
            continue
        fresh.append(snap)
    if fresh:
        st._UNDO_STACK.append({"tool": tool, "ts": time.time(), "files": fresh})
    if len(st._UNDO_STACK) > MAX_UNDO:
        del st._UNDO_STACK[:len(st._UNDO_STACK) - MAX_UNDO]
    _mark_dirty()


def push_if_absent(tool: str, snaps: list[Optional[FileSnapshot]]) -> None:
    """Dedup layer over `push_checkpoint` for the dispatcher (0.47.0): a tool
    body that pushes its own checkpoint (delete_file, sed_edit, edit_file's
    diff mode)
    must not be stacked twice. A path that is already pending gets its
    snapshot UPDATED in place with the incoming one, so the stack keeps
    exactly one pending state per file — the most recently pushed snapshot —
    and one approved call always costs one /undo press."""
    files = [s for s in snaps if s is not None]
    if not files:
        return
    _stamp_after_sha(files)
    existing = set(_pending_paths())
    fresh: list[FileSnapshot] = []
    for snap in files:
        if snap["path"] in existing:
            for entry in st._UNDO_STACK:
                for old in entry["files"]:
                    if old["path"] == snap["path"]:
                        old["existed"] = snap["existed"]
                        old["content"] = snap["content"]
                        if "after_sha" in snap:
                            old["after_sha"] = snap["after_sha"]
                        else:
                            old.pop("after_sha", None)
                        break
            continue
        fresh.append(snap)
    if not fresh:
        _mark_dirty()  # a dedup UPDATE changed the stack in place
        return
    st._UNDO_STACK.append({"tool": tool, "ts": time.time(), "files": fresh})
    if len(st._UNDO_STACK) > MAX_UNDO:
        del st._UNDO_STACK[:len(st._UNDO_STACK) - MAX_UNDO]
    _evict_over_bytes()
    _mark_dirty()


def list_checkpoints() -> list[Checkpoint]:
    """Pending checkpoints, most recent first — for `/undo --list`."""
    return list(reversed(st._UNDO_STACK))


# --- persistence (2.2, 0.47.0) -------------------------------------------------

def _undo_path() -> str:
    # Swarm (P2): one fixed path meant N concurrent nx processes rewrote the
    # same undo.json each turn — last writer wins, /undo across the swarm was
    # meaningless. Namespaced per session id when one exists (st.SESSION_ID is
    # absent in some tests/embedded uses), keeping the legacy path as fallback.
    return undo_path_for(getattr(st, "SESSION_ID", None))


def undo_path_for(session_id: str | None) -> str:
    """Undo file for an ARBITRARY session id (not just the current one) —
    the /delete and --prune GC key off the dead sid, not st.SESSION_ID."""
    if session_id:
        return os.path.join(st.SESSION_HOME, "undo", f"{session_id}.json")
    return os.path.join(st.SESSION_HOME, "undo.json")


def delete_persisted_for(session_id: str | None) -> None:
    """Best-effort removal of a (dead) session's persisted undo file. The
    snapshots hold whole pre-edit file contents — up to MAX_UNDO_BYTES each,
    often secrets — so /delete and --prune must not leave them behind.
    (/reset deliberately KEEPS the old session's file: a later /resume of it
    should still have its undo history — 0.128.3.)"""
    try:
        os.unlink(undo_path_for(session_id))
    except OSError:
        pass


def _mark_dirty() -> None:
    """Flag the stack as needing a write at the next turn boundary (PF-05)."""
    global _UNDO_DIRTY
    _UNDO_DIRTY = True


def flush_undo_stack() -> None:
    """Write the stack if a push since the last flush left it dirty. Called
    once per turn (same place as the session autosave) instead of once per
    edit — this is the amortization that keeps a burst of edits from
    re-serializing every snapshot each time."""
    global _UNDO_DIRTY
    if _UNDO_DIRTY:
        _UNDO_DIRTY = False
        persist_undo_stack()


def persist_undo_stack() -> None:
    """Atomically write the pending stack as `<NX_HOME>/undo.json` (0600).

    Best-effort: an unwritable home degrades to the pre-0.47 memory-only
    behavior; it must never break a tool call that is mid-flight. `indent` is
    deliberately off the persist path (PF-05): the file is machine-written and
    a 40 MB pretty-print costs write amplification for nothing."""
    global _UNDO_DIRTY
    _UNDO_DIRTY = False
    try:
        data = {"version": 1, "entries": st._UNDO_STACK}
        atomic_write_private(_undo_path(), f"{json.dumps(data)}\n")
    except Exception:
        pass


def _load_persisted() -> None:
    """Populate the in-memory stack from disk at startup (skipped by /reset)."""
    path = _undo_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries")
        if not isinstance(entries, list):
            return
        clean: list[Checkpoint] = []
        for e in entries:
            if not isinstance(e, dict) or e.get("tool") is None:
                continue
            files = e.get("files")
            if not isinstance(files, list):
                continue
            snaps = []
            for s in files:
                if (isinstance(s, dict) and isinstance(s.get("path"), str)
                        and isinstance(s.get("existed"), bool)
                        and (s.get("content") is None or isinstance(s.get("content"), str))):
                    snap: FileSnapshot = {"path": s["path"], "existed": s["existed"],
                                          "content": s["content"]}
                    # Old-format entries predate after_sha: keep them, and only
                    # carry the guard forward when the field is present.
                    if isinstance(s.get("after_sha"), str):
                        snap["after_sha"] = s["after_sha"]
                    snaps.append(snap)
            if snaps:
                clean.append({"tool": str(e["tool"]), "ts": float(e.get("ts") or 0.0),
                              "files": snaps})
        if clean:
            st._UNDO_STACK[:] = clean[-MAX_UNDO:]
    except (OSError, ValueError):
        pass  # missing/corrupt file — start empty


def clear_persisted() -> None:
    """Drop both the in-memory stack and its on-disk copy (/reset, /new)."""
    global _UNDO_DIRTY
    st._UNDO_STACK.clear()
    _UNDO_DIRTY = False
    try:
        os.unlink(_undo_path())
    except OSError:
        pass


def swap_session() -> None:
    """Re-key the undo store on an in-process session-id switch (/resume,
    /fork, /reset — 0.128.3). Persist whatever the OUTGOING session left
    unsaved under its own path, then clear the stack and load the incoming
    id's persisted stack. Without this, the in-memory stack survives the id
    swap: /undo in the new session reverts the old session's files and then
    persists those snapshots into the NEW id's undo file."""
    persist_undo_stack()  # best-effort write under the OLD _undo_path()
    st._UNDO_STACK.clear()
    _load_persisted()


def _restore_one(path: str, existed: bool, content: Optional[str]) -> None:
    """Restore one file atomically: temp file + os.replace for a write,
    os.remove for a re-delete. Raises OSError on failure."""
    if existed:
        atomic_write_private(path, content or "")
    elif os.path.exists(path):
        os.remove(path)


def undo_last() -> tuple[Optional[Checkpoint], str]:
    """Revert every file in the most recent checkpointed call.

    Atomic per file (temp + os.replace); a file whose restore fails is dropped
    from the entry and the entry itself is kept pending, so a partially failed
    undo never strands the rest of the checkpoint (0.47.0). Returns (entry, message)."""
    if not st._UNDO_STACK:
        return None, "nothing to undo"
    entry: Checkpoint = st._UNDO_STACK.pop()
    reverted: list[str] = []
    errors: list[str] = []
    for snap in entry["files"]:
        path = snap["path"]
        # Data-loss guard (see _content_sha): only revert a file that still
        # holds what the tool left. A stale/cross-session checkpoint, or any
        # change since the push, would otherwise clobber newer content while
        # the git index keeps it — the inverse-diff mirror state. Snapshots
        # with no after_sha are old persisted entries: restore as before.
        after = snap.get("after_sha")
        if after is not None and _content_sha(path) != after:
            errors.append(f"{path}: changed since checkpoint; not reverted "
                          f"(current content differs from what {entry['tool']} left)")
            continue
        try:
            _restore_one(path, snap["existed"], snap["content"])
            reverted.append(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    if errors and not reverted:
        # Nothing was restored — keep the entry pending for the next press.
        st._UNDO_STACK.append(entry)
        persist_undo_stack()
        return entry, "ERROR: undo failed: " + "; ".join(errors)
    if errors:
        entry["files"] = [s for s in entry["files"] if s["path"] not in
                          {p.split(":")[0] for p in errors}]
    if len(reverted) == 1:
        msg = f"reverted {entry['tool']} on {reverted[0]}"
    else:
        msg = f"reverted {entry['tool']} on {len(reverted)} file(s): {', '.join(reverted)}"
    if errors:
        msg += " (FAILED: " + "; ".join(errors) + ")"
    persist_undo_stack()
    return entry, msg


# --- the /undo fullscreen view (0.152.0) ---------------------------------------
# The same recipe as /diff's ReviewView, /context's ContextView and /memory's
# MemoryView: the view lives in the module that owns the data, renders
# exclusively through `_render.py`, and its ONLY mutation is the EXISTING
# `undo_last()` handed back to the fullscreen shell as a deferred
# `pending_action` (run AFTER the shell leaves the alternate screen, so the
# confirm prompt never paints into it). No new revert logic: `u` calls the
# exact same function bare `/undo` runs.

def _undo_paths_summary(files: list[FileSnapshot]) -> str:
    """The path column `/undo --list` prints: one path, or an N-file summary."""
    if len(files) == 1:
        return files[0]["path"]
    return f"{len(files)} files: " + ", ".join(f["path"] for f in files)


def _undo_when(ts: float | None) -> str:
    """Relative time, the SAME formatter `/undo --list` prints."""
    from ._sessions_cmd import _fmt_when
    return _fmt_when(ts)


class UndoView(View):
    """The `/undo --list` view: a browsable checkpoint history (most recent
    first, the exact rows the inline list prints), `enter` opens one
    checkpoint's per-file detail, `u` reverts the NEWEST checkpoint.

    Deliberately read-only apart from that one action: consuming the stack is
    strictly LIFO (`undo_last()` pops the top), so the view offers NO
    pick-N revert — browsing an older checkpoint is inspection, mutation is
    always "undo newest", same as pressing `/undo` at the prompt.
    """

    surface = "undo"
    title = "undo"

    def __init__(self) -> None:
        super().__init__()
        self.sel = 0
        self.mode = "list"        # "list" | "checkpoint"
        self.entries: list[Checkpoint] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-read the stack — most recent first, the exact rows
        `/undo --list` prints. Called on open AND after a deferred undo (the
        stack shrank; the next repaint must show it)."""
        self.entries = list_checkpoints()
        self.sel = min(self.sel, max(0, len(self.entries) - 1))
        self.mode = "list"

    def render(self, width: int, height: int) -> list[str]:
        from ._ansi import clip_display
        from ._keys import hint_line
        from ._render import RESET, panel, role
        self.sel = min(self.sel, max(0, len(self.entries) - 1))
        lines = [f"{role('ACCENT')}undo{RESET}{role('MUTED')} — "
                 f"{hint_line('undo')}{RESET}"]
        budget = max(1, height - len(lines) - 2)   # header line + box borders
        body: list[str] = []
        n = len(self.entries)
        if self.mode == "checkpoint" and n:
            cp = self.entries[min(self.sel, n - 1)]
            body.append(f"{role('ACCENT')}{cp['tool']}{RESET}"
                        f"{role('MUTED')}  checkpoint "
                        f"{n - min(self.sel, n - 1)} of {n}  "
                        f"({_undo_when(cp.get('ts'))}){RESET}")
            for snap in cp["files"]:
                verb = "restore" if snap["existed"] else "delete"
                size = len(snap.get("content") or "")
                body.append(f"{role('MUTED')}{verb:<8}{RESET} "
                            + clip_display(snap["path"], width - 16)
                            + (role('MUTED') + f"  {size} B" + RESET
                               if snap["existed"] else ""))
            lines.extend(panel(f"{cp['tool']} · {len(cp['files'])} file(s)",
                               [ln[:width] for ln in body],
                               footer=f"newest: {cp['tool']} "
                                      f"({_undo_when(cp.get('ts'))})"))
            if self.notice:
                lines.append(f"{role('WARNING')}{self.notice[:width]}{RESET}")
            return lines[:height]
        if not n:
            body.append(f"{role('MUTED')}  nothing to undo — /undo --list "
                        f"prints the same list inline{RESET}")
        else:
            shown = budget if n <= budget else max(1, budget - 1)
            start = max(0, min(self.sel - shown + 1, n - shown))
            stop = min(n, start + shown)
            for i in range(start, stop):
                cp = self.entries[i]
                marker = "›" if i == self.sel else " "
                text = clip_display(
                    f"{i + 1:>2}  {cp['tool']:<11} "
                    f"{_undo_paths_summary(cp['files'])}  "
                    f"({_undo_when(cp.get('ts'))})",
                    width - 3)
                r = "ACCENT" if i == self.sel else "MUTED"
                mcol = "ACCENT" if i == self.sel else "MUTED"
                body.append(f"{role(mcol)}{marker}{RESET} "
                            f"{role(r)}{text}{RESET}")
            hidden = n - (stop - start)
            if hidden > 0:
                body.append(f"{role('MUTED')}  … {hidden} more checkpoint"
                            f"{'s' if hidden != 1 else ''} (j/k scroll){RESET}")
        lines.extend(panel(f"undo checkpoints · {n}",
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
        last = max(0, len(self.entries) - 1)
        if self.mode == "checkpoint":
            if key in ("enter", "tab", "esc", "left"):
                self.mode = "list"
                return "repaint"
            if key == "u":
                return self._undo(shell)
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
            if self.entries:
                self.mode = "checkpoint"
            return "repaint"
        if key == "u":
            return self._undo(shell)
        if key == "r":
            self.refresh()
            return "repaint"
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"

    def _undo(self, shell: Any) -> str:
        """Queue the EXISTING `undo_last()` (what bare `/undo` runs) as the
        deferred action — confirm via the shell prompt first, the same shape
        as MemoryView's delete. Cancelled ask -> notice, nothing queued."""
        if not self.entries:
            self.notice = "nothing to undo"
            return "repaint"
        newest = self.entries[0]   # list order is most-recent-first
        if shell.ask(f"undo newest checkpoint ({newest['tool']}, "
                     f"{_undo_paths_summary(newest['files'])})? y/n ›") is None:
            self.notice = "undo cancelled"
            return "repaint"
        view = self

        def _act() -> None:
            from ._render import commit_line
            entry, msg = undo_last()
            commit_line(f"  ↳ {msg}")
            view.refresh()

        self.pending_action = _act
        return "run"
