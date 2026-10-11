"""Full-screen working-tree diff review — `/review` (UIUX_SPEC Phase D, 0.137.0).

Three pieces, one file:

* DATA — `working_tree_files()` / `file_diff()` shell out to git read-only
  (`status --porcelain`, `diff --numstat`, `show HEAD:<path>`), so the view has
  no state of its own and nothing here can mutate the repository;
* VIEW — `ReviewView` renders through `_render.py` components only and hands
  the commit back to the fullscreen shell as a deferred action (the shell
  leaves the alternate screen before `git_commit` runs, so the Y/n approval
  prompt is never painted into the alternate buffer);
* INLINE — `_inline()` is the off-tty / `NX_FULLSCREEN=0` path: a plain,
  deterministic list of the same rows, byte-for-byte testable.

The COMMIT path is the existing reviewed-commit machinery, unchanged: staging
into a private index, the secret scan, the Y/n gate, the HEAD-moved preflight
(`_gitcommit.git_commit`). A view gets no new mutation path.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

from . import _state as st
from ._ansi import RESET
from ._fullscreen import View
from ._keys import hint_line
from ._render import diff as render_diff, panel, role

# Same ceiling the git_commit preview uses for its staged diff.
_MAX_DIFF_LINES = 400
_GIT_TIMEOUT = 30

# status code -> semantic role (never a raw palette hue: the Phase B grep
# guard enforces it, and a theme can recolor the roles independently).
_CODE_ROLES = {
    "A": "SUCCESS",   # added
    "?": "SUCCESS",   # untracked
    "D": "ERROR",     # deleted
    "M": "ACCENT",    # modified
    "R": "ACCENT",
    "C": "ACCENT",
    "U": "WARNING",   # unmerged
}


def _repo() -> str:
    return st.PROJECT_ROOT or "."


def _git(repo: str, args: list[str]) -> Optional[str]:
    """Read-only git call. None when git is missing, the call times out, or it
    exits non-zero (a non-repo, an unborn HEAD) — callers degrade, never raise."""
    try:
        proc = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                              text=True, timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def working_tree_files(repo: Optional[str] = None) -> list[dict[str, Any]]:
    """Every changed path in the working tree, with its `git diff --numstat`
    line counts: `[{"path", "code", "added", "removed"}]`.

    Untracked files count as pure additions (their line count), which is what
    the numstat of an `--no-index` diff against /dev/null would say — computed
    locally so a 5k-file tree costs one status pass, not one process per file.
    """
    repo = repo or _repo()
    out = _git(repo, ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    if out is None:
        return []
    files: list[dict[str, Any]] = []
    records = [r for r in out.split("\0") if r]
    i = 0
    while i < len(records):
        rec = records[i]
        i += 1
        if len(rec) < 4:
            continue
        code = rec[:2]
        path = rec[3:]
        if code[0] in ("R", "C"):   # rename/copy: the OLD path follows in -z
            i += 1
        files.append({"path": path, "code": code.strip() or "??",
                      "added": 0, "removed": 0})
    counts = _numstat(repo)
    for f in files:
        added, removed = counts.get(f["path"], (0, 0))
        if added or removed:
            f["added"], f["removed"] = added, removed
        elif f["code"] == "??":
            f["added"] = _count_lines(os.path.join(repo, f["path"]))
    return files


def _numstat(repo: str) -> dict[str, tuple[int, int]]:
    out = _git(repo, ["diff", "--numstat", "HEAD", "--"])
    table: dict[str, tuple[int, int]] = {}
    for line in (out or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, r, path = parts
        try:
            table[path] = (int(a), int(r))
        except ValueError:
            continue          # binary files report "-"; leave them at 0/0
    return table


def _count_lines(path: str, cap: int = 100000) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            n = 0
            for _ in f:
                n += 1
                if n >= cap:
                    break
            return n
    except OSError:
        return 0


def file_diff(path: str, repo: Optional[str] = None) -> list[str]:
    """The rendered diff of one path vs HEAD (an added file diffs against an
    empty old side; a deleted one against its HEAD body). Clipped to the same
    ceiling the commit preview uses."""
    repo = repo or _repo()
    new = _read_worktree(os.path.join(repo, path))
    old = _git(repo, ["show", f"HEAD:{path}"]) or ""
    text = render_diff(old, new)
    if not text:
        return []
    lines = text.splitlines()
    if len(lines) > _MAX_DIFF_LINES:
        lines = lines[:_MAX_DIFF_LINES] + [
            f"… [{len(lines) - _MAX_DIFF_LINES} more diff lines]"]
    return lines


def _read_worktree(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def commit_files(files: list[str], message: str, repo: Optional[str] = None) -> str:
    """Commit `files` with `message` through the existing reviewed-commit tool.
    Returns its report (a refusal text is a normal return value there)."""
    from ._gitcommit import git_commit
    return git_commit(files=" ".join(files), message=message, path=repo or _repo())


def _row(f: dict[str, Any], selected: bool, width: int) -> str:
    code = str(f.get("code") or "??")
    role_name = _CODE_ROLES.get(code[0], "MUTED")
    marker = "›" if selected else " "
    stat = f"+{f.get('added', 0)} -{f.get('removed', 0)}"
    head = (f"{role('ACCENT')}{marker}{RESET} "
            f"{role(role_name)}{code:>2}{RESET} "
            f"{f['path']}")
    pad = max(0, width - len(f["path"]) - 12)
    return f"{head}{' ' * pad}{role('MUTED')}{stat}{RESET}"


def render_review_files(files: list[dict[str, Any]], sel: int, width: int,
                        *, repo: str = "", notice: str = "",
                        mode: str = "files", diff_lines: Optional[list[str]] = None,
                        offset: int = 0) -> list[str]:
    """The whole frame as ready-to-write lines — the ONE renderer both the
    fullscreen view and the inline fallback use, so the two can't drift."""
    n = len(files)
    if mode == "diff" and files:
        title = f"diff · {files[sel]['path']}"
        body = list(diff_lines or [])[offset:]
        footer = f"{len(diff_lines or [])} diff line(s) · tab back to the list"
    else:
        title = f"diff · working tree · {n} file(s) vs HEAD"
        body = [_row(f, i == sel, width) for i, f in enumerate(files)]
        footer = f"{sel + 1}/{n} · enter diff · a commit · r refresh"
    lines = [f"{role('ACCENT')}diff{RESET}{role('MUTED')} — "
             f"{hint_line('diff')}{RESET}"]
    if not files:
        lines.append(f"{role('MUTED')}  working tree clean — nothing to "
                     f"review{RESET}")
    else:
        lines.extend(panel(title, [ln[:width] for ln in body], footer=footer))
    if notice:
        lines.append(f"{role('WARNING')}{notice[:width]}{RESET}")
    return lines


def review_lines(repo: Optional[str] = None, path: str = "",
                 width: int = 100) -> list[str]:
    """The INLINE fallback body for `/review` off a tty (or NX_FULLSCREEN=0).

    With `path`: the list plus that file's rendered diff. Without: just the
    list. Deterministic and side-effect free — this is the output tests pin.
    """
    repo = repo or _repo()
    files = working_tree_files(repo)
    if not files:
        return [f"  working tree clean — nothing to review (vs HEAD, {repo})"]
    # The fallback prints the same rows the full-screen list shows, stripped of
    # the box frame: one line per changed file, then the tip.
    out = [f"{f['code']:>2}  {f['path']}  +{f['added']} -{f['removed']}"
           for f in files]
    out.append(f"  {len(files)} file(s) changed vs HEAD — /diff opens the"
               f" full-screen browser, /diff <path> diffs one file")
    if path:
        out.append(f"  ── {path} ──")
        out.extend(file_diff(path, repo) or [f"  (no textual diff for {path})"])
    return out


def _inline(state: Any = None, args: str = "") -> None:
    """Print `review_lines()` through the shared render path (SPEC_LIVEENGINE:
    the console is owned by `_render`, not by raw stdout writes)."""
    from ._render import commit_line
    for line in review_lines(path=args.strip().split()[0] if args.strip() else ""):
        commit_line(line)


class ReviewView(View):
    """The alternate-screen /diff view: file list → per-file diff, with the
    `a` flow handing a commit back to the fullscreen shell."""

    surface = "diff"
    title = "diff"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.repo = _repo()
        self.sel = 0
        self.offset = 0
        self.mode = "files"
        self.files: list[dict[str, Any]] = []
        self.diff_lines: list[str] = []
        self.notice = ""
        self.refresh()

    def refresh(self) -> None:
        self.files = working_tree_files(self.repo)
        self.sel = min(self.sel, max(0, len(self.files) - 1))
        self.diff_lines = []
        self.mode = "files"

    def _open_diff(self) -> None:
        if not self.files:
            return
        self.diff_lines = file_diff(self.files[self.sel]["path"], self.repo)
        self.offset = 0
        self.mode = "diff"

    def render(self, width: int, height: int) -> list[str]:
        lines = render_review_files(
            self.files, self.sel, width, repo=self.repo, notice=self.notice,
            mode=self.mode, diff_lines=self.diff_lines, offset=self.offset)
        return lines[:height]

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        if key == "j/k":
            # `_keys` advertises the pair as ONE hint label ("j/k move"); the
            # label is not a key, and the two real keys are handled below.
            return "repaint"
        if key in ("j", "down"):
            if self.mode == "diff":
                self.offset += 1
            else:
                self.sel = min(max(0, len(self.files) - 1), self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            if self.mode == "diff":
                self.offset = max(0, self.offset - 1)
            else:
                self.sel = max(0, self.sel - 1)
            return "repaint"
        if key in ("enter", "tab"):
            if self.mode == "diff":
                self.mode = "files"
            else:
                self._open_diff()
            return "repaint"
        if key == "r":
            self.refresh()
            return "repaint"
        if key == "a":
            return self._commit(shell)
        if key in ("esc", "q", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"

    def _commit(self, shell: Any) -> str:
        if not self.files:
            self.notice = "nothing to commit — the working tree is clean"
            return "repaint"
        files = [f["path"] for f in self.files]
        message = shell.ask(f"commit {len(files)} file(s) [{files[0]}…] ›")
        if not message:
            self.notice = "commit cancelled"
            return "repaint"
        repo = self.repo

        def _act() -> None:
            report = commit_files(files, message, repo)
            from ._render import commit_line
            for line in report.splitlines():
                commit_line(line)

        self.pending_action = _act
        return "run"


__all__ = ["ReviewView", "working_tree_files", "file_diff", "commit_files",
           "render_review_files", "review_lines"]
