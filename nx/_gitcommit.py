"""git_commit: the staged-diff review tool (ROADMAP P3-6, 0.51.0).

git add/commit previously existed only as classifier prose — i.e. as raw
`run_bash` reviewed as nothing. This tool makes the commit itself the unit of
review: the model stages paths explicitly, sees the exact staged diff in its
own approval `preview`, and the same text is scanned for secrets before any
confirmation round-trip. Command shape is fixed (no free-text shell — that is
run_bash's job); the user reviews a diff, not an argv.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

from . import _state as st

# git add/commit are slower than a read-only `git status` but still bounded;
# mirrors _run_git's 15s read ceiling with room for a big index refresh.
_DEFAULT_TIMEOUT = 60


def _git(repo: str, args: list[str], timeout: int) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(["git", "-C", repo, *args],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"ERROR: git {' '.join(args[:2])} timed out after {timeout}s"
    except OSError as e:
        return None, f"ERROR: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        return None, (f"ERROR: {(proc.stderr or proc.stdout).strip() or 'git command failed'}")
    return proc.stdout, None


def git_commit(files: str = "", message: str = "", path: str = "", **_: Any) -> str:
    """Stage `files` (space-separated, or "." for every change), show the staged
    diff as the approval preview, and commit with `message`.

    The staged diff — not the command line — is what the user reviews: the
    preview is built from `git diff --staged` AFTER staging, so it always shows
    exactly what the commit will contain. The same text is scanned for
    secret-shaped strings (keys, tokens, PEM blocks); a hit refuses the commit
    before any prompt.

    Args:
      files: files to stage, space-separated (pass "." to stage every change).
      message: commit message (first line; kept verbatim).
      path: repo/directory to commit in (default: the project root).
    """
    from ._bash_jobs import _confirm  # lazy: avoid an import cycle at module load
    repo = (path or "").strip() or st.PROJECT_ROOT or "."
    if not os.path.isdir(os.path.join(repo, ".git")):
        return (f"ERROR: git_commit: {repo} is not a git repository — run inside "
                "the repo (pass `path` if the project root differs)")
    if not message.strip():
        return "ERROR: git_commit: a commit `message` is required"
    if not files.strip():
        return ("ERROR: git_commit: nothing to stage — pass `files` (space-separated "
                "paths, or \".\" for every change); use git_diff first to review")
    timeout = _env_timeout()
    file_list = files.split()
    out, err = _git(repo, ["add", "--", *file_list], timeout)
    if err:
        return err
    staged, err = _git(repo, ["diff", "--staged"], timeout)
    if err:
        return err
    if not (staged or "").strip():
        return ("[git_commit] nothing staged — every path in `files` was already "
                "committed, unchanged, or ignored; nothing was committed")
    from ._bash_jobs import _contains_secret
    hit = _contains_secret(staged)
    if hit:
        return (f"REFUSED: the staged diff contains a secret-shaped string "
                f"({hit}…) — unstage it (git restore --staged), remove the secret, "
                "and commit again. Nothing was committed.")
    short = message.strip().splitlines()[0][:120]
    preview = _preview(staged, short, repo)
    action = f"git commit: {short}"
    if not _confirm(action, preview=preview):
        # Denied AFTER staging: leave the index as the model set it (the diff is
        # still inspectable via git_diff staged=true) but commit nothing.
        return ("DENIED by user — commit not made; the staged changes are still "
                "in the index (git_diff staged=true to review, `git reset` to unstage)")
    out, err = _git(repo, ["commit", "-m", message.strip()], timeout)
    if err:
        return err
    return (f"[git_commit] {short}\n{(out or '').strip()}")


def _env_timeout() -> int:
    from ._env import _env_int
    return _env_int("NX_GITCOMMIT_TIMEOUT", _DEFAULT_TIMEOUT)


def _preview(staged: str, short: str, repo: str) -> str:
    """The approval preview: staged diff + a one-line commit header."""
    from ._ansi import DIM, RESET
    lines = staged.splitlines()
    if len(lines) > 200:
        lines = lines[:200] + [f"... [{len(lines) - 200} more diff lines]"]
    head = (f"{DIM}git commit ({repo}): {short}{RESET}")
    return head + "\n" + "\n".join(lines)
