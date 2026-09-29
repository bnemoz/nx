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
import tempfile
from typing import Any

from . import _state as st

# git add/commit are slower than a read-only `git status` but still bounded;
# mirrors _run_git's 15s read ceiling with room for a big index refresh.
_DEFAULT_TIMEOUT = 60


def _git(repo: str, args: list[str], timeout: int,
         env: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(["git", "-C", repo, *args],
                              capture_output=True, text=True, timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired:
        return None, f"ERROR: git {' '.join(args[:2])} timed out after {timeout}s"
    except OSError as e:
        return None, f"ERROR: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        return None, (f"ERROR: {(proc.stderr or proc.stdout).strip() or 'git command failed'}")
    return proc.stdout, None


def _stage_index(repo: str) -> str:
    """A private temp GIT_INDEX_FILE seeded from the repo's real index.

    Staging runs against this copy, so the preview/secret-scan/denied-commit
    path never touches the shared .git/index: a concurrent nx session (or the
    user's own git commands) can't have its WIP swept into our staging, and a
    denial leaves THEIR index untouched. The real index is only ever written
    by the final `git commit` itself, which reads the tree from this index.
    """
    fd, idx = tempfile.mkstemp(prefix=".nx-index-", dir=os.path.join(repo, ".git"))
    os.close(fd)
    env_seed = dict(os.environ, GIT_INDEX_FILE=idx)
    try:
        subprocess.run(["git", "-C", repo, "read-tree", "HEAD"],
                       env=env_seed, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass  # fresh empty index still stages correctly below
    return idx


def _git_env(index_file: str) -> dict[str, str]:
    return dict(os.environ, GIT_INDEX_FILE=index_file)


def git_commit(files: str = "", message: str = "", path: str = "", **_: Any) -> str:
    """Stage `files` (space-separated, or "." for every change), show the staged
    diff as the approval preview, and commit with `message`.

    The staged diff — not the command line — is what the user reviews: the
    preview is built from `git diff --staged` AFTER staging, so it always shows
    exactly what the commit will contain. The same text is scanned for
    secret-shaped strings (keys, tokens, PEM blocks); a hit refuses the commit
    before any prompt.

    Staging happens in a private temp index (GIT_INDEX_FILE), not the shared
    .git/index: the preview can never include another session's (or the
    user's) already-staged work, and a denial/secret-refusal leaves the
    shared index exactly as it was. Only the final `git commit` (which reads
    the tree from the private index) touches the repo.

    Args:
      files: files to stage, space-separated (pass "." to stage every change).
      message: commit message (first line; kept verbatim).
      path: repo/directory to commit in (default: the project root).
    """
    repo = (path or "").strip() or st.PROJECT_ROOT or "."
    # Accept linked worktrees: their .git is a FILE pointing at the real
    # gitdir, not a directory — isdir(join(repo, ".git")) rejected them and
    # pushed write-tier worktree agents to raw `git` via run_bash (0.103.15).
    if not (os.path.isdir(os.path.join(repo, ".git"))
            or os.path.isfile(os.path.join(repo, ".git"))):
        return (f"ERROR: git_commit: {repo} is not a git repository — run inside "
                "the repo (pass `path` if the project root differs)")
    if not message.strip():
        return "ERROR: git_commit: a commit `message` is required"
    if not files.strip():
        return ("ERROR: git_commit: nothing to stage — pass `files` (space-separated "
                "paths, or \".\" for every change); use git_diff first to review")
    timeout = _env_timeout()
    file_list = files.split()
    idx = _stage_index(repo)
    try:
        return _git_commit_with_index(repo, file_list, message, timeout, idx)
    finally:
        try:
            os.unlink(idx)
        except OSError:
            pass


def _git_commit_with_index(repo: str, file_list: list[str], message: str,
                           timeout: int, idx: str) -> str:
    from ._bash_jobs import _confirm  # lazy: avoid an import cycle at module load
    env = _git_env(idx)
    out, err = _git(repo, ["add", "--", *file_list], timeout, env=env)
    if err:
        return err
    staged, err = _git(repo, ["diff", "--staged"], timeout, env=env)
    if err:
        return err
    if not (staged or "").strip():
        return ("[git_commit] nothing staged — every path in `files` was already "
                "committed, unchanged, or ignored; nothing was committed")
    from ._bash_jobs import _contains_secret
    hit = _contains_secret(staged)
    if hit:
        return (f"REFUSED: the staged diff contains a secret-shaped string "
                f"({hit}…) — remove the secret and commit again. Nothing was "
                "committed and nothing was staged in the shared index.")
    short = message.strip().splitlines()[0][:120]
    preview = _preview(staged, short, repo)
    action = f"git commit: {short}"
    if not _confirm(action, preview=preview):
        # Denied AFTER staging: staging happened in our private index, so the
        # SHARED index is untouched — nothing to unstage, nothing of another
        # session's work was ever swept in.
        return ("DENIED by user — commit not made; the working tree is "
                "unchanged (git_diff staged=true reviews the shared index)")
    out, err = _git(repo, ["commit", "-m", message.strip()], timeout, env=env)
    if err:
        return err
    _sync_shared_index(repo, file_list, timeout)
    return (f"[git_commit] {short}\n{(out or '').strip()}")


def _sync_shared_index(repo: str, file_list: list[str], timeout: int) -> None:
    """Re-cacheinfo the committed paths into the SHARED index (best-effort).

    The commit read its tree from the private index, so the shared index still
    holds the pre-commit blob for those paths — `git status`/`git diff --staged`
    would show a phantom staged change after a successful commit. Rewriting the
    entries to HEAD's blobs (the commit just made) makes the shared index
    consistent again without touching any other path another session staged.
    """
    # -r: without it ls-tree lists only top-level tree entries, so paths in
    # subdirectories (nx/, tests/) were absent from by_path and the sync
    # silently no-oped — the persistent phantom MM.
    out, err = _git(repo, ["ls-tree", "-r", "-z", "HEAD", "--"], timeout)
    if err or not out:
        return
    by_path: dict[str, tuple[str, str]] = {}
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, _, name = rec.partition("\t")
        parts = meta.split()
        if len(parts) >= 3:
            by_path[name] = (parts[0], parts[2])
    wanted = set(file_list) if file_list != ["."] else set(by_path)
    paths = sorted(wanted & set(by_path))
    # Re-hash the actual worktree blobs instead of trusting the index: after a
    # commit the shared-index entry for a committed path is stale. Rewrite it
    # to the WORKTREE blob — worktree == HEAD gives the clean committed entry;
    # worktree != HEAD (live WIP, the reported residual bug) keeps the change
    # as staged without a phantom. The worktree file itself is never touched.
    wt: dict[str, str] = {}
    guarded = True
    if paths:
        hout, herr = _git(repo, ["hash-object", "--", *paths], timeout)
        if herr or not hout:
            guarded = False  # can't verify; sync every path as before
        else:
            hashes = hout.split()
            # A short/misaligned hash list must NOT fall through to the
            # positional zip: writing HEAD's blob for a path whose worktree
            # file differs stages the WIP wrong — the exact MM signature.
            # Bail and leave the shared index for the normal add/commit path.
            if len(hashes) != len(paths):
                return
            for p, h in zip(paths, hashes):
                wt[p] = h
    specs: list[str] = []
    for p in paths:
        mode, blob = by_path[p]
        # Worktree differs from HEAD: write the worktree blob (WIP stays
        # staged), not HEAD's (which would unstage it).
        specs.append(f"{mode},{wt.get(p, blob) if guarded else blob},{p}")
    if specs:
        # ONE --cacheinfo per spec: git's parser consumes exactly one
        # <mode>,<blob>,<path> triple per flag; extra specs after a single
        # flag parse as positional paths and abort the whole update-index,
        # leaving the shared index stale (phantom MM after every multi-file
        # commit — 0.103.8).
        args: list[str] = ["update-index", "--add"]
        for spec in specs:
            args.append("--cacheinfo")
            args.append(spec)
        _git(repo, args, timeout)


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
