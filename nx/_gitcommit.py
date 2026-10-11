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


def _stage_index(repo: str) -> tuple[str, str | None]:
    """A private temp GIT_INDEX_FILE seeded from the repo's real index.

    Staging runs against this copy, so the preview/secret-scan/denied-commit
    path never touches the shared .git/index: a concurrent nx session (or the
    user's own git commands) can't have its WIP swept into our staging, and a
    denial leaves THEIR index untouched. The real index is only ever written
    by the final `git commit` itself, which reads the tree from this index.

    Returns (temp index path, HEAD hash at seed time). The hash is our
    preflight anchor: if HEAD moves before we commit, the commit would
    silently parent onto someone else's new tip (see _head_moved below).
    """
    head = _head_hash(repo)
    fd, idx = tempfile.mkstemp(prefix=".nx-index-", dir=_gitdir(repo))
    os.close(fd)
    env_seed = dict(os.environ, GIT_INDEX_FILE=idx)
    try:
        subprocess.run(["git", "-C", repo, "read-tree", "HEAD"],
                       env=env_seed, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass  # fresh empty index still stages correctly below
    return idx, head


def _gitdir(repo: str) -> str:
    """The real git directory for `repo`, which may be a LINKED WORKTREE.

    In a linked worktree (every `nx` sub-agent worktree, `nx/_worktree.py`)
    `.git` is a FILE holding `gitdir: <path>`, not a directory — the pre-0.137.0
    code passed `<repo>/.git` straight to `mkstemp`, which fails with
    NotADirectoryError and made `git_commit` unusable in exactly the worktrees nx
    itself creates. `git rev-parse --git-dir` resolves both shapes; an
    unresolvable/odd repo falls back to `<repo>/.git` (the common case, and the
    one that already worked).
    """
    try:
        proc = subprocess.run(["git", "-C", repo, "rev-parse", "--absolute-git-dir"],
                              capture_output=True, text=True, timeout=15)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.path.join(repo, ".git")


def _head_hash(repo: str) -> str | None:
    """Current HEAD commit hash, or None (unborn HEAD / git unavailable)."""
    out, err = _git(repo, ["rev-parse", "HEAD"], 15)
    if err:
        return None
    return (out or "").strip() or None


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
    idx, head = _stage_index(repo)
    try:
        return _git_commit_with_index(repo, file_list, message, timeout, idx, head)
    finally:
        try:
            os.unlink(idx)
        except OSError:
            pass


def _git_commit_with_index(repo: str, file_list: list[str], message: str,
                           timeout: int, idx: str, head: str | None = None) -> str:
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
    # Preflight (HEAD-moved guard): another session (or the user) may have
    # committed between _stage_index's read-tree and now. Committing on top of
    # the old anchor would silently parent onto the new tip — refuse instead
    # and name what landed, so the model re-inspects and retries deliberately.
    # `git commit` below still reads the tree from our private index, so a
    # refusal leaves the shared index untouched (the caller's unlink cleans up).
    cur = _head_hash(repo)
    if head is not None and cur is not None and cur != head:
        return _head_moved(repo, head, cur, timeout)
    out, err = _git(repo, ["commit", "-m", message.strip()], timeout, env=env)
    if err:
        return err
    _sync_shared_index(repo, file_list, timeout)
    return (f"[git_commit] {short}\n{(out or '').strip()}")


def _head_moved(repo: str, old: str, new: str, timeout: int) -> str:
    """Refusal text for a HEAD that moved between staging and commit.

    Names the commits that landed in old..new (brief), so the model can see
    whose work it would have parented onto and re-inspect/retry.
    """
    out, err = _git(repo, ["log", "--oneline", f"{old}..{new}"], timeout)
    landed = ""
    if not err and (out or "").strip():
        lines = out.strip().splitlines()
        brief = "; ".join(lines[:5])
        if len(lines) > 5:
            brief += f"; … (+{len(lines) - 5} more)"
        landed = f" — landed here: {brief}"
    return (f"REFUSED: HEAD moved from {old[:12]} to {new[:12]} while you were "
            f"staging{landed}. Another session committed first; nothing was "
            "committed and the shared index is untouched. Re-inspect (git_log, "
            "git_diff) and retry the commit against the new HEAD.")


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
    if paths:
        hout, herr = _git(repo, ["hash-object", "--", *paths], timeout)
        if herr or not hout:
            # Can't verify ANY worktree blob: the old unguarded fallback
            # wrote HEAD's blob for every path, staging HEAD content over a
            # worktree file that differs — the exact MM signature this sync
            # exists to remove. Skip the whole sync instead; the shared index
            # keeps whatever another session staged for the normal add path.
            return
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
    wip: list[str] = []
    for p in paths:
        mode, blob = by_path[p]
        # No verified worktree blob for this path: writing HEAD's blob would
        # stage HEAD content over a differing worktree file (the MM
        # signature again), so leave the path's index entry alone entirely.
        whash = wt.get(p)
        if whash is None:
            continue
        # Worktree differs from HEAD: write the worktree blob (WIP stays
        # staged), not HEAD's (which would unstage it).
        if whash != blob:
            wip.append(p)
        specs.append(f"{mode},{whash},{p}")
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
    if wip:
        _claim_synced_staged(repo, wip)


def _claim_synced_staged(repo: str, paths: list[str]) -> None:
    """Record the paths the sync above LEAVES staged (worktree != HEAD) as
    THIS session's own staged work, so `_clobber.git_index_guard` no longer
    prompts this session about its own WIP. Paths the sync left clean are NOT
    claimed: a later staged entry for one of those came from someone else and
    must still prompt. Best-effort — a failed claim only costs a prompt."""
    try:
        from ._stagedclaim import mark_staged
        mark_staged(os.path.join(repo, p) for p in paths)
    except Exception:  # noqa: BLE001 - bookkeeping must never break a commit
        pass


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
