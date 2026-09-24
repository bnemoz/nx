"""Per-agent git worktree isolation (ROADMAP P3-12, 0.97.0).

A background write-tier subagent shares the parent's working tree, so a
headless swarm (`nx --yolo` with NX_SUBAGENT_WRITE=1) has its implementers
trampling each other's files and racing the parent's own edits. The fix: each
work-tier agent gets its OWN git worktree — `git worktree add <tmp>/nx-wt-<job>
<base>`, created detached (no branch-name collisions between concurrent
agents, no churn in `git branch` output), under the session home so the write
fence admits it.

Isolation contract:
- write_file/edit_file/multi_edit/sed_edit: the write-tier wrapper rewrites
  relative paths into the worktree before the real body runs (absolute paths
  INSIDE the worktree pass through; absolute paths outside are scope-gated
  anyway). The agent's model never sees the worktree path — it edits
  `nx/foo.py` and lands in the worktree.
- run_tests: `targets` are rewritten the same way; the run's cwd is the
  worktree, so the suite tests the agent's own edits, not the parent's tree
  (this was the swarm's worst race: two agents running pytest in one tree).
- git_commit: the `path` (repo) argument defaults to the worktree, so the
  agent's commits land on the worktree's detached HEAD. Nothing is pushed —
  the agent's report names the worktree, and the parent decides whether to
  merge/cherry-pick.

Reads (read_file/search_files/list_dir/run_bash) are NOT rewritten — an
implement agent often must read files outside its scope (references,
tests it must keep green). Under yolo + worktrees the write fence still
constrains writes to the session home; the worktree lives under it, so
out-of-worktree writes are refused by the fence regardless.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from . import _state as st
from ._env import _env_int


def _worktree_enabled() -> bool:
    """NX_SUBAGENT_WORKTREE=1 gives write-tier subagents their own git
    worktree. Reads the live attribute first (tests/REPL), env fallback."""
    import nx as _m
    if "NX_SUBAGENT_WORKTREE" in vars(_m):
        return bool(int(_m.NX_SUBAGENT_WORKTREE or 0))
    return bool(_env_int("NX_SUBAGENT_WORKTREE", 0))


def _parent_worktree_roots() -> set[str]:
    """Real paths of the parent repo + all its linked worktrees — paths an
    agent must NOT be pointed into as ITS worktree (a worktree of a worktree
    fails, and nesting would silently share state)."""
    roots = set()
    base = st.PROJECT_ROOT or os.getcwd()
    roots.add(os.path.realpath(base))
    for cmd in (["git", "-C", base, "rev-parse", "--path-format=absolute",
                 "--git-common-dir"],
                ["git", "-C", base, "worktree", "list", "--porcelain"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0:
            continue
        for line in (r.stdout or "").splitlines():
            if " " not in line:
                continue
            head, _, rest = line.partition(" ")
            if head == "worktree" or head not in (
                    "gitdir", "HEAD", "branch", "bare", "detached", "locked",
                    "ref"):
                roots.add(os.path.realpath(rest))
    return {r for r in roots if os.path.isdir(r)}


def create_worktree(job_id: str) -> tuple[str | None, str | None]:
    """Create a detached worktree for `job_id` from the parent repo's HEAD.
    Returns (worktree_path, None) or (None, error_message)."""
    if not st.PROJECT_ROOT_IS_GIT:
        return None, "no git repo at the project root"
    base = st.PROJECT_ROOT or os.getcwd()
    roots = _parent_worktree_roots()
    if not roots:
        return None, "could not resolve the parent repo's worktree layout"
    # Where the worktree lives: under the session home (inside the write
    # fence — the agent's writes must be allowed without extra fence roots),
    # named by job id so /bg and reports can point at it unambiguously.
    home = st.SESSION_HOME or os.path.expanduser("~/.nx")
    root = os.path.join(home, "worktrees")
    os.makedirs(root, exist_ok=True)
    wt = os.path.realpath(os.path.join(root, f"nx-wt-{job_id}"))
    if os.path.exists(wt):
        return None, f"worktree path already exists: {wt}"
    for r in roots:
        if wt == r or wt.startswith(r + os.sep):
            return None, "worktree path collides with an existing worktree"
    r = subprocess.run(
        ["git", "-C", base, "worktree", "add", "--detach", wt, "HEAD"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout or "git worktree add failed").strip()
    return wt, None


def remove_worktree(path: str) -> None:
    """Best-effort cleanup after the agent's loop ends (the parent may still
    want the tree, so only the registry link is dropped by callers)."""
    if not path or not os.path.isdir(path):
        return
    try:
        subprocess.run(["git", "worktree", "remove", "--force", path],
                       capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    # If the tree was already gone (or rmtree won the race), `git worktree
    # remove` fails and the registration lingers as "prunable" — sweep it so
    # `git worktree list` in the parent repo stays honest.
    base = st.PROJECT_ROOT
    if base:
        try:
            subprocess.run(["git", "-C", base, "worktree", "prune"],
                           capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


def rewrite_path(path: str, worktree: str) -> str:
    """Map an agent-relative path into its worktree.

    Relative (or ~) → joined onto the worktree. Absolute INSIDE the worktree →
    unchanged. Absolute outside → unchanged (the scope gate / write fence
    refuse it)."""
    if not path or not worktree:
        return path
    p = os.path.expanduser(path)
    if os.path.isabs(p):
        rp = os.path.realpath(p)
        return path if rp == worktree or rp.startswith(worktree + os.sep) else path
    return os.path.join(worktree, p)


def in_worktree(path: str, worktree: str) -> bool:
    """True when `path` resolves inside the agent's worktree."""
    if not worktree:
        return False
    rp = os.path.realpath(os.path.expanduser(path))
    return rp == worktree or rp.startswith(worktree + os.sep)
