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
import time

from . import _state as st
from ._env import _env_int


def _worktree_enabled() -> bool:
    """NX_SUBAGENT_WORKTREE=1 gives write-tier subagents their own git
    worktree. Reads the live attribute first (tests/REPL), env fallback."""
    import nx as _m
    if "NX_SUBAGENT_WORKTREE" in vars(_m):
        return bool(int(_m.NX_SUBAGENT_WORKTREE or 0))
    return bool(_env_int("NX_SUBAGENT_WORKTREE", 0))


# --- per-SESSION worktrees (0.103.0, session-isolation phase 3) ----------------
# NX_SESSION_WORKTREE=1 (or `nx --isolate`) puts the whole interactive session
# in its own worktree under SESSION_HOME/worktrees/nx-wt-session-<sid>,
# branched from the repo's HEAD. Two sessions in one repo then share NOTHING:
# separate trees, separate indexes, separate commits — merging is an explicit
# `git merge`/cherry-pick the user controls. st.SESSION_WORKTREE names this
# session's tree (None = normal shared-tree mode).

def _session_worktree_enabled() -> bool:
    import nx as _m
    if "NX_SESSION_WORKTREE" in vars(_m):
        return bool(int(_m.NX_SESSION_WORKTREE or 0))
    return bool(_env_int("NX_SESSION_WORKTREE", 0))


def create_session_worktree(session_id: str) -> tuple[str | None, str | None]:
    """Create the session's own worktree. Unlike create_worktree() this makes
    a BRANCH (nx/session-<sid>) off HEAD so the session's commits have a name,
    and it tolerates reuse: re-entering the same session id re-attaches to the
    existing tree instead of failing. Returns (path, None) or (None, error)."""
    if not st.PROJECT_ROOT_IS_GIT:
        return None, "no git repo at the project root"
    base = st.PROJECT_ROOT or os.getcwd()
    home = st.SESSION_HOME or os.path.expanduser("~/.nx")
    # Per-project slug dir (0.145.2): concurrent projects must not share
    # one flat worktrees/ store.
    import nx as _m
    root = os.path.join(home, "worktrees", _m.project_store_slug())
    os.makedirs(root, exist_ok=True)
    wt = os.path.realpath(os.path.join(root, f"nx-wt-session-{session_id}"))
    branch = f"nx/session-{session_id}"
    if os.path.isdir(wt):
        # Re-entering a session (crash recovery, deliberate resume): keep it.
        return wt, None
    r = subprocess.run(
        ["git", "-C", base, "worktree", "add", "-b", branch, wt, "HEAD"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        # A stale branch from an earlier crashed run: reuse its tip.
        r2 = subprocess.run(
            ["git", "-C", base, "worktree", "add", "--detach", wt, branch],
            capture_output=True, text=True, timeout=30)
        if r2.returncode != 0:
            return None, (r2.stderr or r2.stdout
                          or (r.stderr or r.stdout)
                          or "git worktree add failed").strip()
    return wt, None


def _session_branch_for(path: str) -> str:
    """The session branch a session worktree path implies, from the dir name
    create_session_worktree() gives it (`nx-wt-session-<sid>` ->
    `nx/session-<sid>`). "" when the name carries no session id.

    Needed because `git rev-parse --abbrev-ref HEAD` reports nothing for a
    worktree created by create_session_worktree's FALLBACK (`worktree add
    --detach <stale-branch>`): nothing is checked out there, yet the branch
    still exists in the parent repo and would otherwise leak forever."""
    name = os.path.basename(os.path.realpath(path).rstrip(os.sep))
    prefix = "nx-wt-session-"
    if name.startswith(prefix) and len(name) > len(prefix):
        return "nx/session-" + name[len(prefix):]
    return ""


def remove_session_worktree(path: str) -> None:
    """Best-effort cleanup of the session's tree AND its branch. Only called
    when the session ends with nothing to keep (the normal exit path asks
    first; never auto-run when the tree is dirty)."""
    if not path or not os.path.isdir(path):
        return
    try:
        dirty = subprocess.run(
            ["git", "-C", path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if dirty:
            return  # refuse to destroy work
    except (OSError, subprocess.TimeoutExpired):
        return
    base = st.PROJECT_ROOT
    # Read the branch BEFORE remove_worktree deletes the tree; the path-derived
    # name covers the detached-fallback case where rev-parse reports nothing.
    branch = _branch_of(path) or _session_branch_for(path)
    remove_worktree(path)
    if base and branch:
        try:
            # -D, not -d: a session's commits are merged nowhere by design (the
            # user merges what they want to keep), so -d would always refuse and
            # the branch would leak on every clean exit. Never raise: the tree
            # is already gone and a leftover ref must not break the REPL exit.
            subprocess.run(["git", "-C", base, "branch", "-D", branch],
                           capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _branch_of(worktree: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", worktree, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10)
        name = (r.stdout or "").strip()
        if r.returncode == 0 and name != "HEAD":
            return name
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


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
    import nx as _m
    root = os.path.join(home, "worktrees", _m.project_store_slug())
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
    want the tree, so only the registry link is dropped by callers).

    0.107.0: a dirty agent tree is never destroyed — a slow-but-alive sibling
    past the sweep TTL (or any crashed worker) keeps its uncommitted work.
    The dirty tree is renamed aside (path-dirty) so contents survive for
    manual recovery and a fresh agent can claim the name; a clean or
    already-gone tree is removed as before."""
    if not path or not os.path.isdir(path):
        return
    dirty = False
    try:
        d = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=15)
        dirty = d.returncode == 0 and bool(d.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    if dirty:
        try:
            os.rename(path, path.rstrip(os.sep) + "-dirty")
        except OSError:
            pass
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


def _sweep_agent_tree(path: str) -> None:
    """Reap ONE stale per-agent tree: a live tree at the normal TTL, or a
    `-dirty` tree at its OWN, longer one.

    remove_worktree() renames a dirty tree aside to `<path>-dirty` instead of
    destroying it (0.107.0) so a crashed worker's uncommitted work survives for
    manual recovery — but nothing ever revisited those dirs, so they
    accumulated forever. The rename path is therefore NOT re-entered here: a
    dirty tree past NX_SUBAGENT_DIRTY_WORKTREE_TTL_HOURS (default 72h, vs 24h
    for a live tree — long enough to notice and recover the work, short enough
    that a leak is eventually removed) is deleted outright. Best-effort.

    The clock is the dir's mtime, which `os.rename` preserves — so a tree that
    was ALREADY past the dirty TTL when it was renamed aside is reaped on the
    very next sweep, and one renamed early gets its full 72h from its last
    write. That is the age of the WORK, which is what the TTL is about."""
    dirty = path.rstrip(os.sep).endswith("-dirty")
    hours = _env_int("NX_SUBAGENT_DIRTY_WORKTREE_TTL_HOURS", 72) if dirty \
        else _env_int("NX_SUBAGENT_WORKTREE_TTL_HOURS", 24)
    if hours <= 0:
        return  # 0 = never sweep this kind
    try:
        if os.stat(path).st_mtime >= time.time() - hours * 3600:
            return
    except OSError:
        return
    if dirty:
        shutil.rmtree(path, ignore_errors=True)
    else:
        remove_worktree(path)


def sweep_stale_worktrees() -> None:
    """Reap leftover per-AGENT worktrees older than a TTL (0.103.7).

    The normal path removes a write-tier agent's tree when its finished
    record is evicted, but a killed process leaks `nx-wt-agent-*` dirs under
    SESSION_HOME/worktrees forever. Every write-tier spawn calls this once
    before creating its own tree: a single os.scandir pass, remove_worktree
    only on dirs whose mtime is older than NX_SUBAGENT_WORKTREE_TTL_HOURS
    (default 24; 0 = never sweep). A `-dirty` tree is reaped under its own
    longer TTL by _sweep_agent_tree (NX_SUBAGENT_DIRTY_WORKTREE_TTL_HOURS).
    Session worktrees (`nx-wt-session-*`) are never touched — a live session's
    tree must survive. Best-effort: every
    failure is swallowed; the sweep must never break the spawn it guards.
    One `git worktree prune` at the end (if a base repo is known) so
    registrations whose dirs are already gone are swept too."""
    try:
        # NX_SUBAGENT_WORKTREE_TTL_HOURS=0 still means "never sweep": the
        # whole pass is skipped, dirty trees included (and no `git worktree
        # prune` either, exactly as before). The dirty TTL only ever SHORTENS
        # what that pass reaps, it never widens when the sweep runs.
        if _env_int("NX_SUBAGENT_WORKTREE_TTL_HOURS", 24) <= 0:
            return
        home = st.SESSION_HOME or os.path.expanduser("~/.nx")
        import nx as _m
        root = os.path.join(home, "worktrees", _m.project_store_slug())
        # sweep this project's slug dir, then (transitionally, 0.145.2)
        # only entries sitting DIRECTLY in the legacy flat parent — never
        # recursing into other projects' slug dirs, which scandir's flat
        # listing can't reach anyway (they're dirs, filtered by name).
        roots = [root, os.path.join(home, "worktrees")]
        for d in roots:
            try:
                with os.scandir(d) as it:
                    entries = [e for e in it
                               if e.name.startswith("nx-wt-agent-")]
            except OSError:
                continue
            for e in entries:
                try:  # per-entry: one unreadable dir must not stop the sweep
                    if not e.is_dir(follow_symlinks=False):
                        continue
                    _sweep_agent_tree(e.path)
                except OSError:
                    continue
        base = st.PROJECT_ROOT
        if base:
            try:
                subprocess.run(["git", "-C", base, "worktree", "prune"],
                               capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
    except Exception:  # noqa: BLE001 — pure janitorial work, never fatal
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

