"""The clobber guard (0.104.2): a deterministic, YOLO-overriding check for
commands that would truncate (not append) an existing file.

Deterministic like the write fence (nx/_fence.py), not a judgement: a small
static scan finds shell redirections that REPLACE a file's contents —
``> file``, ``>>``-less heredocs land there too — and, when the target file
already exists and is non-trivial, forces the approval prompt EVEN IN YOLO
mode. The precedent is the run_bash secret guard (NX_BASH_PROMPT_SECRETS'
``_contains_secret``): a cheap pattern match that must not be silently
auto-approved, because the failure it guards against is not recoverable —
a truncated dotfile (an ~/.env with every credential in it) is destroyed
outright, while an ordinary command that misbehaves merely fails.

Why only run_bash: the structured write tools (write_file/edit_file/…)
already show a unified diff through the approval gate and honor the
staleness guard, so a model clobbering a file with them is visible and
revertible (/undo). run_bash is the one write surface with no static
coverage (the write fence's own docstring records this gap) — and it was
exactly how a session destroyed a populated ~/.env with ``cat > ~/.env``:
the user-visible story ("a dotfile was overwritten") traced to an
unreviewed truncation, not an edit tool.

Scope discipline (mirrors the secret guard):
- Only TRUNCATING redirections count. ``>>`` appends are safe by
  construction and never prompt. ``>|`` (noclobber override) counts.
- Only EXISTING, non-empty targets count. Creating a new file, or
  truncating an empty one, is the ordinary way anything gets written —
  prompting there would train the user to mash Y.
- Dotfiles are not special-cased: the scan is content-blind. A populated
  ~/.env and a populated notes.txt deserve the same protection.
- Known-log and ephemeral extensions (.log, .tmp, .out, .swp, .pid, .pyc,
  .o, .so, ...) are exempt: the model legitimately rewrites run logs every
  session, and prompting there trains the user to mash Y. An explicit
  NX_CLOBBER_LOG_EXTENSIONS=0 restores the guard for them; dotfiles and
  lock-like names are never exempt.
- Deterministic, never raises, one regex pass. ``NX_CLOBBER_GUARD=0``
  disables it for environments that find it noisy.

This module also carries the sibling **git-index guard** (bottom of file):
the same shape (deterministic, YOLO-overriding, refuse-in-sub-agent) applied
to the other side effect a raw shell git command has that /undo cannot
repair — a mutating ``git add`` / ``git commit`` sweeping the SHARED staged
index, which may already hold another concurrent session's staged work
(``git_commit`` avoids this by staging in a private ``GIT_INDEX_FILE``; raw
``git add && git commit`` in run_bash has no such isolation). Its
kill-switch is ``NX_GIT_INDEX_GUARD=0``.
"""
from __future__ import annotations

import os
import re
import subprocess

from ._env import _env_int

# A truncating redirection target. Matches:
#   > path        >> would not match     >| path counts (noclobber override)
#   2> path       &> path                 i>&j is FD duplication, NOT a file
# The target is a POSIX word: no quotes handling beyond the simple quoted
# form (the guard is a heuristic backstop, not a shell parser — the write
# fence's docstring covers why full shell parsing is out of scope).
_REDIR_RE = re.compile(
    r"(?:^|[\s;|&(])(\d*&?>{1,2}\|?|\d*>+)\s*(\"[^\"]*\"|'[^']*'|[^\s;&|)<>]+)"
)

# FD-duplication forms (2>&1) and append (>>) are not truncations.
_DUP_RE = re.compile(r"^\d*>>&?$")

# Extensions that are logs or build/ephemeral artifacts. The model
# legitimately truncates these every session (rerunning a benchmark
# rewrites its run log), so prompting trains the user to mash Y.
# NX_CLOBBER_LOG_EXTENSIONS=0 restores the guard for them.
_LOG_EXTENSIONS = frozenset(
    {".log", ".tmp", ".out", ".err", ".swp", ".cache",
     ".pyc", ".pyo", ".o", ".so", ".a", ".obj", ".d"}
)

# Names whose loss is real (locks, env, credentials) — never exempt.
_NEVER_EXEMPT_RE = re.compile(
    r"(^|/)(\.[^/]*|[^/]*\.(lock|env|pem|key))$|env$|cred", re.I
)


def _redirect_targets(command: str) -> list[str]:
    """Paths that ``command`` would truncate, best effort. Quoted targets are
    unquoted; nothing is expanded (~ is, variables are not — the caller
    stats what it can and skips what it can't)."""
    out: list[str] = []
    for m in _REDIR_RE.finditer(command):
        op, target = m.group(1), m.group(2)
        if _DUP_RE.match(op):
            continue  # 2>&1 family: FD duplication, never a file truncation
        if target.startswith(">"):  # shouldn't happen, but never guess
            continue
        if target[0] in "\"'":
            target = target[1:-1]
        target = os.path.expanduser(target)
        if target and "$" not in target:
            out.append(target)
    return out


def clobber_targets(command: str) -> list[str]:
    """Existing, non-empty files `command` would truncate. Empty list =
    nothing at risk (or nothing checkable)."""
    if not _env_int("NX_CLOBBER_GUARD", 1):
        return []
    hits: list[str] = []
    for target in _redirect_targets(command):
        try:
            st = os.stat(target)
        except OSError:
            continue  # doesn't exist / not statable — creating it is fine
        if st.st_size == 0 or target in hits:
            continue
        if (_env_int("NX_CLOBBER_LOG_EXTENSIONS", 1)
                and os.path.splitext(target)[1].lower() in _LOG_EXTENSIONS
                and not _NEVER_EXEMPT_RE.search(target)):
            continue
        hits.append(target)
    return hits


def clobber_guard_message(targets: list[str]) -> str:
    """The prompt line for a clobbering command — phrased as the question the
    user actually needs to answer (destroy these files' contents?)."""
    names = ", ".join(targets[:3]) + ("…" if len(targets) > 3 else "")
    return (
        f"  TRUNCATES existing file(s): {names}. Their current contents will "
        f"be replaced (no backup taken). Run anyway? [Y/n/esc] ")


# --- the git-index guard ------------------------------------------------------
# Same shape as the clobber guard, different unrecoverable side effect. The
# incident: a session ran `git add … && git commit` through run_bash and swept
# the SHARED index — including another session's staged REVERT of a milestone
# commit — into its own commit. nx's git_commit tool is immune (0.103.0 stages
# into a private GIT_INDEX_FILE); a raw shell git op has no such isolation, and
# /undo cannot undo a commit that swallowed someone else's staged work.
#
# Attribution of the staged set to "this session" is not solvable from the
# index (git records no owner), so the check is deliberately conservative:
# ANY staged entry at the moment a mutating git op runs forces the prompt, and
# the prompt NAMES the staged paths so the user can see whose work it is. The
# false-positive cost is one prompt, and the common flow is exempt by
# construction: the check runs BEFORE the command, so `git add mine.py && git
# commit -m x` from a clean index never trips it — only work that was already
# staged by someone else does.

# git's global-flag forms that may sit between `git` and its subcommand. Any
# other token there means we simply do not match — a miss is a prompt that
# wasn't fired, never a wrong action taken.
_GIT_GLOBAL_FLAGS = (
    r"((?:\s+(?:-[cC][=\s]\S+|--(?:git-dir|work-tree|namespace)[=\s]\S+"
    r"|--exec-path[=\s]\S+|--no-pager|--bare))*)"
)

# The mutating ops this guard covers: `git add`, `git commit`, `git reset`
# (any form rewrites the index), plus `git restore --staged` and
# `git merge --squash`, which need their flag to appear in the same segment.
_GIT_OP_RE = re.compile(
    r"(?:^|[\s;&|`(])git\b" + _GIT_GLOBAL_FLAGS +
    r"\s+(add|commit|reset|restore|merge)(?![\w-])([^;&|\n]*)"
)

_GIT_OP_REQUIRED_FLAG = {"restore": "--staged", "merge": "--squash"}


def _in_quotes(command: str, pos: int) -> bool:
    """True when a naive single/double-quote scan says offset `pos` sits
    inside a quoted run: `rg 'git commit'`/`echo "git commit"` SEARCH for
    the words, they do not run them. A quoted `sh -c "git commit"` is
    missed by this — accepted: the same content-blind-limitation stance as
    the redirect scan above (a miss is one unfired prompt, never a wrong
    action)."""
    sq = dq = False
    for i in range(pos):
        c = command[i]
        if c == "'" and not dq:
            sq = not sq
        elif c == '"' and not sq:
            dq = not dq
    return sq or dq

# How long we let `git status --porcelain` take before concluding "nothing
# checkable" (the guard degrades to a no-op, never to a guess).
_GIT_STATUS_TIMEOUT = 5


def git_mutating_index_op(command: str) -> tuple[str, str | None] | None:
    """`(op_label, repo_path)` for the first index-mutating git op in
    `command`, or None when it runs none. `repo_path` is the `-C`/`--git-dir`
    target when the command names one, so the guard checks THAT repo's index
    instead of the process cwd's. Best-effort static scan, not a shell parser
    (the write fence's docstring covers why full parsing is out of scope)."""
    if "git" not in command:
        return None
    for m in _GIT_OP_RE.finditer(command):
        if _in_quotes(command, m.start()):
            continue  # an argument (a search pattern), not an invocation
        flags, op, tail = m.group(1) or "", m.group(2), m.group(3) or ""
        want = _GIT_OP_REQUIRED_FLAG.get(op)
        if want and want not in tail:
            continue  # plain `git restore .` touches the worktree, not the index
        repo = None
        # -C (capital) names a repo; -c names a config setting — only the
        # former is a path to check. A --git-dir target gets normalized to
        # its worktree (git status can't run with cwd inside a bare .git).
        fm = re.search(r"(?:-C[=\s]|--git-dir[=\s])(\S+)", flags)
        if fm:
            repo = os.path.expanduser(fm.group(1))
            if repo.endswith(os.sep + ".git"):
                repo = os.path.dirname(repo) or repo
        return (f"{op} {want}".strip() if want else op), repo
    return None


def staged_index_paths(cwd: str | None = None) -> list[str]:
    """Paths the SHARED index holds right now (index != HEAD), deduped.
    Empty list when there is nothing staged OR nothing checkable (no repo, no
    git, timeout) — the guard stays silent rather than guessing. GIT_INDEX_FILE
    is stripped from the child env on purpose: the question this answers is
    what the SHARED index holds, not what a private index does."""
    env = {k: v for k, v in os.environ.items() if k != "GIT_INDEX_FILE"}
    try:
        proc = subprocess.run(["git", "status", "--porcelain"],
                              cwd=cwd or None, capture_output=True, text=True,
                              timeout=_GIT_STATUS_TIMEOUT, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    out: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4 or line[0] in (" ", "#", "?", "!"):
            continue  # unstaged-only, branch header, untracked, ignored
        path = line[3:].strip('"')
        if " -> " in path:  # rename/add: the new name is the staged path
            path = path.split(" -> ", 1)[1]
        if path and path not in out:
            out.append(path)
    return out


def _staged_root(cwd: str | None) -> str | None:
    """The repo root `staged_index_paths(cwd)`'s paths are relative to (git
    porcelain reports repo-root-relative paths whatever the cwd), or None when
    it can't be resolved — then claims compare the raw strings instead."""
    env = {k: v for k, v in os.environ.items() if k != "GIT_INDEX_FILE"}
    root = ""
    try:
        proc = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                              cwd=cwd or None, capture_output=True, text=True,
                              timeout=_GIT_STATUS_TIMEOUT, env=env)
        if proc.returncode == 0:
            root = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        root = ""
    try:
        return os.path.realpath(root or cwd or os.getcwd())
    except OSError:
        return None


def _staged_pairs(cwd: str | None) -> list[tuple[str, str]]:
    """`(reported, absolute)` for every shared-index staged path."""
    root = _staged_root(cwd)
    return [(p, p if root is None else os.path.join(root, p))
            for p in staged_index_paths(cwd)]


def staged_paths_abs(cwd: str | None = None) -> list[str]:
    """`staged_index_paths` resolved to absolute paths — the form
    `_stagedclaim` records claims in, so a path `git status` reported and the
    same path named absolutely compare equal."""
    return [abs_p for _p, abs_p in _staged_pairs(cwd)]


def git_index_guard(command: str, cwd: str | None = None) -> list[str]:
    """The staged paths at risk if `command` runs NOW: non-empty only when the
    command mutates the git index/commits AND the shared index already holds
    staged entries THIS session did not stage itself. `NX_GIT_INDEX_GUARD=0`
    disables. A clean index keeps `git add`/`git commit` prompt-free, and so
    does a session's own staged WIP (`_stagedclaim`: what git_commit's shared
    index sync left behind, or what this session's earlier raw `git add`
    staged) — only FOREIGN staged paths still prompt. Reported paths are the
    ones `git status` printed (repo-relative), minus the claimed ones."""
    hit = git_mutating_index_op(command)
    if hit is None or not _env_int("NX_GIT_INDEX_GUARD", 1):
        return []
    pairs = _staged_pairs(hit[1] or cwd)
    if not pairs:
        return []
    from ._stagedclaim import claimed_staged
    claimed = claimed_staged([abs_p for _p, abs_p in pairs])
    return [p for p, abs_p in pairs if abs_p not in claimed]


def git_index_guard_message(command: str, paths: list[str]) -> str:
    """The prompt line — the question the user actually has to answer is
    'is this staged work yours to commit?'"""
    hit = git_mutating_index_op(command)
    what = f"`git {hit[0]}`" if hit else "This command"
    names = ", ".join(paths[:3]) + (f" (+{len(paths) - 3} more)"
                                    if len(paths) > 3 else "")
    return (
        f"  SHARED git index already has staged changes: {names}. {what} "
        f"would sweep them into its own result — they may be ANOTHER "
        f"SESSION's staged work (nx's git_commit stages in a private index; "
        f"a raw git op does not). Clear them first with "
        f"`git restore --staged <path>` to keep them out. Run anyway? "
        f"[Y/n/esc] ")
