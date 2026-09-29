"""The sub-agent read-only bash tier: run_bash behind a single-command
read-only allowlist.

Split out of `_subagent.py`. run_bash IS offered to a sub-agent, but the
command must pass `_subagent_bash_allowed()` — a single bare command from an
allowlist anchored on `_risk.py`'s own safe-command table (grep/ls/du/ps/git
log/...). Any metacharacter, write-shaped command, or unknown binary is
refused before it ever reaches run_bash, so a sub-agent can scan logs and
process state without write capability. wait_background stays excluded
(nothing it runs can background)."""
from __future__ import annotations

import shlex
from typing import Any

# Read-only binaries the sub-agent tier may run as a single bare command
# (no pipes/redirection/&&/globs — the same no-metacharacters constraint
# _risk.py applies before its own allowlist). Deliberately EXCLUDES the
# write-capable entries the parent's _SAFE_BASH_COMMANDS contains for its
# approval fast path (mkdir, touch, echo, env, ...) — a sub-agent has no
# user watching a Y/n prompt, so the allowlist itself is the safety.
# git commands are resolved through _local_risk_hint_bash() instead (see
# _subagent_bash_allowed) so only its read-only verdicts pass.
_SUBAGENT_BASH_ALLOW = frozenset((
    "grep", "egrep", "fgrep", "rg", "ls", "pwd", "cat", "head", "tail", "wc",
    "find", "du", "df", "stat", "file", "diff", "tree", "realpath", "basename",
    "dirname", "which", "type", "ps", "pgrep", "md5sum", "sha256sum",
    # harmless text utilities: each reads stdin/files and only prints, so they
    # can't mutate anything even chained conceptually (pipes are still refused
    # by the no-metacharacters rule; these run as single commands only).
    "sort", "uniq", "cut", "paste", "join", "column", "comm", "fold",
    "nl", "rev", "tac", "expand", "unexpand", "seq", "date",
))


def _subagent_bash_allowed(command: str) -> bool:
    """True when `command` is a single bare read-only command the sub-agent
    tier may run: one allowlisted binary (_SUBAGENT_BASH_ALLOW) or a git
    invocation the parent's own _risk.py classifier scores "low"
    (status/diff/log/show/tag/blame, listing-only branch/remote). Anything
    else — pipes, &&, redirection, globs, env-prefixing, unknown binaries,
    write-shaped commands — is refused; the sub-agent reports what it needs
    and the parent runs it."""
    if not isinstance(command, str) or not command.strip():
        return False
    from ._risk import (_local_risk_hint_bash, _strip_shell_comment,
                        _SHELL_METACHARACTERS_RE)
    cmd = _strip_shell_comment(command.strip())
    if _SHELL_METACHARACTERS_RE.search(cmd):
        return False
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return False
    if not parts:
        return False
    argv = parts[1:]
    head = parts[0]
    if head == "git":
        hint = _local_risk_hint_bash(cmd)
        return hint is not None and hint[0] == "low"
    if head not in _SUBAGENT_BASH_ALLOW:
        return False
    # find is allowlisted only in its listing form; these flags delete files or
    # execute arbitrary commands (the same veto _risk.py's classifier applies).
    if head == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir")
                              for a in argv[1:]):
        return False
    return True


def _subagent_bash(command: str, **_: Any) -> str:
    """run_bash replacement installed in the read-only sub-agent dispatch table:
    runs the command only when it passes the read-only allowlist. The sub-loop
    already forces YOLO (read-only tools never prompt), and an allowlisted
    single command can't carry redirection/chaining, so there is no approval
    path to reason about. wait_background is not offered at all — if an
    allowlisted command somehow outlives the poll window, the [background]
    result tells the sub-agent to read the log with read_file.

    A WRITE-tier agent (tier='write'/'implement') is dispatched here too —
    _write_tier_dispatch layers the write tools on top of the read-only table —
    but it gets the parent's real run_bash with its own _confirm approval gate
    instead of this allowlist: the same gate its file edits cross. Building
    (cargo check, uv run pytest, ...) is exactly what a write agent exists to
    do, and no read-only allowlist can express it. A quiet write agent still
    can't prompt — _confirm auto-denies there, same as its edits."""
    from ._subagent_tiers import _subagent_ctx
    ctx = _subagent_ctx()
    if ctx is not None and ctx.write:
        from ._bash_jobs import run_bash as _real_run_bash
        # Anchor the command in the agent's own worktree when it has one:
        # otherwise git add/stash/reset and other side-effecting commands the
        # agent runs "on its tree" hit the PARENT's index/worktree — the
        # primary source of phantom MM in the parent (0.103.15).
        cwd = getattr(ctx, "worktree", None) or None
        return _real_run_bash(command, _cwd=cwd)
    if not _subagent_bash_allowed(command):
        return ("ERROR: sub-agent bash is limited to single read-only commands "
                f"(grep/ls/du/ps/pgrep/git log/git status/...); refused: {command!r}. "
                "Report what you need and let the parent run it.")
    from ._bash_jobs import run_bash
    return run_bash(command)
