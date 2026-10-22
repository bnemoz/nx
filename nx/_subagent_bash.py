"""The sub-agent read-only bash tier: run_bash behind a read-only allowlist.

Split out of `_subagent.py`. run_bash IS offered to a sub-agent, but the
command must pass `_subagent_bash_allowed()`: each pipeline stage must be a
single bare command from an allowlist anchored on `_risk.py`'s own
safe-command table (grep/ls/du/ps/git log/...), and shell globs (`*`, `?`,
`~`, braces) are allowed — expansion is read-only. Any non-pipeline
metacharacter (`;`, `&`, redirection, command substitution), write-shaped
command, or unknown binary is refused before it ever reaches run_bash, so a
sub-agent can scan logs and process state without write capability.
wait_background stays excluded (nothing it runs can background)."""
from __future__ import annotations

import re
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
# Same shape as _risk.py's _SHELL_METACHARACTERS_RE but WITHOUT the glob
# characters: `*`, `?`, `~`, `{`, `}` are read-only shell expansion, and
# refusing them forced sub-agents into one-file-at-a-time greps.
_SHELL_METACHARACTERS_RE_GLOBS_OK = re.compile(r"[;&<>`$()]")


def _split_pipeline(cmd: str) -> list[str]:
    """Split on `|` that is not inside single or double quotes (a quoted `|`
    is an argument character, not a pipe — bash never splits on it)."""
    stages, buf, quote = [], [], None
    for ch in cmd:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == "|":
            stages.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    stages.append("".join(buf).strip())
    return stages


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
    """True when `command` is read-only work the sub-agent tier may run: a
    pipeline whose every stage is one allowlisted binary
    (_SUBAGENT_BASH_ALLOW) or a git invocation the parent's own _risk.py
    classifier scores "low" (status/diff/log/show/tag/blame, listing-only
    branch/remote). Globs are allowed (read-only expansion). Anything else —
    `;`/`&` sequencing or backgrounding, redirection, command substitution,
    env-prefixing, unknown binaries, write-shaped commands — is refused; the
    sub-agent reports what it needs and the parent runs it."""
    if not isinstance(command, str) or not command.strip():
        return False
    from ._risk import (_local_risk_hint_bash, _strip_shell_comment)
    cmd = _strip_shell_comment(command.strip())
    if "|" in cmd:
        # Pipeline: every stage must independently pass the single-command
        # rule — the allowlist itself is the safety, not the chain length.
        # A `|` inside quotes never splits here, so a command like
        # grep 'a|b' f comes back as one stage and falls through to the
        # single-command path below.
        stages = _split_pipeline(cmd)
        if len(stages) > 1:
            return all(_subagent_bash_allowed(stage) for stage in stages)
    # Remaining metacharacters are refused outright: `;`/`&` sequencing or
    # backgrounding, `<>` redirection, `` ` ``/`$`/`(`/`)` command
    # substitution. Globs are read-only and permitted.
    if _SHELL_METACHARACTERS_RE_GLOBS_OK.search(cmd):
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
    # Write/execute-shaped flags of otherwise read-only binaries. The veto
    # scans the full argv INCLUDING the head — the same slice _risk.py:271
    # uses for its find veto (matching the two homes is pinned by a meta-test,
    # tests/test_subagent.py). 0.168.2: fixed the previous argv[1:] off-by-one
    # that let `find -delete …` skip its first argument, and extended the veto
    # to the other in-argv write escapes of allowlisted binaries (audit P1-3).
    if head == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir",
                                    "-fprintf", "-fprint", "-fprint0", "-fls")
                              for a in parts):
        return False
    if head == "sort" and any(a == "-o" or a.startswith("--output=")
                              for a in argv):
        return False
    if head in ("rg", "grep", "egrep", "fgrep") and any(
            a.startswith("--pre") or a == "-O" or a.startswith("--hostname-bin=")
            for a in argv):
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
        return ("ERROR: sub-agent bash is limited to read-only commands — "
                "pipelines of grep/ls/du/ps/pgrep/git log/git status/... are "
                f"fine; refused: {command!r}. "
                "Report what you need and let the parent run it.")
    from ._bash_jobs import run_bash
    return run_bash(command)
