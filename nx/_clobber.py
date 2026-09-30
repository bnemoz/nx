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
- Deterministic, never raises, one regex pass. ``NX_CLOBBER_GUARD=0``
  disables it for environments that find it noisy.
"""
from __future__ import annotations

import os
import re

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
        if st.st_size > 0 and target not in hits:
            hits.append(target)
    return hits


def clobber_guard_message(targets: list[str]) -> str:
    """The prompt line for a clobbering command — phrased as the question the
    user actually needs to answer (destroy these files' contents?)."""
    names = ", ".join(targets[:3]) + ("…" if len(targets) > 3 else "")
    return (
        f"  TRUNCATES existing file(s): {names}. Their current contents will "
        f"be replaced (no backup taken). Run anyway? [Y/n/esc] ")
