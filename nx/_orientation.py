"""One-time session-start orientation note, appended to the initial message
list (not folded into SYSTEM) so the model knows its real working directory
and which approval-free tools to reach for first — the static SYSTEM string
only says "cwd = user's current directory" without ever giving the actual
path. Paid once per fresh session (see _dispatch._initial_messages), so the
per-turn token floor (test_initial_context.py) stays flat.

Defaults on; `NX_ORIENTATION=0` suppresses it (e.g. for a fixed working dir
the user already knows, or to shave a few tokens off a tight context window).
"""
from __future__ import annotations

import os
import subprocess

from ._env import _env_int
from . import _state as st


def _ls_head() -> str:
    """Run `ls | head -20` in the cwd; returns '' on any failure (the note
    then simply omits the listing)."""
    try:
        out = subprocess.run(
            ["ls"], capture_output=True, text=True, timeout=5, check=False
        )
        names = [line for line in out.stdout.splitlines() if line][:20]
        return "\n".join(names)
    except Exception:
        return ""


def _cwd_and_root() -> tuple[str, str | None]:
    """Return (real cwd, project_root-or-None). project_root is only reported
    when it's a real git root that differs from cwd — the risk classifier's
    cwd fallback (running from $HOME) must not masquerade as a project."""
    cwd = os.path.realpath(os.getcwd())
    root = getattr(st, "PROJECT_ROOT", None)
    is_git = bool(getattr(st, "PROJECT_ROOT_IS_GIT", False))
    if root and is_git:
        root = os.path.realpath(root)
        if root == cwd:
            root = None
    else:
        root = None
    return cwd, root


def _load_orientation_note() -> dict[str, str] | None:
    """User-role note giving the model its real cwd (and project_root when
    it's a real git root distinct from cwd) and pointing it at the
    approval-free read-only tools. Returns None when disabled via
    NX_ORIENTATION=0."""
    if _env_int("NX_ORIENTATION", 1) == 0:
        return None
    cwd, root = _cwd_and_root()
    lines = [f"Working directory: {cwd}"]
    if root:
        lines.append(f"Project root: {root}")
    listing = _ls_head()
    if listing:
        lines.append("Top-level entries:\n" + listing)
    lines.append(
        "Orient with run_bash `ls` (read-only, approval-free for pure reads) "
        "for the full listing if it seems relevant."
    )
    return {
        "role": "user",
        "content": "[Orientation note:]\n" + "\n".join(lines),
    }