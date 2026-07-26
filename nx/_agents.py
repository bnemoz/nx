"""Load project-level `AGENTS.md` (repo-root convention for agent instructions,
same idea as CLAUDE.md) into a one-time context note, appended to the initial
message list rather than folded into SYSTEM — keeps the per-turn SYSTEM token
floor (see test_initial_context.py) unaffected by project size, and degrades
silently when no AGENTS.md exists.
"""
import os

AGENTS_MD_MAX_CHARS = 8000


def _find_agents_md(project_root):
    """Return the path to `AGENTS.md` at `project_root`, or None."""
    if not project_root:
        return None
    path = os.path.join(project_root, "AGENTS.md")
    return path if os.path.isfile(path) else None


def _load_agents_md_note(project_root):
    """Read AGENTS.md under `project_root` and return a user-role note dict
    to prepend to a fresh session, or None if there's no AGENTS.md (or it
    can't be read)."""
    path = _find_agents_md(project_root)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None
    content = content.strip()
    if not content:
        return None
    if len(content) > AGENTS_MD_MAX_CHARS:
        content = content[:AGENTS_MD_MAX_CHARS] + "\n[...truncated...]"
    return {
        "role": "user",
        "content": f"[Project instructions from AGENTS.md:]\n{content}",
    }
