"""Session-start delegation guidance: the note that steers the parent model
toward spawn_agent, and its enable check.

Split out of `_subagent.py` (which keeps the spawn machinery). Injected ONLY
when the spawn_agent schema is actually offered (NX_SUBAGENTS=1) — see
`_dispatch._initial_messages()`. The tool description alone wasn't enough:
models left spawn_agent unused and ground through dozens of sequential
read_file/search_files calls themselves unless the user explicitly asked for
a fan-out. A schema description reads as a *permission*, this reads as an
*expectation*, and it's the layer that carries the trigger list (unfamiliar
codebase, cross-module trace, callers of a symbol, 2+ unrelated questions).
Kept out of the static SYSTEM string so a sub-agents-disabled session (the
default) pays nothing for it."""
from __future__ import annotations

from ._env import _env_int

# Session-start delegation guidance, injected ONLY when the spawn_agent schema
# is actually offered (NX_SUBAGENTS=1) — see _dispatch._initial_messages().
# The write-tier guidance (WRITE_TIER_GUIDANCE) is appended at note time only
# when this session actually authorizes write-capable agents
# (NX_SUBAGENT_WRITE=1) — a read-only session must not invite tier:'write'.
DELEGATION_GUIDANCE = (
    "[Delegation note: the spawn_agent tool is available this session — reach "
    "for it PROACTIVELY, without being asked.\n"
    "- Answering or editing about code you haven't read, when the question "
    "spans 2+ files or modules? Spawn one read-only agent per independent area "
    "instead of reading them yourself, then work from their findings.\n"
    "- Independent/unrelated questions go in ONE turn as separate spawn_agent "
    "calls — they run concurrently, and their intermediate greps/reads never "
    "enter your context (that's the whole point: delegation without bloat).\n"
    "- Map an unfamiliar repo, trace a behavior across modules, or enumerate a "
    "symbol's callers: delegate the sweep, keep the answer.\n"
    "- Do it yourself for a file you've already read, one targeted "
    "search_files/read_file, and every edit you're making.\n"
    "- spawn_agent is backgrounded by default: you get a job id now and the "
    "answer arrives as a [Runtime note] plus report path on a later turn; "
    "async=false runs it in the foreground instead.]"
)

# Appended to DELEGATION_GUIDANCE when this session authorizes write-capable
# sub-agents (NX_SUBAGENT_WRITE=1): names the tiers, the scope discipline, and
# the background-mode rule (yolo required; foreground otherwise), so the model
# can delegate implementation — not just investigation — without guessing at
# the contract.
WRITE_TIER_GUIDANCE = (
    "- This session authorizes write-capable sub-agents: pass tier='write' "
    "(plain edits) or 'implement' (adds the implementer discipline prompt: "
    "read before editing, verify as you go, change approach after two "
    "failures, stop and report rather than churn).\n"
    "- Write/implement agents SHOULD carry scope (path globs like "
    "['nx/_turn*.py']); disjoint scopes are what makes parallel implementers "
    "safe. Background (async, the default) write agents work only in yolo "
    "mode; without yolo pass async=false so approval prompts can show.\n"
    "- With NX_SUBAGENT_WORKTREE=1 each write-tier agent gets its own git "
    "worktree (its paths, test runs and commits land there, never the shared "
    "tree), so parallel write agents are safe even with overlapping scopes; "
    "the finished report names the worktree — merge or cherry-pick from it.]"
)

# The marker _add_delegation_note_if_absent scans for (and _memory_note.py's
# session-start note contract mirrors).
DELEGATION_NOTE_MARKER = "[Delegation note:"


def _delegation_enabled() -> bool:
    """True when the spawn_agent schema is being offered this session.
    Resolved through the top-level module attribute (NX_SUBAGENTS) so
    tests/REPL can flip it live; falls back to the env var."""
    import nx as _m
    try:
        return bool(_m.NX_SUBAGENTS)
    except AttributeError:
        return bool(_env_int("NX_SUBAGENTS", 0))


def _write_authorized() -> bool:
    """True when this session's sub-agents may edit files. Reads the live
    top-level attribute first (tests/REPL flip it), then the env var."""
    import nx as _m
    if "NX_SUBAGENT_WRITE" in vars(_m):
        return bool(_m.NX_SUBAGENT_WRITE)
    return bool(_env_int("NX_SUBAGENT_WRITE", 0))


def _load_delegation_note() -> dict[str, str] | None:
    """User-role session-start note with the delegation triggers, or None when
    sub-agents are disabled (nothing to steer toward). The write-tier guidance
    is appended only when NX_SUBAGENT_WRITE authorizes it this session."""
    if not _delegation_enabled():
        return None
    text = DELEGATION_GUIDANCE
    if _write_authorized():
        text = text[:-1] + "\n" + WRITE_TIER_GUIDANCE
    return {"role": "user", "content": text}


def _add_delegation_note_if_absent(messages: list) -> None:
    """Give a resumed session the same note a fresh one gets (its saved
    transcript may predate the feature). Mirrors
    _memory_cmd._add_memory_note_if_absent's marker-scan contract."""
    for m in messages:
        if (m.get("role") == "user" and isinstance(m.get("content"), str)
                and DELEGATION_NOTE_MARKER in m["content"]):
            return
    note = _load_delegation_note()
    if note is not None:
        at = 1 if (messages and messages[0].get("role") == "system") else 0
        messages.insert(at, note)
