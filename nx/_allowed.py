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
import time
from typing import Any, Optional

from . import _state as st

DEFAULT_FILENAME = "allowed.json"


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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    the same text the approval prompt shows."""
    for rule in _load_rules():
        if rule.get("scope", "project") == "project" and (st.PROJECT_ROOT or os.getcwd()) != rule.get("project_root"):
            continue
        if rule.get("tool", "*") not in ("*", tool):
            continue
        if fnmatch.fnmatchcase(action, rule.get("pattern", "*")):
            return (rule.get("pattern", "*"), rule.get("scope", "project"))
    return None


DEFAULT_SCOPE = "project"


def _allowlist_tool(action: str) -> str:
    """The tool name a saved rule must match for `action`. nx's action strings
    all carry the tool's own prefix, so this inverts that mapping."""
    if action.startswith("run: "):
        return "run_bash"
    if action.startswith(("edit ", "write ", "delete ")):
        return ("edit_file" if action.startswith("edit ")
                else "delete_file" if action.startswith("delete ")
                else "write_file")
    if action.startswith("write memory"):
        return "memory_write"
    return "*"


def _rule_from_action(action: str) -> tuple[str, str, str]:
    """Derive (tool, pattern, scope) for an `A`-key save from the action shown
    at the approval prompt. The pattern is deliberately *prefix-shaped*
    (`action + "*"`), not the raw action: 'always allow' means this shape of
    action, not this exact byte-string — e.g. `run: pytest tests/ -q` blesses
    future `pytest` invocations too. Exact-byte blessings would be near-useless
    since tool args almost never repeat verbatim.
    """
    tool = _allowlist_tool(action)
    return (tool, action + "*", DEFAULT_SCOPE)
