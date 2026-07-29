"""Pure risk-classification helpers: project-root/path scoping for the model
risk classifier's user message, and the regex fast path that skips the model
round-trip entirely for unambiguously read-only bash commands.

One cheap non-streaming call per write/bash action goes to the model (see
_assess_risk in __init__.py). Same model, tiny prompt, expects
{"level": "low|medium|high", "reason": "<short>"}. Everything in this module
is the part of that pipeline with no model/network/global-state dependency.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time

from ._openai_client import APIConnectionError

from ._logging import _log_event
from . import _state as st


# Three risk levels; shared with _approval.py (APPROVE_LEVEL comparisons)
# so it lives here rather than there to avoid a circular import (_approval
# already imports _local_risk_hint from this module).
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def _detect_project_root(cwd: str | None = None) -> str:
    """Return the git root for cwd when available, otherwise cwd itself."""
    cwd = os.path.realpath(cwd or os.getcwd())
    if shutil.which("git"):
        try:
            r = subprocess.run(
                ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=2)
            root = r.stdout.strip()
            if r.returncode == 0 and root:
                return os.path.realpath(root)
        except (OSError, subprocess.SubprocessError):
            pass
    return cwd


def _is_under_path(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except (OSError, ValueError):
        return False


def _extract_action_path(action: str) -> str | None:
    """Pull the primary file path from Nx's short approval action string."""
    if not isinstance(action, str):
        return None
    action = action.strip()
    if action.startswith("edit "):
        # edit_file -> "edit <path>"; multi_edit -> "edit <path> (N edits)".
        rest = action[len("edit "):].strip()
        if m := re.match(r"^(.+?)\s+\(\d+\s+edits\)$", rest):
            return m.group(1).strip() or None
        return rest or None
    if m := re.match(r"^write\s+(.+)\s+\(\d+\s+bytes\)$", action):
        return m.group(1).strip() or None
    return None


RISK_SYSTEM = (
    "You are a risk classifier for a coding agent's tool calls. "
    "Given one tool action as JSON, respond with ONLY a JSON object of the form "
    '{"level": "low"|"medium"|"high", "reason": "<one short sentence>"}.\n'
    "The user JSON includes cwd, project_root, and when possible path_scope. "
    "For outside-project decisions, trust project_root/path_scope over folder-name heuristics. "
    "If path_scope is in_project, do not classify high merely because the path is under ~/Downloads; "
    "projects can live in Downloads. If path_scope is outside_project, treat that as outside the project.\n"
    "Levels:\n"
    '- low: read-only or trivially reversible (ls, cat, grep, git status, mkdir, touch, file reads).\n'
    '- medium: modifies state but contained/reversible (writing a single file, editing a file, cp, mv, '
    'pip install in a venv, running tests, git commit).\n'
    '- high: destructive, hard to reverse, or broad scope (rm -rf, git push --force, git reset --hard, '
    'dd, chmod -R, writing outside the project, network sends to external hosts, killing processes, '
    'system-level changes, anything touching dotfiles in $HOME).\n'
    "When in doubt, classify higher. Output ONLY the JSON, no preamble."
)


# A regex-based fast path: commands built only from a short allowlist of
# read-only binaries, invoked standalone (no piping, redirection, chaining,
# or substitution), are exactly what RISK_SYSTEM's own "low" bucket already
# describes ("ls, cat, grep, git status, mkdir, touch, file reads") — so
# skip the model round-trip entirely and return that verdict locally.
# Deliberately conservative: ANY shell metacharacter bails out to the model,
# since a destructive side effect can hide behind an innocuous-looking
# leading command (e.g. `cat foo && rm -rf bar`); this only ever fires for
# a single bare command with no shell tricks in play.
_SAFE_BASH_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep",
    "find", "which", "type", "file", "stat", "echo", "whoami", "date",
    "printenv", "diff", "tree", "basename", "dirname", "realpath", "du",
    "df", "env", "true", "false", "mkdir", "touch",
}
_SAFE_GIT_SUBCOMMANDS = {"status", "diff", "log", "branch", "show", "remote", "tag", "blame"}
_SHELL_METACHARACTERS_RE = re.compile(r"[;&|<>`$(){}\n*?~]")

# Contained-but-not-reversible bash commands, matching RISK_SYSTEM's own
# "medium" bucket verbatim ("cp, mv, pip install in a venv, running tests,
# git commit"). Same no-metacharacters/single-command constraint as the low
# fast path above.
_MEDIUM_BASH_COMMANDS = {"cp", "mv", "pytest"}
_MEDIUM_GIT_SUBCOMMANDS = {"commit", "add", "checkout", "switch", "stash", "fetch", "pull", "merge"}

# Commands/flag combos that are unambiguously RISK_SYSTEM's "high" bucket
# regardless of arguments — destructive, hard to reverse, or system-level.
# Deliberately narrow (exact command names, explicit flag checks) since a
# false "high" only costs an extra prompt, but a false "low"/"medium" would
# skip one entirely.
_HIGH_RISK_BASH_COMMANDS = {"dd", "mkfs", "shutdown", "reboot", "poweroff", "halt", "shred"}
_HIGH_RISK_GIT_PATTERNS = {
    ("push", "--force"), ("push", "-f"), ("reset", "--hard"), ("clean", "-f"),
    ("clean", "-fd"), ("clean", "-fdx"), ("branch", "-D"),
}


def _rm_is_recursive_force(parts: list[str]) -> bool:
    """True when `parts` (an `rm` invocation, argv already split) combines
    recursive and force flags in any order/grouping (-rf, -fr, -r -f, --recursive
    --force, etc.) — the case RISK_SYSTEM calls out by name as high."""
    recursive = force = False
    for a in parts[1:]:
        if a in ("-r", "-R", "--recursive"):
            recursive = True
        elif a in ("-f", "--force"):
            force = True
        elif re.match(r"^-[a-zA-Z]+$", a):
            if "r" in a.lower():
                recursive = True
            if "f" in a:
                force = True
    return recursive and force



# Paths whose write/edit is self-executing or affects trust boundaries beyond
# "this one file's contents" — git hooks run on checkout/commit with no
# further confirmation, CI workflow files run on a shared runner on push/PR,
# and Makefile/build-script targets are commonly invoked blindly. These stay
# "high" even though they resolve in_project, since the in-project "medium"
# bucket assumes contained/reversible effects, which none of these are.
_SENSITIVE_RELPATH_RES = [
    re.compile(r"(^|/)\.git/hooks/"),
    re.compile(r"(^|/)\.github/workflows/"),
    re.compile(r"(^|/)\.githooks/"),
]
_SENSITIVE_BASENAMES = {"makefile", "gnumakefile"}


def _is_sensitive_project_path(abs_path: str, project_root: str) -> bool:
    try:
        rel = os.path.relpath(abs_path, project_root).replace(os.sep, "/")
    except ValueError:
        return False
    if os.path.basename(abs_path).lower() in _SENSITIVE_BASENAMES:
        return True
    return any(rx.search(rel) for rx in _SENSITIVE_RELPATH_RES)


def _local_risk_hint(action: str) -> tuple[str, str] | None:
    """Return (level, reason) when `action` is unambiguously classifiable per
    RISK_SYSTEM's own bucket definitions, without a model round-trip. Returns
    None for anything ambiguous (including parse failures), so the caller
    falls through to _assess_risk() exactly as before this existed."""
    if action.startswith("run: "):
        return _local_risk_hint_bash(action[len("run: "):].strip())
    path = _extract_action_path(action)
    if path is None:
        return None
    info = _classify_action_path(path)
    scope = info.get("path_scope")
    if scope == "in_project":
        if _is_sensitive_project_path(info["primary_path"], st.PROJECT_ROOT):
            return ("high", "writes a self-executing hook/CI/build file")
        return ("medium", "writes a file inside the project")
    # Outside-project writes need judgment calls the model is better placed
    # to make (dotfiles vs. a legitimate sibling project) — fall through.
    return None


def _local_risk_hint_bash(command: str) -> tuple[str, str] | None:
    if not command or _SHELL_METACHARACTERS_RE.search(command):
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    cmd = parts[0]
    if cmd == "rm" and _rm_is_recursive_force(parts):
        return ("high", "recursive forced delete")
    if cmd == "chmod" and any(a in ("-R", "--recursive") for a in parts[1:]):
        return ("high", "recursive chmod")
    if cmd in _HIGH_RISK_BASH_COMMANDS:
        return ("high", f"destructive/system command {cmd}")
    if cmd == "git" and len(parts) > 1:
        sub = parts[1]
        if any(sub == p[0] and p[1] in parts[2:] for p in _HIGH_RISK_GIT_PATTERNS):
            return ("high", f"destructive git {sub}")
        if sub in _SAFE_GIT_SUBCOMMANDS:
            return ("low", f"read-only git {sub}")
        if sub in _MEDIUM_GIT_SUBCOMMANDS:
            return ("medium", f"contained git {sub}")
        return None
    if cmd == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir")
                              for a in parts[1:]):
        return None  # find can destroy or run arbitrary commands with these
    if cmd == "env" and any(not a.startswith("-") and "=" not in a for a in parts[1:]):
        return None  # env <cmd> runs an arbitrary program, not read-only
    if cmd in _SAFE_BASH_COMMANDS:
        return ("low", f"read-only {cmd}")
    if cmd in _MEDIUM_BASH_COMMANDS:
        return ("medium", f"contained {cmd}")
    return None


st.PROJECT_ROOT = _detect_project_root()


def _classify_action_path(
    path: str, cwd: str | None = None, project_root: str | None = None
) -> dict[str, object]:
    cwd = os.path.realpath(cwd or os.getcwd())
    project_root = os.path.realpath(project_root or st.PROJECT_ROOT or cwd)
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(cwd, expanded)
    abs_path = os.path.realpath(expanded)
    home = os.path.realpath(os.path.expanduser("~"))
    downloads = os.path.join(home, "Downloads")
    rel_home = None
    if _is_under_path(abs_path, home):
        try:
            rel_home = os.path.relpath(abs_path, home)
        except ValueError:
            rel_home = None
    home_parts = [] if rel_home in (None, ".") else rel_home.split(os.sep)
    touches_home_dotdir = bool(home_parts and home_parts[0].startswith("."))
    in_project = _is_under_path(abs_path, project_root)
    return {
        "primary_path": abs_path,
        "path_scope": "in_project" if in_project else "outside_project",
        "path_in_cwd": _is_under_path(abs_path, cwd),
        "path_in_downloads": _is_under_path(abs_path, downloads),
        "path_touches_home_dotdir": touches_home_dotdir,
    }


def _risk_user_message(action: str) -> str:
    cwd = os.path.realpath(os.getcwd())
    msg = {
        "action": action,
        "cwd": cwd,
        "project_root": os.path.realpath(st.PROJECT_ROOT or cwd),
        "primary_path": None,
        "path_scope": "unknown",
        "scope_guidance": (
            "Use project_root/path_scope for outside-project decisions. "
            "A file under project_root is in-project even when project_root "
            "is inside ~/Downloads."
        ),
    }
    path = _extract_action_path(action)
    if path:
        msg.update(_classify_action_path(path, cwd=cwd))
    return json.dumps(msg, sort_keys=True)


def _assess_risk(action: str) -> tuple[str, str]:
    """Return (level, reason). level is one of LEVEL_ORDER; reason is a short
    string. On any failure (server down, bad JSON, unknown level) returns
    ("high", "<error>") so the caller falls through to the prompt path."""
    if st.client is None or st.MODEL is None:
        return ("high", "no active source; defaulting to high")
    payload = [
        {"role": "system", "content": RISK_SYSTEM},
        {"role": "user", "content": _risk_user_message(action)},
    ]
    max_conn = max(0, st.RISK_CONNECTION_RETRIES) + 1
    max_parse = 3
    conn_attempts_used = 0
    try:
        for parse_attempt in range(max_parse):
            resp = None
            while conn_attempts_used < max_conn:
                try:
                    _log_event("req", {"model": st.MODEL, "messages": payload,
                                       "stream": False, "_purpose": "risk",
                                       "_attempt": conn_attempts_used + 1,
                                       "_parse_attempt": parse_attempt + 1})
                    resp = st.client.chat.completions.create(
                        model=st.MODEL, messages=payload, stream=False, timeout=15,
                        **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
                    break
                except APIConnectionError:
                    conn_attempts_used += 1
                    if conn_attempts_used >= max_conn:
                        return ("high", f"server unreachable after {max_conn} attempts; defaulting to high")
                    time.sleep(max(0, st.RISK_CONNECTION_RETRY_SECONDS))
            if resp is None:
                return ("high", "no response; defaulting to high")
            try:
                _log_event("resp", {"_purpose": "risk", "data": resp.model_dump()})
            except Exception:
                pass
            text = (resp.choices[0].message.content or "").strip()
            level, reason = None, ""
            try:
                obj = json.loads(text)
                level = (obj.get("level") or "").strip().lower()
                reason = (obj.get("reason") or "").strip()
            except (json.JSONDecodeError, AttributeError, TypeError):
                if m := re.search(r'\b(low|medium|high)\b', text, re.IGNORECASE):
                    level = m.group(1).lower()
                reason = text[:120]
            if level in LEVEL_ORDER:
                return (level, reason or level)
            # Unparseable — add a sharp follow-up instruction and retry
            if parse_attempt < max_parse - 1:
                payload.append({"role": "assistant", "content": text or "(no output)"})
                payload.append({"role": "user", "content": (
                    "Respond with ONLY a single JSON object containing \"level\" and \"reason\" fields. "
                    "No preamble, no explanation, no markdown. Example:\n"
                    '{"level": "medium", "reason": "edits a file in the project"}'
                )})
        return ("high", f"unparseable risk response after {max_parse} attempts: {text[:80]!r}")
    except Exception as e:
        return ("high", f"risk call failed: {type(e).__name__}")
