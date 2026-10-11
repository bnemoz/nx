"""Headless approval policy (UIUX_SPEC C4, 0.135.1).

On headless surfaces (Telegram gateway, async sub-agent bash, `nx run` batch)
there is nobody to answer a Y/n prompt, so a gated call is auto-DENIED — see
the `_subagent_quiet()` check at the top of `_approval._confirm`. That is the
safe default, but it makes "let the headless run do exactly this one kind of
thing" impossible without going full YOLO.

`NX_APPROVAL_POLICY=headless-allow:<glob>` closes that gap: an operator
pre-approves TOOL names matching `glob` (fnmatch, so `bash*`/`edit_file`/
`*git*` work; a comma-separated list of globs is accepted). When a gated call
would otherwise be auto-denied headlessly and its TOOL name matches, the call
runs instead of being denied. Anything that does not match — and any policy
value that is unset, malformed, or not `headless-allow` — keeps the current
auto-deny. The decision is silent: a headless surface must not write to a
shared terminal (batch emits JSONL), so unlike the interactive auto-allow
lines this one leaves no paper trail beyond the tool's own output.

The match is on the TOOL name (the first word of the action string), never on
the whole action, and it grants nothing interactively: with a TTY present the
normal prompt flow runs as before.
"""
from __future__ import annotations

import fnmatch
import os

ENV_VAR = "NX_APPROVAL_POLICY"
_PREFIX = "headless-allow:"


def _policy_globs() -> tuple[str, ...]:
    """Parse NX_APPROVAL_POLICY into tool-name globs; () = deny (default)."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw.lower().startswith(_PREFIX):
        return ()
    spec = raw[len(_PREFIX):].strip()
    if not spec:
        return ()
    return tuple(g.strip() for g in spec.split(",") if g.strip())


def headless_allow(action: str) -> bool:
    """True when a headless auto-deny should instead run `action`: the policy
    is `headless-allow` and the action's TOOL name matches one of its globs."""
    globs = _policy_globs()
    if not globs:
        return False
    tool = action.strip().split(None, 1)[0] if action.strip() else ""
    return any(fnmatch.fnmatchcase(tool, g) for g in globs)
