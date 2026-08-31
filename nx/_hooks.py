"""Org/user tool hooks: NX_PRE_TOOL_HOOK / NX_POST_TOOL_HOOK.

Each is a shell command run (via `sh -c`, like run_bash) around every tool
dispatch in run_tool(), receiving one JSON object on stdin:
  {"tool": "write_file", "args": {...},
   "result": "<tool result string>"   (post-hook only)
  }

Exit code semantics (pre-hook): 0 = allow, non-zero = veto. On veto the tool
does NOT run and the model receives:
  ERROR: blocked by NX_PRE_TOOL_HOOK (<first line of hook stderr/stdout>)
so the model can react, and the user sees the hook's message. The post-hook's
exit code is advisory only (logged to the event log) — the tool already ran.

Hooks are best-effort infrastructure: a missing binary, non-zero exit, or
garbage output must never crash the agent loop. A timeout (NX_HOOK_TIMEOUT,
default 10s) bounds a hung hook. Never set? Zero overhead — not even a
subprocess.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Optional

from ._env import _env_int


def _hook_timeout() -> int:
    """Read at call time (not import time): tests scrub NX_* env wholesale
    during pytest collection, and NX_HOOK_TIMEOUT must reflect the value in
    force when the hook actually runs."""
    return _env_int("NX_HOOK_TIMEOUT", 10)


def _hook_cmd(kind: str) -> str | None:
    v = os.environ.get(kind, "")
    return v.strip() or None


def _run_hook(cmd: str, payload: dict[str, Any], timeout: int) -> tuple[bool, str]:
    """Run a hook; returns (ok, message). ok=False means the hook vetoed/failed."""
    try:
        proc = subprocess.run(
            ["sh", "-c", cmd],
            input=json.dumps(payload, default=str),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"{kind_name(payload)} hook timed out after {timeout}s"
    except OSError as e:
        # can't even spawn the hook — veto rather than silently skip a policy
        return False, f"hook could not run: {e}"
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = msg[0][:200] if msg else f"exit {proc.returncode}"
        return False, detail
    return True, ""


def kind_name(payload: dict[str, Any]) -> str:
    return payload.get("tool", "tool")


def pre_tool_hook(tool: str, args: dict[str, Any]) -> Optional[str]:
    """Return a veto reason string, or None to proceed."""
    cmd = _hook_cmd("NX_PRE_TOOL_HOOK")
    if not cmd:
        return None
    ok, msg = _run_hook(cmd, {"tool": tool, "args": args}, _hook_timeout())
    if ok:
        return None
    from ._subagent import _subagent_quiet
    if not _subagent_quiet():
        print(f"  ↳ pre-hook blocked {tool}: {msg}")
    return f"blocked by NX_PRE_TOOL_HOOK ({msg})"


def post_tool_hook(tool: str, args: dict[str, Any], result: str) -> None:
    """Fire-and-forget observer. Failures are logged, never raised."""
    cmd = _hook_cmd("NX_POST_TOOL_HOOK")
    if not cmd:
        return
    ok, msg = _run_hook(cmd, {"tool": tool, "args": args, "result": result}, _hook_timeout())
    if not ok:
        from ._logging import _log_event
        _log_event("hook", {"tool": tool, "post_hook_error": msg})
