"""Post-edit verification: a free structural check on every .py write, plus an
opt-in NX_VERIFY_CMD that runs the project's own test command after a settled
batch of edits and reports the result back to the model as a [Runtime note].

Part (a) lives in run_tool() (see _dispatch.py): after a successful
write_file/edit_file/multi_edit on a .py file, _ast_check() parses the new
contents in-process — ~1ms, no subprocess — and the outcome is appended to the
tool result, so a syntax error is surfaced on the very turn that caused it
instead of being discovered later (or never).

Part (b) is the settle/batch/launch machinery here: edit tools record their
timestamps in _VERIFY_EDIT_LOG; the turn loop calls _launch_verify_if_settled()
when the model goes quiet; if enough quiet time has passed, the configured
command runs detached via the existing _bash_jobs launcher (run_tool() then
stages its exit status in _PENDING_VERIFY_NOTICE, flushed with the other
runtime notes by _flush_pending_background_notice).

Never raises: verification is best-effort by design and must never turn a
successful edit into an error.
"""
from __future__ import annotations

import ast
import os
import time
from typing import Any

from . import _state as st
from ._env import _env_float

# Timestamps (time.monotonic) of successful .py write_file/edit_file/multi_edit
# calls this session. Consumed/reset by _launch_verify_if_settled().
st._VERIFY_EDIT_LOG = []  # annotated in nx/_state.py
# Set when a verify job is in flight, so a later settle doesn't double-launch
# while one is still running.
st._VERIFY_INFLIGHT = False
# [Runtime note: ...] text staged by the bg-completion scan; flushed with the
# other pending notices by _flush_pending_background_notice.
st._PENDING_VERIFY_NOTICE = None  # annotated in nx/_state.py
# verify-job pids already announced (so a finished verify job is reported once,
# never re-announced on later turns).
st._VERIFY_SEEN = set()  # annotated in nx/_state.py

# Quiet period after the last edit before the verify command fires. Long
# enough to skip dead time between two tool calls in the same burst, short
# enough that the result lands within a turn or two.
_VERIFY_SETTLE_SECS = _env_float("NX_VERIFY_SETTLE_SECS", 20.0)

VERIFY_NOTE_PREFIX = "[Runtime note: NX_VERIFY_CMD"


def verify_enabled() -> bool:
    """NX_VERIFY_CMD is read lazily so /verify set <cmd> (os.environ write)
    takes effect without a restart."""
    return bool(os.environ.get("NX_VERIFY_CMD", "").strip())


def _ast_check(path: str) -> str | None:
    """Parse *path* as Python. Returns None when it parses (or when the check
    doesn't apply), else a one-line 'SYNTAX ERROR ...' message with the
    line/column and the offending text — the model fixes it on the next turn
    with no test run needed."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return None  # vanished mid-flight — not this tool call's problem
    try:
        ast.parse(src, filename=path)
    except SyntaxError as e:
        line = ""
        if e.lineno is not None:
            try:
                line = (src.split("\n")[e.lineno - 1] or "").strip()[:80]
            except IndexError:
                pass
        loc = f"line {e.lineno}" + (f", col {e.offset}" if e.offset else "")
        return f"SYNTAX ERROR in {path} at {loc}: {e.msg}" + (f" → {line}" if line else "")
    except ValueError as e:  # e.g. source with null bytes
        return f"SYNTAX ERROR in {path}: {e}"
    return None


def record_py_edit(path: str) -> str | None:
    """Called by run_tool() after a successful .py edit: (1) the ~1ms in-process
    ast.parse, whose verdict is appended to the tool result; (2) a timestamp in
    the settle log that eventually triggers NX_VERIFY_CMD."""
    problem = _ast_check(path)
    st._VERIFY_EDIT_LOG.append(time.monotonic())
    return problem


def _py_path(path: Any) -> str | None:
    """*path* when it names a .py file we should structurally check (list form
    tolerated — read_file takes lists), else None."""
    if isinstance(path, list):
        path = path[0] if path else None
    if isinstance(path, str) and path.endswith(".py"):
        return path
    return None


def _launch_verify(command: str) -> bool:
    """Start NX_VERIFY_CMD detached via the standard run_bash launcher, so it
    inherits setsid detachment, bg-log rotation, and the completion scan that
    turns its exit status into a [Runtime note]. Returns False when the
    launcher isn't available (run_bash not dispatched, or it returned an
    ERROR:... string — approval veto, no shell). Best-effort, never fatal."""
    import nx as _m
    fn = _m.DISPATCH.get("run_bash")
    if fn is None:
        print("[verify] run_bash unavailable — NX_VERIFY_CMD not run")
        return False
    out = fn(command)  # ty: ignore[missing-argument, invalid-argument-type]  # run_bash(command)
    if not isinstance(out, str) or out.startswith("ERROR") or out.startswith("DENIED"):
        print(f"[verify] NX_VERIFY_CMD did not start: {out if isinstance(out, str) else type(out).__name__}")
        return False
    return True


def _launch_verify_if_settled(force: bool = False) -> None:
    """Call from the turn loop when the model goes quiet. If NX_VERIFY_CMD is
    configured and the last .py edit is ≥_VERIFY_SETTLE_SECS old (or *force*),
    launch it detached once and reset the log so it can't double-fire."""
    if st._VERIFY_INFLIGHT or not verify_enabled():
        return
    log = st._VERIFY_EDIT_LOG
    if not log:
        return
    if not force and (time.monotonic() - log[-1]) < _VERIFY_SETTLE_SECS:
        return
    log.clear()
    st._VERIFY_INFLIGHT = True
    _launch_verify(os.environ["NX_VERIFY_CMD"].strip())


def _is_verify_job(job: dict[str, Any]) -> bool:
    """True when *job* (a list_bg_jobs entry) is an NX_VERIFY_CMD launch from
    this nx process — never another session's strays."""
    cmd = os.environ.get("NX_VERIFY_CMD", "").strip()
    return bool(cmd) and (job.get("command") or "").strip() == cmd \
        and job.get("owner_pid") == os.getpid()


def _stage_verify_outcome() -> None:
    """Scan tracked bg jobs for a finished NX_VERIFY_CMD launch and stage its
    outcome as _PENDING_VERIFY_NOTICE (once per pid). Called from run_tool()'s
    first-tool-of-turn scan; flushed by _flush_pending_background_notice."""
    if st._PENDING_VERIFY_NOTICE or not verify_enabled():
        return
    from ._bash_jobs import list_bg_jobs, _read_bg_exit_status
    for j in list_bg_jobs():
        if not _is_verify_job(j) or j.get("alive"):
            continue
        pid = str(j.get("pid", ""))
        if not pid or pid in st._VERIFY_SEEN:
            continue
        st._VERIFY_SEEN.add(pid)
        st._VERIFY_INFLIGHT = False  # job done — a later settle may launch again
        code = _read_bg_exit_status(pid)[1]
        cmd = j.get("command") or ""
        if code == 0:
            st._PENDING_VERIFY_NOTICE = (
                f"[Runtime note: NX_VERIFY_CMD {cmd!r} passed (exit 0) — "
                "your recent edits verified.]")
        else:
            log_path = j.get("log_path") or ""
            code_s = code if code is not None else "unknown"
            st._PENDING_VERIFY_NOTICE = (
                f"[Runtime note: NX_VERIFY_CMD {cmd!r} FAILED (exit {code_s}) — "
                f"read_file({log_path!r}) for output, fix the breakage, then "
                "re-run it yourself before declaring done.]")
        return
