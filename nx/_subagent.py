"""The spawn_agent tool: run a focused sub-agent turn loop. Read-only by
default, with an opt-in write tier (NX_SUBAGENT_WRITE=1).

A sub-agent is a fresh `messages` list (scoped SYSTEM + the instruction) run
through the same `_run_model_turn_loop` machinery, but with a READ-ONLY toolset
(derived from _CACHEABLE_READ_TOOLS — the tools that never touch state) plus a
read-only bash tier — and YOLO semantics inside the loop, so a read-only
subtask never prompts and never mutates anything. The sub-agent's final answer
comes back as the tool result — the parent's context stays clean of the
exploration (reads, greps, dir walks) the subagent did, which is the point:
delegation without context bloat.

Read-only bash tier: run_bash IS offered, but the command must pass
_subagent_bash_allowed() — a single bare command from an allowlist anchored on
_risk.py's own _SAFE_BASH_COMMANDS table (grep/ls/du/ps/git log/...). Any
metacharacter, write-shaped command, or unknown binary is refused before it
ever reaches run_bash, so a sub-agent can scan logs and process state without
write capability. wait_background stays excluded (nothing it runs can
background).

Concurrent sub-agents: a parent turn containing 2+ spawn_agent calls fans them
out via the _run_read_only_parallel thread pool (_turn.py). Because the
dispatch-table swap below was process-global (a second agent starting mid-loop
would clear the first's restricted table), the "sub-agent is active / which
table" state moved to per-agent threading.local via _subagent_ctx(); run_tool()
reads the TLS-first context, so simultaneous agents each see their own
restricted table. Thread-local falls back to the globals for tests/REPL code
that sets them directly.

Deliberately NOT offered to the sub-agent: final_answer's forced-final mode
(we reuse it), wait_background, and every mutating tool (write/edit/sed/patch/
memory_write) UNLESS the write tier is on (NX_SUBAGENT_WRITE=1). With the
write tier the four file-mutation tool schemas are offered and resolve to
confirm-through wrappers (_write_tier_dispatch): they run the parent's real
body, whose own _confirm() — the same risk-classified Y/n gate with a diff
preview the parent shows — appears while the sub-agent loop is suspended.
Only write/edit/multi_edit/sed_edit are tier-eligible: delete_file is not
(removal is one `edit away` from a recreate, while a delete needs its own
undo snapshot the parent should own), and spawn_agent stays nested-refused.
Because every write still crosses the real approval gate, the sub-loop is
NOT forced YOLO when the tier is on — a sub-agent inherits the parent's
approval posture exactly. That means the wrapper bodies must run outside the
sub-agent context (nested spawn_agent inside a wrapper body would be a
second loop on the same thread): the context is torn down before the body
runs and restored after, so _confirm() sees the parent's approval state.
Esc at a prompt still
raises _EscToChat, which run_tool catches and converts into a DENIED tool
result the sub-loop continues from.
"""
from __future__ import annotations

import json
import shlex
import threading
from typing import Any

from . import _state as st

SUBAGENT_SYSTEM = (
    "You are a focused read-only sub-agent answering ONE question about this "
    "codebase for a parent agent. Read files, search, and inspect; then reply "
    "in plain prose with exactly what was asked (key facts, paths, line "
    "numbers) and nothing else — no preamble, no plan narration, no "
    "next-steps. You cannot write or edit; run_bash is limited to a "
    "read-only allowlist (grep, ls, du, ps, git log, ...)."
)

SUBAGENT_WRITE_SYSTEM = (
    "You are a focused sub-agent completing ONE task for a parent agent. You "
    "may write: write_file, edit_file, multi_edit, and sed_edit are available "
    "(no delete_file, no nested spawn_agent), and every change is shown to "
    "the user as a diff and confirmed by them — the sub-agent pauses at each "
    "prompt. run_bash is limited to a read-only allowlist (git diff, grep, "
    "ls, ...). Read files, make the edits, verify, then reply in plain prose "
    "with what changed (paths + what each edit did) and anything the parent "
    "must finish itself — no preamble, no plan narration."
)

SUBAGENT_TOOL = {"type": "function", "function": {
    "name": "spawn_agent",
    "description": "Delegate a focused investigation or task to a sub-agent that returns only its findings. Read/search tools plus a read-only bash allowlist (grep/ls/du/ps/git log). It can WRITE only when the user opted in via NX_SUBAGENT_WRITE=1 (write_file/edit_file/multi_edit/sed_edit, each still confirmed by the user with a diff preview); otherwise it cannot write — report what to change and let the parent do it. Multiple spawn_agent calls in one turn run concurrently (write-capable ones run one at a time). Use it when the work's output would bloat this conversation; do it yourself when the answer is one read_file away.",
    "parameters": {"type": "object", "properties": {
        "instruction": {"type": "string", "description": "The exact question or task for the sub-agent, with any needed paths/context."},
        "max_turns": {"type": "integer", "description": "Cap on sub-agent tool turns (default 12; 1-40)."},
    }, "required": ["instruction"]},
}}

# Tools the sub-agent may call: the read-only dispatch entries (derived from
# the same tuple run_tool() result-caches, plus list_dir) + run_bash, which is
# gated to the read-only allowlist by _subagent_bash(). Never mutating tools,
# never wait_background.
_READ_ONLY_TOOL_NAMES = (
    "read_file", "list_dir", "search_files", "git_status", "git_diff", "git_log",
    "memory_search", "run_bash",
)

# Per-agent sub-loop context. A global flag/table can't express two agents
# running at once (the second's finally would clear the first's table), so
# spawn_agent() installs its restricted table on a threading.local slot and
# run_tool() resolves tool calls through it. Falls back to the plain globals
# for code (tests, REPL) that sets them directly.
_SUBAGENT_TLS = threading.local()

# Tools offered only when the write tier is on (NX_SUBAGENT_WRITE=1): the
# schemas come from the parent TOOLS list, but inside a tier-enabled sub-loop
# they resolve to the confirm-through wrappers in _write_tier_dispatch().
# delete_file is deliberately excluded (removal, not creation — a delete is
# one edit away from a recreate, but its undo snapshot belongs to the parent);
# so are apply_patch (one call already covers multi-file restructures),
# memory_write, and run_bash (the read-only allowlist tier covers inspection).
_WRITE_TIER_TOOL_NAMES = ("write_file", "edit_file", "multi_edit", "sed_edit")

# Serializes the process-global YOLO/APPROVE_LEVEL snapshot+swap around a
# sub-agent loop (TLS handles per-agent state; this only guards the globals).
_SPAWN_LOCK = threading.Lock()


class _SubagentContext:
    """One running sub-agent's restricted-dispatch state (thread-local)."""

    __slots__ = ("table", "active", "write")

    def __init__(self, table: dict[str, Any], write: bool = False) -> None:
        self.table = table
        self.active = True
        self.write = write  # write tier (NX_SUBAGENT_WRITE=1): write tools resolve


def _subagent_ctx() -> "_SubagentContext | None":
    return getattr(_SUBAGENT_TLS, "ctx", None)


def _subagent_table() -> dict[str, Any] | None:
    """The restricted dispatch table for THIS thread, or None outside a
    sub-agent loop. TLS wins over the global flag (concurrent agents)."""
    ctx = _subagent_ctx()
    if ctx is not None:
        return ctx.table
    return st._SUBAGENT_DISPATCH


def _subagent_active() -> bool:
    if _subagent_ctx() is not None:
        return True
    # global fallback is for serial callers/tests that set it directly; under
    # concurrent agents it's racy (a sibling's True is visible here), so TLS
    # is the authoritative signal and the global is never read on agent
    # threads' entry paths.
    return bool(st._SUBAGENT_ACTIVE)


def _subagent_tools() -> list[dict[str, Any]]:
    """Tool schemas for the sub-loop: read-only entries, plus the write tier
    when NX_SUBAGENT_WRITE=1 (only when a write-capable sub-agent is actually
    being spawned — see spawn_agent)."""
    import nx as _m
    wanted = set(_READ_ONLY_TOOL_NAMES)
    if _write_tier_enabled():
        wanted.update(_WRITE_TIER_TOOL_NAMES)
    return [t for t in _m.TOOLS if t["function"]["name"] in wanted]


def _read_only_dispatch() -> dict[str, Any]:
    from ._dispatch import DISPATCH
    table = {name: DISPATCH[name] for name in _READ_ONLY_TOOL_NAMES if name in DISPATCH}
    table["run_bash"] = _subagent_bash  # allowlist gate in front of run_bash
    return table


def _write_tier_enabled() -> bool:
    """NX_SUBAGENT_WRITE=1 opts into write-capable sub-agents. Read from the
    top-level module attribute first (resolved from the env at import, like
    NX_SUBAGENTS) so tests/REPL code can flip it live; the env is the
    fallback when the attribute isn't there (partially-initialized module)."""
    import nx as _m
    try:
        return bool(int(_m.NX_SUBAGENT_WRITE))
    except (AttributeError, TypeError, ValueError):
        from ._env import _env_int
        return bool(_env_int("NX_SUBAGENT_WRITE", 0))


def _write_tier_dispatch() -> dict[str, Any]:
    """Dispatch table for a write-tier sub-agent: the read-only entries (bash
    still allowlist-gated — git diff/grep for verifying an edit) plus the four
    file-mutation tools bound to _write_tier_wrapper, which runs the parent's
    real tool body OUTSIDE the sub-agent context. The body's own _confirm()
    then hits the normal gate — risk classification, the Y/n prompt with the
    same colored diff preview the parent's writes show — so a sub-agent edit
    is exactly as approved as a parent edit, just tagged (sub-agent) in the
    tool box."""
    table = _read_only_dispatch()
    for name in _WRITE_TIER_TOOL_NAMES:
        table[name] = _write_tier_wrapper(name)
    return table


def _write_tier_wrapper(name: str):
    """Build the confirm-through binding for one write-tier tool.

    Closures over the parent's real body (resolved at spawn time) so a test
    or hook that swaps nx.write_file still governs what actually runs. The
    wrapper tears down this thread's sub-agent context around the body so
    _confirm() sees the parent's real approval state (and a nested
    spawn_agent from inside a hook would be refused, not looped)."""
    from ._dispatch import DISPATCH

    def _run(**args: Any) -> str:
        body = DISPATCH.get(name)
        if body is None:
            return f"ERROR: {name} is unavailable"
        ctx = _subagent_ctx()
        if ctx is None:  # defensive: a wrapper escaped its sub-agent context
            return f"ERROR: {name} write-tier wrapper used outside a sub-agent"
        prev_ctx = _SUBAGENT_TLS.ctx
        _SUBAGENT_TLS.ctx = None  # body + _confirm() run as the parent
        try:
            return body(**args)
        finally:
            _SUBAGENT_TLS.ctx = prev_ctx

    _run.__name__ = f"_subagent_write_{name}"
    return _run


# Read-only binaries the sub-agent tier may run as a single bare command
# (no pipes/redirection/&&/globs — the same no-metacharacters constraint
# _risk.py applies before its own allowlist). Deliberately EXCLUDES the
# write-capable entries the parent's _SAFE_BASH_COMMANDS contains for its
# approval fast path (mkdir, touch, echo, env, ...) — a sub-agent has no
# user watching a Y/n prompt, so the allowlist itself is the safety.
# git commands are resolved through _local_risk_hint_bash() instead (see
# _subagent_bash_allowed) so only its read-only verdicts pass.
_SUBAGENT_BASH_ALLOW = frozenset((
    "grep", "egrep", "fgrep", "rg", "ls", "pwd", "cat", "head", "tail", "wc",
    "find", "du", "df", "stat", "file", "diff", "tree", "realpath", "basename",
    "dirname", "which", "type", "ps", "pgrep", "md5sum", "sha256sum",
))


def _subagent_bash_allowed(command: str) -> bool:
    """True when `command` is a single bare read-only command the sub-agent
    tier may run: one allowlisted binary (_SUBAGENT_BASH_ALLOW) or a git
    invocation the parent's own _risk.py classifier scores "low"
    (status/diff/log/show/tag/blame, listing-only branch/remote). Anything
    else — pipes, &&, redirection, globs, env-prefixing, unknown binaries,
    write-shaped commands — is refused; the sub-agent reports what it needs
    and the parent runs it."""
    if not isinstance(command, str) or not command.strip():
        return False
    from ._risk import (_local_risk_hint_bash, _strip_shell_comment,
                        _SHELL_METACHARACTERS_RE)
    cmd = _strip_shell_comment(command.strip())
    if _SHELL_METACHARACTERS_RE.search(cmd):
        return False
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return False
    if not parts:
        return False
    argv = parts[1:]
    head = parts[0]
    if head == "git":
        hint = _local_risk_hint_bash(cmd)
        return hint is not None and hint[0] == "low"
    if head not in _SUBAGENT_BASH_ALLOW:
        return False
    # find is allowlisted only in its listing form; these flags delete files or
    # execute arbitrary commands (the same veto _risk.py's classifier applies).
    if head == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir")
                              for a in argv[1:]):
        return False
    return True


def _subagent_bash(command: str, **_: Any) -> str:
    """run_bash replacement installed in the sub-agent's dispatch table: runs
    the command only when it passes the read-only allowlist. The sub-loop
    already forces YOLO (read-only tools never prompt), and an allowlisted
    single command can't carry redirection/chaining, so there is no approval
    path to reason about. wait_background is not offered at all — if an
    allowlisted command somehow outlives the poll window, the [background]
    result tells the sub-agent to read the log with read_file."""
    if not _subagent_bash_allowed(command):
        return ("ERROR: sub-agent bash is limited to single read-only commands "
                f"(grep/ls/du/ps/pgrep/git log/git status/...); refused: {command!r}. "
                "Report what you need and let the parent run it.")
    from ._bash_jobs import run_bash
    return run_bash(command)


def spawn_agent(instruction: str, max_turns: int = 12, **_: Any) -> str:
    """Run a sub-agent on `instruction`; return its final answer. Read-only by
    default; NX_SUBAGENT_WRITE=1 upgrades the agent to the write tier (the
    four file-mutation tools, each still gated by the parent's _confirm())."""
    if not isinstance(instruction, str) or not instruction.strip():
        # flash-tier models keep sending {"prompt": ...} instead of the
        # documented `instruction` key; without a hint they burn a whole
        # sub-turn learning the schema by trial and error (seen live on
        # glm-5.3-flash).
        return ("ERROR: instruction is empty (spawn_agent takes {\"instruction\": "
                "\"...\"} — you may have used a different key, e.g. `prompt`; "
                "retry with `instruction`)")
    try:
        max_turns = max(1, min(40, int(max_turns)))
    except (TypeError, ValueError):
        max_turns = 12
    if st.client is None or st.MODEL is None:
        return "ERROR: no active model source"
    if st.YOLO is None:  # pragma: no cover — defensive; YOLO is always a bool
        return "ERROR: approval state unavailable"
    if _subagent_ctx() is not None:  # no recursive fan-out from inside a sub-agent
        return "ERROR: nested spawn_agent refused — sub-agents cannot delegate"

    from ._turn import _run_model_turn_loop  # import-check only (circular-import canary)
    import nx as _m  # lazy: monkeypatch-friendly, same pattern as _repl.py
    # Write tier: opt-in (NX_SUBAGENT_WRITE=1) and refused from inside another
    # sub-agent (only the parent ever holds the approval gate).
    write = _write_tier_enabled()
    if write and _subagent_ctx() is not None:
        return ("ERROR: write-tier spawn_agent refused — sub-agents cannot "
                "delegate; report the change and let the parent make it")
    messages = [{"role": "system", "content": SUBAGENT_WRITE_SYSTEM if write else SUBAGENT_SYSTEM},
                {"role": "user", "content": instruction.strip()}]
    # Force the sub-loop through the restricted dispatch table. run_tool() and
    # model_turn() consult the per-thread sub-agent context first (so two
    # concurrent agents never clobber each other's restricted table). The
    # global st._SUBAGENT_DISPATCH is deliberately NOT written here: a
    # concurrent sibling's run_tool("spawn_agent") would see this thread's
    # restricted table mid-flight and refuse itself. The global stays a
    # manual, serial-only fallback for tests.
    table = _write_tier_dispatch() if write else _read_only_dispatch()
    prev_ctx = _subagent_ctx()
    _SUBAGENT_TLS.ctx = _SubagentContext(table, write=write)
    # YOLO/APPROVE_LEVEL are process-global: snapshot+swap under the spawn lock
    # so two concurrent agents can't restore each other's snapshot out of order
    # (a leaked YOLO=True would silently disable the parent's approval gate).
    # Read-only agents force YOLO (their tools never prompt anyway). Write-tier
    # agents inherit the parent's real approval posture instead — the confirm-
    # through wrappers must see it, and they also drop the TLS context around
    # each body, so _confirm() runs exactly as it would for a parent write.
    with _SPAWN_LOCK:
        prev_yolo, prev_level = st.YOLO, st.APPROVE_LEVEL
        if write:
            st.YOLO, st.APPROVE_LEVEL = prev_yolo, prev_level
        else:
            st.YOLO, st.APPROVE_LEVEL = True, "high"  # read-only tools never prompt anyway
    try:
        summary = _m._run_model_turn_loop(
            messages,
            max_turns_override=max_turns,
            on_step=None,
        )
    except Exception as e:  # noqa: BLE001 — a failed subagent is a tool error, not a crash
        return f"ERROR: sub-agent failed: {type(e).__name__}: {e}"
    finally:
        _SUBAGENT_TLS.ctx = prev_ctx
        with _SPAWN_LOCK:
            st.YOLO, st.APPROVE_LEVEL = prev_yolo, prev_level
    # Extract the last assistant text as the answer (the loop's final turn).
    answer = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and msg["content"].strip():
            answer = msg["content"].strip()
            break
    status = summary.get("status") if isinstance(summary, dict) else "done"
    turns = summary.get("turns", 0) if isinstance(summary, dict) else 0
    if not answer:
        return (f"[sub-agent produced no answer after {turns} turn(s); "
                f"status={status} — try a narrower instruction]")
    if status == "cap":
        return (f"[sub-agent hit its {max_turns}-turn cap; partial answer]\n{answer}")
    return answer
