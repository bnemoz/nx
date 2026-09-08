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

Async (background) sub-agents: `spawn_agent(async=true)` returns a job id
immediately and runs the same turn loop on a daemon thread. The finished
answer is announced exactly like a finished bash job — a one-time [Runtime
note] appended to the parent's messages on a later turn — so long
investigations stop blocking the parent loop. The note carries only a short
head of the answer plus the path of a full markdown report the worker wrote
under SESSION_HOME/agents/<job_id>.md (same head+path contract as
_overflow.py's spill), so a fan-out of agents costs a few hundred chars of
parent context, not their whole answers. Since 0.18.8 the background mode is
the DEFAULT (NX_SUBAGENT_ASYNC=0 restores the synchronous default): the old
foreground behavior rendered every sub-agent's spinner, reasoning collapses
and stats footers inside the parent TUI. Async stays read-only even when
the write tier is on (nobody is watching a Y/n prompt on a detached thread).

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

import os
import shlex
import threading
import time
from typing import Any

from ._private import atomic_write_private, private_dir
from . import _state as st
from ._env import _env_int
from ._progress import note_progress

SUBAGENT_SYSTEM = (
    "You are a focused read-only sub-agent answering ONE question about this "
    "codebase for a parent agent. Read files, search, and inspect; then reply "
    "in plain prose with exactly what was asked (key facts, paths, line "
    "numbers) and nothing else — no preamble, no plan narration, no "
    "next-steps. Keep the investigation tight: the fewest reads that answer "
    "the question, batching independent lookups into parallel tool calls in "
    "one turn. Answer from what you have rather than exhausting every lead. "
    "You cannot write or edit; run_bash is limited to a read-only allowlist "
    "(grep, ls, du, ps, git log, ...)."
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

# Session-start delegation guidance, injected ONLY when the spawn_agent schema
# is actually offered (NX_SUBAGENTS=1) — see _dispatch._initial_messages().
# The tool description alone wasn't enough: models left spawn_agent unused and
# ground through dozens of sequential read_file/search_files calls themselves
# unless the user explicitly asked for a fan-out. A schema description reads as
# a *permission*, this reads as an *expectation*, and it's the layer that
# carries the trigger list (unfamiliar codebase, cross-module trace, callers of
# a symbol, 2+ unrelated questions). Kept out of the static SYSTEM string so a
# sub-agents-disabled session (the default) pays nothing for it.
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


def _delegation_enabled() -> bool:
    """True when the spawn_agent schema is being offered this session.
    Resolved through the top-level module attribute (NX_SUBAGENTS) so
    tests/REPL can flip it live; falls back to the env var."""
    import nx as _m
    try:
        return bool(_m.NX_SUBAGENTS)
    except AttributeError:
        return bool(_env_int("NX_SUBAGENTS", 0))


def _load_delegation_note() -> dict[str, str] | None:
    """User-role session-start note with the delegation triggers, or None when
    sub-agents are disabled (nothing to steer toward)."""
    if not _delegation_enabled():
        return None
    return {"role": "user", "content": DELEGATION_GUIDANCE}


def _add_delegation_note_if_absent(messages: list) -> None:
    """Give a resumed session the same note a fresh one gets (its saved
    transcript may predate the feature). Mirrors
    _memory_cmd._add_memory_note_if_absent's marker-scan contract."""
    for m in messages:
        if (m.get("role") == "user" and isinstance(m.get("content"), str)
                and "[Delegation note:" in m["content"]):
            return
    note = _load_delegation_note()
    if note is not None:
        at = 1 if (messages and messages[0].get("role") == "system") else 0
        messages.insert(at, note)


# The default sub-agent budget and its hard ceiling. A default of 12 turns
# was the other half of the 0.19 cap-outs: real investigations (orient, read
# 3-4 files, search, verify, answer) need ~15-20 turns, and the parent model
# kept passing the schema default through unexamined. 100 gives a read-heavy
# investigation plenty of room, recovery cycles included; the ceiling is
# env-tunable (NX_SUBAGENT_MAX_TURNS, 200) instead of hard-coded. Resolved through
# the top-level module attributes (NX_SUBAGENT_DEFAULT_TURNS /
# NX_SUBAGENT_MAX_TURNS in nx/__init__.py) so tests/REPL can flip them live.
#
# The default is a FLOOR, not only the omitted-value fallback. Several models
# carry their own notion of a turn budget and pass a small max_turns (5/10/12
# scale) even for a broad sweep, then watch the sub-agent die mid-investigation
# — honoring that request was the 0.36 cap-out we keep seeing. A request below
# the floor is raised to it; only the ceiling pulls a request back down.


def _subagent_default_turns() -> int:
    import nx as _m
    try:
        return max(1, int(_m.NX_SUBAGENT_DEFAULT_TURNS))
    except (AttributeError, TypeError, ValueError):
        return _env_int("NX_SUBAGENT_DEFAULT_TURNS", 100)


def _subagent_max_turns() -> int:
    import nx as _m
    try:
        return max(1, int(_m.NX_SUBAGENT_MAX_TURNS))
    except (AttributeError, TypeError, ValueError):
        return _env_int("NX_SUBAGENT_MAX_TURNS", 200)

SUBAGENT_TOOL = {"type": "function", "function": {
    "name": "spawn_agent",
    "description": "Hand one investigation to a read-only sub-agent (changes no files) that returns only its findings. PROACTIVELY spawn one per independent area for multi-file work you haven't read -- callers/uses of a symbol, cross-module traces, unfamiliar code. ONE CALL PER QUESTION, all in one turn; they run concurrently. Not for: one known file, one targeted search, edits in progress.",
    "parameters": {"type": "object", "properties": {
        "instruction": {"type": "string", "description": "The one question to answer, with the paths/context needed. The agent cannot see this conversation and cannot ask you back."},
        "max_turns": {"type": "integer", "description": "Tool-turn budget (default 100, and 100 is a FLOOR: a smaller request is raised to it; ceiling 200). Omit it for broad questions — an agent that runs out spends its last turn answering instead of searching."},
        "async": {"type": "boolean", "description": "Background (default): job id now, answer later. false runs it in the foreground, rendering its searches live and blocking until it finishes."},
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

# Recovery-failsafe retries (reasoning-only stalls, empty turns) inside a
# sub-agent loop. The parent's defaults (NX_REASONING_ONLY_RETRIES=3,
# NX_EMPTY_TURN_RETRIES=3) assume a 200-turn budget; a 12-turn sub-agent
# burning half of it on stall rescues is the main way spawn_agent wastes its
# allowance — so inside a sub-loop all recovery cycles (stalls, empty turns,
# malformed streams/collapses) collapse to this (default 1).
_SUBAGENT_RECOVERY_RETRIES = _env_int("NX_SUBAGENT_RECOVERY_RETRIES", 1)


def _subagent_scoped_retry(default: int) -> int:
    """The retry limit THIS thread's loop should use for a stall/empty-turn
    failsafe: the clamp inside a sub-agent, the parent's own setting outside
    it. Batch workers keep the parent values (they already have a wall-clock
    deadline as their budget guard). A parent that DISABLED a failsafe
    (<= 0 — e.g. benchmark mode's "let the model think") keeps it disabled
    inside sub-agents too: the clamp narrows budgets, never widens them."""
    if default <= 0:
        return default
    if _subagent_ctx() is not None and _SUBAGENT_RECOVERY_RETRIES >= 0:
        return _SUBAGENT_RECOVERY_RETRIES
    return default


def _subagent_stream_cut_limit(default: int) -> int:
    """The malformed-stream / collapse retry limit for THIS thread's loop.

    Same clamp philosophy as _subagent_scoped_retry, but for the retry budget
    that WASN'T clamped: the parent's NX_MALFORMED_STREAM_RETRIES (default 2)
    governs malformed tool-call args and repetition/special-token collapses,
    and each cut burns a full sub-agent turn (model_turn returns, the loop
    re-opens the stream). Two cuts + one reasoning rescue = 3 of 12 turns
    spent re-sampling before the agent ever acted — the 0.19 failure pattern.
    A sub-loop with the recovery clamp at 0 (disabled) gets 0 here too: the
    clamp narrows, never widens."""
    if default <= 0:
        return default
    if _subagent_ctx() is not None and _SUBAGENT_RECOVERY_RETRIES >= 0:
        return _SUBAGENT_RECOVERY_RETRIES
    return default


class _SubagentContext:
    """One running sub-agent's restricted-dispatch state (thread-local)."""

    __slots__ = ("table", "active", "write", "quiet")

    def __init__(self, table: dict[str, Any], write: bool = False,
                 quiet: bool = False) -> None:
        self.table = table
        self.active = True
        self.write = write  # write tier (NX_SUBAGENT_WRITE=1): write tools resolve
        self.quiet = quiet  # no terminal writes (async worker threads)


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
    table["run_bash"] = _subagent_bash  # ty: ignore[invalid-assignment]  # allowlist gate in front of run_bash
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


# --- async (background) sub-agents ------------------------------------------
# `spawn_agent(async=true)` runs the same sub-loop on a daemon thread and
# returns a job id immediately. The completed answer is delivered to the
# PARENT as a one-time [Runtime note]: the worker stages it on st the moment
# it finishes, run_tool()'s first-tool-of-turn scan sweeps any stragglers
# (the same hook that runs _scan_background_completions() for bg bash jobs
# and _stage_verify_outcome() for NX_VERIFY_CMD launches), and
# _compress._flush_pending_background_notice appends it to the parent's
# messages with the other runtime notes. Long investigations therefore stop
# blocking the parent loop.
#
# Quiet mode: an async agent's turn loop must not touch the shared terminal —
# no LifeSpinner, no streamed text, no stats footer, no interrupt-watcher
# arming (the singleton stdin watcher is the PARENT's to arm; an agent thread
# clearing/setting it mid-parent-stream would eat keystrokes or break Esc).
# The sub-agent's answer arrives only as the staged [Runtime note], so the
# visible-by-default read-only tool output would be noise; and a sync
# sub-agent spawned BY an async agent is quiet too (its parent thread is
# already detached from the terminal). Quiet mode is read off the per-thread
# sub-agent context, so it composes with concurrent agents (sync fan-out and
# async workers each keep their own context).
_ASYNC_QUIET_SLOTS = threading.local()


def _subagent_quiet() -> bool:
    """True when THIS thread's loop must not write to the shared terminal:
    async-agent worker threads, and any sub-loop running on one."""
    if getattr(_ASYNC_QUIET_SLOTS, "quiet", False):
        return True
    ctx = _subagent_ctx()
    return bool(ctx is not None and getattr(ctx, "quiet", False))
#
# Thread hygiene: the worker runs _run_sync_agent, which installs its own TLS
# sub-agent context and restores it, so concurrent agents (sync fan-out or
# async) each keep their own restricted dispatch table. Async is read-only
# even when NX_SUBAGENT_WRITE=1 — a detached agent has nobody to show a Y/n
# prompt to, and it must not inherit the process-global YOLO/APPROVE_LEVEL
# swap while the parent turn is mid-flight (a leaked YOLO=True would silently
# disable the parent's approval gate — see _SPAWN_LOCK).
# Async answers go to disk: the worker writes a markdown report per job and the
# [Runtime note] carries a short inline head + the report path — the model
# read_file's the full report only when the head isn't enough. Same shape as
# _overflow.py's spill (head + path, full text on disk), minus the tail (the
# report is prose with its conclusion up front, not a grep dump).
_ASYNC_REPORT_CHARS = 600   # inline head of the answer carried in the [Runtime note]
_ASYNC_REPORT_TTL_DAYS = _env_int("NX_OVERFLOW_TTL_DAYS", 7)


def _agents_report_dir() -> str:
    """Directory holding sub-agent reports (st.SESSION_HOME/agents)."""
    return os.path.join(st.SESSION_HOME, "agents")


def _write_agent_report(job_id: str, instruction: str, status: str,
                        answer: str) -> str | None:
    """Write one agent's full answer as markdown; return the path, or None.

    Written for BOTH done and error finalizations — a failed agent's report is
    what the parent reads to decide whether retrying inline is worth it. Runs
    on the worker thread after the sub-loop finishes; never raises (the note
    must go out even if the disk write fails). Temp file + os.replace so a
    concurrent reader never sees a half-written report.
    """
    try:
        d = _agents_report_dir()
        private_dir(d)
        path = os.path.join(d, f"{job_id}.md")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        body = (
            f"# Sub-agent report — {job_id}\n\n"
            f"- status: {status}\n- finished: {now}\n\n"
            "## Instruction\n\n"
            f"{instruction.strip()}\n\n"
            "## Answer\n\n"
            f"{answer.strip()}\n"
        )
        # Sub-agent reports carry the full instruction and answer verbatim
        # (exactly what the parent saw), so they are written like the other
        # sinks: 0600 at creation, umask-proof, atomically replaced.
        atomic_write_private(path, body)
        _purge_old_agent_reports()
        return path
    except OSError:
        return None


def _purge_old_agent_reports(max_age_days: int = _ASYNC_REPORT_TTL_DAYS) -> None:
    """Remove agent reports older than max_age_days (best-effort, mirrors
    _purge_old_overflow)."""
    d = _agents_report_dir()
    try:
        now = time.time()
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if (now - os.path.getmtime(p)) > max_age_days * 86400:
                    os.remove(p)
            except OSError:
                continue
    except OSError:
        pass


# Job id → agent record. Only agents from this process are tracked/announced
# (same posture as the bg map files' owner_pid check). `announced` marks a
# finished answer whose [Runtime note] has been staged — staged once, never
# re-announced on later turns (mirrors _bash_jobs' _BG_NOTICE_SEEN).
_ASYNC_AGENTS: dict[str, dict[str, Any]] = {}
_ASYNC_LOCK = threading.Lock()
_ASYNC_SEQ = 0

# st slot the finished answer is staged into, flushed (and cleared) by
# _compress._flush_pending_background_notice.
_PENDING_NOTICE_KEY = "_PENDING_SUBAGENT_NOTICE"


def _async_running() -> bool:
    """True while any async sub-agent of this process is still executing its
    sub-loop. The wake poller (_autowake.wake_due) holds off while this is
    set: an agent that is mid-run will stage its own note when it finishes,
    and waking the parent for an *unfinished* agent would only make it wait
    synchronously — the exact behavior async spawn is meant to avoid."""
    with _ASYNC_LOCK:
        return any(rec.get("status") == "running" for rec in _ASYNC_AGENTS.values())


def _new_job_id() -> str:
    """A collision-proof job id: monotonic counter + random suffix, so two
    agents spawned in the same instant (a fan-out turn) still get distinct
    ids the model can tell apart."""
    global _ASYNC_SEQ
    import secrets
    with _ASYNC_LOCK:
        _ASYNC_SEQ += 1
        return f"agent-{_ASYNC_SEQ}-{secrets.token_hex(3)}"


def list_async_agents() -> list[dict[str, Any]]:
    """Dashboard-ready view of this process's async sub-agents (newest first):
    {"id", "alive", "instruction", "status", "tail", "report"}. Used by
    /bg, which renders them alongside the _bash_jobs registry entries."""
    with _ASYNC_LOCK:
        recs = sorted(_ASYNC_AGENTS.items(),
                      key=lambda kv: kv[1].get("started") or 0, reverse=True)
    out = []
    for jid, rec in recs:
        status = rec.get("status", "running")
        answer = rec.get("answer") or ""
        out.append({
            "id": jid,
            "alive": status == "running",
            "instruction": rec.get("instruction", ""),
            "status": status,
            "tail": answer[-160:] if answer else "",
            "report": rec.get("report"),
        })
    return out


def _async_duplicate_running(instruction: str) -> str | None:
    """Job id of a still-running async agent with the SAME instruction, or
    None. Flash-tier parents re-spawn an investigation while the original is
    still in flight, doubling the spend and arriving at the same answer twice."""
    norm = " ".join(instruction.split())
    with _ASYNC_LOCK:
        for jid, rec in _ASYNC_AGENTS.items():
            if rec.get("status") == "running" and \
                    " ".join(str(rec.get("instruction", "")).split()) == norm:
                return jid
    return None


def _run_async_agent(job_id: str, instruction: str, max_turns: int) -> None:
    """Worker body: run the sub-loop, then stage the parent's notice.

    On any failure the job is still finalized (status=error) and staged, so
    the parent gets exactly one note — never a dangling, silently-dead agent
    it waits on forever."""
    try:
        answer = _run_sync_agent(instruction, max_turns)
        # _run_sync_agent turns most failures into an "ERROR: sub-agent
        # failed: ..." answer string rather than raising (a failed agent is a
        # tool error, not a crash) — surface those as FAILED to the parent.
        status = "error" if answer.startswith("ERROR: sub-agent failed") else "done"
    except Exception as e:  # noqa: BLE001 — a failed agent is a note, not a crash
        answer = f"{type(e).__name__}: {e}"
        status = "error"
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS.get(job_id)
        if rec is not None:
            rec["answer"] = answer
            rec["status"] = status
            rec["finished"] = time.time()
    # Full answer to disk first (both done and error), so the runtime note can
    # stay a short head + path instead of dumping thousands of chars of
    # exploration into the parent's context.
    try:
        path = _write_agent_report(job_id, instruction, status, answer)
    except Exception:  # noqa: BLE001 - the report is best-effort; the note is not
        path = None
    if rec is not None:
        rec["report"] = path
    _stage_subagent_answer(job_id)


def _stage_subagent_answer(job_id: str) -> None:
    """Stage the one-time [Runtime note] announcing a finished async agent.

    Idempotent: the first caller to run a finished record wins `announced`,
    so the answer is never announced twice (direct staging by the worker and
    the run_tool() sweep both land here). Merges with any earlier unstaged
    note so a burst of finishing agents is announced together (one flush,
    not N)."""
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS.get(job_id)
        if rec is None or rec.get("announced") or rec.get("status") == "running":
            return
        rec["announced"] = True
        status, answer = rec.get("status", "done"), rec.get("answer") or ""
        report = rec.get("report")
        prev = getattr(st, _PENDING_NOTICE_KEY, None)
        if status == "done":
            head = answer[:_ASYNC_REPORT_CHARS]
            marker = " ...[truncated]" if len(answer) > _ASYNC_REPORT_CHARS else ""
            report_line = ""
            if report:
                report_line = "\nFull report: " + report + " — read_file it for the rest."
            note = ("[Runtime note: background sub-agent " + job_id +
                    " finished. Its answer:\n" + head + marker + report_line + "]")
        elif answer.startswith("ERROR: an identical async agent"):
            # duplicate-in-flight refusal: nothing ran, nothing to wait for —
            # the parent must retry later, so don't announce a [Runtime note]
            # with an answer that will never come
            note = "[Runtime note: background sub-agent " + job_id + \
                " was NOT started — refusal: " + answer[:300] + "]"
        else:
            note = "[Runtime note: background sub-agent " + job_id + \
                " FAILED — " + answer[:600] + "\n" + \
                "Retry the task inline (async=false) if it still matters.]"
        setattr(st, _PENDING_NOTICE_KEY, f"{prev}\n{note}" if prev else note)


def _spawn_async(instruction: str, max_turns: int) -> str:
    """Launch the agent's turn loop on a quiet daemon thread and return the
    job id result the parent's tool message carries."""
    job_id = _new_job_id()
    with _ASYNC_LOCK:
        _ASYNC_AGENTS[job_id] = {
            "instruction": instruction, "status": "running",
            "answer": None, "started": time.time(), "announced": False,
        }

    # A background agent's whole point is that its answer lands later; without
    # this the finish notice would sit staged until the user typed something.
    from ._autowake import arm_wake
    arm_wake()

    def _worker() -> None:
        # Quiet: this thread never writes the shared terminal (no spinner,
        # no streamed text, no footer) and never arms the interrupt watcher —
        # those belong to the parent's loop while this agent runs.
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            _run_async_agent(job_id, instruction, max_turns)
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False

    threading.Thread(target=_worker, daemon=True,
                     name=f"nx-subagent-{job_id}").start()
    return (f"[background sub-agent] job {job_id}\n"
            f"instruction: {instruction[:200]}\n"
            "The agent is running in the background; continue other work in "
            "the meantime. Its answer will arrive automatically as a "
            "[Runtime note] on a later turn (short head + path to its full "
            "markdown report) — do not declare the task done until then.")


def _run_sync_agent(instruction: str, max_turns: int) -> str:
    """The synchronous spawn_agent body, factored out so the sync tool call
    and the async worker run identical machinery."""
    if st.client is None or st.MODEL is None:
        return "ERROR: no active model source"
    if st.YOLO is None:  # pragma: no cover — defensive; YOLO is always a bool
        return "ERROR: approval state unavailable"

    from ._turn import _run_model_turn_loop  # import-check only (circular-import canary)
    import nx as _m  # lazy: monkeypatch-friendly, same pattern as _repl.py
    # Write tier: opt-in (NX_SUBAGENT_WRITE=1) and refused from inside another
    # sub-agent (only the parent ever holds the approval gate).
    write = _write_tier_enabled()
    if write and _subagent_ctx() is not None:
        return ("ERROR: write-tier spawn_agent refused — sub-agents cannot "
                "delegate; report the change and let the parent make it")
    # A sub-loop inherits this thread's quiet mode (an async agent's own
    # nested spawn stays off the terminal).
    quiet = _subagent_quiet()
    messages = [{"role": "system", "content": SUBAGENT_WRITE_SYSTEM if write else SUBAGENT_SYSTEM},
                {"role": "user", "content":
                 f"{instruction.strip()}\n\n[Budget: at most {max_turns} tool turns. "
                 "Spend them answering, not exploring: batch independent lookups "
                 "into parallel tool calls within one turn, and wrap up as soon as "
                 "you can support the answer — do not exhaust every lead. Reserve "
                 "the final turn for the answer itself. In your final answer, "
                 "state what you verified and what you did not finish.]"}]
    # Force the sub-loop through the restricted dispatch table. run_tool() and
    # model_turn() consult the per-thread sub-agent context first (so two
    # concurrent agents never clobber each other's restricted table). The
    # global st._SUBAGENT_DISPATCH is deliberately NOT written here: a
    # concurrent sibling's run_tool("spawn_agent") would see this thread's
    # restricted table mid-flight and refuse itself. The global stays a
    # manual, serial-only fallback for tests.
    table = _write_tier_dispatch() if write else _read_only_dispatch()
    prev_ctx = _subagent_ctx()
    _SUBAGENT_TLS.ctx = _SubagentContext(table, write=write, quiet=quiet)
    # YOLO/APPROVE_LEVEL are process-global: snapshot+swap under the spawn lock
    # so two concurrent agents can't restore each other's snapshot out of order
    # (a leaked YOLO=True would silently disable the parent's approval gate).
    # Read-only agents force YOLO (their tools never prompt anyway) — EXCEPT
    # quiet ones: an async agent's swap would outlive the parent turn and
    # silently disable the approval gate for the agent's whole (minutes-long)
    # run, so quiet agents run with the parent's real posture and rely on
    # their read-only toolset never prompting. Write-tier agents inherit the
    # parent's real approval posture (the confirm-through wrappers must see
    # it; they also drop the TLS context around each body, so _confirm()
    # runs exactly as it would for a parent write).
    with _SPAWN_LOCK:
        prev_yolo, prev_level = st.YOLO, st.APPROVE_LEVEL
        if write or quiet:
            st.YOLO, st.APPROVE_LEVEL = prev_yolo, prev_level
        else:
            st.YOLO, st.APPROVE_LEVEL = True, "high"  # read-only tools never prompt anyway
    try:
        # A foreground agent is the longest-running tool call there is (its own
        # model loop, minutes), and until 0.35.0 the parent showed only
        # `running spawn_agent` for all of it. Each step of its loop now posts a
        # turn count to the parent's activity line through note_progress, which
        # lands on st._ACTIVE_SPINNER — set by run_tool() for this very call.
        # A quiet (background) agent has no parent spinner, so it keeps
        # on_step=None rather than paying for a callback that goes nowhere.
        _steps = [0]

        def _progress_step() -> None:
            _steps[0] += 1
            note_progress(f"agent turn {_steps[0]}/{max_turns}")

        summary = _m._run_model_turn_loop(
            messages,
            max_turns_override=max_turns,
            on_step=(None if quiet else _progress_step),
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
        # The agent died mid-exploration (hit the cap while its final turn
        # still called a search tool, or the backend never emitted text). The
        # last tool observation is still signal the parent can act on — hand
        # it up instead of "produced no answer".
        for msg in reversed(messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) \
                    and msg["content"].strip():
                return (f"[sub-agent hit its {max_turns}-turn cap before writing an "
                        f"answer; last observation]\n{msg['content'].strip()[:2000]}")
        return (f"[sub-agent produced no answer after {turns} turn(s); "
                f"status={status} — try a narrower instruction]")
    if status == "cap":
        return (f"[sub-agent hit its {max_turns}-turn cap; partial answer]\n{answer}")
    return answer


def _async_default() -> bool:
    """True when an unspecified spawn defaults to the background. Reads the
    top-level module attribute first (so tests/REPL can flip it live), env as
    fallback. Default is 1 (background): the old synchronous default rendered
    every sub-agent's spinner, reasoning collapses and stats footers inside the
    parent TUI — the main complaint this knob answers. NX_SUBAGENT_ASYNC=0
    restores the foreground default."""
    import nx as _m
    try:
        return bool(int(_m.NX_SUBAGENT_ASYNC))
    except (AttributeError, TypeError, ValueError):
        from ._env import _env_int
        return bool(_env_int("NX_SUBAGENT_ASYNC", 1))


def spawn_agent(instruction: str, max_turns: Any = None, async_: Any = None,
                **extra: Any) -> str:
    """Run a sub-agent on `instruction`; return its final answer. Read-only by
    default; NX_SUBAGENT_WRITE=1 upgrades the agent to the write tier (the
    four file-mutation tools, each still gated by the parent's _confirm()).
    The tool schema's `async` key (a keyword, so run_tool() can only deliver
    it via **extra) defaults to NX_SUBAGENT_ASYNC (default: background) — a
    background spawn returns a job id immediately and delivers its answer as a
    head + report-path [Runtime note] on a later turn."""
    if not isinstance(instruction, str) or not instruction.strip():
        # flash-tier models keep sending {"prompt": ...} instead of the
        # documented `instruction` key; without a hint they burn a whole
        # sub-turn learning the schema by trial and error (seen live on
        # glm-5.3-flash).
        return ("ERROR: instruction is empty (spawn_agent takes {\"instruction\": "
                "\"...\"} — you may have used a different key, e.g. `prompt`; "
                "retry with `instruction`)")
    # The default is a FLOOR, not just the omitted-value fallback (see the note
    # above SUBAGENT_TOOL): models routinely carry their own turn-budget notion
    # and pass a small max_turns for a broad sweep, then the agent dies mid-
    # investigation. Clamp to the ceiling first, then raise to the floor; the
    # floor is itself ceiling-bounded so an inverted config (default > ceiling)
    # can never exceed the ceiling. int(None) raises → the floor, so an omitted
    # max_turns lands on the default.
    _ceiling = _subagent_max_turns()
    _floor = min(_subagent_default_turns(), _ceiling)
    try:
        max_turns = max(_floor, min(_ceiling, int(max_turns)))
    except (TypeError, ValueError):
        max_turns = _floor
    # Models send booleans as real booleans through native tool calls but as
    # "true"/"1" strings through the text-protocol fallback — accept both.
    # The schema key is `async` — a Python keyword, so it can never bind to a
    # named parameter; run_tool() dispatches fn(**args) and it would silently
    # land in **extra and be dropped (the agent would run synchronously).
    raw_async = extra.pop("async", None)
    if raw_async is not None:
        async_ = raw_async
    # None = the model (or a direct caller) omitted `async` - fall to
    # NX_SUBAGENT_ASYNC (background by default since 0.18.8). An explicit
    # false - bool or "false"/"0" string - must WIN over the knob: it is the
    # documented foreground escape hatch.
    if async_ is None:
        async_ = _async_default()
    want_async = async_ is True or (
        isinstance(async_, str) and async_.strip().lower() in ("1", "true", "yes", "on"))
    if want_async:
        if st.client is None or st.MODEL is None:
            return "ERROR: no active model source"
        if _subagent_ctx() is not None:  # no recursive fan-out from inside a sub-agent
            return "ERROR: nested spawn_agent refused — sub-agents cannot delegate"
        dup = _async_duplicate_running(instruction)
        if dup is not None:
            return (f"ERROR: an identical async agent is already running as job {dup} "
                    "— its answer will arrive as a [Runtime note]; do not re-spawn it")
        return _spawn_async(instruction, max_turns)
    return _run_sync_agent(instruction, max_turns)


# --- parent-side completion sweep --------------------------------------------

def _stage_subagent_completions() -> None:
    """Stage [Runtime note]s for async agents that finished since the model
    last looked. Called from run_tool()'s first-tool-of-turn scan — the same
    hook that runs _scan_background_completions() for bg bash jobs and
    _stage_verify_outcome() for NX_VERIFY_CMD launches — and flushed with the
    other pending notices by _compress._flush_pending_background_notice.

    Normally redundant (the worker stages its own answer the moment it
    finishes) but it guarantees an agent that finished between the worker's
    staging and a lost update still announces exactly once — _stage_subagent_answer
    is idempotent per job id."""
    with _ASYNC_LOCK:
        finished = [jid for jid, rec in _ASYNC_AGENTS.items()
                    if rec.get("status") != "running" and not rec.get("announced")]
    for jid in finished:
        _stage_subagent_answer(jid)
