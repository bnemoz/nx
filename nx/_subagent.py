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

# --- re-exports from the split-out layers (single import surface preserved) ---
from ._delegation_note import (  # noqa: F401 (re-exports)
    DELEGATION_GUIDANCE, DELEGATION_NOTE_MARKER, _add_delegation_note_if_absent,
    _delegation_enabled, _load_delegation_note,
)
from ._subagent_tiers import (  # noqa: F401 (re-exports)
    _READ_ONLY_TOOL_NAMES, _SUBAGENT_TLS, _SubagentContext, _subagent_active,
    _subagent_ctx, _subagent_table, _subagent_tools, _write_tier_dispatch,
    _write_tier_enabled, _write_tier_wrapper, _WRITE_TIER_TOOL_NAMES,
    _read_only_dispatch, _scope_check, _normalize_tier,
)
from ._subagent_bash import (  # noqa: F401 (re-exports)
    _SUBAGENT_BASH_ALLOW, _subagent_bash, _subagent_bash_allowed,
)


from ._prompts import (  # prompt registry (re-exported)
    SUBAGENT_SYSTEM, SUBAGENT_WRITE_SYSTEM, SUBAGENT_IMPLEMENT_SYSTEM,
)

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
        "tier": {"type": "string", "enum": ["read", "write", "implement"], "description": "'read' (default): read-only investigation. 'write' or 'implement': the agent may edit files (needs NX_SUBAGENT_WRITE=1); 'implement' adds the implementer discipline prompt — plan, verify as you go, no churn. In the background (async, the default) they require a yolo session (--yolo); without yolo they are refused — pass async:false so the approval prompt can be shown. With NX_SUBAGENT_WORKTREE=1 a write-tier agent works in its own git worktree: its relative paths, test runs and commits land there, never the shared tree — safe to run several write agents at once; the finished report names the worktree to merge from."},
        "scope": {"type": "array", "items": {"type": "string"}, "description": "Optional path globs (gitignore-style, relative to the project root) the agent's writes are confined to, e.g. ['nx/_turn*.py']. Writes outside the scope are refused with an explicit error. Strongly recommended for write/implement tiers, and required for running more than one write-capable agent."},
    }, "required": ["instruction"]},
}}

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
# Thread hygiene: the worker runs _run_sync_agent_body, which installs its own TLS
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


# Concurrency cap for async fan-out (PF-17, 0.61.0): each background sub-agent
# is its own thread + HTTP stream + parser, so an unbounded fan-out is a
# memory and GIL problem as much as a rate-limit one. A spawn beyond the cap
# parks on this semaphore (its record sits `queued`) instead of racing.
_ASYNC_SEM: "threading.BoundedSemaphore | None" = None
_ASYNC_QUEUED: list[str] = []  # job ids parked on the semaphore, in arrival order

# Finished records are evicted once their answer has been delivered AND the
# parent has moved on (PF-16): the ledger used to hold every answer forever.
# A small floor keeps recent /bg history in a refresher loop.
_MAX_FINISHED_AGENTS = _env_int("NX_SUBAGENT_MAX_FINISHED", 20)


def _async_max_concurrent() -> int:
    """How many async sub-agents may stream at once (NX_SUBAGENT_MAX_CONCURRENT,
    default 4). <=0 means unbounded — the pre-0.61 fan-out."""
    import nx as _m
    n = getattr(_m, "NX_SUBAGENT_MAX_CONCURRENT", None)
    if n is None:
        return _env_int("NX_SUBAGENT_MAX_CONCURRENT", 4)
    try:
        return int(n)
    except (TypeError, ValueError):
        return 4


def _async_semaphore() -> "threading.BoundedSemaphore | None":
    """The process-wide fan-out semaphore, built lazily so the (rare) change
    of the knob between tests is honored rather than frozen at import."""
    global _ASYNC_SEM
    n = _async_max_concurrent()
    if n <= 0:
        return None
    with _ASYNC_LOCK:
        if _ASYNC_SEM is None or _ASYNC_SEM._initial_value != n:  # type: ignore[attr-defined]
            _ASYNC_SEM = threading.BoundedSemaphore(n)
        return _ASYNC_SEM


def _evict_finished_agents() -> None:
    """Drop fully-consumed finished records oldest-first (PF-16, I-8). Pinned
    to a floor of MAX_FINISHED so /bg still has history; running/queued records
    and un-announced answers are never candidates (the parent still needs
    them)."""
    if _MAX_FINISHED_AGENTS <= 0:
        return
    done = [jid for jid, rec in _ASYNC_AGENTS.items()
            if rec.get("status") not in ("running", "queued")
            and rec.get("announced")]
    if len(done) <= _MAX_FINISHED_AGENTS:
        return
    done.sort(key=lambda jid: _ASYNC_AGENTS[jid].get("finished") or 0)
    for jid in done[: len(done) - _MAX_FINISHED_AGENTS]:
        _ASYNC_AGENTS.pop(jid, None)


def _prune_queued() -> None:
    """Stop _ASYNC_QUEUED growing without bound when the parent never reaches
    the spawn that would dequeue it: drop ids whose record is gone or already
    finished. Called under _ASYNC_LOCK."""
    _ASYNC_QUEUED[:] = [jid for jid in _ASYNC_QUEUED
                        if (rec := _ASYNC_AGENTS.get(jid)) is not None
                        and rec.get("status") == "queued"]


def _async_running() -> bool:
    """True while any async sub-agent of this process is still executing its
    sub-loop. The wake poller (_autowake.wake_due) holds off while this is
    set: an agent that is mid-run will stage its own note when it finishes,
    and waking the parent for an *unfinished* agent would only make it wait
    synchronously — the exact behavior async spawn is meant to avoid."""
    with _ASYNC_LOCK:
        return any(rec.get("status") in ("running", "queued")
                   for rec in _ASYNC_AGENTS.values())


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


def _run_async_agent(job_id: str, instruction: str, max_turns: int,
                     tier: str = "read",
                     scope: "list[str] | None" = None) -> None:
    """Worker body: run the sub-loop, then stage the parent's notice.

    On any failure the job is still finalized (status=error) and staged, so
    the parent gets exactly one note — never a dangling, silently-dead agent
    it waits on forever."""
    # Per-agent worktree (0.97.0): a write-tier agent with
    # NX_SUBAGENT_WORKTREE=1 edits/verifies/commits in its own checkout so a
    # headless swarm never tramples the parent's tree or a sibling's. Failure
    # to create one is NOT fatal — the agent falls back to the shared tree
    # (scope gates still apply), and the report says so.
    worktree = None
    wt_note = None
    if tier in ("write", "implement"):
        from ._worktree import _worktree_enabled, create_worktree
        if _worktree_enabled():
            worktree, err = create_worktree(job_id)
            if worktree is None:
                wt_note = (f"worktree isolation unavailable ({err}) — the "
                           "agent ran in the shared tree")
    try:
        answer = _run_sync_agent_body(instruction, max_turns, tier=tier,
                                      scope=scope, worktree=worktree)
        # The unwrapped body: the worker does its OWN report write + [Runtime
        # note] staging below; _run_sync_agent's head+path tool-result wrapper
        # is for the parent's synchronous tool call only. The body turns most
        # failures into an "ERROR: sub-agent
        # failed: ..." answer string rather than raising (a failed agent is a
        # tool error, not a crash) — surface those as FAILED to the parent.
        status = "error" if answer.startswith("ERROR: sub-agent failed") else "done"
    except Exception as e:  # noqa: BLE001 — a failed agent is a note, not a crash
        answer = f"{type(e).__name__}: {e}"
        status = "error"
    if worktree:
        answer += (f"\n\n[Worktree: this agent's edits are in {worktree} "
                   "(detached HEAD from the project root at spawn time) — "
                   "merge or cherry-pick from there; the shared tree is "
                   "untouched.]")
    elif wt_note:
        answer += f"\n\n[{wt_note}]"
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS.get(job_id)
        if rec is not None:
            rec["answer"] = answer
            rec["status"] = status
            rec["finished"] = time.time()
            if worktree:
                rec["worktree"] = worktree
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


def _spawn_async(instruction: str, max_turns: int, tier: str = "read",
                 scope: "list[str] | None" = None) -> str:
    """Launch the agent's turn loop on a quiet daemon thread and return the
    job id result the parent's tool message carries."""
    job_id = _new_job_id()
    sem = _async_semaphore()
    with _ASYNC_LOCK:
        _prune_queued()
        queued = sem is not None and sem._value == 0  # type: ignore[attr-defined]
        _ASYNC_AGENTS[job_id] = {
            "instruction": instruction,
            "status": "queued" if queued else "running",
            "answer": None, "started": time.time(), "announced": False,
        }
        if queued:
            _ASYNC_QUEUED.append(job_id)
            queue_depth = len(_ASYNC_QUEUED)

    # A background agent's whole point is that its answer lands later; without
    # this the finish notice would sit staged until the user typed something.
    from ._autowake import arm_wake
    arm_wake()

    def _worker() -> None:
        # Quiet: this thread never writes the shared terminal (no spinner,
        # no streamed text, no footer) and never arms the interrupt watcher —
        # those belong to the parent's loop while this agent runs.
        acquired = sem is not None
        if sem is not None:
            sem.acquire()
            with _ASYNC_LOCK:
                rec = _ASYNC_AGENTS.get(job_id)
                if rec is not None and rec.get("status") == "queued":
                    rec["status"] = "running"
                    rec["started"] = time.time()
                _prune_queued()
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            _run_async_agent(job_id, instruction, max_turns, tier=tier,
                             scope=scope)
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False
            if acquired and sem is not None:
                sem.release()
            with _ASYNC_LOCK:
                _evict_finished_agents()

    threading.Thread(target=_worker, daemon=True,
                     name=f"nx-subagent-{job_id}").start()
    if queued:
        return (f"[background sub-agent] job {job_id} (queued)\n"
                f"instruction: {instruction[:200]}\n"
                f"{queue_depth} sub-agent(s) queued behind the concurrency cap "
                f"(NX_SUBAGENT_MAX_CONCURRENT={_async_max_concurrent()}). It "
                "starts as soon as a running one finishes; its answer arrives "
                "the same way ([Runtime note] + report path).")
    return (f"[background sub-agent] job {job_id}\n"
            f"instruction: {instruction[:200]}\n"
            "The agent is running in the background; continue other work in "
            "the meantime. Its answer will arrive automatically as a "
            "[Runtime note] on a later turn (short head + path to its full "
            "markdown report) — do not declare the task done until then.")


def _run_sync_agent(instruction: str, max_turns: int, tier: str = "read",
                    scope: "list[str] | None" = None) -> str:
    """The synchronous spawn_agent body, factored out so the sync tool call
    and the async worker run identical machinery.

    Since 0.96.0 BOTH paths apply the same head+path contract (see
    _lean_result): the parent's tool result carries a short head plus the
    report path, never the full text. NX_SUBAGENT_SYNC_INLINE=1 restores the
    old full-text-inline foreground result (handy for tests)."""
    job_id = _new_job_id()
    answer = _run_sync_agent_body(instruction, max_turns, tier=tier, scope=scope)
    if _sync_inline():
        return answer
    return _lean_result(answer, job_id, instruction)


def _sync_inline() -> bool:
    """NX_SUBAGENT_SYNC_INLINE=1 restores the pre-0.96 foreground behavior:
    the full answer inline as the tool result, no report file."""
    import nx as _m
    try:
        return bool(int(_m.NX_SUBAGENT_SYNC_INLINE))
    except (AttributeError, TypeError, ValueError):
        return bool(_env_int("NX_SUBAGENT_SYNC_INLINE", 0))


def _lean_result(answer, job_id, instruction):
    """The head+path tool result shared by both delivery paths (0.96.0): the
    full answer goes to SESSION_HOME/agents/<job>.md and the parent sees a
    short head plus the report path, so a foreground agent's whole
    exploration costs ~600 chars of parent context, not its entire report.
    The model read_file's the report only when the head isn't enough."""
    failed = answer.startswith("ERROR:")
    try:
        path = _write_agent_report(job_id, instruction,
                                   "error" if failed else "done", answer)
    except Exception:  # noqa: BLE001 - the report is best-effort
        path = None
    limit = _ASYNC_REPORT_CHARS
    head = answer[:limit]
    if len(answer) > limit:
        head = head + " ...[truncated]"
    if path:
        head = head + "\nFull report: " + path + " — read_file it for the rest."
    tag = "failed" if failed else "finished"
    return "[sub-agent " + tag + "]\n" + head


def _run_sync_agent_body(instruction: str, max_turns: int, tier: str = "read",
                         scope: "list[str] | None" = None,
                         worktree: "str | None" = None) -> str:
    """The actual foreground sub-loop (wrapped by _run_sync_agent, which adds
    the report-file + head/path staging shared with the async path).
    `worktree` (0.97.0): when set, write-tier calls are rewritten into that
    tree (see _subagent_tiers._worktree_args) — the agent edits its own
    checkout, never the parent's working tree."""
    if st.client is None or st.MODEL is None:
        return "ERROR: no active model source"
    if st.YOLO is None:  # pragma: no cover — defensive; YOLO is always a bool
        return "ERROR: approval state unavailable"

    from ._turn import _run_model_turn_loop  # import-check only (circular-import canary)
    import nx as _m  # lazy: monkeypatch-friendly, same pattern as _repl.py
    # Write tiers (write/implement): opt-in (NX_SUBAGENT_WRITE=1) and refused
    # from inside another sub-agent (only the parent holds the approval gate).
    write = tier in ("write", "implement")
    if write and _subagent_ctx() is not None:
        return ("ERROR: write-tier spawn_agent refused — sub-agents cannot "
                "delegate; report the change and let the parent make it")
    if write and _subagent_quiet() and not st.YOLO:
        return ("ERROR: write-tier spawn_agent refused — a quiet (background/"
                "headless) thread cannot show the approval prompt an edit "
                "raises, and this session is not in yolo mode, so the edits "
                "would be silently auto-denied; spawn it foreground "
                "(async:false) or run the session with --yolo")
    # A sub-loop inherits this thread's quiet mode (an async agent's own
    # nested spawn stays off the terminal).
    quiet = _subagent_quiet()
    system = (SUBAGENT_IMPLEMENT_SYSTEM if tier == "implement"
              else SUBAGENT_WRITE_SYSTEM if write else SUBAGENT_SYSTEM)
    messages = [{"role": "system", "content": system},
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
    ctx = _SubagentContext(table, write=write, quiet=quiet, scope=scope,
                           worktree=worktree)
    from ._subagent_model import install as _install_model_target
    _install_model_target(ctx)
    _SUBAGENT_TLS.ctx = ctx
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
    # A foreground agent is the longest-running tool call there is (its own
    # model loop, minutes), and until 0.35.0 the parent showed only
    # `running spawn_agent` for all of it. Each step of its loop posts a turn
    # count to the parent's activity line through note_progress.
    #
    # But the parent's spinner thread must NOT keep redrawing while the
    # sub-loop owns the terminal: this body runs on the parent thread, so the
    # spinner's 4 fps CLEAR_LINE rewrites interleave with the sub-loop's own
    # spinner, reasoning Live panel and tool-box prints on the same cursor
    # line — the parent line flickers over the agent's output (and vice versa).
    # Same handshake _confirm() uses: stop the spinner around the sub-loop's
    # terminal I/O, restart it after (t0 is stamped once, so the elapsed clock
    # keeps reading as time-in-flight for the whole spawn_agent call). The
    # turn note still lands on spinner.note while stopped and shows the
    # moment the line comes back.
    parent_spinner = st._ACTIVE_SPINNER
    if not quiet and parent_spinner is not None:
        parent_spinner.stop()
    _steps = [0]

    def _progress_step() -> None:
        _steps[0] += 1
        note_progress(f"agent turn {_steps[0]}/{max_turns}")

    try:
        summary = _m._run_model_turn_loop(
            messages,
            max_turns_override=max_turns,
            on_step=(None if quiet else _progress_step),
        )
    except Exception as e:  # noqa: BLE001 — a failed subagent is a tool error, not a crash
        return f"ERROR: sub-agent failed: {type(e).__name__}: {e}"
    finally:
        if not quiet and parent_spinner is not None:
            parent_spinner.start()
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
                tier: Any = None, scope: Any = None, **extra: Any) -> str:
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
    t = _normalize_tier(extra.pop("tier", tier))
    if not t:
        return ("ERROR: tier must be one of read/write/implement "
                "(or omit it — NX_SUBAGENT_WRITE=1 defaults to write)")
    # Explicit read-tier request wins over the env default even when
    # NX_SUBAGENT_WRITE=1; write/implement tiers require the env opt-in.
    if t in ("write", "implement") and not _write_tier_enabled():
        return ("ERROR: write/implement tier requires NX_SUBAGENT_WRITE=1 "
                "(set it in the environment or config, then re-spawn)")
    scope = extra.pop("scope", scope)
    if scope is not None:
        if not (isinstance(scope, list) and scope
                and all(isinstance(x, str) and x.strip() for x in scope)):
            return ("ERROR: scope must be a non-empty list of path glob "
                    "strings (relative to the project root), e.g. "
                    "['nx/_turn*.py']")
        scope = [x.strip() for x in scope if x.strip()]
    # None = the model (or a direct caller) omitted `async` - fall to
    # NX_SUBAGENT_ASYNC (background by default since 0.18.8). An explicit
    # false - bool or "false"/"0" string - must WIN over the knob: it is the
    # documented foreground escape hatch.
    if async_ is None:
        async_ = _async_default()
    want_async = async_ is True or (
        isinstance(async_, str) and async_.strip().lower() in ("1", "true", "yes", "on"))
    if want_async:
        # EXPLICIT refusal, never a silent downgrade or silent auto-denial:
        # a detached thread cannot show the approval prompt a write-tier
        # agent's edits would raise — so async + write tier is refused with
        # the fix spelled out. EXCEPT under YOLO: the approval gate's quiet
        # path (_approval._confirm) auto-approves without prompting when
        # st.YOLO holds, so a yolo headless run CAN host background writers
        # honestly — the user opted out of approval entirely. Refuse only when
        # the posture could not approve a quiet write.
        if t in ("write", "implement") and not st.YOLO:
            return ("ERROR: tier='" + t + "' cannot run in the background — "
                    "a detached thread has nobody to approve its edits (and "
                    "this session is not in yolo mode, so they would be "
                    "silently auto-denied). Re-spawn with async:false, or run "
                    "with --yolo to authorize unattended background writers.")
        if st.client is None or st.MODEL is None:
            return "ERROR: no active model source"
        if _subagent_ctx() is not None:  # no recursive fan-out from inside a sub-agent
            return "ERROR: nested spawn_agent refused — sub-agents cannot delegate"
        dup = _async_duplicate_running(instruction)
        if dup is not None:
            return (f"ERROR: an identical async agent is already running as job {dup} "
                    "— its answer will arrive as a [Runtime note]; do not re-spawn it")
        return _spawn_async(instruction, max_turns, tier=t, scope=scope)
    return _run_sync_agent(instruction, max_turns, tier=t, scope=scope)


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
