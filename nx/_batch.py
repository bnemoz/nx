"""Headless batch mode: `nx run -f tasks.jsonl` — the CI/benchmark harness.

Each line of the input file is ONE task: either a bare JSON string (the
instruction) or a JSON object `{"instruction": "...", "id": "t1",
"session": "my-session"}` (id/session optional). Tasks run through the same
`_run_model_turn_loop` machinery as `nx --prompt-file` one-shot mode — a
fresh messages list, fresh per-task `_Metrics` (installed on the worker
thread's metrics slot — see _state.py — so --parallel workers never race the
process-global recorder), a per-task session id — but N of them per process,
with JSONL results on stdout and a process exit code
that summarizes the whole run:

  0   every task reported status "done"
  3   at least one task hit its turn cap ("cap"); nothing errored
  4   at least one task crashed, was refused, or the endpoint hard-failed
      ("error": task exception, API connection failure, 4xx after fallbacks)
  2   harness-level misuse (unreadable file, zero tasks, no source, bad flag)

Per-task exit codes ride the result lines (`exit_code`), so a consumer that
streams stdout can score each task without waiting for the process to end.

Wall-clock bound: a task is also capped by a per-task timeout — `--timeout
<seconds>` on the command, the task's own `timeout_seconds` field, or
NX_TASK_TIMEOUT_SECONDS (default 0 = off). A lapsed deadline stops the task
at the next model-turn boundary with status "error", an `error` naming the
timeout, and `timed_out: true` on the result line — the turn cap's
forced-final rescue is skipped, so a hung task can't stretch past its
budget by another turn.

Quiet + concurrency: every task's turn loop runs in quiet mode — no spinner,
no streamed text, no stats footer, no interrupt-watcher arming, no tool box —
reusing the async-sub-agent machinery (`_subagent._subagent_quiet()`), so a
`--parallel N` run interleaves tasks on N threads without interleaving their
terminal output. stdout carries ONLY the JSONL result lines; everything
human-facing goes to stderr.

Approval posture: batch is headless — there is nobody to answer a Y/n
prompt — so a task runs with the process's own approval posture and, like a
quiet sub-agent, a prompt can never appear; `_confirm()` auto-denies. Run
batch with `--yolo` (or `NX_APPROVAL=yolo`) when tasks must be allowed to
write and run commands; without it, every gated call comes back
"DENIED by user" and the task reports that.

Sub-agents compose: a task may `spawn_agent(async=true)` exactly as in the
REPL — the answer lands as a [Runtime note] in that task's own transcript.
Since 0.163.0 each task owns its note channel and its background jobs: the
task driver installs a private notice frame (`nx/_notices.py`) and a private
bg-job scope (`nx/_bg_completions.py`) around the whole task, and an async
agent stages into the frame of the turn that spawned it, so a `--parallel N`
sibling can no longer receive (or consume) another task's completion note.

Headless chaining (--autowake, 0.163.0, AUTOWAKE_SPEC Phase E): a task that
backgrounds work does not have to end there. With `--autowake` / the task's
`autowake: true` field the driver starts a SYNTHETIC turn for that task when a
completion note is staged or a sweep finds finished work — the same
user-role instruction and note flush the REPL's wake path uses. The chain is
bounded by the task's own wall-clock deadline and turn cap, which is why an
autowake task with no timeout (NX_TASK_TIMEOUT_SECONDS default 0 = off) is
REFUSED: an unattended, unbounded headless agent is not something to
accidentally configure.
"""
from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ._env import _env_int
from . import _approval
from . import _approval_policy
from . import _state as st
from . import _subagent_model

# Exit codes a batch run (or any one-shot task) can exit with. Exported for
# tests and for consumers that want to mirror the contract.
EXIT_OK = 0        # every task done
EXIT_MISUSE = 2    # bad input: unreadable/empty file, no source, bad flag
EXIT_CAP = 3       # ≥1 task hit its turn cap; nothing errored
EXIT_ERROR = 4     # ≥1 task errored

_MAX_PARALLEL = 64

# Default per-task wall-clock timeout in seconds (0 = off). --timeout or the
# task's own `timeout_seconds` field override it. Off by default so a batch
# without the flag behaves exactly as before; a CI harness that needs a hard
# bound gets one enforced at the next model-turn boundary (a mid-tool-call
# cancel is deliberately NOT attempted).
_TASK_TIMEOUT_DEFAULT = _env_int("NX_TASK_TIMEOUT_SECONDS", 0)

# Per-field caps on a task's JSONL result line. A benchmark consumer streams
# stdout and scores it task by task; a 250KB final_answer (a task that spent
# 85 turns reading the repo) makes one line dominate the stream. The session
# transcript keeps the full text — this caps only the result-line rendition.
_RESULT_ANSWER_CHARS = 4000       # final_answer: head+tail around an elision marker
_RESULT_FIELD_CHARS = 2000        # per tool-call result in the trace

_USAGE_TEXT = """\
usage: nx run -f tasks.jsonl [flags]

Batch mode: run every task in a JSONL file headlessly and emit one JSON
result line per task on stdout. Each input line is one task — a bare JSON
string or an object:

  {"instruction": "refactor foo.py", "id": "t1", "session": "my-session"}

Per-task override keys (object tasks only; each wins over its run flag):
  "approval"         ABSOLUTE posture for this task: "yolo", "low",
                     "medium", "high" (a level headless-auto-approves
                     actions whose LOCAL risk hint is at or below it; a
                     task level inside a --yolo run demotes — e.g.
                     "low" in a --yolo run prompts-deny above low)
  "model"            model for this task, same syntax as --model
  "approval_policy"  "headless-allow:<globs>" for this task (the same
                     value NX_APPROVAL_POLICY takes)
  "autowake"         true|false: auto-continue this task when its background
                     work finishes (the per-task form of --autowake). Requires
                     a timeout (this task's `timeout_seconds`, --timeout or
                     NX_TASK_TIMEOUT_SECONDS) — see --autowake below.

flags:
  -f, --prompt-file <path>   JSONL task file ("-" reads stdin; required)
      --parallel <N>         run up to N tasks concurrently (default 1)
      --continue-on-error    keep going after a task errors (default: stop
                             scheduling new tasks after the first error)
      --yolo / --approval <level>
                             approval posture for the tasks (batch is
                             headless: without yolo, gated calls are DENIED
                             unless a task opts in — see task keys below)
      --model <id|source:id>
                             default model for the tasks (a bare id rides the
                             active source; "source:id" rides that source).
                             A task's own "model" field wins over this.
      --source <name>        endpoint to run the tasks against
      --max-turns <N>        per-task turn cap (default NX_MAX_MODEL_TURNS)
      --timeout <seconds>    per-task wall-clock cap (default
                             NX_TASK_TIMEOUT_SECONDS; 0 = off) — the task
                             stops at the next model-turn boundary and the
                             result line reports status "error", the timeout
                             in `error`, and "timed_out": true
      --autowake             auto-continue a task when its background work
                             finishes (background bash job or async
                             sub-agent): the driver appends the same synthetic
                             wake instruction the REPL uses and runs another
                             turn, until nothing is finished-and-unreported.
                             Off by default (off = unchanged behavior). The
                             chain is bounded by the per-task turn cap AND the
                             per-task timeout, so it REQUIRES a timeout: with
                             no effective timeout each autowake task fails
                             immediately with an error instead of spending
                             unattended forever. A task's own "autowake":
                             true/false field wins over this flag.
      --json                 accepted for symmetry with --prompt-file; JSONL
                             results are always emitted

Per-task exit codes (also on each result line): 0 done, 3 turn cap, 4 error.
Process exit: 0 all done, 3 any cap, 4 any error, 2 harness misuse.
"""


class _BatchUsageError(Exception):
    """A harness-level misuse of `nx run` (bad flag, unreadable file)."""


def _parse_batch_args(argv: list[str]) -> dict[str, Any]:
    """Parse `nx run`'s own flags out of argv (everything after `run`).
    Unknown flags raise _BatchUsageError so a typo can't silently run a
    one-task batch."""
    opts: dict[str, Any] = {
        "file": None, "parallel": 1, "continue_on_error": False,
        "max_turns": None, "timeout": None, "model": None,
        "autowake": False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--prompt-file"):
            if i + 1 >= len(argv):
                raise _BatchUsageError(f"{a} requires a path argument")
            opts["file"] = argv[i + 1]
            i += 2
        elif a == "--parallel":
            if i + 1 >= len(argv):
                raise _BatchUsageError("--parallel requires an integer argument")
            try:
                n = int(argv[i + 1])
            except ValueError:
                raise _BatchUsageError(f"--parallel wants an integer, got {argv[i + 1]!r}")
            if n < 1:
                raise _BatchUsageError("--parallel must be >= 1")
            opts["parallel"] = min(n, _MAX_PARALLEL)
            i += 2
        elif a == "--max-turns":
            if i + 1 >= len(argv):
                raise _BatchUsageError("--max-turns requires an integer argument")
            try:
                n = int(argv[i + 1])
            except ValueError:
                raise _BatchUsageError(f"--max-turns wants an integer, got {argv[i + 1]!r}")
            if n < 1:
                raise _BatchUsageError("--max-turns must be >= 1")
            opts["max_turns"] = n
            i += 2
        elif a == "--timeout":
            if i + 1 >= len(argv):
                raise _BatchUsageError("--timeout requires a seconds argument")
            try:
                t = float(argv[i + 1])
            except ValueError:
                raise _BatchUsageError(f"--timeout wants seconds, got {argv[i + 1]!r}")
            if t < 0:
                raise _BatchUsageError("--timeout must be >= 0 (0 disables the cap)")
            opts["timeout"] = t
            i += 2
        elif a == "--model":
            if i + 1 >= len(argv):
                raise _BatchUsageError("--model requires a model id argument")
            spec = argv[i + 1].strip()
            if not spec:
                raise _BatchUsageError("--model requires a non-empty model id")
            _validate_model_spec(spec, f"--model {spec!r}")
            opts["model"] = spec
            i += 2
        elif a == "--continue-on-error":
            opts["continue_on_error"] = True
            i += 1
        elif a == "--autowake":
            opts["autowake"] = True
            i += 1
        elif a in ("--json",):
            i += 1  # accepted for symmetry; JSONL is always emitted
        elif a in ("--yolo", "--approval", "--source"):
            # consumed globally by nx/__init__.py at import; skip past its value
            i += 2 if a in ("--approval", "--source") and i + 1 < len(argv) else 1
        elif a in ("-h", "--help"):
            raise _BatchUsageError("__help__")
        else:
            raise _BatchUsageError(f"unknown nx run flag: {a!r}")
    if not opts["file"]:
        raise _BatchUsageError("nx run requires -f/--prompt-file <tasks.jsonl>")
    return opts


def _validate_model_spec(spec: str, where: str) -> None:
    """Shared model grammar for the `--model` flag and the task `model` key:
    a bare model id, or "<source>:<model>" with <source> in the registered
    SOURCES registry. An unknown source prefix is a PARSE error (an operator
    authored it — tell them before any tokens burn; sub-agents, by contrast,
    warn-once-and-inherit because a spawn is best-effort)."""
    name, sep, _model = spec.partition(":")
    if sep and name not in st.SOURCES:
        raise _BatchUsageError(
            f"{where}: unknown source {name!r} in model spec {spec!r} "
            f"(registered sources: {', '.join(sorted(st.SOURCES)) or 'none'})")


def _read_task_lines(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise _BatchUsageError(f"cannot read task file {e.filename!r}: {e}")


def _parse_tasks(path: str) -> list[dict[str, Any]]:
    """Parse the JSONL task file into normalized task dicts. Each line must
    be a JSON string (the instruction) or an object with an `instruction`
    key; `id` and `session` are optional. Blank lines and `#` comments are
    skipped. Line numbers are 1-based and preserved in `line`."""
    tasks: list[dict[str, Any]] = []
    task: dict[str, Any]
    for lineno, raw in enumerate(_read_task_lines(path), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as e:
            raise _BatchUsageError(f"task line {lineno} is not valid JSON: {e}")
        if isinstance(parsed, str):
            task = {"instruction": parsed}
        elif isinstance(parsed, dict):
            task = dict(parsed)
            if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
                raise _BatchUsageError(
                    f"task line {lineno} needs a non-empty \"instruction\" string")
            # 0.156.0: the three override keys validate HERE — a bad posture or
            # model must never reach the API (exit 2, no JSONL line). Unknown
            # OTHER keys keep being silently ignored (forward-compat schema).
            # 0.163.0: `autowake` joins them — but as a BOOLEAN, so it gets its
            # own check (a string "false" must not silently mean true).
            if "autowake" in task and not isinstance(task["autowake"], bool):
                raise _BatchUsageError(
                    f"task line {lineno}: autowake must be a boolean, got "
                    f"{type(task['autowake']).__name__}: {task['autowake']!r}")
            for key in ("approval", "model", "approval_policy"):
                if key in task and not isinstance(task[key], str):
                    raise _BatchUsageError(
                        f"task line {lineno}: {key} must be a string, "
                        f"got {type(task[key]).__name__}: {task[key]!r}")
            if "approval" in task:
                kind, _lvl = _approval._normalize_approval(task["approval"])
                if kind is None:
                    raise _BatchUsageError(
                        f"task line {lineno}: invalid approval "
                        f"{task['approval']!r} (expected yolo, low, medium, "
                        f"high, or prompt-all)")
            if "model" in task:
                _validate_model_spec(task["model"].strip(),
                                     f"task line {lineno}: model")
            if "approval_policy" in task:
                spec = task["approval_policy"].strip()
                if not spec.lower().startswith(_approval_policy._PREFIX) \
                        or not spec[len(_approval_policy._PREFIX):].strip():
                    raise _BatchUsageError(
                        f"task line {lineno}: invalid approval_policy "
                        f"{task['approval_policy']!r} (expected "
                        f"\"headless-allow:<globs>\" with non-empty globs)")
        else:
            raise _BatchUsageError(
                f"task line {lineno} must be a JSON string or object, got {type(parsed).__name__}")
        task["line"] = lineno
        if task.get("session"):
            import os
            safe = os.path.basename(str(task["session"]))
            if not safe or safe in (".", "..") or safe != str(task["session"]):
                raise _BatchUsageError(
                    f"task line {lineno}: invalid session id {task['session']!r}")
            task["session"] = safe
        tasks.append(task)
    return tasks


def _batch_active() -> bool:
    """True when THIS thread is running a batch task's turn loop. Checked by
    the shared turn machinery to skip interactive-only behavior (the
    first-tool-of-turn background scan, NX_VERIFY_CMD launches) that belongs
    to the interactive parent — same posture a quiet sub-agent gets via its
    TLS context."""
    return getattr(_TASK_TLS, "task", None) is not None


def _batch_task() -> "dict[str, Any] | None":
    """The task dict for THIS thread's batch task, or None outside batch mode."""
    return getattr(_TASK_TLS, "task", None)


_TASK_TLS = threading.local()

# Quiet-slot guard, set once per process when the first batch task starts:
# a `--parallel N` worker restores `_ASYNC_QUIET_SLOTS.quiet = False` in its
# `finally`, which would silently un-quiet every still-running sibling.
_QUIET_GUARD = threading.Lock()
_QUIET_HOLDERS = 0


def _result_exit_code(status: str | None) -> int:
    """Per-task process-style exit code: 0 done, 3 turn cap, 4 error."""
    if status == "done":
        return EXIT_OK
    if status == "cap":
        return EXIT_CAP
    return EXIT_ERROR


def _task_timeout(task: dict[str, Any], default: float | None) -> float | None:
    """Resolve one task's wall-clock budget in seconds (None = off).
    Precedence: the task's own `timeout_seconds` field, then --timeout,
    then NX_TASK_TIMEOUT_SECONDS. 0 anywhere means off."""
    raw = task.get("timeout_seconds", default)
    if raw is None:
        return None
    try:
        secs = float(raw)
        if secs < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise _BatchUsageError(
            f"task line {task.get('line')}: timeout_seconds must be a "
            f"non-negative number, got {raw!r}") from None
    return secs if secs > 0 else None


def _resolve_model_target(spec: str) -> "tuple[str, tuple[Any, str, dict[str, Any], Any]] | tuple[None, None]":
    """Resolve a batch model spec (bare id, or `source:id` — validated up
    front) into (model_id, call_target) for `call_target_scope`. The same
    normalization `subagent_call_target()` applies to pool entries: a bare id
    rides the active source's client, `source:id` rides that Source's client
    and per-model extra_request_kwargs (0.156.0: the ROUTED model is threaded
    through so thinking separation keys off the task's model, not st.MODEL).
    (None, None) = the run's globals (no spec, or nothing resolvable)."""
    name, sep, model = spec.partition(":")
    if sep:
        src = st.SOURCES.get(name)
        if src is None or getattr(src, "client", None) is None:
            return None, None
        client = src.client
        use_model = model or getattr(src, "model", None)
        if not use_model and hasattr(src, "resolve_model"):
            try:
                use_model = src.resolve_model()
            except Exception:  # noqa: BLE001 — degrade to the globals
                use_model = None
        pin_src = src if hasattr(src, "_request_shapes") else None
    else:
        src = st.ACTIVE
        client = st.client
        use_model = spec
        pin_src = src if (src is not None and hasattr(src, "_request_shapes")) else None
    if not client or not use_model:
        return None, None
    if src is not None and hasattr(src, "extra_request_kwargs"):
        extra = src.extra_request_kwargs(use_model)
    else:
        extra = {}
    return use_model, (client, use_model, extra or {}, pin_src)


def _task_posture(task: dict[str, Any]) -> "tuple[tuple[bool, str | None] | None, str]":
    """Resolve a task's effective approval posture: the task's `approval` key
    is an ABSOLUTE posture (task `"low"` inside a --yolo run demotes — the
    signed-off semantics); absent, the run's own posture (the process globals:
    --yolo/--approval consumed at import) governs, unchanged.

    Returns (scope_or_None, display): scope is the `_posture_scope` force
    tuple when the task set the key, None to keep today's globals; display is
    the effective posture rendered in the NX_APPROVAL vocabulary
    ("yolo" | "low" | "medium" | "high" | "prompt-all")."""
    raw = task.get("approval")
    if raw is None:
        yolo, level = st._posture()
        return None, ("yolo" if yolo else (level or "prompt-all"))
    kind, level = _approval._normalize_approval(raw)
    if kind == "yolo":
        return (True, None), "yolo"
    if kind == "level":
        assert level is not None
        return (False, level), level
    return (False, None), "prompt-all"  # prompt_all


def _task_autowake(task: dict[str, Any], run_autowake: bool) -> bool:
    """Resolve one task's autowake flag: its own field wins over the run flag."""
    raw = task.get("autowake")
    if isinstance(raw, bool):
        return raw
    return run_autowake


def _owner_finished_unstaged_agents(owner_ident: "int | None") -> bool:
    """Async agents SPAWNED BY THIS TASK that finished without their note
    being staged (0.163.0). `nx/_autowake._finished_unstaged_subagents` is the
    interactive twin; the batch version must be owner-scoped or a
    `--parallel N` sibling's finished agent would wake this task (a phantom
    wake by §2's definition). None (no published frame — the driver never got
    this far) scopes to nothing and reports False rather than matching every
    ownerless record in the process."""
    if owner_ident is None:
        return False
    try:
        from ._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
        with _ASYNC_LOCK:
            return any(rec.get("status") not in ("running", "queued")
                       and not rec.get("announced")
                       and rec.get("owner_ident") == owner_ident
                       for rec in _ASYNC_AGENTS.values())
    except Exception:
        return False


def _task_finished_unseen_bg_jobs(jobs) -> bool:
    """Finished, never-announced bg jobs that belong to THIS task (its
    per-task pid scope). Mirrors `nx/_autowake._finished_unseen_bg_jobs`'s
    filters and adds the scope, for the same reason as above."""
    try:
        import os as _os
        from ._verify import _is_verify_job
        from ._bg_completions import bg_scope
        from . import _notices
        from . import _state as _st
        if not jobs:
            return False
        pids = bg_scope()
        seen = _notices.get("_BG_NOTICE_SEEN")
        if seen is None:
            seen = set(getattr(_st, "_BG_NOTICE_BASELINE", set()) or set())
        mine = _os.getpid()
        for j in jobs:
            pid = str(j.get("pid", ""))
            if not pid or pid in seen or j.get("alive"):
                continue
            if j.get("owner_pid") != mine or _is_verify_job(j):
                continue
            if pids is not None and pid not in pids:
                continue
            if not j.get("log_path") or not _os.path.exists(j["log_path"]):
                continue
            return True
    except Exception:
        pass
    return False


def batch_wake_pending(jobs=None) -> bool:
    """The READ-ONLY half of `batch_wake_due` (0.163.1): would a wake turn have
    anything to say, WITHOUT running the completion sweep that answers it.

    Every predicate here is a read: `_any_pending_notice` reads notice slots,
    `_owner_finished_unstaged_agents` reads the agent records, and
    `_task_finished_unseen_bg_jobs` reads the job list plus this task's seen set
    and pid scope (it marks nothing — the sweep does that). A caller that only
    needs to pick a STATUS WORD must use this instead of `batch_wake_due`:
    running the real decision mutates (stages notes, marks pids seen) and a
    driver on its way out drops those notes with the task's frame, so the
    sweep's work would be spent on a decision and then thrown away (the
    0.163.1 review's M1).

    *jobs* lets a caller share an already-computed list_bg_jobs() result."""
    from ._autowake import _any_pending_notice
    owner_ident = getattr(_TASK_TLS, "owner_ident", None)
    if jobs is None:
        try:
            from ._bash_jobs import list_bg_jobs
            jobs = list_bg_jobs()
        except Exception:
            jobs = None
    if _any_pending_notice():
        return True
    if _owner_finished_unstaged_agents(owner_ident):
        return True
    return _task_finished_unseen_bg_jobs(jobs)


def batch_wake_due(jobs=None) -> bool:
    """The batch entry point of the auto-continuation decision (0.163.0,
    AUTOWAKE_SPEC Phase E F2): True when THIS task's driver should start a
    synthetic wake turn.

    `nx/_autowake.wake_due()` is the interactive decision and cannot be
    reused: its liveness guards (`_TURN_ACTIVE`, `_subagent_ctx()`,
    `_batch_active()`, `_async_running()`) all evaluate the shared process
    state from the poller's point of view, and the batch guard at the top of
    it makes it answer False for every batch task — which is exactly right for
    the poller thread and exactly wrong for the driver. §4.3 of the spec is
    explicit that Phase E must key off an explicit entry point rather than the
    ambient thread-local predicate, so this is its own decision:

    * runs the SAME completion sweep `run_tool()` runs on a turn's first tool
      call, threading one `list_bg_jobs()` result through it (the way
      `wake_due()` threads it), so a text-only turn's finished job gets
      announced exactly once;
    * returns True on anything newly staged in THIS task's notice frame, or on
      finished-but-unreported work owned by THIS task (async agent by
      spawning-thread ident, bg job by the task's pid scope);
    * applies NONE of the interactive-only guards — not turn-active (the
      driver is between turns on its own thread by construction), not the
      streak cap (the chain's bound is the task deadline plus the loop's own
      turn cap, both already armed).

    Also deliberately NOT applied, with reasons:
    * Phase C's stall trigger — its guards are the user-returning-to-the-
      keyboard grace period plus a plan revision the model may never touch in
      a headless run, and its "nothing in flight" premise has no owner here.
      A headless chain that ends with work in flight simply ends; §2's
      phantom-wake ban outranks the convenience.
    * Phase D's token budget — its accumulator is fed by the REPL/Telegram
      post-turn seam and reset by a TYPED turn, which a headless task never
      has; wiring a second feed for it would be a new unbounded-in-the-other-
      direction counter. The deadline is the cost bound for batch.

    *jobs* lets a caller that already scanned the pid->log map share it.

    The three predicates below are factored into `batch_wake_pending()` so a
    caller that must NOT mutate (the driver choosing between "done" and
    "cap") can ask the same question read-only."""
    from ._bg_completions import _run_bg_completion_scan
    if jobs is None:
        try:
            from ._bash_jobs import list_bg_jobs
            jobs = list_bg_jobs()
        except Exception:
            jobs = None
    if jobs:
        try:
            _run_bg_completion_scan(jobs=jobs)
        except Exception:
            pass
    return batch_wake_pending(jobs)


def _bg_task_scopes() -> set:
    """This thread's per-task bg-job scope (creating it when absent, so a
    caller inside a task's turn loop — a tool, a test — can register a pid it
    launched). The real driver installs one per task via `bg_scope_install()`;
    this accessor is the stable seam for callers that don't own the lifecycle.
    """
    from ._bg_completions import bg_scope, bg_scope_install
    pids = bg_scope()
    return pids if pids is not None else bg_scope_install()


def _batch_autowake_active() -> bool:
    """True when THIS thread's batch task has autowake enabled. The shared turn
    machinery reads it to decide whether the interactive-parent-only completion
    sweep belongs to a batch turn: off keeps today's gate (no scan), on lets
    the sweep run so a chained task hears about its own finished work."""
    task = _batch_task()
    if task is None:
        return False
    return bool(task.get("_autowake"))


def _reap_task_bg_jobs(pids: "set[str] | None") -> None:
    """Kill every live bg job this task launched (P2 task-end cleanup)."""
    if not pids:
        return
    try:
        from ._bash_jobs import cancel_kill_group, _pid_is_alive
    except ImportError:
        return
    for pid in pids:
        try:
            if _pid_is_alive(int(pid)):
                cancel_kill_group(pid, "")
        except (TypeError, ValueError, OSError):
            pass


def _run_one_task(task: dict[str, Any], run_id: str, idx: int,
                  max_turns: int | None,
                  timeout_seconds: float | None = None,
                  model_default: str | None = None,
                  autowake_default: bool = False) -> dict[str, Any]:
    """Run one task headlessly: fresh messages, fresh metrics, fresh session
    id; the turn loop's chat output is suppressed (quiet). Returns the merged
    result dict (the JSONL line's content). Never raises — a raise is an
    exit_code=4 result.

    timeout_seconds arms a wall-clock deadline on top of the turn cap: the
    loop aborts at the next model-turn boundary with status "error" and
    `timed_out` set on the result line, and the turn cap's forced-final
    rescue is skipped so a hung task cannot stretch one more turn past its
    budget.

    autowake (0.163.0, Phase E): when True, the driver runs a synthetic wake
    turn whenever the completion sweep finds finished work for this task.
    Refuses when no timeout is armed (unbounded headless spend)."""
    import nx as _m  # lazy: the package re-exports what the loop needs
    from ._repl_state import _fresh_metrics
    from ._subagent import _ASYNC_QUIET_SLOTS
    from ._turn import _ensure_ctx_probe, _task_deadline_at
    from ._turn_retry import _task_deadline_exceeded, _task_timeout_summary
    from . import _notices
    from ._bg_completions import bg_scope_install, bg_scope_clear

    instruction = task["instruction"].strip()
    task_id = task.get("id") if isinstance(task.get("id"), str) else f"task-{idx}"
    session_id = task.get("session") or _m._new_session_id()
    global _QUIET_HOLDERS

    result: dict[str, Any] = {
        "index": idx,
        "id": task_id,
        "line": task.get("line"),
        "session": session_id,
        "batch": run_id,
        "exit_code": EXIT_ERROR,
        "status": "error",
        "turns": 0,
        "elapsed_seconds": 0.0,
        "final_answer": None,
        "tokens": None,
        "tool_calls": [],
        "error": None,
        # 0.156.0: per-task provenance — the model that answered and the
        # posture the decisions used (see _task_posture for the rendering).
        "model": st.MODEL,
        "approval": _task_posture(task)[1],
    }
    started = time.time()
    autowake = _task_autowake(task, autowake_default)
    # The result line grows `autowake: true` ONLY for a chained task (off runs
    # stay byte-identical to 0.162.0's stream). Set before the refusal below so
    # a refused autowake task is identifiable in the stream as one too.
    if autowake:
        result["autowake"] = True
    # F4 (0.163.0): refuse an unbounded headless chain rather than spend
    # forever with no keyboard. A clear task error (exit 4), not a crash and
    # not a silent pass — and deliberately BEFORE any per-task state is
    # installed, so the refusal touches no quiet slot, notice frame or job
    # scope (everything below has a matching finally).
    if autowake and timeout_seconds is None:
        result["error"] = (
            "--autowake requires a per-task timeout: the wake chain is bounded "
            "by the task deadline, and with no timeout an unattended headless "
            "task could spend forever. Set --timeout <seconds>, the task's "
            "\"timeout_seconds\" field, or NX_TASK_TIMEOUT_SECONDS (0 means "
            "off), or drop --autowake.")
        result["exit_code"] = EXIT_ERROR
        return result
    # Quiet the whole worker thread (no spinner/stream/footer/tool box) and
    # mark the TLS task slot so the shared turn machinery can skip
    # interactive-only behavior. The REPL never runs inside batch mode, so
    # nothing else contends for the quiet slot. With --parallel, a worker's
    # `finally` may run while siblings are mid-loop — the guard keeps the
    # slot quiet until the LAST batch task on any thread finishes.
    _ASYNC_QUIET_SLOTS.quiet = True
    with _QUIET_GUARD:
        _QUIET_HOLDERS += 1
    _TASK_TLS.task = task
    # Per-task recorder on THIS thread's slot — never the process-global
    # `st._METRICS`: with --parallel > 1 a swap-and-restore of the global
    # races every sibling worker's _record_usage and silently misattributes
    # their token totals (the metrics TLS slot is per thread, like the quiet
    # slot and _TASK_TLS). prev is None except for a nested installation.
    prev_metrics = None
    _task_deadline_at(None if timeout_seconds is None else started + timeout_seconds)
    # 0.156.0: per-task overrides, resolved once and installed as TLS scopes
    # around the try body (all nested-safe, all finally-restored). No task key
    # set → no scope entered → the byte-identical legacy path.
    posture_scope_force, _approval_disp = _task_posture(task)
    model_spec = task.get("model") or model_default
    policy_spec = task.get("approval_policy")
    # 0.163.0 (Phase E): note ownership + job ownership are per TASK, not per
    # process. The frame is what the task's own flush reads and what its
    # agents' notes are routed into (published under this thread's ident);
    # the job scope is the pid set this task launched, which narrows the
    # completion sweep and is killed off at task end. Installed
    # unconditionally (both are prerequisite bug fixes); only the WAKE loop is
    # gated on the autowake flag.
    task["_autowake"] = autowake
    notice_frame = _notices.install_notice_ctx()
    owner_ident = _notices.publish(notice_frame)
    _TASK_TLS.owner_ident = owner_ident
    bg_scope_install()
    scopes = contextlib.ExitStack()
    try:
        # (i) posture: yolo → (True, None); a level or prompt-all → (False, x)
        if posture_scope_force is not None:
            scopes.enter_context(st._posture_scope(posture_scope_force))
        # (ii) model: task field > --model flag > the run's globals (no scope)
        if model_spec:
            model_id, target = _resolve_model_target(model_spec.strip())
            if target is not None:
                scopes.enter_context(_subagent_model.call_target_scope(target))
                result["model"] = model_id
        # (iii) headless threshold opt-in for LEVEL postures only — yolo
        # short-circuits _confirm before the threshold is ever read
        kind = _approval._normalize_approval(task["approval"])[0] \
            if "approval" in task else None
        if kind == "level":
            scopes.enter_context(_approval.task_level_scope(
                _approval._normalize_approval(task["approval"])[1]))
        # (iv) per-task approval policy (headless-allow globs)
        if policy_spec:
            scopes.enter_context(_approval_policy.policy_scope(policy_spec))
        prev_metrics = st._set_thread_metrics(_fresh_metrics(session_id))
        if st.ACTIVE is None:
            raise RuntimeError("no active model source — set NX_BASE_URL/NX_SOURCE_* or use --source")
        _ensure_ctx_probe(st.ACTIVE)
        task_metrics = st._active_metrics()
        messages = _m._initial_messages() + [{"role": "user", "content": instruction}]
        start_idx = len(messages)
        summary = _m._run_model_turn_loop(messages, on_step=None,
                                          max_turns_override=max_turns)
        # --- headless chaining (0.163.0, AUTOWAKE_SPEC Phase E F2/F3) --------
        # The driver — this thread, between turns — decides and acts, exactly
        # like the REPL does; nothing here runs on a poller thread (§2). Each
        # wake is one more `_run_model_turn_loop` call on the SAME messages
        # list, so the transcript (the task's answer channel) grows the way a
        # chained REPL session's does: synthetic user instruction, then the
        # finished work's [Runtime note], then the model's turn.
        #
        # No streak cap: the bound is the wall-clock deadline (armed above,
        # evaluated at every step boundary and re-checked here before each
        # wake) plus the turn cap — and for that second one to be a REAL bound
        # on a chained task it has to be counted ACROSS the chain, since each
        # wake is a fresh `_run_model_turn_loop` call with its own MAX_MODEL_TURNS
        # budget. So the driver accumulates the turns the summaries report and
        # stops at the same number the loop would have stopped at. A chain that
        # stops with finished work still unsaid reports the turn cap ("cap",
        # exit 3) rather than a clean finish; a chain that stops because there
        # is nothing left to report stays "done".
        #
        # Phase C's stall trigger and Phase D's token budget are deliberately
        # NOT applied — the reasons are in batch_wake_due()'s docstring.
        turn_budget = int(max_turns) if max_turns else int(
            getattr(st, "MAX_MODEL_TURNS", 500) or 500)
        total_turns = int(summary.get("turns") or 0) if isinstance(summary, dict) else 0
        wakes = 0
        chain_capped = False
        while autowake and isinstance(summary, dict) and summary.get("status") == "done":
            if _task_deadline_exceeded():
                # A chain that outlived its deadline reports the timeout, not
                # a clean finish — and stops waking (no further spend).
                summary = _task_timeout_summary(total_turns)
                break
            if total_turns >= turn_budget:
                # Only claim the cap if there is genuinely something to say;
                # otherwise the honest answer is still "done". Read-only
                # (0.163.1 M1): this decides a status word on a task that is
                # about to END, and the old `batch_wake_due()` call here ran
                # the real sweep — staging notes and marking pids seen — for
                # notes the task's finally then dropped undelivered.
                chain_capped = batch_wake_pending()
                break
            if not batch_wake_due():
                break
            # 0.163.1 M2: the guard at the top of this loop is not the last
            # moment before the spend — the sweep above (a registry walk plus a
            # log-tail read) runs in between, so a chain that started
            # microseconds before the deadline could still pay for a whole
            # extra model turn. Re-armed here, with nothing but in-process
            # bookkeeping left between the check and the call.
            if _task_deadline_exceeded():
                summary = _task_timeout_summary(total_turns)
                break
            from ._autowake import wake_instruction
            messages.append({"role": "user", "content": wake_instruction()})
            # Same order as the REPL's wake path (_repl.py): the sweep is
            # already done by the decision above, then the notes are flushed
            # up front, because a wake turn may end as plain text and notes
            # otherwise only flush on its first tool call.
            _m._flush_pending_background_notice(messages)
            wakes += 1
            summary = _m._run_model_turn_loop(messages, on_step=None,
                                              max_turns_override=max_turns)
            total_turns += int(summary.get("turns") or 0) if isinstance(summary, dict) else 0
        # Drain whatever the task still owns (0.163.1 review fix 1). Between
        # loops — and on a run with no wake loop at all — this thread's notice
        # frame IS the task's own F0, and an async agent that finished during
        # the FINAL turn staged straight into it (`stage_for_owner`), so with no
        # further turn loop to seed a nested frame from F0 and no wake-loop
        # flush, the note would be dropped by `discard_notice_ctx` below without
        # ever being read (pre-0.163 the worker wrote the module store and the
        # next turn's drain delivered it). One unconditional drain into the
        # task's own transcript covers both paths; every slot it reads is
        # frame-scoped to F0 (or the module store), so it is a no-op when the
        # task has nothing staged, and it can never touch a sibling task's
        # notes — those live in that task's own frame.
        _m._flush_pending_background_notice(messages)
        if wakes:
            result["wakes"] = wakes
        if chain_capped:
            summary = {"status": "cap", "turns": total_turns}
        result.update(_m._one_shot_result(messages, start_idx, summary,
                                          time.time() - started,
                                          metrics=task_metrics))
        if isinstance(summary, dict) and summary.get("timed_out"):
            result["timed_out"] = True
        result["exit_code"] = _result_exit_code(result.get("status"))
        # Persist the transcript exactly like the REPL would — a benchmark
        # task is a real, resumable session.
        _save_task_session(session_id, messages)
    except KeyboardInterrupt:
        result["status"] = "error"
        result["error"] = "interrupted"
        result["exit_code"] = 130
    except Exception as e:  # noqa: BLE001 — a failed task is a result, not a crash
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["exit_code"] = EXIT_ERROR
        result["elapsed_seconds"] = round(time.time() - started, 2)
    finally:
        scopes.close()
        _task_deadline_at(None)
        st._set_thread_metrics(prev_metrics)
        _TASK_TLS.task = None
        # Per-task cleanup, in the order ownership was taken: drop the frame
        # WITHOUT folding (an undelivered note must not leak into the module
        # store for a sibling task to drain), then kill whatever detached jobs
        # this task still owns.
        _notices.unpublish(owner_ident, notice_frame)
        _notices.discard_notice_ctx(notice_frame)
        _TASK_TLS.owner_ident = None
        _reap_task_bg_jobs(bg_scope_clear())
        task.pop("_autowake", None)
        with _QUIET_GUARD:
            _QUIET_HOLDERS -= 1
            if _QUIET_HOLDERS <= 0:
                _ASYNC_QUIET_SLOTS.quiet = False
    return result


def _save_task_session(session_id: str, messages: list[dict[str, Any]]) -> None:
    """Persist a finished task's transcript (best-effort; never raises).

    Reads the task's own recorder via _active_metrics(): with --parallel > 1
    the process-global `st._METRICS` still belongs to whichever REPL (or
    parent process entry point) built it, so merging its totals into this
    task's session JSON would misattribute tokens across sibling tasks."""
    try:
        import nx as _m
        m_obj = st._active_metrics()
        m = dict(m_obj.as_meta()) if m_obj is not None else {}
        m["source"] = st.ACTIVE.name if st.ACTIVE else None
        # 0.156.0: the EFFECTIVE model — the task's routed target when one is
        # installed, else st.MODEL — so a resumed transcript doesn't lie about
        # what answered.
        target = _subagent_model.call_target()
        m["model"] = target[1] if target is not None else st.MODEL
        import os
        m["cwd"] = os.getcwd()
        _m._write_session(session_id, messages, m)
    except Exception:  # noqa: BLE001 — a failed save must not fail the task
        pass


def _trim(text: Any, cap: int) -> Any:
    """Head+tail trim one result-line string to ~cap chars. Non-strings pass
    through untouched. The elision marker names the real size so a consumer
    knows what it didn't get."""
    if not isinstance(text, str) or len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    omitted = len(text) - head - tail
    return (text[:head]
            + f"\n... [{omitted} chars elided; full text in the task's saved session] ..."
            + (text[-tail:] if tail else ""))


def _trim_result_value(value: Any, cap: int) -> Any:
    """_trim for one traced tool result: a content-parts result (read_image)
    becomes its one-line payload-free preview (megabytes of base64 in a JSONL
    stdout line is unrenderable and can't be head+tail trimmed usefully)."""
    if isinstance(value, list):
        from ._images import _parts_preview
        return _parts_preview(value)
    return _trim(value, cap)


def _emit(result: dict[str, Any]) -> None:
    """Write one JSONL result line to stdout, flushed immediately so a
    streaming consumer sees tasks land one by one. final_answer and each
    traced tool result are head+tail trimmed: the transcript is the durable
    record, stdout is the scoring stream."""
    trimmed = dict(result)
    fa = trimmed.get("final_answer")
    if isinstance(fa, str):
        trimmed["final_answer"] = _trim(fa, _RESULT_ANSWER_CHARS)
    tc = trimmed.get("tool_calls")
    if isinstance(tc, list):
        trimmed["tool_calls"] = [
            {**t, "result": _trim_result_value(t.get("result"), _RESULT_FIELD_CHARS)}
            if isinstance(t, dict) else t for t in tc]
    sys.stdout.write(json.dumps(trimmed, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _summary_line(results: list[dict[str, Any]], n_tasks: int, elapsed: float) -> str:
    n_done = sum(1 for r in results if r.get("status") == "done")
    n_cap = sum(1 for r in results if r.get("status") == "cap")
    n_err = sum(1 for r in results if r.get("status") == "error")
    n_timed_out = sum(1 for r in results if r.get("timed_out"))
    skipped = n_tasks - len(results)
    bits = [f"{n_done} done"]
    if n_cap:
        bits.append(f"{n_cap} cap")
    if n_err:
        bits.append(f"{n_err} error")
    if n_timed_out:
        bits.append(f"{n_timed_out} timed out")
    if skipped:
        bits.append(f"{skipped} not run")
    return (f"nx run: {', '.join(bits)} ({len(results)}/{n_tasks} tasks, "
            f"{elapsed:.1f}s)")


def run_batch(argv: list[str]) -> int:
    """The `nx run` entry point. Returns the process exit code (the caller
    sys.exit()s it). Human-facing progress goes to stderr; stdout carries
    ONLY the JSONL result lines."""
    try:
        opts = _parse_batch_args(argv)
    except _BatchUsageError as e:
        if str(e) == "__help__":
            sys.stdout.write(_USAGE_TEXT)
            return EXIT_OK
        sys.stderr.write(f"nx run: {e}\n")
        sys.stderr.write(_USAGE_TEXT)
        return EXIT_MISUSE

    try:
        tasks = _parse_tasks(opts["file"])
    except _BatchUsageError as e:
        sys.stderr.write(f"nx run: {e}\n")
        return EXIT_MISUSE
    if not tasks:
        sys.stderr.write("nx run: no tasks in the task file — nothing to do\n")
        return EXIT_MISUSE
    if st.ACTIVE is None:
        sys.stderr.write("nx run: no source configured — set NX_SOURCE_* env vars or use --source\n")
        return EXIT_MISUSE

    run_id = time.strftime("%Y%m%d-%H%M%S")
    parallel = opts["parallel"]
    max_turns = opts["max_turns"]
    if opts.get("autowake"):
        sys.stderr.write("nx run: --autowake on — chained tasks are bounded by "
                         "their per-task timeout (required) and turn cap\n")
    # Per-task wall-clock budget: --timeout wins over NX_TASK_TIMEOUT_SECONDS;
    # a task's own `timeout_seconds` field (resolved per task, below) wins over
    # both. None/0 = off.
    timeout_default = opts["timeout"]
    if timeout_default is None:
        timeout_default = _TASK_TIMEOUT_DEFAULT or None
    results: list[dict[str, Any]] = []
    stop_scheduling = threading.Event()
    lock = threading.Lock()
    started = time.time()
    sys.stderr.write(f"nx run: {len(tasks)} task(s), parallel={parallel}\n")

    def _worker(task: dict[str, Any], idx: int) -> None:
        if stop_scheduling.is_set():
            return
        try:
            task_timeout = _task_timeout(task, timeout_default)
        except _BatchUsageError as e:
            sys.stderr.write(f"nx run: {e}\n")
            res = {"index": idx, "id": task.get("id") if isinstance(task.get("id"), str) else f"task-{idx}",
                   "line": task.get("line"), "session": None, "batch": run_id,
                   "exit_code": EXIT_MISUSE, "status": "error",
                   "turns": 0, "elapsed_seconds": 0.0, "final_answer": None,
                   "tokens": None, "tool_calls": [], "error": str(e)}
            with lock:
                results.append(res)
            _emit(res)
            stop_scheduling.set()
            return
        res = _run_one_task(task, run_id, idx, max_turns, task_timeout,
                            model_default=opts["model"],
                            autowake_default=bool(opts.get("autowake")))
        with lock:
            results.append(res)
        _emit(res)
        if res.get("exit_code") == EXIT_ERROR and not opts["continue_on_error"]:
            stop_scheduling.set()

    if parallel == 1:
        for i, task in enumerate(tasks):
            _worker(task, i)
    else:
        # Bad timeout_seconds in task N must not hang around until its turn:
        # resolve every task's budget up front when scheduling is parallel.
        for task in tasks:
            try:
                _task_timeout(task, timeout_default)
            except _BatchUsageError as e:
                sys.stderr.write(f"nx run: {e}\n")
                return EXIT_MISUSE
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(_worker, task, i) for i, task in enumerate(tasks)]
            for f in futures:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001 — _worker never raises; defensive
                    sys.stderr.write(f"nx run: worker crashed: {type(e).__name__}: {e}\n")
    elapsed = time.time() - started
    sys.stderr.write(_summary_line(results, len(tasks), elapsed) + "\n")

    codes = [r.get("exit_code", EXIT_ERROR) for r in results] or [EXIT_MISUSE]
    if EXIT_ERROR in codes or 130 in codes:
        return EXIT_ERROR
    if EXIT_CAP in codes:
        return EXIT_CAP
    return EXIT_OK
