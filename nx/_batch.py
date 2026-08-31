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
With --parallel > 1 the pending-note channel is process-global, so an async
agent's note can land in a sibling task's turn; prefer --parallel 1 when a
task relies on async sub-agents.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ._env import _env_int
from . import _state as st

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

flags:
  -f, --prompt-file <path>   JSONL task file ("-" reads stdin; required)
      --parallel <N>         run up to N tasks concurrently (default 1)
      --continue-on-error    keep going after a task errors (default: stop
                             scheduling new tasks after the first error)
      --yolo / --approval <level>
                             approval posture for the tasks (batch is
                             headless: without yolo, gated calls are DENIED)
      --source <name>        endpoint to run the tasks against
      --max-turns <N>        per-task turn cap (default NX_MAX_MODEL_TURNS)
      --timeout <seconds>    per-task wall-clock cap (default
                             NX_TASK_TIMEOUT_SECONDS; 0 = off) — the task
                             stops at the next model-turn boundary and the
                             result line reports status "error", the timeout
                             in `error`, and "timed_out": true
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
        "max_turns": None, "timeout": None,
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
        elif a == "--continue-on-error":
            opts["continue_on_error"] = True
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


def _run_one_task(task: dict[str, Any], run_id: str, idx: int,
                  max_turns: int | None,
                  timeout_seconds: float | None = None) -> dict[str, Any]:
    """Run one task headlessly: fresh messages, fresh metrics, fresh session
    id; the turn loop's chat output is suppressed (quiet). Returns the merged
    result dict (the JSONL line's content). Never raises — a raise is an
    exit_code=4 result.

    timeout_seconds arms a wall-clock deadline on top of the turn cap: the
    loop aborts at the next model-turn boundary with status "error" and
    `timed_out` set on the result line, and the turn cap's forced-final
    rescue is skipped so a hung task cannot stretch one more turn past its
    budget."""
    import nx as _m  # lazy: the package re-exports what the loop needs
    from ._repl_state import _fresh_metrics
    from ._subagent import _ASYNC_QUIET_SLOTS
    from ._turn import _ensure_ctx_probe, _task_deadline_at

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
    }
    started = time.time()
    # Quiet the whole worker thread (no spinner/stream/footer/tool box) and
    # mark the TLS task slot so the shared turn machinery can skip
    # interactive-only behavior. The REPL never runs inside batch mode, so
    # nothing else contends for the quiet slot. With --parallel, a worker's
    # `finally` may run while siblings are mid-loop — the guard keeps the
    # slot quiet until the LAST batch task on any thread finishes.
    _ASYNC_QUIET_SLOTS.quiet = True
    global _QUIET_HOLDERS
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
    try:
        prev_metrics = st._set_thread_metrics(_fresh_metrics(session_id))
        if st.ACTIVE is None:
            raise RuntimeError("no active model source — set NX_BASE_URL/NX_SOURCE_* or use --source")
        _ensure_ctx_probe(st.ACTIVE)
        task_metrics = st._active_metrics()
        messages = _m._initial_messages() + [{"role": "user", "content": instruction}]
        start_idx = len(messages)
        summary = _m._run_model_turn_loop(messages, on_step=None,
                                          max_turns_override=max_turns)
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
        _task_deadline_at(None)
        st._set_thread_metrics(prev_metrics)
        _TASK_TLS.task = None
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
        m["model"] = st.MODEL
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
            {**t, "result": _trim(t.get("result"), _RESULT_FIELD_CHARS)}
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
        res = _run_one_task(task, run_id, idx, max_turns, task_timeout)
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
