"""The top-level CLI entry points: `_run_one_shot()` (headless `--prompt-file`
mode) and `main()` (the orchestrator — handles `sessions`/`--tokens` subcommand
short-circuits, then hands off to the interactive REPL in `_repl.py`).

`main()` deliberately stays a thin dispatcher: by the time this module is
imported, every other concern (sources, approval, tools, the turn loop,
compression, the REPL loop itself) already has a stable home, so this is
"change imports," not "change imports and logic simultaneously."
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from . import _state as st
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from . import _PROMPT_FILE, _JSON_OUTPUT, _initial_messages
from ._repl import _repl, _cli_sessions, _cli_tokens
from ._repl_state import _fresh_metrics
from ._text_fallback import TOOL_PROTOCOL_TAG_RE, parse_text_calls_with_errors
from ._images import _parts_preview

_HELP_TEXT = """\
usage: nx [flags]
       nx run -f tasks.jsonl [flags]
       nx sessions [query]

flags:
  --yolo                          start in never-prompt mode (auto-approve everything)
  --approval <all|low|medium|high|yolo>
                                   start with a non-default approval mode
  --source <name>                 start on a specific source
  --resume [target]               resume a saved session; bare = most recent
  --session <id>                  start a fresh run attached to a specific session id
  --prompt-file <path> / -f <path>
                                   headless one-shot mode; "-" reads stdin
  --json                          (with --prompt-file) emit a JSON result summary
                                   on stdout instead of streamed chat output;
                                   exit 4 if the run errored (endpoint down,
                                   API rejected the request), else 0

Batch mode — run every task in a JSONL file and emit JSONL results:

  nx run -f tasks.jsonl [--parallel N] [--continue-on-error]

Per-task exit codes: 0 done, 3 turn cap, 4 error. `nx run --help` for details.

  --tokens / --bench               print the exact prompt-token floor and exit
  -h, --help                      show this help and exit
  --version / -v                  show the version and exit

Run `nx sessions` to list saved sessions, or `/help` in the REPL for the
full in-session command list. See README.md for the full NX_* env var table.
"""


def _cli_version() -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("nx")
    except PackageNotFoundError:
        return "unknown"


def _cli_help_or_version(argv: list[str]) -> bool:
    if "-h" in argv or "--help" in argv:
        sys.stdout.write(_HELP_TEXT)
        return True
    if "--version" in argv or "-v" in argv:
        print(f"nx {_cli_version()}")
        return True
    return False


def _run_one_shot() -> bool:
    """Non-interactive entry point for --prompt-file. Reads the instruction,
    runs _run_model_turn_loop once, and exits with a status code."""
    if _PROMPT_FILE is None:
        return False  # not in one-shot mode; caller should proceed to REPL
    # --- read the instruction --------------------------------------------------
    if _PROMPT_FILE == "-":
        instruction = sys.stdin.read()
    else:
        try:
            instruction = Path(_PROMPT_FILE).read_text(encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"nx: cannot read --prompt-file {e.filename!r}: {e}\n")
            sys.exit(2)
    instruction = instruction.strip()
    if not instruction:
        sys.stderr.write("nx: --prompt-file is empty — nothing to do\n")
        sys.exit(2)

    # --- resolve the active source ---------------------------------------------
    # switch_source was already called at import time (line ~898) for the
    # --source flag or the first available source. Verify it landed.
    if st.ACTIVE is None:
        sys.stderr.write(
            "nx: no source configured — set NX_SOURCE_* env vars or use --source\n")
        sys.exit(2)

    # Warm the context-window probe in the background (same as the REPL).
    _ensure_ctx_probe(st.ACTIVE)

    # Benchmark harnesses want token accounting too — the REPL sets this up
    # in _repl(), which one-shot mode skips entirely.
    st._METRICS = _fresh_metrics(session_id="one-shot")

    # --- run the turn ----------------------------------------------------------
    messages = _initial_messages() + [{"role": "user", "content": instruction}]
    start_idx = len(messages)
    real_stdout = sys.stdout
    if _JSON_OUTPUT:
        # Streamed chat output still goes through bare print() deep in the
        # turn loop; redirect it to stderr so stdout carries only the final
        # JSON summary a harness can parse.
        sys.stdout = sys.stderr
    started = time.time()
    try:
        summary = _run_model_turn_loop(messages, on_step=None)
    except KeyboardInterrupt:
        sys.stdout = real_stdout
        sys.stderr.write("\n[nx] interrupted\n")
        sys.exit(130)
    except Exception as e:
        sys.stdout = real_stdout
        sys.stderr.write(f"\n[nx] error: {type(e).__name__}: {e}\n")
        sys.exit(1)
    finally:
        sys.stdout = real_stdout

    if _JSON_OUTPUT:
        real_stdout.write(json.dumps(_one_shot_result(
            messages, start_idx, summary, time.time() - started)) + "\n")
    # exit 4 mirrors batch's EXIT_ERROR contract: a turn loop that ended in
    # status "error" (endpoint unreachable / API rejected the request) is a
    # failed run, not a clean one — a harness keying on the exit code must
    # never score it as a pass.
    sys.exit(4 if isinstance(summary, dict) and summary.get("status") == "error" else 0)


def _one_shot_result(messages: list, start_idx: int, summary: dict, elapsed: float,
                     metrics: Any = None) -> dict:
    """Build the --json result payload from the messages this run appended
    (everything from start_idx on) plus the turn-loop's status summary.

    Native tool_call turns pair each result by tool_call_id (a fan-out turn
    emits N results — pairing on trace[-1] used to keep only the last).
    Text-protocol turns append one user message of "Observation (name): ..."
    blocks, one per call, in call order."""
    trace = []
    final_answer = None
    pending_by_id: dict[str, int] = {}
    for m in messages[start_idx:]:
        role = m.get("role")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            for tc in calls:
                fn = tc.get("function", {})
                pending_by_id[tc.get("id", "")] = len(trace)
                trace.append({"tool": fn.get("name"), "args": fn.get("arguments")})
            text = m.get("content") or ""
            if not calls and text:
                # text-protocol turns keep the calls inside the content; parse
                # them so the JSON trace isn't empty on fallback-path models
                parsed, _errs = parse_text_calls_with_errors(text)
                for name, a in parsed:
                    trace.append({"tool": name,
                                  "args": a if isinstance(a, str) else json.dumps(a)})
                if not parsed and not TOOL_PROTOCOL_TAG_RE.search(text):
                    final_answer = text
        elif role == "tool" and trace:
            idx = pending_by_id.get(m.get("tool_call_id", ""))
            content = m.get("content")
            if isinstance(content, list):
                # read_image parts: trace the one-line hint, not megabytes of
                # base64 (the trace is a debugging artifact, not the payload)
                content = _parts_preview(content)
            if idx is not None:
                trace[idx]["result"] = content
            else:
                trace[-1]["result"] = content
        elif role == "user" and trace:
            # text-protocol observations: one user message holding an
            # "Observation (tool): result" block per call, in call order;
            # repeated tool names take their k-th block in turn
            obs_text = m.get("content") or ""
            taken: dict[str, int] = {}
            for t in trace:
                if "result" in t or not t.get("tool"):
                    continue
                marker = "Observation (%s): " % t["tool"]
                j = -1
                for _ in range(taken.get(t["tool"], 0) + 1):
                    j = obs_text.find(marker, j + 1)
                    if j == -1:
                        break
                if j == -1:
                    continue
                taken[t["tool"]] = taken.get(t["tool"], 0) + 1
                rest = obs_text[j + len(marker):]
                nxt = rest.find("\nObservation (")
                t["result"] = rest[:nxt] if nxt != -1 else rest
    m = metrics if metrics is not None else st._METRICS
    tokens = ({"input": m.input_tokens, "output": m.output_tokens,
               "cache_read": m.cache_read_tokens, "reasoning": m.reasoning_tokens,
               "api_calls": m.api_calls} if m is not None else None)
    return {
        "final_answer": final_answer,
        "status": summary.get("status") if summary else "error",
        "error": (summary or {}).get("error"),
        "turns": summary.get("turns") if summary else 0,
        "elapsed_seconds": round(elapsed, 2),
        "tokens": tokens,
        "tool_calls": trace,
    }


def main() -> None:
    # --help/--version short-circuit before any source/network setup.
    if _cli_help_or_version(sys.argv[1:]):
        return
    # `nx run ...` — headless batch mode: every line of a JSONL file is one
    # task, each runs the one-shot machinery on its own quiet thread and
    # emits one JSONL result line; the process exit code summarizes the run.
    if sys.argv[1:2] == ["run"]:
        from ._batch import run_batch
        sys.exit(run_batch(sys.argv[2:]))
    # --prompt-file: headless single-shot mode — run one turn and exit.
    import nx as _m
    if _m._run_one_shot():
        return
    # `nx sessions [query]` — discover + exit, no REPL. Checked first so
    # it short-circuits before the banner / network source resolution that the
    # interactive loop depends on.
    if _cli_sessions(sys.argv[1:]):
        return
    # `nx --tokens` / `--bench` — print the prompt-token floor and exit.
    if _cli_tokens(sys.argv[1:]):
        return
    _repl()
