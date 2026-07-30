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

from . import _state as st
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from . import _PROMPT_FILE, _JSON_OUTPUT, _initial_messages
from ._repl import _repl, _cli_sessions, _cli_tokens
from ._repl_state import _fresh_metrics

_HELP_TEXT = """\
usage: nx [flags]
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
                                   on stdout instead of streamed chat output
  --tokens / --bench               print the exact prompt-token floor and exit
  -h, --help                      show this help and exit
  --version                       show the version and exit

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
    if "--version" in argv:
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
    sys.exit(0)


def _one_shot_result(messages: list, start_idx: int, summary: dict, elapsed: float) -> dict:
    """Build the --json result payload from the messages this run appended
    (everything from start_idx on) plus the turn-loop's status summary."""
    trace = []
    final_answer = None
    for m in messages[start_idx:]:
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                trace.append({"tool": fn.get("name"), "args": fn.get("arguments")})
            if m.get("content") and not m.get("tool_calls"):
                final_answer = m["content"]
        elif role == "tool" and trace:
            trace[-1]["result"] = m.get("content")
    m = st._METRICS
    tokens = ({"input": m.input_tokens, "output": m.output_tokens,
               "cache_read": m.cache_read_tokens, "reasoning": m.reasoning_tokens,
               "api_calls": m.api_calls} if m is not None else None)
    return {
        "final_answer": final_answer,
        "status": summary.get("status") if summary else "error",
        "turns": summary.get("turns") if summary else 0,
        "elapsed_seconds": round(elapsed, 2),
        "tokens": tokens,
        "tool_calls": trace,
    }


def main() -> None:
    # --help/--version short-circuit before any source/network setup.
    if _cli_help_or_version(sys.argv[1:]):
        return
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
