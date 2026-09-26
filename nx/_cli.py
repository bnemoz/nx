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
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import _state as st
from ._turn import _ensure_ctx_probe, _run_model_turn_loop
from . import _PROMPT_FILE, _JSON_OUTPUT, _initial_messages
from ._repl import _repl
from ._sessions_cmd import _cli_sessions, _cli_prune_sessions, _cli_tokens
from ._sessions_cmd import _cli_migrate_sessions
from ._repl_state import _fresh_metrics
from ._text_fallback import TOOL_PROTOCOL_TAG_RE, parse_text_calls_with_errors
from ._images import _parts_preview

_HELP_TEXT = """\
An interactive terminal coding agent.

Usage: nx [OPTIONS] [COMMAND]

Commands:
  run <file>     Batch mode: every JSONL line is one task, one JSON result
                 line per task on stdout. Flags: --parallel N, --max-turns N,
                 --timeout SEC, --continue-on-error, plus the global ones below.
  telegram       Long-poll a Telegram bot and run one turn per allowed-chat
                 message. Needs NX_TELEGRAM_BOT_TOKEN and
                 NX_TELEGRAM_ALLOWED_CHATS (hard allowlist — empty refuses to
                 serve). Gated calls auto-deny unless started with --yolo.
  sessions       List/filter saved sessions and exit (--page N --limit N to
                 page). "nx sessions --prune" runs the store's GC by hand:
                 reconcile vanished rows, collect unreferenced image payloads,
                 enforce NX_IMAGES_QUOTA_MB.
  config         View/check the env layer: list [filter] / get NAME /
                 set NAME VALUE / path / doctor. Never touches the network,
                 so it works against a dead endpoint.
  setup          First-run (or re-run) onboarding: paste an API base URL
                 and key; optionally install/point at SearXNG. Saves to ~/.env.
  eval           Offline eval corpus: replays recorded stub-server scenarios
                 through the real batch harness and scores each
                 (list / run [--source, --out, --filter ID, --parallel N,
                 --json]; exit 0 when all scenarios pass).
  update         Upgrade nx in place via its package manager (uv tool /
                 pipx / pip), or just report the install with --check.

Options:
  -f, --prompt-file <path>  Headless one-shot mode; "-" reads stdin. With
                            --json, emits a JSON result summary on stdout
                            instead of streamed chat output; exit 4 if the
                            run errored (endpoint down, API rejected), else 0
      --json                (with --prompt-file) see above
      --approval <mode>     Start with a non-default approval mode
                            [possible values: all, low, medium, high, yolo]
  -y, --yolo                Start in never-prompt mode (auto-approve everything)
      --source <name>       Start on a specific source
      --resume [target]     Resume a saved session; bare = most recent
      --session <id>        Start a fresh run attached to a specific session id
      --tokens              Print the exact prompt-token floor and exit
      --bench               Alias for --tokens
  -h, --help                Display the concise help for this command
  -V, --version             Display the nx version

Exit codes:
  0     Done
  2     Misuse (unreadable/empty prompt file, no source, bad flags)
  3     Turn cap reached (batch per-task / process summary)
  4     Run errored (endpoint down, API rejected the request)
  130   Interrupted

Use `nx help <command>` or `nx <command> --help` for details on a command.
Run `nx sessions` to list saved sessions, `nx config doctor` to check the env
layer, or `/help` in the REPL for the full in-session command list. See
README.md for the full NX_* env var table.
"""


def _help_colored(text: str) -> str:
    """Render the help text uv/rustup style: section headers (Usage:, Options:,
    Commands:, Exit codes:) in BOLD, the token column of every entry line (the
    `run <file>` part of a Commands row, the `-f, --prompt-file <path>` part of
    an Options row) in the accent color, and descriptions in the default text
    color — never dimmed: a dim tail on top of a dark theme is unreadable.

    Only entry lines get the accent treatment: a line indented inside one of
    the block sections whose first token sits before the 2+ space gap to its
    description. Continuation lines (descriptions wrapping under the token
    column) and trailing prose never start a colored run, even when they
    mention flags ("--parallel N"). Falls back to plain text when stdout is
    not a TTY or NO_COLOR is set — help is written to stdout, so no ANSI may
    leak into a pipe."""
    import os
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    from ._ansi import BOLD, C, RESET
    accent, bold = C.ACCENT, BOLD
    out = []
    in_blocks = False   # inside the Commands:/Options:/Exit codes: body
    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        # section headers
        if stripped and stripped.endswith(":") and len(stripped) < 40 \
                and not stripped.startswith("-"):
            in_blocks = stripped in ("Commands:", "Options:", "Exit codes:")
            out.append(f"{indent}{bold}{stripped}{RESET}")
            continue
        if not in_blocks:
            out.append(line)
            continue
        # an entry line starts at the token column (indent 2, or 6 for an
        # Options entry aligned under a short form); descriptions that wrap
        # under the token column start at column 17/28 and stay plain
        if len(indent) not in (2, 6) or not stripped:
            out.append(line)
            continue
        gap = re.search(r"\S\s{2,}\S", line)
        if gap is None:
            out.append(line)
            continue
        head, tail = line[:gap.start() + 1], line[gap.start() + 1:]
        out.append(f"{indent}{accent}{head.strip()}{RESET}" + ("  " + tail if tail else ""))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _cli_version() -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("nx")
    except PackageNotFoundError:
        return "unknown"


_VERBS = ("run", "telegram", "config", "eval", "sessions", "ls", "list", "prune", "setup", "update")


def _cli_help_or_version(argv: list[str]) -> bool:
    # Verb-first invocations fall through so each subcommand prints its own
    # usage; the top-level flag list describes flags those verbs reject.
    # main()'s verb dispatch handles them.
    if argv and argv[0] in _VERBS:
        return False
    if "-h" in argv or "--help" in argv:
        sys.stdout.write(_help_colored(_HELP_TEXT))
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
    # First-run onboarding (0.77.0): `nx setup` re-runs the wizard on demand;
    # a bare `nx` with NO endpoint configured anywhere walks the user through
    # pasting one before `import nx` fires its startup network probes (which
    # would just hang against a nonexistent endpoint). Needs the env file
    # loaded into os.environ first so _needs_onboarding() sees ~/.env keys.
    if sys.argv[1:2] == ["setup"]:
        from ._onboard import cli_onboard
        sys.exit(cli_onboard(sys.argv[2:]))
    # `nx update` — self-update via the package manager (uv tool / pipx /
    # pip). Purely a local/package-manager operation, so it dispatches before
    # any env-file/onboarding machinery and works against a dead endpoint
    # (nx/__init__.py skips switch_source()'s startup probes for this verb).
    if sys.argv[1:2] == ["update"]:
        from ._update import _cli_update
        sys.exit(_cli_update(sys.argv[2:]))
    if sys.argv[1:2] != ["config"]:
        from ._env import _load_env_file
        _load_env_file()
        from ._onboard import _needs_onboarding
        if _needs_onboarding():
            from ._onboard import run_onboarding
            # A False return (aborted, or non-interactive stdin) falls
            # through to normal startup — the legacy `local` source still
            # registers, and one-shot mode surfaces its usual "no source
            # configured" exit.
            if run_onboarding():
                # On success, re-exec with a clean interpreter: `import nx`
                # above already ran _discover_sources() against the EMPTY
                # pre-onboarding env, and that snapshot isn't rebuilt, so
                # continuing in-process would silently fall back to the
                # default localhost:8080 endpoint instead of the URL the
                # user just pasted. A fresh process sees the saved config.
                os.execv(sys.argv[0], sys.argv)
    # `nx run ...` — headless batch mode: every line of a JSONL file is one
    # task, each runs the one-shot machinery on its own quiet thread and
    # emits one JSONL result line; the process exit code summarizes the run.
    if sys.argv[1:2] == ["run"]:
        from ._batch import run_batch
        sys.exit(run_batch(sys.argv[2:]))
    # `nx telegram` — the Telegram gateway: long-poll one bot, one turn per
    # allowed-chat message. Env-gated (token + allowlist), so it exits 2 with
    # guidance rather than touching the network when unconfigured.
    if sys.argv[1:2] == ["telegram"]:
        from ._env import _load_env_file
        _load_env_file()
        from ._telegram import cli_telegram
        sys.exit(cli_telegram(sys.argv[2:]))
    # `nx config list|get|set|path|doctor` — viewer/doctor over the env layer
    # (P3-9). Read-only except `set`, which goes through _env._set_env_var.
    # Never touches the network: nx/__init__.py skips switch_source()'s
    # startup probes for this verb, so it works against a dead endpoint.
    if sys.argv[1:2] == ["config"]:
        from ._config_cmd import _cli_config
        sys.exit(_cli_config(sys.argv[2:]))
    # `nx eval list|run` — offline eval corpus (P3-14): recorded stub-server
    # scenarios replayed through the real batch harness and scored. Builds its
    # own loopback stub source; never touches the configured endpoints. Like
    # `config`, import skips switch_source() for this verb so it works against
    # a dead endpoint.
    if sys.argv[1:2] == ["eval"]:
        from ._eval_cmd import _eval_main
        sys.exit(_eval_main(sys.argv[2:]))
    # --prompt-file: headless single-shot mode — run one turn and exit.
    import nx as _m
    # Recover any undo checkpoints a previous session left behind (0.47.0,
    # ROADMAP 2.2): a crash no longer loses the /undo story. Never fatal.
    try:
        _m._undo._load_persisted()
    except Exception:
        pass
    if _m._run_one_shot():
        return
    # `nx sessions --prune` — reconcile the store by hand, then exit. Before
    # the listing verb: _cli_sessions treats "--prune" as a search query and
    # would otherwise swallow the flag (it printed "no sessions matching
    # '--prune'" instead of ever pruning).
    if _cli_migrate_sessions(sys.argv[1:]):
        return
    if _cli_prune_sessions(sys.argv[1:]):
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
