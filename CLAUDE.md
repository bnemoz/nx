# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Semver Policy

From now on we follow semver; bump version numbers as changes are made.

## What this is

`nx` is a lightweight coding agent CLI, packaged under `nx/` (entry point
`nx/__init__.py`, ~1500 lines, plus over a dozen focused submodules — `_turn.py`,
`_repl.py`, `_sources.py`, `_chatbox.py`, etc.). It used to be a single ~5300-line
`nx.py` file kept deliberately dependency-minimal; it has since been split into a
proper package (see the "Extract ... into nx/_x.py" commits in the git log) and
picked up real dependencies (`httpx`, `rich`) where they earn their keep. It talks to
the OpenAI-compatible `/chat/completions` and `/models` endpoints directly over `httpx`
(`nx/_openai_client.py`) rather than depending on the `openai` SDK — that package eagerly
imports typed models for admin/batches/evals/graders/responses/azure/bedrock that nx
never touches, which dominated its import-time RSS for a handful of call sites nx's own
thin client now covers directly.
Context-frugality — keeping the per-turn token floor low relative to heavier agent
frameworks — is still a real design value (see `README.md` and `nx --tokens`), but
it is no longer the sole organizing constraint: don't block a clean module boundary or
a useful dependency just to preserve the old one-file/zero-deps posture. Do still keep
tool schemas (`TOOLS`, `FINAL_ANSWER_TOOL`, `SYSTEM`) terse, since those are sent on
every turn regardless of how the code is organized internally.

## Setup

```bash
uv pip install -e .    # httpx, rich
```

To develop against a real backend, point it at a local OpenAI-compatible server (llama.cpp,
vLLM, SGLang) or a remote one (Z.ai, Together, OpenRouter):

```bash
export NX_BASE_URL=http://localhost:8080/v1
export NX_MODEL=your-model-name
export NX_API_KEY=sk-noop
python -c "import nx; nx.main()"
```

Install as an editable console script (`nx` on `$PATH`, edits picked up immediately):

```bash
uv pip install -e .
```

## Running tests

We use `uv` to run tests. There's no test runner config (no pytest.ini/Makefile/CI) — tests
are ad hoc scripts under `tests/`, in two styles:

- **pytest-style** (`def test_...` functions, some using `monkeypatch`/`tmp_path` fixtures):
  run with `uv run pytest tests/` or a single file/test with `uv run pytest
  tests/test_risk_context.py -k test_downloads_project_file_is_in_project`.
- **plain-script style** (top-level code with bare `assert` statements that runs at import
  time, e.g. `test_sources.py`, `test_context_window.py`): run directly with
  `uv run python3 tests/test_sources.py`. These print progress and exit nonzero (or raise) on
  failure — there's no fixture teardown, so side effects (env vars, `importlib.reload`) are
  local to the test's own `sys.path`/env manipulation at the top of the file.

Every test file inserts the repo root onto `sys.path` and imports `nx` (or `nx as m`)
directly — there's no installed package assumption beyond that. Most tests scrub `NX_*`
env vars and point `NX_ENV_FILE` at `/dev/null` (or a tempdir) at the top, so a real
`~/.env` on the dev machine can't leak into test behavior. Follow that pattern for new tests.

No network access is needed for the existing suite — sources/config/parsing/risk-classifier
logic is tested against synthetic state, not live model calls.

## Architecture

The package is organized into one module per concern; grep `^# ---` banners inside
`nx/__init__.py` and the submodules to jump around. The rough execution flow:

1. **Startup**: `_load_env_file()` loads `~/.env`, `_discover_sources()` (`nx/_sources.py`)
   builds the `Source` registry from `NX_SOURCES`/`NX_SOURCE_*` env vars (plus
   auto-registered built-ins for `together`/`openrouter` if their API keys are present),
   `main()` (`nx/_cli.py`) parses CLI flags and either drops into the REPL
   (`nx/_repl.py`) or handles a one-shot/subcommand path (`sessions`, `--resume`,
   `--prompt-file`).

2. **The turn loop**: `model_turn()` (`nx/_turn.py`, the largest function) streams one
   exchange with the model and returns one of the `TURN_*` status constants (`TURN_DONE`,
   `TURN_TOOL`, `TURN_STREAM_CUT`, `TURN_FORCE_FINAL`, `TURN_STALL`, `TURN_EMPTY`, `TURN_ESC`).
   `_run_model_turn_loop()` wraps it to handle retries/recovery. This is where streaming
   parse, tool-call extraction (native or text-fallback via `parse_text_calls()` in
   `nx/_text_fallback.py`), reasoning-content handling, and all the stall/recovery
   guards live.

3. **Tools**: read-only tool bodies (`read_file`, `list_dir`, `search_files`) live in
   `nx/_tools.py`; the state-touching ones (`write_file`, `edit_file`, `multi_edit`,
   `sed_edit`) live in `nx/_edit_tools.py`, and `run_bash`/`wait_background` live in
   `nx/_bash_jobs.py` — all call `_confirm()`/`_ask_approval()` and touch session/
   background-process globals. All are dispatched through `run_tool()`.
   `run_bash` always launches detached (`setsid`) and polls briefly before backgrounding —
   see the `SYSTEM` prompt string in `nx/__init__.py` for the exact background/foreground
   contract given to the model; keep that docstring and the tool's actual behavior in sync
   if you touch either.

4. **Approval gating**: every write/edit/bash call is risk-classified by `_assess_risk()`
   (`nx/_risk.py`, a cheap separate model call) into low/medium/high, then
   `_confirm()`/`_ask_approval()` (`nx/_approval.py`) decide whether to prompt based on
   the active approval mode. Path-based risk also consults `_detect_project_root()` /
   `_is_under_path()` so writes inside the project are treated less suspiciously than writes
   elsewhere.

5. **Sources**: the `Source` class and `_discover_sources()`/`switch_source()`/
   `restore_source()` (`nx/_sources.py`) implement multi-endpoint support — named
   backends with independent base URL, model, API key, extra headers, and provider-routing
   (OpenRouter `/provider`). Context-window size per source/model is probed lazily by
   `_ensure_ctx_probe()` and cached.

6. **Sessions**: `_write_session()`/`_load_session()`/`_list_sessions()` persist the full
   message array + metadata as one JSON file per session under `~/.nx/sessions/`
   (`NX_HOME`/`NX_SESSIONS_DIR` to override). `_maybe_refresh_description()`
   periodically asks the model for a one-line session summary. Token usage accounting lives in
   `_Metrics`/`_normalize_usage()` (`nx/_metrics.py`) and is written into the same session
   JSON.

7. **Context compression**: `compress()` (manual `/compress`) and `_maybe_autocompress()`
   (automatic, triggered by `NX_AUTOCOMPRESS_PERCENT`), both in `nx/_compress.py`,
   fold older turns into a summary message, keeping the most recent turns verbatim.

8. **Terminal UI**: `_chatbox_raw()`/`read_multiline()` (`nx/_chatbox.py`) implement a
   raw-terminal multi-line input box using hand-rolled ANSI escapes (`nx/_ansi.py`, no
   curses/prompt_toolkit — `rich` is used for output rendering, not input handling).
   `_interrupt_watcher()` (`nx/_interrupt.py`) runs on a background thread to catch Esc
   during model generation; it is a *singleton*. Preserve its single-reader and
   parked-watcher invariants, and keep `tests/test_interrupt_watcher_singleton.py` passing
   when changing it.

9. **Project instructions**: `_load_agents_md_note()` (`nx/_agents.py`) reads an
   `AGENTS.md` at the detected project root (same convention as CLAUDE.md itself) and
   prepends it as a one-time user-role message to a fresh session, rather than folding it
   into `SYSTEM` — keeps the per-turn token floor independent of project size.

10. **Process stats**: `process_stats()` (`nx/_resources.py`) samples this process's own
    CPU%/RSS straight from `/proc` (with an `os.times()`/`resource` fallback off-Linux) —
    no `psutil` dependency for a couple of numbers.

Reading order for a new change: find the relevant submodule (or `# --- section ---` banner
inside `nx/__init__.py`) first, then read it — most features are self-contained within
one module plus a touch point in `model_turn()` or `TOOLS`/`SYSTEM`.

## Conventions specific to this repo

- **Env vars, not config files.** New tunables should be `NX_*` env vars with a sane
  default read via `_env_int`/`_env_float`, documented in the README's env var table — not a
  new config file format.
- **Tool schemas are load-bearing on token cost.** Any change to `TOOLS`, `FINAL_ANSWER_TOOL`,
  or `SYSTEM` directly changes the per-turn token floor the project is built around; keep
  descriptions terse and justify additions. This is the one place the old dependency-minimal
  discipline still fully applies — everything else (module layout, real dependencies) is
  fair game.
- **New code goes in a focused submodule, not back into `nx/__init__.py`.** Follow the
  existing extraction pattern (`nx/_turn.py`, `nx/_sources.py`, etc.): a new concern
  gets its own `nx/_x.py` file, imported into `__init__.py` rather than inlined there.
- **Degrade, don't demand.** Features should degrade gracefully for non-conforming servers
  (no native tool-calling, no `usage` object, vLLM vs. llama.cpp sampler support) rather than
  assuming a fully-compliant OpenAI backend — see `NX_BACKEND=vllm` and the text-fallback
  tool-call parser as existing examples.
- For concurrency- or terminal-sensitive changes, document the relevant invariants beside
  the implementation and extend the focused regression tests.
