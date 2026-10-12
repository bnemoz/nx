# Repository Guidelines

## Project Structure & Module Organization

`nx/` contains the Python package for the `nx` coding-agent CLI. Keep concerns in focused private modules and expose/wire them through `nx/__init__.py` only as needed. The formerly monolithic core files were decomposed across 0.87.2–0.103.x; the current map:

- **Turn loop** — `nx/_turn.py` (model_turn, the loop) plus `nx/_turn_status.py` (TURN_* statuses) and `nx/_turn_toolcalls.py` (the tool-call handlers; both split out at 0.90.0 and re-exported from `_turn.py`), `nx/_turn_footer.py` (stats footer, ctx probe), `nx/_tool_prefetch.py` (read-only parallel fan-out, cache prefetch), `nx/_spec_reads.py` (speculative read-only dispatch during streaming), `nx/_argsalvage.py` (malformed-args salvage, breaker nudge), `nx/_turn_retry.py` (step retry ladder, batch deadline TLS).
- **Dispatch** — `nx/_dispatch.py` (DISPATCH table, run_tool) plus `nx/_tool_gates.py` (the tiered registry: `_TOOL_GATES`, `_active_tools`, `_gate_vector`, search_tools, withheld-tool refusals), `nx/_breaker.py` (malformed-args circuit breaker), `nx/_session_preamble.py` (fresh-session message list, one-time nudge strings), `nx/_bg_completions.py` (bg-job completion sweep).
- **Sessions & isolation** — `nx/_sessiondb.py` (0.101.0: one `sessions.sqlite` store, the source of truth), `nx/_sessionlock.py` + `nx/_staleness.py` (cross-session repo advisory lock and edit-staleness guard, 0.103.0), `nx/_session_save.py` (off-critical-path autosave, 0.103.4), `nx/_sessions_io.py` / `nx/_sessions_search.py` (FTS5 session search/export), `nx/_resume.py`, `nx/_sessions_cmd.py` (the `/sessions`/`/resume`/`/save`/`/delete` UI), `nx/_worktree.py` (per-agent and per-session git worktrees).
- **Learning & data** — `nx/_lessons.py` (`recall_lessons`: BM25 cross-session recall over the session store), `nx/_duckdb.py` (SQL over local files), `nx/_obs.py` + `nx/_obsstore.py` (Observation Pack + content-addressed cross-process store), `nx/_memindex.py` (FTS5 memory index).
- **Infra** — `nx/_subagent_model.py` (per-subagent model routing, incl. `NX_SUBAGENT_MODEL_POOL` round-robin), `nx/_lint.py` (`lint_check`, tier-2), `nx/_autowake.py` (self-triggering turns), `nx/_batch.py` (`nx run` headless JSONL batch), `nx/_fusion.py` (action fusion / `then_run`), `nx/_hooks.py`, `nx/_verify.py`.
- **Requests/compression** — `nx/_compress.py` (open_stream, compress, the env-derived request flags) plus `nx/_runtime_notes.py` (the [Runtime note] contract), `nx/_request_opts.py` (recovery sampler, grammar, /toolcalls mode, max_tokens clamp), `nx/_stream_errors.py` (error classification).
- **Sub-agents** — `nx/_subagent.py` (spawn_agent, async machinery) plus `nx/_subagent_tiers.py` (tool tiers, restricted dispatch tables), `nx/_subagent_bash.py` (read-only bash allowlist), `nx/_delegation_note.py`, `nx/_subagent_tls.py`.
- **Memory** — `nx/_memory_cmd.py` (the /memory UI + memory_write) plus `nx/_memory_store.py` (the one scanner/writer), `nx/_memory_search.py`, `nx/_memory_note.py`, `nx/_memory_archive.py`.
- **Prompts** — `nx/_prompts.py` is the single home for the satellite system prompts (RISK_SYSTEM, REDUCER_SYSTEM_PROMPT, DESC_SYSTEM, SUBAGENT_SYSTEM/SUBAGENT_WRITE_SYSTEM/SUBAGENT_IMPLEMENT_SYSTEM); the main `SYSTEM` stays in `nx/__init__.py` (load-bearing on the token floor; its bytes are pinned by `_prefix.py`).

When you split or extend a module, moved names are re-exported from the old home so the import surface (and tests that patch module attributes) keep working. `tests/` contains one test file per behavior area. `README.md` is the user-facing reference; `CONTRIBUTING.md` documents required touch points for slash commands, tools, prompts, and environment variables. `sources.example.env` provides safe configuration examples.

## Semver Policy

From now on we follow semver; bump version numbers as changes are made.

## Build, Test, and Development Commands

Use `uv` for local development:

```bash
uv pip install -e .                 # install the CLI and runtime dependencies
uv run pytest tests/ -n auto        # run pytest-style tests in parallel (~14s; workers capped at 32)
uv run python3 tests/test_sources.py # run a standalone assertion-style test
uv run ruff check .                 # lint gate — must be clean before committing
uv run ty check nx                  # type check; no new diagnostics in touched files
nx --help                           # inspect the installed CLI
```

`-n auto` is the default way to run the suite; plain `uv run pytest tests/` (serial, ~40s) works too and is the fallback if xdist is unavailable.

Target a single pytest test during development with `uv run pytest tests/test_risk_context.py -k test_name`. The suite is designed to run offline; do not depend on a live model endpoint.

### When a full-suite run fails (read before debugging)

The suite runs 12/12 clean on an idle box. A failure you cannot reproduce standalone is almost never a regression — work this checklist in order before spending more than five minutes:

1. **Check the run's wall time.** A clean run takes ~13.6s. If the failing run took 23s+, another pytest/nx session was running concurrently on this machine (verify with `ps aux | grep pytest` and `ls -t ~/.nx/bg-logs/` — including sessions in *other* projects). Re-run once before investigating anything.
2. **Check the known-flake ledger** in `docs/test-architecture.md` §5.0. It lists each recurring flake, its root cause, and its fix — including the residue classes (`_PENDING_*` notice slots, `_TOOL_RESULT_CACHE`, the wake poller, `_BG_NOTICE_SEEN`/`_BG_SCAN_DONE`) that `nx_state_reset` in `tests/_nxtest.py` now drains. If a test fails because some new hidden home wasn't drained, the fix belongs in that fixture — centrally, once — not in the failing test.
3. **Genuinely environmental flakes live in `_FLAKY_RERUN_NODEIDS`** (`tests/conftest.py`): those nodeids get one silent in-process retry; if the retry also fails *and* the test passes standalone, extend the tuple with a one-line comment — do not debug it.
4. **Reproduce serially before bisecting**: `uv run pytest tests/test_suspect.py tests/test_poisoned.py -q` (both orders). Cross-file failures that reproduce serially are residue (fix the fixture or the leaking test); failures that only appear under `-n auto` after that are contention — see step 1.

Do not delete a slow or flaky test to make a run green, and do not add sleeps to a timing-sensitive test — poll on the condition (`_wait_for`) or sync on the event instead. The suite's wall time is dominated by its slowest single file (per-file `--dist loadfile` packing), so before assuming the suite "got slower," check for a straggler file with `--junitxml` + per-class durations before touching distribution settings.

Writing tests that touch process-global state: every module global your test mutates must either be set through `monkeypatch` (auto-restored) or be a home `nx_state_reset` already drains. Tests with no `nx_fresh`-style fixture run against whatever `SESSION_HOME` the module holds — which can be the developer's real `~/.nx` — so any test that triggers a background-completion sweep must pin its own session home.

Ruff and ty are pinned in the `dev` dependency group (installed with `uv sync --group dev`). Ruff is a hard gate; ty's existing ~130 diagnostics are tracked, not owed — see CONTRIBUTING.md's "Lint and type check" section for the policy.

## Coding Style & Naming Conventions

Use standard Python conventions: four-space indentation, `snake_case` functions and variables, `CapWords` classes, and descriptive `NX_*` names for environment settings. Prefer a new focused `nx/_feature.py` module over growing `__init__.py`.

### Prompt & schema budget

We target models with 128K-1M-token context windows, so the per-turn floor of `SYSTEM` prompt + always-on tool schemas is held under a **hard, recorded budget of ~5K tokens** (`BUDGET_TOKENS` in `tests/test_initial_context.py`; currently 5200, measured floor ~5.1K (5120) per the 0.103.22 measurement; `SOFT_TOKENS = BUDGET_TOKENS - 50`). Within that ceiling, favor *clarity over terseness*: every argument the model can pass gets a `description`, overlapping tools name each other, and self-healing behavior is spelled out in the schema. Do not abbreviate prose to save a handful of tokens — the budget is enforced mechanically, so a schema only needs trimming when the floor actually approaches the ceiling.

The budget is a forcing function, not a soft warn: a change that pushes the floor past `BUDGET_TOKENS` fails the suite until either a schema is trimmed or the number is raised **with a ledger entry justifying the cost**. A new always-on tool should first consider displacing an existing one into tier 2 (a `_TOOL_GATES` entry in `nx/_tool_gates.py`, surfaced by `search_tools`) — tier-2 tools cost 0 tokens until used. Adding a gated tool is a **one-file** change: one entry in `_TOOL_GATES` plus one in `_GATE_REFUSALS`, both in `nx/_tool_gates.py` — the gate vector, the withheld-tool refusals, and run_tool's pre-dispatch re-check all derive from that registry. See CONTRIBUTING.md for the tool touch points.

Prompt changes: the satellite system prompts (risk classifier, evidence reducer, session descriptions, sub-agent read/write/implementer scopes) live in `nx/_prompts.py` — edit them there, in one file, and the consuming module picks the constant up. Only the main `SYSTEM` lives in `nx/__init__.py`. `tests/perf/baselines.json`'s `conftest_snapshot` ledger pins the module-count cost of the conftest state rewind: splitting a module means re-recording it (and a large split means raising the budget with a justification, as was done for the 0.89.0 decomposition).

Ruff configuration includes two intentional `per-file-ignores` blocks in `pyproject.toml` — `nx/__init__.py` (F401, E402) and `tests/**` (E402, E702, E731, E741, F841) — plus an explicit `[tool.ruff.lint] select = ["E4", "E7", "E9", "F"]` (ruff 0.16+ otherwise defaults to all rules); preserve these rather than rearranging imports solely for linting.

## Testing Guidelines

Name pytest tests `test_*.py` and test functions `test_*`. Follow existing test setup: insert the repository root on `sys.path`, clear relevant `NX_*` variables, and set `NX_ENV_FILE` to `/dev/null` or a temporary file so local credentials cannot affect results. Add a focused regression test for parsing, approval, terminal, or concurrency changes. Run the relevant test file(s) for the touched area. If the change is behavioral (dispatch, parsing, approval, compression, terminal rendering, or module state), also run the full pytest suite before submitting — a docs-only or fully isolated addition (see CONTRIBUTING.md) doesn't need it. The suite is xdist-safe by construction: `tests/conftest.py` rewinds `NX_*` env vars and all `nx.*` module state around every test, so a new test file must not rely on state left behind by other files — set up what it needs (module scope or a fixture) instead of depending on execution order.

Every change to `nx/` or `tests/` also passes `uv run ruff check .` (hard gate) and adds no new `uv run ty check nx` diagnostics in the touched files. See CONTRIBUTING.md's "Lint and type check" section.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, commonly `fix: ...` for corrections (for example, `fix: preserve dropped tool tail during compression`). Keep commits narrowly scoped. Pull requests should explain the user-visible change, testing performed, and any configuration/documentation impact; link related issues when available. Include terminal screenshots only for UI or rendering changes.

## Project Tools

`.nx/tools/docupathic.py` wraps the `docupathic` research-corpus CLI (BM25 search, get, describe, status). When `NX_PROJECT_TOOLS=1`, discover the exact schemas with `search_project_tools` before first use, then invoke them via `call_project_tool` — do not ask the user to run these, and prefer them over re-deriving searches with `run_bash`. Structured queries (counts, filters, joins) go through the built-in `duckdb` tool against `<root>/docupathic.duckdb`.

## Configuration & Documentation

Use environment variables rather than adding a config-file format. Give new tunables an `NX_*` name, a safe default, and a README entry. New slash commands must be implemented in the appropriate `_repl*`/`_sessions_cmd` module (`nx/_repl.py`, `nx/_repl_settings_cmds.py`, `nx/_sessions_cmd.py`), registered in `_REPL_COMMANDS`, added to `_chatbox.py` completion data, and documented in the README.
