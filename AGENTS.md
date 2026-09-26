# Repository Guidelines

## Project Structure & Module Organization

`nx/` contains the Python package for the `nx` coding-agent CLI. Keep concerns in focused private modules and expose/wire them through `nx/__init__.py` only as needed. The formerly monolithic core files were decomposed across 0.87.2–0.89.2; the current map:

- **Turn loop** — `nx/_turn.py` (model_turn, the loop, tool-call handlers, TURN_* statuses) plus `nx/_turn_footer.py` (stats footer, ctx probe), `nx/_tool_prefetch.py` (read-only parallel fan-out, cache prefetch), `nx/_argsalvage.py` (malformed-args salvage, breaker nudge), `nx/_turn_retry.py` (step retry ladder, batch deadline TLS).
- **Dispatch** — `nx/_dispatch.py` (DISPATCH table, run_tool) plus `nx/_tool_gates.py` (the tiered registry: `_TOOL_GATES`, `_active_tools`, `_gate_vector`, search_tools, withheld-tool refusals), `nx/_breaker.py` (malformed-args circuit breaker), `nx/_session_preamble.py` (fresh-session message list, one-time nudge strings), `nx/_bg_completions.py` (bg-job completion sweep).
- **Requests/compression** — `nx/_compress.py` (open_stream, compress, the env-derived request flags) plus `nx/_runtime_notes.py` (the [Runtime note] contract), `nx/_request_opts.py` (recovery sampler, grammar, /toolcalls mode, max_tokens clamp), `nx/_stream_errors.py` (error classification).
- **Sub-agents** — `nx/_subagent.py` (spawn_agent, async machinery) plus `nx/_subagent_tiers.py` (tool tiers, restricted dispatch tables), `nx/_subagent_bash.py` (read-only bash allowlist), `nx/_delegation_note.py`, `nx/_subagent_tls.py`.
- **Memory** — `nx/_memory_cmd.py` (the /memory UI + memory_write) plus `nx/_memory_store.py` (the one scanner/writer), `nx/_memory_search.py`, `nx/_memory_note.py`, `nx/_memory_archive.py`.
- **Prompts** — `nx/_prompts.py` is the single home for the satellite system prompts (RISK_SYSTEM, REDUCER_SYSTEM_PROMPT, DESC_SYSTEM, SUBAGENT_SYSTEM/SUBAGENT_WRITE_SYSTEM); the main `SYSTEM` stays in `nx/__init__.py` (load-bearing on the token floor; its bytes are pinned by `_prefix.py`).

When you split or extend a module, moved names are re-exported from the old home so the import surface (and tests that patch module attributes) keep working. `tests/` contains one test file per behavior area. `README.md` is the user-facing reference; `CONTRIBUTING.md` documents required touch points for slash commands, tools, prompts, and environment variables. `sources.example.env` provides safe configuration examples.

## Semver Policy

From now on we follow semver; bump version numbers as changes are made.

## Build, Test, and Development Commands

Use `uv` for local development:

```bash
uv pip install -e .                 # install the CLI and runtime dependencies
uv run pytest tests/ -n auto        # run pytest-style tests in parallel (~10s; workers capped at 32)
uv run python3 tests/test_sources.py # run a standalone assertion-style test
uv run ruff check .                 # lint gate — must be clean before committing
uv run ty check nx                  # type check; no new diagnostics in touched files
nx --help                           # inspect the installed CLI
```

`-n auto` is the default way to run the suite; plain `uv run pytest tests/` (serial, ~27s) works too and is the fallback if xdist is unavailable.

Target a single pytest test during development with `uv run pytest tests/test_risk_context.py -k test_name`. The suite is designed to run offline; do not depend on a live model endpoint.

Ruff and ty are pinned in the `dev` dependency group (installed with `uv sync --group dev`). Ruff is a hard gate; ty's existing ~125 diagnostics are tracked, not owed — see CONTRIBUTING.md's "Lint and type check" section for the policy.

## Coding Style & Naming Conventions

Use standard Python conventions: four-space indentation, `snake_case` functions and variables, `CapWords` classes, and descriptive `NX_*` names for environment settings. Prefer a new focused `nx/_feature.py` module over growing `__init__.py`.

### Prompt & schema budget

We target models with 128K-1M-token context windows, so the per-turn floor of `SYSTEM` prompt + always-on tool schemas is held under a **hard, recorded budget of ~5K tokens** (`BUDGET_TOKENS` in `tests/test_initial_context.py`; currently 4800, measured floor ~3.4K). Within that ceiling, favor *clarity over terseness*: every argument the model can pass gets a `description`, overlapping tools name each other, and self-healing behavior is spelled out in the schema. Do not abbreviate prose to save a handful of tokens — the budget is enforced mechanically, so a schema only needs trimming when the floor actually approaches the ceiling.

The budget is a forcing function, not a soft warn: a change that pushes the floor past `BUDGET_TOKENS` fails the suite until either a schema is trimmed or the number is raised **with a ledger entry justifying the cost**. A new always-on tool should first consider displacing an existing one into tier 2 (a `_TOOL_GATES` entry in `nx/_tool_gates.py`, surfaced by `search_tools`) — tier-2 tools cost 0 tokens until used. Adding a gated tool is a **one-file** change: one entry in `_TOOL_GATES` plus one in `_GATE_REFUSALS`, both in `nx/_tool_gates.py` — the gate vector, the withheld-tool refusals, and run_tool's pre-dispatch re-check all derive from that registry. See CONTRIBUTING.md for the tool touch points.

Prompt changes: the satellite system prompts (risk classifier, evidence reducer, session descriptions, sub-agent scopes) live in `nx/_prompts.py` — edit them there, in one file, and the consuming module picks the constant up. Only the main `SYSTEM` lives in `nx/__init__.py`. `tests/perf/baselines.json`'s `conftest_snapshot` ledger pins the module-count cost of the conftest state rewind: splitting a module means re-recording it (and a large split means raising the budget with a justification, as was done for the 0.89.0 decomposition).

Ruff configuration includes intentional import exceptions for `nx/__init__.py`; preserve them rather than rearranging imports solely for linting.

## Testing Guidelines

Name pytest tests `test_*.py` and test functions `test_*`. Follow existing test setup: insert the repository root on `sys.path`, clear relevant `NX_*` variables, and set `NX_ENV_FILE` to `/dev/null` or a temporary file so local credentials cannot affect results. Add a focused regression test for parsing, approval, terminal, or concurrency changes. Run the relevant test file(s) for the touched area. If the change is behavioral (dispatch, parsing, approval, compression, terminal rendering, or module state), also run the full pytest suite before submitting — a docs-only or fully isolated addition (see CONTRIBUTING.md) doesn't need it. The suite is xdist-safe by construction: `tests/conftest.py` rewinds `NX_*` env vars and all `nx.*` module state around every test, so a new test file must not rely on state left behind by other files — set up what it needs (module scope or a fixture) instead of depending on execution order.

Every change to `nx/` or `tests/` also passes `uv run ruff check .` (hard gate) and adds no new `uv run ty check nx` diagnostics in the touched files. See CONTRIBUTING.md's "Lint and type check" section.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, commonly `fix: ...` for corrections (for example, `fix: preserve dropped tool tail during compression`). Keep commits narrowly scoped. Pull requests should explain the user-visible change, testing performed, and any configuration/documentation impact; link related issues when available. Include terminal screenshots only for UI or rendering changes.

## Configuration & Documentation

Use environment variables rather than adding a config-file format. Give new tunables an `NX_*` name, a safe default, and a README entry. New slash commands must be implemented in `_repl.py`, added to `_chatbox.py` completion data, and documented in the README.
