# Repository Guidelines

## Project Structure & Module Organization

`nx/` contains the Python package for the `nx` coding-agent CLI. Keep concerns in focused private modules (for example, `_turn.py`, `_sources.py`, and `_chatbox.py`) and expose/wire them through `nx/__init__.py` only as needed. `tests/` contains one test file per behavior area. `README.md` is the user-facing reference; `CONTRIBUTING.md` documents required touch points for slash commands, tools, and environment variables. `sources.example.env` provides safe configuration examples.

## Semver Policy

From now on we follow semver; bump version numbers as changes are made.

## Build, Test, and Development Commands

Use `uv` for local development:

```bash
uv pip install -e .                 # install the CLI and runtime dependencies
uv run pytest tests/ -n auto        # run pytest-style tests in parallel (~10s; workers capped at 32)
uv run python3 tests/test_sources.py # run a standalone assertion-style test
nx --help                           # inspect the installed CLI
```

`-n auto` is the default way to run the suite; plain `uv run pytest tests/` (serial, ~27s) works too and is the fallback if xdist is unavailable.

Target a single pytest test during development with `uv run pytest tests/test_risk_context.py -k test_name`. The suite is designed to run offline; do not depend on a live model endpoint.

## Coding Style & Naming Conventions

Use standard Python conventions: four-space indentation, `snake_case` functions and variables, `CapWords` classes, and descriptive `NX_*` names for environment settings. Prefer a new focused `nx/_feature.py` module over growing `__init__.py`.

### Prompt & schema budget

We target models with 128K-1M-token context windows, so the per-turn floor of `SYSTEM` prompt + always-on tool schemas is held under a **hard, recorded budget of ~5K tokens** (`BUDGET_TOKENS` in `tests/test_initial_context.py`; currently 4800, measured floor ~3.4K). Within that ceiling, favor *clarity over terseness*: every argument the model can pass gets a `description`, overlapping tools name each other, and self-healing behavior is spelled out in the schema. Do not abbreviate prose to save a handful of tokens — the budget is enforced mechanically, so a schema only needs trimming when the floor actually approaches the ceiling.

The budget is a forcing function, not a soft warn: a change that pushes the floor past `BUDGET_TOKENS` fails the suite until either a schema is trimmed or the number is raised **with a ledger entry justifying the cost**. A new always-on tool should first consider displacing an existing one into tier 2 (a `_TOOL_GATES` entry in `nx/_tool_gates.py`, surfaced by `search_tools`) — tier-2 tools cost 0 tokens until used. See CONTRIBUTING.md for the tool touch points.

Ruff configuration includes intentional import exceptions for `nx/__init__.py`; preserve them rather than rearranging imports solely for linting.

## Testing Guidelines

Name pytest tests `test_*.py` and test functions `test_*`. Follow existing test setup: insert the repository root on `sys.path`, clear relevant `NX_*` variables, and set `NX_ENV_FILE` to `/dev/null` or a temporary file so local credentials cannot affect results. Add a focused regression test for parsing, approval, terminal, or concurrency changes. Run the relevant test file(s) for the touched area. If the change is behavioral (dispatch, parsing, approval, compression, terminal rendering, or module state), also run the full pytest suite before submitting — a docs-only or fully isolated addition (see CONTRIBUTING.md) doesn't need it. The suite is xdist-safe by construction: `tests/conftest.py` rewinds `NX_*` env vars and all `nx.*` module state around every test, so a new test file must not rely on state left behind by other files — set up what it needs (module scope or a fixture) instead of depending on execution order.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, commonly `fix: ...` for corrections (for example, `fix: preserve dropped tool tail during compression`). Keep commits narrowly scoped. Pull requests should explain the user-visible change, testing performed, and any configuration/documentation impact; link related issues when available. Include terminal screenshots only for UI or rendering changes.

## Configuration & Documentation

Use environment variables rather than adding a config-file format. Give new tunables an `NX_*` name, a safe default, and a README entry. New slash commands must be implemented in `_repl.py`, added to `_chatbox.py` completion data, and documented in the README.
