# Repository Guidelines

## Project Structure & Module Organization

`nx/` contains the Python package for the `nx` coding-agent CLI. Keep concerns in focused private modules (for example, `_turn.py`, `_sources.py`, and `_chatbox.py`) and expose/wire them through `nx/__init__.py` only as needed. `tests/` contains one test file per behavior area. `README.md` is the user-facing reference; `CONTRIBUTING.md` documents required touch points for slash commands, tools, and environment variables. `sources.example.env` provides safe configuration examples.

## Build, Test, and Development Commands

Use `uv` for local development:

```bash
uv pip install -e .                 # install the CLI and runtime dependencies
uv run pytest tests/                # run pytest-style tests
uv run python3 tests/test_sources.py # run a standalone assertion-style test
nx --help                           # inspect the installed CLI
```

Target a single pytest test during development with `uv run pytest tests/test_risk_context.py -k test_name`. The suite is designed to run offline; do not depend on a live model endpoint.

## Coding Style & Naming Conventions

Use standard Python conventions: four-space indentation, `snake_case` functions and variables, `CapWords` classes, and descriptive `NX_*` names for environment settings. Prefer a new focused `nx/_feature.py` module over growing `__init__.py`. Keep tool schemas and `SYSTEM` prompt text especially terse because they are sent every model turn. Ruff configuration includes intentional import exceptions for `nx/__init__.py`; preserve them rather than rearranging imports solely for linting.

## Testing Guidelines

Name pytest tests `test_*.py` and test functions `test_*`. Follow existing test setup: insert the repository root on `sys.path`, clear relevant `NX_*` variables, and set `NX_ENV_FILE` to `/dev/null` or a temporary file so local credentials cannot affect results. Add a focused regression test for parsing, approval, terminal, or concurrency changes. Run the relevant test file plus the full pytest suite before submitting.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, commonly `fix: ...` for corrections (for example, `fix: preserve dropped tool tail during compression`). Keep commits narrowly scoped. Pull requests should explain the user-visible change, testing performed, and any configuration/documentation impact; link related issues when available. Include terminal screenshots only for UI or rendering changes.

## Configuration & Documentation

Use environment variables rather than adding a config-file format. Give new tunables an `NX_*` name, a safe default, and a README entry. New slash commands must be implemented in `_repl.py`, added to `_chatbox.py` completion data, and documented in the README.
