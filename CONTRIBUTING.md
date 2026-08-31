# Contributing to nx

Guidance for extending `nx` without leaving loose ends — the kind of edge
case where a feature works when typed out in full but is invisible or
undocumented everywhere else.

## Adding a new slash command

A slash command has **three** touch points. Miss one and the command "works"
but is broken in a way that's easy to not notice:

1. **Dispatch** — `nx/_repl.py`, the `_REPL_COMMANDS` dict (command → handler
   from `_repl_settings_cmds.py` / `_sessions_cmd.py` / `_memory_cmd.py`).
   This is where the command actually runs.
2. **Chatbox hint menu** — `nx/_chatbox.py`, the `SLASH_COMMANDS` list
   (`(command, description)` tuples). Drives the autocomplete dropdown when
   typing `/`. A command missing here still works if typed in full, but users
   who rely on the menu will never discover it.
3. **README** — the `## Commands` table. User-facing reference doc.

Also add the command to `_HELP_COMMANDS` in `_repl_settings_cmds.py` (the
`/help` output) — a fourth touch point that's easy to miss. All four must be
updated together for a new command. When reviewing a diff that adds a
command, grep for the command string across all four files before
considering it done:

```bash
grep -rn '"/yourcommand"' nx/_repl.py nx/_chatbox.py nx/_repl_settings_cmds.py README.md
```

If `/help`-style guidance text exists inline for the command (e.g. the
"unknown arg" error branch), keep the accepted-args list in that message in
sync with the README row too — see `/reasoning`'s `on|off|last` as the
pattern to match.

If the command takes a fixed, enum-like first argument (a closed set of
keywords, not a freeform value like a path or session id), also add it to
`SLASH_ARG_OPTIONS` in `nx/_chatbox.py`. This drives a second autocomplete
dropdown that appears after typing `/cmd ` (mirrors the `/` command menu, but
for the argument). Keep the option list there in sync with whatever actually
validates the argument in `_repl.py` — see `/theme`, `/reasoning`,
`/approval`, and `/autocompress` as existing examples. A command missing here
still works if the argument is typed in full, but users who rely on the
dropdown won't discover valid values.

**Gotcha:** the arg dropdown stays "active" (`kind == "arg"`) as long as the
line is `/cmd <partial-or-full-word>` with no trailing space — including
right after a value has already been selected, since the completed line still
matches that shape. If Enter unconditionally re-inserts the selected value and
loops back to `render()`, the user can never submit — Enter appears to do
nothing. The fix (see the `\r` handler in `nx/_chatbox.py`): only re-insert
and `continue` when the selection actually changes the line; if the line
already equals `cmd + " " + chosen`, fall through to `finish()` and submit.
Any new dropdown-style completion added to the chatbox needs this same
"already-equal" escape hatch or it'll have the identical stuck-on-Enter bug.

## Adding a new tool

Tools (`TOOLS`, `FINAL_ANSWER_TOOL`) are load-bearing on token cost — see
CLAUDE.md. Beyond the schema itself:

- Tool body: read-only tools go in `nx/_tools.py`; state-touching tools
  (anything calling `_confirm()` or touching session/background-process
  globals) go in a focused submodule (e.g. `nx/_edit_tools.py` for the
  write/edit tools, `nx/_memory_cmd.py` for the memory tools), not loose in
  `__init__.py` — the same "prefer a new `nx/_feature.py` over growing
  `__init__.py`" rule from AGENTS.md.
- Dispatch: wire it into `run_tool()`.
- Risk classification: if the tool can write or execute, confirm
  `_assess_risk()` (`nx/_risk.py`) handles it sensibly — don't assume a new
  tool defaults to a safe risk tier.
- `SYSTEM` prompt: if behavior needs explaining to the model (e.g. the
  background/foreground contract for `run_bash`), keep that docstring in
  sync with the tool's actual behavior.

## Adding a new env var

Per CLAUDE.md: env vars, not config files. New tunables:

- `NX_*` prefix, read via `_env_int`/`_env_float` with a sane default.
- Document in the README's environment variables table.

## Keeping the suite fast and offline

The suite runs in ~15s; keep it that way (see tests/conftest.py and
tests/test_test_speed_tunables.py).

- Test modules must not dial the network, including at import time. pytest
  imports every test module before running anything, so a module-scope probe
  (a live compress(), a switch_source() against a real URL) becomes a hidden
  multi-second stall inside "collection". tests/conftest.py defaults
  NX_SKIP_STARTUP_PROBES=1 so `import nx` never probes; if your test needs a
  probe to actually fire, set NX_SKIP_STARTUP_PROBES=0 (or stub the
  client/urlopen) rather than removing that default, and undo any stubs
  before the module finishes loading — pytest keeps one process for all
  modules, so a stubbed resolve_model leaks into every later file.
- Interactive-latency budgets are env-tunable. Don't hardcode sleeps against
  constants like the 2.5s interrupt-restore wait or the watchdog grace: read
  them via _env_int/_env_float (NX_INTERRUPT_RESTORE_WAIT,
  NX_WATCHDOG_POLL_SECONDS, NX_WATCHDOG_KILL_GRACE, NX_HOOK_TIMEOUT) and let
  conftest shrink them. A test that is about the default budget (the
  slow-drain regression, for example) pops the env var for its own duration.
- Read env-var-dependent values at call time, not at module import: the
  standalone assertion-style files scrub NX_* at module scope, so an
  import-time _env_int freezes whatever the shell had during collection.

## Choosing what tests to run

The suite is fast, but "run everything every time" isn't required — scale the
verification to what the diff could actually break:

- **Docs/comments only** (README, CONTRIBUTING, CLAUDE.md, docstrings, or code
  comments with zero token change): no tests. Nothing executes it.
- **Isolated additions** — a new pure function with no callers, a new private
  module, a new env var read behind `_env_int`/`_env_float` with a default:
  its new tests plus the touched module's existing tests.
- **Behavioral changes** to dispatch, parsing, approval, compression,
  terminal rendering, or anything mutating module state (`ACTIVE`, caches,
  the interrupt watcher): the touched test files, then the full suite.

When in doubt (you touched `__init__.py` wiring, or a test that imports other
test helpers), run the full suite — it's the cheaper of the two mistakes.

## General checklist before calling a feature done

- [ ] Does it show up in the chatbox `/` hint menu (if it's a slash command)?
- [ ] If it takes an enum-like keyword argument, is it in `SLASH_ARG_OPTIONS`
      (`nx/_chatbox.py`) so the argument dropdown offers it?
- [ ] Is it in the README (Commands table or env var table)?
- [ ] Does an inline "unknown arg" / usage-error message (if any) match the
      documented syntax?
- [ ] If it touches `TOOLS`/`FINAL_ANSWER_TOOL`/`SYSTEM`, is the added text as
      terse as possible — it's sent every turn?
