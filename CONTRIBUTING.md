# Contributing to nx

Guidance for extending `nx` without leaving loose ends — the kind of edge
case where a feature works when typed out in full but is invisible or
undocumented everywhere else.

## Adding a new slash command

A slash command has **three** touch points. Miss one and the command "works"
but is broken in a way that's easy to not notice:

1. **Dispatch** — `nx/_repl.py`, the `if user == "/x" or user.startswith("/x "):`
   chain in the main REPL loop. This is where the command actually runs.
2. **Chatbox hint menu** — `nx/_chatbox.py`, the `SLASH_COMMANDS` list
   (`(command, description)` tuples). Drives the autocomplete dropdown when
   typing `/`. A command missing here still works if typed in full, but users
   who rely on the menu will never discover it.
3. **README** — the `## Commands` table. User-facing reference doc.

All three must be updated together for a new command. When reviewing a diff
that adds a command, grep for the command string across all three files
before considering it done:

```bash
grep -rn '"/yourcommand"' nx/_repl.py nx/_chatbox.py README.md
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

## General checklist before calling a feature done

- [ ] Does it show up in the chatbox `/` hint menu (if it's a slash command)?
- [ ] If it takes an enum-like keyword argument, is it in `SLASH_ARG_OPTIONS`
      (`nx/_chatbox.py`) so the argument dropdown offers it?
- [ ] Is it in the README (Commands table or env var table)?
- [ ] Does an inline "unknown arg" / usage-error message (if any) match the
      documented syntax?
- [ ] If it touches `TOOLS`/`FINAL_ANSWER_TOOL`/`SYSTEM`, is the added text as
      terse as possible — it's sent every turn?
