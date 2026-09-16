# Contributing to nx

Guidance for extending `nx` without leaving loose ends — the kind of edge
case where a feature works when typed out in full but is invisible or
undocumented everywhere else.

## Adding a new slash command

A slash command has **three** touch points. Miss one and the command "works"
but is broken in a way that's easy to not notice:

1. **Dispatch** — `nx/_repl.py`, the `_REPL_COMMANDS` dict (command → handler
   from `_repl_settings_cmds.py` / `_sessions_cmd.py` / `_memory_cmd.py`, and
   subcommand modules like `_memory_optimize.py` reached via their parent
   handler). This is where the command actually runs.
2. **Chatbox hint menu** — `nx/_chatbox.py`, the `SLASH_COMMANDS` list
   (`(command, description)` tuples). Drives the autocomplete dropdown when
   typing `/`. A command missing here still works if typed in full, but users
   who rely on the menu will never discover it.
3. **README** — the `## Commands` table. User-facing reference doc.

Also add the command to `_HELP_COMMANDS` in `_repl_settings_cmds.py` (the
`/help` output) — a fourth touch point that's easy to miss. All four must be
updated together for a new command. When reviewing a diff that adds a
command, grep for the command string across all four files to verify:

```bash
grep -rn '"/yourcommand"' nx/_repl.py nx/_chatbox.py nx/_repl_settings_cmds.py README.md
```

Subcommands can live in the same module as their parent's dispatch (see
`/memory optimize` in `nx/_memory_optimize.py`, reached by a lazy import from
`_cmd_memory` because it imports `_memory_cmd` at module level, with a test
re-export in `nx/_repl.py`) — the grep above won't surface that indirection, so
mention the module in the commit/PR instead.

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
- Tiering (0.62.0, the feature-cost rule): a new always-on tool adds to the
  per-turn floor that every session pays. Prefer making it tier-2 first: add a
  `_TOOL_GATES` entry in `nx/_dispatch.py` so it is withheld until relevant
  and surfaced on demand by `search_tools` (0 tokens until used). An always-on
  tool may instead raise `BUDGET_TOKENS` in `tests/test_initial_context.py`
  with a ledger entry justifying the cost — the ceiling is ~4K tokens, and the
  point is to stay under it, not to compress every schema to the minimum.
  `tests/perf/baselines.json`'s `active_tools_floor` pins the always-on count.
- Risk classification: if the tool can write or execute, confirm
  `_assess_risk()` (`nx/_risk.py`) handles it sensibly — don't assume a new
  tool defaults to a safe risk tier.
- `SYSTEM` prompt: if behavior needs explaining to the model (e.g. the
  background/foreground contract for `run_bash`), keep that docstring in
  sync with the tool's actual behavior.
- Schema wording: every property the model can pass gets a `description` (an
  undocumented argument is a guessed-at one), required properties are declared
  before optional ones, and overlapping tools name each other (write_file vs
  edit_file, edit_file vs multi_edit). `tests/test_tool_schema_invariants.py`
  enforces all three, including the `UNDOCUMENTED_OK` exemption list — add to
  it only when the property name really is the whole contract.

## Adding a new env var

Per CLAUDE.md: env vars, not config files. New tunables:

- `NX_*` prefix, read via `_env_int`/`_env_float` with a sane default.
- README entry: mandatory (0.62.0, I-10). `tests/test_env_docs_parity.py`
  fails on any `NX_*` read in `nx/` that the README never names; internal
  tunables that shouldn't be user-facing go on that test's `_ALLOWLIST` with a
  reason instead.
- Document in the README's environment variables table.

**Per-source variants.** A setting that can differ per endpoint takes the
`NX_SOURCE_<NAME>_<SETTING>` form (read at call time from the source's `name`,
as `nx/_manifest.py`'s `cache_routing_enabled()` does) and should also have a
global `NX_<SETTING>` default. The per-source value, when set, wins.

## Caching what a server tells you

Anything learned by *asking* an endpoint — its accepted request shape, its
tool-calling protocol, its model id, its context window — is
**(a) keyed by `(source, model)`, (b) cached on the Source instance, and
(c) invalidated on the events that change it** (`/source` switches, a
`/source <name> <model>` override, a `PROBE_GENERATION` bump). `nx/_manifest.py`
is the shared home for the request-shape half and the routing hint; the
capability/thinking/context probes cache on `Source` attributes
(`src._capability`, `src._thinking_kwargs`, `src._context_window`). The rules:

- A cache lives on the `Source`, never in a module global — conftest rewinds
  `nx.*` module state between tests, but a `Source` built inside a test is the
  test's own object.
- A *failure* is cached only when re-probing it would be harmful (a
  silently-clamped over-max_tokens probe); otherwise leave the miss uncached so
  a server that comes up later is still found. `resolve_model()` memoizes only
  a real model id, never the `"local-model"` placeholder.
- Prefer pinning a learned answer over re-deriving it per request: the
  no-tools / no-extra ladder is walked **once** per `(source, model)`, then the
  winning rung is reused (`tests/test_perf_capability_manifest.py`, T-11).
- Every per-turn HTTP request count is a number we can name; if a change adds
  one, pin it in a test rather than letting it drift (PERF.md I-9).

## Adding a new persistent store / index file

Anything that caches state derived from the session transcripts (the
consolidated SQLite store `nx/_storeindex.py`, the image sidecar dir) must
behave as a **cache, never the source of truth**:

- A missing, corrupt, or locked store degrades to the pre-index behavior
  (full transcript parse) — helpers return empty/None, they never raise into
  listing/GC paths. A corrupt SQLite file is set aside (`*.corrupt`), not
  left in the way.
- Rows/files for deleted sessions are pruned opportunistically (indexed
  saves, search refresh) and by `nx sessions --prune` — a stale row must
  never resurface a deleted session id in a listing or a body search.
- Created 0600 in a 0700 dir via `open_private`/`private_dir`
  (`tests/test_private_sinks.py` checks).
- Tests write it through the public helpers (`m._write_session`,
  `m._delete_session`) rather than hand-crafting internals, and each test
  file pins its own `NX_SESSIONS_DIR` (see `tests/test_session_index.py`).

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
  (nx/_images.py got this wrong before 0.38.0: NX_IMAGE_MAX_BYTES was
  resolved at import; now the module attribute is the default and
  _env_int runs inside read_image.)

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
      *clear and complete* as the model benefits from — every argument
      documented, overlapping tools named — rather than squeezed to the fewest
      possible characters? The floor is budgeted (~4K), so favor description
      over terseness.

## Adding a tool: self-healing checklist

The self-healing stack (grammar-constrained decoding, text-protocol parsing,
JSON repair, malformed-args nudges/breaker, `NX_ARG_REPAIR`) is driven off the
schema you declare, so a new tool is only as healable as its schema is
complete. When you add an entry to `TOOLS`:

- [ ] Schema first: flat `properties`, correct `required` list, `required`
      before optional. `NX_ARG_REPAIR` coerces and renames against exactly
      these declarations; a wrong `required` list makes the pre-check refuse
      calls the body would have accepted (or accept calls it can't use).
- [ ] Body signature matches the schema key-for-key (plus `**_` for extras) —
      `tests/test_tool_schema_invariants.py` asserts this, and the
      required-args pre-check refuses before your body ever sees a call.
- [ ] Does the tool have near-miss NAME spellings similarity can never reach
      (a one-word name like `bash`, or another ecosystem's name like
      `str_replace_editor`)? Add them to `_TOOL_ALIASES` in `nx/_argrepair.py`
      — an alias never fires when the spelling is itself a real tool name.
- [ ] Does it have near-miss ARG keys (a `command` vs `cmd`, a `path` that
      models keep sending as `file_path`)? Add a per-tool row to
      `_ARG_ALIASES` in `nx/_argrepair.py`. Targets must be declared
      properties; the no-clobber rule applies automatically.
- [ ] If the model should be steered to a sibling tool when this one's args
      keep truncating, make sure `_malformed_args_nudge`'s fallback map in
      `nx/_turn.py` mentions it (or that the generic "use one of: …" wording
      is acceptable).
- [ ] Unknown-tool refusals name the gate for withheld tools: if your tool is
      conditionally offered by `_active_tools()`, add a branch to
      `_unknown_tool_error()` in `nx/_dispatch.py` saying why it's missing and
      what to do instead — and cover it in `tests/test_unknown_tool.py`.
- [ ] If the tool returns anything but a plain string (a content-parts array,
      like `read_image`), audit every consumer before shipping: the terminal
      preview and the `/tooloutput` stash (nx/_dispatch.py run_tool), the
      text-protocol Observation interpolation (both branches in
      nx/_turn.py), the overrun auto-compress retry (nx/_compress.py),
      `--trace` (nx/_cli.py), batch stdout (nx/_batch.py), session
      flattening (nx/_sessions_io.py), and the post-hook JSON. Route them
      through `_images._parts_preview` / `_strip_image_parts` so a large
      non-text payload can never reach a text channel — the 0.38.0
      read_image work is the worked example.
- [ ] `NX_ARG_REPAIR` covers every tool automatically via `_schemas()`; no
      per-tool registration. If you added a module-level conditional schema,
      make sure it lands in `TOOLS` (or `_EXTRA_TOOL_SCHEMAS`) so repair and
      the unknown-tool name list can see it.
