# nx (pronounced "annex")

A coding agent that doesn't spend 50K tokens of context before you've said
anything. Point it at any OpenAI-compatible endpoint — a local llama.cpp /
vLLM / SGLang server, or a remote API like OpenRouter, Together, or Z.ai —
and get an agent that reads, writes, edits, and runs shell commands in your
project.

## Why context-frugal

Most agent harnesses burn 20K-50K+ tokens of system prompt and tool schemas
before your first message even lands. That's context you can't use for your
own code, and it's paid on *every single turn* — worse on local models, where
you're already tight on context and speed.

nx keeps the per-turn floor small: a terse `SYSTEM` prompt plus a handful of
tool schemas, nothing else. Run `nx --tokens` to print the exact,
mechanically-counted floor for your active backend/model — system prompt +
active tool schemas + a one-token "hey" — instead of an eyeballed number
that drifts as tools are added or removed.

It's also built to survive the rough edges of self-hosted models: falls back
to text-parsed tool-call tags when a server doesn't support native
tool-calling, recovers from malformed streams and reasoning-only stalls
automatically, and probes each source's tool-calling capability at startup so
the right protocol is picked without hand-tuning.

Every write/edit/bash call is risk-classified and gated by an approval prompt
(unless you pass `--yolo`); every session auto-saves so you can `--resume` it
later. `/help` lists all slash commands.

## Quick start

Install it as a `nx` command on your `$PATH`:

```bash
uv tool install git+https://github.com/nmishra93/nx.git
```

Developing in a checkout? Make edits apply immediately with an editable install:

```bash
git clone https://github.com/nmishra93/nx.git && cd nx
uv tool install -e .        # editable: edits in this checkout apply immediately
```

Either way, `uv tool upgrade nx` pulls the latest later on.

### Add an endpoint

`nx` reads credentials from `~/.env` (auto-loaded at startup — no config file):

```bash
export NX_BASE_URL=http://localhost:8080/v1
export NX_MODEL=your-model-name
export NX_API_KEY=sk-noop            # any string; local servers ignore it
python -c "import nx; nx.main()"
```

If `NX_MODEL` is unset, nx asks the server what it's serving.

## Configuration

nx reads config from environment variables and auto-loads `~/.env` at
startup (override the file with `NX_ENV_FILE`). No config file format.

**Single source:**

```
NX_BASE_URL=http://localhost:8080/v1
NX_MODEL=your-model-name
NX_API_KEY=sk-noop
```

**Multiple sources**, switchable at runtime with `/source [name]`:

```
NX_SOURCES=local,zai

NX_SOURCE_LOCAL_BASE_URL=http://localhost:8080/v1
NX_SOURCE_LOCAL_API_KEY=sk-noop

NX_SOURCE_ZAI_BASE_URL=https://api.z.ai/api/paas/v4
NX_SOURCE_ZAI_API_KEY=$zai_test        # $name = look up a key from env / ~/.env
NX_SOURCE_ZAI_MODEL=glm-x-preview
```

See [`sources.example.env`](sources.example.env) for a full annotated example.
Context is preserved across `/source` switches (`/reset` for a clean slate).

The `/source <name> <model>` dropdown lists the ids the server advertises
via `GET /v1/models` and scrolls when the list is long (↑/↓, PageUp/PageDown).
When a proxy's list is stale (retired models still listed, new ones missing),
pin the true lineup with `NX_SOURCE_<NAME>_MODELS` — see
[`sources.example.env`](sources.example.env).

Each source's max context window is probed once and cached, shown in the
`/source` list and the per-turn `ctx 12K/150K` footer (colorized green/yellow/red
by fullness). The footer also shows `cached: N%` — how much of that turn's
prompt was served from the server's KV/prompt cache (llama.cpp `cache_n`,
OpenAI `prompt_tokens_details.cached_tokens`) — green ≥75%, yellow ≥40%, red
below. A steady high number is the append-only discipline working; a sudden
collapse to ~0 means something rewrote early history and silently busted the
prefix. `/ctx` shows the same fraction per turn in its cache column. At
`NX_AUTOCOMPRESS_PERCENT` (default 85%) of the window, older
turns fold into a summary automatically — see `/autocompress`.

### Built-in sources

If their API key is present in the environment, nx auto-registers a built-in
source so you can `/source together` or `/source openrouter` without defining
`NX_SOURCE_*` vars by hand:

| key | source | default model |
| --- | --- | --- |
| `TOGETHER_API_KEY` | `together` | `zai-org/GLM-5.3` |
| `OPENROUTER_API_KEY` | `openrouter` | `z-ai/glm-5.3` (routed to `io-net/fp8`) |

Override either source's model with `/source <name> <model>`, and OpenRouter's
provider routing with `/provider` or `NX_SOURCE_OPENROUTER_EXTRA_BODY`.

**Hybrid thinking models:** some servers (glm-5.3-flash behind a vLLM proxy,
for one) run a thinking model with its reasoning *merged* into `content` —
no separator at all — unless the request carries a template toggle. nx
probes each source/model once (a couple of tiny requests) for a toggle that
makes the server separate `reasoning_content`, then attaches it to every
request. Servers that already separate reasoning are left untouched, and
servers that reject the extra field are unaffected. `NX_THINKING=off`
disables the probe and the attach.

### Flags

| flag | what it does |
| --- | --- |
| `--yolo` | start in never-prompt mode (auto-approve everything) |
| `--approval <all\|low\|medium\|high\|yolo>` | start with a non-default approval mode |
| `--source <name>` | start on a specific source |
| `--resume [target]` | resume a saved session; bare = most recent |
| `--session <id>` | start a fresh run attached to a specific session id |
| `--prompt-file <path>` / `-f <path>` | headless one-shot mode: run one turn and exit (`-` reads stdin) |
| `--json` | (with `--prompt-file`) emit a JSON result summary on stdout instead of streamed chat |
| `--tokens` / `--bench` | print the exact prompt-token floor and exit |
| `-h` / `--help` | print flag usage and exit |
| `--version` / `-v` | print the installed version and exit |

`nx sessions [query]` is a shell subcommand (not a flag): it lists/filters saved
sessions and exits without entering the REPL.

### Key environment variables

| env var | what it does |
| --- | --- |
| `NX_APPROVAL` | persistent default approval mode: `all`/`low`/`medium`/`high`/`yolo` (`/approval set-default <level>` rewrites it from the REPL) |
| `NX_ALLOWED_FILE` | the approval allowlist JSON (default `~/.nx/allowed.json`); rules are saved with `A` at an approval prompt or `/approval allow` and auto-allow matching actions before the risk classifier runs |
| `NX_BASE_URL` / `NX_MODEL` / `NX_API_KEY` | single-source config |
| `NX_SOURCES` / `NX_SOURCE_*` | named multi-source endpoints (`NX_SOURCE_<NAME>_MODELS` pins a source's model-id dropdown list when its `GET /v1/models` is stale) |
| `NX_ACTIVE` | source to start on (persistent version of `--source`) |
| `NX_ENV_FILE` | `.env` file to auto-load (default `~/.env`) |
| `NX_THEME` | `mocha` (default) / `latte` / `dracula` / `nord` / `gruvbox` / `vaporwave` |
| `NX_SPINNER` | the waiting/`running` animation (default `1`): a slow 12-cell sweep bar, at most two glyphs change per frame. `0` disables animation entirely and shows a static status line — recommended for photosensitive users |
| `NX_REASONING_VISIBLE` | stream live reasoning/thinking text (default `0` — collapsed; `1` streams it — toggle live with `/reasoning`) |
| `NX_THINKING` | `auto` (default) probes each source/model for a `chat_template_kwargs` thinking toggle that makes servers like glm-5.3-flash (vLLM) separate reasoning into `reasoning_content` instead of merging it into the answer; `off` keeps the server's stock behavior |
| `NX_TOOL_OUTPUT_VISIBLE` | render each tool call's result body (default `0` — collapsed, only the call header shows; `1` renders the result body — toggle live with `/tooloutput` (`/tools` legacy alias); `todo_write` always renders) |
| `NX_TOOL_RESULT_CHARS` | cap on tool-result characters entering context (default 20000; `0` disables the cap). Oversized results are head/tail-kept and their full text spilled to `~/.nx/overflow/<sha256>.txt` — the result tells the model the path so it can `read_file`/`search_files` the elided middle |
| `NX_TOOL_RESULT_SPILL` | write oversized tool results to `~/.nx/overflow/` (default `1`; `0` falls back to the old blind elision marker) |
| `NX_OVERFLOW_TTL_DAYS` | purge spill files older than N days (default 7, like bg-logs) |
| `NX_OVERFLOW_ECHO_CHARS` | max chars of head/tail content echoed back alongside the spill path (default 2000; `0` = path note only) |
| `NX_HOME` / `NX_SESSIONS_DIR` | where session JSON files are stored |
| `NX_HISTORY_LINES` | how many past prompt submissions Up/Down recalls across runs (`~/.nx/input_history.json`; default 100; `0` disables) |
| `NX_ORIENTATION` | append a one-time working-directory + read-only-tools note to fresh sessions (default `1`; `0` disables) |
| `NX_SUBAGENTS` | offer the read-only `spawn_agent` tool (default off — `1` enables; adds a schema to every turn's floor). Sub-agents get read/search tools plus a bash allowlist (grep/ls/du/ps/git log); 2+ `spawn_agent` calls in one turn run concurrently, and `async: true` runs one in the background (job id now, answer as a `[Runtime note]` on a later turn; `/bg` lists them) |
| `NX_SUBAGENT_WRITE` | opt-in write tier for sub-agents (`NX_SUBAGENT_WRITE=1`): `spawn_agent` additionally gets `write_file`/`edit_file`/`multi_edit`/`sed_edit`, each still confirmed by you through the normal approval gate (risk classification + Y/n with a diff preview) and tagged `(sub-agent)` in the tool box. Write-capable agents never run concurrently and cannot delete files or spawn other agents |
| `NX_VISION` | offer the `read_image` tool on native-tool-call sources (default off; requires the backend to accept image content parts) |
| `NX_IMAGE_MAX_BYTES` | max image size `read_image` will load into context (default 8 MB) |
| `NX_PRE_TOOL_HOOK` / `NX_POST_TOOL_HOOK` | shell command run around every tool call with `{tool, args(, result)}` JSON on stdin; a non-zero pre-hook exit vetoes the call (10s timeout, fail-closed) |
| `NX_BASH_MAX_SECONDS` | watchdog: reap a backgrounded `run_bash` job after N seconds (SIGTERM→SIGKILL; default 0 = off) |
| `NX_BASH_PROMPT_SECRETS` | `1` forces the approval prompt when a `run_bash` command looks like it carries a credential (sk-/ghp_/AKIA/PEM…), even in yolo |
| `NX_PRICE_<SOURCE>_INPUT` / `_OUTPUT` / `_CACHE` | $ per 1M tokens for a source; feeds `/cost`'s $ column (unset sources show token totals only) |
| `NX_AUTOCOMPRESS_PERCENT` | auto-compress threshold (default 85; `0` disables) |
| `NX_MEMORY_SEARCH_LIMIT` | max `memory_search` results per call (default 8; `0` unbounded) — an over-cap response says how many were cut instead of hiding them |
| `NX_VERIFY_CMD` | command auto-run ~20s after the last `.py` edit settles (e.g. `uv run pytest -q`); its pass/fail is injected as a `[Runtime note: …]` when it finishes. Set/clear/force with `/verify [cmd\|off\|run]` |
| `NX_VERIFY_SETTLE_SECS` | quiet period after the last edit before `NX_VERIFY_CMD` fires (default 20) |
| `NX_MAX_TOKENS` | token cap for streaming requests (default 16000; `0` omits it) |
| `NX_MAX_MODEL_TURNS` | cap on tool-call turns per run (default 200) |
| `NX_METRICS_URL` | optional endpoint to push cumulative token usage after each turn |
| `NX_HTTP_RETRIES` / `NX_HTTP_BACKOFF_BASE` | transient-failure retries for every endpoint call: 429/5xx and read/write timeouts are retried up to N times beyond the first attempt (default 2; `0` off) with exponential backoff (base seconds, default 0.5 → 0.5s, 1s, 2s…) or the server's `Retry-After` when it sends one (capped at 30s). Never retried: refused connections (a cold-start fails fast) and anything after the first streamed byte |
| `NX_BACKEND` | `vllm` to disable llama.cpp-only recovery samplers (min_p / repeat_penalty / DRY) that vLLM rejects |
| `NX_NATIVE_TOOLS` | opt into server-side native `tool_calls` (default off — text-protocol fallback is parsed and repaired client-side) |
| `NX_GRAMMAR` | grammar-constrain text-protocol tool calls on llama.cpp servers (opt-in; see below) |
| `NX_SEARCH_EXCLUDE_DIRS` | space-separated dirs the `grep` fallback skips (mirrors ripgrep's `.gitignore` handling; default includes `.venv node_modules __pycache__ build dist`) |
| `NX_<ROLE>_COLOR` | override one of ten semantic roles (see below) |

`nx --help` covers flags; the source (grep `_env_int`/`_env_float` calls in
`nx/`) has the full env var list — dozens more knobs tune recovery samplers,
stall detection, undo depth, log size, etc., all with sane defaults.

### Themes

Each theme supplies eight palette hues — `cyan`, `green`, `yellow`, `red`,
`magenta`, `gray`, `blue`, and `orange` — that ten semantic roles draw from:

| role | default hue | what it colors |
| --- | --- | --- |
| `TEXT` | green | the model's final answer text |
| `REASONING_TEXT` | gray | reasoning/thinking block text |
| `STATUS_BAR` | cyan | per-turn stats footer (tok/s, ctx, ttft, wall) |
| `TOOL_OUTPUT` | magenta | tool/bash result body lines |
| `ACCENT` | blue | borders, model name, chatbox prompt |
| `LABEL` | orange | source name / session id badges |
| `TODO` | orange | `todo_write` list items — **pending** items; **done** items use `SUCCESS` (green), **in_progress** items use `WARNING` (yellow) |
| `SUCCESS` | green | yolo/auto-approve/allow indicators |
| `WARNING` | yellow | medium-risk / approval-prompt / recoverable-cut notices |
| `ERROR` | red | hard failures |

`/theme` switches themes live and saves `NX_THEME` to `~/.env`. Any single
role can be repinned independently of the theme with
`NX_<ROLE>_COLOR=<hue|hex>` — the value is either a palette key (one of the
eight hues above, resolved against the *active* theme so it tracks
`/theme` switches) or a literal 6-digit hex (theme-independent). For example,
`NX_ACCENT_COLOR=magenta` makes borders/prompt magenta in every theme, while
`NX_TOOL_OUTPUT_COLOR=#fdfd96` pins tool output to an exact hex regardless of
theme. Invalid values are ignored with a warning.

### Tool-calling protocols

nx talks to any OpenAI-compatible endpoint, but servers vary widely in how
(or whether) they support tool calls. Rather than assume a fully-compliant
backend, nx probes each source at startup (`switch_source`) and classifies it:

- **Native** — the server returns structured `tool_calls` with valid JSON.
  Nothing extra needed; the server already constrains argument generation.
- **Grammar** — no native `tool_calls`, but a GBNF-constrained tool-call tag
  (derived from the active tool schemas) comes back well-formed. Worth
  attaching the grammar on every turn.
- **Text** — neither works (or the probe failed); fall back to plain-text
  tool-call tag parsing with no grammar constraint.

Servers without native tool-calling fall back to a text protocol, parsed out
of the assistant's reply — small models occasionally emit truncated or
invalid JSON in that tag, which the parser tries to recover from after the
fact. Setting `NX_GRAMMAR=1` against a llama.cpp server instead constrains
generation with a GBNF grammar (derived from the active tool schemas) once
the model emits the tool-call trigger, so malformed JSON becomes unsamplable
rather than something to clean up. It's opt-in and llama.cpp-only — the
`grammar`/`grammar_triggers` request fields are a llama.cpp server extension,
not part of the OpenAI API, and their exact shape has moved across llama.cpp
versions; an unsupported server just has the field ignored or rejected
(falling back automatically, same as an `NX_SOURCE_*_EXTRA_BODY` a server
doesn't recognize).

By default nx sends `tools=` so the schema renders into the prompt but sets
`tool_choice="none"`, leaving the text-protocol tag as the only way to invoke
a tool — which nx parses and can repair itself. Set `NX_NATIVE_TOOLS=1` to
opt into server-side native `tool_calls` for backends known to emit
well-formed arguments.

Qwen-style chat templates (qwen3.8-27b, Qwen3.8-flash-next, …) interleave a
literal newline as a `content` delta between their native `tool_calls`; nx
treats those whitespace-only separators as stream punctuation rather than
prose, so a multi-call turn renders its preamble once instead of once per
call.

## Commands

Run `/help` in the REPL for the full list. The essentials:

| command | what it does |
| --- | --- |
| `/source [name] [model]` | list sources, switch to one, or override its model |
| `/provider [source] [a,b,…\|off]` | show or set OpenRouter provider-routing order |
| `/yolo` | toggle auto-approve for writes and bash |
| `/approval [level]` | show or set risk threshold (`all`/`low`/`medium`/`high`/`yolo`; session-only)
| `/approval set-default <level>` | persist the default approval mode to `~/.env` |
| `/approval allow …` | manage the persistent allowlist: `list`, `forget <n>`, `forget-all`, or save `<tool> <pattern>` |
| `/theme [name]` | list color themes, or switch to one |
| `/reasoning on\|off\|last` | toggle or show the reasoning trace |
| `/tooloutput on\|off\|last` | toggle or show tool-call result visibility (`/tools` is a legacy alias; todo_write always renders) |
| `/sessions [n]` / `/resume [target]` | list or resume a past session (a non-title query falls through to full-text body search) |
| `/export [md] [path]` | write the current session to a markdown transcript |
| `/save [title]` | save the current session |
| `/delete [target]` | delete a saved session |
| `/memory save\|remember\|delete\|list` | manage persistent memory |
| `/compress` / `/compact` | fold older turns into a summary (keep last 2 verbatim; `/ctx`-pinned turns survive) |
| `/ctx [pin\|unpin\|clear]` | per-message token breakdown (total includes the per-turn `tools=` schema cost, shown as its own `+ tools schema` line); pin turns so every later compress keeps them verbatim |
| `/cost` | session usage totals + per-source scoreboard ($ when `NX_PRICE_<SOURCE>_*` is set) |
| `/autocompress [pct\|off\|on]` | show or set the auto-compress threshold; every fold auto-pins the latest `todo_write` turn so the live task plan survives compression |
| `/verify [cmd\|off\|run]` | show/set/clear or force-run the post-edit verify command (`NX_VERIFY_CMD`) — when set, every edited `.py` also gets an in-process `ast.parse` whose verdict rides on the edit's own tool result |
| `/undo` | revert the last write/edit (LIFO) |
| `/recover [note]` | force a low-temp visible checkpoint after a bad stream |
| `/bg [kill <pid>]` | list background `run_bash` jobs, or kill one |
| `/resources` | show this nx process's CPU% and RSS memory |
| `/init` | generate/update an `AGENTS.md` for this project |
| `/todo` | show the model's current `todo_write` list |
| `/reset` / `/clear` / `/new` | clear conversation, start a fresh session |
| `/quit` / `/exit` | exit |

### Keyboard shortcuts

In the REPL input box:

- **F1** opens `/help`, **Ctrl+L** runs `/clear`, **Ctrl+Q** runs `/exit`
  (so you don't have to type the full command).
- **Enter** submits; **Shift+Enter** (or **Ctrl+J**) inserts a newline.
- **Up/Down** navigate history; **Left/Right** move within a line;
  **Home/End** jump to line start/end; **Ctrl+U** clears the line;
  **Ctrl+C** cancels. History persists across runs (last `NX_HISTORY_LINES`
  submissions, no slash commands) — Up on a fresh `nx` reaches earlier
  runs' prompts.
- **Shift+Tab** toggles YOLO mode (mirrors `/yolo`), shown live in the box
  border — also works at an approval prompt.
- **Tab** accepts the highlighted completion.
- Typing **`@`** starts a fuzzy file-path suggestion dropdown (Tab to accept);
  the `@path` mention is expanded into file contents when the message is
  submitted.

A plain `input()` fallback is used when stdin/stdout isn't a TTY.

## Approval modes

Every write/edit/bash call is risk-classified (`low`/`medium`/`high`) by a
cheap model call before it runs:

| setting | prompts at | auto-allows |
| --- | --- | --- |
| _(default)_ / `all` | low + medium + high | — |
| `--approval low` | medium + high | low |
| `--approval medium` | high | low + medium |
| `--approval high` | — | low + medium + high |
| `--yolo` / `yolo` | — | everything; skips classifier |

The prompt shows the risk reason (`[risk: HIGH — recursive force delete in
/tmp]`) so you have context for the decision. **Y**/Enter approves, **n**
denies, **a** saves a persistent allow rule for this action's shape (an
`edit src/main.py` prompt blesses future edits in that file; a `run: pytest -q`
prompt blesses future pytest runs — reviewed with `/approval allow list`),
**Esc** stops the turn and drops back to the prompt. Saved rules are checked
before the risk classifier, so an allowed action costs no model round-trip.

Read-only tool calls in an all-read-only turn (e.g. three `read_file` calls
plus a `search_files`) execute concurrently; anything mutating or
approval-gated keeps strict sequential dispatch.

## Sessions

Every chat auto-saves to `~/.nx/sessions/` as one human-readable JSON file per
session (message array + id/title/description/source/timestamps + cumulative
token usage). Auto-save happens after every turn, so a crash never loses work.
`nx --resume <target>` (short id, prefix, title, or bare for most recent)
restores it, reselects the original source, and replays a recap of the
conversation. `nx sessions [query]` lists/filters sessions from the shell
without entering the REPL; inside the REPL, `/sessions` and `/resume` do the
same.

## Tools

Read-only tools never need approval; state-touching tools (write/edit/bash)
go through the approval gate above.

| tool | args | notes |
| --- | --- | --- |
| `read_file` | `path`, `offset?`, `limit?` | cat -n style (1-based); `path` may be a list |
| `write_file` | `path`, `content` | overwrites; requires confirmation |
| `edit_file` | `path`, `old`, `new`, `replace_all?`, `old_lines?` | `old` must match exactly once (or `replace_all`); `old_lines="10-14"` replaces a 1-based line range instead; unique indentation-only mismatches apply whitespace-tolerantly |
| `multi_edit` | `path`, `edits` | batch of `{old, new}`, one confirmation |
| `delete_file` | `path` | delete one file; restorable via `/undo` (unlike `rm` via bash) |
| `sed_edit` | `pattern`, `replacement`, `path` | regex find/replace over a file or glob |
| `apply_patch` | `diff`, `path?` | apply a unified diff; for multi-file restructure |
| `list_dir` | `path`, `pattern?` | flat list, or recursive glob (`**/*.py`) |
| `search_files` | `pattern`, `path?`, `glob?`, … | regex content search; read-only |
| `git_status` | `path?` | `git status --short -b`; read-only |
| `git_diff` | `path?`, `staged?`, `file?` | `git diff`; read-only |
| `git_log` | `path?`, `max_count?`, `file?` | `git log --oneline`; read-only |
| `run_bash` | `command`, `timeout?` | requires confirmation; backgrounds long-running commands |
| `wait_background` | `pid`, `log_path?`, `timeout?` | only offered after a `run_bash` call has backgrounded |
| `todo_write` | `todos` | rewrites whole list `[{text, status}]`; no per-item id |
| `memory_write` | `title`, `body`, `scope?` | save a durable memory note; requires confirmation |
| `memory_search` | `query`, `scope?` | keyword-search saved memories; title hits rank first; read-only |
| `memory_delete` | `target`, `scope?` | delete one memory by exact title, filename, or unique prefix; requires confirmation |
| `spawn_agent` | `instruction`, `max_turns?`, `async?` | delegate an investigation or task to a sub-agent (`NX_SUBAGENTS=1`); sub-agents can run an allowlisted read-only bash tier (grep/ls/du/ps/git log). Write-capable only with `NX_SUBAGENT_WRITE=1` (write_file/edit_file/multi_edit/sed_edit, each confirmed by the user with a diff preview, tagged `(sub-agent)`; run sequentially, no delete/no nested spawn). Send several read-only ones in one turn to run them concurrently. `async: true` returns a job id immediately and delivers the answer as a one-time `[Runtime note]` on a later turn — for long investigations that would otherwise block the parent loop; async agents are always read-only, and a failed agent still announces (FAILED note). `/bg` lists running/finished agents. Flash-tier models sometimes narrate "I'll spawn…" instead of emitting the call, or mis-key arguments (`prompt` instead of `instruction`) — both self-correct on retry, but heavy delegation wants a non-flash model |
| `read_image` | `path` | show a local image to a vision-capable model (`NX_VISION=1` + native tool calls) |

`read_file`/`list_dir` results are cached per-path and invalidated on writes
or any `run_bash`. `wait_background` is omitted from the tool schemas sent to
the model until a `run_bash` has actually backgrounded, so a session where
nothing ever backgrounds doesn't pay tokens for a tool it will never call.

## Memory

The model can save durable memory notes itself — no manual `/memory save`
needed. Three tools drive it: `memory_search` (recall prior learnings before
re-deriving them), `memory_write` (persist a new one), and `memory_delete`
(retire a superseded one). At session start a short note listing
already-saved titles is appended to context so the model knows what's
remembered; a resumed session gets the same note when its transcript predates
it. Searches require every query term to match, rank title hits above body
hits, and cap results at `NX_MEMORY_SEARCH_LIMIT` (default 8) so a large
store can't flood a turn with snippets.

Memories live in two scopes:

- **project** — `<project_root>/.nx/memories/` (the default; specific to this
  repo, so a memory about one project doesn't clutter another). With no
  project root (bare cwd / outside a repo) project scope falls back to global.
- **global** — `~/.nx/memories/` (cross-project: conventions, tool quirks,
  general learnings).

Each memory is a markdown file (`<timestamp>_<slug>.md`) starting with a
`# Title`. All writes go through one shared writer: files are created
`0600` (dirs `0700`) regardless of umask, written atomically (tmp + rename),
and never clobber — a same-second same-title collision gets a `-2`, `-3`…
suffix instead of overwriting. `memory_write` and `memory_delete` go through
the approval gate like `write_file`/`delete_file`; `memory_search` is
read-only. The manual `/memory save|remember|delete|list` slash commands
remain for distilling a whole session at once (project scope, same
convention) and hand-managing notes.

## Metrics

nx logs token usage (input/output/cache-read/reasoning, per the OpenAI usage
convention, with a llama.cpp `timings` and a client-side character-estimate
fallback) into each session's JSON, always on. The cache-read numbers also
surface live: `cached: N%` in the per-turn footer and the cache column in `/ctx` come from
what the server reports. Three states are distinguished: a fraction (rendered
with the green/yellow/red thresholds above), an explicit `0%` (server said it
served nothing from cache — the busted-prefix signal), and `—` / no `cached`
field (server sent no cache breakdown at all — common on vLLM builds, which
omit `prompt_tokens_details.cached_tokens` unless something actually hit;
some gateways never send it). Only the middle one is a warning; a server that
never reports cache numbers renders nothing rather than a lying `0%`. Set `NX_METRICS_URL` to also
POST cumulative totals to a dashboard after each turn (fire-and-forget,
disables itself on first failure — nothing leaves the machine unless you set
this).

The footer's `tok/s` measures generation pace, not whole-turn wall-clock:
client-side it divides the server's `completion_tokens` by the decode window
(first→last token arrival, so the prefill head — huge on a turn that just
read a big file into the prompt — and the render/teardown tail are excluded).
llama.cpp servers report `timings.predicted_per_second` directly and that
server-measured figure is shown as-is. When the window is under ~0.5s (a
short tool call can arrive in one or two SSE reads) the figure is
pacing-free guessing and renders as `~N.N tok/s` — treat those as order-of-
magnitude only.

## Architecture

`nx/` is a small package (`__init__.py` + focused submodules like `_turn.py`,
`_sources.py`, `_repl.py`, `_chatbox.py`) rather than one monolithic file — see
[`CLAUDE.md`](CLAUDE.md) for the module map and design conventions if you're
extending it, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the required touch
points when adding slash commands, tools, or env vars.

## License

MIT License. See [`LICENSE`](LICENSE).
