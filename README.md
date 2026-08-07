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

Each source's max context window is probed once and cached, shown in the
`/source` list and the per-turn `ctx 12K/150K` footer (colorized green/yellow/red
by fullness). At `NX_AUTOCOMPRESS_PERCENT` (default 85%) of the window, older
turns fold into a summary automatically — see `/autocompress`.

### Built-in sources

If their API key is present in the environment, nx auto-registers a built-in
source so you can `/source together` or `/source openrouter` without defining
`NX_SOURCE_*` vars by hand:

| key | source | default model |
| --- | --- | --- |
| `TOGETHER_API_KEY` | `together` | `zai-org/GLM-5.2` |
| `OPENROUTER_API_KEY` | `openrouter` | `z-ai/glm-5.2` (routed to `parasail/fp8`) |

Override either source's model with `/source <name> <model>`, and OpenRouter's
provider routing with `/provider` or `NX_SOURCE_OPENROUTER_EXTRA_BODY`.

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
| `--version` | print the installed version and exit |

`nx sessions [query]` is a shell subcommand (not a flag): it lists/filters saved
sessions and exits without entering the REPL.

### Key environment variables

| env var | what it does |
| --- | --- |
| `NX_APPROVAL` | persistent default approval mode: `all`/`low`/`medium`/`high`/`yolo` |
| `NX_BASE_URL` / `NX_MODEL` / `NX_API_KEY` | single-source config |
| `NX_SOURCES` / `NX_SOURCE_*` | named multi-source endpoints |
| `NX_ACTIVE` | source to start on (persistent version of `--source`) |
| `NX_ENV_FILE` | `.env` file to auto-load (default `~/.env`) |
| `NX_THEME` | `mocha` (default) / `latte` / `dracula` / `nord` / `gruvbox` / `vaporwave` |
| `NX_REASONING_VISIBLE` | stream live reasoning/thinking text (default `0` — collapsed; `1` streams it — toggle live with `/reasoning`) |
| `NX_TOOL_OUTPUT_VISIBLE` | render each tool call's result body (default `0` — collapsed, only the call header shows; `1` renders the result body — toggle live with `/tooloutput` (`/tools` legacy alias); `todo_write` always renders) |
| `NX_HOME` / `NX_SESSIONS_DIR` | where session JSON files are stored |
| `NX_AUTOCOMPRESS_PERCENT` | auto-compress threshold (default 85; `0` disables) |
| `NX_MAX_TOKENS` | token cap for streaming requests (default 16000; `0` omits it) |
| `NX_MAX_MODEL_TURNS` | cap on tool-call turns per run (default 200) |
| `NX_METRICS_URL` | optional endpoint to push cumulative token usage after each turn |
| `NX_BACKEND` | `vllm` to disable llama.cpp-only recovery samplers (min_p / repeat_penalty / DRY) that vLLM rejects |
| `NX_NATIVE_TOOLS` | opt into server-side native `tool_calls` (default off — text-protocol fallback is parsed and repaired client-side) |
| `NX_GRAMMAR` | grammar-constrain text-protocol tool calls on llama.cpp servers (opt-in; see below) |
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

## Commands

Run `/help` in the REPL for the full list. The essentials:

| command | what it does |
| --- | --- |
| `/source [name] [model]` | list sources, switch to one, or override its model |
| `/provider [source] [a,b,…\|off]` | show or set OpenRouter provider-routing order |
| `/yolo` | toggle auto-approve for writes and bash |
| `/approval [level]` | show or set risk threshold (all/low/medium/high/yolo) |
| `/theme [name]` | list color themes, or switch to one |
| `/reasoning on\|off\|last` | toggle or show the reasoning trace |
| `/tooloutput on\|off\|last` | toggle or show tool-call result visibility (`/tools` is a legacy alias; todo_write always renders) |
| `/sessions [n]` / `/resume [target]` | list or resume a past session |
| `/save [title]` | save the current session |
| `/delete [target]` | delete a saved session |
| `/memory save\|remember\|list` | manage persistent memory |
| `/compress` / `/compact` | fold older turns into a summary (keep last 2 verbatim) |
| `/autocompress [pct\|off\|on]` | show or set the auto-compress threshold |
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
  **Ctrl+C** cancels.
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
denies, **Esc** stops the turn and drops back to the prompt.

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
| `edit_file` | `path`, `old`, `new` | `old` must match exactly once (or `replace_all`) |
| `multi_edit` | `path`, `edits` | batch of `{old, new}`, one confirmation |
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

`read_file`/`list_dir` results are cached per-path and invalidated on writes
or any `run_bash`. `wait_background` is omitted from the tool schemas sent to
the model until a `run_bash` has actually backgrounded, so a session where
nothing ever backgrounds doesn't pay tokens for a tool it will never call.

## Metrics

nx logs token usage (input/output/cache-read/reasoning, per the OpenAI usage
convention, with a llama.cpp `timings` and a client-side character-estimate
fallback) into each session's JSON, always on. Set `NX_METRICS_URL` to also
POST cumulative totals to a dashboard after each turn (fire-and-forget,
disables itself on first failure — nothing leaves the machine unless you set
this).

## Architecture

`nx/` is a small package (`__init__.py` + focused submodules like `_turn.py`,
`_sources.py`, `_repl.py`, `_chatbox.py`) rather than one monolithic file — see
[`CLAUDE.md`](CLAUDE.md) for the module map and design conventions if you're
extending it, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the required touch
points when adding slash commands, tools, or env vars.

## License

MIT License. See [`LICENSE`](LICENSE).
