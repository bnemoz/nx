# nx (pronounced "annex")

A coding agent that doesn't spend 50K tokens of context before you've said
anything. Point it at any OpenAI-compatible endpoint — a local llama.cpp /
vLLM / SGLang server, or a remote API like OpenRouter, and get an agent that 
reads, writes, edits, and runs shell commands in your project.

## Why context-frugal

Most agent harnesses burn 20K-50K+ tokens of system prompt and tool schemas
before your first message even lands. That's context you can't use for your
own code, and it's paid on *every single turn* — worse on local models, where
you're already tight on context and speed.

On a bare `hey`, nx's entire prompt is ~1,400 tokens (system prompt + 7 tool
schemas + framing). Run `nx --tokens` to print the exact, mechanically-counted
floor for your active backend/model — not eyeballed.

It's also built to survive rough edges of self-hosted models: falls back to
text-parsed tool calls when a server doesn't support native tool-calling, 
and recovers from malformed streams and reasoning-only stalls automatically.

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

### Add endpoint

`nx` reads the credentials from ~/.env

```bash
export NX_BASE_URL=http://localhost:8080/v1
export NX_MODEL=your-model-name
export NX_API_KEY=sk-noop            # any string; local servers ignore it
python -c "import nx; nx.main()"
```

If `NX_MODEL` is unset, nx asks the server what it's serving.

## Configuration

nx reads config from environment variables and auto-loads `~/.env` at
startup — no config file format.

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

### Flags

| flag                          | what it does                                              |
| ----------------------------- | -------------------------------------------------------- |
| `--yolo`                      | start in never-prompt mode (auto-approve everything)      |
| `--approval <all\|low\|medium\|high\|yolo>` | start with a non-default approval mode       |
| `--source <name>`             | start on a specific source                                |
| `--resume [target]`           | resume a saved session; bare = most recent                |
| `--session <id>`              | start a fresh run attached to a specific session id       |
| `--tokens` / `--bench`        | print the exact prompt-token floor and exit                |
| `-h` / `--help`               | print flag usage and exit                                  |
| `--version`                   | print the installed version and exit                        |

### Key environment variables

| env var | what it does |
| --- | --- |
| `NX_APPROVAL` | persistent default approval mode: `all`/`low`/`medium`/`high`/`yolo` |
| `NX_BASE_URL` / `NX_MODEL` / `NX_API_KEY` | single-source config |
| `NX_SOURCES` / `NX_SOURCE_*` | named multi-source endpoints |
| `NX_ACTIVE` | source to start on (persistent version of `--source`) |
| `NX_THEME` | `mocha` (default) / `latte` / `dracula` / `nord` / `gruvbox` / `vaporwave` |
| `NX_HOME` / `NX_SESSIONS_DIR` | where session JSON files are stored |
| `NX_AUTOCOMPRESS_PERCENT` | auto-compress threshold (default 85; `0` disables) |
| `NX_MAX_TOKENS` | token cap for streaming requests (default 16000; `0` omits it) |
| `NX_METRICS_URL` | optional endpoint to push cumulative token usage after each turn |
| `NX_GRAMMAR` | grammar-constrain text-protocol tool calls on llama.cpp servers (opt-in; see below) |
| `NX_<ROLE>_COLOR` | override one of nine semantic roles (see below) |

`nx --help` covers flags; the source (grep `_env_int`/`_env_float` calls in
`nx/`) has the full env var list — dozens more knobs tune recovery samplers,
stall detection, undo depth, etc., all with sane defaults.

### Themes

Each theme supplies eight palette hues — `cyan`, `green`, `yellow`, `red`,
`magenta`, `gray`, `blue`, and `orange` — that nine semantic roles draw from:

| role | default hue | what it colors |
| --- | --- | --- |
| `TEXT` | green | the model's final answer text |
| `REASONING_TEXT` | gray | reasoning/thinking block text |
| `STATUS_BAR` | cyan | per-turn stats footer (tok/s, ctx, ttft, wall) |
| `TOOL_OUTPUT` | magenta | tool/bash result body lines |
| `ACCENT` | blue | borders, model name, chatbox prompt |
| `LABEL` | orange | source name / session id badges |
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

Servers without native tool-calling fall back to a `<tool_call>{...}
</tool_call>` text protocol, parsed out of the assistant's reply — small
models occasionally emit truncated or invalid JSON in that tag, which the
parser tries to recover from after the fact. Setting `NX_GRAMMAR=1` against a
llama.cpp server instead constrains generation with a GBNF grammar (derived
from the active tool schemas) once the model emits the `<tool_call>` trigger,
so malformed JSON becomes unsamplable rather than something to clean up.
It's opt-in and llama.cpp-only — the `grammar`/`grammar_triggers` request
fields are a llama.cpp server extension, not part of the OpenAI API, and
their exact shape has moved across llama.cpp versions; an unsupported server
just has the field ignored or rejected (falling back automatically, same as
an `NX_SOURCE_*_EXTRA_BODY` a server doesn't recognize).

## Commands

Run `/help` in the REPL for the full list. The essentials:

| command             | what it does                                            |
| ------------------- | ------------------------------------------------------ |
| `/source [name]`    | list or switch sources (context preserved)              |
| `/yolo`              | toggle auto-approve for writes and bash                 |
| `/approval [level]` | show or set risk threshold                              |
| `/sessions [n]` / `/resume [target]` | list or resume a past session           |
| `/save [title]`     | save the current session                                |
| `/compress` / `/autocompress` | fold older turns into a summary, manually or automatically |
| `/undo`             | revert the last write/edit (LIFO)                       |
| `/recover [note]`   | force a checkpoint after a bad stream                    |
| `/reset`            | start a fresh session                                    |
| `/todo`             | show the model's current `todo_write` list               |

## Approval modes

Every write/edit/bash call is risk-classified (`low`/`medium`/`high`) by a
cheap model call before it runs:

| setting                 | prompts at          | auto-allows       |
| ----------------------- | ------------------- | ----------------- |
| _(default)_ / `all`     | low + medium + high | —                 |
| `--approval low`        | medium + high       | low               |
| `--approval medium`     | high                | low + medium      |
| `--approval high`       | —                   | low + medium + high |
| `--yolo` / `yolo`       | —                   | everything; skips classifier |

The prompt shows the risk reason (`[risk: HIGH — recursive force delete in
/tmp]`) so you have context for the decision. **Y**/Enter approves, **n**
denies, **Esc** stops the turn and drops back to the prompt.

## Sessions

Every chat auto-saves to `~/.nx/sessions/` as one human-readable JSON file per
session (message array + id/title/description/source/timestamps). Auto-save
happens after every turn, so a crash never loses work. `nx --resume <target>`
(short id, prefix, title, or bare for most recent) restores it, reselects the
original source, and replays a recap of the conversation. `nx sessions
[query]` lists/filters sessions from the shell without entering the REPL.

## Tools

| tool           | args                  | notes                           |
| -------------- | --------------------- | ------------------------------- |
| `read_file`    | `path`                |                                 |
| `write_file`   | `path`, `content`     | overwrites; requires confirmation |
| `edit_file`    | `path`, `old`, `new`  | `old` must match exactly once   |
| `list_dir`     | `path`                |                                 |
| `search_files` | `pattern`, `path?`, `glob?`, `max_results?` | regex content search; read-only |
| `multi_edit`   | `path`, `edits`       | batch of `{old, new}`, one confirmation |
| `run_bash`     | `command`             | requires confirmation; backgrounds long-running commands |
| `wait_background` | `pid`, `log_path`, `timeout` | only offered after a `run_bash` call has backgrounded |
| `todo_write`   | `todos`               | rewrites whole list `[{text, status}]`; no per-item id |

## Metrics

nx logs token usage (input/output/cache-read/reasoning, per the OpenAI usage
convention) into each session's JSON, always on. Set `NX_METRICS_URL` to also
POST cumulative totals to a dashboard after each turn (fire-and-forget,
disables itself on first failure — nothing leaves the machine unless you set
this).

## Architecture

`nx/` is a small package (`__init__.py` + focused submodules like `_turn.py`,
`_sources.py`, `_repl.py`) rather than one monolithic file — see
[`CLAUDE.md`](CLAUDE.md) for the module map and design conventions if you're
extending it.

## License

MIT License. See [`LICENSE`](LICENSE).
