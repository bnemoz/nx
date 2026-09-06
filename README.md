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
turns fold into a summary automatically — see `/autocompress`. The guard runs
inside long agentic tasks too, not just between user prompts, and if a server
still refuses a request with a context-length 400, nx folds the older turns
once and retries the same turn instead of reporting the task "done".

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
| `--json` | (with `--prompt-file`) emit a JSON result summary on stdout instead of streamed chat; exits `4` when the run errors (endpoint unreachable, API rejects the request) and `0` otherwise |
| `nx run -f tasks.jsonl` | headless batch mode: every JSONL line is one task — emits one JSON result line per task on stdout, one process exit code for the run (see "Batch mode" below) |
| `--tokens` / `--bench` | print the exact prompt-token floor and exit |
| `-h` / `--help` | print flag usage and exit |
| `--version` / `-v` | print the installed version and exit |

`nx sessions [query]` is a shell subcommand (not a flag): it lists/filters saved
sessions and exits without entering the REPL.

### Key environment variables

| env var | what it does |
| --- | --- |
| `NX_APPROVAL` | persistent default approval mode: `all`/`low`/`medium`/`high`/`yolo` (`/approval set-default <level>` rewrites it from the REPL) |
| `NX_ALLOWED_FILE` | the approval allowlist JSON (default `~/.nx/allowed.json`); rules are saved with `A` at an approval prompt or `/approval allow` and auto-allow matching actions before the risk classifier runs — bash actions containing shell metacharacters (`&&`, `|`, `;`, redirects, …) always fall through to the normal risk check |
| `NX_BASE_URL` / `NX_MODEL` / `NX_API_KEY` | single-source config |
| `NX_SOURCES` / `NX_SOURCE_*` | named multi-source endpoints (`NX_SOURCE_<NAME>_MODELS` pins a source's model-id dropdown list when its `GET /v1/models` is stale) |
| `NX_ACTIVE` | source to start on (persistent version of `--source`) |
| `NX_ENV_FILE` | `.env` file to auto-load (default `~/.env`) |
| `NX_THEME` | `mocha` (default) / `latte` / `dracula` / `nord` / `gruvbox` / `vaporwave` / `amethyst` |
| `NX_SPINNER` | the waiting/`running` animation (default `1`): a slow 12-cell sweep bar, at most two glyphs change per frame. `0` disables animation entirely and shows a static status line — recommended for photosensitive users. While a tool runs the line also carries live progress (`· 12s · 2/3 · compiling 3/40`) — see [Live tool progress](#live-tool-progress); under `0` that tail is suppressed, since a redraw-once line cannot show a moving clock |
| `NX_REASONING_VISIBLE` | stream live reasoning/thinking text (default `0` — collapsed; `1` streams it — toggle live with `/reasoning`) |
| `NX_THINKING` | `auto` (default) probes each source/model for a `chat_template_kwargs` thinking toggle that makes servers like glm-5.3-flash (vLLM) separate reasoning into `reasoning_content` instead of merging it into the answer; `off` keeps the server's stock behavior |
| `NX_REASONING_BUDGET` | cap on the model's reasoning/thinking phase in tokens, sent as `{"reasoning": {"max_tokens": N}}` in the request body (default `0` — off, let the model self-terminate its thinking; ignored by backends that don't consume that field; set per-source with `NX_SOURCE_<NAME>_EXTRA_BODY`; toggle live with `/budget`) |
| `NX_TOOL_OUTPUT_VISIBLE` | render each tool call's result body (default `0` — collapsed, only the call header shows; `1` renders the result body — toggle live with `/tooloutput` (`/tools` legacy alias); `todo_write` always renders). In a parallel turn the collapse hint appears once on the group footer, not once per call |
| `NX_TOOL_RESULT_CHARS` | cap on tool-result characters entering context (default 20000; `0` disables the cap). Oversized results are head/tail-kept and their full text spilled to `~/.nx/overflow/<sha256>.txt` — the result tells the model the path so it can `read_file`/`search_files` the elided middle |
| `NX_TOOL_RESULT_SPILL` | write oversized tool results to `~/.nx/overflow/` (default `1`; `0` falls back to the old blind elision marker) |
| `NX_ARG_REPAIR` | schema-level argument and tool-name repair before a tool body runs (default `1`; `0` = strict pass-through). Coerces wrong-typed args against the tool's declared schema (`"5"` to `5`, `"true"` to `true`, `42` to `"42"`), case-folds a unique enum match (`"In_Progress"` to `"in_progress"`), renames a near-miss key (`maxresults` to `max_results`; difflib similarity 0.8+ and unambiguous only) plus curated per-tool aliases similarity can't reach (`cmd` to `command`, `old_str` to `old`, `file_path` to `path`) — and repairs the tool NAME itself: one alias (`bash` to `run_bash`, `grep` to `search_files`, `str_replace_editor` to `edit_file`) or one unambiguous close match executes under the real name. A missing required argument is refused up front with the declared schema (`ERROR: read_file missing required "path" — schema: …`) instead of a TypeError from the body; values are never fabricated. Every fix is reported in the tool result as an `[arg repair: <note>]` prefix, never applied silently |
| `NX_OVERFLOW_TTL_DAYS` | purge spill files older than N days (default 7, like bg-logs) |
| `NX_OVERFLOW_ECHO_CHARS` | max chars of head/tail content echoed back alongside the spill path (default 2000; `0` = path note only) |
| `NX_HOME` / `NX_SESSIONS_DIR` | where session JSON files are stored |
| `NX_READ_FILE_LINES` | default `read_file` window size in lines (default 400; `limit<=0` reads to end) |
| `NX_READ_FILE_SCAN_LINES` | how far past a windowed `read_file` the header's exact line-count scan may reach (default 200000); beyond it the header reports a lower bound as `N+` instead of scanning an unbounded file |
| `NX_HISTORY_LINES` | how many past prompt submissions Up/Down recalls across runs (`~/.nx/input_history.json`; default 100; `0` disables) |
| `NX_ORIENTATION` | append a one-time working-directory + read-only-tools note to fresh sessions (default `1`; `0` disables) |
| `NX_DELEGATION_NUDGE_READS` | how many consecutive self-performed exploration calls (read_file/list_dir/search_files, cache hits included) with no `spawn_agent` in between before a one-time runtime note tells the model to delegate the rest of the sweep (default `8`; only fires while `NX_SUBAGENTS=1`; `0` disables) |
| `NX_GARBLE_GUARD` | scan every edit's `new` text for model-garble signatures — duplicated lines, stuttered words/phrases, line-boundary splices, mid-sentence abandoned rewrites (default `1`; `0` disables). Unambiguous garble is repaired before the write (the tool result says so); ambiguous garble is refused once, then written as-is if the model resubmits the byte-identical edit |
| `NX_PLAN_MAX_WORDS` | slug length cap for `/plan <goal>` → `.nx/<slug>.md` (default 8 words) |
| `NX_SUBAGENTS` | offer the read-only `spawn_agent` tool (default off — `1` enables; adds a schema to every turn's floor). Sub-agents get read/search tools plus a bash allowlist (grep/ls/du/ps/git log); 2+ `spawn_agent` calls in one turn run concurrently, and spawns run in the background by default (job id now, answer as a short `[Runtime note]` + report path on a later turn; `/bg` lists them; full answer in `~/.nx/agents/<job>.md`). Enabled sessions also get a one-time `[Delegation note:]` steering the model to spawn agents proactively, plus the `NX_DELEGATION_NUDGE_READS` streak nudge |
| `NX_SUBAGENT_ASYNC` | sub-agent execution default (`1` = background, the default): a spawn returns a job id immediately and its answer arrives as a head + report-path `[Runtime note]`, so the parent TUI stays quiet during a fan-out. `0` restores the old foreground default, which renders the sub-agent's search/tool activity live and returns its answer inline |
| `NX_SUBAGENT_WRITE` | opt-in write tier for sub-agents (`NX_SUBAGENT_WRITE=1`): `spawn_agent` additionally gets `write_file`/`edit_file`/`multi_edit`/`sed_edit`, each still confirmed by you through the normal approval gate (risk classification + Y/n with a diff preview) and tagged `(sub-agent)` in the tool box. Write-capable agents never run concurrently and cannot delete files or spawn other agents |
| `NX_SUBAGENT_RECOVERY_RETRIES` | rescue attempts allowed INSIDE a sub-agent loop, per failure family — reasoning-only stalls, empty turns, malformed tool-call streams/collapses, text-protocol corrections (default `1`; the parent keeps its own `NX_REASONING_ONLY_RETRIES`/`NX_EMPTY_TURN_RETRIES`/`NX_MALFORMED_STREAM_RETRIES`). Keeps a flash-tier sub-agent from burning half its `max_turns` on recovery cycles; when a sub-agent has spent its rescue allowance, the loop pre-forces the final answer one turn early so the remaining turns produce an answer, not more recovery. `0` disables the failsafes inside sub-agents entirely |
| `NX_SUBAGENT_DEFAULT_TURNS` | default `max_turns` for a spawn that doesn't specify one (default `100`; was 12, which left real investigations capping out with no answer). Also a **floor**: a parent that passes a smaller `max_turns` — several models carry their own 5/10/12-scale budget notion and were capping their agents out mid-investigation — is raised to this value, so the only way to run a genuinely short agent is to lower this knob. Also what the tool schema advertises to the parent model |
| `NX_SUBAGENT_MAX_TURNS` | hard ceiling on a spawn's `max_turns` after clamping (default `200`; was 40). Wins over the `NX_SUBAGENT_DEFAULT_TURNS` floor if the two are configured inverted |
| `NX_VISION` | offer the `read_image` tool on native-tool-call sources (default off; requires the backend to accept image content parts) |
| `NX_WAKE_PROMPT` | auto-continuation (default `1`): when a background job (a detached `run_bash` command or an async `spawn_agent` sub-agent) finishes while you idle at an empty prompt, nx starts a model turn by itself, feeding it the completion note — no keystroke needed. The note is delivered even when the turn answers without tool calls, and an already-reported job never re-wakes (0.36.3 fixed a loop where a text-only turn left the job unreported and the REPL auto-continued forever; 0.37.1 fixed a phantom banner race — a flag raised just before a user-typed turn no longer fires an extra noteless wake after that turn delivers the note itself; 0.37.4 stopped the "Nothing to do — that was the script from my last turn" wakes: a command that finishes inside its own poll window (or one whose log `wait_background`/`read_file` already returned) counts as reported, so nx never announces output the transcript already contains). Half-typed input is stashed and restored after the wake turn; `0` restores the old behavior, where the note waits for your next message |
| `NX_WAKE_POLL_SECONDS` | how often the wake poller checks for finished background jobs while the prompt idles (default `2.0`, floor `0.2`) — the worst-case delay between a job finishing and the auto-continuation turn starting |
| `NX_IMAGE_MAX_BYTES` | max image size `read_image` will load into context (default 8 MB) |
| `NX_PRE_TOOL_HOOK` / `NX_POST_TOOL_HOOK` | shell command run around every tool call with `{tool, args(, result)}` JSON on stdin; a non-zero pre-hook exit vetoes the call (NX_HOOK_TIMEOUT, default 10s, fail-closed). The pre-hook runs even on cache-hit reads; the post-hook fires on every read too — cache hits carry `"cached": true` in the payload. Prefetched reads on a mixed turn (reads + a write/edit in one tool-call batch) dispatch through the same run_tool() gate, so a veto blocks them and the post-hook sees them like any other read. Since 0.29.5 the pre-hook sees REPAIRED args: schema-level arg repair (`NX_ARG_REPAIR` — coerced types, renamed keys, resolved tool name) runs before the hook, so a policy keyed on `path` sees `path`, not the model's near-miss spelling |
| `NX_INTERRUPT_RESTORE_WAIT` | how long nx waits for the Esc-watcher to restore cooked terminal input mode after a turn/tool wait (default 2.5s — covers the 2s Esc-redirect typing window); shrink only if you never type a redirect immediately after Esc |
| `NX_BASH_MAX_SECONDS` | watchdog: reap a backgrounded `run_bash` job after N seconds (SIGTERM→SIGKILL; default 0 = off) |
| `NX_BASH_SLEEP_CAP` | ceiling on the `sleep N && cmd` auto-extended poll window (default 3600 s; a longer sleep backgrounds with a log path instead of blocking the turn; 0 = uncapped) |
| `NX_WATCHDOG_POLL_SECONDS` / `NX_WATCHDOG_KILL_GRACE` | watchdog granularity: how often the reaper checks whether the job died on its own before the budget elapses (default 0.5s), and how long it waits after SIGTERM before escalating to SIGKILL (default 5s) |
| `NX_BASH_PROMPT_SECRETS` | `1` forces the approval prompt when a `run_bash` command looks like it carries a credential (sk-/ghp_/AKIA/PEM…), even in yolo |
| `NX_PRICE_<SOURCE>_INPUT` / `_OUTPUT` / `_CACHE` | $ per 1M tokens for a source; feeds `/cost`'s $ column (unset sources show token totals only) |
| `NX_AUTOCOMPRESS_PERCENT` | auto-compress threshold (default 85; `0` disables) — checked after every settled model turn, including mid-task; a context-length 400 folds once and retries the turn |
| `NX_MEMORY_SEARCH_LIMIT` | max `memory_search` results per call (default 8; `0` unbounded) — an over-cap response says how many were cut instead of hiding them |
| `NX_MEMORY_OPTIMIZE_BATCH` | notes reviewed per librarian call by `/memory optimize` (default `0` = one pass over the whole bounded store; set a number to split it for a tiny-context model — split reviews carry the previous batch's unactioned titles so straddling near-duplicates still meet) |
| `NX_MEMORY_OPTIMIZE_MAX_NOTES` | most notes one `/memory optimize` run reviews, newest first (default 120; `0` = all) — the unreviewed remainder is reported, never silently skipped |
| `NX_MEMORY_OPTIMIZE_MAX_TOKENS` | completion budget for each `/memory optimize` model call (default 36K; `0` = don't send `max_tokens`, server default applies) — sized so reasoning models don't spend a small server cap on thinking and return no answer; an answer that *was* cut off (`finish_reason=length`) is retried once at 3× this budget |
| `NX_MEMORY_OPTIMIZE_TIMEOUT` | per-call timeout in seconds for each `/memory optimize` model call (default 300; `0` = no client-side cap) — a non-streaming reasoning call can legitimately think for minutes before its first byte |
| `NX_MEMORY_OPTIMIZE_MERGE_WORKERS` | merge round-trips `/memory optimize` runs at once (default `0` = one per merge group, capped at 4; `1` = serial) — the groups are independent calls, so concurrency makes the run's wall clock its slowest merge rather than the sum of them |
| `NX_VERIFY_CMD` | command auto-run ~20s after the last `.py` edit settles (e.g. `uv run pytest -q`); its pass/fail is injected as a `[Runtime note: …]` when it finishes. Set/clear/force with `/verify [cmd\|off\|run]` |
| `NX_VERIFY_SETTLE_SECS` | quiet period after the last edit before `NX_VERIFY_CMD` fires (default 20) |
| `NX_MAX_TOKENS` | token cap for streaming requests (default 16000; `0` omits it) |
| `NX_MAX_CONTINUATIONS` | auto-continues a reply that ended with `finish_reason="length"` (the token cap cut it mid-answer): the partial text becomes a real assistant turn, a resume note is appended, and the stream re-opens (default 2; `0` keeps the old single-turn behavior and marks the partial with a truncation marker) |
| `NX_MAX_MODEL_TURNS` | cap on tool-call turns per run (default 500) |
| `NX_TASK_TIMEOUT_SECONDS` | `nx run` per-task wall-clock cap in seconds (default 0 = off; `--timeout` or the task's `timeout_seconds` field override) — a task that exceeds it stops at the next model-turn boundary and reports `status: "error"`, `timed_out: true` |
| `NX_METRICS_URL` | optional endpoint to push cumulative token usage after each turn |
| `NX_HTTP_RETRIES` / `NX_HTTP_BACKOFF_BASE` / `NX_HTTP_TIMEOUT_CONNECT` / `NX_HTTP_TIMEOUT_READ` | transient-failure retries for every endpoint call: 429/5xx and read/write timeouts are retried up to N times beyond the first attempt (default 2; `0` off) with exponential backoff (base seconds, default 0.5 → 0.5s, 1s, 2s…) or the server's `Retry-After` when it sends one (capped at 30s). Never retried: refused connections (a cold-start fails fast) and anything after the first streamed byte. Per-request timeouts in seconds: connect 10 (refused/unreachable fails fast), read 900 (a slow server may need more — a read-timeout retry notice names this variable); `0` = no cap |
| `NX_SKIP_STARTUP_PROBES` | skip every startup probe — `resolve_model()`'s `/v1/models` + `/props` round-trip, the capability/thinking fingerprint probes, the context-window metadata probes and the over-max_tokens overrun probe (default off; `1` disables). Startup becomes instant but model autodetection is off: set the model explicitly (`NX_MODEL` / `NX_SOURCE_<NAME>_MODEL`); tool-calling falls back to reactive text-protocol parsing until a turn reveals better. Also what keeps `import nx` network-free in test runs |
| `NX_BACKEND` | `vllm` to disable llama.cpp-only recovery samplers (min_p / repeat_penalty / DRY) that vLLM rejects |
| `NX_NATIVE_TOOLS` | opt into server-side native `tool_calls` (default off — text-protocol fallback is parsed and repaired client-side) |
| `NX_MALFORMED_TOOL_BREAKER` | malformed-args circuit breaker: after N malformed/truncated `tool_calls` argument payloads naming the SAME tool (default 3), that tool is dropped from the tool list and the retry nudge steers the model to a sibling tool instead — an endpoint that keeps truncating one tool's args (observed: a local server dying right before `multi_edit`'s `[` array literal) stops being offered it. Since 0.29.9 the breaker DECAYS: one clean execution of the tool (its args parsed and the body ran without error) clears the streak and re-arms it, so two flaky truncations no longer cost the tool for the whole session; `0` disables; `/toolcalls` shows the streaks, `/toolcalls reset [tool]` clears them, `/reset` re-arms everything |
| `NX_GRAMMAR` | grammar-constrain text-protocol tool calls on llama.cpp servers (opt-in; see below) |
| `NX_SEARCH_EXCLUDE_DIRS` | space-separated dirs the `grep` fallback skips (mirrors ripgrep's `.gitignore` handling; default includes `.venv node_modules __pycache__ build dist`) |
| `NX_<ROLE>_COLOR` | override one of ten semantic roles (see below) |

`nx --help` covers flags; the source (grep `_env_int`/`_env_float` calls in
`nx/`) has the full env var list — dozens more knobs tune recovery samplers,
stall detection, undo depth, log size, etc., all with sane defaults.

When repeated reasoning-only stalls exhaust the rescue ladder and nx forces a
final answer, the forced turn is a nudge rather than a straitjacket: a
`final_answer` call still ends the turn, but if the model emits a real tool
call instead, nx runs it (through the normal approval gate) and keeps the
turn going — the counter stays pinned at the last rung so any fresh stall
escalates straight back to forced-final instead of re-climbing the ladder.

Episode awareness also shapes the mid-ladder nudge: once a real tool call has
run since the first stall, a fresh stall is read as the model stalling while
composing its wrap-up — nx asks it to deliver the visible answer
(`STALL_WRAPUP_NUDGE`) instead of demanding another tool call, and a
text-only reply that arrives after any tool work in the episode is accepted
as the final answer rather than re-nudged (the old layout-only check missed
the acts-then-stalls-then-summarizes shape and fired a spurious
"TEXT-ONLY AFTER STALL NUDGE" on legitimate finished answers).

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

`amethyst` is a monochrome purple theme: every hue is a violet/orchid/pink
shade, so the ten roles stay separable by lightness and by position in the
violet→pink band instead of by hue. `gray` (`REASONING_TEXT`) deliberately
stays a neutral gray — reasoning text is the one thing that doesn't go purple.
If you want it purple too, `NX_REASONING_TEXT_COLOR=magenta` does that.

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
| `/toolcalls [text\|native\|auto] [reset [tool]]` | bare: show the session's tool-calling mode and the malformed-args breaker streaks per tool (marked `[broken]` when dropped from the tool list); `reset` clears every streak and re-offers every tool, `reset <tool>` re-arms just that one; `text` forces the grammar-constrained text protocol for this session (the rung past the breaker — an endpoint whose NATIVE args keep truncating gets the client-repairable protocol back without switching sources), `native` re-arms the native flag, `auto` returns to the reactive default |
| `/sessions [n]` / `/resume [target]` | list or resume a past session (a non-title query falls through to full-text body search) |
| `/export [md] [path]` | write the current session to a markdown transcript |
| `/save [title]` | save the current session |
| `/delete [target]` | delete a saved session |
| `/memory save\|remember\|delete\|list` | manage persistent memory |
| `/memory optimize [dry] [focus]` | librarian pass over the memory store: one model review per `NX_MEMORY_OPTIMIZE_BATCH` notes (newest `NX_MEMORY_OPTIMIZE_MAX_NOTES` first) proposes merge groups and retirements, shown as a plan and approved once; merged notes go through the normal writer and every retired note moves to `<memories dir>/archive/<stamp>/` (nothing is unlinked). `dry` prints the plan only; a `focus` argument guides the review |
| `/compress` / `/compact` | fold older turns into a summary (keep last 2 verbatim; `/ctx`-pinned turns survive) |
| `/ctx [pin\|unpin\|clear]` | per-message token breakdown (total includes the per-turn `tools=` schema cost, shown as its own `+ tools schema` line); pin turns so every later compress keeps them verbatim |
| `/cost` | session usage totals + per-source scoreboard ($ when `NX_PRICE_<SOURCE>_*` is set) |
| `/autocompress [pct\|off\|on]` | show or set the auto-compress threshold; every fold auto-pins the latest `todo_write` turn so the live task plan survives compression |
| `/budget [<tokens\|off>]` | cap the model's reasoning phase in tokens (rides as `{"reasoning": {"max_tokens": N}}`; `off` clears; session-only unless `NX_REASONING_BUDGET` is set) |
| `/verify [cmd\|off\|run]` | show/set/clear or force-run the post-edit verify command (`NX_VERIFY_CMD`) — when set, every edited `.py` also gets an in-process `ast.parse` whose verdict rides on the edit's own tool result |
| `/undo` | revert the last write/edit (LIFO) |
| `/recover [note]` | force a low-temp visible checkpoint after a bad stream |
| `/bg [kill <pid>]` | list background `run_bash` jobs, or kill one |
| `/resources` | show this nx process's CPU% and RSS memory |
| `/init` | generate/update an `AGENTS.md` for this project |
| `/todo` | show the model's current `todo_write` list |
| `/plan [goal\|done\|off\|discard]` | plan mode: `/plan <goal>` writes `.nx/<slug>.md` and gates every mutating tool call (`write_file`, `edit_file`, `multi_edit`, `sed_edit`, `apply_patch`, `delete_file`, `run_bash`, `memory_write/delete`) behind a gate message until you run `/plan done`; the model refines the plan with `plan_edit` while gated; `discard` also deletes the file |
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
A saved rule never auto-allows a bash command containing shell control
characters (`&&`, `;`, `|`, redirection, `$(…)`, globs, …) — such chains fall
through to the normal risk check and prompt even when the prefix matches, so
a blessed `run: pytest*` cannot be stretched into `run: pytest && <anything>`.

Read-only tool calls in an all-read-only turn (e.g. three `read_file` calls
plus a `search_files`) execute concurrently; anything mutating or
approval-gated keeps strict sequential dispatch. A concurrent turn renders as
ONE group box drawn by the terminal-owning thread after the calls finish — one
collapse hint and one elapsed time for the group, calls in the order the model
emitted them, `· cached` on results served from the read cache, and a failed
call showing its reason inline — instead of one half-box per thread with
spinners overwriting each other's lines. 2+ `spawn_agent` calls fan out the
same way, and an agent's own tool activity never reaches the parent's terminal
— the group line is the whole story while they run.

## Live tool progress

A long tool call used to look exactly like a hung one: after the cyan args line
the terminal showed a static `running run_bash [▪▪▪▪·······]` for the whole
call, with no clock and no way for the tool to say what it was doing. The
activity line now reports progress while the call is in flight:

```
running run_bash · 12s · pytest tests/ -q 4 passed [▪▪▪▪········]
running 3 tool calls · 2/3 · 4.2s [▪▪▪▪········]
running spawn_agent · 1m48s · agent turn 7/100 [▪▪▪▪········]
```

- **elapsed** — seconds since the call started, on every tool line and on the
  parallel group line. It keeps counting across an approval prompt rather than
  resetting, so a slow risk-classifier round trip reads as the wait it is.
- **`n/m`** — how many of a concurrent group's calls have actually landed.
  It tracks completions, not submissions, so a slow first call no longer has
  the line sitting at `0/3` while two calls are already done.
- **a log tail** — `run_bash` and `wait_background` stream the command's last
  log line into the line (throttled to ~1/s), which is also the only live view
  of a long command, since its output is buffered to a log file until it exits.
- **a tool's own note** — a long call with no other feedback channel can report
  through `nx._progress.note_progress("compiling 3/40")`. A foreground
  `spawn_agent` uses it to show its turn count; `run_bash` needs it not at all.
  Notes are display-only — never sent to the model, never in the transcript.

All of it is one row that never wraps (the label gives up cells to the progress
tail), is suppressed entirely when stdout is not a tty, and is dropped under
`NX_SPINNER=0`, where drawing a clock once and freezing it would misreport the
call.

## Sessions

Every chat auto-saves to `~/.nx/sessions/` as one human-readable JSON file per
session (message array + id/title/description/source/timestamps + cumulative
token usage). Auto-save happens after every turn, so a crash never loses work.

`nx --resume <target>` (short id, prefix, title, or bare for most recent)
restores it, reselects the original source, and replays a recap of the
conversation. `nx sessions [query]` lists/filters sessions from the shell
without entering the REPL; inside the REPL, `/sessions` and `/resume` do the
same.

## Batch mode

`nx run -f tasks.jsonl` is the headless CI/benchmark harness: every line of
the file is one task, each runs as a full one-shot agent run (fresh session,
fresh token counters, its own resumable transcript under `~/.nx/sessions/`),
and one JSON result line lands on stdout per task the moment it finishes.

```
$ nx run -f tasks.jsonl --parallel 4 --yolo
```

Each input line is a bare JSON string (the instruction) or an object:

```
"refactor the retry loop in _openai_client.py"
{"instruction": "summarize README.md", "id": "t1", "session": "bench-07"}
```

Blank lines and `#` comments are skipped. Flags: `--parallel <N>` runs up to N
tasks concurrently (each on its own thread with per-task state — including
per-task token accounting, so a benchmark's per-task token totals stay exact
under `--parallel > 1`), `--max-turns <N>` caps each task's turns,
`--timeout <seconds>` caps each task's wall-clock time (default
`NX_TASK_TIMEOUT_SECONDS`, 0 = off: the task is stopped at the next
model-turn boundary, its result line gets status `error`, an `error`
naming the timeout, and `timed_out: true` — a task can also set its own
`timeout_seconds` field, which wins over the flag),
`--continue-on-error` keeps scheduling after a
task fails (default: stop scheduling new tasks at the first error — a
timed-out task counts as an error),
`--source <name>` picks the endpoint. stdout carries ONLY the JSONL result
lines — every human-facing render is suppressed; progress and the summary go
to stderr.

Per-task exit codes ride each result line (`exit_code` field): `0` done, `3`
turn cap, `4` error/refused. An unreachable endpoint or a request the server
rejects outright (bad key, context-length overrun, 4xx/5xx after fallbacks) is
an `error` task with `error` naming the failure — never a `done` line with
zeroed tokens. The process exit code summarizes the run: `0`
every task done, `3` any cap, `4` any error, `2` harness misuse (unreadable
file, zero tasks, no source, unknown flag). A streaming consumer can score
each task as its line arrives without waiting for the process to end.

Approval: batch is headless — nobody can answer a Y/n prompt. Without
`--yolo`, every gated call is auto-denied and the task reports that; run with
`--yolo` (or `--approval high`) when tasks must write files and run commands.
Each task's transcript is saved like an interactive session's, so a scored run
is also a resumable one.

Result-line shape (one JSON object per task):

```
{"index": 0, "id": "task-0", "line": 1, "session": "20260831-125110-08f950",
 "batch": "20260831-125110", "exit_code": 0, "status": "done", "turns": 3,
 "elapsed_seconds": 12.4, "final_answer": "…", "tokens": {"input": 2630,
 "output": 412, "cache_read": 1980, "reasoning": 0, "api_calls": 4},
 "tool_calls": [{"tool": "read_file", "args": "…", "result": "…"}],
 "error": null}
```

Tasks compose with sub-agents: a task's model may call `spawn_agent`
(read-only by default, write tier with `NX_SUBAGENT_WRITE=1`) — spawns run in
the background by default, so the finished agent's answer arrives in that
task's transcript as a `[Runtime note]`. With `--parallel > 1` the
async-note channel is process-global, so prefer `--parallel 1` when a task
relies on background agents.
`NX_VERIFY_CMD` and the interactive background-completion scan stay off for
batch workers (they're parent-loop machinery).

## Tools

Read-only tools never need approval; state-touching tools (write/edit/bash)
go through the approval gate above.

| tool | args | notes |
| --- | --- | --- |
| `read_file` | `path`, `offset?`, `limit?` | cat -n style (1-based); `path` may be a list; streams — only the window is held in memory |
| `write_file` | `path`, `content` | overwrites; requires confirmation; atomic (temp+rename), preserves the file's permission bits and writes through symlinks to the real file |
| `edit_file` | `path`, `old`, `new`, `replace_all?`, `old_lines?` | `old` must match exactly once (or `replace_all`); `old_lines="10-14"` replaces a 1-based line range instead; unique indentation-only mismatches apply whitespace-tolerantly; `new` passes the garble guard (see `NX_GARBLE_GUARD`) |
| `multi_edit` | `path`, `edits` | batch of `{old, new}`, one confirmation; atomic — a garbled or unmatched edit rolls back the whole batch; each `new` passes the garble guard |
| `delete_file` | `path` | delete one file; restorable via `/undo` (unlike `rm` via bash) |
| `sed_edit` | `pattern`, `replacement`, `path` | regex find/replace over a file or glob |
| `apply_patch` | `diff`, `path?` | apply a unified diff; for multi-file restructure |
| `list_dir` | `path`, `pattern?` | flat list, or recursive glob (`**/*.py`) |
| `search_files` | `pattern`, `path?`, `glob?`, … | regex content search; read-only |
| `git_status` | `path?` | `git status --short -b`; read-only |
| `git_diff` | `path?`, `staged?`, `file?` | `git diff`; read-only |
| `git_log` | `path?`, `max_count?`, `file?` | `git log --oneline`; read-only |
| `run_bash` | `command`, `timeout?` | requires confirmation; backgrounds long-running commands |
| `wait_background` | `pid`, `timeout?` | only offered after a `run_bash` call has backgrounded; waits for the PID and returns its output (a caller-supplied `log_path` is ignored — the path is resolved from the PID) |
| `todo_write` | `todos` | rewrites whole list `[{text, status}]`; no per-item id |
| `memory_write` | `title`, `body`, `scope?` | save a durable memory note; requires confirmation |
| `memory_search` | `query`, `scope?` | keyword-search saved memories; title hits rank first; read-only; ends with the absolute dirs holding the shown hits, because the listed `(name)` is a bare filename — `read_file` needs the full path |
| `memory_delete` | `target`, `scope?` | delete one memory by exact title, filename, or unique prefix; requires confirmation |
| `spawn_agent` | `instruction`, `max_turns?`, `async?` | delegate an investigation or task to a sub-agent (`NX_SUBAGENTS=1`); sub-agents can run an allowlisted read-only bash tier (grep/ls/du/ps/git log). The agent sees its turn budget (default and floor 100 — a parent's smaller `max_turns` is raised to it — ceiling 200) and spends its last turn answering, not searching; rescues inside the agent are capped at one per failure family, and a budget mostly spent on rescues gets its final answer forced early; an agent that dies at the cap still returns its last observation. Write-capable only with `NX_SUBAGENT_WRITE=1` (write_file/edit_file/multi_edit/sed_edit, each confirmed by the user with a diff preview, tagged `(sub-agent)`; run sequentially, no delete/no nested spawn). Send several read-only ones in one turn to run them concurrently — one question per agent; re-spawning an instruction whose background job is still running is refused. Background is the default (`NX_SUBAGENT_ASYNC=0` for foreground): a background spawn returns a job id immediately and delivers a short answer head + the path of its full markdown report (`~/.nx/agents/<job>.md`) as a one-time `[Runtime note]` on a later turn — the parent reads the file when the head isn't enough. Foreground returns the answer inline; while foreground spawns are in flight the turn shows a single group line (`2 tool calls in parallel · spawn_agent ×2`) rather than each agent's own boxes and spinners (0.31.0). Async/background agents are always read-only, and a failed agent still announces (FAILED note). `/bg` lists running/finished agents with their report paths. Flash-tier models sometimes narrate delegation instead of emitting the call, or mis-key arguments (`prompt` instead of `instruction`) — both self-correct on retry, but heavy delegation wants a non-flash model |
| `read_image` | `path` | show a local image to a vision-capable model (`NX_VISION=1` + native tool calls) |

`read_file`/`list_dir` results are cached per-path and invalidated on writes
or any `run_bash`. The `git_status`/`git_diff`/`git_log` cache entries carry a
fingerprint of `.git` state (HEAD/index/`ORIG_HEAD` mtimes) and are dropped by
any edit tool or `run_bash` too — a commit in another terminal, or `git reset`
via `run_bash`, is never served stale. `wait_background` is omitted from the
tool schemas sent to
the model until a `run_bash` has actually backgrounded, so a session where
nothing ever backgrounds doesn't pay tokens for a tool it will never call.

### Garble guard

Small/quantized models sometimes emit textually corrupted replacement text —
a line decoded twice, a phrase that restarts mid-stream, a line boundary
written twice, or a half-abandoned rewrite left behind an ellipsis. Every
`edit_file`/`multi_edit` `new` payload is scanned in-process (`nx/_garble.py`)
before it can land:

- **unambiguous** duplicates (adjacent duplicated lines, period-2 word
  stutters, same-line repeated phrases, line-boundary splices where the shared
  words appear nowhere in the file) are repaired automatically — the diff you
  approve already shows the repaired text, and the tool result says what was
  fixed;
- **ambiguous** corruption (e.g. a mid-sentence abandoned rewrite) is refused
  once with the suspect text quoted; resubmitting the byte-identical edit
  writes it as-is with a warning, so a false positive can never deadlock the
  model;
- repetition the file itself already contains (repeated decorators,
  boilerplate) is left alone, and swapped `old`/`new` arguments get a
  targeted resubmit hint;
- repetition that is ordinary code shape is not flagged at all: a period-2
  echo only counts when BOTH copies sit whole on one line (so an attribute
  saved then reassigned, or `return d` on one line and another `return`
  starting the next, are untouched), a suspected line
  splice is ignored when trimming the shared words would leave an operator
  behind (line 2 was continuing line 1's expression, not echoing it), and
  loop/guard bodies that legitimately repeat (`if not quiet:` + print twice,
  two identical `for` loops) are spared — a repeated phrase counts only when
  its copies sit on one line, and a duplicated line only when nothing but a
  blank line separates the copies.

- when a repair would collapse `new` back to exactly `old` (the only
  "corruption" was a stutter adjacent to the replaced text), the edit is
  refused with a targeted hint instead of being written as a silent no-op;
  resubmitting the byte-identical edit writes `new` as-is.

The guard is passive — no extra tool schema, no tokens per turn — and
`NX_GARBLE_GUARD=0` turns it off.

## Memory

The model can save durable memory notes itself — no manual `/memory save`
needed. Three tools drive it: `memory_search` (recall prior learnings before
re-deriving them), `memory_write` (persist a new one), and `memory_delete`
(retire a superseded one). At session start a short note listing
already-saved titles is appended to context so the model knows what's
remembered. A resumed session's note (`--resume` and `/resume`) is rebuilt
from the live store when the store changed since that transcript was saved —
notes that `/memory optimize` has since archived or merged don't linger as
phantom titles for the model to chase. Searches require every query term to
match, rank title hits above body
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
convention) and hand-managing notes, and `/memory optimize` cleans the store
up once it has outgrown itself.

### `/memory optimize`

A store that only grows turns into noise: after a hundred notes some pair
almost certainly describes the same subsystem from two different weeks, and
older notes describe things the matured project no longer has.
`/memory optimize` is the janitor pass — deliberately a slash command and not
a tool, because a tool schema costs tokens on every turn in every project
forever.

It indexes the store with no model call, then sends a compact index (filename,
age, size, title, one-line digest) to the active model in ONE librarian call
over the whole bounded store (`NX_MEMORY_OPTIMIZE_BATCH` can split it into
several for a tiny-context model) and asks for plan lines only:
`RETIRE <name>`, `MERGE a + b => Title`, or `OK`. A line protocol rather than
JSON, because note bodies are markdown and small models quote that badly; an
unresolvable or malformed line is dropped and shown instead of losing the
whole pass. When the review is split, the previous batch's unactioned titles
ride along, so a near-duplicate pair straddling a batch boundary can still
meet. Each merge group then gets its own call carrying the full bodies (with
the sources' `_scope:`/`_saved:` footer stripped first — the writer re-adds
it) and returns one note, written through the same shared writer as every
other memory, so merged files are conventional: timestamped slug, `0600`,
`_scope:` footer, never clobbering. A merge that spans both scopes lands in
whichever scope held most of its sources (ties go to project). A run applies
at most eight merges (each is a model call); when the librarian finds more
groups than that, the run says so and a second `/memory optimize` picks them
up.

Nothing is unlinked. Retired notes move to
`<memories dir>/archive/<run-stamp>/` inside their own scope, which no scanner
can see (a directory name has no `.md` suffix), so a bad review is reversible
by moving the files back. The plan is printed in full before anything happens
and goes through the approval gate once, so `/approval` rules and yolo apply
as they do elsewhere; `/memory optimize dry` prints the plan and writes
nothing, and a trailing argument guides the review ("focus on the build
notes"). Reviews run newest first over at most `NX_MEMORY_OPTIMIZE_MAX_NOTES`
notes (default 120), so in a bigger store the oldest notes — the strongest
retirement candidates — are never seen; run `/memory optimize` with the
cap set to `0` (all notes) periodically to reach them. Each round-trip sends
`max_tokens` (`NX_MEMORY_OPTIMIZE_MAX_TOKENS`, default 36K) so a reasoning
model can't burn a small server-side cap on thinking and return an empty
answer, and allows `NX_MEMORY_OPTIMIZE_TIMEOUT` seconds per call (default
300; the pre-reasoning fixed 60s could kill a call mid-think — a read
timeout also burns one `NX_HTTP_RETRIES` attempt, so retrying couldn't save
it); if a call still comes back with no visible text the run says why, and a
`finish_reason=length` cut-off — the budget, not the model, ended the call —
is retried once at three times the budget before the run gives up. Merge groups are
independent model calls, so they run concurrently —
`NX_MEMORY_OPTIMIZE_MERGE_WORKERS` at a time (default one per group, capped
at 4; `1` restores serial) — which matters on a reasoning source, where the
run's wall clock is otherwise the sum of several minutes-long merges. After
a real run the cached `memory_search` entries are dropped and
the session's own saved-memories note is rebuilt in place, so the model's view
of the store matches the store — including in a session that started before
the cleanup. Session resume (`--resume` and `/resume`) runs the same
rebuild, so a transcript written before an optimize run in another session
no longer carries its retired titles either.

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
magnitude only. Below ~0.02s the span is SSE arrival quantization rather
than pacing at all (a bursty server can deliver a whole short generation
in one or two reads a fraction of a millisecond apart), so the footer
divides by full turn wall-clock instead — the end-to-end average, a true
lower bound — rather than printing a five-digit artifact.

## Architecture

`nx/` is a small package (`__init__.py` + focused submodules like `_turn.py`,
`_sources.py`, `_repl.py`, `_chatbox.py`) rather than one monolithic file — see
[`CLAUDE.md`](CLAUDE.md) for the module map and design conventions if you're
extending it, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the required touch
points when adding slash commands, tools, or env vars.

## License

MIT License. See [`LICENSE`](LICENSE).
