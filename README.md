# nx (pronounced "annex")

A coding agent that keeps its per-turn floor under ~5K tokens, so a
128K-1M-token context window stays mostly yours. Point it at any
OpenAI-compatible endpoint — a local llama.cpp / vLLM / SGLang server, or a
remote API like OpenRouter, Together, or Z.ai — and get an agent that reads,
writes, edits, and runs shell commands in your project.

### Origins

nx started (v0.1.0) as a fork-inspired experiment based on
[Sentdex/minion](https://github.com/Sentdex/minion) — at the time a single
~5.5K-line monolithic Python file. Since then nx has been rewritten
end-to-end: it is now a package of small focused modules (`nx/_turn.py`,
`nx/_dispatch.py`, ...) with a hard token budget, risk-classified approvals,
multi-source routing, and a test suite. Little of the original code survives;
minion deserves credit as the spark, not the blueprint.

## Why a bounded floor

Most agent harnesses burn 20K-50K+ tokens of system prompt and tool schemas
before your first message even lands — a large slice of a modern context
window spent before you've typed anything, and paid on *every turn*.

nx holds the floor under a **hard, recorded budget of ~5K tokens** (5320): a
`SYSTEM` prompt plus the always-on tool schemas, and nothing else. That is
roomy enough for genuinely descriptive schemas — every argument the model can
pass is documented, tools name each other, and self-healing behavior is
spelled out — while staying trivial against the 128K-1M windows today's
models ship with. The budget is not a vague aspiration: it is an enforced
number, and the machinery that keeps it honest is what lets nx stay bounded
*without* forcing every schema down to a cryptic minimum.

Run `nx --tokens` to print the exact, mechanically-counted floor for your
active backend/model — system prompt + active tool schemas + a one-token
"hey" — instead of an eyeballed number that drifts as tools are added or
removed.

It's also built to survive the rough edges of self-hosted models: accepts
native `tool_calls` by default (`tool_choice="auto"`) and falls back to
text-parsed tool-call tags when a server or model doesn't use them, recovers
from malformed streams and reasoning-only stalls automatically, and probes
each source's tool-calling capability at startup so the right protocol is
picked without hand-tuning.

Every write/edit/bash call is risk-classified and gated by an approval prompt
(unless you pass `--yolo`); every session auto-saves so you can `--resume` it
later. `/help` lists all slash commands.

## Quick start

Install it as a `nx` command on your `$PATH`:

### One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/nmishra93/nx/main/install.sh | sh
```

This bootstraps `uv` if missing (to `~/.local/bin`, no root needed), installs
`nx` as a `uv tool` from this repo, and prints how to upgrade later
(`uv tool upgrade nx`).

On Windows, use the install-from-source form below from PowerShell (nx runs
in a regular terminal; Windows Terminal or any VT-enabled console works).

### Install from source

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

**First run does it for you**: launch `nx` with no endpoint configured and a
short wizard asks you to paste an API base URL and key, then saves them to
`~/.env` — next launch just works. It also offers web-search setup: point it
at an existing SearXNG instance, let it **install a fresh loopback-only one
for you** (Linux or macOS; automates the [docs/SEARXNG_SETUP.md](docs/SEARXNG_SETUP.md) steps — a systemd
user unit on Linux, a LaunchAgent on macOS), or skip.
Re-run any time with `nx setup`.

Prefer to configure by hand? `nx` reads credentials from `~/.env`
(auto-loaded at startup — no config file):

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

`nx config` is a viewer/doctor over that same layer (added 0.52.0, P3-9) —
`nx config list [filter]` shows every var in effect (secrets masked),
`nx config get NAME` prints one value, `nx config set NAME VALUE` persists to
the env file and applies immediately (the only writer, same one as
`/theme`), `nx config path` prints the file being read, and
`nx config doctor` flags contradictions: unreadable env file, unparseable or
double-quoted keys, hyphenated source names whose `NX_SOURCE_*` vars no
shell can set, `NX_VISION` without native tool calls, legacy `NX_BASE_URL`/
`NX_MODEL` silently shadowed by `NX_SOURCE_*` sources. It never touches the
network. `/status` shows the session's view of the same state.

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
via `GET /v1/models`. Every dropdown (slash commands, command arguments,
models, `@file` mentions) uses the same style: long lists scroll instead of
truncating (↑/↓, PageUp/PageDown, `···` marks hidden rows above/below).
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
prefix. `/context` shows the same fraction per turn in its cache column. At
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
| `--resume [target]` / `-r [target]` | resume a saved session; bare = most recent |
| `--session <id>` | start a fresh run attached to a specific session id |
| `--isolate` | run this session in its own git worktree (alias of `NX_SESSION_WORKTREE=1`; see the env table) |
| `--prompt <text>` / `-p <text>` | one-shot with an inline prompt: run one turn on `<text>` and exit |
| `--prompt-file <path>` / `-f <path>` | headless one-shot mode: run one turn and exit (`-` reads stdin) |
| `--json` | (with `--prompt`/`--prompt-file`) emit a JSON result summary on stdout instead of streamed chat; exits `4` when the run errors (endpoint unreachable, API rejects the request) and `0` otherwise |
| `nx run -f tasks.jsonl` | headless batch mode: every JSONL line is one task — emits one JSON result line per task on stdout, one process exit code for the run (see "Batch mode" below) |
| `nx eval list` / `nx eval run` | offline eval corpus: replays recorded stub-server scenarios through the real batch harness and scores each against its expectation (see "Eval corpus" below) |
| `--tokens` / `--bench` | print the exact prompt-token floor and exit |
| `-h` / `--help` | print flag usage and exit |
| `--version` / `-v` | print the installed version and exit |

Every subcommand (`run`, `sessions`, `config`, `eval`, `setup`, `telegram`, `update`) accepts `-h/--help`
and prints its own usage, flags, and exit codes: `nx run --help`,
`nx sessions --help`, `nx config --help`, `nx eval --help`.

`nx sessions [query]` is a shell subcommand (not a flag): it lists/filters saved
sessions and exits without entering the REPL. `nx sessions --migrate` folds any
legacy per-session JSON files into the store by hand (idempotent; the files are
kept as a backup). `nx sessions --prune` reconciles the store by hand: drops
search rows whose transcript row vanished, collects unreferenced image payloads
immediately (no age grace), and enforces the `NX_IMAGES_QUOTA_MB` cap — one
line per action.

### Key environment variables

| env var | what it does |
| --- | --- |
| `NX_APPROVAL` | persistent default approval mode: `all`/`low`/`medium`/`high`/`yolo` (`/approval set-default <level>` rewrites it from the REPL) |
| `NX_ALLOWED_FILE` | the approval allowlist JSON (default `~/.nx/allowed.json`); rules are saved with `A` at an approval prompt or `/approval allow` and auto-allow matching actions before the risk classifier runs — bash actions containing shell metacharacters (`&&`, `|`, `;`, redirects, …) always fall through to the normal risk check |
| `NX_BASE_URL` / `NX_MODEL` / `NX_API_KEY` | single-source config |
| `NX_SOURCES` / `NX_SOURCE_*` | named multi-source endpoints (`NX_SOURCE_<NAME>_MODELS` pins a source's model-id dropdown list when its `GET /v1/models` is stale) |
| `NX_ACTIVE` | source to start on (persistent version of `--source`) |
| `NX_ENV_FILE` | `.env` file to auto-load (default `~/.env`) |
| `NX_APPROVAL_POLICY` | `headless-allow:<glob>[,<glob>…]` pre-approves tool names on headless surfaces — Telegram, background sub-agents, `nx run` batch; match is on the tool name only, unmatched tools still auto-deny; unset/other values keep the auto-deny, and the policy never applies interactively |
| `NX_SESSION_IMAGES` | keep `read_image` payloads out of session JSON (default `1`): the base64 lives content-addressed in `<sessions>/_images/<sha256>.b64` — deduped across sessions, written once, garbage-collected to the images still referenced by any session — and messages carry a small `image_ref` (hash, mime, size, original path) that resume expands back into the live transcript. A missing/corrupt sidecar degrades that image to a text note instead of a 400. `0` restores 0.38.x inline-base64 storage for NEW saves; already-stored payloads are still garbage-collected (since 0.45 they were orphaned forever under `0`) |
| `NX_IMAGES_QUOTA_MB` | size cap on the `_images/` payload dir (default `200` MB, `0` disables): once over quota, the oldest unreferenced payloads are evicted first, then — only if still over — the oldest referenced ones (their resume degrades to the missing-sidecar note). Enforced after every indexed save and by `nx sessions --prune` |
| `NX_SESSION_TEXT_REFS` | keep big tool-output text out of the session store (default `1`): any single text payload over `NX_SESSION_TEXT_REF_MIN` lives content-addressed in `<sessions>/_textrefs/<sha256>.txt` and the event carries a 512-char preview reference that resume expands back into the live transcript (a missing sidecar degrades to the preview, never a 400). Garbage-collected to the payloads still referenced by any session, same as images. `0` stores all text inline |
| `NX_SESSION_TEXT_REF_MIN` | the per-payload size threshold for `NX_SESSION_TEXT_REFS`, in bytes (default `65536`); `0` disables offloading |
| `NX_THEME` | `mocha` (default) / `latte` / `frappe` / `dracula` / `nord` / `gruvbox` / `vaporwave` / `amethyst` |
| `NX_SPINNER` | the waiting/`running` animation (default `1`): a slow 12-cell sweep bar, at most two glyphs change per frame. `0` disables animation entirely and shows a static status line — recommended for photosensitive users. While a tool runs the line also carries live progress (`· 12s · 2/3 · compiling 3/40`) — see [Live tool progress](#live-tool-progress); under `0` that tail is suppressed, since a redraw-once line cannot show a moving clock |
| `NX_REASONING_VISIBLE` | stream live reasoning/thinking text (default `0` — collapsed; `1` streams it — toggle live with `/reasoning`) |
| `NX_THINKING` | `auto` (default) probes each source/model for a `chat_template_kwargs` thinking toggle that makes servers like glm-5.3-flash (vLLM) separate reasoning into `reasoning_content` instead of merging it into the answer; `off` keeps the server's stock behavior. The probe's response budget is `NX_THINKING_PROBE_MAX_TOKENS` (default 256) — generous on purpose: a reasoning model's thinking phase counts against `max_tokens`, and a tight budget cut the probe mid-think before any reasoning token landed, recording the wrong answer for the (source, model) pair |
| `NX_CAPABILITY_PROBE_MAX_TOKENS` | response budget for the startup tool-calling capability probe (default 768; was a hard-coded 64). On hybrid reasoning models (glm / qwen / R1-style) the thinking phase counts against `max_tokens` and can easily burn 100+ tokens before the first tool-call token — the old 64 cut the probe mid-think, returned no `tool_calls`, and misclassified a NATIVE server as TEXT, pinning `tool_choice=none` for the whole session (the model could then never emit a native call to reactively un-confirm it, so every turn stalled reasoning-only until the forced-final salvage printed its reasoning as the answer). One-shot, non-streaming, once per (source, model) |
| `NX_REASONING_BUDGET` | cap on the model's reasoning/thinking phase in tokens, sent as `{"reasoning": {"max_tokens": N}}` in the request body (default `0` — off, let the model self-terminate its thinking; ignored by backends that don't consume that field; set per-source with `NX_SOURCE_<NAME>_EXTRA_BODY`; toggle live with `/budget`) |
| `NX_TOOL_OUTPUT_VISIBLE` | render each tool call's result body (default `0` — collapsed, only the call header shows; `1` renders the result body — toggle live with `/tooloutput` (`/tools` legacy alias); `todo_write` always renders). In a parallel turn the collapse hint appears once on the group footer, not once per call. Approval previews count as tool output too (0.53.1): a `git_commit` staged diff or an `edit_file` before/after shown at the Y/n prompt collapses with the rest — the prompt line itself always shows |
| `NX_TOOL_RESULT_CHARS` | cap on tool-result characters entering context (default 102400; `0` disables the cap). Oversized results are head/tail-kept and their full text spilled to `~/.nx/overflow/<sha256>.txt` — the result tells the model the path so it can `read_file`/`search_files` the elided middle |
| `NX_TOOL_RESULT_SPILL` | write oversized tool results to `~/.nx/overflow/` (default `1`; `0` falls back to the old blind elision marker). Spill files are created `0600` regardless of umask, like every other nx sink |
| `NX_TOOL_RESULT_SPILL_MIN_CHARS` | append-time spill floor: a tool result larger than this (default 102400, equal to the cap) is written to the overflow store the moment it is produced, *before* it is ever sent, and the model gets the bounded head/tail + path stub instead. This pre-empts the context-overrun evictors, which rewrite already-sent bytes and so bust the server prompt-prefix cache (`cached: 0%`) on exactly the turns the prompt is largest. `0` restores the old behaviour (spill only above `NX_TOOL_RESULT_CHARS`; the evictors remain as the last-resort spiller) |
| `NX_TOOL_RESULT_REDACT` | mask known credential shapes in tool results before they are shown, capped, spilled, or persisted (default `1`; `0` = pass-through). The same token shapes the approval-side `NX_BASH_PROMPT_SECRETS` gate checks on the way in, covering the way out: a result that printed a key is replaced with a `[redacted:api-key]`-style placeholder (kind kept, value gone) |
| `NX_ARG_REPAIR` | schema-level argument and tool-name repair before a tool body runs (default `1`; `0` = strict pass-through). Coerces wrong-typed args against the tool's declared schema (`"5"` to `5`, `"true"` to `true`, `42` to `"42"`), case-folds a unique enum match (`"In_Progress"` to `"in_progress"`), renames a near-miss key (`maxresults` to `max_results`; difflib similarity 0.8+ and unambiguous only) plus curated per-tool aliases similarity can't reach (`cmd` to `command`, `old_str` to `old`, `file_path` to `path`) — and repairs the tool NAME itself: one alias (`bash` to `run_bash`, `grep` to `search_files`, `str_replace_editor` to `edit_file`) or one unambiguous close match executes under the real name. A missing required argument is refused up front with the declared schema (`ERROR: read_file missing required "path" — schema: …`) instead of a TypeError from the body; values are never fabricated. Every fix is reported in the tool result as an `[arg repair: <note>]` prefix, never applied silently. Since 0.131.0 each repair is also mirrored onto the usage stream (`repair` event, tool recovered from the `<tool>: …` note) so `/retro` and the session-end retro can count friction per tool |
| `NX_RETRO_ON_EXIT` | silent self-retro at session end (nx #1, default `1` = ON; `0` opts out). On a clean exit (`/quit` or EOF/Ctrl-D) after the final save/flush, `nx/_retro.py` scans this session's `usage.jsonl` for repeated friction (repairs, breaker trips, gates, retries, give-ups, failed tool calls), keeps patterns with 2+ hits, and writes at most 3 notes as project `gotcha` memories — the top patterns only, deduped against existing notes by a tool+event substring scan, so a session that repeats itself doesn't grow the store unboundedly. Runs at most once per session id (a module-level `_done` set) and never raises or blocks the exit; the same pass is available on demand via `/retro` (interactive: prints the patterns and asks before each write) |
| `NX_BUILD_VERBOSE` | `/build` progress lines (default `0`): one line per pipeline step — branch, spec, each package's worktree/merge/conflict retry, each critique round, each fix. Off by default because `/build` prints the spec gate and the final report itself; on is for watching a long run |
| `NX_BUILD_PARALLEL` | run `/build`'s work packages as concurrent agents (default `1`). Concurrency needs a yolo session *and* worktree isolation, so in an attended session the packages run sequentially on the parent thread regardless (a quiet worker thread cannot show the write approval prompt); `0` forces sequential even under `--yolo` |
| `NX_BUILD_WORKTREE` | give each `/build` package agent its own git worktree (default `1`; `0` runs them all in the shared tree — sequential only). Isolation is what makes parallel writers safe; without it the pipeline serializes instead of racing |
| `NX_BUILD_PACKAGE_TURNS` | tool-turn budget per `/build` package agent and per conflict retry (default `80`) |
| `NX_BUILD_FIX_TURNS` | tool-turn budget for the phase-4 follow-up agent that fixes ONE structural critique finding (default `60`) |
| `NX_OVERFLOW_TTL_DAYS` | purge spill files older than N days (default 7, like bg-logs) |
| `NX_OVERFLOW_ECHO_CHARS` | max chars of head/tail content echoed back alongside the spill path (default 2000; `0` = path note only) |
| `NX_HOME` / `NX_SESSIONS_DIR` | where the session store (`sessions.sqlite` and, until imported, the legacy per-session JSON files) lives |
| `NX_READ_FILE_LINES` | default `read_file` window size in lines (default 400; `limit<=0` reads to end) |
| `NX_READ_FILE_MAX_LINES` | ceiling on the `read_file` window in lines (default 25000; `0` disables). A larger `limit` is refused with steering toward `search_files`, so a model cannot fill its context by paging a huge file with ever-bigger windows. `.pdf` paths bypass paging entirely and route to the markdown extractor (`NX_PDF_MAX_PAGES`) |
| `NX_MENTION_INLINE_LIMIT` | max size in bytes for a file `@mention` to inline whole (default 4096). Bigger files get a stub naming the size and pointing at `read_file`/`search_files`. CSV/TSV/`.tab` over 1 KB always take the preview path instead: header + first 10 data rows + row count, since the columns are nearly all a data-file mention is about |
| `NX_MENTION_CSV_SCAN_BYTES` | max CSV/TSV preview size for the row-count scan (default 1048576 = 1 MB); bigger files skip the full read and report `N+` rows instead of an exact count |
| `NX_PDF_MAX_PAGES` | pages a `@doc.pdf` mention extracts to markdown (default 50; `0` = no cap). A longer PDF is extracted up to the cap and the block header names the range carried (`PDF, 300 pages — carried the first 50`), so the truncation is visible and a bigger pull is one env var away. Since 0.100.0 extraction runs via CLI binaries on PATH — `mutool convert -F markdown` (mupdf-tools) → `pdftotext -layout` (poppler-utils) → tesseract OCR for scanned pages (capped at 10 pages) — falling back to `pymupdf4llm` (now an optional extra, `pip install nx[pdf]` — or `uv tool install nx --with pymupdf4llm`) only when none of those binaries exists; a missing toolchain surfaces as an `ERROR:`/`not inlined` line in the mention block instead of erroring out the turn |
| `NX_PDF_TIMEOUT` | wall-clock ceiling in seconds for one CLI PDF extraction, including the OCR pass (default 300; `0` = no timeout). On timeout the mention degrades to an error note rather than holding the turn |
| `NX_READ_FILE_SCAN_LINES` | how far past a windowed `read_file` the header's exact line-count scan may reach (default 200000); beyond it the header reports a lower bound as `N+` instead of scanning an unbounded file |
| `NX_HISTORY_LINES` | how many past prompt submissions Up/Down recalls across runs (`~/.nx/input_history.json`; default 100; `0` disables). Override the file itself with `NX_INPUT_HISTORY_FILE` |
| `NX_PANEL` | show the right status rail on the `/tabs` live region by default (set `1`; otherwise toggled per session with `/panel`) |
| `NX_PANEL_WIDTH` | width of that status rail in columns (default 24; clamped 12..40) |
| `NX_LOG_MAX_BYTES` | size cap on the per-session request log (`~/.nx/logs`, default 10 MB) — the log rolls rather than growing without bound |
| `NX_LOG_REQUESTS` | log request/response bodies on every model call (default `1`; `0` logs a stub line instead — the per-request json.dumps of a large prompt is real harness CPU, and the transcript already holds the payload) |
| `NX_USAGE_LOG` | write a usage event stream to the session logs (`logs/usage.jsonl`; default `1`; `0` disables) — one JSONL line per tool call (duration, outcome, args/result sizes and truncated heads) plus `model_call` (per settled model turn: duration, finish reasons, token split, TTFT, turn number), `agent_spawn`/`agent_done` (sub-agent tier, resolved model, status, turns, elapsed), `give_up` (with turn number), and repair/breaker/gate/retry events, each tagged with the model and agent tier; the raw feed for offline harness/model failure analysis (query it with DuckDB) |
| `NX_MAX_UNDO` | how many `/undo` checkpoints are kept per session (default 20) |
| `NX_MAX_UNDO_BYTES` | byte cap on the persisted undo stack — oldest checkpoints evicted first (default 32 MB; `0` disables the cap). The stack holds whole file contents, so without it a long session against big files is an unbounded RSS term |
| `NX_BASH_POLL_SECONDS` | how long a `run_bash` call polls inline before backgrounding as a fallback beyond its timeout (default 3) |
| `NX_ORIENTATION` | append a one-time working-directory + read-only-tools note to fresh sessions (default `1`; `0` disables) |
| `NX_DELEGATION_NUDGE_READS` | how many consecutive self-performed exploration calls (read_file/list_dir/search_files/duckdb, cache hits included) with no `spawn_agent` in between before a runtime note tells the model to delegate the rest of the sweep (default `5`; the streak resets on each fire so the note RE-ARMS every 2× that many additional consecutive reads, i.e. cumulative `5`, `15`, `25`; only fires while `NX_SUBAGENTS=1`; `0` disables) |
| `NX_GARBLE_GUARD` | scan every edit's `new` text for model-garble signatures — duplicated lines, stuttered words/phrases, line-boundary splices, mid-sentence abandoned rewrites (default `1`; `0` disables). Unambiguous garble is repaired before the write (the tool result says so); ambiguous garble is refused once, then written as-is if the model resubmits the byte-identical edit |
| `NX_PLAN_MAX_WORDS` | slug length cap for `/plan <goal>` → `.nx/<slug>.md` (default 8 words) |
| `NX_SUBAGENTS` | offer the read-only `spawn_agent` tool (default **on** since 0.77.0 — `0` disables; adds a schema to every turn's floor). Sub-agents get read/search tools plus a read-only text/utilities bash allowlist (rg, cat/head/tail/wc, find, diff, sort/uniq/cut, ps, checksums, etc.; git via the read-only risk verdict) — single bare commands only, no pipes/python; 2+ `spawn_agent` calls in one turn run concurrently, and spawns run in the background by default (job id now, answer as a short `[Runtime note]` + report path on a later turn; `/bg` lists them; full answer in `~/.nx/agents/<job>.md`). Enabled sessions also get a one-time `[Delegation note:]` steering the model to spawn agents proactively, plus the `NX_DELEGATION_NUDGE_READS` streak nudge |
| `NX_SUBAGENT_ASYNC` | sub-agent execution default (`1` = background, the default): a spawn returns a job id immediately and its answer arrives as a head + report-path `[Runtime note]`, so the parent TUI stays quiet during a fan-out. `0` restores the old foreground default, which renders the sub-agent's search/tool activity live and returns its answer inline |
| `NX_SUBAGENT_MAX_CONCURRENT` | how many background sub-agents may stream at once (default `12`; `0` = unbounded). Each async agent is its own thread + HTTP stream + parser, so a spawn beyond the cap parks as `queued` (shown by `/bg`) and starts when a running one finishes (PF-17) |
| `NX_SUBAGENT_ANSWER_KEEP` | chars of head+tail kept of a finished sub-agent's answer in the job record (default `4000`; `0` keeps everything) — the rest spills to the report file, so N finished agents don't hold whole answers in RSS |
| `NX_SUBAGENT_MAX_NESTED` | opt-in nested fan-out (default `0`): a `spawn_agent` inside a sub-agent is refused; `1` lets a sub-agent's turn fan out further (two-level swarms multiply threads/streams) |
| `NX_SUBAGENT_MAX_FINISHED` | finished async-agent records kept after their answer has been announced (default `20`; `0` = keep all). Older announced records are evicted so a long session's agent ledger does not grow without bound; a finished-but-unannounced answer is never evicted (PF-16) |
| `NX_EVENT_TAIL_LINES` | lines kept per background sub-agent in the live event sink ring (`nx/_eventsink.py`, default `200`) — what a `/tabs` tail can show for a running agent, and the final tail retained for post-mortem after the agent finishes. Display-only: sink lines never enter the transcript or reach the model. Read once at first sink use; absent/non-positive falls back to the default |
| `NX_TOOL_CACHE_MAX_ENTRIES` / `NX_TOOL_CACHE_MAX_BYTES` | bounds on the read-only tool-result cache (defaults `512` entries / 16 MB of result text; `0` disables that half). A hit refreshes recency, so eviction drops the genuinely cold entries first — this is what keeps a long session's RSS flat (PF-16, invariant I-8) |
| `NX_TOOL_CACHE_TREE_FILES` | entries in the bounded subtree stamp a cached `search_files` anchors on (default `5000`; `0` disables the stamp). The searched root's own mtime never moves when a file in a subdirectory changes, so the stamp is what makes a search hit honest (PF-18) |
| `NX_TABS_HEIGHT` | max rows of the `/tabs` live region painted below the render line while a turn runs (EVENTSURFACE_SPEC §2, default `8`, clamped to `1..20`); a non-int value falls back to the default. Bounded inline overlay, not an alternate screen — a rich `Live` on the shared console (`auto_refresh=False`, `transient=True`; SPEC_LIVEENGINE.md 0.128.0); Ctrl-T toggles it mid-turn, it closes on turn end |
| `NX_DIFF_HUNK_MAX_CHARS` | cap on the unified diff `edit_file`/`multi_edit`/`sed_edit`/`apply_patch` feed back to the model (default `8192`; `0` = uncapped). An elision notice names the dropped size so the model re-reads instead of assuming it saw the whole change (PF-19) |
| `NX_SUBAGENT_WRITE` | opt-in write tier for sub-agents (`NX_SUBAGENT_WRITE=1`): `spawn_agent` additionally gets `write_file`/`edit_file`/`multi_edit`/`sed_edit`, each still confirmed by you through the normal approval gate (risk classification + Y/n with a diff preview) and tagged `(sub-agent)` in the tool box. Write-capable agents cannot delete files or spawn other agents. Background (async) write-capable agents are allowed only in yolo mode (`--yolo` — the approval gate auto-approves quiet writes there); without yolo a background write-tier spawn is silently DOWNGRADED to foreground (0.99.1, same contract the former error message told the model to apply manually), because a detached thread cannot show the Y/n prompt and the write would be silently auto-denied. Scenario: driving a multi-phase plan while you are away — write the phase list with `todo_write` first, then spawn each phase as a background write/implement-tier agent and end the turn after arming the current one; each finished agent wakes the parent to review and spawn the next (the campaign wake instruction), bounded by `NX_WAKE_STREAK_LIMIT` |
| `NX_SUBAGENT_WORKTREE` | per-agent git worktree isolation for write-tier subagents (`1` = on, default `0`): each `tier:"write"`/`"implement"` spawn gets its own checkout under `~/.nx/worktrees/nx-wt-<job>` (detached HEAD from the project root at spawn time), so a headless swarm's implementers never trample the parent's tree or each other. The write-tier tools rewrite the agent's relative paths into its worktree (`nx/foo.py` → the agent's own copy), `run_tests` runs inside it (each agent verifies its own edits, no pytest races), and `git_commit` commits there. The agent never sees the worktree path while it works; its finished report and `[Runtime note]` name it, and the parent decides whether to merge/cherry-pick — nothing is pushed or merged automatically. Since 0.110.1 `NX_SUBAGENT_WRITE=1` alone never upgrades a spawn to write — an omitted `tier` is read; the env var only gates explicit write/implement/advisor tiers (this also means a read investigation from a non-git cwd no longer dies on worktree creation, which since 0.107.0 is a refused spawn for write tiers). The worktree is removed together with its record when the finished agent is evicted from the parent's ledger (`NX_SUBAGENT_WORKTREE_TTL_HOURS` reaps crash leftovers); worktree creation failure is not fatal (the agent runs in the shared tree and says so). Read tools are not rewritten — an implement agent may read references outside its scope. Pair with `scope` for write-location discipline |
| `NX_SUBAGENT_WORKTREE_TTL_HOURS` | TTL for leftover per-agent worktrees (0.103.7, default `24`; `0` = never sweep): a killed process can leak `~/.nx/worktrees/nx-wt-agent-*` checkouts, so every write-tier spawn sweeps those older than this before creating its own tree, then runs one `git worktree prune` on the project root. Session worktrees (`nx-wt-session-*`) and fresh agent trees are never touched |
| `NX_SESSION_WORKTREE` | per-session worktree isolation (0.103.0; `1` = on, default `0`, also `nx --isolate`): the whole interactive REPL runs inside its own worktree under `~/.nx/worktrees/nx-wt-session-<sid>`, branched `nx/session-<sid>` off HEAD. Several isolated sessions in one repo share nothing — separate trees, separate git indexes, separate commits — so agents in one session can never clobber another's work. Merging the session's work back is an explicit `git merge nx/session-<sid>` you run in the main checkout. Re-entering the same session id (`--session <id>`) re-attaches to the existing tree; worktree creation failure is not fatal (a warning prints and the session runs in the shared tree). The undo store, staleness guard and `git_commit`'s isolated staging index key off the session id either way |
| `NX_SESSION_SWEEP_INTERVAL` | amortized store GC (default `300` s, process-wide): stale-FTS-row sweeps, image-payload collection, quota enforcement and the retention sweep run once per interval on save instead of every save; `0` restores the legacy every-save behavior |
| `NX_SESSION_RETAIN_SMALL_DAYS` | opt-in retention gate (0.104.0, default `0` = off): sessions with fewer than `NX_SESSION_RETAIN_SMALL_MAX_MSGS` messages (default `10`) that have been untouched for this many days are deleted by the amortized sweep and by `nx sessions --prune`. Deleting is only ever opt-in via this flag |
| `NX_SESSION_RETAIN_SMALL_MAX_MSGS` | message threshold for the small-session gate above (default `10`): a chat below this count is considered a throwaway once it ages out |
| `NX_SESSION_RETAIN_FLOOR_MSGS` | protection floor for the small-session gate (default `0` = no floor): sessions with at least this many messages are never auto-deleted by the small gate, however old — only you can delete them (`/delete`, `nx sessions --prune` reconciliation). Raise `NX_SESSION_RETAIN_SMALL_MAX_MSGS` and set this together to keep real work safe |
| `NX_SESSION_RETAIN_FLOOR_DAYS` | even big sessions expire (default `0` = never): sessions older than this many days are deleted by the retention sweep regardless of message count. Independent of the small gate; both flags must be explicitly set for anything to be deleted |
| `NX_RESUME_CWD` | on `--resume`, restore the session's saved working directory (default `1`; `0` stays in the current cwd). A saved cwd that no longer exists is skipped with a warning |
| `NX_SESSION_DESC_REFRESH` | re-generate the session's auto description mid-session (interval/trigger knob for the background description writer; unset = the default cadence) |
| `NX_SESSION_LOCK` | per-repo advisory session lock (0.103.0; on by default): a second nx REPL started in the same repo is told which session holds `<repo>/.nx/session.lock` and must Y/n to share the tree (an `NX_SESSION_WORKTREE`-isolated session can't contend and claims it silently). The lock is an `flock` on a dedicated fd, so a crashed session releases it automatically; `0` disables the warning for users who intentionally run parallel sessions in one tree |
| `NX_STALENESS_MAX_PATHS` | cap on the cross-session staleness registry (0.103.0, default `512`): paths this session has `read_file`'d, LRU-evicted oldest-first. `0` disables the staleness guard (write tools then never refuse on "changed since last read") |
| `NX_FILE_CLAIMS` | cross-process file-claim protocol (default ON; `0` disables): before a write, `write_file`/`edit_file`/`multi_edit`/`delete_file`/`sed_edit` atomically claim the target file in the shared session store, so a second nx process (another REPL, a concurrent agent loop, another host on the same tree) editing the same file is refused with an explicit "claimed by another agent" error instead of clobbering it. A claim is `path -> owner_id, pid, claimed_at, cwd, pid_start_time` (the last being the holder's process start-time tick, so a recycled pid can't read as a live holder; a pre-0.119.0 row without one keeps the old bare-pid liveness check); a live pid with a different owner blocks **only when it was claimed from the same working directory** (claims are repo-scoped, so an nx session working in an unrelated folder on this host never refuses an edit here), and a dead/recycled pid, an expired row, or another repo's claim is reclaimed in place; the same owner re-claiming (several tools in one loop) just heartbeats. A clean REPL exit (`/quit`/`/exit`, or `Ctrl-D`/`Ctrl-C` at the prompt) releases this process's claims. Best-effort: any store failure fails open (the edit proceeds) |
| `NX_FILE_CLAIM_TTL` | seconds a file claim stays valid without a heartbeat (default `3600`); a claim older than this from a live pid is reclaimed by the next writer, and the refusal message quotes this value as the remedy. `0` = never expires (only a dead pid frees the file) |
| `NX_SUBAGENT_SYNC_INLINE` | set to 1 to restore the pre-0.96 foreground `spawn_agent` behavior: the full answer inline as the tool result. Default 0 — both foreground and background agents write their full report to `~/.nx/agents/<job>.md` and the parent sees only a ~600-char head + the report path, keeping subagent exploration out of the main context |
| `NX_SUBAGENT_RECOVERY_RETRIES` | rescue attempts allowed INSIDE a sub-agent loop, per failure family — reasoning-only stalls, empty turns, malformed tool-call streams/collapses, text-protocol corrections (default `1`; the parent keeps its own `NX_REASONING_ONLY_RETRIES`/`NX_EMPTY_TURN_RETRIES`/`NX_MALFORMED_STREAM_RETRIES`). Keeps a flash-tier sub-agent from burning half its `max_turns` on recovery cycles; when a sub-agent has spent its rescue allowance, the loop pre-forces the final answer one turn early so the remaining turns produce an answer, not more recovery. `0` disables the failsafes inside sub-agents entirely |
| `NX_SUBAGENT_DEFAULT_TURNS` | default `max_turns` for a spawn that doesn't specify one (default `100`; was 12, which left real investigations capping out with no answer). Also a **floor**: a parent that passes a smaller `max_turns` — several models carry their own 5/10/12-scale budget notion and were capping their agents out mid-investigation — is raised to this value, so the only way to run a genuinely short agent is to lower this knob. Also what the tool schema advertises to the parent model for read-tier spawns |
| `NX_SUBAGENT_MAX_TURNS` | hard ceiling on a spawn's `max_turns` after clamping (default `500` since 0.103.10 — matches the parent loop's `MAX_MODEL_TURNS`; was 40, then 200). Wins over the tier floors if configured inverted |
| `NX_SUBAGENT_CONSULT_BUDGET` | advisor-tier agents (`tier='advisor'`) may call `consult_advisor` — a one-shot, no-tools side-call to the parent model that returns a decision — at most this many times per agent (default `3`; `0` disables consulting entirely). Past the budget the tool refuses with a breaker-style error telling the executor to take the conservative option and log it under '## Deviations' in implementation-notes.md |
| `NX_SUBAGENT_CONSULT_TIMEOUT` | seconds an advisor-tier `consult_advisor` side-call waits for the parent model before giving up (default `60`). On timeout (or an unparseable reply after one retry) the executor is told to take the conservative option and log the deviation, not to stop |
| `NX_SUBAGENT_WRITE_RECOVERY_RETRIES` | recovery-rescue allowance inside a write/implement/advisor-tier sub-loop (default `3`, since 0.121.2) — write tiers keep the parent's default instead of the `NX_SUBAGENT_RECOVERY_RETRIES` read-tier clamp: a single reasoning-only stall forcing the final answer kills a mid-edit agent, and unlike a read sweep a write agent can't cheaply be re-spawned with its plan. `0` disables the failsafes inside write-tier loops entirely |
| `NX_SUBAGENT_WRITE_TURNS` | turn-budget floor for write/implement-tier spawns (default `250` since 0.103.10): delegated multi-file work (read the area, edit, verify, iterate) needs more turns than a read sweep, so a write-tier agent that doesn't ask for more (or asks for fewer — several models pass a 5/10/12-scale budget notion) is raised to this floor. Read-tier spawns floor at `NX_SUBAGENT_DEFAULT_TURNS` (100) |
| `NX_SUBAGENT_SOURCE` / `NX_SUBAGENT_MODEL` | route sub-agent (`spawn_agent`) model calls to a different endpoint/model — e.g. `NX_SUBAGENT_SOURCE=local NX_SUBAGENT_MODEL=glm-5.3-flash` runs every sub-agent on the `local` source's flash model while the parent keeps its main model. `_MODEL` alone switches model id on the parent's client; `_SOURCE` alone uses that source's client + default model. The target is used as-is (no probes, no switch); unknown names fall back to the parent's source with a one-time warning. Resolved per agent start (a `/source` switch mid-session is picked up by the next spawn), and per-thread — concurrent agents never swap process globals |
| `NX_SUBAGENT_MODEL_POOL` | comma-separated sub-agent models handed out round-robin, one per `spawn_agent` — each entry is a bare model id (parent's client) or `source:model` (that endpoint), e.g. `glm-5.3-flash,qwen3.8-flash-next,deepseek-v4.1-flash` spreads a fan-out across three servers. Takes priority over `NX_SUBAGENT_MODEL`/`_SOURCE`; unset → single-model routing. Pair with `NX_SUBAGENT_MAX_CONCURRENT=12` for ~4 agents per model in flight |
| `NX_VISION` | offer the `read_image` tool on native-tool-call sources (default off; requires the backend to accept image content parts). Supported formats: png/jpeg/gif/webp/bmp — chosen by extension, with a magic-byte sniff fallback for missing/unknown extensions, so a real image under any name reads. `~` expands. Also gates the `@image` chat mention (the picture attaches inline on the turn that asks about it; with vision off the mention carries a note naming the gate). Base64 inflates 4/3x, so even a modest image is heavy context (see `NX_IMAGE_MAX_BYTES`); a backend without vision errors on the request *after* the tool call — that error is the signal to turn this off for that source |
| `NX_WEBFETCH` | offer the `web_fetch` tool (default ON; `0` removes it from the tool catalog entirely — for fully offline hosts): keyless web access via the free DuckDuckGo endpoints — no API key, no paid search. Pass `query` for a DuckDuckGo-lite search (title/URL/snippet results) or `url` to fetch a page as readable text (links kept as `[text](url)`; truncated at `NX_WEBFETCH_MAX_BYTES`). Tier-1 always-on (0.113.0 — it is the tool a session needs on turn 1). Courtesy of the free resource is built in: responses are cached GLOBALLY for 7 days (`NX_WEBFETCH_CACHE_TTL`, default 604800 s — persisted under `<NX_HOME>/web-cache/cache.jsonl`, so a hit costs the endpoint nothing even in a fresh session) and stale pages are REVALIDATED with `ETag`/`Last-Modified` (a 304 transfers no body), searches are throttled to at most one per `NX_WEBFETCH_SEARCH_INTERVAL` seconds (default 3) and page fetches of one host to one per `NX_WEBFETCH_HOST_INTERVAL` seconds (default 2), `robots.txt` is honored for page fetches unless `NX_SEARX_URL` is set (`NX_WEBFETCH_ROBOTS` — `1`/`0` forces it either way), `429`/`503` responses back the host off per its `Retry-After`, requests are GET-only with a polite version-accurate User-Agent, and parallel calls for the same URL share one fetch. Failures fail soft with retry guidance. A self-hosted SearXNG (`?format=json`) is the preferred search backend: set `NX_SEARX_URL` to the instance root (e.g. `http://localhost:8080`; searches go to `{NX_SEARX_URL}/search?q=...&format=json`) — unset or unreachable falls back to DuckDuckGo lite (the failure names the likely settings.yml misconfiguration rather than falling back silently) |
| `NX_WEBFETCH_MAX_BYTES` | text size cap on a `web_fetch` page fetch (default `20000`; the tail is elided with a notice so the model fetches a narrower URL instead of assuming it saw everything) |
| `NX_WEBFETCH_SEARCH_INTERVAL` | minimum seconds between searches of the free DuckDuckGo endpoint (default `3`) — the courtesy floor for the shared free resource |
| `NX_WEBFETCH_HOST_INTERVAL` | minimum seconds between page fetches of the SAME host (default `2`); loopback (your own SearXNG, test servers) is exempt. A robots.txt `Crawl-delay` widens this for that host |
| `NX_WEBFETCH_ROBOTS` | honor `robots.txt` for `web_fetch` page fetches (default `1`, except `0` by default when `NX_SEARX_URL` is set — a self-hosted SearXNG implies a trusted setup and blanket robots refusal blocked most of its results; `1`/`0` forces it either way). Parsers are cached per host; unreachable/broken robots.txt means allowed; disallowed URLs are refused before any fetch |
| `NX_WEBFETCH_CACHE_TTL` | seconds a `web_fetch` search/page result stays cached (default `604800` = 7 days); the cache is global and persisted under `<NX_HOME>/web-cache/`, so a fresh session hits it too. Expired pages WITH validators (ETag/Last-Modified) are revalidated instead of refetched. Cache hits are marked `[cached]` and cost the endpoint nothing |
| `NX_WEBFETCH_ALLOW_PRIVATE` | default `0`: the SSRF guard refuses `url=` fetches (and redirect hops, and search-result URLs) that resolve to loopback/private/link-local addresses, so a hostile page cannot pivot the approval-free tool at internal services (metadata endpoints, the loopback SearXNG, RFC1918). Set `1` to lift it on an explicitly trusted network. `NX_SEARX_URL` itself is exempt when it is loopback or https |
| `NX_SEARX_URL` | root URL of a self-hosted SearXNG instance (e.g. `http://localhost:8080`) used as the `web_fetch` search backend (`{url}/search?q=...&format=json`); preferred over the DuckDuckGo-lite fallback, and exempt from the DDG search throttle since it is your own instance. Enable the `json` format in the instance's `settings.yml` (`search: formats: [html, json]`); a configured-but-failing instance falls back to DuckDuckGo with a note naming the misconfiguration |
| `NX_TELEGRAM_BOT_TOKEN` | token for the `nx telegram` gateway (no default; unset = the verb exits with guidance instead of touching the network). Get one from @BotFather. Never echoed to logs or chat output |
| `NX_TELEGRAM_ALLOWED_CHATS` | comma-separated hard allowlist of Telegram chat IDs the gateway serves (no default; EMPTY refuses to start — an open bot is a remote-code-execution surface). Messages from other chats are dropped without reply. Sending `/new` to the bot wipes the gateway's conversation in place for a fresh context (0.85.0) |
| `NX_TELEGRAM_ALLOWED_USERS` | comma-separated hard allowlist of Telegram user IDs whose messages are served (0.95.1); default = the chat allowlist, which is exact for private chats. In a group you must set this explicitly — it stops other group members from driving the host |
| `NX_TELEGRAM_ALLOW_GROUPS` | default off: only private chats are served (0.95.1). Set `1` to serve a group chat you control (pair with `NX_TELEGRAM_ALLOWED_USERS`); without it a group in the chat allowlist is silently dropped |
| `NX_TELEGRAM_POLL_INTERVAL` | seconds between polls after an empty or failed getUpdates round (default 1.0); the long-poll itself blocks up to about 30 s per round, so this is only the backoff floor. Status rendering (0.95.0): one summary bubble per turn — the user's ask, ×N-collapsed tool calls, the latest assistant prose, live sub-agent progress (`🤖 agent turn n/m`), elapsed time — and the final answer always arrives as its own new message (guaranteed delivery; the pre-0.95 edit-on-status contract could leave the answer unrendered until the next turn) |
| `NX_WAKE_PROMPT` | auto-continuation (default `1`): when a background job (a detached `run_bash` command or an async `spawn_agent` sub-agent) finishes while you idle at an empty prompt, nx starts a model turn by itself, feeding it the completion note — no keystroke needed. The note is delivered even when the turn answers without tool calls, and an already-reported job never re-wakes (0.36.3 fixed a loop where a text-only turn left the job unreported and the REPL auto-continued forever; 0.37.1 fixed a phantom banner race — a flag raised just before a user-typed turn no longer fires an extra noteless wake after that turn delivers the note itself; 0.37.4 stopped the "Nothing to do — that was the script from my last turn" wakes: a command that finishes inside its own poll window (or one whose log `wait_background`/`read_file` already returned) counts as reported, so nx never announces output the transcript already contains). Half-typed input is stashed and restored after the wake turn; `0` restores the old behavior, where the note waits for your next message |
| `NX_FUSION` | action fusion (default ON; `0` disables): `write_file`/`edit_file`/`multi_edit` gain an optional `then_run` argument (`{command, timeout?}`) whose validation command runs in the SAME tool call after the edit succeeds, saving a model round-trip. If the edit fails the command is skipped with `[then_run:skipped]`; command outcomes append `[then_run:succeeded]`/`[then_run:failed]` + output to the edit's result. A fused command that outlives the poll window backgrounds exactly like `run_bash` (the result embeds `[background] PID …`), and arms the auto-continuation poller + the `wait_background` guidance the same way, so its finish notice reaches the model |
| `NX_FUSION_POLL_SECONDS` | how long a fused `then_run` command may run before it backgrounds (default `30`; `0` waits indefinitely) |
| `NX_OBS` | observation pack (default off; `1` enables): a tool result larger than `NX_OBS_THRESHOLD` bytes is archived verbatim in the shared content-addressed store (`~/\.nx/obsstore/`, deduped by content across tools/sessions/processes) and, after its first `NX_OBS_FULL_SENDS` provider requests, replaced by a compact handle (`id: obs_...` + head/tail excerpt). The `obs_recall` tool pages the archived original back byte-exactly (`offset`/`next_offset`/`eof`); results containing a reducer receipt line are never packed |
| `NX_OBS_THRESHOLD` | byte size a tool result must exceed to participate in observation packing (default `10240`) |
| `NX_OBS_FULL_SENDS` | how many provider requests a packed result is sent at full weight before the handle replaces it (default `2`) |
| `NX_OBS_STORE_TTL_DAYS` | age prune for the shared tool-result store (default `7`; untouched objects are collected by the amortized sweep or `nx sessions --prune`) |
| `NX_OBS_STORE_MAX_MB` | byte quota for the shared tool-result store (default `512`; `0` disables; over quota the oldest objects are evicted first) |
| `NX_OBS_MAX_TRACKED` | bound on the in-process pack-send counters (default `4096`, drop-oldest) |
| `NX_INTERRUPT_WATCHER` | `0` disables the stdin Esc-interrupt watcher for the whole process (default `1`). For headless runs (`nx run -f`, `--prompt-file`, `--parallel`) there is no tty to watch: this removes the watcher thread and its termios snapshot from every agent in a swarm. A quiet (background sub-agent / batch worker) loop never starts or arms the watcher regardless of this setting |
| `NX_TOOL_FLOOR_SLIM` | `1` withholds the `duckdb` and `read_image` schemas behind `search_tools` (default off). For swarms: every agent pays the always-on tool floor, so demoting rarely-first-turn tools to tier-2 shrinks each agent's prompt by ~400 tokens with `search_tools` as the on-demand lifter |
| `NX_STORE_BUSY_TIMEOUT_MS` | SQLite busy timeout for the session store and memory index (default `750` ms, was a hardcoded 5 s block). Under a shared `NX_HOME` a swarm's per-turn saves contend on one WAL write lock; a shorter wait plus bounded retry beats a 5 s stall, and every consumer degrades instead of raising on timeout |
| `NX_REDUCER` | evidence-preserving reducer (default off; `1` enables): a large diagnostic output (build/test command) may be replaced by a compact receipt produced by a second model call, accepted only if EVERY quoted line verifies byte-for-byte against the on-disk archive (`~/.nx/evidence-preserving-reducer/`); any mismatch, schema miss, or missing failure evidence fails OPEN with the original output |
| `NX_REDUCER_SOURCE` / `NX_REDUCER_MODEL` | route the reducer summary call to a cheaper endpoint (default: the active source/model) |
| `NX_REDUCER_MIN_BYTES` | diagnostic output below this size is never reduced (default `4096`) |
| `NX_REDUCER_MAX_CHARS` | output above this many chars is left alone (default `600000`) |
| `NX_REDUCER_MAX_TOKENS` | answer cap for the summary call (default `2048`) |
| `NX_REDUCER_TIMEOUT` | seconds before the summary call gives up and the original output passes through (default `90`) |
| `NX_REDUCER_MAX_ITEMS` | evidence items accepted in a receipt (default `12`) |
| `NX_REDUCER_MAX_QUOTE_CHARS` | per-quote length cap in a receipt (default `600`) |
| `NX_OCC` | online context compact (default off; `1` enables): a `todo_write` that newly completes a step marks a plan boundary; after the turn settles, the economics check (window pressure vs. cache break-even over the estimated remaining-request horizon) decides whether to run nx native compression early. After a successful fold a `[Runtime note]` tells the model the task is still active and to re-issue `todo_write` with the remaining plan |
| `NX_OCC_KEEP_RECENT_TOKENS` | tokens of recent context the compact fold must keep verbatim (default `20000`) |
| `NX_OCC_CACHE_RATIO` | assumed cache-write:read price ratio in the break-even math (default `12.5`; `0` disables pricing, deferring folds to window pressure only) |
| `NX_WAKE_POLL_SECONDS` | how often the wake poller checks for finished background jobs while the prompt idles (default `2.0`, floor `0.2`) — the worst-case delay between a job finishing and the auto-continuation turn starting |
| `NX_WAKE_STREAK_LIMIT` | cap on consecutive UNATTENDED auto-continuation turns (default `25`, clamped to >=1) (0.121.0): every other wake guard is a liveness guard (no wake with nothing to say), but a wake turn that legitimately spawns new work creates new announced-once jobs — productive-by-construction, hence unbounded when nobody is at the keyboard. Any user-typed turn (including slash commands) resets the streak to `0`, so a chain only runs this many turns deep before wakes stop and nx waits for you |
| `NX_IMAGE_MAX_BYTES` | max image size `read_image` will load into context (default 8 MB). Refusal, not downscaling — no image library is bundled; lower it (a few hundred KB) on small local context windows, where even a small image would otherwise overrun and trigger auto-compression. Read at call time, so a live override works without a restart |
| `NX_PRE_TOOL_HOOK` / `NX_POST_TOOL_HOOK` | shell command run around every tool call with `{tool, args(, result)}` JSON on stdin; a non-zero pre-hook exit vetoes the call (NX_HOOK_TIMEOUT, default 10s, fail-closed). The pre-hook runs even on cache-hit reads; the post-hook fires on every read too — cache hits carry `"cached": true` in the payload. Prefetched reads on a mixed turn (reads + a write/edit in one tool-call batch) dispatch through the same run_tool() gate, so a veto blocks them and the post-hook sees them like any other read. Since 0.29.5 the pre-hook sees REPAIRED args: schema-level arg repair (`NX_ARG_REPAIR` — coerced types, renamed keys, resolved tool name) runs before the hook, so a policy keyed on `path` sees `path`, not the model's near-miss spelling |
| `NX_INTERRUPT_RESTORE_WAIT` | how long nx waits for the Esc-watcher to restore cooked terminal input mode after a turn/tool wait (default 2.5s — covers the 2s Esc-redirect typing window); shrink only if you never type a redirect immediately after Esc |
| `NX_ESC_LADDER_SECONDS` | Esc-escalation window (0.136.0 UIUX C1): a second Esc inside N seconds of the first hard-cancels the running tool (run_bash jobs get SIGTERM→SIGKILL after the grace below; the rest of the tool group is reported to the model as cancelled). Default 0.8; a press faster than 80ms is treated as key-repeat, not the ladder |
| `NX_FULLSCREEN` | `0`/`false`/`off` disables the Phase D alternate-screen views (0.137.0) — `/tabs`, `/diff`, `/sessions` then print their inline output instead. Default unset = on wherever stdin and stdout are a tty; a non-tty (headless, piped, `nx run`) always falls back to the inline form, and a view is never opened while a turn is running |
| `NX_CANCEL_KILL_GRACE` | seconds a hard-cancelled `run_bash`/`wait_background` job has after SIGTERM before its process group is SIGKILLed (default 2.0, UIUX C1). Bounded by design — no unkillable wait; the job is marked `killed` in `/bg` |
| `NX_WRITE_FENCE` / `NX_WRITE_ROOTS` | hard write fence (default `1`): before a path-carrying tool call runs, each path it could touch is resolved exactly as the write itself would resolve it (realpath, symlink followed) and must land inside a writable root — the project root (git root or cwd), the session home `~/.nx`, plus any `NX_WRITE_ROOTS` dirs (colon-separated, `~` expands). A call aiming outside is refused before approval, hooks, or any disk change. Committed-symlink dotfile farms keep working when the link's destination also sits inside a root; destinations in credential or shell-config spots (`~/.ssh`, `~/.aws`, rc files, private keys) stay fenced even from inside a root, since the write tools follow symlinks by design. Applies to `write_file`/`edit_file`/`multi_edit`/`delete_file`/`sed_edit` (literal arg plus each glob match) and `apply_patch` (each `+++` target joined onto the patch base). `run_bash` gets a heuristic backstop layer (`NX_BASH_FENCE`, default `1`): the command is scanned for the common write shapes — redirections, `tee`, write-capable binaries (`cp`/`mv`/`dd`/`tar`/...), and interpreted-language file writes inside here-docs (`open(p, 'w')`, `Path.write_text`) — and any destination outside the writable roots, containing a shell variable, or bare (an unresolvable identifier) refuses the command before it runs. It is a scanner, not a shell parser: exotic shapes can still slip past, and the classifier + approval remain the real shell guard. `0` disables the fence; `NX_BASH_FENCE=0` disables only the bash layer |
| `NX_BASH_MAX_SECONDS` | watchdog: reap a backgrounded `run_bash` job after N seconds (SIGTERM→SIGKILL; default 1800; 0 = unbounded) |
| `NX_BASH_SLEEP_CAP` | ceiling on the `sleep N && cmd` auto-extended poll window (default 120 s; a longer sleep backgrounds with a log path instead of blocking the turn; 0 = uncapped) |
| `NX_BASH_TIMEOUT_CAP` | fail-fast ceiling on an explicit `run_bash`/`then_run` `timeout=` argument (default 120 s; higher values are clamped down and the command backgrounds with a log path; 0 = uncapped) |
| `NX_WAIT_BACKGROUND_CEILING` | max seconds `wait_background` waits when its `timeout` is 0/unset (default 1800; then it returns the log path; 0 = wait indefinitely) |
| `NX_WATCHDOG_POLL_SECONDS` / `NX_WATCHDOG_KILL_GRACE` | watchdog granularity: how often the reaper checks whether the job died on its own before the budget elapses (default 0.5s), and how long it waits after SIGTERM before escalating to SIGKILL (default 5s) |
| `NX_BASH_PROMPT_SECRETS` | `1` forces the approval prompt when a `run_bash` command looks like it carries a credential (sk-/ghp_/AKIA/PEM…), even in yolo |
| `NX_GIT_INDEX_GUARD` | shared-index guard for `run_bash` (default `1` = on): a command with a mutating git op (`git add`, `git commit`, `git reset`, `git restore --staged`, `git merge --squash`) while `git status --porcelain` shows ANY staged entries always prompts, even in yolo, naming the staged paths — the raw op sweeps the SHARED index into its own result, and staged entries may be another concurrent session's work (`git_commit` stages into a private `GIT_INDEX_FILE`; raw shell git has no such isolation, and `/undo` cannot un-commit someone else's staged revert). Attribution is deliberately conservative: a non-empty staged set is enough to ask, so the common clean-index flow (`git add mine.py && git commit -m …`, checked before the command runs) stays prompt-free. Background sub-agents cannot prompt, so they refuse such commands and let the parent run them; `0` disables (same shape as `NX_CLOBBER_GUARD`, independent of it) |
| `NX_CLOBBER_GUARD` | truncation guard for `run_bash` (0.104.2, default `1` = on): a command that would truncate (not append) an existing, non-empty file — `> file`, `2> file`, a heredoc redirect — always prompts, even in yolo, because a truncated dotfile (an `~/.env` full of credentials) is unrecoverable in a way a failed command is not. Background sub-agents cannot prompt, so they refuse such commands and let the parent run them; `0` disables (same shape as `NX_BASH_PROMPT_SECRETS`, independent of it). 0.106.3: known-log/ephemeral extensions (`.log`, `.tmp`, `.out`, `.err`, `.swp`, `.cache`, `.pyc`, `.o`, `.so`, ...) are exempt — rewriting a run log is a normal, reversible operation and prompting there trains the user to mash Y — unless `NX_CLOBBER_LOG_EXTENSIONS=0`; dotfiles, `.lock`, `.env`, `.pem`/`.key` names are never exempt |
| `NX_PRICE_<SOURCE>_INPUT` / `_OUTPUT` / `_CACHE` | $ per 1M tokens for a source; feeds `/cost`'s $ column (unset sources show token totals only). On a multi-model host, `NX_PRICE_<SOURCE>_<MODEL>_<KIND>` (model slug upper-cased, non-alphanumerics folded to `_` — e.g. `NX_PRICE_HOSTED_QWEN3_32B_INPUT` for model `Qwen3-32B` or `qwen/Qwen3-32B`) prices one pinned model specifically and wins over the source-wide key; `/cost` shows each row's model when known. Calls the server answered without a usage object are chars//4 estimates — `/cost` marks them (`~N est`) so an estimate never reads as a measured bill |
| `NX_AUTOCOMPRESS_PERCENT` | auto-compress threshold (default 85; `0` disables) — checked after every settled model turn, including mid-task; a context-length 400 folds once and retries the turn. Since 0.41 a *refused* retry escalates instead of dying: tier 1 halves the kept tail, tier 2 halves it again and spill-evicts the largest tool results (`NX_OVERFLOW_FOLD_TIERS` rungs, default 3). Since 0.47 a fold reuses its previous summary append-only (the digest's bytes never change once written, so the server's prefix cache survives every fold) and spills the foldable region's biggest tool results to disk before summarizing |
| `NX_COMPRESS_KEEP_TOKENS` | token floor for the kept tail after an auto/manual fold (default `0` = message-count semantics; auto folds derive ~window/4 from the context probe when this is unset). A count-keep can leave a tail that alone 400s; the token floor grows the tail to cover recent work instead |
| `NX_COMPRESS_SPILL_MIN_CHARS` / `NX_COMPRESS_SPILL_MAX` | fold-time spill: the smallest tool result in the foldable region worth replacing with a head/tail + spill-path note before summarizing (default 8000 chars) and the per-fold cap (default 6). Pinned turns are never spilled |
| `NX_COMPRESS_SOURCE` / `NX_COMPRESS_MODEL` | route the summary (fold) call to a cheaper endpoint/model — e.g. `NX_COMPRESS_SOURCE=local NX_COMPRESS_MODEL=qwen3-4b`. The fold target is used as-is (no capability probes, no switch); unknown names fall back to the active source with a one-time warning |
| `NX_COMPRESS_SUMMARY_TOKENS` | `max_tokens` for the summary call itself (default 8192), clamped window-aware like every other request — the fold call used to send no cap and could be cut off by a small server limit |
| `NX_OVERFLOW_FOLD_TIERS` / `NX_OVERFLOW_FOLD_KEEP_FLOOR` | context-overflow fold ladder: how many escalation rungs a context-length 400 buys (default 3; `0` restores the old single-fold-then-die) and the floor under which the kept tail is never halved further (default 1) |
| `NX_OVERFLOW_EVICT_MIN_CHARS` / `NX_OVERFLOW_EVICT_MAX` | tier-2 spill eviction: the smallest tool result worth replacing with a head/tail + spill-path note mid-run (default 4000 chars) and the per-fold cap on how many are evicted (default 8). Pinned turns are never evicted |
| `NX_TOKENS_MARGIN` / `NX_MIN_COMPLETION_TOKENS` | when the source's context window is known (probed at startup), every request's `max_tokens` is clamped to `window - prompt_estimate - margin` (default margin 2048) instead of sending the flat `NX_MAX_TOKENS` — on small-window local models this stops the request itself from causing the context 400 and the manufactured length cuts; clamped values are floored at `NX_MIN_COMPLETION_TOKENS` (default 256). An unknown window leaves requests untouched |
| `NX_TOKENIZE_PREFLIGHT` | `1` (default) sizes the prompt for that clamp with the server's exact `/tokenize` endpoint on local sources (never a remote round-trip); `0` uses the chars/4 estimate only |
| `NX_FLOOR_BUDGET` | per-turn token-floor budget enforced by the test suite (default 5320, recorded with its justification ledger in `tests/test_initial_context.py`): a new tool or fatter schema fails CI until trimmed or the budget is consciously re-derived — for CI experiments only, not for shipping past the floor |
| `NX_TEMPERATURE` / `NX_TOP_P` | default sampling params for normal (non-recovery) completions: `NX_TEMPERATURE=0` gives deterministic/greedy output (local evals); `NX_TOP_P` only matters when temperature > 0. A negative value (the default) omits the field so the server uses its own default. A per-source `NX_SOURCE_<NAME>_EXTRA_BODY` temperature/top_p wins over these; recovery retries use `NX_RECOVERY_TEMPERATURE`/`NX_RECOVERY_TOP_P` instead |
| `NX_MEMORY_SEARCH_LIMIT` | max `memory_search` results per call (default 8; `0` unbounded) — an over-cap response says how many were cut instead of hiding them |
| `NX_MEMORY_FTS` | `1` (default) keeps a trigram FTS5 index per memories dir (`.memindex.sqlite`, a self-healing cache reconciled by mtime on every search) so `memory_search` answers from the index instead of reading every note; `0` always runs the plain substring scan. The index only widens recall a little (whitespace folding between query characters) and never changes ranking or the result cap; a sqlite build without FTS5 falls back silently |
| `NX_LESSONS_LIMIT` | max `recall_lessons` results per call (default `5`; `0` unbounded) — the BM25 cross-session recall over past session transcripts shares memory_search's bounded-results contract |
| `NX_MEMORY_OPTIMIZE_BATCH` | notes reviewed per librarian call by `/memory optimize` (default `0` = one pass over the whole bounded store; set a number to split it for a tiny-context model — split reviews carry the previous batch's unactioned titles so straddling near-duplicates still meet) |
| `NX_MEMORY_OPTIMIZE_MAX_NOTES` | most notes one `/memory optimize` run reviews — the **oldest** first, so the best retirement candidates are always in view while the newest notes (the ones the session-start note shows) stay visible (default 120; `0` = all); the unreviewed remainder is reported, never silently skipped |
| `NX_MEMORY_OPTIMIZE_MAX_TOKENS` | completion budget for each `/memory optimize` model call (default 36K; `0` = don't send `max_tokens`, server default applies) — sized so reasoning models don't spend a small server cap on thinking and return no answer; an answer that *was* cut off (`finish_reason=length`) is retried once at 3× this budget |
| `NX_MEMORY_OPTIMIZE_TIMEOUT` | per-call timeout in seconds for each `/memory optimize` model call (default 300; `0` = no client-side cap) — a non-streaming reasoning call can legitimately think for minutes before its first byte |
| `NX_MEMORY_OPTIMIZE_MERGE_WORKERS` | merge round-trips `/memory optimize` runs at once (default `0` = one per merge group, capped at 4; `1` = serial) — the groups are independent calls, so concurrency makes the run's wall clock its slowest merge rather than the sum of them |
| `NX_MEMORY_ARCHIVE_KEEP_RUNS` | archived run dirs kept per memories dir by `/memory optimize` (default `0` = keep every run). The archive is the safety net that makes a bad librarian plan reversible (`/memory archived` restores one), so the default never prunes it; set a number on a long-lived store to bound the disk retired notes occupy. The prune is region-bounded — it can never delete the run it is part of — and reclaims the FTS index's dead pages at the same time |
| `NX_MEMORY_REFLECT_MAX_NOTES` | most notes one `/memory reflect` pass indexes (newest first; default 120; `0` = all) |
| `NX_MEMORY_REFLECT_WORKERS` | write round-trips `/memory reflect` runs at once (default 2, capped at 8) |
| `NX_MEMORY_REFLECT_TIMEOUT` | per-call timeout in seconds for `/memory reflect` model calls (default 600) |
| `NX_VERIFY_CMD` | command auto-run ~20s after the last `.py` edit settles (e.g. `uv run pytest -q`); its pass/fail is injected as a `[Runtime note: …]` when it finishes. Set/clear/force with `/verify [cmd\|off\|run]` |
| `NX_VERIFY_SETTLE_SECS` | quiet period after the last edit before `NX_VERIFY_CMD` fires (default 20) |
| `NX_TESTRUNNER_TIMEOUT` | wall-clock ceiling for one `run_tests` tool call in seconds (default 600; `0` = no cap, a hung suite then wedges the turn). On timeout the result names `run_bash` (which backgrounds) as the investigation path |
| `NX_LINT` | `1` enables the tier-2 `lint_check` tool (default off — 0 schema tokens until enabled; surfaced via `search_tools`) |
| `NX_LINT_CMD` | fixed lint command lint_check runs verbatim (shell-parsed), overriding the built-in per-ecosystem shapes — e.g. `NX_LINT_CMD="ruff check --select I"` |
| `NX_LINT_TIMEOUT` | wall-clock ceiling for one `lint_check` call in seconds (default 120) |
| `NX_PROJECT_TOOLS` | `1` enables project-local tools declared in `.nx/tools/manifest.json` (default `0`). `nx` only reads the manifest while listing tools; it imports a handler module from `.nx/tools/` only after `call_project_tool` validates arguments and, for a `risk: "write"` tool, receives normal user approval. The provider always sees the same two gateway schemas, so project tools do not invalidate the shared tool-schema prompt prefix. |
| `NX_TODO_FILE` | where `todo_write` mirrors its list so fresh sessions pick up the plan (default `.nx/todo.md` under the project root; empty value disables). A leading `# nx todos` header marks it auto-managed |
| `NX_DUCKDB_TIMEOUT` | wall-clock ceiling for one `duckdb` tool call in seconds (default 300; `0` = no cap). On timeout the result names `run_bash` (which backgrounds) as the investigation path for a genuinely long analytical query |
| `NX_DUCKDB_NOTES` | SQL file auto-loaded (`-init`) before every `duckdb` query, so views/macros/types from earlier sessions are present without re-defining them (default `.nx-duckdb-notes.sql` under the project root; missing file = no prologue). The model appends reusable SQL to it with write_file/edit_file — the session-memory idea from duckdb-skills, as a plain file |
| `NX_GITCOMMIT_TIMEOUT` | wall-clock ceiling for one `git_commit` call (the stage + diff + commit sequence) in seconds (default 60) |
| `NX_TURN_RETRIES` | step-level retries for a mid-stream timeout (the server stopped delivering chunks after the stream began — the one failure the HTTP layer's `NX_HTTP_RETRIES` cannot replay). Same backoff curve and default as the HTTP layer (default 2 beyond the first attempt; `0` restores fail-fast). When the ladder is exhausted the turn reports a real error: batch scores it `error`/exit 4, never `done` |
| `NX_MAX_TOKENS` | token cap for streaming requests (default 16000; `0` omits it) |
| `NX_MAX_CONTINUATIONS` | auto-continues a reply that ended with `finish_reason="length"` (the token cap cut it mid-answer): the partial text becomes a real assistant turn, a resume note is appended, and the stream re-opens (default 2; `0` keeps the old single-turn behavior and marks the partial with a truncation marker) |
| `NX_MAX_MODEL_TURNS` | cap on tool-call turns per run (default 500) |
| `NX_TASK_TIMEOUT_SECONDS` | `nx run` per-task wall-clock cap in seconds (default 0 = off; `--timeout` or the task's `timeout_seconds` field override) — a task that exceeds it stops at the next model-turn boundary and reports `status: "error"`, `timed_out: true` |
| `NX_METRICS_URL` | optional endpoint to push cumulative token usage after each turn |
| `NX_HTTP_RETRIES` / `NX_HTTP_BACKOFF_BASE` / `NX_HTTP_TIMEOUT_CONNECT` / `NX_HTTP_TIMEOUT_READ` | transient-failure retries for every endpoint call: 429/5xx and read/write timeouts are retried up to N times beyond the first attempt (default 2; `0` off) with exponential backoff (base seconds, default 0.5 → 0.5s, 1s, 2s…) or the server's `Retry-After` when it sends one (capped at 30s). Never retried: refused connections (a cold-start fails fast) and anything after the first streamed byte. Per-request timeouts in seconds: connect 10 (refused/unreachable fails fast), read 900 (a slow server may need more — a read-timeout retry notice names this variable); `0` = no cap. Background probes never print these notices and drop their remaining retries once a `/source` switch supersedes them (`NX_PROBE_QUIET`) |
| `NX_SKIP_STARTUP_PROBES` | skip every startup probe — `resolve_model()`'s `/v1/models` + `/props` round-trip, the capability/thinking fingerprint probes, the context-window metadata probes and the over-max_tokens overrun probe (default off; `1` disables). Startup becomes instant but model autodetection is off: set the model explicitly (`NX_MODEL` / `NX_SOURCE_<NAME>_MODEL`); tool-calling falls back to reactive text-protocol parsing until a turn reveals better. Also what keeps `import nx` network-free in test runs |
| `NX_PROBE_QUIET` | keep background probes' transient-retry notices out of the terminal (default `1`; `0` restores them when debugging a probe that keeps failing). Only stderr is silenced — probe retries still reach the traffic log. Probes also drop their remaining retries as soon as a `/source` switch supersedes them, regardless of this setting: before 0.53.2 a dead endpoint kept emitting retry notices naming the model you had just switched off, and slept out the whole backoff curve while holding the startup-probe lock, so the source you switched to stayed unprobed behind it |
| `NX_BACKEND` | `vllm` to disable llama.cpp-only recovery samplers (min_p / repeat_penalty / DRY) that vLLM rejects |
| `NX_NATIVE_TOOLS` | native `tool_calls` posture (0.54.0 flip): unset / `1` / `auto` (the default) sends `tool_choice="auto"` — the server MAY return structured `tool_calls`, and text-protocol models are unaffected (`auto` permits, never forces; malformed native args keep the full repair/salvage/breaker ladder, and `/toolcalls text` pins `"none"` for the session). `0` restores the pre-0.54 `tool_choice="none"` posture — text-protocol only — for a source/model whose native arguments are unusable; a reasoning-stall under it re-arms `auto` once per episode with a visible line |
| `NX_PARALLEL_TOOL_CALLS` | attach `"parallel_tool_calls": true` to requests that ride native tool calls (0.103.2, default on; `0` disables): sent only when tools are attached with `tool_choice="auto"` AND the source has already confirmed native `tool_calls` — it tells the server it may return several calls in one turn, which some strict OpenAI-compat backends (vLLM, proxies) otherwise serialize one-per-turn. Non-consumers ignore the field; the SYSTEM prompt also steers the model to batch independent read-only lookups into one turn |
| `NX_SPECULATIVE_READS` | dispatch cacheable read-only tool calls while the model is still streaming them (0.103.3, default on; `0` disables): once a streamed call's name is in the read-only cacheable set (`read_file` / `list_dir` / `search_files` / `git_status` / `git_diff` / `git_log` / `memory_search`) AND its accumulated arguments parse as a balanced JSON object, the call runs immediately on a small background pool through `run_tool()` — same gates (plan mode, pre-hook veto, both hooks) and the same cache insert the post-stream prefetch does — so the real dispatch after the stream closes is an instant cache hit instead of a cold read (native `tool_calls` deltas and closed text-protocol tags both count). Only approval-free read-only tools are ever touched; arguments that fail to parse, or parse to a non-object, are skipped; a sub-agent's stream never warms the parent's cache; and a speculative run that fails is invisible — the real dispatch re-runs the tool and surfaces the error itself. Repeat-call stub accounting is unchanged: the speculative insert carries `hits=0`, so the stub still fires on the same real repeat it fires on behind a prefetch |
| `NX_ASYNC_SAVE` | write per-step session autosaves from a background thread instead of inline on the turn loop (0.103.4, default on; `0` restores the exact synchronous behavior): the turn loop's `on_step()` used to pay a full-transcript store-form copy plus a SQLite read-merge-write and post-write re-read after EVERY model step, which is pure latency between a tool result and the next request. Now it snapshots the transcript and hands it to one daemon writer thread (`nx/_session_save.py`), queue depth 1 with coalescing — a snapshot not yet picked up is REPLACED by the newest, so a burst of autosaves collapses into one write and the newest state always wins. Reads stay honest: `_load_session` / `_list_sessions` / `_delete_session` and an inline `_write_session` drain a pending save first, as do `/quit`, `/save`, `/fork`, `/resume`, `/reset` and `/compress`, plus an `atexit` hook — the only regression is a `SIGKILL` in the (sub-second) window between a step ending and its write landing, which can lose the last snapshot; `0` if you need every turn durable before the next request is built |
| `NX_ASYNC_COMPRESS` | run the auto-compress fold on a background thread instead of blocking the turn loop (0.103.5, default on; `0` restores the synchronous fold): when context usage crosses `NX_AUTOCOMPRESS_PERCENT`, the summary model call used to run inline between turns — a full round-trip against the largest prompt of the session. Now the crossing starts the fold over a private transcript snapshot and returns immediately; the finished fold is applied on the turn thread at the next threshold check, and only if the foldable region still fingerprint-matches the snapshot (a changed region discards the fold and folds synchronously instead, so a threshold crossing is never dropped and a stale summary is never spliced). Only one fold in flight at a time; the context-400 overflow ladder waits up to 2s for an almost-done fold before spending its synchronous tier; the append-only digest reuse (0.47) is untouched |
| `NX_MALFORMED_TOOL_BREAKER` | malformed-args circuit breaker: after N malformed/truncated `tool_calls` argument payloads naming the SAME tool (default 3), that tool is dropped from the tool list and the retry nudge steers the model to a sibling tool instead — an endpoint that keeps truncating one tool's args (observed: a local server dying right before `multi_edit`'s `[` array literal) stops being offered it. 0.56.1 removed one real cause of that truncation: the SSE reader split frames with `str.splitlines()` semantics, so a tool-call delta carrying a character JSON allows raw but Python treats as a line break (U+0085, U+2028, U+2029 — all plausible inside file content an edit tool echoes back) cut one frame in half, dropping that delta while the stream itself ended cleanly at `finish_reason="stop"`. The reader now splits on SSE's own terminators, and the accompanying `⚠ malformed SSE frame(s) dropped` notice actually fires again (it read a counter the logging wrapper never exposed, so it always reported 0). If a breaker trip predates that fix, `/toolcalls reset` re-arms the tool. Since 0.29.9 the breaker DECAYS: one clean execution of the tool (its args parsed and the body ran without error) clears the streak and re-arms it, so two flaky truncations no longer cost the tool for the whole session; since 0.40 it is also scoped to the active `(source, model)` — switching endpoints (including a `/source <name> <model>` override) drops the previous endpoint's streaks and re-offers everything, because truncation is evidence about a server, not about a tool; `0` disables; `/toolcalls` shows the streaks, `/toolcalls reset [tool]` clears them, `/reset` re-arms everything |
| `NX_GRAMMAR` | grammar-constrain text-protocol tool calls on llama.cpp servers (opt-in; see below) |
| `NX_CACHE_ROUTING` | attach a stable per-session routing hint to every request — `user` (plus `prompt_cache_key` in the body) derived from the session id, so a vLLM / SGLang / llama.cpp prefix-cache router can pin this session to a replica or KV block instead of cold-prefilling on a different worker each turn (default on; `0` disables). Constant within a session by construction, and a source that sets its own `user` / `prompt_cache_key` via `EXTRA_BODY` always wins. Per-source: `NX_SOURCE_<NAME>_CACHE_ROUTING` |
| `NX_SOURCE_<NAME>_CACHE_ROUTING` | per-source override of `NX_CACHE_ROUTING` (e.g. `0` for an aggregator that rejects unknown body fields) |
| `NX_SEARCH_EXCLUDE_DIRS` | space-separated dirs the `grep` fallback skips (mirrors ripgrep's `.gitignore` handling; default includes `.venv node_modules __pycache__ build dist`) |
| `NX_<ROLE>_COLOR` | override one of eleven semantic roles (see below) |

`nx --help` covers flags; the source (grep `_env_int`/`_env_float` calls in
`nx/`) has the full env var list — dozens more knobs tune recovery samplers,
stall detection, undo depth, log size, etc., all with sane defaults.

When repeated reasoning-only stalls exhaust the rescue ladder and nx forces a
final answer, the forced turn is a nudge rather than a straitjacket: a
`final_answer` call still ends the turn, but if the model emits a real tool
call instead, nx runs it (through the normal approval gate) and keeps the
turn going — the counter stays pinned at the last rung so any fresh stall
escalates straight back to forced-final instead of re-climbing the ladder.

Before any of that ladder climbs, nx checks the one posture that makes every
rung unsatisfiable: a stall (or an empty turn) on a request that went out
with `tool_choice="none"` — since 0.54.0 only possible when native tool
calls were explicitly disabled (`NX_NATIVE_TOOLS=0`) — means the server may
be suppressing the model's only action channel. Some models (glm / qwen
generations behind vLLM) never fall back to the text-protocol tags when
native calls are off; they think, emit nothing, and every "act now" nudge
is unanswerable. The retry re-opens the stream with `tool_choice="auto"` (a
`↳ retrying with native tool-calling enabled` line says so), which either
un-stalls the model or changes nothing for a text-protocol model — `auto`
permits native calls, it doesn't force them. A `/toolcalls text` override is
never escalated past (you chose it); the flag clears on a successful tool
call or a settled turn, and the next stall re-derives it from its own
evidence.

Episode awareness also shapes the mid-ladder nudge: once a real tool call has
run since the first stall, a fresh stall is read as the model stalling while
composing its wrap-up — nx asks it to deliver the visible answer
(`STALL_WRAPUP_NUDGE`) instead of demanding another tool call, and a
text-only reply that arrives after any tool work in the episode is accepted
as the final answer rather than re-nudged (the old layout-only check missed
the acts-then-stalls-then-summarizes shape and fired a spurious
"TEXT-ONLY AFTER STALL NUDGE" on legitimate finished answers).

Every rescue is injected append-only. A nudge is always a NEW trailing
`[Runtime note:]` user message, never a rewrite of a message the model has
already been sent, so a recovery storm leaves the server's prompt-prefix cache
intact (`cached:` in the footer stays green) on exactly the turns that retry
most. Repeated nudges stack, bounded by each failure family's own retry limit,
which also makes the transcript an honest record of how many rescues fired.
Injected notes do not count as conversation turns: the dangling-tool-result
check looks past them, and `/compress` widens its verbatim keep-window so a
trailing note cannot fold away your last real tool result.

### Themes

Each theme supplies thirteen palette hues — `cyan`, `green`, `yellow`, `red`,
`magenta`, `gray`, `blue`, `orange`, `purple`, `teal`, `pink`, `brown`, and
`slate` —
that eleven semantic roles draw from — plus a twelfth, `SUBAGENT`, which shares
the `teal` hue. All twelve slots are used: the roles take
the original eight, and the say-panel's markdown + code rendering maps the
rest (purple bullets/numbers, teal table borders, brown link URLs and
docstrings, pink decorators):

| role | default hue | what it colors |
| --- | --- | --- |
| `TEXT` | green | the model's final answer text |
| `REASONING_TEXT` | gray | reasoning/thinking block text |
| `STATUS_BAR` | cyan | per-turn stats footer (tok/s, ctx, ttft, wall) |
| `TOOL_OUTPUT` | slate | tool/bash result body lines (pin back to magenta with `NX_TOOL_OUTPUT_COLOR=magenta`) |
| `ACCENT` | blue | borders, model name, chatbox prompt |
| `MEMORY` | purple | memory tool calls (`memory_search`/`memory_write`/`memory_delete`/`recall_lessons`) — box rail and header |
| `SUBAGENT` | teal | `spawn_agent` tool call boxes — delegated sub-agent work |
| `LABEL` | orange | source name / session id badges |
| `TODO` | orange | `todo_write` list items — **pending** items; **done** items use `SUCCESS` (green), **in_progress** items use `WARNING` (yellow) |
| `SUCCESS` | green | yolo/auto-approve/allow indicators |
| `WARNING` | yellow | medium-risk / approval-prompt / recoverable-cut notices |
| `ERROR` | red | hard failures |

`/theme` switches themes live and saves `NX_THEME` to `~/.env`. Any single
role can be repinned independently of the theme with
`NX_<ROLE>_COLOR=<hue|hex>` — the value is either a palette key (one of the
twelve hues above, resolved against the *active* theme so it tracks
`/theme` switches) or a literal 6-digit hex (theme-independent). For example,
`NX_ACCENT_COLOR=magenta` makes borders/prompt magenta in every theme, while
`NX_TOOL_OUTPUT_COLOR=#fdfd96` pins tool output to an exact hex regardless of
theme. Invalid values are ignored with a warning.

Tool dump bodies are layered for legibility: the first line (and any
ERROR/DENIED block) renders at the full `TOOL_OUTPUT` hue, while continuing
body lines render a fainter luma-ramped step of the same hue, so streamed
chatter recedes while the parts you scan for stay strong. The ramp is derived
from the active theme — no configuration needed, and `NX_TOOL_OUTPUT_COLOR`
still pins the base hue the ramp starts from.

On truecolor terminals, tool-call box borders additionally carry a faint
same-hue *background fill* (the accent pre-blended at ~16% over the terminal
background — truecolor SGR has no alpha channel, so the blend is baked into
the RGB), the same large-block declutter Catppuccin's style guide uses for
selection backgrounds. Off by default since 0.142.4; opt in with
`NX_BG_TINT=1`. It is off on 256-color
or non-tty output, where a baked fill would quantize badly.

The assistant's final answer is rendered as markdown through rich; its
heading/code/link/bullet colors come from the same active theme (headings and
links use the `ACCENT` hue, code uses `TEXT`, bullets use `LABEL`), so
`/theme` and the `NX_<ROLE>_COLOR` overrides restyle the markdown panel too —
it is not a separate fixed palette.

`frappe` is Catppuccin Frappé — the medium-dark sibling of the default `mocha`
(dark) and `latte` (light) flavors, with all twelve hues taken from the
official flavor palette.

`amethyst` is a monochrome purple theme: every hue is a violet/orchid/pink
shade, so the ten roles stay separable by lightness and by position in the
violet→pink band instead of by hue (the four extra hues follow the same rule:
`purple`/`teal` sit violet-side, `pink` pushes past magenta, and `brown` is a
muted mauve). `gray` (`REASONING_TEXT`) deliberately
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
  attaching the grammar on every turn. The grammar constrains the same
  text-protocol shape the prompt teaches
  (`<name>tool</name><arguments><arg>value</arg></arguments>`), so constrained
  output is the shape the parser already reads; a required argument becomes a
  tag the model cannot omit.
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

Since 0.54.0 nx sends `tools=` with `tool_choice="auto"` on every normal
turn: a server MAY hand back structured `tool_calls`, and a model that
prefers the prompt-taught text protocol still emits `to-text` tags as
`content`, which nx parses and repairs exactly as before. (`auto` permits
native calls, it doesn't force them.) The pre-0.54 default was
`tool_choice="none"` — text-protocol only, opted into native with
`NX_NATIVE_TOOLS=1` — but that posture was fatal for hybrid reasoning models
behind vLLM (glm-5.3, qwen3.8-flash-next): they only ever act through the
native tool_calls machinery, so `"none"` didn't fall back to the text
protocol, it left them with no action path at all, and a fresh session's
first prompt (which can beat the startup capability probe) burned its whole
reasoning-stall ladder until the forced-final salvage printed the model's
thinking as the answer. Malformed native arguments keep their full recovery
ladder (arg repair → truncated-args salvage → per-tool circuit breaker → the
`/toolcalls text` escape hatch, which pins `"none"` for the session), and
`NX_NATIVE_TOOLS=0` restores the old text-protocol-only posture if a
source's native args are unusable.

A stream that dies while tool-call args are mid-flight (a collapse cut, a
server-side abort) is recovered as a malformed-args turn — the buffered args
are parsed by the normal repair ladder and nudged like any other malformed
call — never as an empty turn or a reasoning stall, so the "EMPTY TURN" rescue
never fires for a model that was visibly mid-call (the `generating tool call
… args N chars` progress line).

Qwen-style chat templates (qwen3.8-27b, Qwen3.8-flash-next, …) interleave a
literal newline as a `content` delta between their native `tool_calls`; nx
treats those whitespace-only separators as stream punctuation rather than
prose, so a multi-call turn renders its preamble once instead of once per
call.

## Commands

Run `/help` in the REPL for the full list. The essentials:

<!-- BEGIN GENERATED COMMANDS (nx --dump-commands; do not edit by hand) -->
| command | what it does |
| --- | --- |
| `/help` | list all slash commands |
| `/init` | generate/update AGENTS.md for this project |
| `/yolo` | toggle auto-approve for writes and bash |
| `/approval [level]` | show or set risk threshold (all/low/medium/high/yolo) |
| `/sessions [n]` | list recent sessions, or show one in full |
| `/export [md] [path]` | write the current session to a markdown file |
| `/resume [target]` | resume a past session (n/id/prefix/title) |
| `/save [title]` | save the current session (optional custom title) |
| `/fork [title]` | branch the conversation into a new session (original kept) |
| `/delete [target]` | delete a saved session |
| `/reset` | clear conversation, start a fresh session |
| `/quit` | exit |
| `/context [all\|pin\|unpin\|clear]` | context breakdown by message; pin/unpin turns against /compress |
| `/cost` | session usage + per-source cost scoreboard |
| `/status` | snapshot: source, model, caps, budgets, fence, approval |
| `/memory [save\|remember\|delete\|list\|archived\|optimize\|reflect]` | manage persistent memory |
| `/retro` | distill this session's friction (usage.jsonl) into gotcha memories |
| `/todo` | show the model's current todo_write list |
| `/wip [note]` | snapshot work state (branch, dirty files, todos, note) into the session |
| `/compress` | summarize older turns into one, keep last 2 verbatim |
| `/autocompress [pct\|off\|on]` | show or set the auto-compress threshold |
| `/recover [note]` | force a low-temp visible checkpoint after a bad stream |
| `/source [name] [model]` | list sources, switch to one, or override its model |
| `/provider [source] [order\|off]` | show or set OpenRouter provider-routing order |
| `/budget [tokens\|off]` | show or set the reasoning token budget |
| `/toolcalls [text\|native\|auto] [reset [tool]]` | tool-calling mode + breaker: text|native|auto, reset [tool], or bare to show |
| `/theme [name]` | list color themes, or switch to one |
| `/resources` | show this nx process's CPU% and RSS memory |
| `/tabs` | live dashboard of running surfaces (main turn, bg jobs, agents); Ctrl-T |
| `/panel` | toggle the right status rail on the live region |
| `/reasoning on\|off\|last` | on|off|last — toggle or show reasoning trace |
| `/tooloutput on\|off\|last` | on|off|last — toggle or show tool-call result |
| `/bg [kill <pid>\|all]` | list this session's background run_bash jobs and sub-agents, or /bg kill <pid>; /bg all lists every session's |
| `/diff [path]` | full-screen working-tree diff review; a commits via the reviewed git flow |
| `/review [focus]` | scoped reviewer over the current work: ranked findings, no fixes (Ctrl+R) |
| `/build [issue]` | spec-driven pipeline: spec → parallel implementers → critique → fixes, on a feature branch |
| `/plan [goal\|done\|off\|discard]` | start/view/exit plan mode: /plan <goal>, /plan, /plan done|off|discard |
| `/verify [cmd\|off\|run]` | set / clear / force-run the post-edit verify command (NX_VERIFY_CMD) |
| `/undo [--list]` | revert the last write_file/edit_file/multi_edit |
<!-- END GENERATED COMMANDS -->

### Keyboard shortcuts

In the REPL input box:

- **F1** opens `/help`, **Ctrl+L** runs `/clear`, **Ctrl+Q** runs `/exit`,
  **Ctrl+S** runs `/compress`, **Ctrl+T** opens `/tabs` (single-byte encoding; raw mode disables
  IXON, so the terminal doesn't swallow it as the output-flow stop — the
  old **Ctrl+Alt+C** was byte-split into Esc + Ctrl-C by some terminals
  and quit the REPL instead), **Ctrl+R** runs `/review` (the scoped
  reviewer; readline history-search convention does not apply here — history
  recall here is Up/Down, so the byte is free)
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
  submitted. The dropdown resolves its candidates before each redraw — the
  project tree is walked once per prompt (cached, re-checked on a short TTL)
  and each longer prefix filters the previous candidate set, so typing stays
  sub-millisecond even in a repo with tens of thousands of files. An
  **`@image`** mention attaches the picture itself so the model
  sees it on that turn (`NX_VISION=1` + native tool calls; with vision off the
  message carries a note naming the gate instead). Attaching via mention skips
  the `read_image` tool round-trip entirely and works on any backend that
  accepts image content parts. A **`@doc.pdf`** mention is extracted to
  markdown (headings, lists, and tables preserved) with
  markdown and inlined as text — since 0.100.0 extraction prefers CLI
  binaries on PATH (`mutool convert -F markdown` → `pdftotext -layout` →
  tesseract OCR for scanned pages), falling back to
  [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) — now
  an optional extra (`pip install nx[pdf]`; with uv: `uv tool install nx --with
  pymupdf4llm`) — only when none of those binaries
  exists; a missing toolchain surfaces as an `ERROR:` line in the mention block
  rather than aborting the turn. The page range carried is capped by
  `NX_PDF_MAX_PAGES` and named in the block header.

A plain `input()` fallback is used when stdin/stdout isn't a TTY.

**Interrupt latency (0.56.0).** Pressing **Esc** during a turn, and the moment
a turn hands the prompt back, are now event-driven rather than polled: the
Esc-watcher is woken by a pipe the instant a turn disarms instead of after its
next ~100ms tick, so the gap between the last token and the prompt is
single-digit milliseconds. The optional metrics push (`NX_METRICS_URL`),
session-description generation (`NX_SESSION_DESC_REFRESH`), and the settle after
a short `run_bash` all moved off the REPL thread / off fixed sleeps, so no turn
pays a blocking side-effect after its answer renders.

**Streaming render cost (0.57.0).** A long answer used to get more expensive per
token the longer it grew: the accumulated text was re-copied on every delta, and
the whole answer was ANSI-stripped and re-parsed as Markdown on every rendered
frame. Now the parser keeps the answer as a list of chunks with an
incrementally-stripped mirror (no whole-buffer copy or re-join per chunk), strips
ANSI once per chunk instead of once per frame, scales the render interval with
answer size past the first ~2 KB, and skips the final re-parse when the last
frame already covers every byte received. A 4000-token answer renders in roughly
a quarter of the previous CPU, and a 36x longer answer no longer costs ~600x the
rendered bytes. The SSE reader also stops re-wrapping every JSON object per
attribute read.

**Prefix-cache integrity (0.58.0).** The server's prompt cache keys on the exact
bytes of the prompt prefix, and four paths used to move them mid-session — each
invisible except as a mysterious `cached: 0%`. The memory-awareness note was
rewritten in place at index 1 on every `/memory optimize` and every resume over
a changed store; it now **appends a fresh note at the tail** instead, leaving
every earlier byte untouched. A context-overrun evictor no longer rewrites tool
results that were already sent — oversized results are spilled to disk the
moment they are produced (`NX_TOOL_RESULT_SPILL_MIN_CHARS`, default 102400), so
their wire bytes never change after the fact. A fresh session and a resumed one
now build the head (SYSTEM + the AGENTS.md / memory / orientation / delegation
notes) from one shared producer (`nx/_prefix.py`) instead of two that disagreed,
and the head is **pinned per session** and checked before every request: if a
future change moves it, you get one warning naming the cause instead of a silent
cache miss. Image payloads are likewise dropped the turn after the model has
seen them, rather than riding along at megabytes per request until an overflow
happens to evict them.

**One cached capability manifest (0.59.0).** A handful of costs were paid *per
turn* to re-learn something nx could only learn by asking: the no-tools /
no-extra request-shape ladder re-sent the full prompt (up to three POSTs) after
a 400, even on an endpoint that had rejected that shape an hour earlier; a vLLM
/ SGLang / llama.cpp router got no stable session identity, so every turn could
cold-prefill on a different worker; and `resolve_model()` re-paid its
`/v1/models` + `/props` round trip on every `/source` and resume. All three now
share one small manifest (`nx/_manifest.py`), keyed per `(source, model)`: the
accepted shape is **pinned** the first time it is discovered and reused after
that, so a turn that used to cost up to 3 POSTs costs exactly 1; the routing
hint rides as `user` (plus `prompt_cache_key` in the body) and is **constant for
the whole session** — a moving value would itself be a cache-buster —
opt-out-able with `NX_CACHE_ROUTING=0` (or `NX_SOURCE_<NAME>_CACHE_ROUTING=0`);
and a successful model autodetect is memoized for the session (a *failed* one
still re-probes, so a local server that comes up after nx started is still
found).

**Bounds and durability (0.61.0).** The last phase-3 items were the ones that
only bite after months: nothing a long session accumulated was ever bounded, and
the terminal discipline of the earlier phases had no cap where it fed text back
to the model.
- The **read-only tool cache is an LRU**, bounded by entries
  (`NX_TOOL_CACHE_MAX_ENTRIES`, 512) *and* bytes (`NX_TOOL_CACHE_MAX_BYTES`,
  16 MB); a hit refreshes recency, so eviction drops the cold entries first and
  a 200-turn session's RSS stays flat instead of growing with every read.
- **Cache validity is anchored on everything the result depends on**: a
  multi-path `read_file` is invalidated by *any* of its paths, and a
  `search_files` hit carries a bounded subtree stamp
  (`NX_TOOL_CACHE_TREE_FILES`, 5000) so a new or changed file in a
  subdirectory invalidates it — the root directory's own mtime never moved, so
  the old anchor served that stale hit forever. Committed edits also re-stat
  every anchor, so `sed_edit`/`apply_patch` and any unnamed path are covered.
- **Background sub-agent fan-out is capped** at `NX_SUBAGENT_MAX_CONCURRENT`
  (12); a spawn past the cap reports `queued` and starts when a slot frees
  (`/bg` shows it), instead of every spawn paying its own thread + HTTP stream +
  parser. Finished agent records evict once announced
  (`NX_SUBAGENT_MAX_FINISHED`, 20), so the agent ledger stops holding every
  answer forever — an un-announced answer is never evicted.
- **Undo persistence is amortized**: a burst of edits marks the stack dirty and
  it is written once per turn (alongside the session autosave) rather than
  re-serializing every snapshot on every edit; the persisted file is compact
  JSON.
- **Diffs fed back to the model are capped** at `NX_DIFF_HUNK_MAX_CHARS`
  (8192) with an elision notice, so a one-line change in a repetitive 30k-line
  file no longer ships a 520 KB hunk into context by relying on a later filter.

**Memory audit follow-ups (0.64.0).** Five defects surfaced by reading the
memory subsystem end to end:
- **FTS snippets were degrading recall.** `_memindex.search_dir` selected FTS5's
  `snippet(..., 16)` — token-bounded, so a hit's preview was a roughly 12-char
  fragment where the plain scan returned four body lines. Since the snippet is
  the only body content the model sees, the accelerator was strictly worse than
  the scan it accelerates. The column is gone; the FTS path now builds its
  preview with the same `_memory_snippet` the scan uses, so the two paths agree
  exactly (asserted in the suite).
- **Nothing reclaimed the index or the archive.** Reconcile deleted archived
  notes' rows but SQLite never shrinks the file (a 300 KB store left a 1.4 MB
  index, 76/335 pages free), and `memories/archive/` grew forever.
  `_memindex.compact()` VACUUMs the index opportunistically after an optimize
  run (free pages above 25% of the file), and `NX_MEMORY_ARCHIVE_KEEP_RUNS`
  bounds the archived runs — region-bounded, so a prune can never delete the run
  it is part of (default `0` = keep all).
- **The session-start note read every body.** The title-only scanner path reads
  only each note's head for the title, so the session-start index and
  `memory_delete`'s resolver no longer read the whole store (roughly half the
  cost at 56 notes).
- **`memory_write` approved blind, and its allow rule was dead.** The approval
  prompt now shows a bounded head of the note; and the allowlist tool mapping no
  longer classifies a memory write/delete as a file write (both start with
  `write `/`delete `), so `A` and `/approval allow memory_write` finally work —
  widened to the whole memory-write family, since a byte-count-shaped pattern
  would match exactly one call.
- **`/memory archived [target]`** lists and restores retired notes (the read
  side of the archive's reversible-by-`mv` promise), and a dead
  `_atomic_write_private` wrapper was removed.

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

**Session isolation (0.103.x).** `nx --isolate` (or `NX_SESSION_WORKTREE=1`)
runs a session in its own git worktree under `~/.nx/worktrees/`, branched
`nx/session-<sid>` off HEAD, so two isolated sessions in one repo share
nothing; a per-repo advisory lock (`NX_SESSION_LOCK`) warns a second REPL
sharing the same tree, a cross-session staleness guard refuses writes to files
changed since this session last read them, and the undo store is per-session
(`~/.nx/undo/<session-id>.json`). See the env table for the knobs.

Every chat auto-saves to a single SQLite store, `~/.nx/sessions/sessions.sqlite`
(0600 file in a 0700 dir; 0.101.0 — was one JSON file per session). Each row
holds the full transcript JSON (message array + id/title/description/source/
timestamps + cumulative token usage); listings read the `updated_at` column and
body search uses an FTS5 table in the same database. Auto-save happens after
every turn — since 0.103.4 off the critical path: a snapshot is queued to one
background writer (coalescing to the newest state per session) so the turn
never waits on disk I/O (`NX_ASYNC_SAVE=0` restores the old synchronous
behavior). The trade: a sudden SIGKILL can additionally lose the last
snapshot (normally sub-second); every path that reads the store flushes
pending saves first, so normal exits lose nothing. Saves are transactional
(BEGIN IMMEDIATE + a per-session `revision` counter), so two nx processes
saving the same session merge instead of losing turns.

Sessions dirs written by older nx versions are migrated automatically: per-
session JSON files are imported (lazily, the first time a listing/load finds
the store empty — or eagerly with `nx sessions --migrate`) and left in place as
a backup; pre-0.46 `.index/` shards fold in too, and rows whose transcript row
is gone are pruned on the next save, so a deleted id can't resurface in a
listing or a body search. Deleting `sessions.sqlite` is always safe — the next
start re-imports the backup files.

`nx --resume <target>` (short id, prefix, title, or bare for most recent)
restores it, reselects the original source, and replays a recap of the
conversation. `nx sessions [query]` lists/filters sessions from the shell
without entering the REPL; inside the REPL, `/sessions` and `/resume` do the
same.

Both entry points run the **same** loader (`nx/_resume.py`), so a resume is
identical whichever way you enter — it expands stored `image_ref` payloads back
into the live transcript (a resumed session is request-ready, never a 400 on
the next turn), reselects the recorded source + model + provider routing,
changes back into the directory the session was recorded in (`NX_RESUME_CWD=0`
opts out; `st.PROJECT_ROOT` is recomputed for it), drops the saved system
prompt for the current one, rebuilds the memory/delegation notes, and reloads
the cumulative token counters so the footer keeps climbing across restarts.

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

## Eval corpus

`nx eval run` is the offline regression harness (added 0.53.0, P3-14): a seed
corpus of scripted stub-server scenarios replayed through the *real* `nx run`
batch machinery, each scored against its expectation. The stub serves
OpenAI-style SSE turns from a per-scenario script — tool calls with real
schemas, silent empty turns, a leaked-token collapse, a stream that stops
delivering mid-answer — keyed on the task instruction, so one loopback server
serves the whole batch and no configured endpoint is ever touched.

```
$ nx eval list                 # show the corpus: ids, scripts, expectations
$ nx eval run                  # run + score; exit 0 only when all pass
$ nx eval run --filter e-timeout --json
```

Flags: `--source NAME` (model-source name to shadow while the stub source is
registered; default shadows the active source), `--out DIR` (artifact
directory, default `<cwd>/.nx/eval`), `--parallel N`, `--filter ID`, `--json`.
Artifacts land in `<out>/`: `tasks.jsonl`, `results.jsonl` (verbatim harness
lines), `summary.json` (scored), and `workspace/` — a seeded directory the
tasks run inside with `--yolo` (headless), so their real tool effects (the
corpus includes real `write_file`/`edit_file` calls) stay confined there.

Scoring pins the harness and recovery contracts, not model quality: a task
that recovers from two silent empty turns must still finish `done` with its
recovery answer, a collapsed stream must cut and recover, and a mid-stream
read timeout must score `status: "error"` naming the timeout — never `done`
(the 0.51.0 honest-timeout contract, enforced here end to end). The seed's
first catch was nx's own: the vendored client disabled every httpx timeout by
passing `timeout=None` into `build_request` (httpx reads an explicit `None`
as "no timeout"), so `NX_HTTP_TIMEOUT_READ` never applied to real requests.

## Tools

Read-only tools never need approval; state-touching tools (write/edit/bash)
go through the approval gate above.

### Project tools

Set `NX_PROJECT_TOOLS=1` to expose the stable `search_project_tools` and
`call_project_tool` gateways. Define project-local handlers in `.nx/tools/`
and declare them in `.nx/tools/manifest.json`; listing tools reads only this
JSON file, while calling a tool imports its module on demand. Project code is
therefore trusted code: enable this only in repositories whose `.nx/tools/`
contents you trust.

```json
{
  "deploy_preview": {
    "module": "preview",
    "handler": "deploy_preview",
    "description": "Create a staging deployment preview.",
    "parameters": {
      "type": "object",
      "properties": {
        "environment": {"type": "string", "enum": ["staging"]}
      },
      "required": ["environment"],
      "additionalProperties": false
    },
    "risk": "write"
  }
}
```

The corresponding `.nx/tools/preview.py` exports
`def deploy_preview(environment): ...`. `risk` is `"read"` or `"write"`
(default: `"write"`); write tools use nx's normal approval prompt and are
blocked during plan mode. Their individual schemas stay in tool results rather
than the provider's `tools=` request, preserving the stable schema prefix.

| tool | args | notes |
| --- | --- | --- |
| `read_file` | `path`, `offset?`, `limit?` | cat -n style (1-based); `path` may be a list; streams — only the window is held in memory; limit capped by `NX_READ_FILE_MAX_LINES` (default 25000); `.pdf` paths extract to markdown instead of erroring |
| `write_file` | `path`, `content` | overwrites; requires confirmation; atomic (temp+rename), preserves the file's permission bits and writes through symlinks to the real file |
| `edit_file` | `path`, `old`, `new`, `replace_all?`, `old_lines?` | `old` must match exactly once (or `replace_all`); `old_lines="10-14"` replaces a 1-based line range instead; unique indentation-only mismatches apply whitespace-tolerantly; `new` passes the garble guard (see `NX_GARBLE_GUARD`) |
| `multi_edit` | `path`, `edits` | batch of `{old, new}`, one confirmation; atomic — a garbled or unmatched edit rolls back the whole batch; each `new` passes the garble guard |
| `delete_file` | `path` | delete one file; restorable via `/undo` (unlike `rm` via bash); fence-checked like the other writers (see `NX_WRITE_FENCE`) |
| `sed_edit` | `pattern`, `replacement`, `path` | regex find/replace over a file or glob; every matched file must sit inside a writable root (see `NX_WRITE_FENCE`) |
| `apply_patch` | `diff`, `path?` | apply a unified diff; for multi-file restructure; each `+++` target is fence-checked against the writable roots (see `NX_WRITE_FENCE`) |
| `list_dir` | `path`, `pattern?` | **retired from the offered schema (0.129.8)** — still dispatches for direct callers (@dir mentions, tests, scripts); models list directories with `run_bash ls` (which also gives filters/counts/metadata the flat listing never had). Emits an actionable refusal pointing at `ls`/`search_files` when a model calls it |
| `search_files` | `pattern`, `path?`, `glob?`, … | regex content search; read-only |
| `git_status` | `path?` | `git status --short -b`; read-only. `path` is a directory (repo), never a file |
| `git_diff` | `path?`, `staged?`, `file?` | `git diff`; read-only. `path` is a directory; pass the target file as `file` (absolute paths ok) |
| `git_log` | `path?`, `max_count?`, `file?` | `git log --oneline`; read-only. `path` is a directory; pass the target file as `file` (absolute paths ok) |
| `run_bash` | `command`, `timeout?` | requires confirmation; backgrounds long-running commands. A command whose heredoc delimiter is a quoting idiom (e.g. the ssh quote-embedding dance fused onto `<< 'EOF'`) is refused pre-flight with a diagnosis naming the fused spelling — the shell would otherwise die at parse time with the exact signature of the old exit-trailer bug |
| `wait_background` | `pid`, `timeout?` | only offered after a `run_bash` call has backgrounded; waits for the PID and returns its output (a caller-supplied `log_path` is ignored — the path is resolved from the PID) |
| `run_tests` | `targets?` | detect the suite (pytest/go/cargo/npm via marker files) and run the fixed per-ecosystem command, returning a failure-focused summary. See below |
| `lint_check` | `paths?` | tier-2 (`NX_LINT=1`): run the detected linter (ruff + ty for python, go vet, cargo check, eslint) read-only over the given files — or, with no paths, the files edited this session (from the undo stack). Fixed command shapes, never free-text shell, never writes, no approval; arbitrary lint flags via `run_bash` or `NX_LINT_CMD` |
| `duckdb` | `sql`, `database?`, `format?` | run analytical SQL with the DuckDB CLI (`duckdb` must be on PATH; install via `curl https://install.duckdb.org \| sh`). Reads CSV/Parquet/JSON directly (`SELECT * FROM 'data/*.parquet'`) — the preferred aggregation path over pandas scripts. Read-only SQL auto-runs; mutating SQL (CREATE/INSERT/COPY/ATTACH/INSTALL and friends) confirms first with a local medium verdict; plan mode gates it; large results fold and spill for `read_file`; `NX_DUCKDB_NOTES` is auto-loaded as a prologue |
| `git_commit` | `files`, `message`, `path?` | stage `files` (space-separated; `.` = every change) and commit `message` after the user reviews the staged diff. See below |
| `obs_recall` | `id`, `offset?` | page back the archived original of a large tool result that was replaced by an observation handle (`NX_OBS=1`; the handle stub names its `obs_...` id). Returns whole lines plus `next_offset`; pass it back in until `eof=true`. Byte-exact: the archive is what the model originally saw |
| `recall_lessons` | `query` | BM25 recall over PAST session transcripts (FTS5 over the session store), always-on; surfaces what a previous chat already learned. Read-only, no approval |
| `final_answer` | `answer` | always-on: end the turn with a final answer instead of leaving the model hanging; also the forced answer when a budget is exhausted |
| `plan_edit` | `old?`, `new`, `old_lines?`, `write?` | splice/replace the current plan file (`.nx/<slug>.md`); offered only while plan mode is active |
| `search_tools` | `pattern?` | list tools withheld this turn (gated/tier-2) with their full schemas, so one can be called next turn — read-only, always offered. This is the tier-2 mechanism (0.62.0): a tool that doesn't need to be always-on costs 0 tokens until the model asks for it here |
| `search_project_tools` | `pattern?` | list enabled project-local tools from `.nx/tools/manifest.json`, including each exact argument schema, without importing the Python handler. Enable with `NX_PROJECT_TOOLS=1` |
| `call_project_tool` | `name`, `arguments` | validate and run one tool returned by `search_project_tools`. The manifest's `risk: "write"` tools require normal approval; `risk: "read"` tools are trusted as read-only declarations |
| `web_fetch` | `query?`, `url?` | keyless web tool (tier-1 always-on; `NX_WEBFETCH=0` removes it): exactly one of `query` (web search via SearXNG when `NX_SEARX_URL` is set, else DuckDuckGo-lite; up to 10 title/URL/snippet results) or `url` (page fetched GET-only and stripped to readable text, links as `[text](url)`, truncated at `NX_WEBFETCH_MAX_BYTES`). Cached globally for 7 days under `<NX_HOME>/web-cache/` (stale pages revalidated via ETag/Last-Modified), per-host + search throttled, robots.txt honored (off by default under SearXNG), and otherwise polite to the free endpoints; see the `NX_WEBFETCH` entry under Configuration |
| `todo_write` | `todos` | rewrites whole list `[{text, status}]`; no per-item id |
| `memory_write` | `title`, `body`, `scope?`, `kind?` | save a durable memory note (`kind`: `preference` for a durable style/behavior note, `fact` otherwise); requires confirmation |
| `memory_search` | `query`, `scope?` | keyword-search saved memories; title hits rank first; read-only; ends with the absolute dirs holding the shown hits, because the listed `(name)` is a bare filename — `read_file` needs the full path |
| `memory_delete` | `target`, `scope?` | delete one memory by exact title, filename, or unique prefix; requires confirmation |
| `spawn_agent` | `instruction`, `max_turns?`, `async?`, `tier?`, `scope?` | delegate a task or investigation to a sub-agent (`NX_SUBAGENTS=1`); sub-agents can run an allowlisted read-only bash tier — a read-only text/utilities allowlist (rg, cat/head/tail/wc, find, diff, sort/uniq/cut, ps, checksums, etc.; git via the read-only risk verdict) — single bare commands only, no pipes/python. For joining/aggregating/filtering data files (CSV/Parquet/JSON), sub-agents and parents alike should prefer the `duckdb` tool — SQL over files directly, no shell scripting needed. Read-tier agents (the default) change no files; write-capable agents (with `NX_SUBAGENT_WRITE=1`) can edit files — independent multi-file edits are a prime delegation target, not just investigations. The agent sees its turn budget (default and floor 100 for read-tier, 250 for write/implement-tier — a parent's smaller `max_turns` is raised to the floor; hard ceiling 500 — see `NX_SUBAGENT_MAX_TURNS`) and spends its last turn answering, not searching; rescues inside the agent are capped at one per failure family, and a budget mostly spent on rescues gets its final answer forced early; an agent that dies at the cap still returns its last observation. Write-capable only with `NX_SUBAGENT_WRITE=1` (write_file/edit_file/multi_edit/sed_edit, each confirmed by the user with a diff preview, tagged `(sub-agent)`; run sequentially, no delete; nested spawn is opt-in via `NX_SUBAGENT_MAX_NESTED=1`). Send several read-only ones in one turn to run them concurrently — one question per agent; re-spawning an instruction whose background job is still running is refused. Tiers (0.96.0): `tier:"read"` (default — an omitted tier is read, never env-upgraded to write), `tier:"write"` or `tier:"implement"` — write-capable agents, gated by `NX_SUBAGENT_WRITE=1`; `implement` adds a discipline prompt (plan, verify as you go, change approach after two failures, no churning). `tier:"advisor"` (0.112.0) is an executor that runs the task end-to-end and consults the parent model (`consult_advisor` side-call) when a decision exceeds its judgment — see the advisor note below the table; also gated by `NX_SUBAGENT_WRITE=1`. In background mode (the default) write/implement/advisor tiers require a yolo session (`--yolo`); without yolo a background write-tier spawn is downgraded to foreground (0.99.1) — pass `async:false` for the same effect, or run `--yolo` to authorize unattended background writers. `scope` optionally confines the agent's writes to path globs (e.g. `["nx/_turn*.py"]`); out-of-scope writes are refused with an explicit error naming the scope. Background is the default (`NX_SUBAGENT_ASYNC=0` for foreground): a background spawn returns a job id immediately and delivers a short answer head + the path of its full markdown report (`~/.nx/agents/<job>.md`) as a one-time `[Runtime note]` on a later turn — the parent reads the file when the head isn't enough. Foreground returns the answer inline; while foreground spawns are in flight the turn shows a single group line (`2 tool calls in parallel · spawn_agent ×2`) rather than each agent's own boxes and spinners (0.31.0). Async/background agents are always read-only unless a tier is passed, and a failed agent still announces (FAILED note). With `NX_SUBAGENT_WORKTREE=1` (0.97.0) each write-tier spawn gets its own git worktree (see the env table) — its edits, test runs and commits land there, never in the shared tree. `/bg` lists running/finished agents with their report paths. Flash-tier models sometimes narrate delegation instead of emitting the call, or mis-key arguments (`prompt` instead of `instruction`) — both self-correct on retry, but heavy delegation wants a non-flash model |
| `critique` | `focus`, `paths?`, `severity?`, `max_turns?` | the scoped minimal reviewer (0.132.0, `NX_SUBAGENTS=1`): hands the work in progress to a READ-ONLY reviewer sub-agent and returns ranked findings — it cannot fix, edit, or run tests, so its findings are not contaminated by having written the code. `focus` is required (what is being reviewed and against what; a scope-free review returns noise), `paths` confines the reading, `severity` sets the floor a finding must clear to be reported (`critical`/`major`/`minor`/`info`, default `major`). Each finding is one line: `SEVERITY — file:line — what is wrong — fix sketch`, at most 5, or the literal `No findings above the floor.` plus what was checked. Does NOT need `NX_SUBAGENT_WRITE=1` (a reviewer has nothing to approve); gated like `spawn_agent`, so it adds 0 tokens to the floor — the budget in `tests/test_initial_context.py` does not move. Use it after a substantive change or before merging something risky. See the critique note below the table |
| `read_image` | `path` | show a local image to a vision-capable model (`NX_VISION=1` + native tool calls). png/jpeg/gif/webp/bmp by extension, with a magic-byte sniff fallback for missing/unknown extensions; `~` works; refusal (not downscale) past `NX_IMAGE_MAX_BYTES`. The base64 rides the tool message; on a text-protocol turn, in `--trace`, or in batch output it renders as a one-line image-attached hint instead. Saved sessions store a hash reference and the payload once in `<sessions>/_images/` (see `NX_SESSION_IMAGES`); if the image contributed to a context overrun, the next auto-compress evicts the payload and keeps its marker line |

**Advisor tier** (0.112.0): `spawn_agent` with `tier:"advisor"` hands the task to an executor backed by an on-call advisor — the parent model itself. The executor consults the advisor (`consult_advisor`, a one-shot no-tools side-call) when a decision exceeds its judgment, keeps `implementation-notes.md` with a Plan and a Deviations log, and takes the conservative option on any advisor timeout or refusal. Consult budget/timeout: `NX_SUBAGENT_CONSULT_BUDGET` (default 3) / `NX_SUBAGENT_CONSULT_TIMEOUT` (default 60 s). Requires `NX_SUBAGENT_WRITE=1` (the executor edits files); background advisor spawns need a yolo session like the other write tiers.

**Critique tier** (0.132.0): `spawn_agent` with `tier:"critique"` runs the scoped minimal reviewer — the read-only toolset with the `CRITIQUE_SYSTEM` prompt. The `critique` tool above is the ergonomic front door to it (it composes the focus/scope/severity-floor instruction); the tier is reachable directly from `spawn_agent` too. The distinction that makes it work is the restriction: no write tool, no `run_tests`, no nested spawn, so a finding cannot be silently "fixed" before the parent judges it, and the parent keeps the approval gate. It needs no `NX_SUBAGENT_WRITE=1`. Like the advisor tier's `consult_advisor`, the reviewer is a sub-agent rather than a main-loop mode, so its reading never enters the parent's context — only the at-most-5 findings do.

`read_file` results are cached per-path and invalidated on writes
or any `run_bash`. `list_dir` results cache the same way for the direct callers
that still dispatch it.

`run_tests` (0.49.0, ROADMAP P3-5) is the closed-loop verification tier: the
model runs the suite after edits without spending a turn composing a shell
command. Detection is marker-based, and the command shape is fixed per
ecosystem, so the tool never runs free-text shell (that is `run_bash`).
Output is folded to the failures: one block per pytest failure with its
traceback middle elided, plus the FAILED and short-summary lines,
whole-result capped; when anything was folded the raw output is written to
a temp file whose path rides the result for `read_file`. `targets` scopes
the run (tests dir, file, node id). Approval rides the normal gate with a
local risk verdict (medium, no classifier round-trip); plan mode gates it;
`NX_TESTRUNNER_TIMEOUT` (default 600 s) caps the call; write-tier sub-agents
get the tool so an agent that edits can verify its own work, read-only
agents are refused. `NX_VERIFY_CMD` stays the hands-off post-edit loop;
`run_tests` is the model-invoked, on-demand one. A `git_*` call that passes a
file where the `path` directory belongs, or an absolute path as `file`, is
repaired in-flight instead of erroring (the file's directory is used and the
command is restricted to that file). The `git_status`/`git_diff`/`git_log`
cache entries carry a
fingerprint of `.git` state (HEAD/index/`ORIG_HEAD` mtimes) and are dropped by
any edit tool or `run_bash` too — a commit in another terminal, or `git reset`
via `run_bash`, is never served stale. `git_commit` (0.51.0, ROADMAP P3-6)
completes the git surface: it stages the paths the model names (`git add --
<files>`, the only argv shape it can run), shows the **staged** diff (taken
after staging, so it is always exactly what the commit will contain) as the
approval `preview`, and scans that same text for secret-shaped strings
(OpenAI/GitHub/Slack/GitLab tokens, AWS key ids, PEM blocks — the same
patterns `run_bash`'s own guard uses) before any confirmation round-trip. A
hit refuses the commit outright; nothing is committed until the user accepts
the diff they were shown. Denial leaves the changes staged (review with
`git_diff` staged=true) and commits nothing. Plan mode gates it, it joins
the sub-agent write tier, and the "git commit: <subject>" action carries a
local medium risk verdict — the model cannot reach `git commit` unreviewed
anymore. A HEAD-moved preflight (0.108.0) re-checks `git rev-parse HEAD`
immediately before the commit: if another session landed a commit between our
staging and the final `git commit`, ours refuses instead of silently parenting
onto the moved tip, names the commit(s) that landed (`old..new`), and leaves the
shared index untouched for a deliberate re-inspect/retry. `run_bash` stays the
tool for every other git operation.
`wait_background` is omitted from the
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
phantom titles for the model to chase. `memory_search` requires every query
term to match, ranks title hits above body hits, and caps results at
`NX_MEMORY_SEARCH_LIMIT` (default 8) so a large store can't flood a turn
with snippets. Search is served from a trigram FTS5 index per memories dir
(`.memindex.sqlite`, cache-only — NX_MEMORY_FTS=0 or a sqlite build without
FTS5 falls back to reading every note) reconciled by mtime on every search,
so a growing store keeps answering without a full-store read per query.
`/memory list` shows both scopes, each row labelled `[global]`/`[project]`.

### Cross-session recall: `recall_lessons` (0.102.0)

Saved notes are curated; the raw material is every past session transcript.
`recall_lessons` queries the session store's FTS index (see Sessions) with an
FTS5 MATCH and ranks the hits with FTS5's built-in **bm25()** — relevance
over whole transcripts, not title substring: "which earlier session figured
out how to make this test suite pass?" answers in one call. It is read-only
(no approval, cacheable like `memory_search`), always-on, and open to
sub-agents' read tier and plan mode's read-only set. Results are capped at
`NX_LESSONS_LIMIT` (default 5); each hit carries its id, title, a
BM25 score, and a bounded snippet. `scope="project"` narrows hits to
sessions whose recorded `cwd` matches the current project root. Every
failure shape (locked or corrupt store, no FTS5, bad query syntax) degrades
to "no hits" instead of raising into the turn, and the same lazy FTS backfill
the `/sessions <query>` search uses keeps store-only sessions recallable.

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
suffix instead of overwriting, and the collision check is part of the atomic
publication itself, so concurrent writers (memory-merge workers) can't erase
each other's note. `memory_write` and `memory_delete` go through
the approval gate like `write_file`/`delete_file` — the prompt shows a bounded
head of the note so you can see what you are approving, and pressing `A`
blesses the whole `write memory *` / `delete memory *` shape (a byte-count- or
path-shaped pattern would have matched exactly one call, so it was widened for
these two tools); `memory_search` is
read-only. The manual `/memory save|remember|delete|list` slash commands
remain for distilling a whole session at once (project scope, same
convention) and hand-managing notes; `/memory archived [target]` lists the
notes `/memory optimize` retired and restores one, and `/memory optimize`
cleans the store up once it has outgrown itself.

### Note kinds (0.68.0)

`memory_write` accepts an optional `kind`: `preference` (a durable style or
tooling choice), `fact`, `gotcha`, or `episodic` (a one-off project event).
The kind is stored in the note's footer (`_kind:`); an unrecognized or absent
kind stores kind-free and behaves exactly as before. Preference-kind notes get
special treatment at session start: the injected saved-memories note lists
them under "User preferences (always apply)" — newest first, up to 4 —
regardless of how many newer untagged notes exist, so your durable
preferences reach the model every session even in a busy store. Ordinary
titles ride the plain newest-first list as before.

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
notes"). Reviews cover at most `NX_MEMORY_OPTIMIZE_MAX_NOTES` notes
(default 120) and take the OLDEST first — the strongest retirement
candidates — so a store bigger than the cap no longer hides its stale notes;
the newest tail is reported as unreviewed. Run with the cap set to `0`
(all notes) to review everything at once. Each round-trip sends
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

### `/memory reflect`

Where optimize is a janitor, `/memory reflect [focus]` is the mirror: it reads
the whole store from outside (both scopes, capped at
`NX_MEMORY_REFLECT_MAX_NOTES`, default 120) and asks the model one question a
janitor never would — what do these notes know collectively that no single
note says? Its outputs, in a line protocol like optimize's:

- `WRITE scope kind` — at most 3 per run. A preference that keeps
  re-appearing inside project scopes becomes one global `preference` note;
  a pattern shared across projects becomes one cross-project insight note.
  Each write gets its own round-trip over the NOTE'd source notes and goes
  through the shared writer (0600, `_kind:` footer), gated by one approval
  for the whole run. Nothing is archived — reflect only ever adds.
- `PROPOSE ...` — contradictions, stale beliefs, or a preference buried in an
  episodic note are printed as advice and never executed (fixing other
  notes' composition is optimize's territory).

An empty store, or one with no cross-cutting signal, writes nothing and says
so. Tunables: `NX_MEMORY_REFLECT_MAX_NOTES` (default 120, `0` = all),
`NX_MEMORY_REFLECT_WORKERS` (default 2, capped at 8), and
`NX_MEMORY_REFLECT_TIMEOUT` (default 600s — a reasoning model can think for
minutes). Like optimize, reflect is a slash command and not a tool: zero
schema tokens, and the user decides when a reflective pass is worth a
round-trip.

### Self-retro (`/retro`, 0.131.0)

Where optimize tidies notes and reflect reads them, `/retro` reads the
*telemetry*: the session's own `usage.jsonl` (`NX_USAGE_LOG`), which already
records every friction signal the harness produces — arg repairs, malformed-args
breaker trips, withheld-tool gates, retry-ladder attempts, give-ups, and tool
calls whose outcome wasn't `ok`. `collect_friction` groups those by
`(event, tool-or-error_class)`, keeps only patterns with 2+ hits (a single blip
is not a pattern), sorts by count and keeps the top 3. Each surviving pattern
becomes one project-scope `gotcha` memory — the note names the event, the tool,
the count and one sample detail line (the model's last words before the
failure, from the usage stream's `ctx_tail`).

The **note contract** that keeps this safe to run unattended:
each pattern is deduped against the existing project notes by a
tool+event substring scan before anything is written (a coarse check on
purpose — a false skip is cheap, a duplicate cluster is not), and the pass
writes **at most 3 notes**, so a session that repeats itself every turn cannot
grow the store without bound. Writes go through the same shared writer as
every other memory (timestamped slug, `0600`, `_kind: gotcha` footer) with
`gate=False` — the approval gate is replaced by the fact that the note is a
derived counter, not model-authored content.

**It fires automatically.** The retro must not depend on the user remembering
to ask, so the session-end hook is **ON by default**: on a clean exit (`/quit`
or EOF/Ctrl-D), after the final save and flush, `nx/_retro.py`'s
`retro_on_exit` runs the non-interactive pass. `NX_RETRO_ON_EXIT=0` opts out
for users who want silence. The hook is defensive by construction: the env
gate, a module-level `_done` set that makes it run at most once per session id
(so the `/quit` path and an atexit path in one process can't double-write), and
a `try/except` that swallows everything — a retro must never raise into, or
delay, an exit. `/retro` remains the manual entry point: it prints the
friction table and asks y/n once per note before writing.

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
`AGENTS.md` for the module map and design conventions if you're
extending it, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the required touch
points when adding slash commands, tools, or env vars. `CLAUDE.md` is a
symlink to `AGENTS.md`, so Claude Code reads the same instructions; `/init`
regenerates `AGENTS.md` in place

## License

MIT License. See [`LICENSE`](LICENSE).
