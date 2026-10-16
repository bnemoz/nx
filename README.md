# nx (pronounced "annex")

A coding agent that keeps its per-turn floor under ~5K tokens, so a
128K-1M-token context window stays mostly yours. Most harnesses burn 20K-50K+
tokens of system prompt and tool schemas before your first message even lands,
paid on *every turn*; nx holds an **enforced, mechanically-counted budget of
~5K tokens** (6000) — a `SYSTEM` prompt plus always-on tool schemas, nothing
else — while keeping genuinely descriptive schemas. Point it at any
OpenAI-compatible endpoint — a local llama.cpp / vLLM / SGLang server, or a
remote API like OpenRouter, Together, or Z.ai — and get an agent that reads,
writes, edits, and runs shell commands in your project. Every write/edit/bash
call is risk-classified and gated by an approval prompt (unless you pass
`--yolo`). Run `nx --tokens` to see the exact counted floor for your backend.

*Origins:* nx started (v0.1.0) as a fork-inspired experiment based on
[Sentdex/minion](https://github.com/Sentdex/minion).

## Quick start

One-line install (Linux/macOS; bootstraps `uv` if missing, installs `nx` on
your `$PATH`, prints how to upgrade later):

```bash
curl -fsSL https://raw.githubusercontent.com/nmishra93/nx/main/install.sh | sh
```

(Or install from source: `uv tool install git+https://github.com/nmishra93/nx.git`;
developing in a checkout: `uv tool install -e .`.)

**Add an endpoint — first run does it for you**: launch `nx` with no endpoint
configured and a short wizard asks for an API base URL and key, then saves them
to `~/.env` — next launch just works. It also offers web-search setup (an
existing SearXNG instance, or a fresh loopback-only one installed for you — see
[docs/SEARXNG_SETUP.md](docs/SEARXNG_SETUP.md)). Re-run any time with `nx setup`.

Prefer to configure by hand? `nx` reads credentials from `~/.env`
(auto-loaded at startup — no config file):

```bash
export NX_BASE_URL=http://localhost:8080/v1
export NX_MODEL=your-model-name
export NX_API_KEY=sk-noop                    # any string; local servers ignore it
python -c "import nx; nx.main()"
```

If `NX_MODEL` is unset, nx asks the server what it's serving. Then just run:

```bash
nx
```

## Worth knowing first

| flag / env var | what it does |
| --- | --- |
| `--yolo` | skip all approval prompts — every tool call runs without asking |
| `--approval <level>` | approval gate: `all` (default)/`low`/`medium`/`high`/`yolo` |
| `NX_APPROVAL` | persistent default approval mode (`/approval set-default <level>` rewrites it) |
| `--resume [target]` / `-r` | resume a saved session; bare = most recent (sessions auto-save) |
| `--prompt <text>` / `-p` | one-shot with an inline prompt: run one turn and exit |
| `--source <name>` | start on a specific named model source |
| `NX_BASE_URL` / `NX_MODEL` / `NX_API_KEY` | endpoint, model id, and key for the default source |
| `NX_SOURCES` / `NX_SOURCE_*` | named multi-source endpoints (comma-separated names + per-source vars) |
| `NX_ENV_FILE` | override the env file read at startup (default `~/.env`) |

## Headline capabilities

- **Risk-gated approvals** — every write/edit/bash call is risk-classified
  (`low`/`medium`/`high`) and prompted, with persistent allow rules
  (`/approval allow list` to review).
- **Sessions & `--resume`** — every session auto-saves; resume the most recent
  or a named one.
- **Multi-source switching** — define several OpenAI-compatible endpoints
  (`NX_SOURCES`) and switch between them with `/source` mid-session.
- **Auto-compression** — long conversations are summarized as they grow, so the
  context window stays yours.
- **Web search** — SearXNG-backed, with an installer for a loopback-only
  instance (`nx setup`).
- **Batch mode** — `nx run` executes task files headlessly, in parallel.
- **Memory / recall** — persistent notes and past-session recall, searchable.

## Full documentation

Everything else lives in **[docs/REFERENCE.md](docs/REFERENCE.md)**: the
complete configuration-variable and slash-command tables, approval modes,
live tool progress, sessions, batch mode, the eval corpus, tools, memory,
metrics, and architecture. Also in `docs/`:

- [docs/SEARXNG_SETUP.md](docs/SEARXNG_SETUP.md) — web-search setup details
- [docs/TELEGRAM_SETUP.md](docs/TELEGRAM_SETUP.md) — Telegram headless surface
- [docs/PERF.md](docs/PERF.md) — performance notes

Annotated multi-source example: [`sources.example.env`](sources.example.env).

## License

MIT License. See [`LICENSE`](LICENSE).
