# Telegram gateway — setup & notes (nx 0.79.0)

The implementation lives in `nx/_telegram.py` (wired as `nx telegram` in
`nx/_cli.py`); tests in `tests/test_telegram.py`. This file is the *operator*
side: how to run it on this machine and the one open issue (network).

## Status

- [x] Implementation (gateway, PDF routing in web_fetch, docs, tests) — in nx
- [ ] Network egress from `thor` — **BLOCKED**, see below
- [ ] Token revoked + reissued (old one was pasted into a shell — treat as burned)

## Setup steps (once network is solved)

1. **Bot token** — @BotFather in Telegram: `/newbot`, or `/revoke` the old
   token. Save it with `nx config set NX_TELEGRAM_BOT_TOKEN "<token>"` so it
   never lands in shell history.

2. **Chat ID** — message the bot once from the phone, then:
   ```bash
   curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" \
     | python3 -m json.tool | grep '"id"'
   ```
   First `"id"` under `chat` (positive number = private chat). Then:
   ```bash
   nx config set NX_TELEGRAM_ALLOWED_CHATS "<id>"
   ```
   The allowlist is HARD: empty → the gateway refuses to start. Other chats
   are dropped without reply.

3. **Run it** (persistent — tmux / systemd user unit):
   ```bash
   nx telegram            # approval auto-deny (safe, ingestion-only)
   nx telegram --yolo     # writes/shell allowed — only in a vault you accept
   ```
   Approval posture matches batch mode: no terminal to answer Y/n, so gated
   calls auto-DENY unless launched with `--yolo`. Middle ground: pre-bless
   specific action shapes with `/approval allow` rules in a local session —
   the allowlist file is shared.

## Usage from the phone

- **Web link** — sent as text; the model fetches it itself via `web_fetch`
  (readable-text extraction, links kept, 7-day cache). Multi-hop works: it
  can follow a landing page to the PDF, search the DOI, etc.
- **PDF attachment** — caption + file: downloaded via getFile and inlined as
  extracted text (`_pdf.extract_pdf`, page-capped via `NX_PDF_MAX_PAGES`).
- **arXiv/PMC/journal PDF URLs** — `web_fetch` now sniffs PDFs
  (content-type or `%PDF-` magic bytes) and routes them through the same
  extractor, so "read this arXiv link" and an uploaded PDF converge.
- Progress: a `⚙ working…` message live-edits once per model-turn boundary
  (tool name + target + result size). Final answer arrives in ≤4096-char
  chunks. One conversation per process; a restart starts fresh.

## ⚠ Open issue: thor cannot reach api.telegram.org

Institute firewall (Scripps) DNS-blackholes Telegram:

```
api.telegram.org → CNAME blocked.scripps.edu → 172.29.200.35
TLS handshake returns a *.scripps.edu cert — the firewall answers in
Telegram's place. 100% block, no client-side CA/proxy trust fix.
```

Constraint that shapes the fix: **the LLM endpoint is an institute server**,
so the gateway must stay on thor (or in-network) for the model, while only
Telegram traffic escapes. Both options below are split-tunnel:

### Option A — Tailscale exit node with LAN access (cleanest)

On thor (plus any off-network node — VPS, home box, even the phone):
```bash
sudo tailscale up --exit-node=<node> --exit-node-allow-lan-access
nx telegram   # no other config; localhost:8080 model stays direct
```

### Option B — HTTP proxy + no_proxy

Plain urllib does NOT speak SOCKS — needs an HTTP CONNECT proxy. On any
off-network SSH box: `tinyproxy` (Port 8888, Allow 127.0.0.1), then tunnel:
```bash
ssh -L 8888:127.0.0.1:8888 -fN user@remote-box
export https_proxy=http://127.0.0.1:8888
export no_proxy="localhost,127.0.0.1,.scripps.edu,172.29.*"
export HTTPS_PROXY="$https_proxy"; export NO_PROXY="$no_proxy"
nx telegram
```
Note: if the model endpoint URL is `http://` (not https), also set
`http_proxy` — `no_proxy` governs both.

## Env vars (README has the full entries)

| var | purpose |
|---|---|
| `NX_TELEGRAM_BOT_TOKEN` | from @BotFather; unset → `nx telegram` exits with guidance |
| `NX_TELEGRAM_ALLOWED_CHATS` | comma-separated chat-ID allowlist; empty refuses to start |
| `NX_TELEGRAM_POLL_INTERVAL` | backoff between empty/failed polls (default 1.0 s) |

## Security notes

- A Telegram-exposed coding agent that can run shell is an RCE surface with
  the allowlist as the only gate — treat `--yolo` accordingly.
- The token was exposed (pasted in a shell command); revoke it at @BotFather
  before first real use, and store via `nx config set`, not shell history.
- `.scripps.edu` block list likely covers all Telegram hosts
  (web.telegram.org included) — it's policy, not transient.
