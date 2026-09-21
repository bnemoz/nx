# SearXNG for nx `web_fetch` — setup guide

Goal: a loopback-only SearXNG instance that nx uses as its preferred search
backend via `NX_SEARX_URL`. No Docker, no public exposure. Reproduce these
steps on any machine (currently running on the primary workstation; see
"Checklist" at the bottom).

## 1. Install SearXNG from source

```bash
git clone https://github.com/searxng/searxng.git ~/.local/share/searxng-src
cd ~/.local/share/searxng-src
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

(`uv` is the only extra dependency; any Python 3.10+ venv works the same way.)

If the editable install fails with `ModuleNotFoundError: msgspec` (or `yaml` /
`typing_extensions`) from `searx/__init__.py`, upstream `setup.py` imports
`searx` at build time without declaring those build deps. Workaround
(verified 2026-09-20 on the second machine):

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-build-isolation -e .
```

## 2. Settings

Write `~/.local/share/searxng/settings.yml`:

```yaml
use_default_settings: true

server:
  # generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
  secret_key: "PASTE_RANDOM_HEX_HERE"
  bind_address: "127.0.0.1"   # loopback ONLY — never 0.0.0.0
  port: 8888
  limiter: false              # private instance, rate limiter unnecessary

search:
  formats:
    - html
    - json                    # required: nx queries ?format=json
```

Keys differ per machine — generate a fresh `secret_key` on each one; do not
copy it between machines.

## 3. systemd user service

Write `~/.config/systemd/user/searxng.service`:

```ini
[Unit]
Description=SearXNG metasearch (local, for nx web_fetch)

[Service]
ExecStart=%h/.local/share/searxng-src/.venv/bin/python -m searx.webapp
Environment=SEARXNG_SETTINGS_PATH=%h/.local/share/searxng/settings.yml
WorkingDirectory=%h/.local/share/searxng-src
Restart=on-failure

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now searxng
systemctl --user status searxng
```

Boot-time note: a user service starts at login/boot only if lingering is
enabled for the user. Check once with `loginctl show-user $USER -p Linger`;
if it shows `Linger=no`, run `sudo loginctl enable-linger $USER`.

## 4. Point nx at it

Set in your environment (e.g. `~/.profile` or environment.d, wherever
`NX_*` vars live on this machine):

```bash
export NX_SEARX_URL=http://127.0.0.1:8888
```

Unset or unreachable, nx silently falls back to DuckDuckGo-lite — see step 5
to make sure it didn't.

## 5. Verify

All three should pass:

```bash
# a) loopback-only bind (must show 127.0.0.1:8888, not 0.0.0.0)
ss -tln | grep 8888

# b) JSON endpoint answers (expect HTTP 200)
curl -s -o /dev/null -w '%{http_code}\n' \
  'http://127.0.0.1:8888/search?q=test&format=json'

# c) full payload shape nx expects (expect ~30 results, the 5 keys)
curl -s 'http://127.0.0.1:8888/search?q=test&format=json' | python3 -c \
  "import sys,json;d=json.load(sys.stdin);print(len(d['results']));print(list(d)[:5])"
# expect: 34 / ['query','results','answers','corrections','infoboxes']
```

Then in a fresh nx session, confirm `web_fetch` with a `query` uses SearXNG
(no DDG fallback hint in the result).

## Exposure notes (why this config is safe)

- SearXNG binds `127.0.0.1` only — unreachable from the LAN or internet at
  the socket level, independent of any firewall. Verify with `ss` (step 5a),
  not just the config file.
- `limiter: false` and JSON format are fine because the instance is
  loopback-only; do not relax `bind_address` instead of them.
- No container is needed: Docker would add image-update and daemon
  dependencies with no isolation gain for a loopback-bound process.
- Decided 2026-09-19 on the primary machine; this file is the source of
  truth for reproducing it elsewhere.

## Checklist for the second machine

(Or: let the nx onboarding wizard do all of this — `nx setup`, option [2]
automates steps 1–4 and verifies with the step-5 probe. The manual path is
below.)

1. Clone source + venv (step 1)
2. Write `settings.yml` with a **fresh** `secret_key`, loopback bind, json format (step 2)
3. Install + enable the systemd unit; check lingering (step 3)
4. Export `NX_SEARX_URL=http://127.0.0.1:8888` (step 4)
5. Run all three verification commands (step 5)
6. Optionally update the nx project memory note for that machine
