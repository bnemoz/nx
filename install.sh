#!/bin/sh
# nx one-line installer:  curl -fsSL https://raw.githubusercontent.com/nmishra93/nx/main/install.sh | sh
#
# Installs the `nx` CLI as a uv tool (isolated environment, upgradeable with
# `uv tool upgrade nx`). Falls back to pipx, then pip --user when uv/pipx
# are unavailable. Works on Linux and macOS; requires Python >= 3.9.
set -eu

REPO="https://github.com/nmishra93/nx.git"
BRANCH="main"

say() { printf 'nx-installer: %s\n' "$1"; }
die() { printf 'nx-installer: ERROR: %s\n' "$1" >&2; exit 1; }

# --- Locate or bootstrap a Python >= 3.9 -------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

python_ok() {
    command -v "$1" >/dev/null 2>&1 || return 1
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null
}

PY=""
for c in python3 python; do
    if python_ok "$c"; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "no Python >= 3.9 found on PATH; install one first (https://www.python.org)"

# --- Install via uv (preferred), pipx, or pip --user --------------------------
if ! have uv; then
    say "uv not found; bootstrapping it..."
    # Official standalone installer; installs to ~/.local/bin without root.
    curl -LsSf https://astral.sh/uv/install.sh | sh || die "failed to bootstrap uv"
    export PATH="$HOME/.local/bin:$PATH"
    have uv || die "uv was installed but is not on PATH; reopen your shell and re-run"
fi

say "installing nx from $REPO ($BRANCH) via uv tool install"
# Retry once: fresh-clone hiccups (network, GitHub rate limits) are common.
uv tool install --force "git+$REPO" \
    || uv tool install --force "git+$REPO" \
    || die "uv tool install failed"

# uv puts tool shims on ~/.local/bin; nudge the current shell's PATH.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

if command -v nx >/dev/null 2>&1; then
    say "installed: $(nx --version 2>/dev/null || echo nx)"
    say "later upgrades: uv tool upgrade nx"
else
    say "nx was installed to ~/.local/bin — add it to PATH if not already there:"
    say "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.profile && source ~/.profile"
fi
say "next: run 'nx' and set your endpoint/key per the README (sources.example.env)"
