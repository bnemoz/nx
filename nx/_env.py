"""Load ~/.env (or NX_ENV_FILE) into os.environ without clobbering vars
already set in the shell. Lets source config / API keys live in one place
instead of being exported in every terminal."""
from __future__ import annotations

import os


def _env_file_path() -> str:
    return os.path.expanduser(os.environ.get("NX_ENV_FILE", "~/.env"))


def _parse_env_value(value: str) -> str:
    """Return a .env value, removing comments outside quoted values.

    A ``#`` begins a comment only when it follows whitespace. This keeps
    values such as URLs or secrets containing a literal ``#`` intact while
    supporting the usual ``KEY=value # explanation`` form.
    """
    value = value.strip()
    if value and value[0] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end]
    comment_at = next(
        (i for i, char in enumerate(value)
         if char == "#" and i > 0 and value[i - 1].isspace()),
        None,
    )
    return value[:comment_at].rstrip() if comment_at is not None else value


def _load_env_file() -> None:
    try:
        with open(_env_file_path(), encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
        if not k or k in os.environ:
            continue
        os.environ[k] = _parse_env_value(v)


def _matches_key(line: str, name: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("export "):
        stripped = stripped[len("export "):]
    if "=" not in stripped:
        return False
    return stripped.split("=", 1)[0].strip() == name


def _set_env_var(name: str, value: str) -> None:
    """Persist `name=value` into ~/.env (or NX_ENV_FILE), replacing an
    existing `[export ]name=...` line if present, else appending one. Used by
    `/theme` so a runtime switch survives a restart without hand-editing the
    file — mirrors how NX_ACTIVE/NX_THEME are read by _load_env_file()."""
    env_file = _env_file_path()
    try:
        with open(env_file, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        lines = []

    new_line = f"{name}={value}\n"
    for i, line in enumerate(lines):
        if _matches_key(line, name):
            lines[i] = new_line
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
