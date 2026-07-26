"""Load ~/.env (or NX_ENV_FILE) into os.environ without clobbering vars
already set in the shell. Lets source config / API keys live in one place
instead of being exported in every terminal."""
import os

def _load_env_file():
    env_file = os.path.expanduser(os.environ.get("NX_ENV_FILE", "~/.env"))
    try:
        with open(env_file, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k.startswith("export "):
                    k = k[len("export "):].strip()
                if not k or k in os.environ:
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                os.environ[k] = v
    except (OSError, IOError):
        pass


def _set_env_var(name, value):
    """Persist `name=value` into ~/.env (or NX_ENV_FILE), replacing an
    existing `[export ]name=...` line if present, else appending one. Used by
    `/theme` so a runtime switch survives a restart without hand-editing the
    file — mirrors how NX_ACTIVE/NX_THEME are read by _load_env_file()."""
    env_file = os.path.expanduser(os.environ.get("NX_ENV_FILE", "~/.env"))
    try:
        with open(env_file, encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, IOError):
        lines = []

    def _matches(line):
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        return stripped.split("=", 1)[0].strip() == name if "=" in stripped else False

    new_line = f"{name}={value}\n"
    for i, line in enumerate(lines):
        if _matches(line):
            lines[i] = new_line
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
