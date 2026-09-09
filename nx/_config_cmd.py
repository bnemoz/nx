"""ROADMAP P3-9: `nx config` + `/status` — a viewer/doctor over the env layer.

Env vars stay the substrate (no config-file format — CONTRIBUTING policy).
Everything here reads the same os.environ state that nx/__init__.py builds
from ~/.env (NX_ENV_FILE) at import time; `nx config set` is the only writer
and goes through _env._set_env_var (the same line-replacing writer /theme and
/approval set-default use) so the file's comments and other keys survive.

`nx config` must never touch the network: nx/__init__.py skips
switch_source()'s startup probes when argv[1] == "config", so the verbs read
purely import-time env state (st.SOURCES is still populated).
"""
from __future__ import annotations

import os
import re
import sys

from . import _state as st
from ._ansi import BOLD, C, DIM, RESET
from ._env import _env_file_path, _parse_env_value

_USAGE = """\
usage: nx config list [filter]        every config var in effect (NX_* plus the
                                      non-NX source keys), optional substring filter
       nx config get NAME [NAME...]   print one value (`NAME=value`; exit 1 if unset)
       nx config set NAME VALUE       persist to the env file and apply now
       nx config path                 print the env file being read/written
       nx config doctor               check the env layer for contradictions"""

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# Any var whose name ends in one of these is masked in list/get output — the
# viewer runs in terminals and scrollbacks, and an api key pasted into a
# share/screenshot is the one config leak that matters.
_SENSITIVE_SUFFIXES = ("API_KEY", "_KEY", "KEY", "TOKEN", "SECRET", "PASSWORD")
_MASK = "•••••••• (set — use printenv NAME to inspect)"

# Non-NX env keys nx reads for source discovery (README "Built-in sources").
_NON_NX_KEYS = ("TOGETHER_API_KEY", "OPENROUTER_API_KEY")


def _is_sensitive(name: str) -> bool:
    return name.endswith(_SENSITIVE_SUFFIXES)


def _display_value(name: str, value: str) -> str:
    if value == "":
        return "(empty)"
    if _is_sensitive(name):
        return _MASK
    return value


def _quote_env_value(value: str) -> str:
    """Quote a value written to the env file when its raw spelling would not
    survive _parse_env_value() on reload: a ' #' becomes a comment, and a
    leading quote would start an (unterminated) quoted value."""
    if value and value[0] in ("'", '"'):
        value = '"' + value.replace('"', '\\"') + '"'
    elif re.search(r"\s#", value):
        value = '"' + value.replace('"', '\\"') + '"'
    return value


def _source_names() -> list[str]:
    return list(st.SOURCE_ORDER)


def _env_var_for_source(name: str) -> str:
    """The NX_SOURCE_ prefix a source's settings live under — and the doc for
    the P3-9 audit finding: the mapping is `.upper()` only, so a hyphenated
    name yields a var no shell can set. doctor flags it; the fix (a sanitizing
    rule) would change working setups' names, so it stays a diagnosis."""
    return f"NX_SOURCE_{name.upper()}_*"


def _cmd_list(args: list[str]) -> int:
    flt = args[0].lower() if args else None
    path = _env_file_path()
    names = sorted(k for k in os.environ if k.startswith("NX_"))
    names += [k for k in _NON_NX_KEYS if k in os.environ]
    if flt:
        names = [k for k in names if flt in k.lower()]
    if not names:
        print(f"{DIM}  no config vars matching"
              f"{f' {args[0]!r}' if flt else ''} — nx is on defaults{RESET}")
        return 0
    print(f"{DIM}  env file: {path} · {len(names)} vars in effect"
          f"{f' (filter: {args[0]})' if flt else ''}{RESET}")
    for k in names:
        print(f"  {C.ACCENT}{k}{RESET}={_display_value(k, os.environ[k])}")
    return 0


def _cmd_get(args: list[str]) -> int:
    if not args:
        sys.stderr.write(_USAGE + "\n")
        return 2
    rc = 0
    for name in args:
        if name in os.environ:
            print(f"{name}={_display_value(name, os.environ[name])}")
        else:
            sys.stderr.write(f"nx config: {name} is not set\n")
            rc = 1
    return rc


def _cmd_set(args: list[str]) -> int:
    if len(args) != 2:
        sys.stderr.write("usage: nx config set NAME VALUE\n")
        return 2
    name, value = args
    if not _IDENT_RE.match(name):
        sys.stderr.write(
            f"nx config: {name!r} is not a valid env var name — letters, "
            "digits, underscore, no leading digit\n")
        return 2
    from ._env import _set_env_var
    _set_env_var(name, _quote_env_value(value))
    os.environ[name] = value  # apply to the (already imported) session too
    shown = _MASK if _is_sensitive(name) else value
    print(f"  {C.ACCENT}{name}{RESET}={shown} "
          f"{DIM}(saved to {_env_file_path()}){RESET}")
    return 0


def _cmd_path(_args: list[str]) -> int:
    print(_env_file_path())
    return 0


def _cmd_doctor(_args: list[str]) -> int:
    path = _env_file_path()
    issues: list[tuple[str, str]] = []
    notes: list[str] = []

    def bad(text: str) -> None:
        issues.append(text)

    def warn(text: str) -> None:
        notes.append(text)

    # --- env file readability ------------------------------------------------
    if os.environ.get("NX_ENV_FILE") == "/dev/null":
        warn("env file is NX_ENV_FILE=/dev/null — nothing persists")
    elif not os.path.exists(path):
        warn(f"env file {path} does not exist yet (created on first "
             "`nx config set`)")
    elif not os.access(path, os.R_OK):
        bad(f"env file {path} is not readable (permission denied)")
    else:
        # Key syntax + quote hygiene in the file itself. _load_env_file()
        # silently skips lines it can't parse; doctor makes them visible.
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            bad(f"env file {path} cannot be read: {e}")
            lines = []
        for i, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            if not _IDENT_RE.match(key):
                bad(f"{path}:{i}: {key!r} is not a settable env var name "
                    "(this line is silently ignored at load)")
            resolved = _parse_env_value(line.split("=", 1)[1])
            if resolved and resolved[0] in ("'", '"'):
                bad(f"{path}:{i}: {key} still has a leading quote after "
                    "parsing (unterminated quoted value?)")

    # --- source wiring ---------------------------------------------------------
    for name in _source_names():
        if not re.fullmatch(r"[a-z0-9_]+", name or ""):
            p = _env_var_for_source(name)
            bad(f"source {name!r} maps to {p} — hyphens/dots can't be set "
                "from a shell; rename the source in NX_SOURCES")
        src = st.SOURCES.get(name)
        if src is not None and src.api_key in (None, "", "sk-noop"):
            warn(f"source {name!r} has no API key configured "
                 f"({_env_var_for_source(name)}API_KEY or $indirection)")
    # ACTIVE / NX_ACTIVE agreement (import-time selection: --source > NX_ACTIVE
    # > first source; only meaningful when sources exist).
    active_name = st.ACTIVE.name if st.ACTIVE is not None else None
    if active_name is None and _source_names():
        bad("no active source selected")
    if st.ACTIVE is not None and active_name not in _source_names():
        bad(f"active source {active_name!r} is not in st.SOURCES")
    for var in ("NX_ACTIVE", "NX_BASE_URL", "NX_API_KEY", "NX_MODEL"):
        if var in os.environ and _source_names():
            warn(f"{var} is set but ignored while NX_SOURCE_* sources exist "
                 "(legacy single-source vars only apply with no sources)")

    # --- contradictory / shadowed flags ----------------------------------------
    # Shell-vs-file precedence: _load_env_file() is first-set-wins against
    # os.environ, so a var exported in the shell always beats the env file.
    # A differing pair is the classic "why won't my setting stick" — flag it.
    if os.path.exists(path) and os.access(path, os.R_OK):
        try:
            with open(path, encoding="utf-8") as f:
                for i, raw in enumerate(f.readlines(), 1):
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key = line.split("=", 1)[0].strip()
                    if key.startswith("export "):
                        key = key[len("export "):].strip()
                    if key in os.environ and key.startswith("NX_") \
                            and os.environ[key] != _parse_env_value(line.split("=", 1)[1]):
                        warn(f"{key} differs between the shell ({os.environ[key]!r}) "
                             f"and {path} — the shell wins; edit the shell or "
                             "unset it there")
        except OSError:
            pass
    if os.environ.get("NX_VISION", "") not in ("", "0"):
        from . import _compress as _c
        if getattr(_c, "NX_NATIVE_TOOLS", False):
            bad("NX_VISION=1 but native tool calls are off (NX_NATIVE_TOOLS=0) — "
                "image mentions will carry a gate note instead of the picture")

    # --- report ------------------------------------------------------------------
    print(f"{BOLD}nx config doctor{RESET} {DIM}— env file {path} · "
          f"{len(_source_names())} source(s){RESET}")
    for note in notes:
        print(f"  {C.WARNING}•{RESET} {note}")
    if issues:
        for text in issues:
            print(f"  {C.ERROR}✗ {text}{RESET}")
        print(f"{DIM}  {len(issues)} issue(s); env vars stay the substrate — "
              f"fix them in the file, not a new format{RESET}")
        return 1
    print(f"  {C.SUCCESS}✓{RESET} no issues found "
          f"{DIM}({len(notes)} note(s)){RESET}")
    return 0


def _cli_config(args: list[str]) -> int:
    """`nx config` verb — returns the process exit code (0 ok, 1 doctor found
    issues / get missed a name, 2 usage)."""
    verb = args[0] if args else "list"
    rest = args[1:]
    if verb in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    handlers = {
        "list": _cmd_list,
        "get": _cmd_get,
        "set": _cmd_set,
        "path": _cmd_path,
        "doctor": _cmd_doctor,
    }
    handler = handlers.get(verb)
    if handler is None:
        sys.stderr.write(f"nx config: unknown subcommand {verb!r}\n" + _USAGE + "\n")
        return 2
    return handler(rest)


# --- /status -----------------------------------------------------------------
def render_status() -> str:
    """The `/status` snapshot: source, model, caps, budget, fence, approval —
    the same env-layer state `nx config doctor` reasons about, as the session
    sees it. Pure text so it stays testable and terminal-dumpable."""
    import nx as _m

    sep = f"{DIM} · {RESET}"
    lines: list[str] = []
    lines.append(f"{BOLD}nx{RESET}{sep}"
                 f"{C.ACCENT}{st.MODEL or '(no model)'}{RESET}")
    if st.ACTIVE is not None:
        src = st.ACTIVE
        ctx = src._context_window
        ctx_s = f"{ctx // 1000}K" if isinstance(ctx, int) and ctx > 0 else "?"
        lines.append(f"{DIM}  source{RESET}    {src.name}{sep}{src.base_url}"
                     f"{sep}ctx {ctx_s}")
        lines.append(f"{DIM}  env vars{RESET}  {_env_var_for_source(src.name)}"
                     f"{sep}file {_env_file_path()}")
    else:
        lines.append(f"{DIM}  source{RESET}    (none — set NX_SOURCE_* or "
                     "NX_BASE_URL)")
    # Tool-calling protocol: the probed capability, else the env gate.
    cap = getattr(st.ACTIVE, "_capability", None) if st.ACTIVE else None
    if cap is None:
        from . import _compress as _c
        cap = ("none (NX_NATIVE_TOOLS=0)" if getattr(_c, "NX_NATIVE_TOOLS", False)
               else "native (auto)" if getattr(_c, "NATIVE_TOOLS_DEFAULT", True)
               else ("grammar" if getattr(_c, "NX_GRAMMAR", False) else "pending"))
    lines.append(f"{DIM}  tools{RESET}     {cap}"
                 + (f"{sep}vision on" if os.environ.get("NX_VISION", "") not in ("", "0")
                    else ""))
    lines.append(f"{DIM}  budget{RESET}    reasoning "
                 f"{st.REASONING_BUDGET_TOKENS or 'off'}"
                 f"{sep}floor {os.environ.get('NX_FLOOR_BUDGET', '(unset)')} tok"
                 f"{sep}last prompt {st._LAST_PROMPT_TOKENS} tok")
    from ._fence import _writable_roots
    from ._env import _env_int
    fence = "off" if not _env_int("NX_WRITE_FENCE", 1) else "on"
    roots = _writable_roots()
    lines.append(f"{DIM}  fence{RESET}     {fence}{sep}writable: "
                 f"{', '.join(roots) or '-'}")
    lines.append(f"{DIM}  approval{RESET}  {_m._approval_display()}")
    return "\n".join(lines)
