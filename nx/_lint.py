"""lint_check tool (tier-2, gated on NX_LINT): fast style/lint feedback for
the files just edited, without an approval-gated run_bash round-trip.

Today the only lint path is `run_bash ruff ...` — an approval prompt, a
spinner, and a free-text shell command for what is a fixed, read-only
invocation. lint_check closes that gap: fixed command shapes per linter
(never free-text shell), read-only (never writes — no approval needed),
and tier-2 gated on NX_LINT (default OFF; 0 schema tokens until surfaced
via search_tools) so sessions that never lint don't pay for it.

Scope contract: `paths` is a space-separated list of existing files/dirs
(relative or absolute, `~` expands). With no paths, the changed files of
the current session's undo checkpoints are used — the natural "check what
I just edited" default — falling back to nothing (error) rather than
linting the whole tree.

Approval contract: fully read-only — no _confirm call, like git_status.
Plan-mode note: read-only tools are not gated by plan mode, and neither is
this (it writes nothing).

Command shapes (one per linter, chosen by marker/binary detection):
  python  -> ruff check <paths> + ty check           (fallback: python -m
                                                       py_compile when ruff
                                                       is absent; ty runs
                                                       package-wide)
  go      -> go vet <paths>
  rust    -> cargo check --message-format short
  js/ts   -> npx --no-install eslint <paths>   (when eslint is installed)
Detection order: a linter named explicitly wins; otherwise the first
supported ecosystem found in the paths' project. NX_LINT_CMD overrides the
whole command shape for teams with house lint configs (e.g.
"NX_LINT_CMD=ruff check --select I").
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from . import _state as st
from ._env import _env_int

LINT_ENABLED = _env_int("NX_LINT", 0)

# Cap on the returned output; linters can be extremely chatty on a big glob.
_MAX_OUT_CHARS = 8000
_TIMEOUT_SECS = _env_int("NX_LINT_TIMEOUT", 120)


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _session_touched_files() -> list[str]:
    """Files this session edited (from the undo stack's checkpoints), most
    recent first — the no-args default scope. Read-only peek at _undo state;
    a missing/empty stack returns []."""
    try:
        from ._undo import list_checkpoints
        seen: list[str] = []
        for cp in reversed(list_checkpoints()):
            for f in getattr(cp, "files", ()) or ():
                p = getattr(f, "path", None)
                if p and p not in seen:
                    seen.append(p)
        return seen
    except Exception:  # noqa: BLE001 — the default scope must never raise
        return []


def _detect_linters(paths: list[str]) -> list[tuple[str, list[str]]]:
    """(linter, base_argv) runs for the first supported ecosystem among paths'
    project root(s). Python projects get BOTH checks when installed —
    `ty check` (types) and `ruff check` (style/lint) are one workflow gate —
    falling back to `python -m py_compile` when ruff is absent. Other
    ecosystems: go vet, cargo check, eslint via npx (only when installed)."""
    roots = []
    for p in paths:
        d = os.path.dirname(os.path.abspath(_expand(p)))
        while d != os.path.dirname(d):
            roots.append(d)
            d = os.path.dirname(d)
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    runs: list[tuple[str, list[str]]] = []
    for d in dict.fromkeys(roots):
        try:
            names = set(os.listdir(d))
        except OSError:
            continue
        if any(n in names for n in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")) or any(
                e in exts for e in (".py",)):
            # ty (types) then ruff (style/lint) — the standard python gate;
            # py_compile is the last-resort syntax check when ruff is absent.
            if shutil.which("ty"):
                runs.append(("python", ["ty", "check"]))
            if shutil.which("ruff"):
                runs.append(("python", ["ruff", "check"]))
            elif not runs:
                runs.append(("python", [os.sys.executable, "-m", "py_compile"]))
        if "go.mod" in names:
            runs.append(("go", ["go", "vet"]))
        if "Cargo.toml" in names:
            runs.append(("rust", ["cargo", "check", "--message-format", "short"]))
        if "package.json" in names and shutil.which("npx") and shutil.which("eslint"):
            runs.append(("js", ["npx", "--no-install", "eslint"]))
        if runs:
            break
    return runs


def lint_check(paths: str = "", **_: Any) -> str:
    """Tool body. `paths`: space-separated files/dirs; empty = this session's
    edited files. Returns the linter(s') (truncated) output with a verdict."""
    argv_override = os.environ.get("NX_LINT_CMD", "").strip()
    plist = [_expand(p) for p in (paths or "").split() if p.strip()]
    if not plist:
        plist = _session_touched_files()
        if not plist:
            return ("ERROR: lint_check: no paths given and no files edited this "
                    "session — pass `paths` (space-separated files or dirs)")
    missing = [p for p in plist if not os.path.exists(p)]
    if missing:
        return f"ERROR: lint_check: path(s) not found: {', '.join(missing)}"
    if argv_override:
        import shlex
        runs = [("custom (NX_LINT_CMD)", shlex.split(argv_override))]
    else:
        runs = _detect_linters(plist)
        if not runs:
            return ("ERROR: lint_check: no supported linter ecosystem detected "
                    "(python/pyproject, go.mod, Cargo.toml, package.json). Set "
                    "NX_LINT_CMD to a fixed lint command for this project.")
    blocks = []
    for linter, argv in runs:
        if linter == "go":
            argv = argv + plist
        elif linter == "rust":
            pass  # cargo check ignores explicit paths; runs the workspace
        elif linter == "python" and argv[:1] == ["ty"]:
            pass  # ty check takes no paths; it resolves the package from cwd
        else:
            argv = argv + plist[:50]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, check=False,
                               timeout=_TIMEOUT_SECS, cwd=st.PROJECT_ROOT or None)
        except subprocess.TimeoutExpired:
            blocks.append(f"[lint_check: {linter}] TIMED OUT after {_TIMEOUT_SECS}s")
            continue
        except FileNotFoundError:
            return f"ERROR: lint_check: {argv[0]!r} not found on PATH — install it or set NX_LINT_CMD"
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if len(out) > _MAX_OUT_CHARS:
            out = out[:_MAX_OUT_CHARS] + f"\n... [lint output truncated at {_MAX_OUT_CHARS} chars]"
        verdict = "clean" if r.returncode == 0 else f"{linter} reported issues (exit {r.returncode})"
        blocks.append(f"[lint_check: {linter}] {verdict}\n{out}")
    return "\n".join(blocks)


LINT_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "lint_check",
        "description": "Run the project's linter (ruff + ty for python, go vet, cargo check, eslint) read-only over given files — or, no paths, the files edited this session. Fixed command shapes (never free-text shell, never writes, no approval). Arbitrary lint flags/fixes -> run_bash.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {"type": "string", "description": "Space-separated files/dirs to lint (default: files edited this session)"},
            },
            "required": [],
        },
    },
}
