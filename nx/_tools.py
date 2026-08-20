"""Read-only tool bodies with no approval/global-state dependency: read_file,
list_dir, search_files, and the line-number-stripping helper shared by the
edit tools. The mutating tools live in their own submodules instead:
write_file/edit_file/multi_edit/sed_edit in _edit_tools.py, run_bash/
wait_background in _bash_jobs.py — both need _confirm() (reads the YOLO/
APPROVE_LEVEL globals) and other state this module deliberately avoids.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any

from ._env import _env_int


def read_file(path: str | list[str], offset: int = 1, limit: int | None = None, **_: Any) -> str:
    """Read a file as numbered lines (1-based, `cat -n` style: right-aligned
    line number, a tab, then the line). Large files return only a window — pass
    offset (start line) and limit (max lines) to page through the rest. A header
    announces the visible range + total line count so the model can ask for more
    instead of being head/tail-capped by _sanitize_tool_result.

    path may also be a list of paths; each is read with the same offset/limit
    and the results are concatenated under a "==> path <==" header per file."""
    if isinstance(path, list):
        return "\n".join(
            f"==> {p} <==\n{read_file(p, offset=offset, limit=limit)}" for p in path
        )
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    except UnicodeDecodeError:
        return f"ERROR: {path} is not a valid UTF-8 text file (binary or wrong encoding)"
    total = len(lines)
    try:
        start = max(1, int(offset)) - 1
    except (TypeError, ValueError):
        start = 0
    if total == 0:
        return f"[{path}: empty file]"
    if start >= total:
        return f"[{path}: {total} lines; offset {start + 1} is past end of file]"
    if limit is None:
        limit = _env_int("NX_READ_FILE_LINES", 400)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 400
    end = total if limit <= 0 else min(total, start + limit)
    body = "".join(f"{i + 1:6d}\t{lines[i]}" for i in range(start, end))
    if body and not body.endswith("\n"):
        body += "\n"
    if start > 0 or end < total:
        tip = ""
        if start == 0 and end < total:
            tip = (" Looking for one function/section rather than the whole "
                   "file? STOP paging -- call search_files(pattern, context=N) "
                   "instead, it finds it in one shot.")
        return (f"[{path}: lines {start + 1}-{end} of {total}; "
                f"call read_file with offset/limit to page.{tip}]\n" + body)
    return body


_LINE_NUM_PREFIX = re.compile(r"^ *\d+\t")


def _strip_line_numbers(text: str) -> str:
    """Remove read_file's `<n>\\t` line-number prefixes from a pasted block.
    Only strips when EVERY non-empty line carries one (i.e. the whole block came
    from numbered read_file output) — so ordinary edits, and code that merely
    starts a line with digits, are never altered."""
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty and all(_LINE_NUM_PREFIX.match(ln) for ln in nonempty):
        return "\n".join(_LINE_NUM_PREFIX.sub("", ln) for ln in lines)
    return text


def list_dir(path: str = ".", pattern: str | None = None, **_: Any) -> str:
    """List a directory. Without `pattern`: flat, alphabetical (like `ls`).
    With `pattern` (e.g. "**/*.py", "src/**/*.ts"): recursive glob search
    rooted at `path`, results sorted newest-modified first — mirrors Claude
    Code's Glob tool, for "find files matching X" instead of "what's in this
    one directory."""
    if pattern:
        import glob as _glob
        try:
            matches = _glob.glob(os.path.join(path, pattern), recursive=True)
        except (OSError, ValueError) as e:
            return f"ERROR: {type(e).__name__}: {e}"
        if not matches:
            return f"[no files matching {pattern!r} under {path}]"
        matches.sort(key=lambda p: -(os.path.getmtime(p) if os.path.exists(p) else 0))
        return "\n".join(matches)
    try:
        return "\n".join(sorted(os.listdir(path)))
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"


def search_files(
    pattern: str | list[str],
    path: str = ".",
    glob: str | None = None,
    max_results: int = 100,
    context: int = 0,
    ignore_case: bool = False,
    output_mode: str = "content",
    **_: Any,
) -> str:
    """Search file contents for a regex `pattern` and return `path:line:match`
    hits (ripgrep-style). Falls back to `grep -rn` when `rg` isn't installed, so
    the tool works with only a POSIX grep present — nx adds no new runtime
    dependency.

    Read-only and a known-safe operation, so like `read_file` / `list_dir` it
    never goes through the approval / risk classifier — that's the whole point
    of a native search tool instead of shelling out to grep via `run_bash`
    (which *would* need its own risk-classification pass). `path:line` hits map
    directly onto `read_file(path, offset=line)`; pass `context` when the
    surrounding lines are needed too, so a targeted question doesn't force a
    full read_file of a large file just to see what's around a match.

    Args:
      pattern: regex (not a glob); pass a list to OR multiple patterns in one search.
      path: file or directory to search under (default ".").
      glob: optional include filter, e.g. "*.py" (rg `-g` / grep `--include`).
      max_results: cap on matches returned (default 100; <=0 -> 1000).
      context: lines of context before/after each match (rg/grep `-C`).
      ignore_case: case-insensitive match (rg/grep `-i`).
      output_mode: "content" (default, path:line:match), "files_with_matches"
        (just matching file paths, one per line), or "count" (path:count).
    """
    patterns = pattern if isinstance(pattern, list) else [pattern]
    patterns = [str(p) for p in patterns if p and str(p).strip()]
    if not patterns:
        return "ERROR: `pattern` is empty"
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 100
    if max_results <= 0:
        max_results = 1000
    try:
        context = max(0, int(context))
    except (TypeError, ValueError):
        context = 0
    if output_mode not in ("content", "files_with_matches", "count"):
        output_mode = "content"
    use_rg = bool(shutil.which("rg"))
    pattern_flags = [f for p in patterns for f in ("-e", p)]
    # Directories ripgrep skips by default (it honors .gitignore, which lists
    # .venv/, node_modules/, __pycache__/, build/, dist/, etc.). The grep
    # fallback does NOT read .gitignore, so without these --exclude-dir flags
    # it recursively scans the whole virtualenv on every call — the dominant
    # cost of search_files when rg isn't installed. The list mirrors what rg
    # ignores out of the box; NX_SEARCH_EXCLUDE_DIRS overrides it.
    exclude_dirs = [d.strip() for d in os.environ.get(
        "NX_SEARCH_EXCLUDE_DIRS",
        ".git .hg .svn .venv venv env node_modules __pycache__ .mypy_cache "
        ".pytest_cache .ruff_cache .tox dist build .eggs",
    ).split() if d.strip()]
    if use_rg:
        cmd = ["rg", "--no-heading", "--color=never", *pattern_flags]
        if output_mode == "files_with_matches":
            cmd += ["-l"]
        elif output_mode == "count":
            cmd += ["-c"]
        else:
            cmd += ["-n"]
            if context:
                cmd += ["-C", str(context)]
        if ignore_case:
            cmd += ["-i"]
        if glob:
            cmd += ["-g", str(glob)]
        cmd += ["--", str(path)]
    else:
        cmd = ["grep", "-rE", "--color=never",
               *[f"--exclude-dir={d}" for d in exclude_dirs],
               *pattern_flags]
        if output_mode == "files_with_matches":
            cmd += ["-l"]
        elif output_mode == "count":
            cmd += ["-c"]
        else:
            cmd += ["-n"]
            if context:
                cmd += ["-C", str(context)]
        if ignore_case:
            cmd += ["-i"]
        if glob:
            cmd += ["--include=" + str(glob)]
        cmd += ["--", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as e:
        return f"ERROR: no search backend found (rg/grep): {e}"
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out after 30s"
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    err = (proc.stderr or "").strip()
    # grep/rg exit 1 = no matches (normal); exit 2 = real error. Only surface
    # stderr as an error when there's no stdout to return.
    if err and not proc.stdout:
        return f"ERROR: {err.splitlines()[-1] if err else 'search failed'}"
    lines = proc.stdout.splitlines()
    if output_mode == "count":
        # grep -rc emits "path:0" for non-matching files; rg -c omits them.
        # Normalize so both backends only report files with hits.
        lines = [ln for ln in lines if not ln.endswith(":0")]
    if not lines:
        return f"[no matches for {patterns!r} under {path}]"
    shown = "\n".join(lines[:max_results])
    if len(lines) > max_results:
        shown += f"\n... [{len(lines) - max_results} more matches; raise max_results to see more]"
    return shown


def _run_git(args: list[str], path: str) -> tuple[str | None, str | None]:
    if not shutil.which("git"):
        return None, "ERROR: git not found on PATH"
    try:
        proc = subprocess.run(["git", "-C", str(path), *args],
                               capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return None, "ERROR: git command timed out after 15s"
    except OSError as e:
        return None, f"ERROR: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        return None, f"ERROR: {(proc.stderr or proc.stdout).strip() or 'git command failed'}"
    return proc.stdout, None


def git_status(path: str = ".", **_: Any) -> str:
    """`git status --short -b`, read-only like `git diff`/`git log` — never
    goes through the approval / risk classifier (same rationale as
    `search_files`: a dedicated native tool beats shelling out via `run_bash`
    just to earn a risk-classifier round-trip for a command that's always
    read-only anyway)."""
    out, err = _run_git(["status", "--short", "-b"], path)
    if err or out is None:
        return err or "[clean working tree]"
    return out.strip() or "[clean working tree]"


def git_diff(path: str = ".", staged: bool = False, file: str | None = None, **_: Any) -> str:
    """`git diff` (or `git diff --staged` / `git diff -- <file>`), read-only —
    see `git_status` docstring for why this is a native tool rather than
    `run_bash("git diff")`.

    Args:
      path: repo/directory to run inside (default ".").
      staged: pass True for `git diff --staged` (staged changes only).
      file: optional path to restrict the diff to a single file.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if file:
        args += ["--", str(file)]
    out, err = _run_git(args, path)
    if err or out is None:
        return err or "[no changes]"
    return out.strip() or "[no changes]"


def git_log(path: str = ".", max_count: int = 20, file: str | None = None, **_: Any) -> str:
    """`git log --oneline` (one commit per line: short hash + subject),
    read-only -- see `git_status` docstring for why this is a native tool
    rather than `run_bash("git log")`.

    Args:
      path: repo/directory to run inside (default ".").
      max_count: cap on commits returned (default 20).
      file: optional path to restrict history to a single file.
    """
    args = ["log", "--oneline", f"-{max(1, int(max_count))}"]
    if file:
        args += ["--", str(file)]
    out, err = _run_git(args, path)
    if err or out is None:
        return err or "[no commits]"
    return out.strip() or "[no commits]"
