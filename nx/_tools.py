"""Read-only tool bodies with no approval/global-state dependency: read_file,
list_dir, search_files, and the line-number-stripping helper shared by the
edit tools. write_file/edit_file/multi_edit/run_bash/wait_background stay in
__init__.py because they call _confirm() (reads the YOLO/APPROVE_LEVEL
globals) and, for the bash tools, SESSION_HOME/_USER_INTERRUPTED/_log_event.
"""
import os
import re
import shutil
import subprocess

from ._env import _env_int


def read_file(path, offset=1, limit=None, **_):
    """Read a file as numbered lines (1-based, `cat -n` style: right-aligned
    line number, a tab, then the line). Large files return only a window — pass
    offset (start line) and limit (max lines) to page through the rest. A header
    announces the visible range + total line count so the model can ask for more
    instead of being head/tail-capped by _sanitize_tool_result."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
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


def _strip_line_numbers(text):
    """Remove read_file's `<n>\\t` line-number prefixes from a pasted block.
    Only strips when EVERY non-empty line carries one (i.e. the whole block came
    from numbered read_file output) — so ordinary edits, and code that merely
    starts a line with digits, are never altered."""
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty and all(_LINE_NUM_PREFIX.match(ln) for ln in nonempty):
        return "\n".join(_LINE_NUM_PREFIX.sub("", ln) for ln in lines)
    return text


def list_dir(path=".", **_):
    try:
        return "\n".join(sorted(os.listdir(path)))
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"


def search_files(pattern, path=".", glob=None, max_results=100, context=0, **_):
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
      pattern: regex (not a glob).
      path: file or directory to search under (default ".").
      glob: optional include filter, e.g. "*.py" (rg `-g` / grep `--include`).
      max_results: cap on matches returned (default 100; <=0 -> 1000).
      context: lines of context before/after each match (rg/grep `-C`).
    """
    if not pattern or not str(pattern).strip():
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
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--color=never", "-e", str(pattern)]
        if context:
            cmd += ["-C", str(context)]
        if glob:
            cmd += ["-g", str(glob)]
        cmd += ["--", str(path)]
    else:
        cmd = ["grep", "-rn", "-E", "--color=never", "--exclude-dir=.git",
               "-e", str(pattern)]
        if context:
            cmd += ["-C", str(context)]
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
    if not lines:
        return f"[no matches for {pattern!r} under {path}]"
    shown = "\n".join(lines[:max_results])
    if len(lines) > max_results:
        shown += f"\n... [{len(lines) - max_results} more matches; raise max_results to see more]"
    return shown


def _run_git(args, path):
    if not shutil.which("git"):
        return None, "ERROR: git not found on PATH"
    try:
        proc = subprocess.run(["git", "-C", str(path)] + args,
                               capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return None, "ERROR: git command timed out after 15s"
    except OSError as e:
        return None, f"ERROR: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        return None, f"ERROR: {(proc.stderr or proc.stdout).strip() or 'git command failed'}"
    return proc.stdout, None


def git_status(path=".", **_):
    """`git status --short -b`, read-only like `git diff`/`git log` — never
    goes through the approval / risk classifier (same rationale as
    `search_files`: a dedicated native tool beats shelling out via `run_bash`
    just to earn a risk-classifier round-trip for a command that's always
    read-only anyway)."""
    out, err = _run_git(["status", "--short", "-b"], path)
    if err:
        return err
    return out.strip() or "[clean working tree]"


def git_diff(path=".", staged=False, file=None, **_):
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
    if err:
        return err
    return out.strip() or "[no changes]"
