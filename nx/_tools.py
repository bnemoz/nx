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
from ._overflow import OVERFLOW_READ_MARKER, is_overflow_path


def _count_tail_lines(path: str, skip: int, cap: int) -> tuple[int, bool]:
    """Count the lines of `path` after the first `skip` lines, streaming and
    O(1) memory. Returns (count, exact): exact=False means the count was cut
    short (cap reached, or the file became unreadable), so `count` is only a
    lower bound. Used to total the line count in read_file's paging header
    without ever buffering the whole file."""
    try:
        with open(path, encoding="utf-8") as f:
            for _ in range(skip):
                if not f.readline():
                    return 0, True
            n = 0
            for _ in f:
                n += 1
                if n > cap:
                    return cap, False
            return n, True
    except (OSError, UnicodeDecodeError):
        return 0, False


# Window ceiling. A model that keeps paging a huge file with ever-larger
# limits (or offset++ scans) is the single fastest way to fill a context
# window; the ceiling makes that strategy impossible rather than merely
# discouraged. Set NX_READ_FILE_MAX_LINES=0 to disable.
READ_FILE_MAX_LINES_DEFAULT = 25000


def _read_file_max_lines() -> int:
    return _env_int("NX_READ_FILE_MAX_LINES", READ_FILE_MAX_LINES_DEFAULT)


def read_file(path: str | list[str], offset: int = 1, limit: int | None = None, **_: Any) -> str:
    """Read a file as numbered lines (1-based, `cat -n` style: right-aligned
    line number, a tab, then the line). Large files return only a window — pass
    offset (start line) and limit (max lines) to page through the rest. A header
    announces the visible range + total line count so the model can ask for more
    instead of being head/tail-capped by _sanitize_tool_result.

    Only the requested window is held in memory; the file is streamed, never
    read whole. The header's total comes from a bounded streaming tail scan
    (NX_READ_FILE_SCAN_LINES lines, default 200000) — beyond it the header
    reports a lower bound as "N+" rather than reading an unbounded file.

    path may also be a list of paths; each is read with the same offset/limit
    and the results are concatenated under a "==> path <==" header per file."""
    if isinstance(path, list):
        return "\n".join(
            f"==> {p} <==\n{read_file(p, offset=offset, limit=limit)}" for p in path
        )
    try:
        start = max(1, int(offset)) - 1
    except (TypeError, ValueError):
        start = 0
    if limit is None:
        limit = _env_int("NX_READ_FILE_LINES", 400)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 400
    # Window ceiling: clamp oversized limits instead of honoring them. The
    # header tells the model exactly how to proceed (target a line via
    # search_files) rather than letting it brute-force with bigger windows.
    max_lines = _read_file_max_lines()
    if max_lines > 0 and limit > max_lines:
        return (f"[read_file: limit {limit} exceeds the {max_lines}-line window "
                f"ceiling (NX_READ_FILE_MAX_LINES). Do NOT page through a file "
                f"this way -- each window is context you pay for. Find the "
                f"section you need with search_files(pattern, path=\"{path}\", "
                f"context=N) and read_file(path, offset=<hit line>) just that "
                f"part, or raise NX_READ_FILE_MAX_LINES if you truly need the "
                f"whole file.]")
    if path.lower().endswith(".pdf"):
        # A PDF read through read_file used to fall into the UTF-8 error path,
        # which models read as "try harder" — a 50K-page text binary read
        # through 500 pages of `head` is how a context window dies. Route to
        # the markdown extractor (page-capped, never raises) instead.
        from ._pdf import extract_pdf
        return extract_pdf(path)
    out: list[str] = []
    skipped = 0
    try:
        with open(path, encoding="utf-8") as f:
            while skipped < start and f.readline():
                skipped += 1
            if skipped == start:
                for line in f:  # one line in memory at a time
                    out.append(f"{start + len(out) + 1:6d}\t{line}")
                    if 0 < limit <= len(out):
                        break
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    except UnicodeDecodeError:
        return f"ERROR: {path} is not a valid UTF-8 text file (binary or wrong encoding)"
    if not out:
        if skipped == 0:  # any nonempty file skips >= 1 line
            return f"[{path}: empty file]"
        return f"[{path}: {skipped} lines; offset {start + 1} is past end of file]"
    body = "".join(out)
    if not body.endswith("\n"):
        body += "\n"
    end = start + len(out)
    truncated = 0 < limit <= len(out)  # stopped at limit, EOF not yet seen
    total: int | None = None
    plus = ""
    if truncated:
        cap = _env_int("NX_READ_FILE_SCAN_LINES", 200000)
        tail, exact = _count_tail_lines(path, end, cap)
        if exact:
            if tail == 0:
                truncated = False  # the window was the whole rest of the file
            else:
                total = end + tail
        else:
            total = end + cap  # lower bound only
            plus = "+"
    if start == 0 and not truncated:
        # A whole-file read: if this is an overflow spill file, tag it so
        # _sanitize_tool_result serves it verbatim instead of re-spilling the
        # read (pointer-chasing: the stub of a stub points at a fresh file).
        if is_overflow_path(path):
            return _OVERFLOW_READ_MARKER + body
        return body  # whole file (or whole rest of file) — no header
    if total is None:
        total = end  # window with start>0 reached EOF
    tip = ""
    if is_overflow_path(path):
        # Same anti-respill tag for a windowed read; see the whole-file branch.
        tip = ""
    elif start == 0:
        tip = (" Looking for one function/section rather than the whole "
               "file? STOP paging -- call search_files(pattern, context=N) "
               "instead, it finds it in one shot.")
    else:
        # Escalating steer: a mid-file window usually means the model is
        # walking the file with incrementing offsets, which is how a big file
        # eats a context window. Name the alternative and the exact next move.
        tip = (f" Paging mid-file. If you are walking toward a target, STOP -- "
               f"one search_files(pattern, path=\"{path}\") call lands on the "
               f"right offset; then read_file(path, offset=<hit line>, limit=N) "
               f"just that section.")
    return (f"[{path}: lines {start + 1}-{end} of {total}{plus}; "
            f"call read_file with offset/limit to page.{tip}]\n"
            + (_OVERFLOW_READ_MARKER if is_overflow_path(path) else "")
            + body)


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


def _abs_to_rel(a: str, run_dir: str) -> str:
    """Turn an absolute pathspec into one relative to run_dir; git rejects
    absolute pathspecs as outside the repository. Unresolvable paths are
    returned untouched so git reports them itself."""
    if not os.path.isabs(a):
        return a
    try:
        rel = os.path.relpath(a, run_dir)
    except ValueError:      # different drive (Windows)
        return a
    return a if rel.startswith("..") else rel


def _anchor_file(path: str, file: str | None) -> str:
    """Where to run a git command that names an absolute `file`. An explicit
    directory `path` that contains the file wins (a subrepo/subdir may be
    meant); otherwise — including the default `path="."`, which may not be a
    repository at all — anchor to the file's own directory. Relative `file`
    args are left alone."""
    f = str(file or "")
    if not f or not os.path.isabs(f):
        return path
    p = path or "."
    if p != "." and os.path.isdir(p):
        try:
            inside = not os.path.relpath(f, p).startswith("..")
        except ValueError:
            inside = False
        if inside:
            return p
    return os.path.dirname(f)


def _run_git(args: list[str], path: str) -> tuple[str | None, str | None]:
    if not shutil.which("git"):
        return None, "ERROR: git not found on PATH"

    # Models keep passing a FILE where a directory belongs (git_diff with
    # path=src/x.py instead of file=src/x.py); `git -C <file>` hard-fails with
    # "cannot change to ...: Not a directory". Run inside the file's directory
    # and restrict the command to that file so the intent still lands.
    run_dir = str(path) or "."
    if os.path.isfile(os.path.abspath(run_dir)):
        fname = os.path.basename(os.path.abspath(run_dir))
        run_dir = os.path.dirname(os.path.abspath(run_dir))
        if "--" not in args:
            args = [*args, "--", fname]
    i = args.index("--") if "--" in args else -1
    if i >= 0:
        args = [*args[:i + 1], *(_abs_to_rel(a, run_dir) for a in args[i + 1:])]
    try:
        proc = subprocess.run(["git", "-C", run_dir, *args],
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
      path: repo/directory to run inside (default ".") — a directory, not a
        file; to scope the diff use `file` instead.
      staged: pass True for `git diff --staged` (staged changes only).
      file: optional path to restrict the diff to a single file.
    """
    out, err = _run_git(["diff"] + (["--staged"] if staged else [])
                        + (["--", str(file)] if file else []),
                        _anchor_file(path, str(file) if file else None))
    if err or out is None:
        return err or "[no changes]"
    return out.strip() or "[no changes]"


def git_log(path: str = ".", max_count: int = 20, file: str | None = None, **_: Any) -> str:
    """`git log --oneline` (one commit per line: short hash + subject),
    read-only -- see `git_status` docstring for why this is a native tool
    rather than `run_bash("git log")`.

    Args:
      path: repo/directory to run inside (default ".") — a directory, not a
        file; to scope history use `file` instead.
      max_count: cap on commits returned (default 20).
      file: optional path to restrict history to a single file.
    """
    out, err = _run_git(["log", "--oneline", f"-{max(1, int(max_count))}"]
                        + (["--", str(file)] if file else []),
                        _anchor_file(path, str(file) if file else None))
    if err or out is None:
        return err or "[no commits]"
    return out.strip() or "[no commits]"
