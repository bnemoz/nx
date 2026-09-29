"""File-mutation tool bodies: write_file, edit_file, multi_edit, sed_edit.

Split out of __init__.py as a focused submodule for the tools that write to
disk (as opposed to nx/_tools.py's read-only ones). Each still goes through
_confirm()/risk classification from _approval.py.

Every mutating edit first crosses nx/_garble.py's garble_check(): the `new`
payload is scanned for the duplicated-line/repeated-phrase/spliced-boundary
corruption quantized models produce, repaired when unambiguous, hard-errored
when not (see that module's docstring).

`_confirm` is monkeypatched by tests via `m._confirm = fake` (the top-level
`nx` module attribute), so it's called through a lazy `import inspect
import nx as _m;
_m._confirm(...)` rather than bound directly at import time — see the note
at the top of _repl.py for the same pattern.
"""
from __future__ import annotations

import glob
import inspect
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from ._ansi import DIM, RESET
from ._approval import _diff_preview, _diff_hunk
from ._garble import garble_check, swapped_args
from ._tools import _strip_line_numbers
from ._undo import push_checkpoint as _undo_push, snapshot as _undo_snapshot


def _confirm(action: str, preview: str | None = None) -> bool:
    import nx as _m
    return _m._confirm(action, preview=preview)


def _atomic_write(path: str, content: str) -> None:
    """Write *content* to *path* without ever leaving a truncated file on
    disk: write to a sibling temp file first, then rename over the target.
    `os.replace` is atomic on POSIX, so a crash mid-write leaves either the
    old file or the new one, never a half-written one.

    A symlink target is resolved first so the link is followed and the real
    file rewritten in place (replacing the link itself would silently defeat
    symlink-farm dotfile management). The existing file's permission bits are
    carried over too — mkstemp creates 0600, which would otherwise strip e.g.
    the executable bit off a script on every edit."""
    # Follow symlinks: write through to the real file, never replace the link.
    real = os.path.realpath(path)
    d = os.path.dirname(real) or "."
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".nx-tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        try:
            mode = os.stat(real).st_mode & 0o7777
        except OSError:
            pass  # new file: keep mkstemp's 0600
        else:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, real)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_with_mtime(path: str) -> tuple[str, float | None]:
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        mtime = None
    with open(path) as f:
        return f.read(), mtime


def _staleness_error(path: str) -> str | None:
    """Cross-session staleness refusal (nx/_staleness.py): the file changed
    after this session last READ it — likely another session or an editor.
    Silently skipped when there's no read record (a never-read file writes
    freely; write_file creating a new file has no record by construction)."""
    from ._staleness import staleness_error
    return staleness_error(path)


def write_file(path: str, content: str, **_: Any) -> str:
    stale = _staleness_error(path)
    if stale:
        return stale
    try:
        before, before_mtime = _read_with_mtime(path)
    except (FileNotFoundError, IsADirectoryError):
        before, before_mtime = "", None
    if not _confirm(f"write {path} ({len(content)} bytes)", preview=_diff_preview(before, content)):
        return "DENIED by user"
    try:
        after_mtime = os.stat(path).st_mtime
    except OSError:
        after_mtime = None
    if after_mtime != before_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
    _atomic_write(path, content)
    return f"wrote {len(content)} bytes to {path}"


def _apply_line_range(src: str, start: int, end: int, new: str) -> tuple[str, int, int]:
    """Replace the 1-based inclusive line range [start, end] with `new` text.
    Returns (new_src, replaced_count, new_line_count). Raises ValueError when
    the range is outside the file."""
    lines = src.split("\n")
    n = len(lines)
    if src.endswith("\n"):
        n -= 1  # the trailing element after the final \n is '', not a line
    if start < 1 or end < start or end > n:
        raise ValueError(f"line range {start}-{end} outside 1..{n}")
    # Preserve the file's trailing-newline convention.
    had_nl = src.endswith("\n")
    prefix = "\n".join(lines[:start - 1])
    suffix = "\n".join(lines[end:])
    new_body = new.rstrip("\n")
    out = "\n".join([prefix, new_body, suffix]) if prefix or suffix else new_body
    # normalize: ensure exactly the original trailing-newline shape
    if had_nl and not out.endswith("\n"):
        out += "\n"
    elif not had_nl and out.endswith("\n"):
        out = out[:-1]
    return out, end - start + 1, new_body.count("\n") + 1


def _whitespace_tolerant_match(src: str, old: str) -> str | None:
    """Find the unique region of `src` matching `old` modulo leading/trailing
    whitespace and blank-line count — the common model failure of getting
    indentation wrong. Returns the exact source slice to replace, or None.
    Only tried when the exact match already failed, and only accepted when it
    matches exactly ONE region (ambiguity never guesses)."""
    if not old.strip():
        return None
    old_lines = old.strip("\n").split("\n")
    sig = [(ln.strip(), bool(ln.strip())) for ln in old_lines]
    sig = [s for s in sig if s[1]]  # ignore blank lines for matching
    if not sig:
        return None
    src_lines = src.split("\n")
    sig = [s[0] for s in sig]  # content lines only (blanks already dropped)
    hits = []
    for i in range(len(src_lines)):
        # the window must contain the content lines contiguously, with blank
        # lines allowed between/around them (the model may include blanks the
        # source has more/fewer of)
        window = src_lines[i:i + len(sig) + 4]
        if len(window) < len(sig):
            break
        wi = 0
        ok = True
        for s in sig:
            # skip up to 4 blank lines in the window between content lines
            while wi < len(window) and window[wi].strip() == "" and wi < len(sig) + 3:
                wi += 1
            if wi >= len(window) or window[wi].strip() != s:
                ok = False
                break
            wi += 1
        if ok and wi == len(sig) + sum(1 for w in window[:wi] if w.strip() == ""):
            hits.append((i, i + wi))
    if len(hits) != 1:
        return None
    i, j = hits[0]
    return "\n".join(src_lines[i:j])


def edit_file(path: str, old: str = "", new: str = "", replace_all: bool = False,
              old_lines: str = "", **_: Any) -> str:
    """Replace `old` with `new` in `path`. By default `old` must match exactly
    once (ambiguous matches error rather than guessing). Pass replace_all=True
    to replace every occurrence instead — e.g. renaming a variable throughout
    a file — without needing multi_edit for a single literal string.

    Alternative targeting: pass `old_lines="10-14"` (1-based, inclusive, the
    numbering read_file prints) instead of `old` to replace those lines with
    `new` directly — no exact-match hunting, useful after read_file paging.
    `old_lines` and `old` are mutually exclusive.

    Fallback when the exact match fails uniquely at the wrong indentation:
    the edit applies whitespace-tolerantly (leading/trailing indentation
    ignored, blank-line runs collapsed) and the result says so, so a model
    that mis-indents doesn't have to re-read the file and retry."""
    src, src_mtime = _read_with_mtime(path)
    stale = _staleness_error(path)
    if stale:
        return stale
    ws_fallback_used = False
    garble_note = ""
    if old_lines:
        if old:
            return "ERROR: pass either `old` or `old_lines`, not both"
        try:
            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(old_lines))
            if not m:
                m2 = re.fullmatch(r"\s*(\d+)\s*", str(old_lines))
                if not m2:
                    raise ValueError
                start = end = int(m2.group(1))
            else:
                start, end = int(m.group(1)), int(m.group(2))
        except ValueError:
            return "ERROR: old_lines wants 'N' or 'N-M' (1-based lines, e.g. '10-14')"
        status, msg, repaired = garble_check(new, src, path=path)
        if status == "error":
            return msg or "ERROR: garble guard rejected the edit"
        if status == "repaired":
            assert repaired is not None and msg is not None
            new = repaired
            garble_note = msg
        try:
            new_src, _n_repl, _n_lines = _apply_line_range(src, start, end, new)
        except ValueError as e:
            return f"ERROR: {e}"
        if new_src == src:
            return f"ERROR: no change to {path} (new content identical to lines {start}-{end})"
        action = f"edit {path} (lines {start}-{end})"
        if not _confirm(action, preview=_diff_preview(src, new_src)):
            return "DENIED by user"
        try:
            now_mtime = os.stat(path).st_mtime
        except OSError:
            now_mtime = None
        if now_mtime != src_mtime:
            return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
        _atomic_write(path, new_src)
        return (f"edited {path} (replaced lines {start}-{end})\n{_diff_hunk(src, new_src)}"
                + (f"\n{garble_note}" if garble_note else ""))
    if not isinstance(old, str) or not old:
        return "ERROR: provide `old` (text to find) or `old_lines` (line range)"
    if not replace_all and src.count(old) != 1:
        # The model may have pasted read_file's numbered output into `old`.
        # Only as a fallback (exact match already failed), retry with the
        # numbering stripped — and strip `new` the same way so we never write
        # line numbers into the file. Normal edits hit the exact match above
        # and never reach here, so genuine content is never touched.
        stripped_old = _strip_line_numbers(old)
        if stripped_old != old and src.count(stripped_old) == 1:
            old, new = stripped_old, _strip_line_numbers(new)
    count = src.count(old)
    if count == 0:
        # Last-resort whitespace-tolerant match (unique only) — see docstring.
        ws_slice = _whitespace_tolerant_match(src, old)
        if ws_slice is not None:
            old = ws_slice
            new = _strip_line_numbers(new)
            ws_fallback_used = True
            count = 1
        else:
            hint = swapped_args(new, src, old)
            if hint:
                return hint
            return "ERROR: `old` matched 0 times (need at least 1)"
    if not replace_all and count != 1:
        return f"ERROR: `old` matched {count} times (need exactly 1, or pass replace_all=True)"
    status, msg, repaired = garble_check(new, src, old, path=path)
    if status == "error":
        return msg or "ERROR: garble guard rejected the edit"
    if status == "repaired":
        new = repaired
        garble_note = msg
    new_src = src.replace(old, new) if replace_all else src.replace(old, new, 1)
    if new_src == src:
        return f"ERROR: no change to {path} (`old` and `new` are identical)"
    action = f"edit {path}" + (f" ({count} occurrences)" if replace_all and count > 1 else "")
    if not _confirm(action, preview=_diff_preview(src, new_src)):
        return "DENIED by user"
    try:
        now_mtime = os.stat(path).st_mtime
    except OSError:
        now_mtime = None
    if now_mtime != src_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
    _atomic_write(path, new_src)
    note = "\n(edit applied whitespace-tolerantly — `old`'s indentation was normalized to match)" if ws_fallback_used else ""
    if garble_note:
        note += f"\n{garble_note}"
    return f"edited {path}\n{_diff_hunk(src, new_src)}{note}"




# --- action fusion (0.67.0, SoL-Pi port) -------------------------------------
# A call carrying `then_run` (NX_FUSION, default ON) runs its follow-up validation
# command in the same tool call via nx/_fusion.run_fused. Declared here (not
# in _dispatch) so DISPATCH[name] stays the same object the schema-introspection
# tests and the sub-agent write tier see, and so the wrapper accepts `then_run`
# as a real named parameter.
def _fused_body(body):
    real_params = {n for n in inspect.signature(body).parameters}
    def _inner(*pos: Any, then_run: dict[str, Any] | None = None, **kw: Any):
        # preserve the body's positional signature (edit_file(path, old, new))
        path = pos[0] if pos else kw.pop("path", "")
        rest = pos[1:]
        if not then_run and not _fusion_enabled():
            return body(path, *rest, **kw)
        if not then_run:
            return body(path, *rest, **kw)
        params = list(inspect.signature(body).parameters)[1:]
        call_kw = dict(zip(params, rest)) if rest else {}
        call_kw.update(kw)
        call_kw.pop("then_run", None)
        args = {"path": path, "then_run": then_run, **call_kw}
        return _fusion_run_fused(args, lambda: body(path, **call_kw))
    # expose the body's real parameter names (and then_run) to introspection
    _inner.__wrapped_signature_params__ = real_params | {"then_run"}
    _inner.__name__ = getattr(body, "__name__", "fused")
    _inner.__doc__ = getattr(body, "__doc__", None)
    return _inner


from ._fusion import run_fused as _fusion_run_fused, fusion_enabled as _fusion_enabled  # noqa: E402

def multi_edit(path: str, edits: list[dict[str, str]], **_: Any) -> str:
    """Apply multiple exact find-and-replace edits to one file atomically:
    either every edit lands or none do (a failed match rolls back the whole
    batch — the file is only written after all edits match). Each edit is an
    object {"old": <str>, "new": <str>}; `old` must match exactly once in the
    running text (edits apply in list order, so a later edit may target text an
    earlier edit produced). Pasted `read_file` line-number prefixes are stripped
    automatically, like `edit_file`. One approval covers the whole batch — a
    five-line refactor becomes one tool call, not five.
    """
    src, src_mtime = _read_with_mtime(path)
    stale = _staleness_error(path)
    if stale:
        return stale
    if not isinstance(edits, list) or not edits:
        return "ERROR: `edits` must be a non-empty list of {old, new} objects"
    garble_notes: list[str] = []
    work = src
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"ERROR: edit #{i + 1} is not an object"
        old = edit.get("old")
        new = edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return f"ERROR: edit #{i + 1} needs string `old` and `new`"
        # Tolerate pasted read_file line numbers exactly like edit_file.
        if work.count(old) != 1:
            stripped = _strip_line_numbers(old)
            if stripped != old and work.count(stripped) == 1:
                old, new = stripped, _strip_line_numbers(new)
        count = work.count(old)
        if count != 1:
            hint = "" if count else (swapped_args(new, work, old) or "")
            return (f"ERROR: edit #{i + 1} `old` matched {count} times "
                    f"(need exactly 1); no edits applied" + (f"\n{hint}" if hint else ""))
        status, msg, repaired = garble_check(new, work, old, path=path)
        if status == "error":
            detail = msg or "garble guard rejected the edit"
            if detail.startswith("ERROR: "):
                detail = detail[len("ERROR: "):]
            return f"ERROR: edit #{i + 1} {detail} no edits applied"
        if status == "repaired":
            assert repaired is not None and msg is not None
            new = repaired
            garble_notes.append(msg)
        work = work.replace(old, new, 1)
    if work == src:
        return f"no change to {path} (edits were no-ops)"
    if not _confirm(f"edit {path} ({len(edits)} edits)", preview=_diff_preview(src, work)):
        return "DENIED by user"
    try:
        now_mtime = os.stat(path).st_mtime
    except OSError:
        now_mtime = None
    if now_mtime != src_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-read and retry"
    _atomic_write(path, work)
    tail = "\n" + "\n".join(dict.fromkeys(garble_notes)) if garble_notes else ""
    return f"edited {path} ({len(edits)} edits applied)\n{_diff_hunk(src, work)}{tail}"


def delete_file(path: str, **_: Any) -> str:
    """Delete one file, with undo support: the snapshot pushed here is what
    /undo restores (a plain `rm` via run_bash bypasses the undo stack — this
    tool is the recoverable path). Refuses directories and dotfiles-at-home;
    goes through the normal approval gate."""
    if not path or path in (".", "..") or os.path.basename(path).startswith("."):
        return f"ERROR: refusing to delete {path!r} (need a concrete non-dotfile path)"
    if os.path.isdir(path):
        return f"ERROR: {path} is a directory — delete_file only removes files"
    stale = _staleness_error(path)
    if stale:
        return stale
    existed = os.path.exists(path)
    if not existed:
        return f"ERROR: {path} does not exist"
    snap = _undo_snapshot(path)
    # mtime BEFORE _confirm (0.47.0, ROADMAP 2.2): the "changed while awaiting
    # approval" guard used to take both of its stats after _confirm returned —
    # a tautology that could never fire. This is the pre-approval reading the
    # post-approval re-stat below is compared against (same pattern as
    # edit_file/multi_edit).
    try:
        before_mtime = os.stat(path).st_mtime
    except OSError:
        before_mtime = None
    if not _confirm(f"delete {path}"):
        return "DENIED by user"
    try:
        now_mtime = os.stat(path).st_mtime
    except OSError:
        now_mtime = None
    if now_mtime != before_mtime:
        return f"ERROR: {path} changed on disk while awaiting approval; re-check and retry"
    try:
        os.remove(path)
    except OSError as e:
        return f"ERROR: {type(e).__name__}: {e}"
    _undo_push("delete_file", [snap])
    return f"deleted {path} (restorable with /undo)"


# Extensions never worth opening as text; skipped before the read/decode
# attempt so a broad glob (e.g. "**/*") doesn't burn time round-tripping
# through UnicodeDecodeError on every image/archive/compiled artifact.
_BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac",
    ".db", ".sqlite", ".sqlite3",
})

# Diff previews beyond this many files are elided to a summary line so a
# broad glob match doesn't blow up the approval prompt.
_MAX_PREVIEW_FILES = 12


def sed_edit(pattern: str, replacement: str, path: str, count: int = 0, flags: str = "", **_: Any) -> str:
    """Regex find-and-replace (Python `re` syntax) over one file or a glob of
    files (e.g. "src/**/*.py", recursive `**` supported). `flags` is a string
    of single-letter re flags: i=IGNORECASE, m=MULTILINE, s=DOTALL. `count`
    caps replacements *per file* (not a total across matched files; 0 =
    unlimited). One approval covers every matched file, with a combined diff
    preview (truncated past the first few files).
    """
    try:
        reflags = 0
        for ch in flags:
            reflags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}.get(ch, 0)
        rx = re.compile(pattern, reflags)
    except re.error as e:
        return f"ERROR: invalid regex: {e}"
    files = sorted(glob.glob(path, recursive=True)) if glob.has_magic(path) else [path]
    files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() not in _BINARY_EXTS]
    if not files:
        return f"ERROR: no files matched {path}"
    from ._staleness import staleness_error as _se
    _stale = [f for f in files if _se(f)]
    if _stale:
        return ("ERROR: " + ", ".join(_stale) + " changed on disk since this session "
                "last read it (likely another session or editor) — re-read and "
                "retry; refusing to overwrite possibly-newer content")
    # replacement is always literal text, never a backreference template:
    # escape backslashes so `\1`, `\U`, a Windows path, etc. don't get
    # interpreted as re.subn's replacement-string syntax.
    literal_replacement = replacement.replace("\\", "\\\\")
    changes = []
    previews = []
    unreadable = []
    for f in files:
        try:
            src, mtime = _read_with_mtime(f)
        except (OSError, UnicodeDecodeError) as e:
            unreadable.append(f"{f}: {e}")
            continue
        new_src, n = rx.subn(literal_replacement, src, count=count or 0)
        if n == 0:
            continue
        diff = _diff_preview(src, new_src)
        if diff and len(changes) < _MAX_PREVIEW_FILES:
            previews.append(f"{DIM}--- {f} ---{RESET}\n{diff}")
        changes.append((f, src, new_src, n, mtime))
    if not changes:
        prefix = f"no matches for pattern in {len(files)} file(s)"
        return prefix + ("\nskipped (unreadable): " + ", ".join(unreadable) if unreadable else "")
    if len(changes) > _MAX_PREVIEW_FILES:
        previews.append(f"{DIM}... and {len(changes) - _MAX_PREVIEW_FILES} more file(s) not shown{RESET}")
    total = sum(n for _, _, _, n, _ in changes)
    action = f"sed_edit {len(changes)} file(s), {total} replacement(s): " + ", ".join(f for f, _, _, _, _ in changes)
    if not _confirm(action, preview="\n".join(previews)):
        return "DENIED by user"
    stale = []
    for f, _src, _new_src, _n, mtime in changes:
        try:
            now_mtime = os.stat(f).st_mtime
        except OSError:
            now_mtime = None
        if now_mtime != mtime:
            stale.append(f)
    if stale:
        return ("ERROR: " + ", ".join(stale) + " changed on disk while awaiting approval; "
                "re-read and retry (no files were modified)")
    # After the staleness abort (0.47.0, ROADMAP 2.2): nothing is pushed
    # unless the files were actually written. 0.103.29: pushed AFTER the
    # write loop so _stamp_after_sha sees the post-edit content and the
    # /undo guard (refuse to revert a file changed since the checkpoint)
    # can cover sed_edit too — a pre-write push would stamp the pre-edit
    # hash and make every sed_edit /undo refuse.
    ok: list[tuple[str, str, str]] = []
    for f, src, new_src, _n, _mtime in changes:
        try:
            _atomic_write(f, new_src)
            ok.append((f, src, new_src))
        except OSError:
            pass  # failed file: leave it out of the checkpoint, keep the rest
    if not ok:
        return "ERROR: no files could be written"
    _undo_push("sed_edit", [{"path": f, "existed": True, "content": src} for f, src, _new_src in ok])
    hunks = "\n".join(
        f"--- {f} ---\n{_diff_hunk(src, new_src)}" for f, src, new_src in ok)
    return (f"edited {len(ok)} file(s), {total} replacement(s): "
            + ", ".join(f"{f} ({n})" for f, _s, _ns, n, _m in changes
                        if any(f == of and _s == osrc for of, osrc, _ in ok))
            + "\n" + hunks)


def _patch_affected_files(diff_text: str) -> list[str]:
    files = []
    for line in diff_text.splitlines():
        m = re.match(r"^\+\+\+ (?:b/)?(\S+)", line)
        if m and m.group(1) != "/dev/null" and m.group(1) not in files:
            files.append(m.group(1))
    return files


def apply_patch(diff: str, path: str = ".", **_: Any) -> str:
    """Apply a unified diff (`diff -u`, `git diff`, or a hand-written patch) to
    every file it touches, atomically, in one call -- for a multi-file
    restructuring that's awkward to express as several multi_edit/sed_edit
    calls. Prefers `git apply`, falls back to the POSIX `patch` command if git
    is missing or the target isn't a git repo. Dry-run checked before the
    approval prompt so a patch that can't apply cleanly never gets a partial
    write.

    Args:
      diff: unified diff text (must have `--- a/x` / `+++ b/x` file headers).
      path: directory the patch paths are relative to (default ".").
    """
    if not diff or not diff.strip():
        return "ERROR: `diff` is empty"
    files = _patch_affected_files(diff)
    if not files:
        return "ERROR: no `--- a/x` / `+++ b/x` file headers found in diff"
    has_git = bool(shutil.which("git"))
    has_patch = bool(shutil.which("patch"))
    if not has_git and not has_patch:
        return "ERROR: neither `git` nor `patch` found on PATH"
    text = diff if diff.endswith("\n") else diff + "\n"
    fd, tmp_path = tempfile.mkstemp(prefix=".nx-patch-", suffix=".diff")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        backend = "git" if has_git else "patch"
        if backend == "git":
            check = subprocess.run(["git", "apply", "--check", tmp_path],
                                    cwd=path, capture_output=True, text=True)
            if check.returncode != 0 and has_patch:
                backend = "patch"
        if backend == "patch":
            check = subprocess.run(["patch", "--dry-run", "-p1", "-i", tmp_path],
                                    cwd=path, capture_output=True, text=True)
        if check.returncode != 0:
            return f"ERROR: patch does not apply cleanly: {(check.stderr or check.stdout).strip()}"
        preview = diff if len(diff) < 4000 else diff[:4000] + "\n... [truncated]"
        if not _confirm(f"apply_patch: {len(files)} file(s): {', '.join(files)}", preview=preview):
            return "DENIED by user"
        befores = [_undo_snapshot(os.path.join(path, f)) for f in files]
        if backend == "git":
            result = subprocess.run(["git", "apply", tmp_path], cwd=path, capture_output=True, text=True)
        else:
            result = subprocess.run(["patch", "-p1", "-i", tmp_path], cwd=path, capture_output=True, text=True)
        if result.returncode != 0:
            return f"ERROR: patch application failed: {(result.stderr or result.stdout).strip()}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    _undo_push("apply_patch", befores)
    return f"applied patch ({backend}) to {len(files)} file(s): {', '.join(files)}"


write_file = _fused_body(write_file)
edit_file = _fused_body(edit_file)
multi_edit = _fused_body(multi_edit)
