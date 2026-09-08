"""Private filesystem writes — the one place that decides how nx creates files.

Everything nx writes under `~/.nx` can carry secrets: session transcripts hold
tool output and command results, background logs hold whatever a command printed
(image payloads can be a screenshot of someone's terminal), and the overflow
spill holds the COMPLETE untruncated text of oversized tool results. On a shared
host a group-readable file is a leak, and the process umask is not a control —
it is whatever the caller's shell happened to set.

So: directories are created 0700 and re-chmodded (an existing directory created
by an older nx, or by `mkdir` in a stray shell, keeps its old mode), and files
are created 0600 via `os.open`, which applies the mode at creation — before any
byte exists — rather than chmod-after-write.

`_sessions_fs`, `_input_history`, `_images`, `_logging`, `_memory_cmd`, the
sub-agent report writer, the approval allowlist store and the memory archive
all grew their own version of this before 0.40; they now import it from here
(the last two keep a mode-literal only where they hand-craft something the
helpers don't cover, e.g. the archive's nested dir move).
"""
from __future__ import annotations

import os

DIR_MODE = 0o700
FILE_MODE = 0o600


def private_dir(path: str) -> str:
    """Create `path` (and parents) as a private directory; return `path`.

    The explicit chmod is not redundant: `os.makedirs(mode=...)` only applies to
    the leaf AND is masked by umask, and an already-existing directory keeps
    whatever mode it was created with — including a group-writable one from a
    pre-0.40 nx run.
    """
    os.makedirs(path, exist_ok=True, mode=DIR_MODE)
    try:
        os.chmod(path, DIR_MODE)
    except OSError:
        pass  # a read-only mount or a foreign-owned dir: still usable for reads
    return path


def open_private(path: str, flags: int = os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                 text_mode: str = "w", encoding: str = "utf-8"):
    """Open `path` for writing with 0600 regardless of umask.

    Caller closes it (it is a real file object). Mirrors the signature
    `_sessions_fs._open_private` had before this moved here."""
    fd = os.open(path, flags, FILE_MODE)
    return os.fdopen(fd, text_mode, encoding=encoding)


def touch_private(path: str) -> str:
    """Create an empty 0600 `path` if absent (and make its parent private).

    For files that something ELSE will write — a shell redirection in a
    `run_bash` background log, for instance, which `sh` creates with the umask
    unless the file already exists. Pre-creating it here is what keeps the log
    private without changing the command we hand to the shell.
    """
    parent = os.path.dirname(path)
    if parent:
        private_dir(parent)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, FILE_MODE)
    os.close(fd)
    return path


def atomic_write_private(path: str, data: str) -> None:
    """Atomically write `data` to `path` at 0600 (temp file in the same dir +
    os.replace, so a concurrent reader never sees a half-written file). The
    pattern _sessions_fs and _overflow hand-roll; new sinks should call this."""
    parent = os.path.dirname(path)
    if parent:
        private_dir(parent)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open_private(tmp) as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
