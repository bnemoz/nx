"""Cross-session repo advisory lock (phase 2 of session isolation, 0.103.0).

Two nx REPLs started in the same repo share one working tree, one git index
and one `<repo>/.nx/` state dir — the second session's agents clobber the
first's in-progress work. This module keeps an advisory presence file per
repo: `<repo>/.nx/session.lock`, holding the claiming session id + pid.

- acquire at REPL startup (and on every /resume into this repo): flock'd
  while held, so a crashed session's lock evaporates with the process.
- a second session in the SAME repo fails the non-blocking probe, prints who
  holds the repo, and the REPL asks Y/n to continue anyway (read-only-ish
  posture is the user's choice — nothing is technically enforced; the lock is
  a warning gate, not a fence).
- release on clean REPL exit; flock auto-releases on crash.

The lock file lives under `<repo>/.nx/` so it never leaves a stray file at
the repo root and is naturally per-project.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os

from . import _state as st


def _lock_path() -> str:
    root = st.PROJECT_ROOT or os.getcwd()
    # PROJECT_ROOT may be the repo subdir of the worktree base; realpath so a
    # trailing slash or symlinked tmp dir doesn't fork the lock location.
    return os.path.join(os.path.realpath(root), ".nx", "session.lock")


def _read_holder(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        sid = data.get("session_id") or "?"
        pid = data.get("pid") or "?"
        cwd = data.get("cwd") or "?"
        return f"session {sid} (pid {pid}, cwd {cwd})"
    except (OSError, ValueError):
        return "another session"


def acquire(ask_confirm=None) -> bool:
    """Claim the repo's advisory lock. Returns True when this session holds
    it (or the user chose to share it anyway); False when it must not run
    here (holder present AND the caller had no way to ask).

    `ask_confirm` is a `prompt: str -> bool` callable (the REPL's _confirm);
    when absent, a contended repo refuses instead of prompting.
    """
    path = _lock_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return True  # can't even create the file — degrade to unlocked
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                return True
            holder = _read_holder(path)
            os.close(fd)
            if ask_confirm is None:
                return False
            return bool(ask_confirm(
                f"another nx session already holds this repo ({holder}). "
                "Concurrent edits can clobber each other's work. Continue "
                "anyway?"))
        # Hold the flock for the process lifetime on a DEDICATED fd we keep
        # open; write the description file for humans/readers.
        holder = {"session_id": st.SESSION_ID, "pid": os.getpid(),
                  "cwd": os.getcwd()}
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(holder) + "\n")
        except OSError:
            pass
        # leak fd deliberately: keeping it open IS the lock. Registered for
        # release() so /resume into another repo can drop it.
        _HELD.append(fd)
        _HELD_PATHS.append(path)
        return True
    except Exception:
        os.close(fd)
        return True  # lock bookkeeping must never block a session


_HELD: list[int] = []
_HELD_PATHS: list[str] = []


def release() -> None:
    """Drop any locks this process holds (clean REPL exit / repo switch)."""
    while _HELD:
        fd = _HELD.pop()
        path = _HELD_PATHS.pop()
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        try:
            os.unlink(path + ".lock")
        except OSError:
            pass
