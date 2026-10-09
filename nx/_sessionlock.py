"""Cross-session repo advisory lock (phase 2 of session isolation, 0.103.0).

Two nx REPLs started in the same repo share one working tree, one git index
and one `<repo>/.nx/` state dir — the second session's agents clobber the
first's in-progress work. This module keeps an advisory presence file per
repo: `<repo>/.nx/session.lock`, holding the claiming session id + pid.

- acquire at REPL startup (and on every /resume into this repo): the lock
  file is held open (flock on POSIX, msvcrt byte-range lock on Windows — the
  split lives in nx/_platform.py), so a crashed session's lock evaporates
  with the process.
- a second session in the SAME repo fails the non-blocking probe, prints who
  holds the repo, and the REPL asks Y/n to continue anyway (read-only-ish
  posture is the user's choice — nothing is technically enforced; the lock is
  a warning gate, not a fence).
- release on clean REPL exit; the OS auto-releases on crash.

The lock file lives under `<repo>/.nx/` so it never leaves a stray file at
the repo root and is naturally per-project.
"""
from __future__ import annotations

import json
import os

from . import _platform
from . import _state as st


def _lock_path() -> str:
    root = st.PROJECT_ROOT or os.getcwd()
    # PROJECT_ROOT may be the repo subdir of the worktree base; realpath so a
    # trailing slash or symlinked tmp dir doesn't fork the lock location.
    return os.path.join(os.path.realpath(root), ".nx", "session.lock")


def _close_fd(fd: int) -> None:
    """Close an fd whose lock attempt failed, ignoring a double close.

    POSIX try_lock_fd leaves the fd open on failure; the Windows path already
    closed it (msvcrt locks are released by closing the fd), so the second
    close raises EBADF — harmless, and swallowed rather than escaping as a
    lock bookkeeping failure."""
    try:
        os.close(fd)
    except OSError:
        pass


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

    The probe is platform-agnostic (_platform.try_lock_fd) — it reports a
    genuine contention (POSIX EACCES/EAGAIN) and a filesystem that refuses
    the lock outright the same way. Both mean "this process holds nothing",
    but they differ in what the caller may do: a real holder is only ever
    *warned about* (the prompt), and an unacquirable lock must never fence a
    session out — exactly the old non-EACCES/EAGAIN degrade, folded into the
    same contended branch because try_lock_fd's OSError → False is
    indistinguishable from EAGAIN.
    """
    path = _lock_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return True  # can't even create the file — degrade to unlocked
    try:
        if not _platform.try_lock_fd(fd):
            # Contended (EACCES/EAGAIN) or the platform refused the lock —
            # either way we hold nothing. Read the description file (it names
            # the other session, or nothing) and let the caller decide; the
            # non-EACCES/EAGAIN case used to return True (unlocked) here, and
            # the contended one prompted. Prompting is the safe superset: an
            # absent ask_confirm still answers False rather than sharing
            # silently, and _HELD stays empty so release() touches nothing.
            holder = _read_holder(path)
            _close_fd(fd)
            if ask_confirm is None:
                return False
            return bool(ask_confirm(
                f"another nx session already holds this repo ({holder}). "
                "Concurrent edits can clobber each other's work. Continue "
                "anyway?"))
        # Hold the lock for the process lifetime on a DEDICATED fd we keep
        # open; write the description file for humans/readers.
        holder = {"session_id": st.SESSION_ID, "pid": os.getpid(),
                  "cwd": os.getcwd()}
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(holder) + "\n")
        except OSError:
            pass
        # leak fd deliberately: keeping it open IS the lock (on Windows the
        # msvcrt byte lock dies when that fd closes too). Registered for
        # release() so /resume into another repo can drop it.
        _HELD.append(fd)
        _HELD_PATHS.append(path)
        return True
    except Exception:
        _close_fd(fd)
        return True  # lock bookkeeping must never block a session


_HELD: list[int] = []
_HELD_PATHS: list[str] = []


def release(unlink: bool = True) -> None:
    """Drop any locks this process holds (clean REPL exit / repo switch).

    `unlink=False` (repo switch, 0.129.0): drop OUR flock and fd but leave the
    files on disk. Between our unlock_fd and our unlink a waiter can win the
    flock on the same inode; unlinking then deletes the file out from under
    its NEW holder, so a third process would create a fresh file and lock it
    while the second still holds an orphaned inode. Leaving the stale
    session.lock costs one cosmetic wrong "holder" line later; the race
    costs a silently unenforced lock."""
    while _HELD:
        fd = _HELD.pop()
        path = _HELD_PATHS.pop()
        # unlock_fd releases AND closes the fd on both platforms (POSIX: flock
        # LOCK_UN then close; Windows: the close is the release), so there is
        # no separate close here.
        _platform.unlock_fd(fd)
        if not unlink:
            continue
        try:
            os.unlink(path)
        except OSError:
            pass
        try:
            os.unlink(path + ".lock")
        except OSError:
            pass


def reacquire(ask_confirm=None) -> bool:
    """Re-claim the repo lock after a cwd change (/resume into another repo,
    0.129.0). No-op unless this process already holds a lock — the startup
    `nx --resume` path runs resume_session BEFORE the first acquire, so _HELD
    is empty there and the normal startup acquire in _repl.py does the work.
    Otherwise: drop the OLD repo's lock without unlinking its files (see
    release), then acquire() against the freshly restored cwd/PROJECT_ROOT.
    Same contract as acquire: never fatal, degrades open."""
    if not _HELD:
        return True
    try:
        release(unlink=False)
    except Exception:
        pass
    try:
        return acquire(ask_confirm=ask_confirm)
    except Exception:
        return True  # lock bookkeeping must never block a session
