"""Cross-platform primitives for the POSIX/Windows split.

One small module so the platform branches stay out of the feature modules:
`fcntl`-style advisory file locking (msvcrt on Windows), a pid→log map
compatible pid-liveness probe, a kill-tree (the Windows analog of
os.killpg on a job's process group), and shell-preference detection.
No platform checks elsewhere import fcntl/termios directly anymore.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Iterator

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import msvcrt
else:
    import fcntl


# ------------------------------------------------------------- file locking ---

@contextlib.contextmanager
def exclusive_file_lock(fd: int, *, blocking: bool = True,
                        timeout: float | None = None) -> Iterator[bool]:
    """Hold an advisory interprocess exclusive lock on an open fd.

    Yields True when the lock was acquired, False when a non-blocking
    attempt found it held or `timeout` elapsed. `timeout` (seconds) bounds
    the wait: instead of blocking forever, the lock is retried non-blocking
    in a 50 ms poll loop and False is yielded once the budget is spent —
    callers degrade (proceed unlocked / skip) rather than hang. On Windows, msvcrt.locking() requires the
    file object it locks to stay open for the lifetime of the lock — the
    caller therefore passes a file-like object, not a raw fd, when it can.

    On POSIX this is fcntl.flock (LOCK_EX [| LOCK_NB]); on Windows it is
    msvcrt.locking (LK_NBLCK / LK_LOCK), which locks byte ranges. The lock
    always covers byte 0 — sufficient for a whole-file advisory gate, and
    the convention every nx lock file uses.
    """
    if IS_WINDOWS:
        # msvcrt.locking works on the current file position's byte range;
        # normalize to offset 0 and lock 1 byte.
        fobj = fd if hasattr(fd, "seek") else os.fdopen(fd, "a+", encoding="utf-8")
        owned = fobj is not fd
        try:
            fobj.seek(0)  # type: ignore[union-attr]  # guarded by hasattr above
            if timeout is not None:
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        # pyrefly: ignore[missing-attribute]  # Windows-only branch
                        msvcrt.locking(fobj.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            yield False
                            return
                        time.sleep(0.05)
            else:
                try:
                    msvcrt.locking(fobj.fileno(),  # pyrefly: ignore[missing-attribute]  # Windows-only branch
                                   # pyrefly: ignore [missing-attribute]
                                   (msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK), 1)
                except OSError:
                    yield False
                    return
            try:
                yield True
            finally:
                try:
                    fobj.seek(0)  # type: ignore[union-attr]  # guarded by hasattr above
                    msvcrt.locking(fobj.fileno(), msvcrt.LK_UNLCK, 1)  # pyrefly: ignore[missing-attribute]  # Windows-only branch
                except OSError:
                    pass
        finally:
            if owned:
                # pyrefly: ignore [missing-attribute]
                fobj.close()
        return

    # --- POSIX ---
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    if timeout is not None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, flags | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.05)
    else:
        try:
            fcntl.flock(fd, flags)
        except OSError:
            yield False
            return
    try:
        yield True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


def try_lock_fd(fd: int) -> bool:
    """One-shot non-blocking exclusive lock on an open fd (flock semantics).

    Returns True when acquired (and LEAVES it locked — the fd staying open
    IS the lock, exactly like _sessionlock.py's held-fd pattern). Returns
    False when another process holds it or the platform refuses. Unlock
    with unlock_fd()."""
    if IS_WINDOWS:
        try:
            fobj = os.fdopen(fd, "a+", encoding="utf-8")
        except OSError:
            return False
        try:
            fobj.seek(0)
            msvcrt.locking(fobj.fileno(), msvcrt.LK_NBLCK, 1)  # pyrefly: ignore[missing-attribute]  # Windows-only branch
            return True
        except OSError:
            fobj.close()
            return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def unlock_fd(fd: int) -> None:
    """Best-effort release of a lock taken by try_lock_fd."""
    if IS_WINDOWS:
        # The lock is released by closing the fd open at lock time; the
        # fd object lives in the caller's _HELD list — closing is enough.
        try:
            os.close(fd)
        except OSError:
            pass
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


# ------------------------------------------------------------- process mgmt ---

def pid_is_alive(pid: int, expected_start_time: str | None = None) -> bool:
    """Whether pid is alive (and, when a start time is known, still the same
    process). POSIX: os.kill(pid, 0). Windows: OpenProcess-based liveness —
    os.kill(pid, 0) is not implemented there."""
    if IS_WINDOWS:
        import ctypes
        SYNCHRONIZE = 0x00100000
        STILL_ACTIVE = 259
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pyrefly: ignore[missing-attribute]  # Windows-only branch
        h = k32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not h:
            return False
        try:
            code = ctypes.wintypes.DWORD(0)
            if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                # A reused pid that already exited reports a nonzero exit
                # code; a still-running (or zombie-unreaped) one reports 259.
                return code.value == STILL_ACTIVE
            return False
        finally:
            k32.CloseHandle(h)
    if expected_start_time is not None:
        current = proc_start_time(pid)
        if current is not None and current != expected_start_time:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


def proc_start_time(pid: int | str) -> str | None:
    """Linux's immutable start-time tick for pid (PID-reuse guard), else None."""
    if IS_WINDOWS:
        return None
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            stat = f.read()
        fields = stat.rsplit(") ", 1)[1].split()
        return fields[19]
    except (OSError, ValueError, IndexError):
        return None


def kill_process_tree(pid: int, sig_name: str = "TERM") -> bool:
    """Terminate a job's whole tree. POSIX: SIGTERM (then SIGKILL when
    sig_name == 'KILL') to the process group (the job runs in its own
    session, so the group covers its children). Windows: taskkill /T /F —
    there is no process-group kill; taskkill walks the pid tree instead.

    Returns True when the signal was delivered."""
    if IS_WINDOWS:
        flag = "/F"  # /T always force-terminates; a soft TERM doesn't exist
        try:
            r = subprocess.run(["taskkill", "/T", flag, "/PID", str(pid)],
                               capture_output=True, timeout=10)
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    import signal
    try:
        sig = getattr(signal, "SIG" + sig_name)
        os.killpg(os.getpgid(pid), sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def bash_launch_args() -> dict:
    """kwargs for subprocess launch of a *background* job (detached)."""
    if IS_WINDOWS:
        # No setsid on Windows; CREATE_NEW_PROCESS_GROUP detaches from the
        # console group so terminal Ctrl+C (which signals a group) can't
        # reach the job, and CREATE_NO_WINDOW stops a console flash.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP  # pyrefly: ignore[missing-attribute]  # Windows-only branch
                | subprocess.CREATE_NO_WINDOW}  # pyrefly: ignore[missing-attribute]  # Windows-only branch
    return {"start_new_session": True}


def shell_command(argv: list[str]) -> list[str] | str:
    """The command form subprocess.run expects for the platform's shell."""
    return argv if not IS_WINDOWS else argv


def null_device() -> str:
    """os.devnull (nul on Windows, /dev/null elsewhere)."""
    return os.devnull
