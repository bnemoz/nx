"""Cross-platform terminal primitives: raw-mode entry, timed readiness wait,
single-byte read. Centralizes the POSIX/Windows split that used to be
scattered as bare `termios`/`select`/`os.read` calls across `_chatbox.py`,
`_approval.py`, and `_interrupt.py`. Call sites speak only this module's
three primitives (`raw_mode`, `wait_ready`, `read_byte`) and carry no
platform branches themselves.

POSIX path is the original termios + select + os.read logic, unchanged in
behavior. Windows path targets the real interactive console (the only
Windows terminal `nx` supports for its editor / Esc-watcher / approval
prompt): `SetConsoleMode` enables `ENABLE_VIRTUAL_TERMINAL_INPUT` so
special keys arrive as the same CSI sequences the editor already parses,
and clears `ENABLE_LINE_INPUT`/`ENABLE_ECHO_INPUT` (the ICANON|ECHO
analog). Readiness uses `WaitForSingleObject` on the console input handle
(`select.select` cannot poll console-input handles on Windows). Non-tty
fds (pipes, the path tests exercise) fall through to a `PeekNamedPipe`
poll so `_drain_redirect_text` keeps working over `os.pipe()`.
"""
from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _STD_INPUT_HANDLE = wintypes.DWORD(-10)
    _ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
    _ENABLE_LINE_INPUT = 0x0002
    _ENABLE_ECHO_INPUT = 0x0004
    _ENABLE_PROCESSED_INPUT = 0x0001
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 0x102

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GetStdHandle = _kernel32.GetStdHandle
    _GetStdHandle.argtypes = [wintypes.DWORD]
    _GetStdHandle.restype = wintypes.HANDLE
    _GetConsoleMode = _kernel32.GetConsoleMode
    _GetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
    _GetConsoleMode.restype = wintypes.BOOL
    _SetConsoleMode = _kernel32.SetConsoleMode
    _SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _SetConsoleMode.restype = wintypes.BOOL
    _WaitForSingleObject = _kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _WaitForSingleObject.restype = wintypes.DWORD
    _PeekNamedPipe = _kernel32.PeekNamedPipe
    _PeekNamedPipe.argtypes = [
        wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
        wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD,
    ]
    _PeekNamedPipe.restype = wintypes.BOOL
else:
    import select
    import termios


# ---------------------------------------------------------------- raw_mode ---

@contextlib.contextmanager
def raw_mode(fd: int, *, icrnl: bool = False) -> Iterator[list | None]:
    """Put `fd` into raw (non-canonical, no-echo) mode for the duration of the
    block, restoring the original attributes on exit. Centralizes the
    termios save/restore/flag-twiddle shared by `_chatbox.py`'s editor and
    `_approval.py`'s Y/n/esc prompt.

    `icrnl` disables CR->NL mapping (POSIX-only; Windows has no equivalent
    and Enter arrives as `\\r`, which the chatbox already treats as submit).

    On Windows the console is put into virtual-terminal-input mode with
    line-input and echo cleared; `ENABLE_PROCESSED_INPUT` stays on so Ctrl+C
    still generates SIGINT. The SIGTERM-restore handler is a POSIX-only
    concern (Windows never delivers SIGTERM the tty way) and is skipped.
    """
    if IS_WINDOWS:
        if not os.isatty(fd):
            yield None
            return
        handle = _GetStdHandle(_STD_INPUT_HANDLE)
        old = wintypes.DWORD(0)
        if not _GetConsoleMode(handle, ctypes.byref(old)):
            yield None
            return
        new = (old.value | _ENABLE_VIRTUAL_TERMINAL_INPUT)
        new &= ~(_ENABLE_LINE_INPUT | _ENABLE_ECHO_INPUT)
        new |= _ENABLE_PROCESSED_INPUT  # keep Ctrl+C -> SIGINT
        _SetConsoleMode(handle, wintypes.DWORD(new))
        try:
            yield [old.value]
        finally:
            try:
                _SetConsoleMode(handle, old)
            except Exception:
                pass
        return

    # --- POSIX ---
    old = termios.tcgetattr(fd)
    new = old[:]
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[0] &= ~termios.IXON
    if icrnl:
        new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0

    def _on_sigterm(signum, frame):
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    prev_handler = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _on_sigterm)
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        yield old
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, prev_handler)


# --------------------------------------------------------------- wait_ready ---

def wait_ready(fd: int, timeout: float) -> bool:
    """Return True if a byte is available to read from `fd` within `timeout`
    seconds, False on timeout. Used by the chatbox escape-sequence drain,
    the Esc-watcher's bare-Esc disambiguation, and `_drain_redirect_text`.

    POSIX: `select.select([fd], [], [], timeout)`.
    Windows console: `WaitForSingleObject` on the console input handle.
    Windows pipe (tests): `PeekNamedPipe` poll loop (select can't poll
    console handles or pipes on Windows).
    """
    if IS_WINDOWS:
        # Console: WaitForSingleObject on the stdin handle.
        try:
            handle = _GetStdHandle(_STD_INPUT_HANDLE)
        except Exception:
            handle = None
        if handle:
            ms = max(0, int(timeout * 1000))
            r = _WaitForSingleObject(handle, wintypes.DWORD(ms))
            return r == _WAIT_OBJECT_0
        # Non-console fd (pipe/file): PeekNamedPipe poll.
        try:
            wh = msvcrt.get_osfhandle(fd)
        except OSError:
            return False
        avail = wintypes.DWORD(0)
        end = time.time() + timeout
        while time.time() <= end:
            if _PeekNamedPipe(wh, None, 0, None, ctypes.byref(avail), None):
                if avail.value > 0:
                    return True
            time.sleep(0.01)
        return False

    # POSIX
    ready, _, _ = select.select([fd], [], [], timeout)
    return bool(ready)


def wait_ready_any(fds: list[int], timeout: float) -> int | None:
    """Wait until *any* fd in `fds` is readable, up to `timeout` seconds.
    Returns the first readable fd, or None on timeout.

    POSIX-only (PF-06): the interrupt watcher uses this to block on stdin *and*
    a wake pipe at once, so a disarm can end its wait immediately instead of
    after the next poll interval. Non-POSIX callers fall back to polling the
    first fd, which is the pre-existing behaviour on those platforms.
    """
    if IS_WINDOWS or not fds:
        # Windows console handles can't be select()ed; fall back to the single
        # -fd poll the watcher used before (the wake pipe is POSIX-only).
        return fds[0] if fds and wait_ready(fds[0], timeout) else None
    ready, _, _ = select.select(list(fds), [], [], timeout)
    return ready[0] if ready else None


# ------------------------------------------- interrupt-watcher input mode ---
# The Esc watcher (_interrupt.py) can't use the `raw_mode` context manager:
# it sets raw input mode when armed and restores cooked mode when disarmed,
# across many cycles inside one persistent thread. These primitives give it
# the same POSIX termios save/restore (raw, no-echo, ISIG ON so Ctrl+C still
# generates SIGINT, VMIN=0/VTIME=0 for non-blocking reads) and the Windows
# console equivalent (VT input on, line/echo off, processed on) without it
# having to know which platform it's on.

def capture_input_mode(fd: int) -> Any:
    """Snapshot the current input mode of `fd`. Returns an opaque value to
    pass to `apply_interrupt_mode`/`restore_input_mode`, or None if `fd` is
    not a controlling terminal we can manipulate (non-tty, no fileno, etc.)
    — callers treat None as "watcher can't run" and park forever."""
    if not os.isatty(fd):
        return None
    try:
        if IS_WINDOWS:
            handle = _GetStdHandle(_STD_INPUT_HANDLE)
            old = wintypes.DWORD(0)
            if not _GetConsoleMode(handle, ctypes.byref(old)):
                return None
            return ("win", handle, old.value)
        return ("posix", termios.tcgetattr(fd))
    except Exception:
        return None


def apply_interrupt_mode(fd: int, old: Any) -> None:
    """Switch `fd` into the watcher's raw input mode (ISIG/processed kept ON
    so Ctrl+C still interrupts). Raises on failure; caller parks/exits."""
    if old is None:
        return
    if IS_WINDOWS:
        _kind, handle, old_mode = old
        new = (old_mode | _ENABLE_VIRTUAL_TERMINAL_INPUT)
        new &= ~(_ENABLE_LINE_INPUT | _ENABLE_ECHO_INPUT)
        new |= _ENABLE_PROCESSED_INPUT
        _SetConsoleMode(handle, wintypes.DWORD(new))
        return
    _kind, attrs = old
    new = attrs[:]
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 0
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSADRAIN, new)


def restore_input_mode(fd: int, old: Any) -> None:
    """Restore a mode captured by `capture_input_mode`. No-op if old is None."""
    if old is None:
        return
    try:
        if IS_WINDOWS:
            _kind, handle, old_mode = old
            _SetConsoleMode(handle, wintypes.DWORD(old_mode))
            return
        _kind, attrs = old
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except Exception:
        pass


# --------------------------------------------------------------- read_byte ---

def read_byte(fd: int) -> bytes:
    """Read exactly one byte from `fd`, blocking until it arrives. Returns
    b"" on a closed pipe (the EOF signal the chatbox turns into EOFError).
    On a Windows console under VT-input mode, `os.read` returns one byte of
    the UTF-8/CSI sequence the console synthesized from the physical key."""
    return os.read(fd, 1)


# Windows-only convenience for callers (the interrupt watcher) that want
# to poll stdin without a timeout-driven loop differently; kept here so the
# Windows-only import surface is in one module.
if IS_WINDOWS:
    def kbhit() -> bool:
        """True if a key is available on the Windows console right now
        (non-blocking). Returns False for non-console stdin."""
        return bool(msvcrt.kbhit())
else:
    def kbhit() -> bool:
        # POSIX non-blocking readiness on stdin's fd.
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            return False
        return wait_ready(fd, 0.0)