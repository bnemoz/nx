#!/usr/bin/env python3
"""Regression coverage for resize repaint anchoring."""
import contextlib
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-chatbox-resize-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _chatbox  # noqa: E402


class _Output:
    def __init__(self):
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        pass


class _Input:
    def __init__(self, fd: int):
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


def test_resize_repaints_from_the_erased_frame_top(monkeypatch):
    """A resize must not insert a line between erase and replacement paint."""
    read_fd, write_fd = os.pipe()
    output = _Output()
    # First paint, idle resize check, resize repaint, stale-frame erase.
    widths = iter((120, 80, 80, 80))

    monkeypatch.setattr(_chatbox.sys, "stdin", _Input(read_fd))
    monkeypatch.setattr(_chatbox.sys, "stdout", output)
    monkeypatch.setattr(_chatbox, "raw_mode", lambda *_a, **_kw: contextlib.nullcontext())
    monkeypatch.setattr(_chatbox.shutil, "get_terminal_size",
                        lambda _fallback: os.terminal_size((next(widths), 24)))
    ready = iter((False, True))
    monkeypatch.setattr(_chatbox, "wait_ready", lambda *_a: next(ready))
    monkeypatch.setattr(_chatbox, "_raw_read_key", lambda _fd: "\x11")
    try:
        assert _chatbox._chatbox_raw() == "/exit"
    finally:
        os.close(read_fd)
        os.close(write_fd)

    erase_at = next(i for i, text in enumerate(output.writes) if "\x1b[J" in text)
    replacement_at = next(
        i for i in range(erase_at + 1, len(output.writes))
        if output.writes[i].startswith("\r\x1b[2K"))
    replacement = output.writes[replacement_at]
    assert "\n" not in output.writes[erase_at + 1:replacement_at]
    assert replacement.startswith("\r\x1b[2K"), replacement.encode()
