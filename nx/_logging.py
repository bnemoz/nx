"""Append-only JSONL record of every byte we ship to / receive from the
server. Lives next to this package so it's easy to find. `_LoggingStream`
(see _turn.py) tees every single streamed SSE chunk through here, which on a
fast local server can be hundreds of times a second — a blocking disk write
(or the flush a line-buffered file does on every "\\n") on that path adds
real latency to token rendering. `_AsyncLogWriter` hands the line to a
background thread instead, so the caller's `.write()` is just a
`queue.put()` and never waits on disk I/O. Rotates once the file passes
LOG_MAX_BYTES so it stays bounded without needing to be rotated by hand.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from typing import Any

from . import _state as st
from ._env import _env_int

# Under NX_HOME (same convention as sessions), not the installed package dir —
# a system-wide/read-only install must not crash on import, and the log holds
# full unredacted request/response bodies that shouldn't be world-readable.
LOG_PATH = os.path.join(st.SESSION_HOME, "logs", "llamacpp.log")
LOG_MAX_BYTES = _env_int("NX_LOG_MAX_BYTES", 10000000)


class _AsyncLogWriter:
    """Drop-in replacement for a file object's `.write(line)` that hands the
    line to a dedicated background thread instead of writing inline. Used
    as a drop-in for `_llog`, which both `_log_event` and `_LoggingStream`
    call `.write()` on."""

    def __init__(self, path: str, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="nx-log-writer")
        self._thread.start()

    def write(self, line: str) -> None:
        self._queue.put(line)

    def flush(self) -> None:
        """Block until every currently queued line has been written. Called
        at interpreter exit so a normal shutdown never drops trailing log
        lines that were still in flight on the writer thread."""
        self._queue.join()

    def _open(self):
        """Lazy, fail-soft: a read-only/system-wide install must not crash
        the writer thread (or, transitively, `import nx`) just because the
        log directory can't be created."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True, mode=0o700)
            fd = os.open(self._path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            return os.fdopen(fd, "a", buffering=1)
        except OSError:
            return None

    def _run(self) -> None:
        f = self._open()
        try:
            while True:
                line = self._queue.get()
                try:
                    if f is None:
                        continue
                    if self._max_bytes > 0 and f.tell() >= self._max_bytes:
                        f.close()
                        try:
                            os.replace(self._path, self._path + ".1")
                        except OSError:
                            pass
                        f = self._open()
                    f.write(line)
                except OSError:
                    pass
                finally:
                    self._queue.task_done()
        finally:
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass


_llog = _AsyncLogWriter(LOG_PATH, LOG_MAX_BYTES)
atexit.register(_llog.flush)


def _log_event(direction: str, payload: Any) -> None:
    """direction: 'req' (outgoing) or 'resp' (incoming SSE chunk). Enqueues
    onto the background writer thread (see _AsyncLogWriter) rather than
    writing inline, so a slow disk never adds latency to the caller."""
    _llog.write(json.dumps({"ts": time.time(), "dir": direction, "data": payload}) + "\n")
