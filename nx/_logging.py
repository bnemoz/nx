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
import atexit
import json
import os
import queue
import threading
import time

from ._env import _env_int

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "llamacpp.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
LOG_MAX_BYTES = _env_int("NX_LOG_MAX_BYTES", 10000000)


class _AsyncLogWriter:
    """Drop-in replacement for a file object's `.write(line)` that hands the
    line to a dedicated background thread instead of writing inline. Used
    as a drop-in for `_llog`, which both `_log_event` and `_LoggingStream`
    call `.write()` on."""

    def __init__(self, path, max_bytes):
        self._path = path
        self._max_bytes = max_bytes
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="nx-log-writer")
        self._thread.start()

    def write(self, line):
        self._queue.put(line)

    def flush(self):
        """Block until every currently queued line has been written. Called
        at interpreter exit so a normal shutdown never drops trailing log
        lines that were still in flight on the writer thread."""
        self._queue.join()

    def _run(self):
        f = open(self._path, "a", buffering=1)
        try:
            while True:
                line = self._queue.get()
                try:
                    if self._max_bytes > 0 and f.tell() >= self._max_bytes:
                        f.close()
                        try:
                            os.replace(self._path, self._path + ".1")
                        except OSError:
                            pass
                        f = open(self._path, "a", buffering=1)
                    f.write(line)
                except OSError:
                    pass
                finally:
                    self._queue.task_done()
        finally:
            try:
                f.close()
            except OSError:
                pass


_llog = _AsyncLogWriter(LOG_PATH, LOG_MAX_BYTES)
atexit.register(_llog.flush)


def _log_event(direction, payload):
    """direction: 'req' (outgoing) or 'resp' (incoming SSE chunk). Enqueues
    onto the background writer thread (see _AsyncLogWriter) rather than
    writing inline, so a slow disk never adds latency to the caller."""
    _llog.write(json.dumps({"ts": time.time(), "dir": direction, "data": payload}) + "\n")
