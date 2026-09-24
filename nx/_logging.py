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
from ._private import private_dir

# Under NX_HOME (same convention as sessions), not the installed package dir —
# a system-wide/read-only install must not crash on import, and the log holds
# full unredacted request/response bodies that shouldn't be world-readable.
LOG_PATH = os.path.join(st.SESSION_HOME, "logs", "llamacpp.log")
LOG_MAX_BYTES = _env_int("NX_LOG_MAX_BYTES", 10000000)

# Swarm P4: the request log json.dumps's the ENTIRE prompt (and httpx
# serializes it again for the wire) on every model call — pure harness CPU on
# the agent's thread, and with N agents the shared queue is an unbounded RAM
# backlog. NX_LOG_REQUESTS=0 (also settable per run with NX_LOG_REQUESTS=0 in
# the environment) drops req/resp bodies to a stub line: streams still log
# chunk-free metadata elsewhere, and the request is still fully recoverable
# from the session transcript. Default ON (1) preserves the old behavior.
LOG_REQUESTS = _env_int("NX_LOG_REQUESTS", 1)


class _AsyncLogWriter:
    """Drop-in replacement for a file object's `.write(line)` that hands the
    line to a dedicated background thread instead of writing inline. Used
    as a drop-in for `_llog`, which both `_log_event` and `_LoggingStream`
    call `.write()` on."""

    def __init__(self, path: str, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        # Bounded (swarm P4): the writer is one line-buffered daemon thread;
        # if it falls behind N streaming agents, an unbounded queue is an
        # unbounded RAM backlog. drop-oldest past 4096 lines (~tens of MB of
        # worst-case lines): diagnostics degrade, the agents never block.
        self._queue: queue.Queue[str] = queue.Queue(maxsize=4096)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="nx-log-writer")
        self._thread.start()

    def write(self, line: str) -> None:
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            # Drop-oldest, keep the new line: the freshest diagnostics matter.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                pass
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                pass

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
            private_dir(os.path.dirname(self._path))
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
    writing inline, so a slow disk never adds latency to the caller.

    NX_LOG_REQUESTS=0 (swarm P4) logs a stub for req/resp bodies instead of
    the full payload: the per-request json.dumps of a 100K-token prompt is
    ~400KB of CPU per model call per agent, and the transcript already holds
    everything the log would."""
    if not LOG_REQUESTS and direction in ("req", "resp"):
        stub = {k: v for k, v in payload.items()
                if not isinstance(v, (list, dict))} if isinstance(payload, dict) else {}
        if isinstance(payload, dict):
            # keep byte-sizes so the stub is still useful for debugging
            stub["_logged_bytes"] = {k: len(str(v)) for k, v in payload.items()
                                     if isinstance(v, (list, dict))}
        payload = {"_stub": True, **stub}
    _llog.write(json.dumps({"ts": time.time(), "dir": direction, "data": payload}) + "\n")
