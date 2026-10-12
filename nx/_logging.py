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

# v1 usage metrics: rich per-tool-call telemetry (harness-vs-model failure
# diagnosis) on its OWN channel so old req/resp/repair records are never
# confused with these. Separate stream (usage.jsonl, same async writer class
# and rotation policy) rather than a dir tag on llamacpp.log: the request log
# is a firehose and these events are meant to be grep-able on their own.
# NX_USAGE_LOG=0 disables every emitter at the flag check below.
USAGE_PATH = os.path.join(st.SESSION_HOME, "logs", "usage.jsonl")
USAGE_LOG = _env_int("NX_USAGE_LOG", 1) != 0

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
        # unbounded RAM backlog. Bounded twice: 1024 lines AND 64 MB of queued
        # bytes, drop-oldest on either — a 100K-token prompt serializes to
        # ~400 KB per req line, so a line-only cap could still park ~1 GB in
        # a swarm whose agents outpace the single writer thread.
        self._queue: queue.Queue[str] = queue.Queue(maxsize=1024)
        self._max_queued_bytes = 64 * 1024 * 1024
        self._queued_bytes = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="nx-log-writer")
        self._thread.start()

    def write(self, line: str) -> None:
        must_drop = self._queue.full() or (
            self._queued_bytes + len(line)) > self._max_queued_bytes
        if must_drop:
            # Drop-oldest until the new line fits under both caps: the
            # freshest diagnostics matter, and the agents never block.
            while self._queue.qsize() > 1 or (
                    self._queued_bytes + len(line)) > self._max_queued_bytes:
                try:
                    dropped = self._queue.get_nowait()
                    self._queued_bytes -= len(dropped)
                    self._queue.task_done()
                except queue.Empty:
                    break
        try:
            self._queue.put_nowait(line)
            self._queued_bytes += len(line)
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
                    if f is not None:  # fail-soft: a failed reopen drops the line
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

_ulog = _AsyncLogWriter(USAGE_PATH, LOG_MAX_BYTES)
atexit.register(_ulog.flush)


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
            # keep byte-sizes so the stub is still useful for debugging.
            # 0.108.6: lists/dicts are sized structurally, NOT via str(v) —
            # that repr'ed the whole transcript + tool schemas on every
            # model call even with NX_LOG_REQUESTS=0 (transient, but a
            # full-prompt copy on the hot path).
            stub["_logged_bytes"] = {
                k: (len(v) if isinstance(v, (list, dict, str)) else -1)
                for k, v in payload.items() if isinstance(v, (list, dict))
            }
        payload = {"_stub": True, **stub}
    _llog.write(json.dumps({"ts": time.time(), "dir": direction, "data": payload}) + "\n")


# --- v1 usage-metrics emitters (see the module docstring of nx/_usage_ctx) ----

def _log_usage(event: str, payload: Any) -> None:
    """One usage event onto the usage.jsonl stream: same async writer
    pattern as _log_event, distinct file + `event` key instead of `dir`, so
    old channel records and usage records never mix. Stamps `ts` here.
    Caller-checked USAGE_LOG flag re-checked defensively; never raises."""
    if not USAGE_LOG:
        return
    try:
        _ulog.write(json.dumps({"ts": time.time(), "dir": "usage",
                                "event": event, "data": payload}) + "\n")
    except Exception:  # noqa: BLE001 — telemetry must never break a call
        pass


def _agent_tag() -> str:
    """'main', or 'subagent:<tier>' when a sub-agent context owns this
    thread (the same TLS context run_tool() resolves its dispatch table
    through)."""
    try:
        from ._subagent_tls import _subagent_ctx
        ctx = _subagent_ctx()
        if ctx is not None:
            return "subagent:" + (getattr(ctx, "tier", None) or "read")
    except Exception:  # noqa: BLE001 — telemetry must never break a call
        pass
    return "main"


def _event_model() -> str | None:
    """The model to attribute this thread's usage events to: a sub-agent's
    ROUTED model (its model_target — the pool round-robin entry) when one is
    resolved, else the parent's active model. The per-model comparison is the
    point of the stream, so an agent running on qwen must not be billed to
    the glm session that spawned it."""
    try:
        from ._subagent_tls import _subagent_ctx
        ctx = _subagent_ctx()
        if ctx is not None:
            mt = getattr(ctx, "model_target", None)
            if mt and mt[1]:
                return str(mt[1])
    except Exception:  # noqa: BLE001 — telemetry must never break a call
        pass
    return st.MODEL


def _usage_emit(event: str, fields: dict, tail: bool = False) -> None:
    """One usage event: common fields (model/agent/session) + `fields`,
    onto the usage stream. `tail=True` attaches CTX_TAIL (see _usage_ctx).
    Cheap no-op when NX_USAGE_LOG=0. Never raises."""
    if not USAGE_LOG:
        return
    try:
        payload = dict(fields)
        payload.setdefault("model", _event_model())
        payload.setdefault("agent", _agent_tag())
        sid = st.SESSION_ID
        if sid:
            payload.setdefault("session", sid)
        if tail:
            from ._usage_ctx import tail as _tail
            payload["ctx_tail"] = _tail()
        _log_usage(event, payload)
    except Exception:  # noqa: BLE001 — telemetry must never break a call
        pass


def usage_tool_call(name: str, args: dict, result, duration_ms: float,
                    outcome: str, error_kind: str, cache_hit=None) -> None:
    """One `tool_call` event per run_tool() dispatch (successes included).
    `result` is the final string-ish result (content-parts already reduced to
    their one-line preview)."""
    if not USAGE_LOG:
        return
    try:
        args_raw = json.dumps(args, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — unserializable args: repr is enough
        args_raw = str(args)
    result_str = result if isinstance(result, str) else str(result)
    fields = {
        "tool": name,
        "duration_ms": round(duration_ms, 3),
        "outcome": outcome,               # ok | error | denied
        "args_bytes": len(args_raw),
        "args_raw": args_raw[:8192],
        "result_bytes": len(result_str),
        "result_head": result_str[:500],
        "error_kind": error_kind,         # exc class name | prefix | "ok"
        "subagent": _agent_tag() != "main",
    }
    if cache_hit is not None:
        fields["cache_hit"] = bool(cache_hit)
    _usage_emit("tool_call", fields, tail=outcome != "ok")


def usage_gate(name: str, reason: str) -> None:
    """A withheld-tool refusal: the model asked for `tool` this turn even
    though its schema was never offered (missing-tool signal)."""
    _usage_emit("gate", {"tool": name, "reason": reason}, tail=True)


def usage_breaker(name: str, trip_count: int) -> None:
    """The malformed-args breaker just OPENED for `tool` after
    `trip_count` consecutive malformed-args failures."""
    _usage_emit("breaker", {"tool": name, "trip_count": trip_count}, tail=True)


def usage_repair(kind: str, what: str, tool: str = "") -> None:
    """One applied/rejected args repair (see `_argsalvage._repair_log`).
    Mirrors the llamacpp.log `repair` record onto the usage stream so the
    friction counters (self-retro) see repairs per tool alongside breaker/
    gate/retry without parsing the request-log firehose. `tool` when the
    repair names one; `what` truncated since the CTX_TAIL carries the rest."""
    fields = {"kind": kind, "what": (what or "")[:512]}
    if tool:
        fields["tool"] = tool
    _usage_emit("repair", fields, tail=True)


def usage_retry(attempt: int, error_class: str) -> None:
    """One step-retry-ladder attempt for a failed turn (per ladder step,
    never per chunk). `error_class` is the turn-level stream error class."""
    _usage_emit("retry", {"attempt": attempt, "error_class": error_class},
                tail=False)


# --- v2 usage metrics (model_call / agent lifecycle / turn numbers) ---------

# The 1-based step number of the turn loop that is currently running on THIS
# thread (set by _run_model_turn_loop_inner each iteration; never set on a
# thread outside a turn loop). Thread-local, like the metrics/quiet slots: a
# --parallel batch worker's give_up must report ITS step, not a sibling's.
_turn_no = threading.local()


def set_turn_no(n) -> None:
    """Record this thread's current turn-loop step (1-based); None clears it
    (a thread outside any turn loop reports no turn number at all)."""
    try:
        if n is None:
            _turn_no.__dict__.pop("n", None)
        else:
            _turn_no.n = n
    except Exception:  # noqa: BLE001 — telemetry must never break a loop
        pass


def usage_model_call(duration_ms: float, finish_reasons, tokens,
                     streamed_chars: int = 0, ttft_ms: float | None = None,
                     prompt_tokens: int | None = None) -> None:
    """One `model_call` event per settled stream (the same common path
    _record_usage rides, so error early-returns are included).
    `tokens` is _normalize_usage()'s dict (input/output/cache_read/reasoning/
    estimated) or None when the server reported nothing; `ttft_ms` is the
    client-measured time to first token (parser.t_first)."""
    if not USAGE_LOG:
        return
    fields = {
        "duration_ms": round(duration_ms, 3),
        "finish_reasons": [str(fr) for fr in (finish_reasons or [])],
        "streamed_chars": int(streamed_chars),
        "subagent": _agent_tag() != "main",
    }
    n = getattr(_turn_no, "n", None)
    if n is not None:
        fields["turn"] = n
    if ttft_ms is not None:
        fields["ttft_ms"] = round(ttft_ms, 3)
    if tokens:
        fields["tokens"] = tokens
    if prompt_tokens is not None:
        fields["prompt_tokens"] = prompt_tokens
    _usage_emit("model_call", fields)


def usage_give_up(fields: dict) -> None:
    """A turn that surrendered (collapse / reasoning-only stall / malformed
    stream). Mirrors the llamacpp.log `give_up` record onto the usage stream
    and adds the turn number (v1 gap), so give-ups are countable per model
    alongside the tool_call/model_call events of the same session."""
    payload = dict(fields)
    n = getattr(_turn_no, "n", None)
    if n is not None:
        payload.setdefault("turn", n)
    _usage_emit("give_up", payload, tail=True)


def usage_agent(event: str, fields: dict) -> None:
    """Sub-agent lifecycle: `agent_spawn` (tier, resolved model, async,
    max_turns) when a sub-loop starts and `agent_done` (status, turns,
    elapsed, exit_state) when it finalizes. The pair makes swarm behavior —
    fan-out width, per-tier completion rates, agent duration by model —
    queryable without touching transcripts."""
    _usage_emit("agent_" + event, fields)
