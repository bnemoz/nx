"""Minimal drop-in replacement for the bits of the `openai` SDK nx actually
uses: a chat-completions client (streaming + non-streaming), GET /models, and
two exception types. nx only ever talks to /chat/completions and /models on
an OpenAI-compatible endpoint (llama.cpp, vLLM, Z.ai, OpenAI itself, ...) —
the full SDK eagerly imports typed models for admin/batches/evals/graders/
responses/azure/bedrock that nx never touches, which was most of its ~27MB
import-time RSS cost for a handful of call sites. This module (plus `httpx`,
already a direct dependency) covers that handful directly.

Response objects are generic dict-attribute wrappers (`_Obj`) rather than
typed models — nx's own code (`_stream_parse.py`, `_metrics.py`) only ever
does `getattr`/`.get()`-style access on them, so exact typing isn't needed.

Transient-failure retry: before the first byte of a response, a 429/5xx or a
broken read is retried up to NX_HTTP_RETRIES times (default 2) with
exponential backoff (NX_HTTP_BACKOFF_BASE, default 0.5s → 0.5s, 1s, 2s, …)
or the server's Retry-After header when it sends one. Retries stop dead the
moment a stream is open — see _send's docstring.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal, overload

from ._env import _env_float, _env_int

# httpx is deliberately NOT imported at module top: it (plus httpcore, anyio,
# h11, certifi) costs ~45 ms of every `import nx` — including `nx --help`,
# `nx sessions`, and the whole test suite — and is only actually needed once a
# real client is constructed or an exception is classified. Everything below
# references `httpx.X` lazily via this module-global alias; _load_httpx() runs
# on the OpenAI() constructor path (the only path that reaches the network) so
# all later attribute lookups are plain dict hits on a fully-loaded module.
httpx: Any = None  # replaced by the real module on first _load_httpx()


def _load_httpx():
    """Import httpx on first use and bind it as this module's `httpx` global.
    Returns the module. Cheap idempotent no-op after the first call.

    Raises ImportError (not silently skipped) if httpx is missing — nx's
    pyproject lists it as a hard dependency, so this only fires in a broken
    install and should say so at the exact call site that needed it."""
    global httpx
    if httpx is None:
        try:
            import httpx as _httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required to talk to a model endpoint "
                "(pip install httpx)") from e
        httpx = _httpx
    return httpx


def __getattr__(name: str):
    # Legacy entry point: `from ._openai_client import APIError` used to mean
    # "import nx", which pulled httpx. Now these import nothing heavy.
    if name == "__path__":
        raise AttributeError(name)
    return globals()[name]


class APIError(Exception):
    """Raised for a non-2xx response. `.body` is the parsed JSON error body
    (or None), mirroring the attribute nx's context-window overrun probe
    (`_sources.py:_ctx_from_overrun_probe`) reads to extract the server's
    error message."""
    def __init__(
        self,
        message: str = "",
        *,
        body: Any = None,
        status_code: int | None = None,
        request: httpx.Request | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.body = body
        self.status_code = status_code
        self.request = request


class APIConnectionError(APIError):
    """Raised when the request never reached a server (DNS/refused/timeout),
    as opposed to APIError for a reachable server returning an error status.
    `.retryable` records whether `_send` judged the transport failure
    transient (a connection existed and broke/timed out — worth a caller-level
    retry) or not (the server refused/was unreachable — backoff can't help)."""
    def __init__(self, message: str = "connection error", *,
                 request: httpx.Request | None = None,
                 retryable: bool = True):
        super().__init__(message, request=request)
        self.retryable = retryable


# --- transient-error retry ---------------------------------------------------
# Default attempts *beyond the first* and the base of the exponential backoff
# (base * 2**attempt: 0.5s, 1s, 2s, …). Read per-request via the env (not
# cached at import) so tests can retune them without a reload.
_HTTP_RETRIES_DEFAULT = 2
_HTTP_BACKOFF_BASE_DEFAULT = 0.5
# A Retry-After (or runaway backoff) can never stall a turn longer than this.
_RETRY_DELAY_CAP = 30.0
# Statuses worth a blind retry: 429 (rate limit — usually clears in seconds)
# and 5xx (upstream/overloaded). Deliberately NOT 4xx generally: 400/401/403/
# 404 are deterministic, and retrying them just multiplies the latency before
# the same error (a 400 also feeds nx's context-window overrun probe, which
# must see the error immediately).
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _is_transient_net_exc(e: BaseException) -> bool:
    """True for transport failures where a connection existed and then broke or
    stalled (read/write/remote reset/timeout) — the busy-server shapes a
    backoff can plausibly outlast. A refused/never-established connection
    (ConnectError/ConnectTimeout) means the server is down, not busy: no
    retry, so a cold-start against a not-yet-up local server fails fast
    exactly as it always has."""
    return isinstance(e, (httpx.ReadError, httpx.WriteError,
                          httpx.ReadTimeout, httpx.WriteTimeout,
                          httpx.PoolTimeout, httpx.RemoteProtocolError))


def _retry_after_seconds(headers: Any) -> float | None:
    """Retry-After as seconds (delay or HTTP-date form). None when absent or
    unparseable, so the caller falls back to exponential backoff."""
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        import datetime as _dt
        from email.utils import parsedate_to_datetime
        delta = parsedate_to_datetime(raw) - _dt.datetime.now(_dt.timezone.utc)
        return max(0.0, delta.total_seconds())
    except Exception:
        return None


def _backoff_delay(attempt: int, retry_after: float | None) -> float:
    """Seconds to wait before retry `attempt + 1` (0-based). A usable
    Retry-After wins as-is; otherwise base * 2**attempt. Capped either way."""
    if retry_after is not None:
        return max(0.0, min(retry_after, _RETRY_DELAY_CAP))
    base = max(0.0, _env_float("NX_HTTP_BACKOFF_BASE", _HTTP_BACKOFF_BASE_DEFAULT))
    if base <= 0:
        return 0.0
    return min(base * (2 ** max(0, attempt)), _RETRY_DELAY_CAP)


def _notice(req: httpx.Request, attempt: int, retries: int, why: str,
            delay: float, retry_after: float | None, hint: str = "") -> None:
    """One-line stderr notice + traffic-log entry per retry. stderr, not
    stdout: stdout is strictly chat rendering (and --json harness output).
    Silent while a probe_scope() is active on this thread (see below) — but
    only the print: probe retries stay in the traffic log for debugging.
    `hint` is appended verbatim when a failure shape has an actionable tunable
    worth naming (a read timeout, most commonly)."""
    quiet = _probe_quiet()
    basis = "Retry-After" if retry_after is not None else "backoff"
    if not quiet:
        print(f"nx: {req.method} {req.url.path} failed ({why}); retrying "
              f"{attempt + 1}/{retries} in {delay:.1f}s ({basis}){hint}",
              file=sys.stderr)
    from ._logging import _log_event  # lazy: _logging builds on NX_HOME state
    _log_event("http_retry", {"method": req.method, "url": str(req.url),
                              "why": why, "attempt": attempt + 1,
                              "retries": retries, "delay": delay,
                              "retry_after": retry_after})


# --- background-probe mode ---------------------------------------------------
# The startup and /source probes (capability, thinking, context-window) send
# real chat-completions requests from daemon threads, each holding a captured
# Source and model id. Against a dead endpoint that was doubly visible
# (0.53.2): every backoff pass printed a retry notice from a background thread,
# landing over a chatbox frame whose rendering had already moved on and naming
# the model the probe captured rather than the active one; and the loop then
# slept out the whole curve while holding STARTUP_PROBE_LOCK, so the source the
# user switched TO stayed unprobed behind the one they switched away from.
# probe_scope() makes this thread's retry loop silent and abortable. The caller
# owns the staleness question -- only it knows what this probe was armed for.


class _ProbeAbort(Exception):
    """Raised inside a probe's request loop when its staleness predicate fires.
    Reaches the probe thread's own `except Exception`, which already treats a
    failed probe as "stay unprobed and let a later turn re-arm" -- so no probe
    needs to name it, and it can never surface in a user turn."""


# threading.local, not a contextvar: each probe runs in its own daemon thread
# and a bare Thread does not inherit a contextvar snapshot, so the flag cannot
# leak into the main thread's user turns.
_PROBE_MODE = threading.local()
# NX_PROBE_QUIET=0 restores probe retry notices for debugging. The abort
# behavior below is not gated by it -- that part is correctness, not noise.
_PROBE_QUIET_DEFAULT = 1


def _probe_active() -> bool:
    return getattr(_PROBE_MODE, "stale", None) is not None


def _probe_quiet() -> bool:
    return (_probe_active()
            and _env_int("NX_PROBE_QUIET", _PROBE_QUIET_DEFAULT) != 0)


def _probe_stale() -> None:
    """Abort this thread's probe retry loop if it has been superseded. Safe to
    call from the main thread's turns: `_probe_active()` is False there, so this
    is a no-op and a user turn can never be cancelled by it."""
    fn = getattr(_PROBE_MODE, "stale", None)
    if fn is not None and fn():
        raise _ProbeAbort()


@contextmanager
def probe_scope(stale: Callable[[], bool]):
    """Silence retry notices on this thread and abort backoff while `stale()`
    is true. Checked before each attempt and before each backoff sleep, so a
    superseded probe releases STARTUP_PROBE_LOCK promptly instead of sleeping
    out the rest of the curve against an endpoint the user abandoned.

    Checks land only between attempts, never during a send, so an in-flight
    request always completes and nothing is cancelled mid-stream."""
    _PROBE_MODE.stale = stale
    try:
        yield
    finally:
        _PROBE_MODE.stale = None


_MISSING = object()  # PF-22: sentinel for _Obj's attribute-wrapper cache
_DONE = object()   # _ChatStream._line(): the terminal [DONE] frame was seen


class _Obj:
    """Wraps a JSON dict so callers can use attribute access
    (`chunk.choices[0].delta.content`) the way they would against a real
    pydantic model. `.model_dump()` returns the raw dict; `.model_extra`
    returns the same raw dict (a harmless superset of "unknown" fields —
    every caller only ever does `.get(key)` on it, never iterates expecting
    exclusivity)."""
    __slots__ = ("_raw", "_cache")

    def __init__(self, raw: Any):
        object.__setattr__(self, "_raw", raw if isinstance(raw, dict) else {})
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str) -> Any:
        # PF-22: memoize the child wrapper. Callers read the same path several
        # times per token (`chunk.choices[0].delta`), and re-wrapping the dict
        # (plus every dict in a list) on each read was ~5-8 allocations/token.
        cache = self._cache
        v = cache.get(name, _MISSING)
        if v is not _MISSING:
            return v
        v = self._raw.get(name)
        if isinstance(v, dict):
            v = _Obj(v)
        elif isinstance(v, list):
            v = [_Obj(x) if isinstance(x, dict) else x for x in v]
        cache[name] = v
        return v

    def __repr__(self) -> str:
        return f"_Obj({self._raw!r})"

    def model_dump(self) -> dict[str, Any]:
        return self._raw

    @property
    def model_extra(self) -> dict[str, Any]:
        return self._raw


class _ChatStream:
    """Iterates an SSE `text/event-stream` response, yielding one `_Obj` per
    `data:` line (skipping the terminal `[DONE]`). `.close()` releases the
    underlying connection — mirrors the openai SDK stream object's contract
    that `_LoggingStream`/`open_stream()` callers rely on."""
    def __init__(self, resp: httpx.Response):
        self._resp = resp
        self.dropped_lines = 0

    def __iter__(self) -> Iterator[_Obj]:
        # NOT iter_lines(): httpx's LineDecoder splits with str.splitlines()
        # semantics, whose NEWLINE_CHARS are the nine characters
        # "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029". SSE's only terminators
        # are \n, \r\n and \r, and a JSON string may legally carry the other
        # seven as raw UTF-8 -- so one tool-args delta containing U+0085,
        # U+2028, U+2029 or a vertical tab (all plausible inside an edit
        # tool's new/diff value) splits ONE frame into two lines: the first
        # half fails json.loads and is counted as a dropped frame, the second
        # does not start with the data: prefix and is skipped outright. The
        # delta is gone, the args stay truncated, and the stream itself
        # finishes cleanly -- which is why it surfaced as a malformed tool
        # call with finish_reason=stop rather than as a transport error.
        # Splitting raw bytes on b"\n" (then trimming a trailing b"\r", which
        # covers \r\n and a lone \r) is both SSE-correct and immune: those
        # code points never appear as an ASCII newline byte.
        buf = b""
        for raw in self._resp.iter_bytes():
            if not raw:
                continue
            buf += raw
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line, buf = buf[:i], buf[i + 1:]
                if line.endswith(b"\r"):
                    line = line[:-1]
                obj = self._line(line)
                if obj is _DONE:
                    return
                if obj is not None:
                    yield obj
        obj = self._line(buf.rstrip(b"\r"))
        if obj is not None and obj is not _DONE:
            yield obj

    def _line(self, line: bytes) -> Any:
        """Decode one SSE line into a chunk, or None for a blank/non-data line
        and for a malformed frame (counted in `dropped_lines`). `_DONE` tells
        the iterator to stop."""
        if not line.startswith(b"data:"):
            return None
        data = line[len(b"data:"):].strip()
        if not data:
            return None
        if data == b"[DONE]":
            return _DONE
        try:
            return _Obj(json.loads(data))
        except json.JSONDecodeError:
            # A malformed SSE frame here can silently drop a tool-call
            # argument delta, later surfacing as a bogus "malformed tool
            # call" JSON error once the truncated buffer is parsed.
            self.dropped_lines += 1
            return None

    def close(self) -> None:
        self._resp.close()


class _Completions:
    def __init__(self, client: "OpenAI"):
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: Literal[False] = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream_options: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _Obj: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: Literal[True],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream_options: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _ChatStream: ...

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream_options: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _ChatStream | _Obj:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stream_options is not None:
            payload["stream_options"] = stream_options
        payload.update(kwargs)  # max_tokens, temperature, sampler overrides, ...
        if extra_body:
            payload.update(extra_body)
        # httpx semantics gotcha (found by the eval corpus, P3-14): an explicit
        # timeout=None passed to build_request DISABLES the timeout entirely —
        # the client's configured Timeout only applies when the per-request
        # kwarg is ABSENT. Passing None unconditionally turned every request
        # (models.list's 3s/10s caps included) into a no-timeout request: the
        # connect-fail-fast and NX_HTTP_TIMEOUT_READ deadlocks never fired and
        # the mid-stream timeout recovery ladder (P3-2) was unreachable. Only
        # attach the per-request timeout when the caller actually set one.
        if timeout is not None:
            req = self._client._client.build_request(
                "POST", "/chat/completions", json=payload, timeout=timeout)
        else:
            req = self._client._client.build_request(
                "POST", "/chat/completions", json=payload)
        resp = self._client._send(req, stream=stream)
        if stream:
            return _ChatStream(resp)
        return _Obj(resp.json())


class _Chat:
    def __init__(self, client: "OpenAI"):
        self.completions = _Completions(client)


class _Models:
    def __init__(self, client: "OpenAI"):
        self._client = client

    def list(self, timeout: float | httpx.Timeout | None = None) -> _Obj:
        # Explicit branch instead of `**({"timeout": ...})` unpack: httpx's
        # overloads can't type a conditional-dict `**kwargs`, but the named
        # kwarg types fine (behavior identical either way).
        if timeout is None:
            req = self._client._client.build_request("GET", "/models")
        else:
            req = self._client._client.build_request("GET", "/models",
                                                     timeout=timeout)
        resp = self._client._send(req, stream=False)
        return _Obj(resp.json())


class OpenAI:
    """Just enough of `openai.OpenAI` for nx: base_url + api_key + optional
    default_headers, `.chat.completions.create()`, `.models.list()`.

    `transport` (optional) replaces httpx's default connection pooling —
    tests pass an httpx.MockTransport to exercise _send's retry loop without
    a live server."""
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        _load_httpx()
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if default_headers:
            headers.update(default_headers)
        # Per-request timeouts (NX_HTTP_TIMEOUT_CONNECT / NX_HTTP_TIMEOUT_READ,
        # read here so tests can retune them): short connect so an unreachable
        # server fails fast (matches the openai SDK's practical behavior);
        # generous read since local models can take minutes before a
        # non-streaming answer's first byte. 0 (or negative) means no cap —
        # httpx reads a literal 0.0 as a *zero-second* timeout, which would
        # kill every request instantly.
        connect = _env_float("NX_HTTP_TIMEOUT_CONNECT", 10.0)
        read = _env_float("NX_HTTP_TIMEOUT_READ", 900.0)
        timeout = httpx.Timeout(
            connect=connect if connect > 0 else None,
            read=read if read > 0 else None,
            write=30.0, pool=600.0)
        self._client = httpx.Client(base_url=self.base_url, headers=headers,
                                     timeout=timeout, transport=transport)
        self.chat = _Chat(self)
        self.models = _Models(self)

    def _send(self, req: httpx.Request, stream: bool) -> httpx.Response:
        """Send `req`, retrying transient failures before the first byte.

        Retried (NX_HTTP_RETRIES times beyond the first attempt, exponential
        NX_HTTP_BACKOFF_BASE backoff, or the server's Retry-After when sent):

          • 429 / 502 / 503 / 504 / 500 responses — rate limiting and
            upstream overload, both usually gone in seconds
          • a connection that existed and then broke or timed out
            (read/write/reset — the busy-server shapes)

        Never retried:

          • other 4xx — deterministic (400/401/403/404); a 400 also feeds the
            context-window overrun probe, which needs the error immediately
          • ConnectError/ConnectTimeout — the server is down, not busy; a
            cold-start probe must keep failing fast
          • anything after the first streamed byte. `send(stream=True)`
            returns as soon as response *headers* arrive, before any SSE
            payload, so this loop is structurally incapable of retrying
            mid-stream — and `_ChatStream`'s read errors stay un-retried on
            purpose: a partially-consumed stream can't be replayed safely
            (nx's turn loop owns recovery for those, with correct bookkeeping
            around the tokens already rendered).
        """
        retries = max(0, _env_int("NX_HTTP_RETRIES", _HTTP_RETRIES_DEFAULT))
        attempt = 0
        while True:
            _probe_stale()
            try:
                resp = self._client.send(req, stream=stream)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt < retries and _is_transient_net_exc(e):
                    retry_after = None
                    delay = _backoff_delay(attempt, retry_after)
                    hint = ""
                    if isinstance(e, httpx.ReadTimeout):
                        read = self._client.timeout.read
                        read_s = "no limit" if read is None else f"{read:g}s"
                        hint = (f" — the server sent no response bytes within "
                                f"the {read_s} read timeout; a slow server needs "
                                "NX_HTTP_TIMEOUT_READ raised")
                    _notice(req, attempt, retries, f"connection error: {e}", delay,
                            retry_after, hint)
                    _probe_stale()
                    attempt += 1
                    time.sleep(delay)
                    continue
                raise APIConnectionError(str(e), request=req,
                                         retryable=_is_transient_net_exc(e)) from e
            if resp.status_code < 400:
                return resp
            resp.read()
            body = None
            try:
                body = resp.json()
            except Exception:
                pass
            msg = None
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    msg = err.get("message")
                elif isinstance(err, str):
                    msg = err
            if not msg:
                msg = resp.text or f"HTTP {resp.status_code}"
            if attempt < retries and resp.status_code in _RETRY_STATUSES:
                retry_after = _retry_after_seconds(resp.headers)
                delay = _backoff_delay(attempt, retry_after)
                _notice(req, attempt, retries,
                        f"HTTP {resp.status_code} — {msg[:160]}", delay, retry_after)
                resp.close()
                _probe_stale()
                attempt += 1
                time.sleep(delay)
                continue
            resp.close()
            raise APIError(f"Error code: {resp.status_code} - {json.dumps(body) if body else msg}",
                           body=body, status_code=resp.status_code, request=req)
