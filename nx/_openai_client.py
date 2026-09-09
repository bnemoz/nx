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
import time
from typing import Any, Iterator, Literal, overload

import httpx

from ._env import _env_float, _env_int


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
    `hint` is appended verbatim when a failure shape has an actionable
    tunable worth naming (a read timeout, most commonly)."""
    basis = "Retry-After" if retry_after is not None else "backoff"
    print(f"nx: {req.method} {req.url.path} failed ({why}); retrying "
          f"{attempt + 1}/{retries} in {delay:.1f}s ({basis}){hint}", file=sys.stderr)
    from ._logging import _log_event  # lazy: _logging builds on NX_HOME state
    _log_event("http_retry", {"method": req.method, "url": str(req.url),
                              "why": why, "attempt": attempt + 1,
                              "retries": retries, "delay": delay,
                              "retry_after": retry_after})


class _Obj:
    """Wraps a JSON dict so callers can use attribute access
    (`chunk.choices[0].delta.content`) the way they would against a real
    pydantic model. `.model_dump()` returns the raw dict; `.model_extra`
    returns the same raw dict (a harmless superset of "unknown" fields —
    every caller only ever does `.get(key)` on it, never iterates expecting
    exclusivity)."""
    __slots__ = ("_raw",)

    def __init__(self, raw: Any):
        object.__setattr__(self, "_raw", raw if isinstance(raw, dict) else {})

    def __getattr__(self, name: str) -> Any:
        v = self._raw.get(name)
        if isinstance(v, dict):
            return _Obj(v)
        if isinstance(v, list):
            return [_Obj(x) if isinstance(x, dict) else x for x in v]
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
        for line in self._resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                # A malformed SSE frame here can silently drop a tool-call
                # argument delta, later surfacing as a bogus "malformed tool
                # call" JSON error once the truncated buffer is parsed.
                self.dropped_lines += 1
                continue
            yield _Obj(obj)

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
        req = self._client._client.build_request(
            "GET", "/models",
            **({} if timeout is None else {"timeout": timeout}))
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
                attempt += 1
                time.sleep(delay)
                continue
            resp.close()
            raise APIError(f"Error code: {resp.status_code} - {json.dumps(body) if body else msg}",
                           body=body, status_code=resp.status_code, request=req)
