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
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Literal, overload

import httpx


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
    as opposed to APIError for a reachable server returning an error status."""
    def __init__(self, message: str = "connection error", *, request: httpx.Request | None = None):
        super().__init__(message, request=request)


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
        req = self._client._client.build_request(
            "POST", "/chat/completions", json=payload, timeout=timeout)
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
        req = self._client._client.build_request("GET", "/models", timeout=timeout)
        resp = self._client._send(req, stream=False)
        return _Obj(resp.json())


class OpenAI:
    """Just enough of `openai.OpenAI` for nx: base_url + api_key + optional
    default_headers, `.chat.completions.create()`, `.models.list()`."""
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if default_headers:
            headers.update(default_headers)
        # Short connect timeout so an unreachable server fails fast (matches
        # the openai SDK's practical behavior); generous read timeout since
        # local models can take a while to start streaming.
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=600.0)
        self._client = httpx.Client(base_url=self.base_url, headers=headers,
                                     timeout=timeout)
        self.chat = _Chat(self)
        self.models = _Models(self)

    def _send(self, req: httpx.Request, stream: bool) -> httpx.Response:
        try:
            resp = self._client.send(req, stream=stream)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.NetworkError, httpx.TimeoutException) as e:
            raise APIConnectionError(str(e), request=req) from e
        if resp.status_code >= 400:
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
            resp.close()
            raise APIError(f"Error code: {resp.status_code} - {json.dumps(body) if body else msg}",
                           body=body, status_code=resp.status_code, request=req)
        return resp
