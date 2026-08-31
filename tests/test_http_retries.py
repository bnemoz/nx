#!/usr/bin/env python3
"""Transient-error retry in nx/_openai_client._send.

429/5xx responses and connections that broke mid-request are retried with
exponential backoff (or the server's Retry-After when it sends one);
refused/unreachable endpoints and deterministic 4xx are not; and — the safety
invariant — nothing is ever retried once a streaming response has started
yielding bytes. All against httpx.MockTransport, no live server; time.sleep is
patched out so backoff delays are recorded, never waited.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-httpretry-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_HOME"] = _tmp
for k in ("NX_HTTP_RETRIES", "NX_HTTP_BACKOFF_BASE", "NX_BASE_URL", "NX_MODEL",
          "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
import pytest  # noqa: E402

from nx._openai_client import APIConnectionError, APIError, OpenAI  # noqa: E402


@pytest.fixture()
def sleeps(monkeypatch):
    """Record backoff delays instead of waiting them. _openai_client calls
    time.sleep, i.e. the global module's attr — monkeypatch restores it."""
    seen = []
    monkeypatch.setattr("nx._openai_client.time.sleep", seen.append)
    return seen


def _client(handler):
    return OpenAI(base_url="http://test/v1", api_key="sk-test",
                  transport=httpx.MockTransport(handler))


def _ok(content=b'{"data": []}'):
    return httpx.Response(200, content=content)


def test_429_retried_with_exponential_backoff_then_success(sleeps, capsys):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return _ok(b'{"choices": [{"message": {"content": "ok"}}]}')

    resp = _client(handler).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=False)
    assert resp.choices[0].message.content == "ok"
    assert attempts["n"] == 3          # 1 attempt + 2 default retries
    assert sleeps == [0.5, 1.0]        # base * 2**attempt, default base 0.5
    err = capsys.readouterr().err
    assert err.count("retrying 1/2") == 1 and err.count("retrying 2/2") == 1
    assert "rate limited" in err


def test_retry_after_header_wins_over_backoff(sleeps):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, headers={"Retry-After": "7"},
                                  json={"error": {"message": "overloaded"}})
        return _ok()

    _client(handler).models.list()
    assert attempts["n"] == 2
    assert sleeps == [7.0]             # server's value, not 0.5 backoff


def test_retries_exhausted_raises_api_error(sleeps):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(502, json={"error": {"message": "bad gateway"}})

    with pytest.raises(APIError) as exc:
        _client(handler).models.list()
    assert exc.value.status_code == 502
    assert "bad gateway" in exc.value.message
    assert attempts["n"] == 3          # default NX_HTTP_RETRIES=2
    assert sleeps == [0.5, 1.0]


def test_zero_retries_disables_the_loop(sleeps, monkeypatch):
    monkeypatch.setenv("NX_HTTP_RETRIES", "0")
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    with pytest.raises(APIError):
        _client(handler).models.list()
    assert attempts["n"] == 1
    assert sleeps == []


def test_deterministic_4xx_is_never_retried(sleeps):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(APIError) as exc:
        _client(handler).models.list()
    assert exc.value.status_code == 401
    assert attempts["n"] == 1
    assert sleeps == []


def test_connect_refused_is_never_retried(sleeps):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(APIConnectionError) as exc:
        _client(handler).models.list()
    assert exc.value.retryable is False   # down, not busy — fail fast
    assert attempts["n"] == 1
    assert sleeps == []


def test_read_timeout_is_retried_then_success(sleeps):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return _ok()

    _client(handler).models.list()
    assert attempts["n"] == 2
    assert sleeps == [0.5]


def test_stream_retry_happens_before_first_byte(sleeps):
    """stream=True: a 503 before the response headers is still retried — the
    guard is "no retries once the stream is open", not "no retries when
    stream=True was requested"."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        body = (b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
                b"data: [DONE]\n\n")
        return httpx.Response(200, content=body)

    stream = _client(handler).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    assert attempts["n"] == 2
    assert sleeps == [0.5]
    chunks = [c.choices[0].delta.content for c in stream]
    assert chunks == ["hi"]
    assert stream.dropped_lines == 0


def test_never_retry_mid_stream(sleeps):
    """The invariant: once the response headers are in and the SSE body has
    started, a mid-stream connection break must surface to the caller, not be
    transparently replayed — tokens were already rendered."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1

        def body():
            yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            raise httpx.ReadError("connection reset mid-stream")

        return httpx.Response(200, content=body())

    stream = _client(handler).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    with pytest.raises(httpx.ReadError):
        list(stream)
    assert attempts["n"] == 1           # no second attempt
    assert sleeps == []


def test_backoff_base_env_knob(sleeps, monkeypatch):
    monkeypatch.setenv("NX_HTTP_BACKOFF_BASE", "0.1")
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return _ok()

    _client(handler).models.list()
    assert sleeps == [0.1]              # base * 2**0


def test_retry_after_is_capped(sleeps, monkeypatch):
    monkeypatch.setenv("NX_HTTP_RETRIES", "1")
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3600"},
                                  json={"error": {"message": "slow down"}})
        return _ok()

    _client(handler).models.list()
    assert attempts["n"] == 2
    assert sleeps == [30.0]             # capped: never stall a turn for 1h
