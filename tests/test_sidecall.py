"""nx/_sidecall.py (0.156.3): the shared one-shot JSON side call. Pins the
(helper contract, no network): (dict, raw) on success, None-on-every-failure
with raw still set, metrics recorded with BOTH chars//4 fallbacks, the
retry_prompt payload append firing only between parse attempts, the shared
connection-retry ladder, and loose_json brace-slicing."""
import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

from nx import _sidecall
from nx import _state as st
from nx._openai_client import APIConnectionError


class _FakeResp:
    def __init__(self, text, usage=None):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=text))]
        self.usage = usage

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeClient:
    """Scripted replies: str -> that reply; Exception -> raised."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self._lock = threading.Lock()

    class _Chat:
        def __init__(self, outer):
            self._outer = outer

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kw):
                with self._outer._lock:
                    # deepcopy: the helper appends to its payload list
                    # in-place; record a snapshot of what THIS call saw
                    import copy
                    self._outer.calls.append(copy.deepcopy(kw))
                    if not self._outer.replies:
                        raise AssertionError("no scripted reply left")
                    item = self._outer.replies.pop(0)
                if isinstance(item, Exception):
                    raise item
                return _FakeResp(item)

        @property
        def completions(self):
            return self._Completions(self._outer)

    @property
    def chat(self):
        return self._Chat(self)


@pytest.fixture()
def source(monkeypatch):
    """A client on st (ACTIVE None: no extra_request_kwargs) + a live metrics
    slot so _record_api_call lands somewhere we can inspect. nx_state_reset
    is not needed — nothing here adds module globals."""
    from nx._metrics import _Metrics
    m = _Metrics("test-sidecall")
    monkeypatch.setattr(st, "_METRICS", m, raising=False)
    monkeypatch.setattr(st, "_BATCH_METRICS", None, raising=False)
    monkeypatch.setattr(st, "ACTIVE", None, raising=False)
    monkeypatch.setattr(st, "MODEL", "test-model", raising=False)
    yield m


def _client(monkeypatch, replies):
    c = _FakeClient(replies)
    monkeypatch.setattr(st, "client", c, raising=False)
    return c


def _unpatched_client(monkeypatch):
    monkeypatch.setattr(st, "client", None, raising=False)
    monkeypatch.setattr(st, "MODEL", None, raising=False)


# ------------------------------------------------------------- the contract

def test_success_returns_dict_and_raw(source, monkeypatch):
    payload_txt = json.dumps({"level": "low", "reason": "read-only ls"})
    c = _client(monkeypatch, [f"  {payload_txt}  "])
    obj, raw = _sidecall.side_call_json("SYS", "USER")
    assert obj == {"level": "low", "reason": "read-only ls"}
    assert raw == payload_txt  # stripped
    call = c.calls[0]
    assert call["stream"] is False
    assert call["timeout"] == 30.0
    assert call["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "USER"}]


def test_no_source_returns_none_empty(source, monkeypatch):
    _unpatched_client(monkeypatch)
    assert _sidecall.side_call_json("S", "U") == (None, "")


def test_unparseable_after_one_attempt_is_none_with_raw(source, monkeypatch):
    c = _client(monkeypatch, ["not json"])
    obj, raw = _sidecall.side_call_json("S", "U")
    assert obj is None
    assert raw == "not json"
    assert len(c.calls) == 1  # parse_attempts=1: exactly one call


def test_json_scalar_or_list_is_none(source, monkeypatch):
    c = _client(monkeypatch, ["[1, 2]"])
    obj, raw = _sidecall.side_call_json("S", "U")
    assert obj is None
    assert raw == "[1, 2]"


# ---------------------------------------------------------------- metrics

def test_metrics_recorded_with_both_fallbacks(source, monkeypatch):
    c = _client(monkeypatch, ['{"a": 1}'])
    obj, raw = _sidecall.side_call_json("SYS", "USER")
    assert obj == {"a": 1}
    assert source.api_calls == 1
    # usage=None -> chars//4 estimates on BOTH axes (the prompt-char fallback
    # is the one /build's own call used to miss)
    expected_in = max(1, (len("SYS") + len("USER")) // 4)
    expected_out = max(1, len('{"a": 1}') // 4)
    assert source.estimated_calls == 1
    assert source.input_tokens == expected_in
    assert source.output_tokens == expected_out


def test_metrics_recorded_once_per_attempt(source, monkeypatch):
    c = _client(monkeypatch, ["nope", '{"a": 2}'])
    obj, raw = _sidecall.side_call_json("S", "U", parse_attempts=2)
    assert obj == {"a": 2}
    assert len(c.calls) == 2
    assert source.api_calls == 2


# ------------------------------------------------- retry_prompt / attempts

def test_retry_prompt_appended_only_between_attempts(source, monkeypatch):
    c = _client(monkeypatch, ["garbage", '{"a": 3}'])
    obj, raw = _sidecall.side_call_json(
        "S", "U", parse_attempts=3, retry_prompt="ONLY JSON.")
    assert obj == {"a": 3}
    assert len(c.calls) == 2
    # first call: pristine two-message payload
    assert c.calls[0]["messages"] == [{"role": "system", "content": "S"},
                                      {"role": "user", "content": "U"}]
    # second call: assistant echo + the re-ask appended
    second = c.calls[1]["messages"]
    assert second[:2] == c.calls[0]["messages"]
    assert second[2] == {"role": "assistant", "content": "garbage"}
    assert second[3] == {"role": "user", "content": "ONLY JSON."}


def test_retry_prompt_not_appended_after_last_attempt(source, monkeypatch):
    c = _client(monkeypatch, ["x1", "x2"])
    obj, raw = _sidecall.side_call_json(
        "S", "U", parse_attempts=2, retry_prompt="ONLY JSON.")
    assert obj is None
    assert raw == "x2"
    assert len(c.calls) == 2
    # the second (final) call's payload still has no retry tail
    assert len(c.calls[1]["messages"]) == 4  # appended once, after attempt 1
    assert len(c.calls[-1]["messages"]) == len(c.calls[1]["messages"])


def test_parse_attempts_without_retry_prompt_reshapes_nothing(source, monkeypatch):
    c = _client(monkeypatch, ["nope", '{"ok": true}'])
    obj, raw = _sidecall.side_call_json("S", "U", parse_attempts=2)
    assert obj == {"ok": True}
    assert c.calls[0]["messages"] == c.calls[1]["messages"]


# ------------------------------------------------------------ conn ladder

def test_conn_retries_ladder_shared_across_parse_attempts(source, monkeypatch):
    c = _client(monkeypatch, [APIConnectionError("down"), APIConnectionError("down"),
                              '{"a": 1}'])
    obj, raw = _sidecall.side_call_json("S", "U", conn_retries=2,
                                        conn_retry_seconds=0.0)
    assert obj == {"a": 1}
    assert len(c.calls) == 3


def test_conn_ladder_exhausted_returns_none(source, monkeypatch):
    c = _client(monkeypatch, [APIConnectionError("down")] * 3)
    obj, raw = _sidecall.side_call_json("S", "U", conn_retries=2,
                                        parse_attempts=3, conn_retry_seconds=0.0)
    assert obj is None
    assert raw == ""
    # the ladder is consumed once for the WHOLE call, not per parse attempt
    assert len(c.calls) == 3


def test_other_exception_never_raises(source, monkeypatch):
    _client(monkeypatch, [RuntimeError("boom")])
    assert _sidecall.side_call_json("S", "U") == (None, "")


# --------------------------------------------------------------- loose_json

def test_loose_json_brace_slices_prose(source, monkeypatch):
    _client(monkeypatch, ['Sure! Here you go:\n{"path": "a.py", "content": "x"}\nDone.'])
    obj, raw = _sidecall.side_call_json("S", "U", loose_json=True)
    assert obj == {"path": "a.py", "content": "x"}
    assert raw.startswith("Sure!")


def test_strict_rejects_the_same_prose(source, monkeypatch):
    _client(monkeypatch, ['wrap {"a": 1} wrap'])
    obj, raw = _sidecall.side_call_json("S", "U")
    assert obj is None
    assert raw == 'wrap {"a": 1} wrap'


def test_loose_json_no_braces_is_none(source, monkeypatch):
    _client(monkeypatch, ["no braces here"])
    obj, raw = _sidecall.side_call_json("S", "U", loose_json=True)
    assert obj is None
    assert raw == "no braces here"


# ----------------------------------------------------- extra_request_kwargs

def test_extra_request_kwargs_spread(source, monkeypatch):
    c = _client(monkeypatch, ['{"a": 1}'])
    monkeypatch.setattr(st, "ACTIVE",
                        types.SimpleNamespace(
                            name="test-src",  # _Metrics.add stamps the source name
                            extra_request_kwargs=lambda model=None: {"temperature": 0.2}),
                        raising=False)
    obj, raw = _sidecall.side_call_json("S", "U")
    assert obj == {"a": 1}
    assert c.calls[0]["temperature"] == 0.2


# ----------------------------------------------------------- log_event seam

def test_caller_log_event_override_used(source, monkeypatch):
    _client(monkeypatch, ['{"a": 1}'])
    seen = []
    obj, raw = _sidecall.side_call_json("S", "U",
                                        log_event=lambda d, p: seen.append(d))
    assert obj == {"a": 1}
    assert seen == ["req", "resp"]
