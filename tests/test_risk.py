"""_risk._assess_risk after the 0.156.3 side-call migration: the shared
helper owns the call/metrics/parse-retry machinery; this file pins what
_assess_risk still owns — the (level, reason) mapping, the regex level-word
fallback over raw prose (only when the reply never parsed as JSON), the
conservative high default, and the three-attempt ladder with the sharp
follow-up."""
import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

from nx import _risk
from nx import _state as st
from nx._openai_client import APIConnectionError


class _FakeResp:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=text))]

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeClient:
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
def risk_env(monkeypatch):
    monkeypatch.setattr(st, "ACTIVE", None, raising=False)
    monkeypatch.setattr(st, "MODEL", "test-model", raising=False)
    monkeypatch.setattr(st, "RISK_CONNECTION_RETRY_SECONDS", 0.0, raising=False)


def _client(monkeypatch, replies):
    c = _FakeClient(replies)
    monkeypatch.setattr(st, "client", c, raising=False)
    return c


def test_no_source_defaults_high(monkeypatch, risk_env):
    monkeypatch.setattr(st, "client", None, raising=False)
    monkeypatch.setattr(st, "MODEL", None, raising=False)
    assert _risk._assess_risk("run: ls") == ("high", "no active source; defaulting to high")


def test_json_verdict_parsed(monkeypatch, risk_env):
    c = _client(monkeypatch, [json.dumps({"level": "Medium", "reason": "  edits x  "})])
    assert _risk._assess_risk("edit x") == ("medium", "edits x")
    assert c.calls[0]["stream"] is False
    assert c.calls[0]["timeout"] == 15
    assert c.calls[0]["messages"][0]["role"] == "system"


def test_empty_reason_falls_back_to_level(monkeypatch, risk_env):
    _client(monkeypatch, [json.dumps({"level": "low", "reason": ""})])
    assert _risk._assess_risk("run: ls")[0] == "low"


def test_prose_finds_level_word(monkeypatch, risk_env):
    _client(monkeypatch, ["this is a High risk change, delete it"] * 3)
    level, reason = _risk._assess_risk("run: rm -rf /")
    assert level == "high"
    assert reason.startswith("this is a High")


def test_parsed_json_with_unknown_level_stays_high(monkeypatch, risk_env):
    # regex fallback must NOT fire when JSON parsed (matching a stray word
    # inside a reason like "lowest impact" could under-rate the action)
    _client(monkeypatch, [json.dumps({"level": "extreme", "reason": "low impact"})] * 3)
    level, reason = _risk._assess_risk("edit x")
    assert level == "high"
    assert "unparseable risk response" in reason


def test_three_attempt_ladder_with_sharp_follow_up(monkeypatch, risk_env):
    c = _client(monkeypatch, ["garbage one", "garbage two",
                              json.dumps({"level": "low", "reason": "ok"})])
    assert _risk._assess_risk("run: ls") == ("low", "ok")
    assert len(c.calls) == 3
    # attempt 1: pristine payload; attempt 2: assistant echo + follow-up
    assert len(c.calls[0]["messages"]) == 2
    msgs = c.calls[1]["messages"]
    assert msgs[2] == {"role": "assistant", "content": "garbage one"}
    assert msgs[3]["role"] == "user"
    assert "ONLY a single JSON object" in msgs[3]["content"]


def test_unparseable_after_three_attempts_defaults_high(monkeypatch, risk_env):
    c = _client(monkeypatch, ["no level word here"] * 3)
    level, reason = _risk._assess_risk("edit x")
    assert level == "high"
    assert "unparseable risk response after 3 attempts" in reason
    assert len(c.calls) == 3


def test_connection_ladder_exhausted_defaults_high(monkeypatch, risk_env):
    monkeypatch.setattr(st, "RISK_CONNECTION_RETRIES", 2, raising=False)
    c = _client(monkeypatch, [APIConnectionError("down")] * 3)
    level, reason = _risk._assess_risk("edit x")
    assert level == "high"
    # the ladder is spent once for the whole call, not per parse attempt
    assert len(c.calls) == 3


def test_connection_flake_then_answer(monkeypatch, risk_env):
    monkeypatch.setattr(st, "RISK_CONNECTION_RETRIES", 2, raising=False)
    _client(monkeypatch, [APIConnectionError("down"),
                          json.dumps({"level": "low", "reason": "ls"})])
    assert _risk._assess_risk("run: ls") == ("low", "ls")
