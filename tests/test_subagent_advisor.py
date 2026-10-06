"""Advisor-tier sub-agents (0.112.0): the executor consults the parent model
via consult_advisor — a one-shot, no-tools side-call. Covers tier gating, the
dispatch/tool-schema surface, budget exhaustion, timeout/malformed fallbacks,
the stop signal, and that the tool never reaches the always-on token floor."""
import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for var in ("NX_ENV_FILE", "NX_SUBAGENT_WRITE", "NX_SUBAGENT_CONSULT_BUDGET",
            "NX_SUBAGENT_CONSULT_TIMEOUT", "NX_SUBAGENT_MODEL", "NX_SUBAGENT_SOURCE"):
    os.environ.pop(var, None)
os.environ["NX_ENV_FILE"] = "/dev/null"

import pytest

from nx import _subagent_tiers as tiers
from nx import _subagent as sub


# ---------------------------------------------------------------- tier gating

def test_normalize_tier_accepts_advisor():
    assert tiers._normalize_tier("advisor") == "advisor"
    assert tiers._normalize_tier("ADVISOR") == "advisor"
    assert tiers._normalize_tier(" implement ") == "implement"
    assert tiers._normalize_tier(None) == "read"
    assert tiers._normalize_tier("bogus") == ""


def test_advisor_dispatch_is_write_table_plus_consult():
    table = tiers._advisor_tier_dispatch()
    assert "consult_advisor" in table
    for name in tiers._WRITE_TIER_TOOL_NAMES:
        assert name in table
    assert "run_bash" in table


def test_consult_tool_not_on_always_on_floor():
    import nx
    names = {t["function"]["name"] for t in nx.TOOLS}
    assert "consult_advisor" not in names


def test_advisor_tier_requires_write_optin(monkeypatch):
    monkeypatch.setattr(nx := __import__("nx"), "NX_SUBAGENT_WRITE", 0, raising=False)
    out = sub.spawn_agent("do the thing", tier="advisor", async_=False)
    assert "NX_SUBAGENT_WRITE=1" in out


def test_read_tier_has_no_consult_tool():
    ctx = tiers._SubagentContext(tiers._read_only_dispatch())
    tiers._SUBAGENT_TLS.ctx = ctx
    try:
        names = {t["function"]["name"] for t in tiers._subagent_tools()}
        assert "consult_advisor" not in names
    finally:
        tiers._SUBAGENT_TLS.ctx = None


def test_advisor_tools_include_consult_schema():
    ctx = tiers._SubagentContext(tiers._advisor_tier_dispatch())
    ctx.advisor = True
    tiers._SUBAGENT_TLS.ctx = ctx
    try:
        names = {t["function"]["name"] for t in tiers._subagent_tools()}
        assert "consult_advisor" in names
    finally:
        tiers._SUBAGENT_TLS.ctx = None


def test_write_tier_does_not_get_consult_tool():
    ctx = tiers._SubagentContext(tiers._write_tier_dispatch(), write=True)
    tiers._SUBAGENT_TLS.ctx = ctx
    try:
        names = {t["function"]["name"] for t in tiers._subagent_tools()}
        assert "consult_advisor" not in names
    finally:
        tiers._SUBAGENT_TLS.ctx = None


# ------------------------------------------------------- the consult side-call

class _FakeResp:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _FakeClient:
    def __init__(self, replies, lock=None):
        self.replies = list(replies)
        self.calls = []
        self._lock = lock or threading.Lock()

    class _Chat:
        def __init__(self, outer):
            self._outer = outer

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kw):
                with self._outer._lock:
                    self._outer.calls.append(kw)
                    if not self._outer.replies:
                        raise RuntimeError("no scripted reply")
                    item = self._outer.replies.pop(0)
                if isinstance(item, Exception):
                    raise item
                import time as _t
                if isinstance(item, tuple):
                    _t.sleep(item[0])
                    item = item[1]
                return _FakeResp(item)

        @property
        def completions(self):
            return self._Completions(self._outer)

    @property
    def chat(self):
        return self._Chat(self)


@pytest.fixture()
def advisor_ctx():
    ctx = tiers._SubagentContext(tiers._advisor_tier_dispatch())
    ctx.advisor = True
    tiers._SUBAGENT_TLS.ctx = ctx
    yield ctx
    tiers._SUBAGENT_TLS.ctx = None


def _with_source(monkeypatch, client, model="test-model"):
    from nx import _state as st
    monkeypatch.setattr(st, "client", client, raising=False)
    monkeypatch.setattr(st, "MODEL", model, raising=False)
    monkeypatch.setattr(st, "ACTIVE", None, raising=False)


def test_consult_returns_decision_and_guidance(monkeypatch, advisor_ctx):
    client = _FakeClient([json.dumps({"decision": "option B", "guidance": "safer", "stop": False})])
    _with_source(monkeypatch, client)
    out = tiers._consult_advisor(question="A or B?", options=["A", "B"],
                                 context_digest="plan: x")
    assert "[advisor] decision: option B" in out
    assert "[advisor] guidance: safer" in out
    call = client.calls[0]
    assert call["stream"] is False
    assert call["model"] == "test-model"
    body = call["messages"]
    assert body[0]["role"] == "system"
    assert "advisor" in body[0]["content"].lower()
    assert "A or B?" in body[1]["content"]
    assert "- A" in body[1]["content"]


def test_consult_budget_exhausts_to_conservative(monkeypatch):
    os.environ["NX_SUBAGENT_CONSULT_BUDGET"] = "2"
    advisor_ctx = tiers._SubagentContext(tiers._advisor_tier_dispatch())
    advisor_ctx.advisor = True
    tiers._SUBAGENT_TLS.ctx = advisor_ctx
    try:
        client = _FakeClient([json.dumps({"decision": "x", "guidance": "", "stop": False})] * 2)
        _with_source(monkeypatch, client)
        for _ in range(2):
            out = tiers._consult_advisor(question="q")
            assert out.startswith("[advisor] decision: x")
        out = tiers._consult_advisor(question="q")
        assert out.startswith("ERROR: advisor consult budget exhausted")
        assert "conservative" in out
        assert "## Deviations" in out
        assert len(client.calls) == 2  # no call was made past the budget
    finally:
        os.environ.pop("NX_SUBAGENT_CONSULT_BUDGET", None)


def test_consult_timeout_falls_back(monkeypatch, advisor_ctx):
    os.environ["NX_SUBAGENT_CONSULT_TIMEOUT"] = "1"
    try:
        # (2s, reply) exceeds the 1s deadline -> the loop exits before create
        client = _FakeClient([(2.0, json.dumps({"decision": "x", "stop": False}))])
        _with_source(monkeypatch, client)
        out = tiers._consult_advisor(question="q")
        assert out.startswith("ERROR: advisor did not answer in time")
        assert "conservative" in out
    finally:
        os.environ.pop("NX_SUBAGENT_CONSULT_TIMEOUT", None)


def test_consult_malformed_reply_retries_then_falls_back(monkeypatch, advisor_ctx):
    client = _FakeClient(["not json at all", "still not json"])
    _with_source(monkeypatch, client)
    out = tiers._consult_advisor(question="q")
    assert out.startswith("ERROR: advisor reply unusable")
    assert "unparseable" in out
    assert len(client.calls) == 2  # one parse retry


def test_consult_stop_signal(monkeypatch, advisor_ctx):
    client = _FakeClient([json.dumps({"decision": "", "guidance": "out of scope", "stop": True})])
    _with_source(monkeypatch, client)
    out = tiers._consult_advisor(question="should I stop?")
    assert "STOP" in out
    assert "implementation-notes.md" in out


def test_consult_requires_question(monkeypatch, advisor_ctx):
    client = _FakeClient([])
    _with_source(monkeypatch, client)
    out = tiers._consult_advisor(question="  ")
    assert out.startswith("ERROR: consult_advisor requires question")
    assert not client.calls  # budget not spent on a malformed call


def test_consult_outside_subagent_refused(monkeypatch):
    tiers._SUBAGENT_TLS.ctx = None
    out = tiers._consult_advisor(question="q")
    assert out.startswith("ERROR: consult_advisor used outside a sub-agent")
