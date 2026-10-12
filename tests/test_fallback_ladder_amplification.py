#!/usr/bin/env python3
"""open_stream's no-tools/no-extra fallback ladder only re-sends on a FIELD rejection.

The gap this pins down: the ladder used to fire for ANY exception that wasn't
APIConnectionError/httpx.HTTPError. But _send (nx/_openai_client.py) already
retries 429/5xx NX_HTTP_RETRIES times with backoff before raising APIError —
so the ladder was a second, invisible retry layer on top of it: one degraded
turn became up to 3 logical requests x 3 attempts = 9 POSTs (plus sleep
backoff per attempt) against an endpoint already drowning, and tens of
seconds of latency before the turn gave up.

Contract under test:
- _is_request_field_rejection(): True only for a 400/422 APIError (the
  server parsed the request and objected to a field); False for 429/5xx,
  401/403/404, and status-less exotic shapes — those must propagate.
- open_stream() with a 429/5xx APIError performs exactly ONE create() call
  (no ladder rungs), latches st._LAST_API_ERROR, returns None.
- Same for a status-less error shape (e.g. a proxy SDK raising something
  unexpected): one call, no amplification.
- A genuine tools=-rejected 400 still walks the ladder (no-tools, then
  no-tools-no-extra) — the fallback the ladder exists for still works.
- A ladder rung failing with a 5xx surfaces immediately instead of
  advancing to the next rung.

Offline: the OpenAI client is faked; never a real endpoint.

Tier A: nx is imported ONCE and never reloaded. The old nx_fresh fixture's
importlib.reload(nx) only re-read the NX_HOME/NX_SESSIONS_DIR env and rewound
the handful of nx globals these tests touch (the fake source entry plus the
latched _LAST_API_ERROR/_LAST_CONTEXT_OVERRUN signals). With the reload gone,
conftest's universal env rewind covers the env, and every mutated nx global
goes through monkeypatch (per-test, auto-restored) instead of the module-dict
rewind.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_SOURCES", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


class _Field400(Exception):
    """Server rejected the tools= field: deterministic 400 with a body."""
    def __init__(self, msg="Error code: 400 - tools is not supported"):
        super().__init__(msg)
        self.message = msg
        self.body = {"error": {"message": msg}}
        self.status_code = 400


class _StatusErr(Exception):
    """Shape of the APIError _send raises after exhausting NX_HTTP_RETRIES on
    a 429/5xx (the outage-amplifier case)."""
    def __init__(self, msg="Error code: 503 - upstream overloaded"):
        super().__init__(msg)
        self.message = msg
        self.body = {"error": {"message": msg}}
        self.status_code = 503


class _Auth401(Exception):
    def __init__(self):
        super().__init__("Error code: 401 - invalid api key")
        self.message = "Error code: 401 - invalid api key"
        self.body = {"error": {"message": "invalid api key"}}
        self.status_code = 401


class _Shapeless(Exception):
    """Unexpected error shape: no status_code at all."""
    def __init__(self):
        super().__init__("malformed proxy reply")
        self.message = "malformed proxy reply"


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    monkeypatch.chdir(tmp_path)
    import nx
    # No reload: nx is imported once (Tier A). The reload used to rewind the
    # globals these tests latch; pin the two latched signals through monkeypatch
    # so each test starts from a clean slate and teardown restores them (this
    # stands in for the module-dict rewind). NX_* env rewind is universal via
    # conftest, so no manual pop on teardown.
    monkeypatch.setattr(nx.st, "_LAST_API_ERROR", None)
    monkeypatch.setattr(nx.st, "_LAST_CONTEXT_OVERRUN", None)
    yield nx, tmp_path


def _install_source(nx, monkeypatch):
    # The old reload re-ran nx's import-time source discovery; here the tests
    # install their own fake source. Register the mutated globals through
    # monkeypatch (the SOURCES dict, SOURCE_ORDER list, and ACTIVE binding are
    # replaced wholesale and auto-restored) instead of the Tier-B module-dict
    # rewind.
    from nx import _sources as _sources_mod
    src = _sources_mod.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model")
    monkeypatch.setattr(nx.st, "SOURCES", {**nx.st.SOURCES, "fake": src})
    monkeypatch.setattr(nx.st, "SOURCE_ORDER", [*nx.st.SOURCE_ORDER, "fake"])
    monkeypatch.setattr(nx.st, "ACTIVE", src)
    return src


def _quiet(monkeypatch):
    from nx._subagent import _ASYNC_QUIET_SLOTS
    monkeypatch.setattr(_ASYNC_QUIET_SLOTS, "quiet", True, raising=False)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def _client(monkeypatch, behavior):
    """A client whose create() runs `behavior(calls)`; `calls` is the shared
    list the test inspects for how many POSTs the turn produced."""
    calls: list = []

    def create(*a, **k):
        calls.append(1)
        return behavior(calls)

    cp = type("Cp", (), {"create": staticmethod(lambda *a, **k: create(*a, **k))})
    ch = type("Ch", (), {"completions": cp()})
    return type("C", (), {"chat": ch()})(), calls


# --- classifier -------------------------------------------------------------------

def test_field_rejection_classifier(nx_fresh):
    nx, _ = nx_fresh
    from nx._compress import _is_request_field_rejection
    yes = [
        _Field400(),
        _Field400("Error code: 422 - Unrecognized request argument: tools"),
        type("E422", (_Field400,), {})("Error code: 422 - extra_forbidden"),
    ]
    yes[2].status_code = 422
    no = [
        _StatusErr(),      # 503 after retries — transport/status grade
        _Auth401(),        # 401 — deterministic, not a body-shedding fix
        _Shapeless(),      # no status_code — no evidence the server parsed us
    ]
    e429 = _StatusErr("Error code: 429 - rate limited")
    e429.status_code = 429
    no.append(e429)
    for e in yes:
        assert _is_request_field_rejection(e), e
    for e in no:
        assert not _is_request_field_rejection(e), e


# --- open_stream: one POST, no ladder, honest latch -------------------------------

@pytest.mark.parametrize("exc", [_StatusErr(), _Auth401(), _Shapeless()],
                         ids=["503", "401", "shapeless"])
def test_open_stream_no_ladder_on_status_or_shape_errors(nx_fresh, monkeypatch, exc):
    nx, _ = nx_fresh
    _install_source(nx, monkeypatch)
    _quiet(monkeypatch)

    def behavior(calls):
        raise exc
    client, calls = _client(monkeypatch, behavior)
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is None
    assert calls == [1], "no fallback re-send for a non-field rejection"
    assert nx.st._LAST_API_ERROR == f"{type(exc).__name__}: {exc}"
    assert nx.st._LAST_CONTEXT_OVERRUN is None


def test_open_stream_still_walks_ladder_on_tools_400(nx_fresh, monkeypatch):
    """The ladder's reason for existing: a server that 400s on the tools= /
    tool_choice fields gets the no-tools then no-tools-no-extra re-sends."""
    nx, _ = nx_fresh
    _install_source(nx, monkeypatch)
    _quiet(monkeypatch)
    stages: list = []

    def behavior(calls):
        n = len(calls)
        if n == 1:
            stages.append("with-tools")
            raise _Field400()
        if n == 2:
            stages.append("no-tools")
            raise _Field400("Error code: 400 - extra_body rejected")
        return "stream"

    client, calls = _client(monkeypatch, behavior)
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    stream = nx.open_stream([{"role": "user", "content": "hi"}])
    assert stream is not None, "no-tools-no-extra rung must succeed"
    assert stages == ["with-tools", "no-tools"]
    assert len(calls) == 3
    assert nx.st._LAST_API_ERROR is None


def test_ladder_rung_5xx_surfaces_dont_advance(nx_fresh, monkeypatch):
    """A rung failing with a 5xx stops the ladder right there — advancing to
    the next rung would just re-send into the same degradation."""
    nx, _ = nx_fresh
    _install_source(nx, monkeypatch)
    _quiet(monkeypatch)

    def behavior(calls):
        if len(calls) == 1:
            raise _Field400()
        raise _StatusErr()

    client, calls = _client(monkeypatch, behavior)
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is None
    assert len(calls) == 2, "5xx on a rung must not advance to the next rung"
    assert nx.st._LAST_API_ERROR and "503" in nx.st._LAST_API_ERROR


def test_ladder_rung_overrun_latches_context_signal(nx_fresh, monkeypatch):
    """A rung that hits the context-400 wording folds into the overrun
    recovery path (compress + retry), not the terminal-error path."""
    nx, _ = nx_fresh
    _install_source(nx, monkeypatch)
    _quiet(monkeypatch)

    def behavior(calls):
        if len(calls) == 1:
            raise _Field400()
        raise _Field400("Error code: 400 - maximum context length is 8192 tokens")

    client, calls = _client(monkeypatch, behavior)
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is None
    assert len(calls) == 2
    assert nx.st._LAST_CONTEXT_OVERRUN and "maximum context length" in nx.st._LAST_CONTEXT_OVERRUN
    assert nx.st._LAST_API_ERROR is None


# --- terminal mapping is unchanged -------------------------------------------------

def test_status_error_still_maps_to_api_error_turn(nx_fresh, monkeypatch):
    nx, _ = nx_fresh
    _install_source(nx, monkeypatch)
    _quiet(monkeypatch)
    client, calls = _client(monkeypatch, lambda calls: (_ for _ in ()).throw(_StatusErr()))
    monkeypatch.setattr(nx.st, "client", client)
    monkeypatch.setattr(nx.st, "MODEL", "fake-model")
    messages = nx._initial_messages() + [{"role": "user", "content": "x"}]
    assert nx.model_turn(messages) == nx._turn.TURN_API_ERROR
    assert calls == [1], "one degraded turn, one logical request"
