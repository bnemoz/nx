#!/usr/bin/env python3
"""One cached capability manifest per (source, model) — PERF.md 0.59.0.

Three findings, one fix (see nx/_manifest.py):

- **T-11 / PF-11** — every turn's HTTP request count is pinned. Against a local
  mock server, a plain turn and a turn with one tool call each issue an exact,
  recorded number of POSTs. The no-tools / no-extra field-rejection ladder is
  walked **once** per (source, model) and then the learned shape is reused —
  the pre-0.59.0 code re-derived it by failing, up to 3 full-prompt POSTs per
  turn, against an endpoint that had already rejected the first one.
- **PF-10** — `extra_request_kwargs()` carries a stable per-session routing
  identity (`user`, plus `prompt_cache_key` in the body): constant within a
  session so a router can pin it to a replica/KV block, and opt-out-able.
- **PF-21** — `Source.resolve_model()` is memoized, so a `/source` switch and a
  `resolve_context_window()` fallback do not re-pay `/v1/models` + `/props`.

Everything is offline: a fake `create()` counts calls, a fake client counts
`/v1/models` requests, and the mock HTTP server is a local
`ThreadingHTTPServer` shut down at the end of each test.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _k in list(os.environ):
    if _k.startswith("NX_"):
        del os.environ[_k]
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SKIP_STARTUP_PROBES"] = "1"

import nx  # noqa: E402
import nx._state as st  # noqa: E402
from nx import _manifest, _sources  # noqa: E402

# Recorded per-turn POST counts (PERF.md §5: recorded, not derived). A plain
# turn is one logical request; a turn plus one native tool call is two.
PLAIN_TURN_POSTS = 1
TOOL_CALL_TURN_POSTS = 2


def _fresh_source(name="fake", model="fake-model"):
    src = _sources.Source(name, "http://127.0.0.1:9/v1", "k", model)
    st.SOURCES[name] = src
    st.SOURCE_ORDER.append(name)
    st.ACTIVE = src
    st.client = src.client
    st.MODEL = model
    _manifest.reset(src)  # a shared Source must not leak a pin between tests
    return src


class _Field400(Exception):
    """A 400 naming the tools= field: the shape the ladder exists for."""

    def __init__(self, msg="Error code: 400 - tools is not supported"):
        super().__init__(msg)
        self.message = msg
        self.body = {"error": {"message": msg}}
        self.status_code = 400


def _client(monkeypatch, behavior):
    """A fake client whose create() runs behavior(calls, kwargs); `calls` is
    inspected for how many POSTs a turn produced."""
    calls = []

    def create(*a, **k):
        calls.append(k)
        return behavior(calls, k)

    cp = type("Cp", (), {"create": staticmethod(lambda *a, **k: create(*a, **k))})
    ch = type("Ch", (), {"completions": cp()})
    monkeypatch.setattr(st, "client", type("C", (), {"chat": ch()})())
    return calls


# --- T-11 / PF-11: the shape is learned once, then pinned ---------------------

def test_shape_ladder_is_walked_once_and_then_pinned(monkeypatch):
    """First turn walks full -> no-tools (2 POSTs, the server's 400 drives it).
    Second turn opens at the pinned rung, so 1 POST -- the whole point of
    PF-11. The pre-0.59.0 code paid the 2 POSTs on every single turn."""
    src = _fresh_source()
    posted = []
    attempt = []

    def behavior(calls, kwargs):
        attempt.append(1)
        posted.append(kwargs.get("tools") is not None)
        if len(attempt) == 1:
            raise _Field400()
        return "stream"

    _client(monkeypatch, behavior)
    stream = nx.open_stream([{"role": "user", "content": "hi"}])
    assert stream is not None, "open_stream failed"
    assert posted == [True, False], f"first turn must walk full -> no-tools: {posted}"
    assert _manifest.pinned_index(src, st.MODEL) == 1, "rung not pinned"

    posted.clear()
    assert nx.open_stream([{"role": "user", "content": "again"}]) is not None
    assert posted == [False], f"pinned rung not reused: {posted}"


def test_pin_only_ever_lowers_the_shape():
    """A later success on a more complete rung must not un-learn a rejection
    the server already gave us."""
    src = _fresh_source()
    _manifest.pin(src, st.MODEL, 2)
    assert _manifest.pinned_index(src, st.MODEL) == 2
    _manifest.pin(src, st.MODEL, 1)  # a fuller shape succeeded
    assert _manifest.pinned_index(src, st.MODEL) == 2, "pin was un-lowered"


def test_model_override_forgets_the_old_models_shape():
    """A per-switch model override points the same Source at a different model;
    the old model's accepted shape says nothing about the new one."""
    src = _fresh_source()
    _manifest.pin(src, "model-a", 2)
    assert _manifest.pinned_index(src, "model-a") == 2
    _sources.switch_source(src.name, model_override="model-b")
    assert st.MODEL == "model-b"
    assert _manifest.pinned_index(src, "model-b") == 0, "new model inherited a pin"


# --- PF-10: stable per-session routing identity -------------------------------

def test_routing_identity_is_stable_within_a_session_and_attached(monkeypatch):
    """The `user` field must be identical for every request of a session (a
    moving value would itself be a cache-buster) and must ride along with the
    extra_body cache key."""
    from nx._metrics import _Metrics
    src = _fresh_source()
    monkeypatch.setattr(st, "_METRICS", _Metrics("20260101-000000-abcdef"),
                        raising=False)
    first = src.extra_request_kwargs()
    second = src.extra_request_kwargs()
    assert first == second, "routing identity moved within a session"
    assert first["user"] == "nx-20260101-000000-abcdef", first
    assert first["extra_body"]["prompt_cache_key"] == first["user"]


def test_routing_identity_absent_without_a_session(monkeypatch):
    """No session recorder (import-time probes, pre-install batch workers) --
    nothing is attached; an empty user field would be worse than none."""
    src = _fresh_source()
    monkeypatch.setattr(st, "_METRICS", None, raising=False)
    assert "user" not in src.extra_request_kwargs()


def test_routing_identity_is_opt_out_able(monkeypatch):
    """NX_CACHE_ROUTING=0 disables the hint globally; the per-source
    NX_SOURCE_FAKE_CACHE_ROUTING wins over it when set."""
    from nx._metrics import _Metrics
    src = _fresh_source()
    monkeypatch.setattr(st, "_METRICS", _Metrics("sid"), raising=False)

    monkeypatch.setenv("NX_CACHE_ROUTING", "0")
    assert "user" not in src.extra_request_kwargs()

    # Per-source opt-IN over a global opt-out.
    monkeypatch.setenv("NX_SOURCE_FAKE_CACHE_ROUTING", "1")
    assert src.extra_request_kwargs()["user"] == "nx-sid"

    # Per-source opt-out over a global opt-in.
    monkeypatch.delenv("NX_CACHE_ROUTING")
    monkeypatch.setenv("NX_SOURCE_FAKE_CACHE_ROUTING", "0")
    assert "user" not in src.extra_request_kwargs()


def test_configured_user_field_wins_over_the_hint(monkeypatch):
    """A source that set its own user (via EXTRA_BODY) keeps it."""
    from nx._metrics import _Metrics
    src = _sources.Source("fake", "http://127.0.0.1:9/v1", "k", "fake-model",
                          extra_body={"prompt_cache_key": "mine"})
    monkeypatch.setattr(st, "_METRICS", _Metrics("sid"), raising=False)
    kw = src.extra_request_kwargs()
    assert kw["extra_body"]["prompt_cache_key"] == "mine"
    assert kw["user"] == "nx-sid"


# --- T-11 / PF-21: HTTP request counts against a local mock -------------------

class _Handler(BaseHTTPRequestHandler):
    """Counts GET /v1/models — PF-21's repeated probe was the drift T-11 pins."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.server.models_hits += 1
        body = json.dumps({"data": [{"id": "served-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _mock_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.models_hits = 0
    srv.daemon_threads = True
    thr = threading.Thread(target=srv.serve_forever, daemon=True)
    thr.start()
    return srv


def test_resolve_model_is_memoized_against_a_live_endpoint():
    """PF-21: the first resolve_model() pays one GET /v1/models; every later
    call for the session is free. Pre-0.59.0 each /source / resume /
    resolve_context_window() fallback re-paid the round trip."""
    srv = _mock_server()
    try:
        port = srv.server_address[1]
        os.environ["NX_SKIP_STARTUP_PROBES"] = "0"  # opt back into the probe
        src = _sources.Source("mock", f"http://127.0.0.1:{port}/v1", "k", None)
        assert src.resolve_model() == "served-model"
        assert srv.models_hits == 1, srv.models_hits
        assert src.resolve_model() == "served-model"
        assert src.resolve_model() == "served-model"
        assert srv.models_hits == 1, "resolve_model re-probed after memoizing"
    finally:
        os.environ["NX_SKIP_STARTUP_PROBES"] = "1"
        srv.shutdown()
        srv.server_close()


def test_plain_turn_posts_exactly_plain_turn_posts(monkeypatch):
    """T-11: a plain turn is PLAIN_TURN_POSTS POSTs — no probe, no ladder."""
    _fresh_source()
    monkeypatch.setattr(st, "_METRICS", None, raising=False)
    calls = _client(monkeypatch, lambda c, k: "stream")
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is not None
    assert len(calls) == PLAIN_TURN_POSTS, len(calls)


def test_prepinned_source_never_walks_the_ladder(monkeypatch):
    """T-11 for the tool-call shape: once the rung is known, a turn is exactly
    one POST. A turn that then executes a tool dispatch issues the follow-up
    request as a *second* turn (TOOL_CALL_TURN_POSTS = one per turn), so the
    count stays linear in turns rather than turns x rungs."""
    src = _fresh_source()
    _manifest.pin(src, st.MODEL, 1)  # learned in an earlier session
    monkeypatch.setattr(st, "_METRICS", None, raising=False)
    posted = []

    def behavior(calls, kwargs):
        posted.append(kwargs)
        return "stream"

    calls = _client(monkeypatch, behavior)
    assert nx.open_stream([{"role": "user", "content": "hi"}]) is not None
    assert len(calls) == PLAIN_TURN_POSTS
    assert posted[0]["tools"] is None, "pinned no-tools rung must omit tools"

