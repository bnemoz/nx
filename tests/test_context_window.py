#!/usr/bin/env python3
"""Tests for Source.resolve_context_window() — the max-context probe that
backs the footer's `ctx N/<max>` suffix and the /source listing's `· N ctx`.

Tier A: nx is imported ONCE and never reloaded. The old file's import-time
env scrub + importlib.reload(nx) (Test 8 re-read NX_SOURCES knobs) are
replaced by the autouse `_tier_a` fixture: scrub NX_* env, re-run
_discover_sources()/switch_source() to rebind the model source, and
monkeypatch-pin every piece of nx module state the tests touch (ACTIVE/
client/MODEL/SOURCES/SOURCE_ORDER/PROBE_GENERATION). NX_SKIP_STARTUP_PROBES
is read at call time, so the fixture flips it to "0" AFTER the probe-free
wiring — the over-max_tokens probe then actually fires (tests 3-6b, 7).
urlopen and the fake client classes stand in for the network; everything
stays offline and deterministic. Conftest's per-test env rewind plus
monkeypatch undo replace the old Tier-B module-dict rewind.
"""
import json
import re
import urllib.error
import urllib.request

import pytest

import nx as m
import nx._state as st
from nx._sources import _discover_sources, switch_source  # noqa: F401


@pytest.fixture(autouse=True)
def _tier_a(monkeypatch):
    """Fresh single-'local' source world per test, with nx state pinned."""
    # Pin the module state these tests mutate (undo() restores at teardown —
    # replaces the Tier-B module-dict rewind).
    for attr in ("ACTIVE", "client", "MODEL", "SOURCES", "SOURCE_ORDER",
                 "PROBE_GENERATION", "_NATIVE_TOOL_CALLS_CONFIRMED"):
        monkeypatch.setattr(st, attr, getattr(st, attr))
    # Scrub env so _discover_sources builds a clean single 'local' source
    # (env rewind is universal in conftest; monkeypatch.delenv is belt+braces).
    for k in [k for k in m.os.environ if k.startswith("NX_")]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NX_ENV_FILE", "/dev/null")
    monkeypatch.setenv("NX_SOURCE_LOCAL_BASE_URL", "http://localhost:8080/v1")
    # Wiring runs with conftest's NX_SKIP_STARTUP_PROBES=1 (no network);
    # the probe tests need the flag OFF and it is read at call time.
    _discover_sources()
    assert switch_source(st.SOURCE_ORDER[0]), "fixture source wiring failed"
    monkeypatch.setenv("NX_SKIP_STARTUP_PROBES", "0")


# --- helpers: fake model-list entries -----------------------------------------
class _FakeModel:
    """Stand-in for a GET /v1/models entry. resolve_context_window() reads
    meta.n_ctx / context_length out of the raw dict via model_dump()."""
    def __init__(self, mid, extra=None):
        self.id = mid
        self.created = 0
        self.object = "model"
        self.owned_by = "test"
        self._extra = extra or {}

    def model_dump(self):
        return {"id": self.id, "created": self.created, "object": self.object,
                "owned_by": self.owned_by, **self._extra}


class _FakeModelsPage:
    def __init__(self, data):
        self.data = data


class _FakeChatCompletions:
    """Raises a BadRequestError-shaped exception carrying the context-length
    message, mimicking Together's over-max_tokens 400 response."""
    def __init__(self, msg=None):
        self._msg = msg

    def create(self, **kwargs):
        if self._msg is None:
            # Simulate a provider that accepts the request and would generate
            # — the probe should NOT parse anything from a successful call.
            return object()
        e = RuntimeError(self._msg)
        e.body = json.loads(self._msg.split(" - ", 1)[1]) if " - " in self._msg else None
        raise e


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeModels:
    def __init__(self, page):
        self._page = page

    def list(self, **kwargs):
        return self._page


class _FakeClient:
    def __init__(self, models_page=None, chat=None):
        self.models = _FakeModels(models_page)
        self.chat = chat or _FakeChat(_FakeChatCompletions())


def _make_source(models_page=None, chat=None):
    src = m.Source("test", "http://localhost:8080/v1", "sk-noop", "test-model")
    src.client = _FakeClient(models_page, chat)
    return src


_OVERMAX_ERR = ("Error code: 400 - " + json.dumps({
    "error": {"message": "This model's maximum context length is 262144 tokens, "
              "but the request requires 5000013 tokens."}}))


# --- Test 1: llama.cpp /v1/models carries n_ctx in data[0].meta ---------------
def test_1_llamacpp_n_ctx_from_models_meta():
    page = _FakeModelsPage([_FakeModel("test-model", {"meta": {"n_ctx": 131072}})])
    src = _make_source(models_page=page, chat=_FakeChat(_FakeChatCompletions()))
    n = src.resolve_context_window()
    assert n == 131072, f"expected 131072 from meta.n_ctx, got {n}"


# --- Test 2: /props fallback (models list empty) ----------------------------
def test_2_props_fallback(monkeypatch):
    props = {"default_generation_settings": {"n_ctx": 65536},
             "model_alias": "test-model"}

    class _FakeResp:
        def __init__(self, body):
            self._body = body.encode() if isinstance(body, str) else body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(url, timeout=None):
        if "/props" in str(url):
            return _FakeResp(json.dumps(props))
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    src = _make_source(models_page=_FakeModelsPage([]),
                       chat=_FakeChat(_FakeChatCompletions()))
    n = src.resolve_context_window()
    assert n == 65536, f"expected 65536 from /props, got {n}"


# --- Test 3: Together — over-max_tokens 400 names the limit -----------------
def test_3_over_max_tokens_probe(monkeypatch):
    # For a remote host, /props doesn't exist and models list is empty, so the
    # over-max_tokens 400 is the only path that should succeed. Keep urlopen mocked
    # to fail so /props can't accidentally reach a real local server.
    def _props_refused(url, timeout=None):
        raise urllib.error.URLError("no /props on remote host")

    monkeypatch.setattr(urllib.request, "urlopen", _props_refused)
    src = _make_source(models_page=_FakeModelsPage([]),
                       chat=_FakeChat(_FakeChatCompletions(msg=_OVERMAX_ERR)))
    src._context_window = None  # fresh — don't inherit any cached value
    n = src.resolve_context_window()
    assert n == 262144, f"expected 262144 from overrun probe, got {n}"


class _CountingChat(_FakeChatCompletions):
    def __init__(self, probe_calls):
        super().__init__(msg=_OVERMAX_ERR)
        self._probe_calls = probe_calls
    def create(self, **kwargs):
        self._probe_calls[0] += 1
        return super().create(**kwargs)


# --- Test 4+5: caching — second call doesn't re-probe; force re-probes ------
def test_4_and_5_caching_and_force(monkeypatch):
    probe_calls = [0]

    def _props_refused(url, timeout=None):
        raise urllib.error.URLError("no /props")

    monkeypatch.setattr(urllib.request, "urlopen", _props_refused)
    src = _make_source(models_page=_FakeModelsPage([]),
                       chat=_FakeChat(_CountingChat(probe_calls)))
    src._context_window = None  # ensure fresh
    n1 = src.resolve_context_window()
    n2 = src.resolve_context_window()  # cached
    assert n1 == 262144 and n2 == 262144
    assert probe_calls[0] == 1, f"second call should hit cache, probes={probe_calls[0]}"
    # force=True bypasses the cache
    n3 = src.resolve_context_window(force=True)
    assert n3 == 262144 and probe_calls[0] == 2, "force should re-probe"


class _NoInfoChat(_FakeChatCompletions):
    def create(self, **kwargs):
        raise RuntimeError("some unrelated error")


# --- Test 6: total miss → None (all probes fail, no crash) ------------------
def test_6_total_miss_returns_none(monkeypatch):
    def _refused(*a, **k):
        raise urllib.error.URLError("conn refused")

    monkeypatch.setattr(urllib.request, "urlopen", _refused)
    src = _make_source(models_page=_FakeModelsPage([]),
                       chat=_FakeChat(_NoInfoChat()))
    src._context_window = None
    n = src.resolve_context_window()
    assert n is None, f"expected None on total miss, got {n}"


# --- Test 6b: silently-clamped probes are negatively cached -----------------
def test_6b_silent_clamp_negatively_cached(monkeypatch):
    # The clamping.example overrun probe must actually FIRE here; stub urlopen
    # so the /props leg can't reach a real host.
    def _refused(url, timeout=None):
        raise urllib.error.URLError("conn refused")

    monkeypatch.setattr(urllib.request, "urlopen", _refused)
    silent_probe_calls = [0]

    class _SilentClampChat(_FakeChatCompletions):
        def create(self, **kwargs):
            silent_probe_calls[0] += 1
            return object()  # accepted request: no 400/context length to parse

    src = m.Source("silent-clamp", "https://clamping.example/v1", "sk-noop",
                   "test-model")
    src.client = _FakeClient(_FakeModelsPage([]),
                             _FakeChat(_SilentClampChat()))
    n1 = src.resolve_context_window()
    n2 = src.resolve_context_window()
    assert n1 is None and n2 is None
    assert src._context_window_probe_attempted, "miss should be cached"
    assert silent_probe_calls[0] == 1, \
        "a silently-clamped probe must not run again on later turns"


# --- Test 7: remote source leads with over-max_tokens probe (no /v1/models) --
def test_7_remote_leads_with_overrun_probe(monkeypatch):
    # A remote host (Together et al.) returns an empty /v1/models and has no
    # /props, but the over-max_tokens 400 names the limit. The probe must resolve
    # on the FIRST chat request WITHOUT ever hitting /v1/models (which would cost
    # a multi-second TLS round-trip returning nothing). We assert ordering by
    # giving /v1/models a non-empty page that WOULD be consulted if it ran first
    # (context_length=999999) — if the overrun probe ran first we get 262144, not
    # the models page's 999999.
    _remote_models_calls = [0]

    class _CountingModels(_FakeModels):
        def list(self, **kwargs):
            _remote_models_calls[0] += 1
            return self._page

    # Remote base_url → _is_local() is False → overrun probe leads.
    remote_page = _FakeModelsPage([_FakeModel("test-model",
                                              {"context_length": 999999})])
    src = m.Source("together", "https://api.together.xyz/v1", "sk-noop",
                   "zai-org/GLM-5.3")
    src.client = _FakeClient(_CountingModels(remote_page),
                             _FakeChat(_FakeChatCompletions(msg=_OVERMAX_ERR)))
    src._context_window = None
    n = src.resolve_context_window()
    assert n == 262144, f"remote should resolve via overrun probe, got {n}"
    assert _remote_models_calls[0] == 0, \
        "remote must not hit /v1/models when the overrun probe succeeds"


# --- Test 7b: _is_local() classification ------------------------------------
def test_7b_is_local_classification():
    _local = m.Source("local", "http://localhost:8080/v1", "sk-noop")
    _lan = m.Source("lan", "http://192.168.1.50:8080/v1", "sk-noop")
    _remote1 = m.Source("together", "https://api.together.xyz/v1", "sk-noop")
    _remote2 = m.Source("zai", "https://api.z.ai/api/paas/v4", "sk-noop")
    assert _local._is_local(), "localhost should be local"
    assert _lan._is_local(), "192.168.x should be local"
    assert not _remote1._is_local(), "api.together.xyz should be remote"
    assert not _remote2._is_local(), "api.z.ai should be remote"


# --- Test 8: switch_source invalidates the cached window ---------------------
def test_8_switch_source_invalidates_cache(monkeypatch):
    # Re-discover a two-source world (the fixture pinned st.SOURCES/
    # st.SOURCE_ORDER, so undo restores the fixture's single-source state) and
    # swap; the cached window should be cleared on switch. Replaces the old
    # importlib.reload(nx) env-knob re-read.
    monkeypatch.setenv("NX_SOURCES", "alpha,beta")
    monkeypatch.setenv("NX_SOURCE_ALPHA_BASE_URL", "http://localhost:8081/v1")
    monkeypatch.setenv("NX_SOURCE_BETA_BASE_URL", "http://localhost:8082/v1")
    _discover_sources()
    m.SOURCES["alpha"]._context_window = 999
    m.SOURCES["beta"]._context_window = 888
    m.SOURCES["alpha"]._context_window_probe_attempted = True
    m.SOURCES["beta"]._context_window_probe_attempted = True
    assert m.switch_source("alpha")
    assert m.SOURCES["alpha"]._context_window is None, "switch should clear alpha cache"
    assert not m.SOURCES["alpha"]._context_window_probe_attempted, \
        "switch should clear alpha's negative cache"
    assert m.switch_source("beta")
    assert m.SOURCES["beta"]._context_window is None, "switch should clear beta cache"
    assert not m.SOURCES["beta"]._context_window_probe_attempted, \
        "switch should clear beta's negative cache"


# --- Test 9: _ctx_field() — colored "cur/max ctx" footer field ----------------
def test_9_ctx_field_colorization(monkeypatch):
    _ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def _plain(s):
        return _ANSI.sub("", s)

    # Set up a source with a known 170K window.
    src = _make_source(models_page=_FakeModelsPage([
        _FakeModel(m.MODEL or "x", {"meta": {"n_ctx": 170000}})]),
        chat=_FakeChat(_FakeChatCompletions()))
    src._context_window = None
    m.ACTIVE = src
    src.resolve_context_window()

    # Green: under 30% of 170000 (= 51000). Use 667.
    f = m._ctx_field(667)
    assert m.GREEN in f, "under 30% should be green"
    assert m.CYAN in f, "max should be cyan"
    assert "/170K" in _plain(f), "should contain /170K"
    assert "667" in _plain(f), "should contain current 667"

    # Yellow: 30%–60% of 170000 (51000–102000). Use 70000.
    f = m._ctx_field(70000)
    assert m.YELLOW in f and m.GREEN not in f, "30-60% should be yellow, not green"

    # Red: at/above 60% of 170000 (>= 102000). Use 128000 (75%).
    f = m._ctx_field(128000)
    assert m.RED in f and m.YELLOW not in f and m.GREEN not in f, ">=60% should be red only"

    # Unknown max → no color, no slash, just "NNN ctx".
    def _props_refused3(url, timeout=None):
        raise urllib.error.URLError("no /props")
    monkeypatch.setattr(urllib.request, "urlopen", _props_refused3)
    src2 = _make_source(models_page=_FakeModelsPage([]),
                        chat=_FakeChat(_NoInfoChat()))
    src2._context_window = None
    m.ACTIVE = src2
    f = m._ctx_field(1234)
    assert m.GREEN not in f and m.CYAN not in f and m.RED not in f, "no color when max unknown"
    assert "/" not in _plain(f), "no slash when max unknown"
    assert "1.2K" in _plain(f) and "ctx" in _plain(f), "should still show current + 'ctx'"
