"""2.5 cost correctness: side-call accounting, estimate provenance,
per-model pricing, reasoning-token dual read, by_source persistence.

Each test pins one gap the ROADMAP audit found in the accounting pipeline:
calls that were billed by the provider but invisible to /cost, estimates that
masqueraded as measured numbers, a reasoning split that read only one of the
two field shapes servers actually send, a scoreboard that lost its per-source
attribution on resume, and a price key that couldn't distinguish two models
hosted behind one source.

Tier A: nx is imported ONCE and never reloaded. The old fixture's
importlib.reload(nx) existed to rebuild the model-source wiring and the
pricing/env knobs from this fixture's env; env is read live (pricing keys in
nx._cost are read with os.environ.get at call time, and the conftest rewinds
NX_* env universally), and the source wiring is rebuilt without a reload via
_discover_sources()/switch_source() — resolve_model() reads NX_MODEL
("test-model"), so no network round-trip happens. Bare nx/_state mutations are
monkeypatch.setattr'd (their teardown used to ride the module-dict rewind);
nx_state_reset covers the rest of the hidden state.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest

import nx  # noqa: E402


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = "/dev/null"
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    # Tier A: no importlib.reload(nx). Rebuild only the model-source wiring
    # the reload used to redo: re-discover sources and switch to the first
    # one (binds st.client + st.MODEL from the env set above). 127.0.0.1:9
    # is never contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    yield nx, tmp_path


def _usage_ns(prompt=100, cached=None, completion=50, reasoning=None,
              details="output_tokens_details"):
    """A usage object with only the fields a server might send."""
    ns = types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    if details == "output_tokens_details":
        if cached is not None:
            ns.prompt_tokens_details = types.SimpleNamespace(cached_tokens=cached)
        if reasoning is not None:
            ns.output_tokens_details = types.SimpleNamespace(reasoning_tokens=reasoning)
    else:
        if cached is not None:
            ns.prompt_tokens_details = types.SimpleNamespace(cached_tokens=cached)
        if reasoning is not None:
            ns.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=reasoning)
    return ns


# --- reasoning_tokens dual read -------------------------------------------------

def test_reasoning_read_from_output_tokens_details(nx_fresh):
    from nx._metrics import _normalize_usage
    d = _normalize_usage(_usage_ns(reasoning=30), None)
    assert d["reasoning_tokens"] == 30
    assert d["output_tokens"] == 20  # completion 50 - reasoning 30


def test_reasoning_read_from_completion_tokens_details(nx_fresh):
    """OpenAI's own field shape — pre-0.43.1 this read only
    output_tokens_details, so the column sat at 0 there."""
    from nx._metrics import _normalize_usage
    d = _normalize_usage(_usage_ns(reasoning=30, details="completion_tokens_details"), None)
    assert d["reasoning_tokens"] == 30
    assert d["output_tokens"] == 20


def test_reasoning_zero_when_neither_shape_present(nx_fresh):
    from nx._metrics import _normalize_usage
    d = _normalize_usage(_usage_ns(), None)
    assert d["reasoning_tokens"] == 0


# --- estimate provenance ---------------------------------------------------------

def test_estimate_flagged_in_delta(nx_fresh):
    from nx._metrics import _normalize_usage
    d = _normalize_usage(None, None, fallback_chars=400, fallback_prompt_chars=800)
    assert d["estimated"] == 1
    from nx import _tokens as _tk
    assert d["output_tokens"] == _tk.estimate_tokens_from_chars(400)
    assert d["input_tokens"] == _tk.estimate_tokens_from_chars(800)


def test_measured_not_flagged(nx_fresh):
    from nx._metrics import _normalize_usage
    assert _normalize_usage(_usage_ns(), None)["estimated"] == 0


def test_metrics_track_estimated_calls(nx_fresh, monkeypatch):
    from nx._metrics import _Metrics
    nx, _tmp = nx_fresh
    monkeypatch.setattr(nx._state, "ACTIVE", type("S", (), {"name": "s"})())
    m = _Metrics(session_id="t", source="s")
    m.add({"input_tokens": 200, "output_tokens": 100, "cache_read_tokens": 0,
           "reasoning_tokens": 0, "estimated": 1})
    m.add({"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0,
           "reasoning_tokens": 0, "estimated": 0})
    assert m.api_calls == 2 and m.estimated_calls == 1
    assert m.estimated_input_tokens == 200 and m.estimated_output_tokens == 100
    assert m.by_source["s"]["estimated_calls"] == 1


def test_cost_report_marks_estimated_calls(nx_fresh, capsys):
    from nx._metrics import _Metrics
    from nx._cost import print_cost_report
    m = _Metrics(session_id="t", source="s")
    m.add({"input_tokens": 200, "output_tokens": 100, "cache_read_tokens": 0,
           "reasoning_tokens": 0, "estimated": 1})
    print_cost_report(m)
    out = capsys.readouterr().out
    assert "estimated" in out


# --- by_source survives resume ----------------------------------------------------

def test_by_source_saved_and_restored(nx_fresh):
    from nx._metrics import _Metrics
    m = _Metrics(session_id="s1", source="local")
    m.by_source = {"local": {"input_tokens": 10, "output_tokens": 2,
                             "cache_read_tokens": 0, "reasoning_tokens": 0,
                             "api_calls": 1, "estimated_calls": 0}}
    meta = m.as_meta()
    assert meta["by_source"] == m.by_source  # saved into session JSON


def _patch_reload_state(monkeypatch, st, saved):
    """Tier-A stand-in for the module-dict rewind around
    _reload_metrics_totals: patch _load_session where that function actually
    looks it up (its own module dict) and pin every nx._state name the call
    writes — _METRICS (which we replace) plus the todos/plan/wip snapshot it
    re-reads from the saved dict."""
    from nx._metrics import _Metrics
    monkeypatch.setattr(nx._repl_state, "_load_session", lambda sid: saved)
    for name in ("_METRICS", "_TODOS", "_PLAN_ACTIVE", "_PLAN_GOAL",
                 "_PLAN_SLUG", "_PLAN_STARTED", "_WIP"):
        monkeypatch.setattr(st, name, getattr(st, name), raising=False)
    monkeypatch.setattr(st, "_METRICS", _Metrics(session_id="sid", source="local"))


def test_resume_reloads_by_source(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    from nx._metrics import _Metrics
    from nx import _repl_state
    m = _Metrics(session_id="sid", source="local")
    m.by_source = {"local": {"input_tokens": 111, "output_tokens": 9,
                             "cache_read_tokens": 0, "reasoning_tokens": 0,
                             "api_calls": 1, "estimated_calls": 0}}
    saved = {"input_tokens": 111, "output_tokens": 9, "cache_read_tokens": 0,
             "reasoning_tokens": 0, "api_calls": 1, "started_at": 1.0,
             "by_source": m.by_source}
    st = nx._state
    _patch_reload_state(monkeypatch, st, saved)
    _repl_state._reload_metrics_totals("sid")
    assert st._METRICS.by_source["local"]["input_tokens"] == 111


def test_resume_without_by_source_keeps_fresh_buckets(nx_fresh, monkeypatch):
    """Older session files have no by_source — must not fabricate one."""
    nx, _tmp = nx_fresh
    from nx import _repl_state
    saved = {"input_tokens": 5, "output_tokens": 1, "cache_read_tokens": 0,
             "reasoning_tokens": 0, "api_calls": 1}
    st = nx._state
    _patch_reload_state(monkeypatch, st, saved)
    _repl_state._reload_metrics_totals("sid")
    assert st._METRICS.by_source == {}


# --- per-model pricing ------------------------------------------------------------

def test_per_model_price_key_wins(nx_fresh):
    from nx._cost import _price_for
    os.environ["NX_PRICE_HOSTED_INPUT"] = "1.0"
    os.environ["NX_PRICE_HOSTED_BIG_MODEL_INPUT"] = "5.0"
    os.environ["NX_PRICE_HOSTED_BIG_MODEL_OUTPUT"] = "20.0"
    try:
        assert _price_for("hosted", "input", "big-model") == 5.0
        assert _price_for("hosted", "output", "big-model") == 20.0
        # a model without its own key falls back to the source-wide price
        assert _price_for("hosted", "input", "small-model") == 1.0
        assert _price_for("hosted", "input", None) == 1.0
    finally:
        for k in ("NX_PRICE_HOSTED_INPUT", "NX_PRICE_HOSTED_OUTPUT",
                  "NX_PRICE_HOSTED_BIG_MODEL_INPUT", "NX_PRICE_HOSTED_BIG_MODEL_OUTPUT"):
            os.environ.pop(k, None)


def test_model_slug_folds_punctuation(nx_fresh):
    from nx._cost import _price_for
    os.environ["NX_PRICE_HOSTED_QWEN3_32B_INPUT"] = "2.0"
    os.environ["NX_PRICE_HOSTED_QWEN_QWEN3_32B_INPUT"] = "4.0"
    try:
        # the slug folds every non-alphanumeric to _ (predictable, documented):
        # "Qwen3-32B" -> QWEN3_32B, "Qwen/Qwen3-32B" -> QWEN_QWEN3_32B
        assert _price_for("hosted", "input", "Qwen3-32B") == 2.0
        assert _price_for("hosted", "input", "qwen3_32B") == 2.0
        assert _price_for("hosted", "input", "Qwen/Qwen3-32B") == 4.0
    finally:
        os.environ.pop("NX_PRICE_HOSTED_QWEN3_32B_INPUT", None)
        os.environ.pop("NX_PRICE_HOSTED_QWEN_QWEN3_32B_INPUT", None)


def test_source_cost_takes_model_key(nx_fresh):
    from nx._cost import _source_cost
    os.environ["NX_PRICE_HOSTED_INPUT"] = "1.0"
    os.environ["NX_PRICE_HOSTED_OUTPUT"] = "2.0"
    os.environ["NX_PRICE_HOSTED_BIG_INPUT"] = "3.0"
    os.environ["NX_PRICE_HOSTED_BIG_OUTPUT"] = "6.0"
    try:
        bucket = {"input_tokens": 1_000_000, "output_tokens": 100_000,
                  "cache_read_tokens": 0, "reasoning_tokens": 0}
        assert abs(_source_cost(bucket, "hosted", "big") - (3.0 + 0.6)) < 1e-9
        assert abs(_source_cost(bucket, "hosted") - (1.0 + 0.2)) < 1e-9
    finally:
        for k in list(os.environ):
            if k.startswith("NX_PRICE_HOSTED"):
                os.environ.pop(k, None)


# --- side-calls recorded -----------------------------------------------------------

def test_risk_call_recorded(nx_fresh, monkeypatch):
    nx, tmp = nx_fresh
    import nx._risk as risk_mod
    recorded = []

    class Resp:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(
            content='{"level":"low","reason":"r"}'))]
        usage = _usage_ns(prompt=500, completion=20)

        def model_dump(self):
            return {}

    monkeypatch.setattr(risk_mod, "_log_event", lambda *a, **k: None)
    # _assess_risk reads st.client / st.MODEL (the _state module), so the
    # fake client must land there, not on nx._risk
    monkeypatch.setattr(nx._state, "client", types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **kw: Resp()))))
    monkeypatch.setattr(nx._state, "MODEL", "test-model")
    # _record_api_call is imported inside _assess_risk at call time, so the
    # patch must land on the module it is imported FROM
    import nx._metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "_record_api_call",
                        lambda resp, **kw: recorded.append(resp.usage))
    risk_mod._assess_risk("run: true")
    assert recorded and recorded[0].prompt_tokens == 500


def test_memory_optimize_still_recorded(nx_fresh, monkeypatch):
    """Guard the existing accounting (the audit found 5 recorded sites) —
    _record_api_call must not have regressed them."""
    from nx._metrics import _record_api_call
    resp = types.SimpleNamespace(usage=_usage_ns(prompt=10, completion=5))
    seen = []
    import nx._metrics as mm
    orig = mm._record_usage
    monkeypatch.setattr(mm, "_record_usage",
                        lambda u, **kw: seen.append(u) if u else None)
    _record_api_call(resp)
    assert seen and seen[0].prompt_tokens == 10
