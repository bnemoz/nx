"""F10: /cost — per-source usage bucketing + price-driven $ scoreboard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.chdir(tmp_path)
    import nx
    # Tier A: no importlib.reload. The conftest imports nx once under a scrubbed
    # env, so instead of a reload rebuild just the model-source wiring from the
    # env set above: re-discover sources and switch to the first one (binds
    # st.client + st.MODEL). resolve_model() reads NX_MODEL ("test-model"), so
    # no network round-trip happens; 127.0.0.1:9 is never contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # Tests below swap nx._state.ACTIVE for fake source objects; the conftest
    # module-dict rewind used to clean that up, so restore it explicitly here.
    _prev_active = nx._state.ACTIVE
    try:
        yield nx, tmp_path
    finally:
        nx._state.ACTIVE = _prev_active


def _metrics(nx, active_name):
    from nx._metrics import _Metrics
    m = _Metrics(session_id="t", source=active_name)
    nx._state.ACTIVE = type("S", (), {"name": active_name})()
    return m


def _delta(i=0, o=0, c=0, r=0):
    return {"input_tokens": i, "output_tokens": o, "cache_read_tokens": c, "reasoning_tokens": r}


def test_by_source_bucketing_on_source_switch(nx_fresh):
    nx, tmp = nx_fresh
    m = _metrics(nx, "local")
    m.add(_delta(i=1000, o=100))
    # switch to zai mid-session
    nx._state.ACTIVE = type("S", (), {"name": "zai"})()
    m.add(_delta(i=2000, o=200))
    assert m.by_source["local"]["input_tokens"] == 1000
    assert m.by_source["zai"]["input_tokens"] == 2000
    assert m.api_calls == 2
    # grand totals unchanged
    assert m.input_tokens == 3000 and m.output_tokens == 300


def test_cost_computation_with_prices(nx_fresh):
    nx, tmp = nx_fresh
    os.environ["NX_PRICE_ZAI_INPUT"] = "0.5"    # $/1M
    os.environ["NX_PRICE_ZAI_OUTPUT"] = "2.0"
    os.environ["NX_PRICE_ZAI_CACHE"] = "0.1"
    from nx._cost import _source_cost
    bucket = {"input_tokens": 1_000_000, "output_tokens": 100_000,
              "cache_read_tokens": 500_000, "reasoning_tokens": 0, "api_calls": 3}
    # 1M*0.5 + 0.5M*0.1 + 0.1M*2.0 = 0.5 + 0.05 + 0.2
    assert abs(_source_cost(bucket, "zai") - 0.75) < 1e-9
    # unpriced source → None
    assert _source_cost(bucket, "local") is None


def test_cost_cache_defaults_to_input_price(nx_fresh):
    nx, tmp = nx_fresh
    os.environ["NX_PRICE_LOCAL_INPUT"] = "1.0"
    os.environ["NX_PRICE_LOCAL_OUTPUT"] = "1.0"
    from nx._cost import _source_cost
    bucket = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 1_000_000,
              "reasoning_tokens": 0, "api_calls": 1}
    assert abs(_source_cost(bucket, "local") - 1.0) < 1e-9


def test_print_cost_report_renders(nx_fresh, capsys):
    nx, tmp = nx_fresh
    os.environ["NX_PRICE_ZAI_INPUT"] = "0.5"
    os.environ["NX_PRICE_ZAI_OUTPUT"] = "2.0"
    m = _metrics(nx, "local")
    m.add(_delta(i=4000, o=500))
    nx._state.ACTIVE = type("S", (), {"name": "zai"})()
    m.add(_delta(i=9000, o=1500))
    from nx._cost import print_cost_report
    print_cost_report(m)
    out = capsys.readouterr().out
    assert "session usage" in out and "2 model calls" in out
    assert "per source" in out
    assert "local" in out and "zai" in out
    assert "$" in out           # zai priced
    assert "—" in out           # local unpriced
    assert "NX_PRICE_<SOURCE>_INPUT" in out  # hint when partially priced


def test_print_cost_report_empty(nx_fresh, capsys):
    nx, tmp = nx_fresh
    from nx._cost import print_cost_report
    print_cost_report(None)
    assert "no usage recorded" in capsys.readouterr().out
    m = _metrics(nx, "local")
    print_cost_report(m)
    out = capsys.readouterr().out
    assert "no per-source data" in out


def test_cmd_cost_registered(nx_fresh):
    nx, tmp = nx_fresh
    from nx._repl import _REPL_COMMANDS
    from nx._chatbox import SLASH_COMMANDS
    assert "/cost" in _REPL_COMMANDS
    assert any(cmd == "/cost" for cmd, _ in SLASH_COMMANDS)
