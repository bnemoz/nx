"""Reasoning token budget (NX_REASONING_BUDGET / /budget): attach shape via
Source.extra_request_kwargs(), precedence vs a source's own EXTRA_BODY, env
resolution, and the /budget handler's parsing. No live server needed."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_REASONING_BUDGET", "NX_SOURCES", "NX_BASE_URL", "NX_MODEL",
          "NX_API_KEY", "NX_ACTIVE"):
    os.environ.pop(k, None)

import pytest


def _fresh_source(**kw):
    from nx import _sources
    return _sources.Source("t", "http://127.0.0.1:9/v1", "sk-test",
                           model="test-model", **kw)


@pytest.fixture(autouse=True)
def _restore_budget_state():
    """Never leak a budget into later test files (module state is global)."""
    yield
    import nx
    nx.REASONING_BUDGET_TOKENS = None
    os.environ.pop("NX_REASONING_BUDGET", None)


# --- attach shape ------------------------------------------------------------

def test_default_off_sends_no_body():
    import nx
    assert nx.REASONING_BUDGET_TOKENS is None
    assert _fresh_source().extra_request_kwargs() == {}


def test_budget_attaches_nested_reasoning_object():
    import nx
    nx.REASONING_BUDGET_TOKENS = 4096
    kw = _fresh_source().extra_request_kwargs()
    assert kw == {"extra_body": {"reasoning": {"max_tokens": 4096}}}


def test_zero_and_negative_budget_are_ignored():
    import nx
    for bad in (0, -5):
        nx.REASONING_BUDGET_TOKENS = bad
        assert _fresh_source().extra_request_kwargs() == {}


# --- precedence vs the source's own EXTRA_BODY --------------------------------

def test_source_extra_body_reasoning_wins_over_global_budget():
    import nx
    nx.REASONING_BUDGET_TOKENS = 4096
    src = _fresh_source(extra_body={"reasoning": {"max_tokens": 1024}})
    assert src.extra_request_kwargs() == \
        {"extra_body": {"reasoning": {"max_tokens": 1024}}}


def test_budget_merges_alongside_other_extra_body_keys():
    import nx
    nx.REASONING_BUDGET_TOKENS = 2048
    src = _fresh_source(extra_body={"provider": {"order": ["a"]}})
    assert src.extra_request_kwargs() == {
        "extra_body": {"provider": {"order": ["a"]},
                       "reasoning": {"max_tokens": 2048}}}


# --- env resolution -----------------------------------------------------------

def test_env_var_resolves_on_reload():
    os.environ["NX_REASONING_BUDGET"] = "2048"
    try:
        importlib.reload(nx := importlib.import_module("nx"))
        assert nx.REASONING_BUDGET_TOKENS == 2048
    finally:
        os.environ.pop("NX_REASONING_BUDGET", None)
        importlib.reload(nx)


def test_garbage_env_var_falls_back_to_off():
    os.environ["NX_REASONING_BUDGET"] = "not-a-number"
    try:
        importlib.reload(nx := importlib.import_module("nx"))
        assert nx.REASONING_BUDGET_TOKENS is None
    finally:
        os.environ.pop("NX_REASONING_BUDGET", None)
        importlib.reload(nx)


# --- /budget handler ----------------------------------------------------------

def _run(arg):
    from nx._repl_settings_cmds import _cmd_budget
    _cmd_budget(arg, None)  # state unused by the handler


def test_budget_cmd_set_and_clear(capsys):
    import nx
    _run("/budget 4096")
    assert nx.REASONING_BUDGET_TOKENS == 4096
    assert os.environ["NX_REASONING_BUDGET"] == "4096"
    _run("/budget off")
    assert nx.REASONING_BUDGET_TOKENS is None
    assert "NX_REASONING_BUDGET" not in os.environ


def test_budget_cmd_rejects_garbage_and_negatives(capsys):
    import nx
    _run("/budget 4096")
    for bad, needle in (("/budget abc", "usage"),
                        ("/budget -5", "out of range")):
        _run(bad)
        assert needle in capsys.readouterr().out.lower()
        assert nx.REASONING_BUDGET_TOKENS == 4096, bad


def test_budget_cmd_show_lists_current_value(capsys):
    import nx
    nx.REASONING_BUDGET_TOKENS = 512
    _run("/budget")
    out = capsys.readouterr().out
    assert "512" in out
