"""Prompt-cache visibility: footer `cached: N%` + /ctx per-turn cache column.

Regression signal for append-only discipline: a busted prompt prefix (memory_write
cache-key invalidation, /undo-style message edits) collapses the cache-hit
fraction to ~0, which these surfaces make visible instead of silent.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest

import nx  # reloaded by nx_fresh; fixture yields the same module object


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    import nx
    importlib.reload(nx)
    yield nx
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL", "NX_API_KEY"):
        os.environ.pop(k, None)
    nx._state._CTX_PINNED = []


class _PTD:
    def __init__(self, cached):
        self.cached_tokens = cached


class _Usage:
    """OpenAI-style usage object with the cache detail broken out."""

    def __init__(self, prompt, cached, completion=10):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _PTD(cached)
        self.output_tokens_details = None


# --- footer -------------------------------------------------------------------

def test_footer_omits_cached_when_no_cache_numbers(nx_fresh, capsys):
    from nx._turn import _print_stats_footer
    nx._state._LAST_CACHE_HIT = None
    _print_stats_footer(_Usage(500, 0), None, 0, 0.1, 1.0, "hi", {})
    out = capsys.readouterr().out
    assert "cached" not in out


def test_footer_shows_cached_percent(nx_fresh, capsys):
    from nx._turn import _print_stats_footer
    nx._state._LAST_CACHE_HIT = 0.78
    _print_stats_footer(_Usage(500, 390), None, 0, 0.1, 1.0, "hi", {})
    out = capsys.readouterr().out
    # ANSI codes sit between the number and the word — check both parts
    assert "78%" in out and "cached" in out


def test_capture_last_cache_hit_usage_path(nx_fresh):
    from nx._turn import _capture_last_prompt_tokens
    nx._state._LAST_CACHE_HIT = None
    nx._state._LAST_PROMPT_TOKENS = 0
    _capture_last_prompt_tokens(_Usage(1000, 800), None)
    assert nx._state._LAST_PROMPT_TOKENS == 1000
    assert nx._state._LAST_CACHE_HIT == pytest.approx(0.8)


def test_capture_last_cache_hit_timings_path(nx_fresh):
    """llama.cpp: no usage object, cache_n on the timings dict instead."""
    from nx._turn import _capture_last_prompt_tokens
    nx._state._LAST_CACHE_HIT = None
    nx._state._LAST_PROMPT_TOKENS = 0
    _capture_last_prompt_tokens(None, {"prompt_n": 2000, "cache_n": 1500,
                                       "predicted_n": 40})
    assert nx._state._LAST_PROMPT_TOKENS == 2000
    assert nx._state._LAST_CACHE_HIT == pytest.approx(0.75)


def test_capture_last_cache_hit_zero_when_uncached(nx_fresh):
    """cache_n absent/0 → 0.0, not None — a busted prefix is a real signal."""
    from nx._turn import _capture_last_prompt_tokens
    nx._state._LAST_CACHE_HIT = 0.9
    _capture_last_prompt_tokens(_Usage(1000, 0), None)
    assert nx._state._LAST_CACHE_HIT == 0.0


def test_capture_ignores_zero_prompt(nx_fresh):
    from nx._turn import _capture_last_prompt_tokens
    nx._state._LAST_CACHE_HIT = 0.5
    nx._state._LAST_PROMPT_TOKENS = 123
    _capture_last_prompt_tokens(_Usage(0, 0), None)
    assert nx._state._LAST_CACHE_HIT == 0.5
    assert nx._state._LAST_PROMPT_TOKENS == 123


def test_cache_hit_field_colors(nx_fresh):
    from nx._turn import _cache_hit_field
    assert _cache_hit_field(None) == ""
    assert "cached" in _cache_hit_field(0.9)
    assert "cached" in _cache_hit_field(0.5)
    assert "cached" in _cache_hit_field(0.1)


# --- /ctx cache column --------------------------------------------------------

def _msgs(n=6):
    msgs = [{"role": "system", "content": "sys prompt"}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"msg {i}: " + "x" * (200 if i % 2 else 40)})
    return msgs


def test_cache_hits_padded_and_truncated(nx_fresh):
    from nx._ctxreport import _cache_hits_by_turn
    nx._state._METRICS = nx._metrics._Metrics("s", model="m")
    for cached, prompt in ((0, 100), (900, 1000)):
        nx._state._METRICS.add({"input_tokens": prompt - cached,
                                "output_tokens": 1,
                                "cache_read_tokens": cached,
                                "reasoning_tokens": 0})
    # 2 calls, 7 messages → left-padded with None
    hits = _cache_hits_by_turn(7)
    assert len(hits) == 7
    assert hits[:5] == [None] * 5
    assert hits[5] == 0.0
    assert hits[6] == pytest.approx(0.9)
    # 4 calls, 2 messages → right-truncated (keeps the latest)
    nx._state._METRICS.add({"input_tokens": 0, "output_tokens": 1,
                            "cache_read_tokens": 0, "reasoning_tokens": 0})
    nx._state._METRICS.add({"input_tokens": 0, "output_tokens": 1,
                            "cache_read_tokens": 100, "reasoning_tokens": 0})
    hits = _cache_hits_by_turn(2)
    assert len(hits) == 2
    assert hits[-1] == pytest.approx(1.0)


def test_cache_hits_empty_without_metrics(nx_fresh):
    from nx._ctxreport import _cache_hits_by_turn
    nx._state._METRICS = None
    assert _cache_hits_by_turn(5) == []


def test_report_includes_cache_and_cmd_renders(nx_fresh, capsys):
    from nx._ctxreport import _messages_token_report
    from nx._repl_settings_cmds import _cmd_ctx
    from nx._repl_state import _ReplState
    nx._state._METRICS = nx._metrics._Metrics("s", model="m")
    nx._state._METRICS.add({"input_tokens": 20, "output_tokens": 5,
                            "cache_read_tokens": 80, "reasoning_tokens": 0})
    msgs = _msgs()
    rep = _messages_token_report(msgs)
    assert "cache" in rep and len(rep["cache"]) == len(msgs)
    assert rep["cache"][-1] == pytest.approx(0.8)
    state = _ReplState(msgs, "t", False, [])
    _cmd_ctx("/ctx", state)
    out = capsys.readouterr().out
    assert "cache = that turn's" in out
    assert " 80%" in out
    assert " — " in out or "—" in out  # padded turns render as em-dash


def test_metrics_ring_buffer_capped(nx_fresh):
    from nx._metrics import _TURN_RING_MAX
    nx._state._METRICS = nx._metrics._Metrics("s", model="m")
    for _ in range(_TURN_RING_MAX + 10):
        nx._state._METRICS.add({"input_tokens": 10, "output_tokens": 1,
                                "cache_read_tokens": 0, "reasoning_tokens": 0})
    assert len(nx._state._METRICS.turns) == _TURN_RING_MAX
    assert nx._state._METRICS.turns[-1]["prompt_tokens"] == 10
