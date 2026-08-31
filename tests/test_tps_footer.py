#!/usr/bin/env python3
"""tok/s footer correctness: the decode window, not wall-clock.

The old footer divided the server's completion_tokens by (wall - t_first),
where wall was taken AFTER the final Rich render, usage computation, and
teardown — and t_first sat after the prefill head. On a turn whose prompt
just grew by a freshly-read file the head dwarfed the decode (tok/s read
absurdly low); on short tool-call turns the tail dwarfed it from the other
side (tok/s swung high). Both directions of the reported "massive jumps".

Fix under test here, in two halves:
  1. StreamParser.gen_window() — first→last token-bearing delta span (+
     the usage chunk when it lands after the last delta), with a trust
     floor: below _TPS_TRUST_WINDOW a whole generation can arrive in one
     or two SSE reads, so the figure is pacing-free guessing and renders
     with a `~` marker instead of five decimals of false precision.
  2. _print_stats_footer() — uses that window when a parser is supplied,
     keeps the server-measured llama.cpp timings branch bare (no `~`), and
     still renders when called the old way (tests/legacy callers).
"""
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest  # noqa: E402  (env setup must precede imports)
from rich.console import Console  # noqa: E402


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


# --- fakes ---------------------------------------------------------------------

class _Fn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, index=0, name="read_file", arguments=""):
        self.index = index
        self.id = "call_1"
        self.function = _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = None
        self.output_tokens_details = None


class _Chunk:
    """Bare SSE chunk: .usage, .choices, .model_extra (timings live there)."""

    def __init__(self, delta=None, usage=None, finish_reason=None, timings=None):
        self.usage = usage
        self.choices = [] if delta is None and finish_reason is None \
            else [_Choice(delta, finish_reason)]
        self.model_extra = {"timings": timings} if timings else {}


class _Clock:
    """Deterministic stand-in for time.time() in nx._stream_parse."""

    def __init__(self, ticks):
        self.ticks = list(ticks)

    def time(self):
        return self.ticks.pop(0) if self.ticks else float("inf")


def _new_parser():
    from nx._stream_parse import StreamParser
    return StreamParser(Console(file=open(os.devnull, "w")), 0.0)


# --- 1. gen_window() math --------------------------------------------------------

def test_gen_window_spans_first_to_last_token_not_wall():
    p = _new_parser()
    p.t_first, p.t_last, p.t_usage = 2.1, 5.6, 5.62
    window, trusted = p.gen_window(elapsed_total=9.9)
    assert abs(window - 3.52) < 1e-9         # usage−first (5.62−2.1), NOT 9.9−2.1
    assert trusted is True                   # well above the 0.5s floor


def test_gen_window_extends_to_late_usage_chunk():
    p = _new_parser()
    p.t_first, p.t_last, p.t_usage = 2.1, 3.0, 5.6
    window, _ = p.gen_window(elapsed_total=9.9)
    assert abs(window - 3.5) < 1e-6          # usage − first, when later than last


def test_gen_window_ignores_early_usage_chunk():
    p = _new_parser()
    p.t_first, p.t_last, p.t_usage = 2.1, 5.6, 2.2
    window, _ = p.gen_window(elapsed_total=9.9)
    assert abs(window - 3.5) < 1e-9          # usage before last delta changes nothing


def test_gen_window_no_usage_never_zero_from_span_alone():
    p = _new_parser()
    p.t_first, p.t_last = 2.1, 5.6
    window, _ = p.gen_window(elapsed_total=99.0)
    assert abs(window - 3.5) < 1e-9          # plain span, unaffected by wall


def test_gen_window_no_tokens_falls_back_to_wall_untrusted():
    p = _new_parser()  # nothing ever arrived (empty turn)
    window, trusted = p.gen_window(elapsed_total=4.0)
    assert abs(window - 4.0) < 1e-9
    assert trusted is False


def test_gen_window_below_floor_is_untrusted():
    from nx._stream_parse import _TPS_TRUST_WINDOW
    p = _new_parser()
    p.t_first, p.t_last = 0.0, _TPS_TRUST_WINDOW - 0.01
    _, trusted = p.gen_window(elapsed_total=9.9)
    assert trusted is False


def test_gen_window_burst_span_falls_back_to_wall_not_quantization():
    # GLM-5.3-flash / vLLM on a short tool-call turn: the whole generation
    # lands in two SSE reads 0.0004s apart. The old code divided 22 tok by
    # that span and printed ~51008.7 tok/s — arrival quantization, not
    # pacing. Below the _TPS_MIN_SPAN floor the window gives way to the
    # end-to-end average (22 / 0.6 ≈ 36.7), a true lower bound.
    from nx._stream_parse import _TPS_MIN_SPAN
    first = 0.5
    p = _new_parser()
    p.t_first = first
    p.t_last = first + 0.0004
    window, trusted = p.gen_window(elapsed_total=0.6)
    assert trusted is False
    assert abs(window - 0.6) < 1e-9
    from nx._turn import _window_tps
    assert _window_tps(22, window) == pytest.approx(22 / 0.6, rel=1e-3)
    assert _TPS_MIN_SPAN < 0.05  # floor is sub-trust-window by design


def test_gen_window_burst_span_keeps_window_when_longer_than_wall():
    # Defensive mirror: if elapsed is somehow tiny (clock edge), never let
    # the fallback shrink below the span we did observe.
    from nx._stream_parse import _TPS_MIN_SPAN
    p = _new_parser()
    p.t_first = 0.5
    p.t_last = 0.5 + (_TPS_MIN_SPAN / 2)
    window, trusted = p.gen_window(elapsed_total=0.01)
    assert trusted is False
    assert window == pytest.approx(_TPS_MIN_SPAN / 2)


def test_gen_window_just_above_min_span_is_kept():
    # Real multi-read pacing just above the quantization floor is untouched.
    from nx._stream_parse import _TPS_MIN_SPAN
    p = _new_parser()
    p.t_first = 0.5
    p.t_last = 0.5 + _TPS_MIN_SPAN + 0.001
    window, trusted = p.gen_window(elapsed_total=9.9)
    assert trusted is False                    # still under the 0.5s trust floor
    assert window == pytest.approx(_TPS_MIN_SPAN + 0.001)  # but kept as the span


def test_gen_window_zero_span_single_event():
    p = _new_parser()
    p.t_first = p.t_last = 3.3  # whole generation in one SSE event
    window, trusted = p.gen_window(elapsed_total=9.9)
    assert window == 0.0
    assert trusted is False


# --- 2. end-to-end arrival capture through process() -----------------------------

def test_process_records_token_arrivals_relative_to_t0(monkeypatch):
    import nx._stream_parse as spp
    p = _new_parser()
    p._t0 = 1000.0
    # show_tool_status() also reads the clock (throttle check) on each
    # tool-call chunk, so meaningful ticks are interleaved with duplicates
    # that absorb those extra reads.
    monkeypatch.setattr(spp, "time", _Clock([1000.2, 1000.5, 1000.5, 1000.9, 1000.9, 1001.0]))
    assert p.process(_Chunk(usage=_Usage(500, 0))) is False       # usage first, no tokens
    assert p.t_usage == pytest.approx(0.2)
    assert p.process(_Chunk(delta=_Delta(tool_calls=[_TC(arguments="{\"pa")]))) is False
    assert p.process(_Chunk(delta=_Delta(tool_calls=[_TC(arguments="th\":1}")]),
                            finish_reason="tool_calls")) is False
    assert p.t_first == pytest.approx(0.5) and p.t_last == pytest.approx(0.9)
    window, _ = p.gen_window(elapsed_total=99.0)
    assert window == pytest.approx(0.4)


# --- 3. footer rendering ----------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _footer_out(mod, capsys, *args, **kwargs):
    from nx._turn import _print_stats_footer
    mod._state._LAST_CACHE_HIT = None
    _print_stats_footer(*args, **kwargs)
    return _ANSI.sub("", capsys.readouterr().out)


class _ParserStub:
    """Just the gen_window() surface the footer consumes."""

    def __init__(self, window, trusted):
        self._w, self._t = window, trusted

    def gen_window(self, elapsed_total):
        return self._w, self._t


def test_footer_trusted_window_renders_bare_tps(nx_fresh, capsys):
    out = _footer_out(nx_fresh, capsys, _Usage(500, 105), None, 0, 0.1, 6.0, "ok", {},
                      parser=_ParserStub(3.5, True))
    assert " 30.0 tok/s" in out   # 105 / 3.5 — not 105/5.9 wall ≈ 17.8
    assert "~" not in out


def test_footer_untrusted_window_marks_tps_approximate(nx_fresh, capsys):
    out = _footer_out(nx_fresh, capsys, _Usage(500, 40), None, 0, 0.1, 6.0, "ok", {},
                      parser=_ParserStub(0.05, False))
    assert "~800.0 tok/s" in out  # 40 / 0.05 — huge AND visibly untrusted


def test_footer_burst_turn_renders_wall_average_not_quantization_artifact(nx_fresh, capsys):
    # The reported glm-5.3-flash shape, end to end through the footer:
    # 22 completion tokens whose whole arrival span is 0.0004s across a
    # 0.6s turn. The footer must print the wall-time average (~36.7),
    # never the ~51008.7 artifact.
    from nx._stream_parse import StreamParser
    p = StreamParser(Console(file=open(os.devnull, "w")), 0.0)
    p.t_first = 0.5
    p.t_last = 0.5004
    out = _footer_out(nx_fresh, capsys, _Usage(24912, 22), None, 0, 0.5, 0.6, "ok", {},
                      parser=p)
    assert "~" in out and "tok/s" in out
    assert "51008" not in out and "~ 36.7 tok/s" in out


def test_footer_trusted_tps_tucks_into_padded_slot(nx_fresh, capsys):
    # the ~ marker occupies the %5.1f padding slot, so column width is stable:
    # a trusted 800.0 renders with the leading space the ~ replaced
    out = _footer_out(nx_fresh, capsys, _Usage(500, 40), None, 0, 0.1, 6.0, "ok", {},
                      parser=_ParserStub(0.05, True))
    assert " 800.0 tok/s" in out


def test_footer_window_wider_than_wall_impossible_case(nx_fresh, capsys):
    # defensive: a stubbed window can't exceed elapsed here, but the math must
    # not go negative/ZeroDivision if a caller ever passes one that does
    out = _footer_out(nx_fresh, capsys, _Usage(500, 30), None, 0, 0.1, 1.0, "ok", {},
                      parser=_ParserStub(0.5, True))
    assert " 60.0 tok/s" in out


def test_footer_legacy_call_still_renders(nx_fresh, capsys):
    # no parser kwarg: pre-existing callers (and old sessions' tests) keep working
    out = _footer_out(nx_fresh, capsys, _Usage(500, 30), None, 0, 0.1, 6.0, "ok", {})
    assert "tok/s" in out and "~" in out   # unparsered estimate is untrusted by definition


def test_footer_llama_timings_stay_bare(nx_fresh, capsys):
    timings = {"prompt_n": 500, "predicted_n": 120, "predicted_per_second": 48.2,
               "prompt_ms": 1200}
    out = _footer_out(nx_fresh, capsys, None, timings, 0, 0.1, 6.0, "ok", {})
    assert " 48.2 tok/s" in out   # server-measured: no ~ marker
    assert "~" not in out         # (also proves timings didn't divide by a window)


def test_footer_llama_timings_zero_tps_safe(nx_fresh, capsys):
    timings = {"prompt_n": 500, "predicted_n": 120, "predicted_per_second": 0,
               "prompt_ms": 0}
    out = _footer_out(nx_fresh, capsys, None, timings, 0, 0.2, 6.0, "ok", {})
    assert "tok/s" in out         # 1/0 guard: no ZeroDivisionError


def test_footer_char_fallback_uses_window(nx_fresh, capsys):
    # no usage, no timings: 4 chars/token estimate over the decode window
    out = _footer_out(nx_fresh, capsys, None, None, 1600, 0.1, 6.0, "x" * 1600, {},
                      parser=_ParserStub(4.0, True))
    assert " 100.0 tok/s" in out  # 1600 chars ≈ 400 tok / 4.0s, not / 5.9 wall


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
