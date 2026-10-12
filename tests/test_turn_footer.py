#!/usr/bin/env python3
"""Per-user-turn footer aggregation: intermediate steps suppress their footer
and record into the accumulator; the flush renders ONE aggregated row."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR"):
    os.environ.pop(k, None)

import pytest  # noqa: E402
import re  # noqa: E402

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = None
        self.output_tokens_details = None


class _ParserStub:
    def __init__(self, window, trusted):
        self._w, self._t = window, trusted

    def gen_window(self, elapsed_total):
        return self._w, self._t


@pytest.fixture()
def nx_fresh(tmp_path):
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    import nx
    importlib.reload(nx)
    yield nx
    for k in ("NX_HOME", "NX_SESSIONS_DIR", "NX_BASE_URL", "NX_MODEL",
              "NX_API_KEY"):
        os.environ.pop(k, None)
    nx._state._CTX_PINNED = []


def _out(mod, capsys, *args, **kwargs):
    from nx._turn import _print_stats_footer
    mod._state._LAST_CACHE_HIT = None
    _print_stats_footer(*args, **kwargs)
    return _ANSI.sub("", capsys.readouterr().out)


def test_intermediate_steps_suppress_and_final_aggregates(nx_fresh, capsys):
    tf = nx_fresh._turn_footer
    tf.begin_user_turn_footer()
    # step 1 (intermediate): records, prints NOTHING
    _out(nx_fresh, capsys, _Usage(500, 40), None, 0, 0.1, 6.0, "", {},
         parser=_ParserStub(0.5, True))
    assert capsys.readouterr().out == ""
    # step 2 (last): also records only — the flush renders the aggregate
    _out(nx_fresh, capsys, _Usage(500, 105), None, 0, 0.1, 3.5,
         "final answer", {}, parser=_ParserStub(3.5, True))
    assert capsys.readouterr().out == ""
    tf.end_user_turn_footer()
    out = _ANSI.sub("", capsys.readouterr().out)
    assert "145 tok" in out               # 40 + 105
    assert "9.5s wall" in out             # 6.0 + 3.5
    assert "~ 15.3 tok/s" in out          # 145 / 9.5; aggregated = untrusted


def test_single_step_flush_rerenders_identical_row(nx_fresh, capsys):
    tf = nx_fresh._turn_footer
    tf.begin_user_turn_footer()
    _out(nx_fresh, capsys, _Usage(500, 105), None, 0, 0.1, 6.0, "ok",
         {}, parser=_ParserStub(3.5, True))
    assert capsys.readouterr().out == ""
    tf.end_user_turn_footer()
    out = _ANSI.sub("", capsys.readouterr().out)
    assert "105 tok" in out and " 30.0 tok/s" in out and "~" not in out


def test_quiet_steps_never_record(nx_fresh, capsys):
    tf = nx_fresh._turn_footer
    tf.begin_user_turn_footer()
    _out(nx_fresh, capsys, _Usage(500, 40), None, 0, 0.1, 6.0, "", {},
         parser=_ParserStub(0.5, True), quiet=True)
    assert tf._acc() is not None and tf._acc().steps == 0
    tf.end_user_turn_footer()
    assert capsys.readouterr().out == ""


def test_mixed_sources_still_sum(nx_fresh, capsys):
    tf = nx_fresh._turn_footer
    timings = {"prompt_n": 500, "predicted_n": 120,
               "predicted_per_second": 48.2, "prompt_ms": 1200}
    tf.begin_user_turn_footer()
    _out(nx_fresh, capsys, None, timings, 0, 0.1, 6.0, "", {})
    _out(nx_fresh, capsys, _Usage(500, 25), None, 0, 0.1, 3.0, "done", {},
         parser=_ParserStub(1.0, True))
    _out(nx_fresh, capsys, _Usage(500, 25), None, 0, 0.1, 3.0, "done",
         {}, parser=_ParserStub(1.0, True))
    tf.end_user_turn_footer()
    out = _ANSI.sub("", capsys.readouterr().out)
    assert "170 tok" in out  # 120 (timings) + 25 + 25 (usage)


def test_unbracketed_call_renders_immediately(nx_fresh, capsys):
    # legacy direct callers (tests, REPL-less paths) behave exactly as before
    out = _out(nx_fresh, capsys, _Usage(500, 105), None, 0, 0.1, 6.0, "ok",
               {}, parser=_ParserStub(3.5, True))
    assert "105 tok" in out and " 30.0 tok/s" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
