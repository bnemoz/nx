"""Phase 3 (per-turn event log, /turns) of the explainable harness: the
_turnlog report over the metrics rings, the TurnsView surface, and the /turns
command wiring — same fixture patterns as test_ctxreport.py."""

from nx import _metrics as metrics_mod
from nx import _turnlog


def _metrics(turns, events=None):
    m = metrics_mod._Metrics(session_id="t")
    m.turns = list(turns)
    m.events = list(events or [])
    return m


def test_report_lists_settled_calls_with_cache_pct():
    m = _metrics([
        {"prompt_tokens": 1000, "cache_read_tokens": 500},
        {"prompt_tokens": 1000, "cache_read_tokens": -1},
        {"prompt_tokens": 1000, "cache_read_tokens": 0},
    ])
    lines = _turnlog._turnlog_report_lines(m)
    assert "[0]" in lines[1] and "50%" in lines[1]
    assert "—" in lines[2]          # -1 sentinel → no data, not 0%
    assert "0%" in lines[3]         # real zero → a genuine miss


def test_report_appends_attributed_event_notes():
    m = _metrics(
        [{"prompt_tokens": 10, "cache_read_tokens": 0},
         {"prompt_tokens": 10, "cache_read_tokens": 0}],
        [{"ts": 0.0, "kind": "retry", "detail": "ladder step 2", "call": 1}])
    lines = _turnlog._turnlog_report_lines(m)
    assert "retry: ladder step 2" in lines[2]
    assert "retry" not in lines[1]


def test_report_unattributed_events_get_their_own_section():
    m = _metrics([{"prompt_tokens": 10, "cache_read_tokens": 0}],
                 [{"ts": 0.0, "kind": "salvage", "detail": "args rewritten",
                   "call": None}])
    lines = _turnlog._turnlog_report_lines(m)
    assert any("unattributed" in ln for ln in lines)
    assert any("salvage: args rewritten" in ln for ln in lines)


def test_empty_ring_placeholder():
    assert _turnlog._turnlog_report_lines(_metrics([])) == \
        ["  no settled calls yet"]
    assert _turnlog._turnlog_report_lines(None) == \
        ["  no settled calls yet"]


def test_garbage_entries_do_not_raise():
    m = _metrics(["nope", {"prompt_tokens": "x"}, None],
                 ["junk", {"kind": None}])
    lines = _turnlog._turnlog_report_lines(m)
    assert "[0] —" in lines[1]


def test_metrics_event_ring_appends_and_caps():
    m = metrics_mod._Metrics(session_id="t")
    for i in range(metrics_mod._TURN_RING_MAX + 5):
        m.event("retry", f"step {i}", call=0)
    assert len(m.events) == metrics_mod._TURN_RING_MAX
    assert m.events[-1]["detail"] == f"step {metrics_mod._TURN_RING_MAX + 4}"


def test_record_turn_flags_a_big_cache_miss():
    m = metrics_mod._Metrics(session_id="t")
    m.add({"input_tokens": 4096, "cache_read_tokens": 0})
    m.add({"input_tokens": 4096, "cache_read_tokens": 4096})
    kinds = [e["kind"] for e in m.events]
    assert kinds == ["cache-miss"]          # the hit (call 1) is not flagged
    assert m.events[0]["call"] == 0


def test_turns_view_paints_the_same_lines():
    from nx import _fullscreen as fs
    m = _metrics([{"prompt_tokens": 100, "cache_read_tokens": 100}])
    import nx._state as st
    saved = st._METRICS
    st._METRICS = m
    try:
        view = _turnlog.turns_view()
        inline = _turnlog._turnlog_report_lines(m)
        painted = "\n".join(view.render(240, 40))
        assert "100%" in painted and "[0]" in painted
        assert fs.make_view("turns") is not None
        assert view.on_key("esc", None) == "quit"
    finally:
        st._METRICS = saved


def test_context_view_t_jumps_to_turns_view():
    from nx import _ctxreport
    pushed = []

    class _Shell:
        def push(self, v):
            pushed.append(v)

    class _State:
        messages = [{"role": "user", "content": "hi"}]

    view = _ctxreport.context_view(_State())
    assert view.on_key("t", _Shell()) == "repaint"
    assert len(pushed) == 1 and type(pushed[0]).__name__ == "TurnsView"
    assert view.on_key("t", None) == "repaint"     # shell=None degrades


def test_cmd_turns_inline_prints_the_report(monkeypatch, capsys):
    import nx._state as st
    m = _metrics([{"prompt_tokens": 64, "cache_read_tokens": 0}])
    monkeypatch.setattr(st, "_METRICS", m, raising=False)
    monkeypatch.setattr("nx._fullscreen.wants_fullscreen", lambda: False)
    _turnlog._cmd_turns("")
    out = capsys.readouterr().out
    assert "[0]" in out and "cached 0%" in out
