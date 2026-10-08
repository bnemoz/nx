#!/usr/bin/env python3
"""Per-job event sink (EVENTSURFACE_SPEC Design §1, Phase 1): ring bounds
(NX_EVENT_TAIL_LINES, default 200), tail ordering (oldest -> newest),
drop_sink's retain-then-drop + retained-tail cap, N concurrent writers
under the sink lock, the zero-cost path (sink_for on an unknown id is
None), invalid kinds stored verbatim (caller's contract), and that
nothing ever prints."""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k in [k for k in os.environ if k.startswith("NX_")]:
    del os.environ[_k]
os.environ["NX_ENV_FILE"] = "/dev/null"

import pytest  # noqa: E402

from nx import _eventsink as es  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_sinks():
    """Registry/retention/env-cache are process globals; rewind them
    around every test so order can't matter."""
    def _reset():
        with es._LOCK:
            es._SINKS.clear()
            es._RETAINED_TAILS.clear()
        es._TAIL_LINES = None
    _reset()
    yield
    _reset()


# --- ring bounds ---------------------------------------------------------------

def test_ring_defaults_to_200_lines(monkeypatch):
    monkeypatch.delenv("NX_EVENT_TAIL_LINES", raising=False)
    sink = es.register_sink("job-bounds")
    for i in range(250):
        sink.write("job-bounds", "stream", f"line {i}")
    t = sink.tail("job-bounds")
    assert len(t) == 200, "ring must hold exactly NX_EVENT_TAIL_LINES (default)"
    # oldest-survivor: the last 200 of 250 are lines 50..249
    assert t[0] == "[stream] line 50"
    assert t[-1] == "[stream] line 249"


def test_ring_honors_nx_event_tail_lines(monkeypatch):
    monkeypatch.setenv("NX_EVENT_TAIL_LINES", "5")
    sink = es.register_sink("job-env")
    for i in range(8):
        sink.write("job-env", "tool", str(i))
    t = sink.tail("job-env")
    assert len(t) == 5
    assert t == ["[tool] 3", "[tool] 4", "[tool] 5", "[tool] 6", "[tool] 7"]


def test_tail_lines_env_parses_once(monkeypatch):
    monkeypatch.setenv("NX_EVENT_TAIL_LINES", "7")
    sink = es.register_sink("job-once")
    assert len(sink._lines) == 0 and sink._lines.maxlen == 7
    monkeypatch.setenv("NX_EVENT_TAIL_LINES", "3")  # read once at first use
    sink2 = es.register_sink("job-once2")
    assert sink2._lines.maxlen == 7, "env must be read once, not per sink"


@pytest.mark.parametrize("raw,expect", [
    ("abc", 200),     # non-int -> default
    ("", 200),        # empty -> default
    ("0", 200),       # zero -> default (a zero-line ring is a dead sink)
    ("-4", 200),      # negative -> default
    ("42", 42),       # valid
    (" 11 ", 11),     # surrounding whitespace tolerated
])
def test_tail_lines_env_parsing(monkeypatch, raw, expect):
    monkeypatch.setenv("NX_EVENT_TAIL_LINES", raw)
    assert es._tail_lines() == expect


def test_tail_lines_absent_is_default(monkeypatch):
    monkeypatch.delenv("NX_EVENT_TAIL_LINES", raising=False)
    assert es._tail_lines() == es.DEFAULT_TAIL_LINES == 200


# --- tail ordering / shape -------------------------------------------------------

def test_tail_returns_oldest_to_newest():
    sink = es.register_sink("job-order")
    for i in range(5):
        sink.write("job-order", "status", f"turn {i}")
    assert sink.tail("job-order") == [f"[status] turn {i}" for i in range(5)]


def test_tail_n_cuts_most_recent():
    sink = es.register_sink("job-n")
    for i in range(10):
        sink.write("job-n", "stream", str(i))
    assert sink.tail("job-n", lines=3) == ["[stream] 7", "[stream] 8", "[stream] 9"]
    assert sink.tail("job-n", lines=0) == []
    assert len(sink.tail("job-n", lines=1000)) == 10  # N over length is fine


def test_invalid_kind_is_stored_verbatim():
    sink = es.register_sink("job-kind")
    sink.write("job-kind", "banana", "not a contract violation")
    assert sink.tail("job-kind") == ["[banana] not a contract violation"]


# --- zero-cost path ---------------------------------------------------------------

def test_sink_for_unknown_job_is_none():
    assert es.sink_for("never-registered") is None
    assert es.sink_for(None) is None
    assert es.sink_for("") is None


def test_sink_for_registered_job():
    sink = es.register_sink("job-here")
    assert es.sink_for("job-here") is sink


# --- drop_sink: retain-then-drop ----------------------------------------------------

def test_drop_sink_retains_final_tail():
    sink = es.register_sink("job-drop")
    for i in range(4):
        sink.write("job-drop", "stream", f"l{i}")
    before = sink.tail("job-drop")
    es.drop_sink("job-drop")
    assert es.sink_for("job-drop") is None, "registry entry must be gone"
    kept = es.retained_tail("job-drop")
    assert kept == before, "final tail must survive the drop, same order"
    assert es.retained_tail("job-drop", lines=2) == ["[stream] l2", "[stream] l3"]


def test_drop_sink_capped_ring_is_what_retains(monkeypatch):
    monkeypatch.setenv("NX_EVENT_TAIL_LINES", "3")
    sink = es.register_sink("job-cap")
    for i in range(10):
        sink.write("job-cap", "tool", str(i))
    es.drop_sink("job-cap")
    # the RETAINED copy is the bounded ring, not the full history
    assert es.retained_tail("job-cap") == ["[tool] 7", "[tool] 8", "[tool] 9"]


def test_retained_tails_capped_at_last_16_jobs():
    for i in range(es.MAX_RETAINED_JOBS + 4):
        s = es.register_sink(f"job-evict-{i}")
        s.write(f"job-evict-{i}", "status", "done")
        es.drop_sink(f"job-evict-{i}")
    assert len(es._RETAINED_TAILS) == es.MAX_RETAINED_JOBS
    # oldest-dropped are evicted, newest kept (insertion order = drop order)
    for i in range(4):
        assert es.retained_tail(f"job-evict-{i}") == []
    for i in range(4, es.MAX_RETAINED_JOBS + 4):
        assert es.retained_tail(f"job-evict-{i}") == ["[status] done"]


def test_drop_sink_unknown_job_is_noop():
    es.drop_sink("ghost")
    assert es.retained_tail("ghost") == []
    assert "ghost" not in es._RETAINED_TAILS


# --- concurrency ---------------------------------------------------------------------

def test_concurrent_writers_lose_nothing_under_lock(monkeypatch):
    monkeypatch.setenv("NX_EVENT_TAIL_LINES", "1000")
    sink = es.register_sink("job-race")
    n_threads, per_thread = 8, 50

    def worker(w):
        for i in range(per_thread):
            sink.write("job-race", "tool", f"w{w}-{i}")

    threads = [threading.Thread(target=worker, args=(w,))
               for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    t = sink.tail("job-race")
    assert len(t) == n_threads * per_thread, "no append may be lost"
    assert len(set(t)) == len(t), "no torn/duplicated line"
    for w in range(n_threads):
        for i in range(per_thread):
            assert f"[tool] w{w}-{i}" in t


def test_concurrent_write_and_tail_and_drop():
    # A repaint (tail) racing drop_sink must never observe a half-retired
    # sink: any snapshot is either the live ring or the retained copy,
    # never neither, never a mix at the boundary.
    sink = es.register_sink("job-retire")
    stop = threading.Event()
    anomalies: list[object] = []

    def writer():
        i = 0
        while not stop.is_set():
            sink.write("job-retire", "stream", f"s{i}")
            i += 1

    def reader():
        while not stop.is_set():
            live = es.sink_for("job-retire")
            if live is not None:
                continue
            kept = es.retained_tail("job-retire")
            if not kept:  # gone from BOTH sides -> half-retired window
                anomalies.append("neither live nor retained")

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    for _ in range(200):
        sink.write("job-retire", "status", "tick")
    es.drop_sink("job-retire")
    stop.set()
    w.join()
    r.join()
    assert not anomalies
    assert es.retained_tail("job-retire"), "retained tail must exist post-drop"


# --- display-only: nothing prints ------------------------------------------------------

def test_nothing_ever_prints(capsys):
    sink = es.register_sink("job-quiet")
    for i in range(300):
        sink.write("job-quiet", "stream", f"chatter {i}")
    sink.tail("job-quiet")
    es.sink_for("job-quiet")
    es.drop_sink("job-quiet")
    es.retained_tail("job-quiet")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
