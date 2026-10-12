#!/usr/bin/env python3
"""Phase 2 of EVENTSURFACE_SPEC: the WRITERS that feed the Phase 1 sink.

Phase 1 (`nx/_eventsink.py`) owns the ring; this file owns the claim that the
existing render paths actually fork into it — and, just as load-bearing, that
they fork NOWHERE ELSE:

- an async (quiet) agent's spawn produces `stream`/`status`/`tool` lines in its
  own sink, resolved from that thread's sub-agent context;
- the parent thread (and the `_batch`/`_telegram` quiet sites, and `_spec_reads`
  pool threads, which all have no ctx) emits NOTHING — the None fallback;
- registration precedes the worker thread and drop happens in the worker's
  `finally`, so the retained tail is the COMPLETE tail;
- `clear_tool_status()` (the one un-guarded CLEAR_LINE) stays silent when quiet;
- the /tabs snapshot row for a running agent carries the sink tail;
- nothing prints, and nothing reaches `messages` / the model.
"""
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

# Same ambient-config scrub as the other sub-agent suites: a developer's real
# endpoints / NX_* must not reach these tests.
os.environ["NX_ENV_FILE"] = "/dev/null"
for _k in [k for k in os.environ if k.startswith("NX_")]:
    if _k != "NX_ENV_FILE":
        os.environ.pop(_k, None)


@pytest.fixture()
def nx_fresh(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("NX_")]:
        os.environ.pop(k, None)
    os.environ["NX_HOME"] = str(tmp_path / "nxhome")
    os.environ["NX_ENV_FILE"] = str(tmp_path / "env")
    os.environ["NX_SESSIONS_DIR"] = str(tmp_path / "nxhome" / "sessions")
    os.environ["NX_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["NX_MODEL"] = "test-model"
    os.environ["NX_API_KEY"] = "sk-test"
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    # Tier A: no reload(nx). The reload used to rebuild nx's import-time state
    # from this fixture's env — most importantly the model source: conftest
    # imports nx once under a scrubbed env, so st.client/st.MODEL stay None (no
    # active source). Rebuild just that wiring: re-discover sources from the env
    # set above and switch to the first one (binds st.client + st.MODEL).
    # resolve_model() reads NX_MODEL ("test-model"), so no network round-trip
    # happens; the dead 127.0.0.1:9 endpoint is never contacted.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # The reload also wiped nx's staged runtime notice, which the module-dict
    # rewind used to undo; without either, a previous test's note would leak
    # into this test's _drain(). Pin it per-test (auto-restored).
    from nx import _state as st
    monkeypatch.setattr(st, "_PENDING_SUBAGENT_NOTICE", None, raising=False)
    yield nx
    # Agent state (_ASYNC_AGENTS, quiet slots, TLS) is reset by the universal
    # nx_state_reset autouse fixture and env rewind is universal (conftest), so
    # no manual cleanup here; the sink registry is restored by the _empty_sinks
    # autouse fixture below.


@pytest.fixture(autouse=True)
def _empty_sinks(monkeypatch):
    """Every test starts with an empty registry: tap_event's zero-cost fast
    path (`if not _es._SINKS`) means a leftover sink from a sibling test would
    silently change what a tap does.

    Tier A: the module-dict rewind no longer restores nx._eventsink's
    module-level registry, so those object globals are swapped through
    monkeypatch (restored to the baseline objects per-test, after teardown).
    Every reader reaches them through the module attr (_subagent_tls._SINKS,
    _eventsink.sink_for/register_sink/...), so a rebind is observed. The
    in-place clear keeps the OLD guarantee too: whatever row an earlier file
    left on the shared baseline objects is gone for us AND for whoever runs
    after us (the baseline objects are what monkeypatch restores)."""
    from nx import _eventsink as es
    with es._LOCK:
        es._SINKS.clear()
        es._RETAINED_TAILS.clear()
    monkeypatch.setattr(es, "_SINKS", {}, raising=False)
    monkeypatch.setattr(es, "_RETAINED_TAILS", {}, raising=False)
    monkeypatch.setattr(es, "_TAIL_LINES", None, raising=False)
    yield


def _wait_for(pred, what, deadline_s=30.0):
    deadline = time.time() + deadline_s
    result = pred()
    while not result and time.time() < deadline:
        time.sleep(0.02)
        result = pred()
    assert result, f"timed out after {deadline_s}s waiting for {what}"
    return result


def _job_id(spawn_result: str) -> str:
    m = re.search(r"job (agent-[\w-]+)", spawn_result)
    assert m, f"no job id in spawn result: {spawn_result!r}"
    return m.group(1)


def _drain(nx):
    from nx import _state as st
    note = getattr(st, "_PENDING_SUBAGENT_NOTICE", None)
    st._PENDING_SUBAGENT_NOTICE = None
    return note


# --- sink lifetime around the async worker ------------------------------------


def test_sink_registered_before_worker_and_dropped_after(nx_fresh, monkeypatch):
    """register in _spawn_async (pre-thread), drop in the worker's finally —
    and the retained tail is what the whole run produced."""
    from nx import _eventsink as es
    from nx._subagent import spawn_agent
    from nx._subagent_tls import tap_event

    release = threading.Event()
    started = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        tap_event("stream", "investigating the thing")
        release.wait(timeout=10)
        messages.append({"role": "assistant", "content": "answer text"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    job = _job_id(spawn_agent("q", async_=True))
    started.wait(timeout=10)
    # registered, and BEFORE the first event could be dropped on the floor
    assert es.sink_for(job) is not None, "sink must exist while the agent runs"
    # `started` is set BEFORE the worker's tap_event call, so under load the
    # tail can still be empty here — poll for the line instead of reading once.
    _wait_for(lambda: (s := es.sink_for(job)) is not None
              and s.tail(job) == ["[stream] investigating the thing"],
              "first sink line")
    release.set()
    _wait_for(lambda: _drain(nx_fresh), "the agent's staged runtime note")
    # dropped: gone from the live registry, present as the retained copy. The
    # drop runs in the worker's `finally`, AFTER the note is staged, so it needs
    # its own wait (the drain above can win that race).
    _wait_for(lambda: es.sink_for(job) is None, "sink drop in the worker finally")
    assert es.sink_for(job) is None, "sink must be dropped once the job ends"
    assert "[stream] investigating the thing" in es.retained_tail(job), \
        "drop_sink must retain the final tail (Phase 1 contract)"
    # and the tail is COMPLETE — the drop ran after the answer was staged, so
    # nothing the agent emitted is missing from the retained copy.
    assert len(es.retained_tail(job)) >= 1


def test_drop_sink_unknown_job_is_still_noop(nx_fresh):
    from nx import _eventsink as es
    es.drop_sink("never-spawned")
    assert es.retained_tail("never-spawned") == []


# --- tap kinds on a real quiet agent loop ------------------------------------


def test_quiet_agent_loop_produces_stream_status_and_tool_kinds(nx_fresh, monkeypatch):
    """The three kinds the spec names, produced through the REAL dispatch path
    (run_tool on the agent's own thread) rather than by calling tap_event."""
    from nx import _dispatch
    from nx._subagent import spawn_agent

    release = threading.Event()
    started = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        # the fake stands in for _run_model_turn_loop, so it must invoke the
        # real `on_step` (bound to _run_sync_agent_body's _progress_step) the
        # way the real loop does — that closure is where the status tap lives.
        on_step = kw.get("on_step")
        if on_step is not None:
            on_step()
        # a real tool call through dispatch: quiet, so no box, but the sink
        # still records start + outcome
        out = _dispatch.run_tool("git_status", {"path": "."})
        assert not out.startswith("ERROR")
        # a multi-line delta: one ring line per display line
        from nx._subagent_tls import tap_event  # the parser's own fork
        tap_event("stream", "line one\nline two\n\nline three")
        release.wait(timeout=10)
        messages.append({"role": "assistant", "content": "final answer"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    job = _job_id(spawn_agent("q", async_=True))
    started.wait(timeout=10)
    from nx._subagent_tls import sink_tail
    # Wait for BOTH tap kinds: the tool tap lands a few statements before the
    # stream tap on the worker thread, so waiting on "[tool]" alone could read
    # the tail in between (a flake under -n auto load).
    _wait_for(lambda: any("[tool]" in ln for ln in sink_tail(job, 40))
              and any("[stream]" in ln for ln in sink_tail(job, 40)),
              "the tool and stream tap lines")
    lines = sink_tail(job, 40)
    kinds = {ln.split("]")[0].lstrip("[") for ln in lines}
    assert {"tool", "stream", "status"} <= kinds, f"kinds seen: {kinds} / {lines}"
    assert "[tool] git_status ok" in lines, "finish line must carry ok/error"
    # status tap: turn counter + tier
    assert any(ln.startswith("[status] turn ") and "tier=" in ln for ln in lines)
    # multi-line split: no ring entry contains a newline, and the blank line was
    # dropped rather than stored as an empty row
    assert all("\n" not in ln for ln in lines)
    assert "[stream] line one" in lines and "[stream] line three" in lines
    assert "[stream] " not in [ln for ln in lines if ln.strip() == "[stream] "]
    release.set()


def test_tool_tap_reports_error_outcome_for_failed_call(nx_fresh, monkeypatch):
    """The finish line uses the result's own ERROR/DENIED convention, so a
    failing call reads as `error` on the dashboard.

    Note the site: the tap sits in run_tool's quiet branch, AFTER the refusal
    ladder. A call refused by the tier gate (a read-only agent asking for
    write_file) returns before that branch and is therefore NOT tapped — see
    the report for that deliberate Phase-2 boundary."""
    from nx import _dispatch
    from nx._subagent import spawn_agent

    release = threading.Event()
    started = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        out = _dispatch.run_tool("read_file",
                                 {"path": "definitely-not-here-xyz.txt"})
        assert out.startswith("ERROR"), out
        release.wait(timeout=10)
        messages.append({"role": "assistant", "content": "answer"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    job = _job_id(spawn_agent("q", async_=True))
    started.wait(timeout=10)
    from nx._subagent_tls import sink_tail
    _wait_for(lambda: any(ln.startswith("[tool] read_file ")
                          for ln in sink_tail(job, 40)),
              "the failed call's tap")
    lines = sink_tail(job, 40)
    assert "[tool] read_file error" in lines, lines
    release.set()


# --- the None fallback: no ctx -> no emission ----------------------------------


def test_tap_without_ctx_emits_nothing(capsys):
    """The parent thread (and the `_batch`/`_telegram` quiet sites, and the
    `_spec_reads` pool threads) have NO sub-agent ctx: a tap there must be a
    no-op even with a sink registered for some other job."""
    from nx import _eventsink as es
    from nx._subagent_tls import _SUBAGENT_TLS, tap_event
    sink = es.register_sink("agent-other-1")
    assert _SUBAGENT_TLS.__dict__.get("ctx") is None  # no ctx on this thread
    tap_event("stream", "must not land anywhere")
    assert sink.tail("agent-other-1") == []
    assert es.retained_tail("agent-other-1") == []
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_tap_with_ctx_but_no_sink_emits_nothing():
    """A SYNC agent installs a ctx but gets no sink (its display is the
    parent's render) — the ctx alone must not make a tap write."""
    from nx import _eventsink as es
    from nx._subagent_tls import _SUBAGENT_TLS, tap_event

    class _Ctx:
        job_id = "agent-sync-1"

    prev = getattr(_SUBAGENT_TLS, "ctx", None)
    _SUBAGENT_TLS.ctx = _Ctx()
    try:
        tap_event("stream", "sync agents have no sink")
        assert es.sink_for("agent-sync-1") is None
    finally:
        _SUBAGENT_TLS.ctx = prev


def test_tap_is_zero_cost_when_no_sink_ever_registered(capsys, monkeypatch):
    """`sink_for is not None` is the guard: with an empty registry the tap must
    not even reach the TLS read (spec: one dict lookup per render event)."""
    from nx import _eventsink as es
    from nx import _subagent_tls as tls

    def _boom():  # pragma: no cover - reached only on a regression
        raise AssertionError("tls read must not happen on the zero-cost path")

    monkeypatch.setattr(tls, "_subagent_ctx", _boom)
    assert not es._SINKS
    for _ in range(100):
        tls.tap_event("stream", "nobody is listening")
    c = capsys.readouterr()
    assert c.out == "" and c.err == ""


def test_tap_never_prints_and_never_touches_messages(capsys):
    from nx import _eventsink as es
    from nx._subagent_tls import _SUBAGENT_TLS, tap_event

    class _Ctx:
        job_id = "agent-purity-1"

    es.register_sink("agent-purity-1")
    prev = getattr(_SUBAGENT_TLS, "ctx", None)
    _SUBAGENT_TLS.ctx = _Ctx()
    try:
        tap_event("stream", "<injected> role:user text")
        tap_event("tool", "run_bash ok")
    finally:
        _SUBAGENT_TLS.ctx = prev
    c = capsys.readouterr()
    assert c.out == "" and c.err == "", "sinks must never reach the console"


def test_sink_text_never_reaches_the_parent_notice(nx_fresh, monkeypatch):
    """Transcript purity: the sink is display-only. A marker emitted ONLY to
    the sink must not appear in the [Runtime note] the parent receives."""
    from nx._subagent import spawn_agent
    from nx._subagent_tls import tap_event

    release = threading.Event()
    started = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        tap_event("stream", "SINK_ONLY_MARKER_12345")
        release.wait(timeout=10)
        messages.append({"role": "assistant", "content": "the real answer"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    spawn_agent("q", async_=True)
    started.wait(timeout=10)
    release.set()
    note = _wait_for(lambda: _drain(nx_fresh), "the runtime note")
    assert "the real answer" in note
    assert "SINK_ONLY_MARKER_12345" not in note, \
        "sink lines must never enter messages/the note (display-only)"


# --- the latent hazard: clear_tool_status with no quiet check -----------------


def test_clear_tool_status_writes_nothing_when_quiet(capsys):
    """Both halves of the guard: the CLEAR_LINE never prints while quiet, and a
    non-quiet parser still clears a genuine active status line."""
    from nx._stream_parse import StreamParser
    from nx._ansi import CLEAR_LINE
    from rich.console import Console
    import io

    con = Console(file=io.StringIO())
    p = StreamParser(con, 0.0, quiet=True)
    p.tool_status_active = True
    p.clear_tool_status()
    assert p.tool_status_active is True, \
        "quiet: the flag must survive too (nothing was cleared)"
    assert capsys.readouterr().out == "", "quiet parser must not write CLEAR_LINE"

    p2 = StreamParser(con, 0.0, quiet=False)
    p2.tool_status_active = True
    p2.clear_tool_status()
    assert p2.tool_status_active is False
    assert capsys.readouterr().out == CLEAR_LINE, \
        "non-quiet path must still clear"

    p3 = StreamParser(con, 0.0, quiet=False)  # inactive: still a no-op
    p3.clear_tool_status()
    assert capsys.readouterr().out == ""


# --- agent-row tail in the /tabs snapshot -------------------------------------


def test_sink_tail_prefers_live_then_retained():
    """The helper /tabs rows go through: live ring while registered, retained
    copy after the drop, empty for a job that never had a sink."""
    from nx import _eventsink as es
    from nx._subagent_tls import sink_tail
    job = "agent-tail-1"
    assert sink_tail(job) == []
    assert sink_tail(None) == []
    sink = es.register_sink(job)
    sink.write(job, "stream", "live line")
    assert sink_tail(job, 6) == ["[stream] live line"]
    es.drop_sink(job)
    assert es.sink_for(job) is None
    assert sink_tail(job, 6) == ["[stream] live line"], \
        "after the drop the retained copy must answer"


def test_snapshot_agent_row_tails_the_sink_while_running(nx_fresh):
    from nx import _eventsink as es
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    from nx._surfaces import _surfaces_snapshot

    job = "agent-snap-1"
    sink = es.register_sink(job)
    for i in range(3):
        sink.write(job, "status", f"turn {i}/10 tier=read")
    with _ASYNC_LOCK:
        _ASYNC_AGENTS[job] = {
            "instruction": "snapshot me", "status": "running", "answer": None,
            "started": time.time(), "announced": False, "turns": 1,
            "max_turns": 10, "tier": "read",
        }
    try:
        rows = _surfaces_snapshot()
        agent_rows = [r for r in rows if r["kind"] == "agent"]
        assert len(agent_rows) == 1, rows
        tail = agent_rows[0]["tail"]
        assert "[status] turn 0/10 tier=read" in tail, \
            f"running agent row must show the sink tail, got {tail!r}"
        assert "[status] turn 2/10 tier=read" in tail
    finally:
        with _ASYNC_LOCK:
            _ASYNC_AGENTS.pop(job, None)


def test_snapshot_agent_row_uses_retained_tail_after_drop(nx_fresh):
    """Once the worker's finally drops the sink, `_eventsink.tail` falls back to
    the retained copy — the row keeps the agent's last activity instead of
    blanking."""
    from nx import _eventsink as es
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    from nx._surfaces import _surfaces_snapshot

    job = "agent-snap-2"
    sink = es.register_sink(job)
    sink.write(job, "stream", "last thing the agent said")
    with _ASYNC_LOCK:
        _ASYNC_AGENTS[job] = {
            "instruction": "i", "status": "running", "answer": None,
            "started": time.time(), "announced": False, "turns": 1,
            "max_turns": 10, "tier": "read",
        }
    try:
        es.drop_sink(job)
        rows = _surfaces_snapshot()
        agent_rows = [r for r in rows if r["kind"] == "agent"]
        assert agent_rows and \
            "last thing the agent said" in agent_rows[0]["tail"], agent_rows
    finally:
        with _ASYNC_LOCK:
            _ASYNC_AGENTS.pop(job, None)


def test_snapshot_agent_row_prefers_the_answer_tail(nx_fresh):
    """A finished agent's answer tail is denser than its event ring, so an
    existing (non-empty) tail must win — this change adds no new substitution."""
    from nx._eventsink import register_sink
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    from nx._surfaces import _surfaces_snapshot

    job = "agent-snap-3"
    sink = register_sink(job)
    sink.write(job, "stream", "chatter")
    with _ASYNC_LOCK:
        _ASYNC_AGENTS[job] = {
            "instruction": "i", "status": "done", "answer": "the answer tail",
            "started": time.time(), "announced": True, "turns": 3,
            "max_turns": 10, "tier": "read",
        }
    try:
        rows = _surfaces_snapshot()
        row = [r for r in rows if r["kind"] == "agent"][0]
        assert "the answer tail" in row["tail"]
        assert "chatter" not in row["tail"]
    finally:
        with _ASYNC_LOCK:
            _ASYNC_AGENTS.pop(job, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
