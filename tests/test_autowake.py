#!/usr/bin/env python3
"""Tests for the auto-continuation ("wake") feature (nx/_autowake.py, 0.36.0).

A backgrounded run_bash job or an async spawn_agent stages its completion
note, but until 0.36.0 that note only reached the model on the first tool
call of a LATER turn — i.e. it sat parked until the user typed something.
These tests pin the three pieces that fix that:

  1. wake_due() — the "a wake is warranted" predicate (armed gating, notice
     slots, the read-only bg-job/sub-agent sweeps, the not-mid-turn guard).
  2. the editor — read_multiline(wake=True) returns WAKE_INPUT instead of
     blocking, stashing typed text so the wake never eats half a message.
  3. the REPL — the WAKE sentinel becomes a synthetic user turn that flushes
     the staged note and calls model_turn, with no history/slash side effects.
  4. the 0.36.3 loop fix — a bg job finished after its turn answered in plain
     text used to re-wake forever (every wake turn also ended text-only, so
     nothing ever marked the job seen); wake_due() must run the completion
     sweep itself, and a text-only wake turn must deliver its note.
"""
import builtins
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m
from nx import _autowake

_REAL_PRINT = builtins.print


def _silence():
    builtins.print = lambda *a, **k: None


def _restore():
    builtins.print = _REAL_PRINT


def setup_function(_function):
    m.st._WAKE_ARMED = False
    m.st._WAKE_DUE = False
    m.st._WAKE_PREFILL = ""
    m.st._TURN_ACTIVE = False
    os.environ["NX_WAKE_PROMPT"] = "1"
    # Don't leave a real poller thread running between tests.
    _autowake._POLLER = None


# ---------------------------------------------------------------------------
# 1. wake_due(): armed gate
# ---------------------------------------------------------------------------
def test_wake_due_requires_armed_session():
    # A fresh session that never backgrounded anything must never wake.
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: finished.]"
    assert m.st._WAKE_ARMED is False
    assert _autowake.wake_due() is False
    # ...and arming alone isn't enough either: nothing pending.
    m.st._PENDING_SUBAGENT_NOTICE = None
    m.st._WAKE_ARMED = True
    assert _autowake.wake_due() is False


def test_wake_due_on_staged_notice():
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = "[Runtime note: sub-agent finished.]"
    try:
        assert _autowake.wake_due() is True
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None


def test_wake_due_disabled_by_env():
    m.st._WAKE_ARMED = True
    m.st._PENDING_BG_COMPLETION_NOTICE = "note"
    os.environ["NX_WAKE_PROMPT"] = "0"
    try:
        assert _autowake.wake_due() is False
    finally:
        os.environ["NX_WAKE_PROMPT"] = "1"
        m.st._PENDING_BG_COMPLETION_NOTICE = None


def test_wake_due_not_while_turn_active():
    m.st._WAKE_ARMED = True
    m.st._TURN_ACTIVE = True
    m.st._PENDING_SUBAGENT_NOTICE = "note"
    try:
        # A notice staged mid-turn is that turn's own machinery's job to
        # flush; waking concurrently would race the shared transcript.
        assert _autowake.wake_due() is False
    finally:
        m.st._TURN_ACTIVE = False
    assert _autowake.wake_due() is True


def test_wake_due_holds_off_while_agent_running():
    m.st._WAKE_ARMED = True
    from nx import _subagent
    jid = "agent-test-running"
    with _subagent._ASYNC_LOCK:
        _subagent._ASYNC_AGENTS[jid] = {
            "instruction": "t", "status": "running",
            "answer": None, "started": time.time(), "announced": False,
        }
    try:
        assert _subagent._async_running() is True
        assert _autowake.wake_due() is False
        # Finishing it makes the (unannounced) answer wake-worthy.
        with _subagent._ASYNC_LOCK:
            _subagent._ASYNC_AGENTS[jid]["status"] = "done"
            _subagent._ASYNC_AGENTS[jid]["answer"] = "the answer"
        assert _autowake.wake_due() is True
    finally:
        with _subagent._ASYNC_LOCK:
            _subagent._ASYNC_AGENTS.pop(jid, None)


def test_wake_due_on_finished_unseen_bg_job(monkeypatch):
    m.st._WAKE_ARMED = True
    fake_jobs = [{"pid": "424242", "alive": False,
                  "owner_pid": os.getpid(), "command": "true"}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)
    assert _autowake._finished_unseen_bg_jobs() == ["424242"]
    assert _autowake.wake_due() is True
    # Already-announced pids don't re-wake (mirrors _BG_NOTICE_SEEN).
    m.st._BG_NOTICE_SEEN = {"424242"}
    assert _autowake._finished_unseen_bg_jobs() == []
    m.st._BG_NOTICE_SEEN = None


# ---------------------------------------------------------------------------
# 2. the editor: wake-aware prompt
# ---------------------------------------------------------------------------
class _FakeFd:
    """Stands in for the stdin fd the editor polls."""

    def __init__(self):
        self.ready = False


def test_read_multiline_wake_returns_sentinel(monkeypatch):
    # Force the raw-editor branch: read_multiline looks up _chatbox_raw in
    # its own module's globals, and isatty is checked on the real streams.
    import nx._chatbox as cb
    monkeypatch.setattr(cb.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cb.sys.stdout, "isatty", lambda: True)
    got = {}

    def fake_raw(initial, history, wake=None):
        got["wake"] = wake
        return "typed"

    monkeypatch.setattr(cb, "_chatbox_raw", fake_raw)
    m.read_multiline(wake=True)  # type: ignore[call-arg]
    assert got["wake"] is _autowake.WAKE_INPUT
    # wake=False keeps the old plain-editor contract.
    m.read_multiline()  # type: ignore[call-arg]
    assert got["wake"] is None


def test_chatbox_raw_wake_stashes_prefill_and_returns_sentinel(monkeypatch):
    # Drive _chatbox_raw's key loop with a scripted stdin: WAKE_DUE is set
    # and stdin never becomes ready, so the loop must stash the typed buffer
    # and return the sentinel without any keystroke.
    import nx._chatbox as cb
    m.st._WAKE_DUE = True
    monkeypatch.setattr(cb, "wait_ready", lambda fd, timeout: False)
    monkeypatch.setattr(cb, "raw_mode", lambda fd, icrnl=False: _null_ctx())
    monkeypatch.setattr(cb.sys.stdin, "fileno", lambda: 0)

    class _Sink:
        def write(self, *a, **k):
            pass

        def flush(self):
            pass

    real_stdout = sys.stdout
    sys.stdout = _Sink()
    try:
        out = cb._chatbox_raw("half-typed line", [], wake=_autowake.WAKE_INPUT)
    finally:
        sys.stdout = real_stdout
    assert out is _autowake.WAKE_INPUT
    assert m.st._WAKE_PREFILL == "half-typed line"


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# 3. the REPL: WAKE_INPUT → synthetic turn
# ---------------------------------------------------------------------------
def test_repl_wake_runs_turn_and_flushes_notice(monkeypatch):
    _silence()
    saved = {k: getattr(m, k) for k in
             ("read_multiline", "model_turn", "_banner", "_record_input",
              "_flush_pending_background_notice")}
    m._banner = lambda: ""
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = ("[Runtime note: background sub-agent "
                                     "agent-1 finished. Its answer: done]")
    flushed = {}

    def fake_flush(messages):
        # Mirror the real one: append staged notices as user messages, clear.
        if m.st._PENDING_SUBAGENT_NOTICE:
            messages.append({"role": "user",
                             "content": m.st._PENDING_SUBAGENT_NOTICE})
            m.st._PENDING_SUBAGENT_NOTICE = None
        flushed["called"] = flushed.get("called", 0) + 1

    calls = {"n": 0, "last_user": None}

    def fake_model_turn(messages, *a, **kw):
        calls["n"] += 1
        calls["last_user"] = [mm for mm in messages if mm.get("role") == "user"][-1]
        assert calls["last_user"]["content"] == _autowake.WAKE_INSTRUCTION
        assert any("[Runtime note: background sub-agent" in
                   (mm.get("content") or "") for mm in messages), \
            "wake turn must flush the staged completion note"
        return m.TURN_DONE

    history = []
    m._record_input = lambda user, hist: history.append(user)
    turns = iter([_autowake.WAKE_INPUT, "/quit"])
    m.read_multiline = lambda initial="", history=None, wake=False: next(turns)
    m.model_turn = fake_model_turn
    m._flush_pending_background_notice = fake_flush
    try:
        m.main()
        assert calls["n"] == 1, "exactly one auto-continuation turn"
        assert flushed["called"] == 1
        assert history == [], "synthetic wake input must not enter Up-history"
        assert m.st._WAKE_DUE is False
        assert m.st._PENDING_SUBAGENT_NOTICE is None
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
        m.st._PENDING_SUBAGENT_NOTICE = None
        _restore()


def test_repl_wake_restores_typed_prefill(monkeypatch):
    """Text typed before the wake must reappear at the next prompt, not be
    swallowed by the auto-continuation turn."""
    _silence()
    saved = {k: getattr(m, k) for k in
             ("read_multiline", "model_turn", "_banner", "_record_input")}
    m._banner = lambda: ""
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = "note"
    # Simulate the editor having stashed a half-typed line before returning
    # the sentinel.
    m.st._WAKE_PREFILL = "half-typed line"

    def fake_model_turn(messages, *a, **kw):
        return m.TURN_DONE

    prompts = []

    m.model_turn = fake_model_turn
    m._record_input = lambda user, hist: None
    # The editor returns the sentinel once, then /quit. A real editor stashes
    # its (initial-seeded) buffer on the wake path — mirror that here.
    seq = iter([_autowake.WAKE_INPUT, "/quit"])
    m.read_multiline = lambda initial="", history=None, wake=False: (
        prompts.append(initial)
        or (setattr(m.st, "_WAKE_PREFILL", initial) if _autowake.WAKE_INPUT else None,
            next(seq))[1])
    try:
        m.main()
        # First prompt saw the restored prefill... wait: the prefill is
        # consumed by the FIRST read (before the wake turn), stashed by the
        # editor, and restored at the SECOND read.
        assert prompts[1] == "half-typed line", prompts
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
        m.st._WAKE_PREFILL = ""
        _restore()


# ---------------------------------------------------------------------------
# 4. the 0.36.3 loop fix: wake_due() runs the sweep; a text-only wake turn
#    delivers the note so the loop terminates
# ---------------------------------------------------------------------------
def test_wake_due_runs_sweep_and_marks_seen(monkeypatch):
    """A finished job nothing ever announced must be staged by wake_due()
    itself (the containing turn ended as plain text, so run_tool()'s scan
    never ran) — and the NEXT wake_due() must be False, not an endless
    'background job finished — auto-continuing' loop."""
    _silence()
    m.st._WAKE_ARMED = True
    fake_jobs = [{"pid": "424243", "alive": False,
                  "owner_pid": os.getpid(), "command": "true",
                  "tail": "done"}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)
    monkeypatch.setattr("nx._bash_jobs._read_bg_exit_status",
                        lambda pid: (True, 0, "/tmp/x.log"))
    m.st._BG_NOTICE_SEEN = None
    m.st._PENDING_BG_COMPLETION_NOTICE = None
    try:
        assert _autowake.wake_due() is True
        note = m.st._PENDING_BG_COMPLETION_NOTICE
        assert note and "424243" in note, "the sweep must stage the exit note"
        # The wake turn flushes the note it staged; only then is the job both
        # seen AND delivered — so the next wake_due() must be False.
        m._flush_pending_background_notice([])
        assert _autowake.wake_due() is False
    finally:
        m.st._BG_NOTICE_SEEN = None
        m.st._PENDING_BG_COMPLETION_NOTICE = None
        _restore()


def test_repl_wake_delivers_note_on_text_only_turn(monkeypatch):
    """The wake turn itself must run the sweep before flushing: a wake turn
    that ends as plain text (no tool call) still delivers the finished job's
    exit line into the transcript."""
    _silence()
    saved = {k: getattr(m, k) for k in
             ("read_multiline", "model_turn", "_banner", "_record_input",
              "_flush_pending_background_notice", "_run_bg_completion_scan")}
    m._banner = lambda: ""
    m.st._WAKE_ARMED = True
    # Pre-stage nothing: the sweep must be what produces the note.
    m.st._PENDING_BG_COMPLETION_NOTICE = None
    m.st._BG_NOTICE_SEEN = None
    fake_jobs = [{"pid": "424244", "alive": False,
                  "owner_pid": os.getpid(), "command": "true", "tail": "ok"}]
    monkeypatch.setattr("nx._bash_jobs.list_bg_jobs", lambda: fake_jobs)
    monkeypatch.setattr("nx._bash_jobs._read_bg_exit_status",
                        lambda pid: (True, 0, "/tmp/x.log"))
    seen_in_turn = {}

    def fake_model_turn(messages, *a, **kw):
        # Model emits plain text: no tool call, so run_tool() never scans.
        users = [(i, mm.get("content") or "") for i, mm in enumerate(messages)
                 if mm.get("role") == "user"]
        seen_in_turn["note"] = any("424244" in c for _, c in users)
        seen_in_turn["instruction_idx"] = next(
            (i for i, c in users if c == _autowake.WAKE_INSTRUCTION), None)
        seen_in_turn["note_idx"] = next(
            (i for i, c in users if "424244" in c), None)
        return m.TURN_DONE

    turns = iter([_autowake.WAKE_INPUT, "/quit"])
    m.read_multiline = lambda initial="", history=None, wake=False: next(turns)
    m.model_turn = fake_model_turn
    m._record_input = lambda user, hist: None
    try:
        m.main()
        assert seen_in_turn["note"], "text-only wake turn must deliver the note"
        assert seen_in_turn["instruction_idx"] is not None
        assert seen_in_turn["note_idx"] is not None
        assert seen_in_turn["note_idx"] > seen_in_turn["instruction_idx"], \
            "the completion note rides right after the wake instruction"
        # The delivered job is marked seen: no further wake is due.
        assert _autowake.wake_due() is False
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
        m.st._BG_NOTICE_SEEN = None
        m.st._PENDING_BG_COMPLETION_NOTICE = None
        _restore()


def test_poller_raises_due_flag():
    # Deterministic poller check with a tiny interval — no sleeps of seconds.
    os.environ["NX_WAKE_POLL_SECONDS"] = "0.2"
    m.st._WAKE_ARMED = True
    m.st._PENDING_SUBAGENT_NOTICE = "note"
    try:
        _autowake.ensure_poller()
        deadline = time.time() + 5
        while not m.st._WAKE_DUE and time.time() < deadline:
            time.sleep(0.02)
        assert m.st._WAKE_DUE is True, "poller must raise _WAKE_DUE"
    finally:
        m.st._PENDING_SUBAGENT_NOTICE = None
        m.st._WAKE_DUE = False
        os.environ["NX_WAKE_POLL_SECONDS"] = "2.0"


def test_arm_wake_noop_when_disabled():
    os.environ["NX_WAKE_PROMPT"] = "0"
    try:
        _autowake.arm_wake()
        assert m.st._WAKE_ARMED is False
    finally:
        os.environ["NX_WAKE_PROMPT"] = "1"
