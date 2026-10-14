"""Async (background) sub-agents: spawn_agent(async=true) returns a job id,
runs the same read-only sub-loop on a daemon thread, and delivers the answer
to the PARENT as a one-time [Runtime note] — staged by the worker, swept by
run_tool()'s first-tool-of-turn scan, flushed by
_flush_pending_background_notice. Also covers: no re-announcement, failure
finalization (a dead agent still produces exactly one note), the sub-agent
loop never consuming the parent's staged notice, string-boolean `async`
acceptance, /bg listing, and the NX_SUBAGENTS gate carrying over.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# 0.96.0: foreground spawn results are head+report-path by default; these
# tests assert on the full inline answer, so opt into the legacy inline mode.
os.environ["NX_SUBAGENT_SYNC_INLINE"] = "1"


@pytest.fixture(autouse=True)
def _sync_inline(monkeypatch):
    # Load-bearing (not a no-op): nx/_subagent._sync_inline prefers the module
    # attr _m.NX_SUBAGENT_SYNC_INLINE over the env. nx_fresh's reload()
    # re-executes nx/__init__ in the SAME module dict without defining this
    # name, so setting it here (autouse runs before nx_fresh) survives the
    # reload and pins inline mode. raising=False because the attr is not
    # defined at import time.
    import nx as _nx
    monkeypatch.setattr(_nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    yield

os.environ.setdefault("NX_ENV_FILE", "/dev/null")
# Clear ambient NX_* config so a developer's ~/.env / shell (real endpoints,
# keys, NX_VERIFY_CMD, ...) can't affect results — per repo test guidelines.
_AMBIENT_NX = [k for k in os.environ if k == "NX_ENV_FILE" or k.startswith("NX_")]
for k in _AMBIENT_NX:
    os.environ.pop(k, None)
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ.pop("NX_VERIFY_CMD", None)
os.environ.pop("NX_BASH_PROMPT_SECRETS", None)
for k in ("NX_BASE_URL", "NX_MODEL", "NX_API_KEY", "NX_HOME", "NX_SESSIONS_DIR", "NX_SUBAGENTS"):
    os.environ.pop(k, None)


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
    os.environ["NX_SUBAGENT_ASYNC"] = "0"  # these tests exercise the foreground sub-loop; the 0.18.8 default is background (covered in test_subagent_async.py)
    monkeypatch.setattr(sys, "argv", ["nx"])
    import nx
    # The old `importlib.reload(nx)` rebuilt nx's import-time state from this
    # fixture's env — most importantly the model source. Without a reload,
    # rewire just that: re-discover sources from the env we just set and switch
    # to the first one (binds st.client + st.MODEL). resolve_model() reads
    # NX_MODEL ("test-model"), so no network round-trip happens.
    from nx._sources import _discover_sources, switch_source
    _discover_sources()
    assert switch_source(nx._state.SOURCE_ORDER[0]), "fixture source wiring failed"
    # nx reads these knobs through the MODULE ATTR first, falling back to the
    # env (see _subagent._is_async_default / _sync_inline); the attr holds the
    # import-time default, which the reload used to rebuild from the env.
    monkeypatch.setattr(nx, "NX_SUBAGENT_SYNC_INLINE", 1, raising=False)
    monkeypatch.setattr(nx, "NX_SUBAGENT_ASYNC", 0, raising=False)
    # The reload also wiped nx's staged runtime notice; without it a previous
    # test's note leaks into this test's _drain(). Reset it per-test.
    from nx import _state as st
    monkeypatch.setattr(st, "_PENDING_SUBAGENT_NOTICE", None, raising=False)
    yield nx
    # env rewind is universal (conftest); async-agent state (_ASYNC_AGENTS,
    # quiet slots, TLS) is reset by the universal nx_state_reset autouse fixture.


def _drain(nx):
    """Return (and clear) the pending async-agent notice on st, if any."""
    from nx import _state as st
    note = getattr(st, "_PENDING_SUBAGENT_NOTICE", None)
    st._PENDING_SUBAGENT_NOTICE = None
    return note


def _wait_for(pred, what, deadline_s=30.0):
    """Poll `pred` until it returns a truthy value, up to `deadline_s`.

    These tests only assert that events EVENTUALLY arrive (not latency), so
    the deadline is generous — a failure under load means the mechanism is
    broken, not that 5s was too short. On expiry, raise with the observed
    state so the failure is diagnosable.
    """
    deadline = time.time() + deadline_s
    result = pred()
    while not result and time.time() < deadline:
        time.sleep(0.02)
        result = pred()
    if not result:
        from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
        with _ASYNC_LOCK:
            agents = {k: dict(v) for k, v in _ASYNC_AGENTS.items()}
        from nx import _state as st
        raise AssertionError(
            f"poll timed out after {deadline_s}s waiting for {what}; "
            f"observed state: agents={agents!r} "
            f"pending_notice={getattr(st, '_PENDING_SUBAGENT_NOTICE', None)!r}"
        )
    return result


def test_async_returns_job_id_and_announces_on_later_turn(nx_fresh, monkeypatch):
    from nx import _dispatch
    from nx._subagent import spawn_agent, list_async_agents
    import nx as m

    release = threading.Event()
    started = threading.Event()

    def fake_loop(messages, *a, **kw):
        started.set()
        # the async agent runs the same restricted read-only loop
        out = _dispatch.run_tool("write_file", {"path": "evil.txt", "content": "x"})
        assert "not available to the read-only sub-agent" in out
        release.wait(timeout=5)
        messages.append({"role": "assistant", "content": "findings: X is in nx/_x.py"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(m, "_run_model_turn_loop", fake_loop)
    result = spawn_agent("where is X?", async_=True)
    assert result.startswith("[background sub-agent] job agent-")
    assert "Runtime note" in result  # the parent is told how the answer arrives
    started.wait(timeout=5)  # agent mid-loop
    # job id is in the registry as running
    agents = list_async_agents()
    assert len(agents) == 1 and agents[0]["alive"] and agents[0]["status"] == "running"
    release.set()
    # wait for the worker to finish and stage its note
    note = _wait_for(lambda: _drain(nx_fresh), "agent's staged runtime note")
    assert "findings: X is in nx/_x.py" in note
    assert note.startswith("[Runtime note: background sub-agent agent-")
    # done, announced, never re-staged
    agents = list_async_agents()
    assert not agents[0]["alive"] and agents[0]["status"] == "done"
    assert _drain(nx_fresh) is None  # announced exactly once


def test_run_tool_scan_sweeps_unstaged_completions(nx_fresh, monkeypatch):
    """A finished-but-unstaged agent (worker staged nothing, e.g. between
    finish and flush) is caught by run_tool()'s first-tool-of-turn sweep."""
    from nx import _dispatch
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    with _ASYNC_LOCK:
        _ASYNC_AGENTS["agent-9-deadbeef"] = {
            "instruction": "q", "status": "done", "answer": "the answer",
            "started": time.time(), "announced": False,
        }
    # reload(nx) doesn't reload submodules — a previous test's run_tool call
    # may have left the per-turn flag set; reset it like a fresh turn would.
    from nx import _state as st
    st._BG_SCAN_DONE = False
    out = _dispatch.run_tool("list_dir", {"path": "."})
    assert not out.startswith("ERROR")
    note = _drain(nx_fresh)
    assert note and "the answer" in note
    # swept once: the sweep marks announced, so a later scan doesn't re-stage
    _drain(nx_fresh)
    from nx._subagent import _stage_subagent_completions
    _stage_subagent_completions()
    assert _drain(nx_fresh) is None


def test_flush_appends_subagent_note_to_parent_messages(nx_fresh):
    from nx._compress import _flush_pending_background_notice
    from nx import _state as st
    messages = [{"role": "user", "content": "q"}]
    st._PENDING_SUBAGENT_NOTICE = "[Runtime note: background sub-agent agent-1-aa finished. Its answer:\nfound it]"
    _flush_pending_background_notice(messages)
    assert any(m.get("role") == "user" and "found it" in m.get("content", "")
               for m in messages)
    assert st._PENDING_SUBAGENT_NOTICE is None  # flushed exactly once


def test_failed_agent_still_announces_exactly_once(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent

    def boom(messages, *a, **kw):
        raise RuntimeError("model socket exploded")

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", boom)
    result = spawn_agent("long task", async_=True)
    assert result.startswith("[background sub-agent] job agent-")
    note = _wait_for(lambda: _drain(nx_fresh), "failed agent's runtime note")
    assert "FAILED" in note and "model socket exploded" in note
    assert _drain(nx_fresh) is None  # one note, not two


def test_subagent_loop_never_consumes_parents_notice(nx_fresh, monkeypatch):
    """run_tool() inside a sub-agent loop must not sweep/stage the PARENT's
    pending notices (async agents run turns long after the parent's per-turn
    reset — without the ctx guard an agent thread would steal the note)."""
    from nx import _dispatch
    from nx import _state as st
    from nx._subagent import spawn_agent
    st._BG_SCAN_DONE = False
    st._PENDING_BG_COMPLETION_NOTICE = "[Runtime note: background job(s) you started have finished: pid 1]"
    st._PENDING_SUBAGENT_NOTICE = "[Runtime note: background sub-agent agent-1-aa finished. Its answer:\npayload]"

    def fake_loop(messages, *a, **kw):
        _dispatch.run_tool("list_dir", {"path": "."})  # first tool call of the agent's turn
        messages.append({"role": "assistant", "content": "done"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    assert spawn_agent("q") == "done"
    # both notices are still staged for the parent
    assert st._PENDING_BG_COMPLETION_NOTICE and st._PENDING_SUBAGENT_NOTICE
    st._PENDING_BG_COMPLETION_NOTICE = None
    st._PENDING_SUBAGENT_NOTICE = None


def test_async_string_boolean_and_gating(nx_fresh, monkeypatch):
    from nx._subagent import spawn_agent
    # text-protocol models send booleans as strings — accepted
    out = spawn_agent("q", async_="true")
    assert out.startswith("[background sub-agent] job agent-")
    out2 = spawn_agent("q2", async_="1")  # different instruction: not a duplicate
    assert out2.startswith("[background sub-agent] job agent-")
    # ...but "false"/"0"/None stay synchronous (real model turn, no job id)
    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "sync"})
        return {"status": "done", "turns": 1}
    # monkeypatch, NOT a bare assignment: this used to set nx._run_model_turn_loop
    # directly and rely on the fixture's old importlib.reload(nx) to undo it.
    # Without the reload the fake loop survives the test and every later
    # sub-agent test on this worker returns "sync" (observed: the whole of
    # test_subagent_write.py failing when it shares a worker with this file).
    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    assert spawn_agent("q", async_="false") == "sync"
    assert spawn_agent("q", async_=False) == "sync"
    # no model source: async is refused like sync
    from nx import _state as st
    prev_client, prev_model = st.client, st.MODEL
    st.client, st.MODEL = None, None
    try:
        assert spawn_agent("q", async_=True) == "ERROR: no active model source"
    finally:
        st.client, st.MODEL = prev_client, prev_model


def test_schema_async_key_routes_via_extra_kwargs(nx_fresh, monkeypatch):
    """The schema key is `async` — a Python keyword, so run_tool()'s
    fn(**args) can only deliver it through **extra. It must still trigger
    backgrounding (not silently run the agent synchronously)."""
    from nx import _dispatch
    nx_fresh.NX_SUBAGENTS = 1  # run_tool() refuses a gated spawn (0.28.1); this tests the routing

    def fake_loop(messages, *a, **kw):  # a sync run would reach this
        messages.append({"role": "assistant", "content": "SYNC LEAK"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = _dispatch.run_tool("spawn_agent", {"instruction": "q", "async": True})
    assert out.startswith("[background sub-agent] job agent-"), out
    assert "SYNC LEAK" not in out


def test_async_worker_posture_is_quiet_and_thread_local(nx_fresh, monkeypatch):
    """The worker thread runs quiet (no terminal writes) and installs its own
    thread-local posture: an async agent must not leave YOLO behind, and — since
    0.155.0 — it must not write the process globals either, because the parent
    turn is mid-flight for its whole (minutes-long) run.

    (Pre-0.155.0 this test asserted `st.YOLO is False` DURING the worker loop,
    which the worker's quiet branch satisfied by simply not swapping the
    globals. That assertion is vacuous now: nothing swaps them at all. The pin
    has to look at the WORKER's posture install instead.)"""
    from nx._subagent import spawn_agent, _subagent_quiet
    from nx import _state as st
    nx_fresh.YOLO = False
    seen = {}

    def fake_loop(messages, *a, **kw):
        seen["quiet"] = _subagent_quiet()
        # The worker's own view: quiet agents INHERIT (nothing installed), so
        # `_posture()` falls back to the globals it was spawned with...
        seen["posture"] = st._posture()
        # ...and NOTHING is installed on the worker's thread.
        tls = getattr(st, "_POSTURE_TLS", None)
        seen["tls"] = {} if tls is None else dict(vars(tls))
        messages.append({"role": "assistant", "content": "ok"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q", async_=True)
    assert out.startswith("[background sub-agent] job agent-")
    _wait_for(lambda: "quiet" in seen, "worker loop to record quiet/posture state")
    assert seen.get("quiet") is True            # worker runs quiet
    assert seen.get("posture") == (False, None)  # inherits the parent's posture
    assert seen.get("tls") == {}                # nothing installed on the worker
    _wait_for(lambda: _drain(nx_fresh) is not None, "worker's staged runtime note")
    assert (st.YOLO, st.APPROVE_LEVEL) == (False, None)  # globals untouched
    assert dict(vars(st._POSTURE_TLS)) == {}

    # READ-ONLY, foreground: this is the shape that FORCES a posture. It must
    # install it on ITS OWN thread and nowhere else. (Pre-0.155.0 the same spawn
    # wrote st.YOLO=True process-wide for the whole run.)
    st.YOLO = False
    st.APPROVE_LEVEL = "medium"
    forced = {}

    def forced_loop(messages, *a, **kw):
        forced["posture"] = st._posture()
        forced["globals"] = (st.YOLO, st.APPROVE_LEVEL)
        messages.append({"role": "assistant", "content": "ro"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", forced_loop)
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_ASYNC", 0)
    assert spawn_agent("q") == "ro"
    assert forced["posture"] == (True, "high"), forced
    assert forced["globals"] == (False, "medium"), (
        "a read-only spawn wrote the process globals: " + repr(forced["globals"]))
    assert (st.YOLO, st.APPROVE_LEVEL) == (False, "medium")
    assert dict(vars(st._POSTURE_TLS)) == {}


def test_async_readonly_agent_leaves_the_parents_posture_alone(nx_fresh,
                                                              monkeypatch):
    """§5.3 — while a read-only async agent runs, the PARENT thread's posture
    reads are the parent's. Pre-0.155.0 the worker wrote the process globals
    BEFORE installing anything else, so `st.YOLO` read True (and
    `_confirm()`'s YOLO fast path auto-allowed) for the agent's whole run."""
    import threading

    from nx import _state as st
    from nx._subagent import spawn_agent

    monkeypatch.setattr(st, "YOLO", False)
    monkeypatch.setattr(st, "APPROVE_LEVEL", "medium")
    entered = threading.Event()
    release = threading.Event()
    seen = {}

    def fake_loop(messages, *a, **kw):
        # The worker runs QUIET, so it installs nothing: its posture read is the
        # globals — still the parent's values, which is the inherit contract.
        seen["worker_posture"] = st._posture()
        entered.set()
        assert release.wait(30), "release never signalled"
        messages.append({"role": "assistant", "content": "ok"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("q", async_=True)
    assert out.startswith("[background sub-agent] job agent-")
    try:
        assert entered.wait(30), "the worker never entered the loop"
        # Parent-side reads while the agent is mid-run.
        assert (st.YOLO, st.APPROVE_LEVEL) == (False, "medium"), (
            "a background agent moved the process globals")
        assert st._posture() == (False, "medium"), st._posture()
    finally:
        release.set()
    _wait_for(lambda: _drain(nx_fresh) is not None, "worker's staged runtime note")
    assert seen["worker_posture"] == (False, "medium")
    assert (st.YOLO, st.APPROVE_LEVEL) == (False, "medium")


def test_quiet_run_tool_prints_nothing(nx_fresh, monkeypatch, capsys):
    """Inside a quiet loop run_tool dispatches normally but draws no tool
    box — the terminal belongs to the parent while the agent runs."""
    from nx import _dispatch
    from nx._subagent import _ASYNC_QUIET_SLOTS
    _ASYNC_QUIET_SLOTS.quiet = True
    try:
        out = _dispatch.run_tool("list_dir", {"path": "."})
        assert not out.startswith("ERROR")
        captured = capsys.readouterr()
        assert captured.out == ""
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
    # sanity: the same call outside quiet mode DOES draw the box (clear the
    # read cache first — a cache hit returns before the box is drawn)
    from nx import _state as st
    st._TOOL_RESULT_CACHE.clear()
    _dispatch.run_tool("list_dir", {"path": "."})
    assert "╭─" in capsys.readouterr().out


def test_quiet_confirm_never_prompts(nx_fresh, monkeypatch):
    """_confirm short-circuits on a quiet thread — it never prompts and never
    reaches the risk classifier. The verdict still honors the approval
    posture: quiet + yolo approves (a detached agent's bash tier is
    allowlist-gated anyway), quiet + no yolo denies (batch is headless:
    nothing may approve a write the user never opted into)."""
    from nx._approval import _confirm
    from nx._subagent import _ASYNC_QUIET_SLOTS

    def _never(*a, **kw):  # the risk classifier would hang on a network call
        raise AssertionError("quiet loop must not reach the risk classifier")

    monkeypatch.setattr(nx_fresh, "_assess_risk", _never)
    _ASYNC_QUIET_SLOTS.quiet = True
    try:
        nx_fresh.YOLO = True
        assert _confirm("write: x") is True  # explicit yolo: approve, silently
        nx_fresh.YOLO = False
        assert _confirm("write: x") is False  # no yolo: headless = deny
        nx_fresh.YOLO = True
    finally:
        _ASYNC_QUIET_SLOTS.quiet = False
        nx_fresh.YOLO = False


def test_bg_after_async_finish_shows_done_status(nx_fresh, monkeypatch, capsys):
    from nx._repl_settings_cmds import _cmd_bg
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "the answer"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    spawn_agent("count things", async_=True)
    _wait_for(lambda: _drain(nx_fresh) is not None, "agent's staged runtime note")
    _cmd_bg("/bg", None)
    out = capsys.readouterr().out
    assert "done" in out and "agent-" in out and "count things" in out


def test_bg_lists_async_agents(nx_fresh, monkeypatch, capsys):
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    with _ASYNC_LOCK:
        _ASYNC_AGENTS["agent-1-aaaaaa"] = {
            "instruction": "investigate the flaky test", "status": "running",
            "answer": None, "started": time.time(), "announced": False,
        }
        _ASYNC_AGENTS["agent-2-bbbbbb"] = {
            "instruction": "count the tests", "status": "done",
            "answer": "42 tests", "started": time.time() - 100, "announced": True,
        }
    from nx._repl_settings_cmds import _cmd_bg
    _cmd_bg("/bg", None)
    out = capsys.readouterr().out
    assert "background sub-agents" in out
    assert "agent-2-bbbbbb" in out and "agent-1-aaaaaa" in out
    assert "investigate the flaky test" in out
    assert "newest first" in out


def test_concurrent_async_agents_merge_into_one_note(nx_fresh, monkeypatch):
    """Two agents finishing before the next flush announce together (one
    merged note), each exactly once."""
    from nx._subagent import spawn_agent
    gate = threading.Barrier(3, timeout=30)  # parent + 2 agents

    def fake_loop(messages, *a, **kw):
        content = messages[1]["content"]
        gate.wait(timeout=5)
        messages.append({"role": "assistant", "content": "ans:" + content})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    spawn_agent("task A", async_=True)
    spawn_agent("task B", async_=True)
    gate.wait(timeout=30)  # both agents are past their turn loops
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK

    def _both_announced() -> bool:
        # NON-destructive poll: _drain() CLEARS the slot, so polling through it
        # could consume agent A's half-note before B merges into the same slot,
        # leaving the merged note unobservable. Staging the note and setting
        # `announced` happen under one _ASYNC_LOCK hold, so "every record
        # announced" implies the merged note is complete in the slot (records
        # are only evicted after that hold ends).
        with _ASYNC_LOCK:
            flags = [rec.get("announced") for rec in _ASYNC_AGENTS.values()]
        return bool(flags) and all(flags)  # bool(): empty dict is not "announced"

    _wait_for(_both_announced, "both agents announced")
    note = _drain(nx_fresh)  # one destructive read, after both announced
    assert note and "task A" in note and "task B" in note, note
    assert "ans:task A" in note and "ans:task B" in note
    assert note.count("[Runtime note: background sub-agent") == 2


# --- 0.18.8: background by default + report files -----------------------------

def test_spawn_defaults_to_background(nx_fresh, monkeypatch):
    """With no `async` argument at all, a spawn goes to the background
    (NX_SUBAGENT_ASYNC defaults to 1): job id now, [Runtime note] later."""
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "findings"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    # the fixture pins the knob to 0; restore the SHIPPED default for this test
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_ASYNC", 1)
    out = spawn_agent("what changed?")  # no async kwarg
    assert out.startswith("[background sub-agent] job agent-")
    note = _wait_for(lambda: _drain(nx_fresh), "agent's staged runtime note")
    assert "findings" in note


def test_explicit_async_false_stays_foreground(nx_fresh, monkeypatch):
    """The documented foreground escape hatch survives the new default: an
    explicit async=false runs synchronously even with the knob at 1."""
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "sync answer"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    assert spawn_agent("q", async_=False) == "sync answer"
    assert spawn_agent("q", async_="false") == "sync answer"
    assert _drain(nx_fresh) is None  # nothing was backgrounded


def test_async_default_respects_nx_subagent_async_0(nx_fresh, monkeypatch):
    """NX_SUBAGENT_ASYNC=0 restores the foreground default: an omitted `async`
    runs synchronously (explicit true still backgrounds)."""
    from nx._subagent import spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "fg"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    monkeypatch.setattr(nx_fresh, "NX_SUBAGENT_ASYNC", 0)
    assert spawn_agent("q") == "fg"  # omitted -> knob -> foreground
    out = spawn_agent("bg q", async_=True)  # explicit true still wins
    assert out.startswith("[background sub-agent] job agent-")


def test_finished_agent_writes_report_and_note_carries_path(nx_fresh, monkeypatch):
    """The worker writes SESSION_HOME/agents/<job>.md (full answer) and the
    runtime note carries a short head + the report path instead of the whole
    answer."""
    from nx._subagent import _agents_report_dir, spawn_agent
    long_answer = "HEAD-PART. " + ("detail " * 400)  # > _ASYNC_REPORT_CHARS

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": long_answer})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("big investigation", async_=True)
    jid = out.split("job ")[1].split("\n")[0]
    note = _wait_for(lambda: _drain(nx_fresh), "report-note for the long answer")
    assert long_answer.strip() not in note  # full answer is NOT in context
    assert note.count(long_answer[:20]) <= 1  # head appears at most once
    assert "Full report: " in note
    report = os.path.join(_agents_report_dir(), jid + ".md")
    assert report in note  # the note names the report path
    body = open(report, encoding="utf-8").read()
    assert "# Sub-agent report" in body
    assert "- status: done" in body
    assert "## Instruction" in body and "big investigation" in body
    assert "detail detail" in body  # the FULL answer is on disk


def test_partial_agent_is_not_announced_as_finished(nx_fresh, monkeypatch, capsys):
    """0.103.9 regression: a turn-capped async agent whose report is only an
    interim note used to be announced as "finished" with status done — the
    parent trusted it. It must now be announced as PARTIAL (status: partial in
    the report file, "stopped early" in the [Runtime note], and "partial" in
    /bg)."""
    from nx._subagent import _agents_report_dir, spawn_agent

    def fake_loop(messages, *a, **kw):
        messages.append({"role": "assistant", "content": "still exploring the schema"})
        return {"status": "cap", "turns": 100}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("sweep the repo", async_=True)
    jid = out.split("job ")[1].split("\n")[0]
    note = _wait_for(lambda: _drain(nx_fresh), "partial agent's runtime note")
    assert "PARTIAL" in note and "stopped early" in note
    assert "finished. Its answer" not in note
    report = os.path.join(_agents_report_dir(), jid + ".md")
    body = open(report, encoding="utf-8").read()
    assert "- status: partial" in body
    # /bg renders partial distinctly from done/error.
    from nx._repl_settings_cmds import _cmd_bg
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    with _ASYNC_LOCK:
        _ASYNC_AGENTS["agent-1-partial"] = {
            "instruction": "sweep the repo", "status": "partial",
            "answer": "still exploring", "started": time.time(),
            "announced": True, "report": "/tmp/fake/agents/agent-1-partial.md",
        }
    _cmd_bg("/bg", None)
    bg_out = capsys.readouterr().out
    assert "partial" in bg_out


def test_failed_agent_writes_error_report(nx_fresh, monkeypatch):
    """A failed agent's report is written too (status: error) — it is what the
    parent reads to decide whether an inline retry is worth it."""
    from nx._subagent import _agents_report_dir, spawn_agent

    def boom(messages, *a, **kw):
        raise RuntimeError("model socket exploded")

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", boom)
    out = spawn_agent("doomed task", async_=True)
    jid = out.split("job ")[1].split("\n")[0]
    note = _wait_for(lambda: _drain(nx_fresh), "failed agent's runtime note")
    assert "FAILED" in note
    report = os.path.join(_agents_report_dir(), jid + ".md")
    body = open(report, encoding="utf-8").read()
    assert "- status: error" in body
    assert "model socket exploded" in body


def test_bg_lists_report_path(nx_fresh, monkeypatch, capsys):
    """/bg shows a finished agent's report path (and omits it while running)."""
    from nx._repl_settings_cmds import _cmd_bg
    from nx._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    with _ASYNC_LOCK:
        _ASYNC_AGENTS["agent-1-cccccc"] = {
            "instruction": "explore", "status": "done", "answer": "found",
            "started": time.time(), "announced": True,
            "report": "/tmp/fake/agents/agent-1-cccccc.md",
        }
        _ASYNC_AGENTS["agent-2-dddddd"] = {
            "instruction": "still running", "status": "running",
            "answer": None, "started": time.time() - 100, "announced": False,
        }
    _cmd_bg("/bg", None)
    out = capsys.readouterr().out
    assert "report: /tmp/fake/agents/agent-1-cccccc.md" in out
    assert out.count("report:") == 1  # the running agent has no report yet
def test_async_agent_writes_intent_at_spawn_and_clears_at_finalize(nx_fresh, monkeypatch):
    """0.147.2 effect sandwich, steps 1+3: an intent record exists while the
    worker is mid-flight, and is gone once the result is durably persisted."""
    import json
    from nx._subagent import _agents_report_dir, spawn_agent

    def fake_loop(messages, *a, **kw):
        ipath = os.path.join(_agents_report_dir(), jid + ".intent.json")
        assert os.path.exists(ipath)  # worker mid-flight: intent exists NOW
        rec = json.load(open(ipath, encoding="utf-8"))
        assert rec["instruction"] == "crash probe"
        assert rec["tier"] == "read"
        assert rec["owner_pid"] == os.getpid()
        messages.append({"role": "assistant", "content": "done"})
        return {"status": "done", "turns": 1}

    monkeypatch.setattr(nx_fresh, "_run_model_turn_loop", fake_loop)
    out = spawn_agent("crash probe", async_=True)
    jid = out.split("job ")[1].split("\n")[0]
    _wait_for(lambda: _drain(nx_fresh), "agent's runtime note")
    # Finalized: the result is on disk, the intent is retired.
    assert not os.path.exists(os.path.join(_agents_report_dir(),
                                           jid + ".intent.json"))
    assert os.path.exists(os.path.join(_agents_report_dir(), jid + ".md"))


def test_orphan_intent_scan_announces_dead_owner_once(nx_fresh):
    """The sweep finds an intent whose owner PID is dead, announces it with
    the instruction, deletes the file, and is silent on a second pass."""
    import json
    from nx._subagent import (_agents_report_dir, _scan_orphan_agent_intents,
                              _write_agent_intent)
    dead_pid = 2**22 - 7  # above Linux's default pid_max (4194304)
    rec = {"job_id": "agent-orphan1", "instruction": "investigate the flake",
           "tier": "read", "owner_pid": dead_pid,
           "owner_pid_start_time": None, "started": "2026-10-11 12:00:00"}
    os.makedirs(_agents_report_dir(), exist_ok=True)
    path = os.path.join(_agents_report_dir(), "agent-orphan1.intent.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    lines = _scan_orphan_agent_intents()
    assert len(lines) == 1
    assert "agent-orphan1" in lines[0]
    assert "investigate the flake" in lines[0]
    assert not os.path.exists(path)  # announced once, then removed
    assert _scan_orphan_agent_intents() == []  # silent on the second pass
    # A live owner (this process) is never announced.
    _write_agent_intent("agent-live1", "still running elsewhere", "read")
    assert _scan_orphan_agent_intents() == []
    os.remove(os.path.join(_agents_report_dir(), "agent-live1.intent.json"))
