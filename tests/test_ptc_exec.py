"""PTC Phase 2 (docs/ptc-migration-spec.md): the run_tools execution engine.

Tier A: nx is imported ONCE and never reloaded; state writes go through
monkeypatch. The subprocess harness is exercised for real (the launcher +
pipe protocol are the deliverable), but tool dispatch is faked by
monkeypatching nx._dispatch.run_tool — no write tool is ever dispatched, and
the one test that does use the real read_file body proves the cache interplay
against st._TOOL_RESULT_CACHE.

Coverage (spec §Phase 2 tests + the turn-layer semantics):
  1. Allowlist == _CACHEABLE_READ_TOOLS ∩ PLAN_READONLY_TOOLS (Audit
     correction #2 — spawn_agent/todo_write/plan_edit NOT callable).
  2. A script calling write_file is refused over the pipe; the fake dispatch
     never sees the name; no file is written.
  3. Breaker parity: a script-level exception is ONE error result for
     run_tools + exactly one malformed-args breaker increment.
  4. Esc/_EscToChat kills the child's process group (killpg) and the child is
     reaped (in-flight captures skip-filled before the kill; nothing hangs).
  5. Timeout: NX_PTC_TIMEOUT-derived cap, TimeoutExpired path kills the group
     and folds one error result (+ breaker increment).
  6. Cache reuse: two identical stub reads inside one script populate exactly
     one st._TOOL_RESULT_CACHE entry (real read_file body).
  7. Prompt surface: static PTC note gated on NX_PTC in prefix_notes(); stub
     source staged ONCE as a runtime-note tail (never the pinned prefix);
     one-time nudge constant shape.
"""
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
import nx._state as st  # noqa: E402
from nx import _dispatch, _ptc_exec  # noqa: E402
from nx._approval import _EscToChat  # noqa: E402
from nx._breaker import _breaker_views  # noqa: E402
from nx._session_preamble import PTC_FANOUT_GUIDANCE  # noqa: E402


@pytest.fixture()
def ptc_on(monkeypatch):
    """NX_PTC on + clean per-test PTC state flags."""
    monkeypatch.setattr(m, "NX_PTC", 1)
    monkeypatch.setattr(st, "_PTC_NUDGED", False)
    monkeypatch.setattr(st, "_PTC_SOURCE_SHIPPED", False)
    monkeypatch.setattr(st, "_PTC_BREAKER_TRIPPED", 0)
    return monkeypatch


@pytest.fixture()
def fake_dispatch(monkeypatch):
    """Subprocess-harness fake: dispatch records (name, args) and answers with
    a marker string. No real tool body ever runs through the pipe."""
    calls: list[tuple[str, dict]] = []

    def _fake(name, args):
        calls.append((name, dict(args)))
        return f"FAKE[{args.get('path', name)}]"

    monkeypatch.setattr(_dispatch, "run_tool", _fake)
    return calls


# --- 1. allowlist --------------------------------------------------------------

def test_allowlist_is_the_intersection_not_the_union():
    from nx._dispatch import _CACHEABLE_READ_TOOLS
    from nx._plan import PLAN_READONLY_TOOLS
    allow = _ptc_exec.ptc_allowlist()
    assert allow == frozenset(_CACHEABLE_READ_TOOLS) & frozenset(PLAN_READONLY_TOOLS)
    # Audit correction #2: the plan set alone would expose these to scripts.
    for banned in ("spawn_agent", "todo_write", "plan_edit", "read_image",
                   "wait_background", "write_file", "edit_file", "run_bash"):
        assert banned not in allow
    # The pure reads stay.
    assert {"read_file", "search_files", "list_dir"} <= allow


# --- 2. write refusal over the pipe (fake harness) ------------------------------

def test_script_write_call_refused_never_dispatched(ptc_on, fake_dispatch, tmp_path,
                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    res = _ptc_exec.run_tools(
        "import stubs\n"
        "print('R:' + stubs.read_file(path='a.txt'))\n"
        "w = stubs.write_file(path='b.txt', content='nope')\n"
        "print('W:' + w)\n"
        "s = stubs.todo_write(todos=[{'text': 'x', 'status': 'pending'}])\n"
        "print('S:' + s)\n", timeout=60)
    assert "FAKE[a.txt]" in res
    # Refused names never reach dispatch at all (allowlist, before run_tool).
    assert [n for n, _ in fake_dispatch] == ["read_file"]
    assert "ERROR:" in res and "write_file is not callable" in res
    assert "todo_write is not callable" in res
    assert "2 refused/failed" in res
    assert not (tmp_path / "b.txt").exists()


# --- 3. breaker parity -----------------------------------------------------------

def test_script_exception_folds_one_error_and_increments_breaker(ptc_on, monkeypatch,
                                                                 tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(st, "MALFORMED_TOOL_BREAKER_LIMIT", 3)
    failures_before = _breaker_views()[0].get("run_tools", 0)
    res = _ptc_exec.run_tools("x = 1/0\nprint('never')", timeout=30)
    assert res.startswith("ERROR")
    assert "ZeroDivisionError" in res
    assert "never" not in res  # the script's post-crash print never ran
    failures_after = _breaker_views()[0].get("run_tools", 0)
    assert failures_after == failures_before + 1
    # Success re-dispatch decays (JSON-mode parity: one clean call clears the
    # streak) — go through the real run_tool so the SAME decay path the JSON
    # turn uses runs, proving the increment landed on the shared streak.
    res2 = _dispatch.run_tool("run_tools", {"code": "print('ok')", "timeout": 30})
    assert res2.startswith("[run_tools:")
    assert _breaker_views()[0].get("run_tools", 0) < failures_after


# --- 4. Esc kill + skip-fill -------------------------------------------------------

def test_esc_kills_process_group_and_child_is_reaped(ptc_on, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = {"done": False}

    def _fake(name, args):
        if not first["done"]:
            first["done"] = True
            raise _EscToChat("read_file")
        return "FAKE"

    monkeypatch.setattr(_dispatch, "run_tool", _fake)
    killpgs: list[tuple[int, int]] = []
    real_killpg = os.killpg

    def _spy_killpg(pid, sig):
        killpgs.append((pid, sig))
        return real_killpg(pid, sig)

    monkeypatch.setattr(os, "killpg", _spy_killpg)
    with pytest.raises(_EscToChat):
        _ptc_exec.run_tools(
            "import stubs\n"
            "print(stubs.read_file(path='x.txt'))\n"
            "print(stubs.read_file(path='y.txt'))\n", timeout=30)
    # The group was SIGTERM'd (start_new_session => pgid == child pid) and the
    # child reaped — no orphan process group outlives the turn. The in-flight
    # capture was skip-filled with the cancel error BEFORE the kill (the
    # launcher unblocks, so the kill lands on an exiting child — nothing
    # hangs; the hang would show up as the 2s SIGTERM-grace + SIGKILL path).
    assert killpgs and killpgs[0][1] == signal.SIGTERM
    pid = killpgs[0][0]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert st._ACTIVE_PTC_PROC is None


# --- 5. timeout ---------------------------------------------------------------------

def test_timeout_kills_group_and_folds_error(ptc_on, fake_dispatch, monkeypatch,
                                             tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(st, "MALFORMED_TOOL_BREAKER_LIMIT", 3)
    killpgs: list[int] = []
    real_killpg = os.killpg
    monkeypatch.setattr(os, "killpg",
                        lambda pid, sig: (killpgs.append(pid), real_killpg(pid, sig))[1])
    before = _breaker_views()[0].get("run_tools", 0)
    res = _ptc_exec.run_tools("import time\ntime.sleep(120)\nprint('late')",
                              timeout=1)
    assert res.startswith("ERROR")
    assert "timed out after 1s" in res
    assert killpgs  # process group killed
    # A timeout is NOT a malformed-args breaker event (correct-but-slow script
    # must never withdraw run_tools via the fallback ladder).
    assert _breaker_views()[0].get("run_tools", 0) == before
    assert st._ACTIVE_PTC_PROC is None


def test_timeout_env_default(ptc_on, fake_dispatch, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NX_PTC_TIMEOUT", "1")
    res = _ptc_exec.run_tools("import time\ntime.sleep(60)", timeout=0)
    assert "timed out after 1s" in res


# --- 6. cache interplay (real read body) ----------------------------------------------

def test_repeated_stub_read_reuses_tool_result_cache(ptc_on, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    st._TOOL_RESULT_CACHE.clear()
    try:
        res = _ptc_exec.run_tools(
            "import stubs\n"
            "a = stubs.read_file(path=%r)\n"
            "b = stubs.read_file(path=%r)\n"
            "print('SAME' if a == b else 'DIFF')\n" % (str(target), str(target)),
            timeout=60)
        assert "SAME" in res
        keys = [k for k in st._TOOL_RESULT_CACHE if k[0] == "read_file"]
        assert len(keys) == 1  # one entry served twice, not two misses
    finally:
        st._TOOL_RESULT_CACHE.clear()


# --- 7. prompt surface -----------------------------------------------------------

def test_ptc_note_gated_in_prefix(ptc_on):
    from nx._prefix import prefix_notes
    notes = prefix_notes()
    assert any(n["content"].startswith("[PTC note:") for n in notes)


def test_ptc_note_absent_when_gate_closed(monkeypatch):
    monkeypatch.setattr(m, "NX_PTC", 0)
    from nx._prefix import prefix_notes
    notes = prefix_notes()
    assert not any(n["content"].startswith("[PTC note:") for n in notes)
    assert _ptc_exec._load_ptc_note() is None


def test_stub_source_shipped_once_as_runtime_note_tail(ptc_on, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Isolate the SOURCE staging from the fan-out nudge (same slot, separate
    # behavior — the nudge has its own test below).
    monkeypatch.setattr(st, "_PTC_NUDGED", True)
    st._READ_STREAK = 0
    monkeypatch.setitem(st._NOTICE_DEFAULTS, "_PENDING_PTC_NOTICE", None)
    st.__dict__.pop("_PENDING_PTC_NOTICE", None)
    # The first dispatched tool call ships the source as a note...
    _dispatch.run_tool("read_file", {"path": str(tmp_path / "nope.txt")})
    note = getattr(st, "_PENDING_PTC_NOTICE", None)
    assert note and note.startswith("[Runtime note:")
    assert "def read_file" in note and "def _capture" in note
    # ...exactly once (the source is dynamic; it must ride the append-only
    # tail, never the pinned prefix, and never twice).
    st._PENDING_PTC_NOTICE = None
    _dispatch.run_tool("read_file", {"path": str(tmp_path / "nope2.txt")})
    assert getattr(st, "_PENDING_PTC_NOTICE", None) is None
    from nx._prefix import is_prefix_note
    assert not is_prefix_note({"role": "user", "content": note})
    st._READ_STREAK = 0


def test_fanout_nudge_fires_once_with_reads(ptc_on, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Real run_tool here (the nudge lives inside it); source staging is marked
    # shipped so the two notes sharing one slot cannot cross-contaminate.
    monkeypatch.setattr(st, "_PTC_SOURCE_SHIPPED", True)
    st._READ_STREAK = 0
    st._DELEGATION_NUDGE_COUNT = 0
    monkeypatch.setattr(st, "_PENDING_PTC_NOTICE", None, raising=False)
    try:
        _dispatch.run_tool("read_file", {"path": "a"})
        assert getattr(st, "_PENDING_PTC_NOTICE", None) is None
        _dispatch.run_tool("search_files", {"pattern": "x"})
        note = getattr(st, "_PENDING_PTC_NOTICE", None)
        assert note and PTC_FANOUT_GUIDANCE in note
        assert "≥2 independent READS" in note
        st._PENDING_PTC_NOTICE = None
        _dispatch.run_tool("read_file", {"path": "b"})
        assert getattr(st, "_PENDING_PTC_NOTICE", None) is None  # one-time
    finally:
        st._READ_STREAK = 0


def test_nudge_absent_when_gate_closed(monkeypatch):
    monkeypatch.setattr(m, "NX_PTC", 0)
    monkeypatch.setattr(st, "_PTC_NUDGED", False)
    st._READ_STREAK = 0
    monkeypatch.setattr(st, "_PENDING_PTC_NOTICE", None, raising=False)
    try:
        _dispatch.run_tool("read_file", {"path": "a"})
        _dispatch.run_tool("search_files", {"pattern": "x"})
        assert getattr(st, "_PENDING_PTC_NOTICE", None) is None
    finally:
        st._READ_STREAK = 0


# --- engine plumbing details ------------------------------------------------------

def test_gate_and_refusal_registered():
    from nx._tool_gates import _GATE_REFUSALS, _TOOL_GATES
    assert "run_tools" in _TOOL_GATES and "run_tools" in _GATE_REFUSALS
    assert "NX_PTC is off" in _GATE_REFUSALS["run_tools"]
    assert "run_tools" in {t["function"]["name"] for t in m.TOOLS}


def test_run_tools_not_recursed_into_stubs():
    from nx import _ptc_stubs
    assert "run_tools" in _ptc_stubs.EXCLUDED_TOOLS


def test_empty_code_and_bad_timeout_folds(ptc_on, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert "empty" in _ptc_exec.run_tools("", timeout=10)
    assert "empty" in _ptc_exec.run_tools("   \n", timeout=10)


def test_per_call_failure_returns_error_string_not_raise(ptc_on, monkeypatch,
                                                         tmp_path):
    """A refused/broken per-call failure must NOT kill the script (JSON-mode
    parity: an ERROR tool result doesn't kill the turn either)."""
    monkeypatch.chdir(tmp_path)

    def _fake(name, args):
        return "ERROR: simulated per-call failure"

    monkeypatch.setattr(_dispatch, "run_tool", _fake)
    res = _ptc_exec.run_tools(
        "import stubs\n"
        "r = stubs.read_file(path='a')\n"
        "print('AFTER' if r.startswith('ERROR') else 'RAISED')\n", timeout=30)
    assert "AFTER" in res  # script kept running after the failed call
    assert res.startswith("[run_tools:")


# --- Phase 3: batch fan-out, launcher reader thread, telemetry ------------------

def test_dispatch_pipe_batch_concurrent_and_id_ordered():
    """Batch of 2 with staggered sleeps: replies arrive in id order AND the
    fast call does not wait for the slow one (real concurrency)."""
    import time as _t
    started: list[float] = []
    finished: dict[str, float] = {}

    def _fake(name, args):
        started.append(_t.monotonic())
        _t.sleep({"read_file": 0.3, "search_files": 0.0}[name])
        finished[args["cid"]] = _t.monotonic()
        return f"FAKE[{name}]"

    calls: list[dict] = []
    orig = _dispatch.run_tool
    _dispatch.run_tool = _fake
    try:
        _ptc_exec._dispatch_pipe_batch(
            [(1, "read_file", {"cid": "slow"}),
             (2, "search_files", {"cid": "fast"})],
            lambda payload: calls.append(payload))
    finally:
        _dispatch.run_tool = orig
    assert [c["id"] for c in calls] == [1, 2]
    assert calls[0]["result"] == "FAKE[read_file]"
    assert calls[1]["result"] == "FAKE[search_files]"
    # call 2 (0.0s sleep) finished well before call 1 (0.3s) would have allowed
    assert finished["fast"] - started[0] < 0.25
    assert finished["slow"] - started[0] >= 0.25


def test_dispatch_pipe_batch_fast_path_no_pool(monkeypatch):
    """A 1-frame batch stays serial: zero ThreadPoolExecutor instantiations."""
    created: list = []

    class _SpyExecutor(_ptc_exec.ThreadPoolExecutor):
        def __init__(self, *a, **k):
            created.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(_ptc_exec, "ThreadPoolExecutor", _SpyExecutor)
    orig = _dispatch.run_tool
    _dispatch.run_tool = lambda name, args: f"FAKE[{name}]"
    got: list[dict] = []
    try:
        _ptc_exec._dispatch_pipe_batch([(7, "read_file", {})],
                                       lambda p: got.append(p))
    finally:
        _dispatch.run_tool = orig
    assert got == [{"id": 7, "result": "FAKE[read_file]"}]
    assert not created


def test_e2e_threaded_script_two_stub_reads(ptc_on, fake_dispatch, tmp_path,
                                            monkeypatch):
    """A child script calling stubs from 2 concurrent threads gets both
    results through the launcher's reader-thread pipe."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
    res = _ptc_exec.run_tools(
        "import stubs, threading\n"
        "out = {}\n"
        "def one(): out['a'] = stubs.read_file(path='a.txt')\n"
        "def two(): out['b'] = stubs.read_file(path='b.txt')\n"
        "t1, t2 = threading.Thread(target=one), threading.Thread(target=two)\n"
        "t1.start(); t2.start(); t1.join(); t2.join()\n"
        "print(out['a'] + '|' + out['b'])\n", timeout=60)
    # both threads got their own (correct) result through the pipe
    assert "FAKE[a.txt]|FAKE[b.txt]" in res
    assert len(fake_dispatch) == 2


def test_fanout_scripts_counter_and_footer_field(ptc_on, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    start = st._PTC_FANOUT_SCRIPTS
    _ptc_exec.run_tools(
        "import stubs\n"
        "print(stubs.read_file(path='a.txt'), stubs.read_file(path='b.txt'))\n",
        timeout=30)
    assert st._PTC_FANOUT_SCRIPTS == start + 1
    # a 1-call script must NOT increment
    _ptc_exec.run_tools("import stubs\nprint(stubs.read_file(path='a.txt'))\n",
                        timeout=30)
    assert st._PTC_FANOUT_SCRIPTS == start + 1
    # footer field
    monkeypatch.setattr(st, "_PTC_TURN_TOOL_CALLS", 1)
    monkeypatch.setattr(st, "_PTC_FANOUT_SCRIPTS", 1)
    from nx import _turn_footer
    assert "script-fan-out x1" in _turn_footer._ptc_baseline_field()


def test_cacheable_tuple_is_one_source_of_truth():
    """Audit correction #4: prefetch/spec_reads import _dispatch's tuple —
    recall_lessons included, identity shared, and maybe_speculate dispatches."""
    from nx import _spec_reads, _tool_prefetch
    assert (_tool_prefetch._CACHEABLE_READ_TOOLS
            is _dispatch._CACHEABLE_READ_TOOLS)
    assert "recall_lessons" in _tool_prefetch._CACHEABLE_READ_TOOLS
    assert "recall_lessons" in _ptc_exec.ptc_allowlist()
    monkey_called: list = []
    orig = _dispatch.run_tool
    _dispatch.run_tool = lambda name, args: (
        monkey_called.append(name) or "LESSONS")
    try:
        _spec_reads.maybe_speculate("recall_lessons", '{"query": "x"}')
    finally:
        _dispatch.run_tool = orig
    assert monkey_called == ["recall_lessons"]


# --- 8. hardening (0.165.3): batch deadline + seccomp net-deny ------------------

def test_batch_deadline_fires_on_hung_dispatch(ptc_on, fake_dispatch, tmp_path,
                                               monkeypatch):
    """The reviewer's finding #2: the deadline was only checked BETWEEN
    batches — a hung run_tool body inside _dispatch_pipe_batch blocked past
    it. With the future-based batch the wait is bounded and the script folds
    into the timeout error result."""
    monkeypatch.chdir(tmp_path)
    calls: list = []

    def hung_run_tool(name, args):
        calls.append(name)
        import time as _t
        _t.sleep(60)  # far past the 3s budget
        return "NEVER"

    monkeypatch.setattr(_dispatch, "run_tool", hung_run_tool)
    res = _ptc_exec.run_tools(
        "import stubs\nprint(stubs.search_files(pattern='x'))\n", timeout=3)
    assert "timed out after 3s" in res
    assert calls  # the hung dispatch was actually entered
    # and the harness returned promptly, not after 60s


def test_seccomp_blocks_child_network(ptc_on, fake_dispatch, tmp_path,
                                      monkeypatch):
    """Reviewer's finding #1 mitigation: the child gets _ptc_preexec —
    seccomp denies the socket family (EACCES) and prctl sets NO_NEW_PRIVS.
    The pipe (inherited fds) is unaffected. Skips when libseccomp is absent
    (macOS, minimal containers) — the preexec is best-effort by design."""
    import ctypes.util
    if not ctypes.util.find_library("seccomp"):
        pytest.skip("libseccomp not available")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    res = _ptc_exec.run_tools(
        "import socket\n"
        "try:\n"
        "    s = socket.socket()\n"
        "    s.connect(('93.184.216.34', 80))\n"
        "    print('NET FAIL')\n"
        "except OSError as e:\n"
        "    print('blocked', e.errno)\n"
        "import stubs\n"
        "print(stubs.read_file(path='a.txt'))\n", timeout=30)
    assert "blocked 13" in res  # EACCES from the seccomp filter
    assert "FAKE[a.txt]" in res  # pipe protocol unaffected


def test_env_scrubbed_in_child(ptc_on, fake_dispatch, tmp_path, monkeypatch):
    """0.165.2's scrub, pinned: no canary secret reaches the child env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PTC_CANARY_SECRET", "supersecret123")
    res = _ptc_exec.run_tools(
        "import os\n"
        "print('LEAK' if 'PTC_CANARY_SECRET' in os.environ else 'clean')\n"
        "import stubs\nprint(stubs.read_file(path='a.txt'))\n", timeout=30)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert "clean" in res
    assert "LEAK" not in res
