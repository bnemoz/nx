"""Contract tests for the unified per-test teardown (nx_state_reset_body).

Phase 1 (docs/test-architecture.md 5.0): the drain contract lives ONCE, in
tests/_nxtest.nx_state_reset_body, called by both the autouse fixture and
conftest's _restore(). These tests pin that contract: for each drained home,
plant residue, force the body, assert the home is clean. They are also the
tripwire for the two ways the contract regresses - someone adds a new
_PENDING_* slot in nx/_state.py (the derivation must see it) and someone
re-forks conftest's copy (the conftest-wired test fails).
"""
import importlib
import os
import sys

import nx._state as st
from _nxtest import nx_state_reset_body, _pending_slots


def test_wake_residue_is_swept():
    st._WAKE_DUE = True
    st._WAKE_ARMED = True
    st._WAKE_STREAK = 7
    # 0.161.0 (Phase C): the wake-state section of the reset body is hand-
    # listed, so the plan-stall slots are pinned here too — they are dir(st)
    # _WAKE_* names but NOT _PENDING_* ones, so the derived notice oracle
    # below cannot see them.
    st._WAKE_STALL_AT = 1234.5
    st._WAKE_STALL_REVISION = 99
    st._WAKE_STALL_HOLD = True
    st._WAKE_STALL_PENDING = True
    # 0.162.0 (Phase D): the chain cost accumulators are hand-listed here too.
    st._WAKE_CHAIN_SPEND = 999999
    st._WAKE_CHAIN_BASE = 4242
    nx_state_reset_body()
    assert st._WAKE_DUE is False
    assert st._WAKE_ARMED is False
    assert st._WAKE_STREAK == 0
    assert st._WAKE_STALL_AT == 0.0
    assert st._WAKE_STALL_REVISION == -1
    assert st._WAKE_STALL_HOLD is False
    assert st._WAKE_STALL_PENDING is False
    assert st._WAKE_CHAIN_SPEND == 0
    assert st._WAKE_CHAIN_BASE == 0


def test_last_prompt_tap_records_role_index_digest():
    """0.166.0 (EXPLAINABLE_HARNESS_SPEC Phase 1): the last-prompt tap stores
    one digest row per message — role, position, and a sha256 over
    role + the message's canonical text — and nothing else. The /ctx as-sent
    view reads exactly these fields, so the row shape is pinned here."""
    st._LAST_PROMPT = None
    st.record_last_prompt([
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": [{"type": "text", "text": "part a"},
                                     {"type": "text", "text": " part b"}]},
        {"role": "assistant", "content": ""},
    ])
    snap = st._LAST_PROMPT
    assert [r["role"] for r in snap] == ["system", "user", "assistant"]
    assert [r["index"] for r in snap] == [0, 1, 2]
    assert all(len(r["sha256"]) == 64 for r in snap)
    # No baseline to diff against on the first recorded call.
    assert [r["changed"] for r in snap] == [False, False, False]


def test_last_prompt_digest_tracks_content_not_identity():
    """Same content ⇒ same digest (so an unchanged row reads `identical to
    live`); different content ⇒ different digest; the role is part of the hash
    input, so the same text under two roles does not collide. The list-of-parts
    projection is the same `message_content_text` /ctx renders, so a parts form
    and its joined string hash alike."""
    st.record_last_prompt([{"role": "user", "content": "hello"}])
    first = st._LAST_PROMPT[0]["sha256"]
    st.record_last_prompt([{"role": "user", "content": "hello"}])
    assert st._LAST_PROMPT[0]["sha256"] == first
    assert st._LAST_PROMPT[0]["changed"] is False
    st.record_last_prompt([{"role": "user", "content": "goodbye"}])
    other = st._LAST_PROMPT[0]["sha256"]
    assert other != first
    assert st._LAST_PROMPT[0]["changed"] is True
    st.record_last_prompt([{"role": "assistant", "content": "goodbye"}])
    assert st._LAST_PROMPT[0]["sha256"] != other
    st.record_last_prompt([{"role": "user",
                            "content": [{"type": "text", "text": "hello"}]}])
    assert st._LAST_PROMPT[0]["sha256"] == first


def test_last_prompt_tap_is_drained():
    """The tap is a session-scope global, so per AGENTS.md the drain lives in
    the central fixture, not in a test. Residue would make the /ctx as-sent view
    report provenance for messages a later test never sent."""
    st.record_last_prompt([{"role": "user", "content": "residue"}])
    assert st._LAST_PROMPT
    nx_state_reset_body()
    assert st._LAST_PROMPT is None


def test_last_compression_ledger_is_drained():
    """Same session-scope global, same drain contract (Phase 2). Residue would
    make the /ctx before/after view describe a fold a later test never ran."""
    st.record_last_compression(kept=1, summarized=2, summary_tokens_est=10,
                               pinned=0, before_messages=5, after_messages=3,
                               ts=1.0)
    assert st._LAST_COMPRESSION is not None
    nx_state_reset_body()
    assert st._LAST_COMPRESSION is None


def test_metrics_tls_is_cleared():
    st._METRICS_TLS.metrics = {"t": 1}
    st._METRICS_TLS.other = "residue"
    nx_state_reset_body()
    assert vars(st._METRICS_TLS) == {}


def test_call_target_tls_is_cleared():
    """0.156.0 (phase M): the per-TASK model routing target
    (`_subagent_model.call_target_scope` / `_CALL_TARGET_TLS`) is a TLS home
    with the same leak shape as _POSTURE_TLS — installed by the scope, deleted
    in its finally, so only a crash between install and restore (or a test
    driving the scope by hand) leaves it set. Plant, drain, assert clean."""
    import nx._subagent_model as sm
    sm._CALL_TARGET_TLS.target = ("client", "model", {}, None)
    nx_state_reset_body()
    assert vars(sm._CALL_TARGET_TLS) == {}
    assert sm.call_target() is None


def test_task_level_tls_is_cleared():
    """0.156.0 (batch task overrides): the per-task headless threshold
    (`_approval.task_level_scope` / `_TASK_LEVEL_TLS`) — same plant/drain/
    assert shape as the slot above. Residue would auto-approve actions whose
    local risk hint is at/below the stale level on every later quiet decision."""
    import nx._approval as ap
    ap._TASK_LEVEL_TLS.level = "high"
    nx_state_reset_body()
    assert vars(ap._TASK_LEVEL_TLS) == {}
    assert ap._task_level() is None


def test_policy_tls_is_cleared():
    """0.156.0 (batch task overrides): the per-task approval policy
    (`_approval_policy.policy_scope` / `_POLICY_TLS`) — residue would keep
    policy-allowing tools on every later quiet decision on this thread."""
    from nx import _approval_policy as apol
    apol._POLICY_TLS.spec = "headless-allow:bash*"
    nx_state_reset_body()
    assert vars(apol._POLICY_TLS) == {}
    # env-only again after the drain (NX_APPROVAL_POLICY unset in this suite)
    assert apol._policy_globs() == ()


def test_eventsink_and_steer_homes_are_swept():
    """0.155.1 pin: the sink registry, retained tails and per-agent steer
    queues are keyed by job id; residue there leaks into the next test's
    snapshot rows (the test_event_taps answer-tail flake). Plant, drain,
    assert clean."""
    import nx._eventsink as es
    import nx._surfaces as surf
    sink = es.register_sink("agent-snap-3")
    sink.write("agent-snap-3", "stream", "chatter")
    es._RETAINED_TAILS["agent-snap-3"] = ["[stream] retained residue"]
    with surf._STEER_LOCK:
        surf._AGENT_QUEUES["agent-snap-3"] = ["steer residue"]
    nx_state_reset_body()
    assert es.sink_for("agent-snap-3") is None
    assert es.retained_tail("agent-snap-3") == []
    assert surf.agent_steering_pending("agent-snap-3") == 0


def test_pending_slots_are_derived_from_the_store():
    """dir()-oracle: every _PENDING_* attr on nx._state plus every routed
    default must be swept - a NEW slot added in nx/_state.py is covered by
    construction, not by editing a hand list."""
    seen = set(_pending_slots(st))
    by_dir = {a for a in dir(st) if a.startswith("_PENDING_")}
    assert by_dir <= seen
    assert set(st._NOTICE_DEFAULTS) <= seen
    # and the sweep actually resets them through the notices store
    import nx._notices as no
    for slot in sorted(by_dir):
        no.set_(slot, "residue")
    nx_state_reset_body()
    for slot in sorted(by_dir):
        assert no.get(slot) in (None, False)


def test_notice_ctx_and_bg_scan_state_are_swept():
    import nx._notices as no
    no.install_notice_ctx()
    st._BG_NOTICE_SEEN = {"123"}
    st._BG_SCAN_DONE = True
    nx_state_reset_body()
    assert no._CURRENT.get() is None
    assert st._BG_NOTICE_SEEN is None
    assert st._BG_SCAN_DONE is False


def test_conftest_restore_wires_the_same_body():
    """conftest._restore() must run the shared body - the unification
    contract. Plant residue, call _restore directly (the pattern the
    conftest race-regression test uses), assert the body's drains ran."""
    import conftest
    st._WAKE_DUE = True
    st._METRICS_TLS.hit = True
    conftest._restore(__file__)
    assert st._WAKE_DUE is False
    assert vars(st._METRICS_TLS) == {}


def test_conftest_restore_rewinds_sys_argv():
    """0.155.4 family A: a test that leaves a fake sys.argv behind (bare
    assignment or monkeypatch leak) must be rewound by _restore — otherwise a
    later importlib.reload(nx) re-scans the poisoned argv and re-arms posture
    (st.YOLO=True inside unrelated tests)."""
    import conftest
    saved = list(sys.argv)
    try:
        sys.argv[:] = ["nx.py", "--resume", "--yolo"]
        conftest._restore(__file__)
        assert sys.argv == saved, sys.argv
    finally:
        sys.argv[:] = saved


def test_reload_after_argv_rewind_leaves_posture_clean():
    """Second family-A pin: with a clean argv a reload leaves posture alone;
    with '--yolo' injected, _restore()'s argv rewind (NOT the reload) is what
    keeps a later reload from re-arming it. The pin restores the posture it
    mutated: reload once more after rewinding argv and assert clean."""
    import conftest
    import nx
    saved_argv = list(sys.argv)
    saved_yolo, saved_level = st.YOLO, st.APPROVE_LEVEL
    try:
        importlib.reload(nx)
        assert st.YOLO is False and st.APPROVE_LEVEL is None, \
            (st.YOLO, st.APPROVE_LEVEL)
        sys.argv[:] = ["nx.py", "--yolo"]
        conftest._restore(__file__)
        assert sys.argv == saved_argv, sys.argv
        importlib.reload(nx)
        assert st.YOLO is False and st.APPROVE_LEVEL is None, \
            "the rewound argv must not re-arm posture on a later reload"
    finally:
        sys.argv[:] = saved_argv
        importlib.reload(nx)
        st.YOLO = saved_yolo
        st.APPROVE_LEVEL = saved_level


def _stage_finished_bg_job(home, command="echo residue-01554"):
    """Stage a finished, owned bg job the way test_bg_completion_seen stages
    them: a real backgrounded bash job whose log still exists. Caller pins
    st.SESSION_HOME to `home` (monkeypatch) BEFORE calling — the completion
    sweep lists jobs from SESSION_HOME, so it must stay pointed there."""
    bj = sys.modules.get("nx._bash_jobs")
    if bj is None:
        import nx._bash_jobs as bj
    pid, _log = bj._run_bash(command)
    assert bj._poll_pid(pid, 30.0) == "exited"
    jobs = bj.list_bg_jobs(mine_only=True)
    job = [j for j in jobs if str(j["pid"]) == str(pid)]
    return pid, (job[0] if job else None)


def test_bg_baseline_seeding_silences_worker_residue(tmp_path, monkeypatch):
    """0.155.4 family B: file-based bg residue from the worker's shared home
    (finished, log present, owner_pid == worker pid) used to be announced as a
    user-role [Runtime note] by the completion sweep on a later turn's first
    tool call. nx_state_reset_body now seeds st._BG_NOTICE_BASELINE once per
    worker from exactly those jobs. Pin both directions."""
    import nx._bg_completions as bc
    home = str(tmp_path / "nxhome")
    os.makedirs(home, exist_ok=True)
    monkeypatch.setattr(st, "SESSION_HOME", home)
    pid, job = _stage_finished_bg_job(home)
    assert job and not job.get("alive") and os.path.exists(job["log_path"])
    # the harness seed itself: reset the once-per-worker latch, clear any
    # baseline, and let _seed_bg_notice_baseline compute it from the same
    # public view the sweep reads.
    import _nxtest
    monkeypatch.setattr(st, "_BG_NOTICE_BASELINE", None, raising=False)
    monkeypatch.setattr(_nxtest, "_BG_NOTICE_BASELINE_SEEDED", False)
    _nxtest._seed_bg_notice_baseline(st)
    assert st._BG_NOTICE_BASELINE == {str(pid)}, st._BG_NOTICE_BASELINE
    # direction 1: with the baseline seeded (what _seed_bg_notice_baseline
    # computes), the sweep announces nothing for the residue.
    monkeypatch.setattr(st, "_BG_NOTICE_BASELINE", {str(pid)}, raising=False)
    st._BG_NOTICE_SEEN = None
    assert bc._scan_background_completions() is None
    # direction 2: WITHOUT the baseline seeding the note lands.
    monkeypatch.setattr(st, "_BG_NOTICE_BASELINE", set(), raising=False)
    st._BG_NOTICE_SEEN = None
    note = bc._scan_background_completions()
    assert note is not None and "[Runtime note:" in note, note


def test_bg_note_is_silent_on_a_turn_with_seeded_baseline(tmp_path, monkeypatch):
    """End-to-end shape of the forced_final flake: a staged finished owned bg
    job + a turn path that flushes module-store notices must NOT append a
    user-role runtime note when the harness seeded the baseline."""
    from nx._runtime_notes import _is_runtime_note, _flush_module_store_notes
    home = str(tmp_path / "nxhome")
    os.makedirs(home, exist_ok=True)
    monkeypatch.setattr(st, "SESSION_HOME", home)
    pid, job = _stage_finished_bg_job(home)
    assert job and not job.get("alive") and os.path.exists(job["log_path"])
    old_base = getattr(st, "_BG_NOTICE_BASELINE", None)
    old_seen = st._BG_NOTICE_SEEN
    try:
        st._BG_NOTICE_BASELINE = {str(pid)}   # what _seed_bg_notice_baseline computes
        st._BG_NOTICE_SEEN = None
        note = sys.modules["nx._bg_completions"]._scan_background_completions()
        assert note is None, note
        messages = []
        _flush_module_store_notes(messages, set())
        assert not any(_is_runtime_note(m) for m in messages), messages
        # and without the baseline the note DOES land (both directions pinned)
        st._BG_NOTICE_BASELINE = set()
        st._BG_NOTICE_SEEN = None
        note = sys.modules["nx._bg_completions"]._scan_background_completions()
        assert note is not None and "[Runtime note:" in note, note
    finally:
        st._BG_NOTICE_BASELINE = old_base
        st._BG_NOTICE_SEEN = old_seen


def test_leaked_live_is_stopped_not_just_unstacked():
    """The completed drain: a Live left RUNNING on the shared console by a
    crashed dashboard test must be *stopped* (rich's own rendering state
    released), not merely unstacked.

    `_live_stack.clear()` alone makes `live_active()` False, so the old drain
    passed every stack-shape assertion while leaving the Live object started:
    `_started` stays True (so the next Live on this console is treated as
    `_nested` and never erases its frame) and the Live stays in the console's
    `_render_hooks`. Both of those are asserted here — they are the checks that
    fail without stop()."""
    import io

    import nx._rich as rich
    from rich.live import Live

    console = rich.shared_console()
    saved_file = console._file
    live = Live("leaked frame", console=console, auto_refresh=False,
                transient=True)
    console._file = io.StringIO()   # as a crashed capture would leave it
    live.start(refresh=False)
    assert console._live_stack, "precondition: the Live is really on the stack"
    try:
        nx_state_reset_body()
        assert live._started is False, (
            "the drain cleared the stack without stopping the Live — rich's "
            "render state (hiding cursor, nested Live handling) stays live")
        assert not console._live_stack
        assert live not in console._render_hooks, (
            "the stopped Live left its render hook on the console")
        assert rich.live_active() is False
    finally:
        try:
            live.stop()
        except Exception:
            pass
        console._file = saved_file


def test_drain_restores_a_capture_swapped_console_file():
    """Second half of the same leak: `captured_console_output` swaps the shared
    console's file to a StringIO; a test that dies before its finally leaves
    every later rich-branch write going into a dead buffer. The drain hands the
    console's lazy `file` back to `sys.stdout` (`_file = None`)."""
    import io

    import nx._rich as rich

    console = rich.shared_console()
    saved_file = console._file
    try:
        console.file = io.StringIO()
        nx_state_reset_body()
        assert not isinstance(getattr(console, "_file", None), io.StringIO), (
            "a leaked capture buffer is still the shared console's file")
        assert console.file is sys.stdout
    finally:
        console._file = saved_file


def test_captured_console_output_restores_none_say_console():
    """The old_say restore bug: a pre-capture _say_console of None was never
    restored, leaking the temp console into later tests on the worker."""
    import nx._rich as rich
    import nx._turn_footer as footer
    from _nxtest import captured_console_output

    saved = footer._say_console
    try:
        footer._say_console = None
        with captured_console_output({"TTY_INTERACTIVE": "1"}):
            assert footer._say_console is not None  # temp console bound
        assert footer._say_console is None  # restored, not left dangling
    finally:
        footer._say_console = saved
        rich._SHARED_CONSOLE = rich.themed_console()
