"""Contract tests for the unified per-test teardown (nx_state_reset_body).

Phase 1 (docs/test-architecture.md 5.0): the drain contract lives ONCE, in
tests/_nxtest.nx_state_reset_body, called by both the autouse fixture and
conftest's _restore(). These tests pin that contract: for each drained home,
plant residue, force the body, assert the home is clean. They are also the
tripwire for the two ways the contract regresses - someone adds a new
_PENDING_* slot in nx/_state.py (the derivation must see it) and someone
re-forks conftest's copy (the conftest-wired test fails).
"""
import sys

import nx._state as st
from _nxtest import nx_state_reset_body, _pending_slots


def test_wake_residue_is_swept():
    st._WAKE_DUE = True
    st._WAKE_ARMED = True
    st._WAKE_STREAK = 7
    nx_state_reset_body()
    assert st._WAKE_DUE is False
    assert st._WAKE_ARMED is False
    assert st._WAKE_STREAK == 0


def test_metrics_tls_is_cleared():
    st._METRICS_TLS.metrics = {"t": 1}
    st._METRICS_TLS.other = "residue"
    nx_state_reset_body()
    assert vars(st._METRICS_TLS) == {}


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
