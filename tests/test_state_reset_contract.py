"""Contract tests for the unified per-test teardown (nx_state_reset_body).

Phase 1 (docs/test-architecture.md 5.0): the drain contract lives ONCE, in
tests/_nxtest.nx_state_reset_body, called by both the autouse fixture and
conftest's _restore(). These tests pin that contract: for each drained home,
plant residue, force the body, assert the home is clean. They are also the
tripwire for the two ways the contract regresses - someone adds a new
_PENDING_* slot in nx/_state.py (the derivation must see it) and someone
re-forks conftest's copy (the conftest-wired test fails).
"""
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
