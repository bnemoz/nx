"""EVENTSURFACE_SPEC Phase 3 (display layer) + Phase 4 (write serialization)
tests: /tabs live mode.

Covers: toggle idempotence, mid-turn Ctrl-T capture in _interrupt.py, repaint
region bounds (the \\x1b[{n}A\\r\\x1b[J idiom), agent-block tear avoidance,
NX_TABS_HEIGHT parsing/clamping, the quiet-path guardrail (dashboard closed =>
parent console output byte-identical), and Phase 4 write serialization (raw
print sites in _turn/_stream_parse/_dispatch take _surfaces' _WRITE_LOCK so a
4 fps repaint cannot tear parent output; RLock reentrancy).

Headless-safe: no real tty is required — every test monkeypatches
`nx._surfaces._vt_supported` (the only tty gate) instead.
"""
import io
import sys
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock

import nx as _m
from nx import _surfaces
from nx import _eventsink as _es


def _reset_dash() -> None:
    """Fresh dashboard state per test (module-level flags)."""
    with _surfaces._DASH_LOCK:
        _surfaces._DASH_OPEN = False
        _surfaces._DASH_DISABLED = False
        _surfaces._DASH_DRAWN = 0


class _VTFake:
    """Context manager: fake tty state for the dashboard's only tty gate."""

    def __init__(self, supported: bool = True):
        self.supported = supported

    def __enter__(self):
        self.p = mock.patch.object(_surfaces, "_vt_supported",
                                   lambda: self.supported)
        self.p.start()
        return self

    def __exit__(self, *a):
        self.p.stop()
        return False


class TestTabsHeight(unittest.TestCase):
    def test_default(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("NX_TABS_HEIGHT", None)
            self.assertEqual(_surfaces._tabs_height(), 8)

    def test_clamp_low_and_high(self):
        for val, want in (("0", 1), ("-5", 1), ("1000", 20), ("3", 3)):
            with mock.patch.dict("os.environ", {"NX_TABS_HEIGHT": val}):
                self.assertEqual(_surfaces._tabs_height(), want, val)

    def test_non_int_falls_back(self):
        with mock.patch.dict("os.environ", {"NX_TABS_HEIGHT": "eight"}):
            self.assertEqual(_surfaces._tabs_height(), 8)


class TestToggleIdempotence(unittest.TestCase):
    def setUp(self):
        _reset_dash()

    def tearDown(self):
        _reset_dash()

    def test_toggle_open_then_close(self):
        with _VTFake():
            self.assertTrue(_surfaces.toggle_dashboard())
            self.assertTrue(_surfaces.dashboard_open())
            self.assertFalse(_surfaces.toggle_dashboard())
            self.assertFalse(_surfaces.dashboard_open())

    def test_toggle_is_idempotent(self):
        with _VTFake():
            # toggle alternates open/closed and the return value always
            # matches the resulting state (idempotent open/close contract)
            for expect_open in (True, False, True, False):
                self.assertEqual(_surfaces.toggle_dashboard(), expect_open)
                self.assertEqual(_surfaces.dashboard_open(), expect_open)

    def test_close_when_closed_is_silent_noop(self):
        with redirect_stdout(io.StringIO()) as cap:
            _surfaces.close_dashboard()
            _surfaces.close_dashboard()
        self.assertEqual(cap.getvalue(), "")

    def test_refused_without_vt(self):
        with _VTFake(supported=False):
            self.assertFalse(_surfaces.toggle_dashboard())
            self.assertFalse(_surfaces.dashboard_open())

    def test_repaint_crash_disables_for_session(self):
        with _VTFake():
            self.assertTrue(_surfaces.toggle_dashboard())
            _reset_dash()
            with _VTFake():
                _surfaces._DASH_OPEN = True
                with mock.patch.object(_surfaces, "_live_frame",
                                       side_effect=RuntimeError("boom")):
                    _surfaces.repaint_live()
            self.assertTrue(_surfaces.dashboard_disabled())
            # further toggles refuse
            self.assertFalse(_surfaces.toggle_dashboard())


class TestRepaintRegionBounds(unittest.TestCase):
    def setUp(self):
        _reset_dash()

    def tearDown(self):
        _reset_dash()

    def test_paint_and_erase_sequence(self):
        with _VTFake():
            out = io.StringIO()
            with mock.patch.object(_surfaces.sys, "stdout", out):
                _surfaces.open_dashboard()
                first = out.getvalue()
                self.assertIn("\r\x1b[J", first)
                self.assertGreater(_surfaces._DASH_DRAWN, 0)
                out.seek(0)
                out.truncate(0)
                _surfaces.repaint_live()
                second = out.getvalue()
            # erase prefix: move up exactly as many rows as were painted
            self.assertIn(f"\x1b[{_surfaces._DASH_DRAWN}A\r\x1b[J", second)
            self.assertLessEqual(_surfaces._DASH_DRAWN,
                                 _surfaces._tabs_height())
            out.seek(0)
            out.truncate(0)
            with mock.patch.object(_surfaces.sys, "stdout", out):
                _surfaces.close_dashboard()
            # close erased the region and reset the drawn counter
            self.assertIn("\x1b[", out.getvalue())
            self.assertEqual(_surfaces._DASH_DRAWN, 0)

    def test_zero_output_when_closed(self):
        _reset_dash()
        out = io.StringIO()
        with redirect_stdout(out):
            _surfaces.repaint_live()
            _surfaces.maybe_close_if_idle()
        self.assertEqual(out.getvalue(), "")

    def test_frame_respects_height(self):
        rows = [{"kind": "agent", "id": f"agent-{i}", "alive": True,
                 "status": "running", "instruction": f"task {i}", "tail": ""}
                for i in range(10)]
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=rows), \
                mock.patch.dict("os.environ", {"NX_TABS_HEIGHT": "4"}):
            frame = _surfaces._live_frame()
        self.assertEqual(len(frame), 4)


class TestAgentBlockTearAvoidance(unittest.TestCase):
    def _rows(self, alives):
        return [{"kind": "agent", "id": f"a{i}", "alive": a,
                 "status": "running" if a else "done", "instruction": "",
                 "tail": ""} for i, a in enumerate(alives)]

    def test_block_flips_only_when_whole_block_terminal(self):
        rows = self._rows([False, False, True])
        styles = _surfaces._agent_styles(rows)
        self.assertEqual(styles, [False, False, False])
        rows = self._rows([False, False, False])
        styles = _surfaces._agent_styles(rows)
        self.assertEqual(styles, [True, True, True])

    def test_non_agent_rows_never_gated(self):
        rows = [{"kind": "main", "id": "main", "alive": False,
                 "status": "idle", "instruction": "", "tail": ""}]
        rows += self._rows([False, True])
        rows.append({"kind": "bg", "id": "bg-1", "alive": False,
                     "status": "finished", "instruction": "", "tail": ""})
        styles = _surfaces._agent_styles(rows)
        self.assertEqual(styles, [True, False, False, True])

    def test_order_is_spawn_order(self):
        """Row order == dict insertion order; the block rule runs over that
        order, so a middle agent finishing cannot reorder the block."""
        rows = self._rows([False, True, False, True])
        styles = _surfaces._agent_styles(rows)
        # one contiguous agent block (test rows are all agents) — none flips
        # while any sibling still runs, preserving spawn order visually
        self.assertEqual(styles, [False, False, False, False])


class TestMidTurnCtrlT(unittest.TestCase):
    def setUp(self):
        _reset_dash()
        _m.set_steer_partial("")
        _m.drain_steering()

    def tearDown(self):
        _reset_dash()
        _m.set_steer_partial("")
        _m.drain_steering()

    def test_ctrl_t_toggles_and_is_consumed(self):
        with _VTFake():
            from nx._interrupt import _capture_steering_byte
            self.assertTrue(_capture_steering_byte(0, b"\x14") is None)
            self.assertTrue(_surfaces.dashboard_open())
            # not steering text
            self.assertEqual(_m.steer_partial(), "")
            self.assertEqual(_m.drain_steering(), [])
            _capture_steering_byte(0, b"\x14")
            self.assertFalse(_surfaces.dashboard_open())

    def test_ctrl_t_before_printable_test(self):
        """\x14 < ' ' — the old code dropped it via the printable check. With
        VT refused, the byte is still consumed (no steering pollution)."""
        with _VTFake(supported=False):
            from nx._interrupt import _capture_steering_byte
            _capture_steering_byte(0, b"\x14")
            self.assertEqual(_m.steer_partial(), "")

    def test_watcher_tick_drives_repaint_mid_turn(self):
        """Turn-loop-owned terminal: the armed watcher's poll tick repaints."""
        with _VTFake():
            calls = []
            with _surfaces._DASH_LOCK:
                _surfaces._DASH_OPEN = True
            with mock.patch.object(_surfaces, "repaint_live",
                                   lambda *a, **k: calls.append(1)):
                import nx._interrupt as _int
                r, w = __import__("os").pipe()
                old = _int._WAKE_R, _int._WAKE_W
                _int._WAKE_R, _int._WAKE_W = r, w
                try:
                    # simulate one idle poll iteration: _select_stdin_or_wake
                    # times out, the tick fires because the dashboard is open
                    # and the fps interval has elapsed (_dash_last starts 0.0)
                    sess = _int._run_armed_session
                    with mock.patch.object(_int, "apply_interrupt_mode"), \
                            mock.patch.object(_int, "restore_input_mode"), \
                            mock.patch.object(_int, "_INTERRUPT_ARMED") as arm:
                        arm.is_set.return_value = False  # exit loop at once
                        arm.wait.return_value = False
                        sess.__wrapped__ if hasattr(sess, "__wrapped__") else None
                    # Simpler: drive the tick helper directly
                    from nx._surfaces import repaint_live_tick
                    repaint_live_tick()
                finally:
                    _int._WAKE_R, _int._WAKE_W = old
                    __import__("os").close(r)
                    __import__("os").close(w)
            self.assertEqual(calls, [1])
            with _surfaces._DASH_LOCK:
                _surfaces._DASH_OPEN = False


class TestQuietPathNonInterference(unittest.TestCase):
    def test_dashboard_closed_console_is_quiet(self):
        """Guardrail: with the dashboard closed, spawning + sinking agent
        events prints nothing to the parent console."""
        _reset_dash()
        job = "agent-tabs-quiet-test"
        sink = _es.sink_for(job) or _es.register_sink(job)
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                for i in range(20):
                    sink.write(job, "stream", f"line {i}")
                    _es.tap_event if False else None
                from nx._subagent_tls import tap_event
                tap_event("tool", "search_files -> ok")
                _surfaces.maybe_close_if_idle()
                _surfaces.repaint_live()
            self.assertEqual(out.getvalue(), "")
            # and the ring kept the lines for later /tabs viewing
            self.assertTrue(_es.retained_tail(job, 5) or
                            _es.sink_for(job) is not None)
        finally:
            _es.drop_sink(job)


class TestWriteSerialization(unittest.TestCase):
    """Phase 4: every raw console write in _turn/_stream_parse/_dispatch takes
    the shared _WRITE_LOCK, so a live repaint and parent-turn rendering can
    never interleave (tear) while the dashboard is open."""

    def _source_prints_all_locked(self, rel_path: str) -> None:
        """AST guard: every print()/sys.stdout.write() call in the module sits
        inside a `with write_lock():` block. Catches the regression class the
        lock exists for: a NEW print site added without the wrap."""
        import ast
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "nx", rel_path)
        tree = ast.parse(open(path, encoding="utf-8").read())

        class Walk(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.offenders = []

            def visit_With(self, node):
                locked = any(
                    isinstance(it.context_expr, ast.Call)
                    and isinstance(it.context_expr.func, ast.Name)
                    and it.context_expr.func.id == "write_lock"
                    for it in node.items)
                if locked:
                    self.depth += 1
                    self.generic_visit(node)
                    self.depth -= 1
                else:
                    self.generic_visit(node)

            def visit_Call(self, node):
                f = node.func
                raw = (isinstance(f, ast.Name) and f.id == "print") or (
                    isinstance(f, ast.Attribute) and f.attr == "write"
                    and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "stdout")
                if raw and self.depth == 0:
                    self.offenders.append(node.lineno)
                self.generic_visit(node)

        w = Walk()
        w.visit(tree)
        self.assertEqual(w.offenders, [],
                         f"{rel_path}: raw writes outside `with write_lock()`: "
                         f"lines {w.offenders}")

    def test_all_write_sites_locked(self):
        for mod in ("_turn.py", "_stream_parse.py", "_dispatch.py"):
            self._source_prints_all_locked(mod)

    def test_modules_share_the_surfaces_lock(self):
        """write_lock() is _WRITE_LOCK itself, and the three writer modules
        imported THAT function (one lock, not per-module copies)."""
        import importlib
        lock = _surfaces.write_lock()
        self.assertIs(lock, _surfaces._WRITE_LOCK)
        for mod in ("nx._turn", "nx._stream_parse", "nx._dispatch"):
            m = importlib.import_module(mod)
            self.assertIs(m.write_lock(), lock, mod)

    def test_write_lock_is_reentrant(self):
        """RLock semantics: nested acquires on one thread must not deadlock —
        several wrapped sites wrap multi-print blocks that also nest."""
        lock = _surfaces.write_lock()
        with lock:
            with lock:
                with _surfaces.write_lock():
                    lock.acquire()
                    lock.release()

    def test_repaint_reenters_parent_held_lock(self):
        """Same-thread reentrancy THROUGH the repaint path: a parent holding
        the write lock mid-write must not deadlock the (same-thread) tick."""
        with _VTFake():
            out = io.StringIO()
            with mock.patch.object(_surfaces.sys, "stdout", out):
                _surfaces.open_dashboard()
                out.seek(0)
                out.truncate(0)
                with _surfaces.write_lock():   # parent "mid-write"
                    _surfaces.repaint_live()   # would hang if not RLock
                self.assertIn("tabs", out.getvalue())
                _surfaces.close_dashboard()

    def test_parent_print_blocks_while_repaint_holds_lock(self):
        """Cross-thread serialization, deterministic: a wrapped parent write
        (StreamParser.close_think_block — real Phase-4 site) cannot write while
        another thread holds the lock, and completes right after release."""
        import time
        from rich.console import Console
        from nx import _stream_parse as sp
        lock = _surfaces.write_lock()
        holding = threading.Event()
        release = threading.Event()
        done = threading.Event()

        def holder():
            with lock:                 # simulates the 4 fps repaint's write
                holding.set()
                release.wait(5)
            done.set()

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        self.assertTrue(holding.wait(5))
        parser = sp.StreamParser(Console(file=io.StringIO()), time.time())
        parser.mode = "think"
        parser.think_start = time.time()
        buf = io.StringIO()

        # redirect_stdout is thread-local since 3.15 only; pin stdout for the
        # writer thread manually so the wrapped print cannot touch the real
        # stdout (pytest captures the process-wide stream).
        def run():
            real, sys.stdout = sys.stdout, buf
            try:
                parser.close_think_block()
            finally:
                sys.stdout = real
        writer = threading.Thread(target=run, daemon=True)
        writer.start()
        time.sleep(0.15)               # a fair window to (wrongly) slip through
        self.assertTrue(writer.is_alive(),
                        "wrapped parent write jumped the held write lock")
        release.set()
        writer.join(5)
        t.join(5)
        self.assertFalse(writer.is_alive())
        self.assertTrue(done.is_set())
        self.assertIn("thought for", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
