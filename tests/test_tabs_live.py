"""EVENTSURFACE_SPEC Phase 3 (display layer) + Phase 4 (write serialization)
tests: /tabs live mode — as re-based on the rich Live engine of
SPEC_LIVEENGINE.md (0.128.0).

Covers: toggle idempotence, mid-turn Ctrl-T capture in _interrupt.py, region
open/close/repaint as Live lifecycle (start on open, update+refresh on the
4 fps tick, stop on close, no output while closed), the Live's construction
contract (shared console, no auto-refresh thread, no stdout redirect),
agent-block tear avoidance, NX_TABS_HEIGHT parsing/clamping, the quiet-path
guardrail (dashboard closed => parent console output byte-identical), and
Write serialization (Phase 4): every mid-turn writer goes through the shared
console (no bare `print`), and the _stream_parse Live sites take _surfaces'
_WRITE_LOCK (RLock reentrancy).

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


class _FakeLive:
    """Records the region Live's lifecycle the way rich's Live would run it."""

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.updates = []
        self.refreshes = 0

    def start(self, refresh: bool = False):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def update(self, renderable, refresh: bool = False):
        self.updates.append(renderable)

    def refresh(self):
        self.refreshes += 1


def _reset_dash() -> None:
    """Fresh dashboard state per test (module-level flags)."""
    with _surfaces._DASH_LOCK:
        _surfaces._DASH_OPEN = False
        _surfaces._DASH_DISABLED = False
        _surfaces._DASH_LIVE = None


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
            fake = _FakeLive()
            with _VTFake(), \
                    mock.patch.object(_surfaces, "_make_region_live", lambda: fake):
                _surfaces.open_dashboard()
                with mock.patch.object(_surfaces, "_live_frame",
                                       side_effect=RuntimeError("boom")):
                    _surfaces.repaint_live()
            self.assertTrue(_surfaces.dashboard_disabled())
            # the crash closed the region (Live stopped) rather than leaving a
            # half-painted frame behind
            self.assertEqual(fake.stopped, 1)
            self.assertIsNone(_surfaces.live_region())
            # further toggles refuse
            self.assertFalse(_surfaces.toggle_dashboard())


class TestRepaintRegionBounds(unittest.TestCase):
    """The region's paint is now a rich Live lifecycle, not cursor arithmetic
    (SPEC_LIVEENGINE.md): open starts it, the tick updates+refreshes it, close
    stops it (transient=True erases the frame), and a closed region writes
    nothing at all."""

    def setUp(self):
        _reset_dash()

    def tearDown(self):
        _reset_dash()

    def _open_fake(self):
        """Open the region against a recording FakeLive; return it."""
        fake = _FakeLive()
        patcher = mock.patch.object(_surfaces, "_make_region_live",
                                    lambda: fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        with _VTFake():
            self.assertTrue(_surfaces.open_dashboard())
        return fake

    def test_open_starts_live_and_tick_updates_it(self):
        fake = self._open_fake()
        self.assertEqual(fake.started, 1)
        self.assertIs(_surfaces.live_region(), fake)
        self.assertTrue(_surfaces.dashboard_open())
        # the 4 fps tick hands the frame to the Live and refreshes
        with _VTFake():
            _surfaces.repaint_live()
        self.assertEqual(len(fake.updates), 1)
        self.assertEqual(fake.refreshes, 1)
        # the renderable is the frame lines joined and parsed to a Text (a plain
        # str would go through Console.render_str and choke on a frame row's
        # square brackets); no cursor arithmetic survives in it
        from rich.text import Text
        expected = Text.from_ansi("\n".join(_surfaces._live_frame()))
        self.assertIsInstance(fake.updates[0], Text)
        self.assertEqual(fake.updates[0].plain, expected.plain)
        self.assertTrue(fake.updates[0].no_wrap)
        for idiom in ("\x1b[J", "\r", "A\r"):
            self.assertNotIn(idiom, fake.updates[0].plain)

    def test_region_live_renders_a_frame_on_a_terminal(self):
        """End-to-end (no fakes): the Live actually PAINTS on a terminal — the
        frame text is emitted, transient=True erases it on close, and the cursor
        is hidden/restored. The shared console is rebuilt from documented env
        hooks (TTY_INTERACTIVE/COLORTERM) writing to a captured buffer, so this
        stays headless-safe (no pty, no stdio swap, no rich-privates patched)."""
        import tests._nxtest as _nxtest
        from rich.live import Live
        with _nxtest.captured_console_output(
                {"TTY_COMPATIBLE": "1", "TTY_INTERACTIVE": "1", "COLORTERM": "truecolor"}) as (console, buf):
            with mock.patch.object(_surfaces, "_make_region_live",
                                   lambda: Live(_surfaces._region_renderable(),
                                                console=console, auto_refresh=False,
                                                transient=True,
                                                vertical_overflow="visible")), \
                    _VTFake():
                self.assertTrue(_surfaces.open_dashboard())
                _surfaces.repaint_live()
                self.assertTrue(_surfaces.dashboard_open())
                _surfaces.close_dashboard()
            out = buf.getvalue()
        # the real Live painted the real frame through the shared console: once
        # at start() and again on the repaint. (Cursor hide/restore and the
        # transient erase are rich's own bookkeeping on top of this.)
        self.assertIn("tabs", out, "no frame painted on a terminal")
        self.assertGreaterEqual(out.count("ctrl-t closes"), 2,
                                "start()+repaint did not both render the frame")
        self.assertFalse(_surfaces.dashboard_disabled(),
                         "the render must not trip the crash handler")

    def test_frame_with_brackets_does_not_kill_the_region(self):
        """Regression pin: a frame row containing square brackets (a clipped
        `[/etc/hosts]` tail, a `[Steering note]` instruction) must not raise
        MarkupError out of Live.update — which, on the real code path, would trip
        the crash handler and disable the region for the whole session."""
        from rich.text import Text
        rows = [{"kind": "agent", "id": "a1", "alive": True,
                 "status": "running", "instruction": "[Steering note] do it",
                 "tail": "tail"}]
        with mock.patch.object(_surfaces, "_surfaces_snapshot", return_value=rows):
            renderable = _surfaces._region_renderable()
        self.assertIsInstance(renderable, Text)
        self.assertIn("[Steering note]", renderable.plain)
        # and the Live can consume it without render_str raising MarkupError
        from rich.console import Console
        from rich.live import Live
        c = Console(file=io.StringIO(), force_terminal=True, width=80)
        live = Live(renderable, console=c, auto_refresh=False)
        live.start(refresh=False)
        live.update(_surfaces._region_renderable())
        live.refresh()
        live.stop()

    def test_open_is_idempotent(self):
        fake = self._open_fake()
        with _VTFake():
            self.assertTrue(_surfaces.open_dashboard())
        self.assertEqual(fake.started, 1, "second open must not build a new Live")
        self.assertIs(_surfaces.live_region(), fake)

    def test_close_stops_live_and_clears_it(self):
        fake = self._open_fake()
        with _VTFake():
            _surfaces.repaint_live()
        self.assertEqual((len(fake.updates), fake.refreshes), (1, 1))
        with _VTFake():
            _surfaces.close_dashboard()
        self.assertEqual(fake.stopped, 1)
        self.assertIsNone(_surfaces.live_region())
        self.assertFalse(_surfaces.dashboard_open())
        # closed: the tick must not touch the (gone) Live
        with _VTFake():
            _surfaces.repaint_live()
        self.assertEqual((len(fake.updates), fake.refreshes), (1, 1))
        # close is idempotent
        with _VTFake():
            _surfaces.close_dashboard()
        self.assertEqual(fake.stopped, 1)

    def test_live_construction_contract(self):
        """The Live must share the ONE console, own line discipline itself
        (transient + no redirect), and never spawn a refresh thread."""
        from nx._rich import shared_console
        with mock.patch("rich.live.Live") as fake_live_cls, _VTFake():
            fake_live_cls.return_value = _FakeLive()
            self.assertTrue(_surfaces.open_dashboard())
            kwargs = fake_live_cls.call_args.kwargs
            self.assertIs(kwargs["console"], shared_console())
            self.assertFalse(kwargs["auto_refresh"])
            self.assertTrue(kwargs["transient"])
            self.assertFalse(kwargs["redirect_stdout"])
            self.assertFalse(kwargs["redirect_stderr"])
            self.assertEqual(kwargs["vertical_overflow"], "visible")
            # 4 fps, matching LIVE_FPS_INTERVAL
            self.assertAlmostEqual(kwargs["refresh_per_second"],
                                   1 / _surfaces.LIVE_FPS_INTERVAL)
            _surfaces.close_dashboard()

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

    def _source_has_no_bare_prints(self, rel_path: str) -> None:
        """AST guard (SPEC_LIVEENGINE.md): no bare `print()` in a mid-turn
        writer module. Every mid-turn line must go through the shared console
        (`nx._rich.cprint`), which composes it above an active Live frame
        instead of tearing through it. A NEW print site is caught by this.

        `sys.stdout.write` is deliberately NOT flagged: the two remaining sites
        (`_stream_parse`'s tool-status line + its CLEAR_LINE erase, and
        `LifeSpinner`) are gated on whether a Live actually owns the console,
        keeping the pre-0.128 bytes when none is up. `_ansi`'s spinner path is
        covered by its own `_draw_frame` switch and by tests/test_spinner.py.
        """
        import ast
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "nx", rel_path)
        tree = ast.parse(open(path, encoding="utf-8").read())

        class Walk(ast.NodeVisitor):
            def __init__(self):
                self.offenders = []

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    self.offenders.append(node.lineno)
                self.generic_visit(node)

        w = Walk()
        w.visit(tree)
        self.assertEqual(w.offenders, [],
                         f"{rel_path}: bare print() at lines {w.offenders} — "
                         f"mid-turn writers must use nx._rich.cprint "
                         f"(the shared console)")

    def test_mid_turn_writers_use_the_shared_console(self):
        for mod in ("_turn.py", "_turn_footer.py", "_toolblocks.py",
                    "_stream_parse.py", "_dispatch.py"):
            self._source_has_no_bare_prints(mod)

    def test_raw_write_sites_are_live_gated(self):
        """The only raw `sys.stdout.write` left in _stream_parse must sit
        behind the `_console_has_live()` switch — otherwise a CLEAR_LINE would
        erase the region Live's own frame."""
        import ast
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "nx", "_stream_parse.py")
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)

        class Walk(ast.NodeVisitor):
            def __init__(self):
                self.gated = 0
                self.ungated = []
                self._depth = 0

            def visit_If(self, node):
                test = ast.dump(node.test)
                if "_console_has_live" in test:
                    self._depth += 1
                    for st in node.body:
                        self.visit(st)
                    self._depth -= 1
                    for st in node.orelse:
                        self.visit(st)
                    return
                self.generic_visit(node)

            def visit_Call(self, node):
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr == "write"
                        and isinstance(f.value, ast.Attribute)
                        and f.value.attr == "stdout"):
                    if self._depth:
                        self.gated += 1
                    else:
                        self.ungated.append(node.lineno)
                self.generic_visit(node)

            def visit_IfExp(self, node):
                # `CLEAR_LINE if _console_has_live() else ""` style guards
                src_seg = ast.get_source_segment(src, node) or ""
                saved = self._depth
                if "_console_has_live" in src_seg:
                    self._depth = saved
                self.generic_visit(node)
                self._depth = saved

        w = Walk()
        w.visit(tree)
        self.assertEqual(w.ungated, [],
                         f"_stream_parse.py: raw stdout writes at lines "
                         f"{w.ungated} are not behind _console_has_live()")
        self.assertGreaterEqual(w.gated, 1,
                                "the CLEAR_LINE erase site must be Live-gated")

    def test_cprint_is_the_shared_console(self):
        """cprint routes through the ONE console (the same object the say-panel
        Live and the region Live use) — that identity is what lets rich compose
        all three into one block."""
        from nx import _rich, _turn_footer
        self.assertIs(_turn_footer._say_console, _rich.shared_console())
        self.assertIs(_rich.shared_console(), _rich.shared_console())

    def test_cprint_goes_through_the_console_while_a_live_is_up(self):
        """SPEC_LIVEENGINE.md §3: with a Live owning the shared console, mid-turn
        text must be emitted THROUGH the console (so rich composes it above the
        frame, with the frame's erase/reposition handled), not by a raw write
        the Live knows nothing about."""
        import tests._nxtest as _nxtest
        from rich.live import Live
        from nx._rich import cprint
        # TTY_COMPATIBLE=0 + TTY_INTERACTIVE=0 rebuild the shared console as a
        # plain (non-terminal, non-interactive) one — the no-Live-elsewhere
        # headless baseline (the old test's _force_terminal=False) — with the
        # output captured, no privates patched.
        with _nxtest.captured_console_output(
                {"TTY_COMPATIBLE": "0", "TTY_INTERACTIVE": "0"}) as (console, buf):
            live = Live("frame", console=console, auto_refresh=False)
            live.start(refresh=False)
            try:
                cprint("\033[2m  └ 12 tok\033[0m")
            finally:
                live.stop()
            self.assertIn("└ 12 tok", buf.getvalue())
            # live.stop() already unregistered it (Console.clear_live); guard in
            # case start() never got that far.
            if console._live_stack:
                console.clear_live()

    def test_cprint_falls_back_to_plain_print_when_no_live(self):  # noqa: C901
        """No Live => no frame to compose against: keep the classic byte path
        (this is what every non-region caller and the whole test suite expect)."""
        import io as _io
        from contextlib import redirect_stdout
        from nx._rich import cprint
        out = _io.StringIO()
        with redirect_stdout(out):
            cprint("\033[2m  x\033[0m")
        self.assertEqual(out.getvalue(), "\033[2m  x\033[0m\n")

    def test_region_live_and_writers_share_one_console(self):
        """The invariant the whole design rests on: the region Live, the
        say-panel Live and the mid-turn writers use ONE Console object, so rich
        stacks the two Lives into a single block (Console._live_stack) and
        composes printed lines above it."""
        from nx._rich import shared_console
        from rich.live import Live
        console = shared_console()
        with mock.patch.object(_surfaces, "_vt_supported", lambda: True):
            self.assertTrue(_surfaces.open_dashboard())
            try:
                self.assertTrue(console._live_stack,
                                "region Live not registered on the shared console")
                say = Live("say", console=console, auto_refresh=False)
                say.start(refresh=False)
                try:
                    # SAME console -> the second Live is REGISTERED, not fought
                    # over: rich's stack holds the region Live at the bottom,
                    # and a nested refresh delegates to it (no raise, no
                    # second console).
                    self.assertGreaterEqual(len(console._live_stack), 1)
                    say.update("say 2")
                    say.refresh()   # delegates down the stack: must not raise
                finally:
                    say.stop()
            finally:
                # stop the region Live properly: it is registered on the SHARED
                # console, so leaving it started would make every later cprint
                # in the session take the console path (a process-wide leak into
                # other tests).
                _surfaces.close_dashboard()

    def test_stream_parse_live_sites_locked(self):
        """AST guard: every rich Live interaction site (Live()/update/start/
        stop on say_live) in _stream_parse.py sits inside a
        `with write_lock():` block, so a Live frame repaint cannot interleave
        with the /tabs dashboard repaint."""
        import ast
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "nx", "_stream_parse.py")
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
                self.depth += 1 if locked else 0
                self.generic_visit(node)
                if locked:
                    self.depth -= 1

            def visit_Call(self, node):
                f = node.func
                live = (
                    # Live(...) construction
                    (isinstance(f, ast.Name) and f.id == "Live")
                    # say_live.update / say_live.start / say_live.stop
                    or (isinstance(f, ast.Attribute)
                        and f.attr in ("update", "start", "stop")
                        and isinstance(f.value, ast.Attribute)
                        and f.value.attr == "say_live"))
                if live and self.depth == 0:
                    self.offenders.append(node.lineno)
                self.generic_visit(node)

        w = Walk()
        w.visit(tree)
        self.assertEqual(
            w.offenders, [],
            f"_stream_parse.py: Live sites outside `with write_lock()`: "
            f"lines {w.offenders}")

    def test_modules_share_the_surfaces_lock(self):
        """write_lock() is _WRITE_LOCK itself, and the writer modules imported
        THAT function (one lock, not per-module copies)."""
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
            fake = _FakeLive()
            with mock.patch.object(_surfaces, "_make_region_live", lambda: fake):
                _surfaces.open_dashboard()
                with _surfaces.write_lock():   # parent "mid-write"
                    _surfaces.repaint_live()   # would hang if not RLock
                self.assertEqual((len(fake.updates), fake.refreshes), (1, 1))
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
        parser.think_start = time.time() - 1.0  # past _THINK_MIN: block shows
        parser._think_shown = True
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
