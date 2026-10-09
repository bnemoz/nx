"""Right status rail (SPEC_PANEL.md, 0.127.0): panel_cells content, the
NX_PANEL / NX_PANEL_WIDTH gates, and _live_frame composition on/off.
Headless-safe like the rest of the suite (no tty required)."""
import importlib
import os
import re
import shutil
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

from nx import _panel, _surfaces  # noqa: E402
from nx import _state as st  # noqa: E402
from nx._ansi import C  # noqa: E402

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


ROWS = [
    {"kind": "main", "id": "main", "alive": True, "status": "running"},
    {"kind": "bg", "id": "bg-1", "alive": True, "status": "running"},
    {"kind": "bg", "id": "bg-2", "alive": False, "status": "finished"},
    {"kind": "agent", "id": "agent-1-abc", "alive": True, "status": "running"},
    {"kind": "agent", "id": "agent-2-def", "alive": False, "status": "done"},
]


class _PanelEnv(unittest.TestCase):
    """Base: NX_* gate vars cleared + the /panel override reset per test."""

    def setUp(self):
        self._saved = {}
        for var in ("NX_PANEL", "NX_PANEL_WIDTH"):
            self._saved[var] = os.environ.pop(var, None)
        self._override = _panel._OVERRIDE
        _panel._OVERRIDE = None

    def tearDown(self):
        for var, val in self._saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
        _panel._OVERRIDE = self._override


class TestPanelCells(_PanelEnv):
    def test_ctx_counts_and_layout(self):
        class _Src:
            _context_window = 128000
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 12345), \
             mock.patch.object(st, "ACTIVE", _Src()):
            cells = _panel.panel_cells(24, ROWS)
        self.assertTrue(cells[0].startswith("\x1b["), "ctx cell styled")
        self.assertIn("ctx", _strip(cells[0]))
        self.assertIn("12k", _strip(cells[0]))
        self.assertIn("/128k", _strip(cells[0]))  # green <30% utilization
        self.assertIn(C.SUCCESS, cells[0])
        self.assertIn("jobs 1", _strip(cells[1]))    # one ALIVE bg row
        self.assertIn("agents 1", _strip(cells[2]))  # one ALIVE agent row

    def test_ctx_unknown_max_degrades(self):
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 832), \
             mock.patch.object(st, "ACTIVE", None):
            cells = _panel.panel_cells(24, ROWS)
        self.assertEqual(_strip(cells[0]), "832 ctx")

    def test_ctx_red_at_high_utilization(self):
        class _Src:
            _context_window = 10000
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 7000), \
             mock.patch.object(st, "ACTIVE", _Src()):
            cells = _panel.panel_cells(24, ROWS)
        self.assertIn(C.ERROR, cells[0])

    def test_steer_cell_hidden_when_zero_present_when_nonzero(self):
        with mock.patch.object(_surfaces, "steer_pending", return_value=0):
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("steer" in _strip(c) for c in cells))
        with mock.patch.object(_surfaces, "steer_pending", return_value=2):
            cells = _panel.panel_cells(24, ROWS)
        self.assertTrue(any("steer 2" in _strip(c) for c in cells))

    def test_cells_truncated_to_width(self):
        class _Src:
            _context_window = 1234567
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 1234567), \
             mock.patch.object(st, "ACTIVE", _Src()):
            for w in (12, 18):
                for c in _panel.panel_cells(w, ROWS):
                    self.assertLessEqual(len(_strip(c)), w)


class TestGates(_PanelEnv):
    def setUp(self):
        super().setUp()
        # the test harness's terminal is often 80 cols, which the >=60-free-
        # cols gate would refuse; pin a wide size so these tests assert the
        # env semantics, not the harness width
        self._size = mock.patch.object(shutil, "get_terminal_size",
                                       return_value=os.terminal_size((120, 24)))
        self._size.start()

    def tearDown(self):
        self._size.stop()
        super().tearDown()

    def test_default_off_env_on(self):
        self.assertFalse(_panel.panel_enabled())  # NX_PANEL unset
        os.environ["NX_PANEL"] = "1"
        self.assertTrue(_panel.panel_enabled())
        os.environ["NX_PANEL"] = "TRUE"
        self.assertTrue(_panel.panel_enabled())
        os.environ["NX_PANEL"] = "0"
        self.assertFalse(_panel.panel_enabled())

    def test_narrow_terminal_gate(self):
        os.environ["NX_PANEL"] = "1"
        # 80 - 24 = 56 < 60 -> silently off; 100 - 24 = 76 >= 60 -> on
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=os.terminal_size((80, 24))):
            self.assertFalse(_panel.panel_enabled())
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=os.terminal_size((100, 24))):
            self.assertTrue(_panel.panel_enabled())

    def test_width_clamped(self):
        self.assertEqual(_panel.panel_width(), 24)  # default
        os.environ["NX_PANEL_WIDTH"] = "4"
        self.assertEqual(_panel.panel_width(), 12)
        os.environ["NX_PANEL_WIDTH"] = "99"
        self.assertEqual(_panel.panel_width(), 40)
        os.environ["NX_PANEL_WIDTH"] = "notanint"
        self.assertEqual(_panel.panel_width(), 24)

    def test_set_panel_enabled_override(self):
        _panel.set_panel_enabled(True)
        self.assertTrue(_panel.panel_enabled())  # on despite unset env
        _panel.set_panel_enabled(False)
        os.environ["NX_PANEL"] = "1"
        self.assertFalse(_panel.panel_enabled())  # toggle beats env


class TestLiveFrame(_PanelEnv):
    """_live_frame composition: disabled -> no rail at all; enabled -> every
    line carries the separator + a cell column."""

    def test_disabled_output_has_no_rail(self):
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            lines = _surfaces._live_frame()
        self.assertTrue(lines)
        for ln in lines:
            self.assertNotIn("│", ln)

    def test_enabled_appends_separator_and_cells(self):
        # driven via NX_PANEL, not set_panel_enabled(): tests/conftest.py
        # reaps nx._panel from sys.modules between tests (it isn't in the
        # post-`import nx` baseline), so _live_frame's lazy
        # `from ._panel import ...` can land on a freshly imported module —
        # the env var crosses that boundary, a module-level flag does not
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=list(ROWS))
        size = mock.patch.object(shutil, "get_terminal_size",
                                 return_value=os.terminal_size((120, 24)))
        os.environ["NX_PANEL"] = "1"
        with size, snap:
            on = _surfaces._live_frame()
        self.assertTrue(on)
        for ln in on:
            self.assertIn("│", ln)
        body = [_strip(ln) for ln in on]
        self.assertTrue(any("jobs 1" in ln for ln in body))
        self.assertTrue(any("agents 1" in ln for ln in body))
        # same frame with the rail off: line count must match, and the
        # enabled lines must start with the (padded) disabled lines
        del os.environ["NX_PANEL"]
        with size, snap:
            off = _surfaces._live_frame()
        self.assertEqual(len(off), len(on))
        pw = _panel.panel_width()
        lw = max(1, 120 - pw - 2)
        for o, n in zip(off, on):
            self.assertTrue(n.startswith(o[:lw].ljust(lw)))


class TestCmdPanelToggle(_PanelEnv):
    """_cmd_panel flips the flag only; no prompt-time region open/close
    (EVENTSURFACE_SPEC §2/§3 — the region is owned by stdin's holder)."""

    def setUp(self):
        super().setUp()
        # conftest reaps nx._panel between tests; _cmd_panel lazily imports it,
        # so test the SAME module instance the command handler will see.
        self._p = importlib.import_module("nx._panel")
        # panel_enabled() carries the >=60-free-cols width gate
        self._size = mock.patch.object(shutil, "get_terminal_size",
                                       return_value=os.terminal_size((120, 24)))
        self._size.start()

    def tearDown(self):
        self._size.stop()
        super().tearDown()

    def _call(self):
        importlib.import_module("nx._repl")._cmd_panel("/panel", None)

    def _no_open(self):
        def _boom(*a, **k):
            raise AssertionError("open_dashboard called at the prompt")
        return mock.patch.object(_surfaces, "open_dashboard", _boom)

    def test_on_flips_flag_without_opening(self):
        with self._no_open():
            self._call()
        self.assertTrue(self._p.panel_enabled())

    def test_off_when_region_closed_does_not_touch_surfaces(self):
        self._p.set_panel_enabled(True)
        with mock.patch.object(_surfaces, "dashboard_open", return_value=False), \
             self._no_open(), \
             mock.patch.object(_surfaces, "close_dashboard",
                               side_effect=AssertionError("closed while closed")):
            self._call()
        self.assertFalse(self._p.panel_enabled())

    def test_off_when_region_open_closes_it(self):
        self._p.set_panel_enabled(True)
        closed = []
        with mock.patch.object(_surfaces, "dashboard_open", return_value=True), \
             mock.patch.object(_surfaces, "close_dashboard",
                               side_effect=lambda: closed.append(True)):
            self._call()
        self.assertFalse(self._p.panel_enabled())
        self.assertEqual(closed, [True])


class TestEnsureRegion(_PanelEnv):
    """ensure_region(): turn-start open, only when enabled and closed."""

    def setUp(self):
        super().setUp()
        self._p = importlib.import_module("nx._panel")
        # panel_enabled() carries the >=60-free-cols gate; pin a wide terminal
        # so these tests assert the flag semantics, not the harness width.
        self._size = mock.patch.object(shutil, "get_terminal_size",
                                       return_value=os.terminal_size((120, 24)))
        self._size.start()

    def tearDown(self):
        self._size.stop()
        super().tearDown()

    def test_disabled_does_nothing(self):
        opened = []
        with mock.patch.object(_surfaces, "dashboard_open", return_value=False), \
             mock.patch.object(_surfaces, "open_dashboard",
                               side_effect=lambda: opened.append(True)):
            _panel.ensure_region()
        self.assertEqual(opened, [])

    def test_enabled_when_closed_opens(self):
        self._p.set_panel_enabled(True)
        opened = []
        with mock.patch.object(_surfaces, "dashboard_open", return_value=False), \
             mock.patch.object(_surfaces, "open_dashboard",
                               side_effect=lambda: opened.append(True)):
            _panel.ensure_region()
        self.assertEqual(opened, [True])

    def test_enabled_when_already_open_is_idempotent(self):
        self._p.set_panel_enabled(True)
        opened = []
        with mock.patch.object(_surfaces, "dashboard_open", return_value=True), \
             mock.patch.object(_surfaces, "open_dashboard",
                               side_effect=lambda: opened.append(True)):
            _panel.ensure_region()
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
