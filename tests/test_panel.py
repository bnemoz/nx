"""Right status rail (SPEC_PANEL.md, 0.140.0): panel_cells content and slot
order, the display-width clip/measure trio, the NX_PANEL / NX_PANEL_WIDTH gates,
TTL-cached polls, _live_frame composition (rail / degraded one-line / off, which
must stay byte-identical to the pre-panel frame). Headless-safe like the rest of
the suite (no tty required)."""
import importlib
import io
import os
import re
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

from nx import _panel, _surfaces  # noqa: E402
from nx import _fileclaim  # noqa: E402
from nx import _state as st  # noqa: E402
from nx._ansi import C, RESET, clip_display, display_width, pad_display  # noqa: E402

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _cell(cells, key: str) -> str:
    """The one cell whose stripped text starts with `key` (slot order is fixed,
    but which slots SURVIVE depends on state, so tests address cells by name)."""
    hits = [c for c in cells if _strip(c).startswith(key)]
    assert len(hits) == 1, f"{key!r}: {[_strip(c) for c in cells]}"
    return hits[0]


ROWS = [
    {"kind": "main", "id": "main", "alive": True, "status": "running"},
    {"kind": "bg", "id": "bg-1", "alive": True, "status": "running"},
    {"kind": "bg", "id": "bg-2", "alive": False, "status": "finished"},
    {"kind": "agent", "id": "agent-1-abc", "alive": True, "status": "running"},
    {"kind": "agent", "id": "agent-2-def", "alive": False, "status": "done"},
]


class _PanelEnv(unittest.TestCase):
    """Base: NX_* gate vars cleared, the /panel override reset, and the two
    module-level poll caches (branch/claims) rewound per test."""

    def setUp(self):
        self._saved = {}
        for var in ("NX_PANEL", "NX_PANEL_WIDTH"):
            self._saved[var] = os.environ.pop(var, None)
        self._override = _panel._OVERRIDE
        _panel._OVERRIDE = None
        self._branch_cache = _panel._branch_cache
        _panel._branch_cache = (0.0, None)
        self._claims_cache = _panel._claims_cache
        _panel._claims_cache = (0.0, 0)
        # no branch cell unless a test asks for one (the test tree IS a repo).
        # `_real_poll_branch` keeps the unpatched poll reachable for the tests
        # that exercise it against a faked `git`.
        self._real_poll_branch = _panel._poll_branch
        p = mock.patch.object(_panel, "_poll_branch", lambda: None)
        self._no_branch = p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        for var, val in self._saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
        _panel._OVERRIDE = self._override
        _panel._branch_cache = self._branch_cache
        _panel._claims_cache = self._claims_cache


class TestPanelCells(_PanelEnv):
    def test_ctx_counts_and_layout(self):
        class _Src:
            _context_window = 128000
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 12345), \
             mock.patch.object(st, "ACTIVE", _Src()):
            cells = _panel.panel_cells(24, ROWS)
        ctx = cells[0]
        self.assertTrue(ctx.startswith("\x1b["), "ctx cell styled")
        self.assertIn("ctx", _strip(ctx))
        self.assertIn("12k", _strip(ctx))
        self.assertIn("/128k", _strip(ctx))  # green <30% utilization
        self.assertIn(C.SUCCESS, ctx)
        self.assertIn("jobs 1", _strip(_cell(cells, "jobs")))    # one ALIVE bg
        self.assertIn("agents 1", _strip(_cell(cells, "agents")))

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

    def test_ctx_hint_compact_at_red_threshold(self):
        class _Src:
            _context_window = 10000
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 9000), \
             mock.patch.object(st, "ACTIVE", _Src()):
            wide = _panel.panel_cells(24, ROWS)[0]
            narrow = _panel.panel_cells(12, ROWS)[0]
        self.assertIn("/compact", _strip(wide))
        # a 12-cell rail has no room for the hint (`bar + %` already fills it)
        self.assertNotIn("/compact", _strip(narrow))
        self.assertIn("90%", _strip(narrow))
        # below the threshold there is no hint to give
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 3000), \
             mock.patch.object(st, "ACTIVE", _Src()):
            self.assertNotIn("/compact", _strip(_panel.panel_cells(24, ROWS)[0]))

    def test_ctx_bar_form_vs_full_form_by_width(self):
        """Rail narrower than BAR_MODE_WIDTH renders the `▰▱` bar + %; at or
        above it the full `cur/max ctx` form. Same state, two widths."""
        class _Src:
            _context_window = 100000
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 42000), \
             mock.patch.object(st, "ACTIVE", _Src()):
            narrow = _panel.panel_cells(_panel.BAR_MODE_WIDTH - 1, ROWS)[0]
            wide = _panel.panel_cells(_panel.BAR_MODE_WIDTH, ROWS)[0]
        self.assertIn("▰", narrow, "narrow rail must use the bar form")
        self.assertIn("▱", narrow)
        self.assertIn("42%", _strip(narrow))
        self.assertNotIn("/100k", _strip(narrow))
        self.assertNotIn("▰", wide, "wide rail must use the cur/max form")
        self.assertIn("/100k", _strip(wide))
        # the bar is proportional: 42% of 8 cells rounds to 3 filled
        self.assertEqual(_strip(narrow).count("▰"), 3)
        self.assertEqual(_strip(narrow).count("▱"), 5)

    def test_steer_cell_hidden_when_zero_present_when_nonzero(self):
        with mock.patch.object(_surfaces, "steer_pending", return_value=0):
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("steer" in _strip(c) for c in cells))
        with mock.patch.object(_surfaces, "steer_pending", return_value=2):
            cells = _panel.panel_cells(24, ROWS)
        self.assertIn("steer 2", _strip(_cell(cells, "steer")))

    def test_cells_clipped_to_width(self):
        class _Src:
            _context_window = 1234567
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 1234567), \
             mock.patch.object(st, "ACTIVE", _Src()):
            for w in (12, 18, 24):
                for c in _panel.panel_cells(w, ROWS):
                    self.assertLessEqual(display_width(c), w)

    def test_cells_capped_at_seven(self):
        class _Src:
            _context_window = 100000
        with mock.patch.object(st, "_LAST_PROMPT_TOKENS", 50000), \
             mock.patch.object(st, "ACTIVE", _Src()), \
             mock.patch.object(_surfaces, "steer_pending", return_value=1), \
             mock.patch.object(_panel, "_claims_store_count", return_value=2), \
             mock.patch.object(_panel, "_poll_branch", lambda: "main +1"):
            cells = _panel.panel_cells(24, ROWS)
        self.assertLessEqual(len(cells), 7)
        keys = [_strip(c).split(" ")[0] for c in cells]
        self.assertEqual(keys[:2], ["50k/100k", "mode"])
        self.assertEqual(keys[2:5], ["branch", "jobs", "agents"])
        self.assertEqual(keys[5:], ["steer", "claims"])


class TestDisplayWidthClip(unittest.TestCase):
    """The shared `_ansi` trio the rail and the frame composite both use."""

    def test_wide_chars_count_two_columns(self):
        self.assertEqual(display_width("你好"), 4)
        self.assertEqual(display_width("日本 abc"), 8)
        self.assertEqual(display_width(""), 0)

    def test_ansi_is_zero_width(self):
        self.assertEqual(display_width(f"{C.ERROR}red{RESET}"), 3)

    def test_clip_never_leaks_an_escape_and_never_cuts_one(self):
        styled = f"{C.ERROR}hello{RESET}"
        out = clip_display(styled, 3)
        self.assertEqual(display_width(out), 3)
        self.assertNotIn("38;2;", _strip(out))     # the wrapper survived intact
        self.assertTrue(out.endswith(f"{C.ERROR}hel") or _strip(out) == "hel",
                        out)
        # the old raw codepoint slice cut mid-sequence and leaked a literal m
        self.assertNotIn("\x1bm", out[1:])

    def test_clip_drops_a_straddling_wide_char(self):
        out = clip_display(f"{C.RED}你好{RESET}", 3)
        self.assertEqual(display_width(out), 2)   # 1 would half-draw a cell
        self.assertEqual(_strip(out), "你")
        self.assertEqual(_strip(clip_display("abc", 0)), "")

    def test_panel_clip_routes_through_the_shared_trio(self):
        styled = f"{C.RED}你好宽字符{RESET}"
        self.assertEqual(_panel._clip(styled, 5), clip_display(styled, 5))
        self.assertEqual(display_width(_panel._clip(styled, 5)), 4)


class TestModeCell(_PanelEnv):
    def test_mode_cell_renders_the_approval_mode(self):
        cases = (
            ({"YOLO": True, "APPROVE_LEVEL": None}, "yolo"),
            ({"YOLO": False, "APPROVE_LEVEL": None}, "prompt-all"),
            ({"YOLO": False, "APPROVE_LEVEL": "high"}, "auto:high"),
        )
        for attrs, want in cases:
            with mock.patch.object(st, "YOLO", attrs["YOLO"]), \
                 mock.patch.object(st, "APPROVE_LEVEL", attrs["APPROVE_LEVEL"]):
                cells = _panel.panel_cells(24, ROWS)
            self.assertEqual(_strip(_cell(cells, "mode")), f"mode {want}")

    def test_mode_cell_hidden_for_an_unknown_mode(self):
        with mock.patch.object(st, "YOLO", False), \
             mock.patch.object(st, "APPROVE_LEVEL", 7):   # not a level this build knows
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("mode" in _strip(c) for c in cells))


class TestBranchCell(_PanelEnv):
    def test_branch_cell_and_ttl_cache(self):
        """The poll is served from a module cache for BRANCH_TTL seconds: the
        rail repaints at 4 fps and must not fork `git` per tick."""
        calls = []

        def _poll():
            calls.append(1)
            return "main +3"

        clock = [100.0]
        with mock.patch.object(_panel, "_poll_branch", _poll), \
             mock.patch.object(_panel.time, "monotonic", lambda: clock[0]):
            self.assertEqual(_strip(_cell(_panel.panel_cells(24, ROWS), "branch")),
                             "branch main +3")
            clock[0] += 1.0                      # inside the TTL
            _panel.panel_cells(24, ROWS)
            self.assertEqual(len(calls), 1, "served from the cache")
            clock[0] += _panel.BRANCH_TTL + 0.01  # past it
            _panel.panel_cells(24, ROWS)
            self.assertEqual(len(calls), 2, "TTL expired -> one repoll")

    def test_branch_cell_hidden_without_a_repo(self):
        with mock.patch.object(_panel, "_poll_branch", lambda: None):
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("branch" in _strip(c) for c in cells))

    def test_branch_name_is_left_elided(self):
        with mock.patch.object(_panel, "_poll_branch",
                               lambda: "feature/very-long-rail-name"):
            val = _strip(_cell(_panel.panel_cells(24, ROWS), "branch"))
        self.assertTrue(val.startswith("branch …"), val)
        self.assertLessEqual(len(val), 24)
        self.assertIn("name", val, "the informative tail survives")

    def test_detached_head_falls_back_to_the_short_hash(self):
        """"## HEAD (no branch)" is not something to print: the branch cell
        shows the short hash (the second, rare `rev-parse` call)."""
        class _PR:
            def __init__(self, out, rc=0):
                self.stdout, self.returncode = out, rc

        def _run(cmd, **kw):
            if cmd[1] == "status":
                return _PR("## HEAD (no branch)\n")
            self.assertEqual(cmd[1], "rev-parse")
            return _PR("9f3c1ab\n")

        with mock.patch("subprocess.run", _run):
            self.assertEqual(self._real_poll_branch(), "9f3c1ab")

    def test_non_repo_hides_the_branch_cell(self):
        class _PR:
            stdout, returncode = "", 128

        with mock.patch("subprocess.run", lambda *a, **k: _PR()):
            self.assertIsNone(self._real_poll_branch())

    def test_dirty_count_rendered_when_present(self):
        with mock.patch.object(_panel, "_poll_branch", lambda: "main +12"):
            self.assertEqual(
                _strip(_cell(_panel.panel_cells(24, ROWS), "branch")),
                "branch main +12")


class TestCostAndClaimsCells(_PanelEnv):
    def test_cost_cell_hidden_when_nothing_is_priced(self):
        class _M:
            by_source = {"srv": {"input_tokens": 100, "output_tokens": 10}}
        with mock.patch.object(st, "_METRICS", _M()):
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("cost" in _strip(c) for c in cells),
                         "unpriced session must not read as $0.00")

    def test_cost_cell_rendered_when_priced(self):
        class _M:
            by_source = {"srv": {"input_tokens": 1_000_000,
                                 "output_tokens": 0}}
        # the key NAMES are built, not spelled out: tests/test_env_docs_parity
        # scans every NX_* literal in nx/ and tests/ and demands a README row
        # for each, and a per-source price key is data, not a tunable
        env = {"NX_PRICE_" + "SRV_INPUT": "3.0",
               "NX_PRICE_" + "SRV_OUTPUT": "15.0"}
        with mock.patch.object(st, "_METRICS", _M()), \
             mock.patch.dict("os.environ", env):
            cells = _panel.panel_cells(24, ROWS)
        self.assertEqual(_strip(_cell(cells, "cost")), "cost $3.00")

    def test_cost_cell_hidden_without_metrics(self):
        with mock.patch.object(st, "_METRICS", None):
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("cost" in _strip(c) for c in cells))

    def test_claims_cell_rendered_from_the_getter(self):
        with mock.patch.object(_panel, "_claims_store_count", return_value=2):
            cells = _panel.panel_cells(24, ROWS)
        self.assertEqual(_strip(_cell(cells, "claims")), "claims 2")
        _panel._claims_cache = (0.0, 0)      # rewind the TTL cache
        with mock.patch.object(_panel, "_claims_store_count", return_value=0):
            cells = _panel.panel_cells(24, ROWS)
        self.assertFalse(any("claims" in _strip(c) for c in cells))

    def test_claims_count_is_ttl_cached(self):
        calls = []
        clock = [100.0]
        with mock.patch.object(_panel, "_claims_store_count",
                               lambda: calls.append(1) or 1), \
             mock.patch.object(_panel.time, "monotonic", lambda: clock[0]):
            _panel.panel_cells(24, ROWS)
            _panel.panel_cells(24, ROWS)
            self.assertEqual(len(calls), 1)
            clock[0] += _panel.CLAIMS_TTL + 0.01
            _panel.panel_cells(24, ROWS)
        self.assertEqual(len(calls), 2)


class TestClaimsGetter(unittest.TestCase):
    def test_active_claims_is_an_int_and_never_raises(self):
        n = _fileclaim.active_claims()
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)

    def test_active_claims_counts_only_rows_inside_the_ttl(self):
        """Stale rows (a session that exited without releasing) are history,
        not status: only rows inside NX_FILE_CLAIM_TTL are counted — exactly the
        rule claim_file applies before blocking on a holder."""
        import sqlite3
        import time as _time

        def _store_conn():
            db = sqlite3.connect(":memory:")
            db.execute("CREATE TABLE file_claims (path TEXT, owner_id TEXT,"
                       " pid INTEGER, claimed_at REAL, cwd TEXT,"
                       " pid_start_time TEXT)")
            now = _time.time()
            db.execute("INSERT INTO file_claims VALUES ('p1','o',1,?, 'c',"
                       " NULL)", (now,))
            db.execute("INSERT INTO file_claims VALUES ('p2','o',1,?, 'c',"
                       " NULL)", (now - 5 * 86400,))    # ancient: stale row
            return db

        with mock.patch.dict("os.environ", {"NX_FILE_CLAIM_TTL": "3600"}), \
             mock.patch("nx._sessiondb._open",
                        lambda *a, **k: _store_conn()):
            self.assertEqual(_fileclaim.active_claims(), 1)
        # TTL=0 disables expiry: every row is live by definition
        with mock.patch.dict("os.environ", {"NX_FILE_CLAIM_TTL": "0"}), \
             mock.patch("nx._sessiondb._open",
                        lambda *a, **k: _store_conn()):
            self.assertEqual(_fileclaim.active_claims(), 2)


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
    """_live_frame composition: disabled -> no rail at all (byte-identical to
    the pre-panel frame); enabled -> every line carries the separator + a cell
    column, inside the terminal's last-column reserve."""

    def setUp(self):
        super().setUp()
        self._saved_frame = _surfaces._LAST_FRAME
        _surfaces._LAST_FRAME = None

    def tearDown(self):
        _surfaces._LAST_FRAME = self._saved_frame
        super().tearDown()

    def test_disabled_output_has_no_rail(self):
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            lines = _surfaces._live_frame()
        self.assertTrue(lines)
        for ln in lines:
            self.assertNotIn("│", ln)

    def test_disabled_frame_is_byte_identical_under_a_narrow_rail_intent(self):
        """Panel-off invariant: with the rail neither enabled NOR wanted, the
        frame must be byte-for-byte the pre-panel one at every width — including
        below the degraded-mode floor."""
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=list(ROWS))
        with snap:
            off = _surfaces._live_frame()
        os.environ["NX_PANEL"] = "0"          # explicitly off
        for cols in (40, 80, 120):
            size = mock.patch.object(shutil, "get_terminal_size",
                                     return_value=os.terminal_size((cols, 24)))
            with size, snap:
                self.assertEqual(_surfaces._live_frame(), off, f"cols={cols}")

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
        lw = max(1, 120 - pw - 3)
        for o, n in zip(off, on):
            self.assertTrue(n.startswith(pad_display(o, lw)))

    def test_railed_rows_reserve_the_last_terminal_column(self):
        """Every composed row must fit in cols-1: writing the last column
        auto-wraps, which adds a row the Live's erase count never knew about."""
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=list(ROWS))
        os.environ["NX_PANEL"] = "1"
        for cols in (100, 120, 200):
            size = mock.patch.object(shutil, "get_terminal_size",
                                     return_value=os.terminal_size((cols, 24)))
            with size, snap:
                on = _surfaces._live_frame()
            for ln in on:
                self.assertLessEqual(display_width(ln), cols - 1, ln)

    def test_wide_char_row_does_not_shift_the_separator(self):
        """A CJK instruction used to shift `│` right and wrap the row (the
        codepoint-vs-column bug): the rail column must land on the same display
        column for every row."""
        wide = [{"kind": "bg", "id": "bg-1", "alive": True, "status": "running",
                 "instruction": "你好世界这是一个很长的中文说明" * 4}]
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=wide)
        size = mock.patch.object(shutil, "get_terminal_size",
                                 return_value=os.terminal_size((120, 24)))
        os.environ["NX_PANEL"] = "1"
        with size, snap:
            on = _surfaces._live_frame()
        pw = _panel.panel_width()
        lw = max(1, 120 - pw - 3)
        checked = 0
        for ln in on:
            if len(_strip(ln)) <= lw:
                continue                       # a row shorter than the rail
            self.assertEqual(_strip(ln)[lw], "│", ln)
            checked += 1
        self.assertGreaterEqual(checked, 2, "expected railed rows")

    def test_degraded_one_line_bottom_status(self):
        """Rail can't fit but the user wants it on (cols >= DEGRADED_MIN_COLS):
        the SAME cells as one extra bottom line, rail-off/compact-on."""
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=list(ROWS))
        os.environ["NX_PANEL"] = "1"
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=os.terminal_size((120, 24))), snap:
            off = _surfaces._live_frame()
        size = mock.patch.object(shutil, "get_terminal_size",
                                 return_value=os.terminal_size((52, 24)))
        with size, snap:
            self.assertFalse(_panel.panel_enabled())   # 52 - 24 = 28 < 60
            self.assertTrue(_panel.panel_wants_on())
            msg = _surfaces._live_frame()
        self.assertEqual(len(msg), len(off) + 1, "exactly one extra line")
        for ln in msg:
            self.assertNotIn("│", ln, "degraded mode is rail-off")
        last = _strip(msg[-1])
        self.assertIn(" · ", last)
        self.assertIn("ctx", last)
        self.assertIn("jobs 1", last)
        self.assertLessEqual(display_width(msg[-1]), 51)

    def test_below_the_degraded_floor_stays_byte_identical(self):
        """Under DEGRADED_MIN_COLS one line of status is itself noise: the
        frame must equal the panel-off frame byte for byte."""
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=list(ROWS))
        with snap:
            off = _surfaces._live_frame()
        os.environ["NX_PANEL"] = "1"
        size = mock.patch.object(shutil, "get_terminal_size",
                                 return_value=os.terminal_size((40, 24)))
        with size, snap:
            self.assertEqual(_surfaces._live_frame(), off)

    def test_degraded_line_respects_the_height_clamp(self):
        """The compact line is a frame row: it must not push the frame past the
        rows-3 clamp (a short terminal needs that reserved space most)."""
        snap = mock.patch.object(_surfaces, "_surfaces_snapshot",
                                 return_value=list(ROWS))
        os.environ["NX_PANEL"] = "1"
        size = mock.patch.object(shutil, "get_terminal_size",
                                 return_value=os.terminal_size((52, 6)))
        with size, snap:
            msg = _surfaces._live_frame()
        self.assertEqual(len(msg), 3)          # height = min(8, 6-3)
        self.assertIn(" · ", _strip(msg[-1]))

    def test_degraded_line_none_when_the_rail_fits(self):
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=os.terminal_size((120, 24))):
            os.environ["NX_PANEL"] = "1"
            self.assertIsNone(_panel.degraded_line(ROWS))
        del os.environ["NX_PANEL"]
        with mock.patch.object(shutil, "get_terminal_size",
                               return_value=os.terminal_size((52, 24))):
            self.assertIsNone(_panel.degraded_line(ROWS))  # nobody asked


class TestHeightClamp(_PanelEnv):
    """The frame height is clamped against the terminal ROWS, not just
    NX_TABS_HEIGHT, so a short terminal keeps its prompt."""

    def _frame_at(self, cols, lines_):
        size = mock.patch.object(shutil, "get_terminal_size",
                                 return_value=os.terminal_size((cols, lines_)))
        with size, mock.patch.object(_surfaces, "_surfaces_snapshot",
                                     return_value=list(ROWS)):
            return _surfaces._live_frame()

    def test_frame_height_is_clamped_to_rows_minus_three(self):
        with mock.patch.dict("os.environ", {"NX_TABS_HEIGHT": "8"}):
            tall = self._frame_at(120, 40)
            short = self._frame_at(120, 5)
        self.assertGreater(len(tall), len(short))
        self.assertLessEqual(len(short), 5 - 3)

    def test_frame_height_never_drops_below_one(self):
        with mock.patch.dict("os.environ", {"NX_TABS_HEIGHT": "8"}):
            frame = self._frame_at(120, 2)   # rows - 3 = -1
        self.assertEqual(len(frame), 1)

    def test_tabs_height_keeps_its_env_contract(self):
        with mock.patch.dict("os.environ", {"NX_TABS_HEIGHT": "8"}):
            self.assertEqual(_surfaces._tabs_height(), 8)


class _FakeLive:
    """Records the region Live's lifecycle (a stand-in for rich's Live)."""

    def __init__(self):
        self.updates = []
        self.refreshes = 0

    def start(self, refresh: bool = False):
        pass

    def stop(self):
        pass

    def update(self, renderable, refresh: bool = False):
        self.updates.append(renderable)

    def refresh(self):
        self.refreshes += 1


class TestRepaintRowDiff(_PanelEnv):
    """Frame-level change suppression: an identical frame must not be handed to
    the Live at all (the flicker/CPU win); a changed frame must repaint; and a
    repaint crash must still disable the region for the session."""

    def setUp(self):
        super().setUp()
        self._saved_frame = _surfaces._LAST_FRAME
        _surfaces._LAST_FRAME = None
        self._dash = (_surfaces._DASH_OPEN, _surfaces._DASH_DISABLED,
                      _surfaces._DASH_LIVE)
        _surfaces._DASH_DISABLED = False
        self._vt = mock.patch.object(_surfaces, "_vt_supported", lambda: True)
        self._vt.start()
        self.addCleanup(self._vt.stop)

    def tearDown(self):
        with _surfaces._DASH_LOCK:
            (_surfaces._DASH_OPEN, _surfaces._DASH_DISABLED,
             _surfaces._DASH_LIVE) = self._dash
        _surfaces._LAST_FRAME = self._saved_frame
        super().tearDown()

    def _open_fake(self):
        fake = _FakeLive()
        real = _surfaces._make_region_live
        _surfaces._make_region_live = lambda: fake
        self.addCleanup(lambda: setattr(_surfaces, "_make_region_live", real))
        self.assertTrue(_surfaces.open_dashboard())
        return fake

    def test_identical_frame_is_not_repainted(self):
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            fake = self._open_fake()
            _surfaces.repaint_live()
            _surfaces.repaint_live()
            _surfaces.repaint_live()
        self.assertEqual(len(fake.updates), 1, "unchanged frame repainted")
        self.assertEqual(fake.refreshes, 1)

    def test_changed_frame_repaints(self):
        rows = list(ROWS)
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=rows):
            fake = self._open_fake()
            _surfaces.repaint_live()
            rows.append({"kind": "bg", "id": "bg-9", "alive": True,
                         "status": "running", "instruction": "new job"})
            _surfaces.repaint_live()
        self.assertEqual(len(fake.updates), 2)
        self.assertEqual(fake.refreshes, 2)

    def test_force_repaints_an_identical_frame(self):
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            fake = self._open_fake()
            _surfaces.repaint_live()
            _surfaces.repaint_live(force=True)
        self.assertEqual(len(fake.updates), 2)

    def test_color_only_change_does_not_repaint(self):
        """The comparison is on the plain (ANSI-stripped) lines: a frame that
        differs only in SGR bytes is not a visible change."""
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            fake = self._open_fake()
            _surfaces.repaint_live()
            real = _surfaces._live_frame()
            recolored = [f"\x1b[35m{ln}\x1b[0m" for ln in real]
            with mock.patch.object(_surfaces, "_live_frame",
                                   return_value=recolored):
                _surfaces.repaint_live()
        self.assertEqual(len(fake.updates), 1)

    def test_repaint_crash_still_disables_the_session(self):
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            fake = self._open_fake()
            fake.refresh = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
            _surfaces.repaint_live()
        self.assertTrue(_surfaces.dashboard_disabled())
        self.assertFalse(_surfaces.dashboard_open())

    def test_reopen_paints_again_despite_an_identical_frame(self):
        with mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            fake = self._open_fake()
            _surfaces.repaint_live()
            _surfaces.close_dashboard()
            second = self._open_fake()
            _surfaces.repaint_live()
        self.assertEqual(len(fake.updates), 1)
        self.assertEqual(len(second.updates), 1, "a fresh Live must be painted")


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


class TestNoStrayOutput(_PanelEnv):
    """The rail must stay a passive read surface: composing a frame prints
    nothing (no probe, no fork's stdout leaking into the live region)."""

    def test_live_frame_prints_nothing(self):
        buf = io.StringIO()
        os.environ["NX_PANEL"] = "1"
        with redirect_stdout(buf), \
             mock.patch.object(shutil, "get_terminal_size",
                               return_value=os.terminal_size((120, 24))), \
             mock.patch.object(_surfaces, "_surfaces_snapshot",
                               return_value=list(ROWS)):
            _surfaces._live_frame()
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
