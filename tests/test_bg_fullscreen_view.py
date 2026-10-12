"""The /bg fullscreen view (0.148.6): render, empty state, keys, dispatch.

Background jobs used to ride in the /tabs rows; they now have their own
Phase-D view (`_bg_completions.BackgroundJobsView`, entered by bare `/bg` or
Ctrl+B) while /tabs keeps only the live-status rows. Modelled on
tests/test_todo_fullscreen_view.py: the view (rows from `list_bg_jobs()`, the
shared `_surfaces._process_status` word, an explicit empty state, j/k
scrolling), the /tabs split, and the dispatch path of both new keybindings —
all without needing a real terminal.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-bg-fs-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _fullscreen  # noqa: E402
from nx import _state as st  # noqa: E402,F401


def _seed_jobs(monkeypatch, jobs):
    """Patch the registry the view reads (refresh() imports `list_bg_jobs`
    lazily, so the patch must land on the defining module)."""
    import nx._bash_jobs as bj
    import nx._repl_settings_cmds as rsc
    monkeypatch.setattr(bj, "list_bg_jobs", lambda mine_only=True: jobs)
    # the inline fallback (`_cmd_bg`) holds its OWN imported reference
    monkeypatch.setattr(rsc, "list_bg_jobs", lambda mine_only=True: jobs, raising=False)


def _job(pid, alive, command, *, killed=False, tail="", started=None):
    return {"pid": pid, "alive": alive, "command": command, "started": started,
            "log_path": "", "killed": killed, "tail": tail}


def test_bg_view_renders_rows_and_status(monkeypatch):
    from nx import _bg_completions, _render
    _seed_jobs(monkeypatch, [_job("1234", True, "pytest -x -q",
                                  tail="collecting"),
                             _job("999", False, "make docs", killed=True)])
    body = "\n".join(_bg_completions.bg_jobs_view().render(80, 24))
    assert _render.BOX_TL in body and _render.BOX_BL in body
    assert "pytest -x -q" in body and "make docs" in body
    # the shared status word: alive means running, killed stays killed (C1)
    assert "running" in body and "killed" in body
    # counts ride in the panel title; the footer points at the kill form
    assert "2 shown" in body and "1 running" in body
    assert "/bg kill" in body


def test_bg_view_windows_finished_rows_and_scrolls(monkeypatch):
    from nx import _bg_completions, _keys
    _seed_jobs(monkeypatch, [_job(str(100 + i), False, f"job {i}")
                             for i in range(9)])
    view = _bg_completions.bg_jobs_view()
    view.render(80, 24)   # first paint refreshes the rows (TabsView pattern)
    # only the newest _BG_DONE_KEEP finished rows are shown (a live view; the
    # full history stays in the inline /bg print and the log dir)
    assert len(view.rows) == _bg_completions._BG_DONE_KEEP
    assert len(view.jobs) == 9
    view.sel = 0
    assert view.on_key("j", None) == "repaint" and view.sel == 1
    assert view.on_key("k", None) == "repaint" and view.sel == 0
    assert view.on_key("q", None) == "quit"
    # every advertised key is a key the view actually handles
    for key, _meaning in _keys.keys_for("bgjobs"):
        assert view.on_key("j" if key == "j/k" else key, None) in (
            "repaint", "quit", "run"), key


def test_bg_view_empty_state(monkeypatch):
    from nx import _bg_completions, _render
    _seed_jobs(monkeypatch, [])
    body = "\n".join(_bg_completions.bg_jobs_view().render(80, 24))
    assert _render.BOX_TL in body, "empty state must still be a panel"
    assert "no background jobs" in body
    assert "bg jobs · none" in body


def test_tabs_view_no_longer_lists_bg_jobs(monkeypatch):
    """The split: /tabs keeps the live rows, /bg owns the job rows."""
    from nx import _surfaces
    live = {"kind": "main", "id": "main", "alive": True, "status": "running",
            "instruction": "main", "tail": ""}
    bg = {"kind": "bg", "id": "bg-1", "alive": True, "status": "running",
          "instruction": "sleep 99", "tail": ""}
    monkeypatch.setattr(
        _surfaces, "_surfaces_snapshot",
        lambda **kw: [live] if not kw.get("include_bg", True) else [live, bg])
    view = _surfaces.tabs_view()
    body = "\n".join(view.render(80, 24))
    assert [r["kind"] for r in view.rows] == ["main"]
    assert "sleep 99" not in body
    assert "1 row(s)" in body


def test_bg_command_opens_the_view_with_the_inline_fallback(capsys, monkeypatch):
    """Dispatch: bare `/bg` opens BackgroundJobsView through
    `_fullscreen.open_view`, and the fallback is the pre-view inline print;
    an argument (`/bg kill 1`, `/bg all`) stays on the one-shot `_cmd_bg`."""
    import importlib
    rp = importlib.import_module("nx._repl")
    _seed_jobs(monkeypatch, [_job("4242", True, "sleep 1", tail="t")])
    opened = []

    def _fake_open(view, fallback=None):
        opened.append(view)
        fallback()
        return True

    monkeypatch.setattr(_fullscreen, "open_view", _fake_open)
    rp._cmd_bg_view("/bg", None)
    assert len(opened) == 1 and opened[0].surface == "bgjobs"
    out = capsys.readouterr().out
    assert "4242" in out and "sleep 1" in out

    # /bg kill and /bg all never take the alternate screen
    opened.clear()
    rp._cmd_bg_view("/bg all", None)
    assert opened == []
    assert capsys.readouterr().out.strip() != ""


def test_ctrl_b_and_ctrl_x_return_the_right_commands():
    """The chatbox's single-byte shorthands: the byte-to-slash-string contract
    Ctrl+G followed. Source-level (the editor needs a real tty), plus the
    precedence check that pins the byte audit behind the choice of \\x18."""
    import inspect
    from nx import _chatbox
    src = inspect.getsource(_chatbox._chatbox_raw)
    assert 'elif c == "\\x02":' in src and 'return "/bg"' in src
    assert 'elif c == "\\x18":' in src and 'return "/diff"' in src
    # ^D (EOF on an empty buffer) is TAKEN and its branch comes FIRST: \x04
    # stays the quit gesture, which is why /diff got \x18 instead
    assert src.index('elif c == "\\x04":') < src.index('elif c == "\\x18":')
    # and the erase-then-return discipline Ctrl+T/Ctrl+G established
    block = src[src.index('elif c == "\\x02":'):src.index('elif c == "\\x07":')]
    assert "erase_stale_frame()" in block and "finish()" in block
    assert _fullscreen.bgjobs().surface == "bgjobs"
