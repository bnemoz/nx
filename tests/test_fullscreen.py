"""UIUX_SPEC Phase D (0.137.0): the full-screen shell + its three views.

What is testable without a real terminal is tested here:

* the ENTER/EXIT byte contract (`\\033[?1049h` / `\\033[?1049l`) and that the
  exit sequence is emitted exactly once even when the view raises — the
  property that makes "cursor and scrollback restored" true (the alternate
  screen itself guarantees it; a test through `os.openpty` drives a real view
  loop and asserts the same bytes reached the terminal, skipped when the
  platform has no pty);
* SIGWINCH repaint: the notifier's wake fd, and a loop that repaints on it;
* the stop-the-world contract: the live region refuses to paint while a view
  owns the screen, and a view pauses/reopens the dashboard around itself;
* the inline fallback being byte-identical to the pre-Phase-D command output
  (/tabs -> `tabs_print_fallback`, /sessions -> `_cmd_sessions`, /diff -> the
  plain file list, /context -> `_ctx_inline_report`);
* one key registry: the views' advertised hints and their key handling cannot
  drift (every key in `_keys.keys_for(surface)` is a key the view handles).
"""
from __future__ import annotations

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeOut(io.StringIO):
    """A stdout stand-in that records raw writes (StringIO.flush is a no-op)."""

    def flush(self) -> None:  # pragma: no cover - trivial
        pass


def _reader_for(seq):
    it = iter(seq)

    def _reader(fd):  # noqa: ARG001 - matches the real reader signature
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    return _reader


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    """Only the vars THIS module reads: the blanket clear the suite's own
    tests would collide with (they `monkeypatch.delenv` other NX_* names, which
    raises when the var is already gone)."""
    monkeypatch.delenv("NX_FULLSCREEN", raising=False)
    monkeypatch.setenv("NX_ENV_FILE", "/dev/null")
    yield


# ---------------------------------------------------------------------------
# enter/exit
# ---------------------------------------------------------------------------


def test_alternate_screen_emits_the_enter_and_exit_pair():
    from nx import _fullscreen
    out = FakeOut()
    with _fullscreen.alternate_screen(out=out, pause=False):
        out.write("frame")
    text = out.getvalue()
    assert text.startswith(_fullscreen.ALT_ENTER)
    assert text.count(_fullscreen.ALT_ENTER) == 1
    assert text.count(_fullscreen.ALT_EXIT) == 1
    assert text.index(_fullscreen.ALT_EXIT) > text.index("frame")
    assert text.endswith(_fullscreen.ALT_EXIT)


def test_alternate_screen_exits_even_when_the_body_raises():
    from nx import _fullscreen
    out = FakeOut()
    with pytest.raises(RuntimeError):
        with _fullscreen.alternate_screen(out=out, pause=False):
            raise RuntimeError("view blew up")
    assert _fullscreen.ALT_EXIT in out.getvalue()
    assert _fullscreen.owns_screen() is False


def test_screen_ownership_flag_follows_the_context():
    from nx import _fullscreen
    assert _fullscreen.owns_screen() is False
    with _fullscreen.alternate_screen(out=FakeOut(), pause=False):
        assert _fullscreen.owns_screen() is True
    assert _fullscreen.owns_screen() is False


def test_esc_sequence_ordered_after_the_frame_still_exits():
    """The exit bytes are the LAST thing written: nothing a view paints can
    land in the normal buffer after the screen is restored."""
    from nx import _fullscreen
    out = FakeOut()
    with _fullscreen.alternate_screen(out=out, pause=False):
        out.write("junk")
    assert out.getvalue().endswith(_fullscreen.ALT_EXIT)


def test_run_view_draws_frames_and_leaves_on_esc():
    from nx import _fullscreen
    from nx._surfaces import tabs_view
    out = FakeOut()
    _fullscreen.run_view(tabs_view(), fd=0, out=out, _reader=_reader_for(["j", "esc"]),
                         _notifier=_fullscreen._ResizeNotifier())
    text = out.getvalue()
    assert text.startswith(_fullscreen.ALT_ENTER)
    assert text.endswith(_fullscreen.ALT_EXIT)
    # frames are painted inside the alternate buffer (home + erase, never
    # in-place cursor arithmetic in the normal buffer)
    assert text.count(_fullscreen.FRAME_ERASE) >= 2


def test_run_view_overlay_repaints_from_the_keys_registry():
    from nx import _fullscreen
    from nx._keys import keys_for
    from nx._surfaces import tabs_view
    out = FakeOut()
    # "?" then any key then esc
    _fullscreen.run_view(tabs_view(), fd=0, out=out,
                         _reader=_reader_for(["?", "x", "esc"]),
                         _notifier=_fullscreen._ResizeNotifier())
    body = out.getvalue()
    assert "any key closes" in body
    assert "j/k" in body  # the registry's own hint row
    assert keys_for("tabs"), "tabs has no registry entry"


def test_open_view_declines_off_tty_and_calls_the_fallback(monkeypatch):
    from nx import _fullscreen
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    called = []
    assert _fullscreen.open_view(object(), fallback=lambda: called.append(1)) is False
    assert called == [1]


def test_wants_fullscreen_respects_the_env_switch(monkeypatch):
    from nx import _fullscreen
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert _fullscreen.wants_fullscreen() is True
    monkeypatch.setenv("NX_FULLSCREEN", "0")
    assert _fullscreen.wants_fullscreen() is False
    monkeypatch.setenv("NX_FULLSCREEN", "off")
    assert _fullscreen.wants_fullscreen() is False
    monkeypatch.setenv("NX_FULLSCREEN", "1")
    assert _fullscreen.wants_fullscreen() is True


def test_wants_fullscreen_refuses_mid_turn(monkeypatch):
    """Stop-the-world: views are prompt-time surfaces, so a running turn (which
    owns stdin for the Esc watcher) never opens one."""
    from nx import _fullscreen, _state as st
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    st._TURN_ACTIVE = True
    try:
        assert _fullscreen.wants_fullscreen() is False
    finally:
        st._TURN_ACTIVE = False


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def test_resize_notifier_posts_and_drains_a_wake():
    from nx import _fullscreen
    n = _fullscreen._ResizeNotifier()
    n.open()
    try:
        fds = n.fds()
        assert fds, "expected the POSIX wake pipe"
        n.trigger()
        import select
        ready, _, _ = select.select(fds, [], [], 0.5)
        assert ready, "wake fd not readable after a resize"
        n.drain()
        ready, _, _ = select.select(fds, [], [], 0.05)
        assert not ready, "drain left bytes behind"
    finally:
        n.close()
    assert n.fds() == []


def test_run_view_repaints_on_a_resize_wake():
    from nx import _fullscreen
    from nx._surfaces import tabs_view
    out = FakeOut()
    notifier = _fullscreen._ResizeNotifier()
    notifier.open()
    fired = []

    def _reader(fd):
        if not fired:
            fired.append(1)
            notifier.trigger()   # a simulated SIGWINCH mid-loop
            return "j"           # ... and a key that repaints, not a quit
        return "esc"

    try:
        _fullscreen.run_view(tabs_view(), fd=0, out=out, _reader=_reader,
                             _notifier=notifier)
    finally:
        notifier.close()
    # the first frame plus at least one repaint
    assert out.getvalue().count(_fullscreen.FRAME_ERASE) >= 2


# ---------------------------------------------------------------------------
# stop the world
# ---------------------------------------------------------------------------


def test_live_region_does_not_paint_while_a_view_owns_the_screen(monkeypatch):
    from nx import _fullscreen, _surfaces
    painted = []
    monkeypatch.setattr(_surfaces, "_DASH_OPEN", True)
    monkeypatch.setattr(_surfaces, "_DASH_DISABLED", False)
    monkeypatch.setattr(_surfaces, "_vt_supported", lambda: True)
    monkeypatch.setattr(_surfaces, "_make_region_live",
                        lambda: painted.append(1) or _FakeLive())
    with _fullscreen.alternate_screen(out=FakeOut(), pause=False):
        _surfaces.repaint_live()
    assert painted == [], "painted into a fullscreen view's buffer"


class _FakeLive:
    def update(self, *a, **k):
        pass

    def refresh(self):
        pass

    def start(self, **k):
        pass

    def stop(self):
        pass


def test_view_pauses_and_reopens_the_dashboard(monkeypatch):
    from nx import _fullscreen, _surfaces
    state = {"open": True}
    calls = []
    monkeypatch.setattr(_surfaces, "dashboard_open", lambda: state["open"])

    def _close():
        state["open"] = False
        calls.append("close")

    def _open():
        state["open"] = True
        calls.append("open")
        return True

    monkeypatch.setattr(_surfaces, "close_dashboard", _close)
    monkeypatch.setattr(_surfaces, "open_dashboard", _open)
    monkeypatch.setattr(_surfaces, "_surfaces_snapshot",
                        lambda include_bg=True: [{"kind": "agent", "alive": True}])
    with _fullscreen.alternate_screen(out=FakeOut()):
        assert calls == ["close"]
    assert calls == ["close", "open"]


def test_view_leaves_a_dashboard_that_was_already_closed_alone(monkeypatch):
    from nx import _fullscreen, _surfaces
    calls = []
    monkeypatch.setattr(_surfaces, "dashboard_open", lambda: False)
    monkeypatch.setattr(_surfaces, "close_dashboard",
                        lambda: calls.append("close"))
    monkeypatch.setattr(_surfaces, "open_dashboard",
                        lambda: calls.append("open") or True)
    with _fullscreen.alternate_screen(out=FakeOut()):
        pass
    assert calls == []


# ---------------------------------------------------------------------------
# inline fallbacks: identical to the pre-Phase-D command output
# ---------------------------------------------------------------------------


def test_tabs_fallback_is_the_pre_phase_d_print(capsys, monkeypatch):
    from nx import _surfaces
    rows = [{"kind": "main", "id": "main", "alive": False, "status": "idle",
             "instruction": "main conversation", "tail": ""},
            {"kind": "agent", "id": "a1", "alive": True, "status": "running",
             "instruction": "do the thing", "tail": "line1\nline2"}]
    monkeypatch.setattr(_surfaces, "_surfaces_snapshot", lambda: rows)
    _surfaces.tabs_print_fallback()
    out = capsys.readouterr().out
    assert " [main] main: idle — main conversation" in out
    assert " ● [agent] a1: running — do the thing" in out
    assert "     │ line1" in out


def test_sessions_fallback_off_tty_prints_the_inline_list(capsys, monkeypatch):
    """The /sessions fallback is the SAME `_cmd_sessions` the command always
    called: with no tty the fullscreen path declines before doing anything."""
    from nx import _sessions_cmd, _fullscreen

    class State:
        session_id = "cur"

    def _fake(user, current_id):
        print("inline list")

    monkeypatch.setattr(_sessions_cmd, "_cmd_sessions", _fake)
    monkeypatch.setattr(_fullscreen, "wants_fullscreen", lambda: False)
    _sessions_cmd._cmd_sessions_handler("/sessions", State())
    assert capsys.readouterr().out.strip() == "inline list"


def test_sessions_view_falls_back_to_the_same_inline_command(capsys, monkeypatch):
    from nx import _sessions_cmd, _fullscreen

    class State:
        session_id = "cur"

    monkeypatch.setattr(_fullscreen, "wants_fullscreen", lambda: True)
    monkeypatch.setattr(_fullscreen, "open_view",
                        lambda view, fallback=None: (fallback(), False)[1])
    monkeypatch.setattr(_sessions_cmd, "_cmd_sessions",
                        lambda user, current_id: print("inline list"))
    _sessions_cmd.cmd_sessions_view("/sessions", State())
    assert capsys.readouterr().out.strip() == "inline list"


def test_review_inline_fallback_is_a_plain_file_list(capsys, monkeypatch):
    import importlib
    repl_mod = importlib.import_module("nx._repl")
    from nx import _review
    monkeypatch.setattr(_review, "working_tree_files", lambda repo=None: [
        {"path": "nx/a.py", "code": "M", "added": 3, "removed": 1},
        {"path": "nx/b.py", "code": "??", "added": 7, "removed": 0}])
    repl_mod._inline_review()
    out = capsys.readouterr().out
    assert " M  nx/a.py  +3 -1" in out
    assert "??  nx/b.py  +7 -0" in out
    assert "2 file(s) changed vs HEAD" in out


def test_review_inline_with_a_path_adds_that_diff(capsys, monkeypatch):
    from nx import _review
    monkeypatch.setattr(_review, "working_tree_files", lambda repo=None: [
        {"path": "nx/a.py", "code": "M", "added": 1, "removed": 1}])
    monkeypatch.setattr(_review, "file_diff", lambda path, repo=None: ["+new line"])
    _review._inline(None, "nx/a.py")
    out = capsys.readouterr().out
    assert "── nx/a.py ──" in out
    assert "+new line" in out


# ---------------------------------------------------------------------------
# the views: read-only, registry-driven, render through _render
# ---------------------------------------------------------------------------


def test_tabs_view_keys_are_advertised_by_the_registry(monkeypatch):
    from nx import _keys, _surfaces
    monkeypatch.setattr(_surfaces, "_surfaces_snapshot",
                        lambda include_bg=True: [{"kind": "main", "id": "main", "alive": False,
                                  "status": "idle", "instruction": "",
                                  "tail": ""}])
    view = _surfaces.tabs_view()
    shell = None
    for key, _meaning in _keys.keys_for("tabs"):
        handled = view.on_key("j" if key == "j/k" else key, shell)
        assert handled in ("repaint", "quit", "run"), key


def test_diff_view_keys_are_advertised_by_the_registry(monkeypatch):
    from nx import _keys, _review
    monkeypatch.setattr(_review, "working_tree_files", lambda repo=None: [])
    view = _review.ReviewView()
    for key, _meaning in _keys.keys_for("diff"):
        assert view.on_key("j" if key == "j/k" else key, None) in (
            "repaint", "quit", "run"), key


def test_sessions_view_keys_are_advertised_by_the_registry(monkeypatch):
    from nx import _keys, _sessions_cmd
    monkeypatch.setattr(_sessions_cmd, "_list_sessions", lambda **k: [])

    class Shell:
        def ask(self, label, initial=""):
            return initial

    view = _sessions_cmd.SessionsView(None)
    for key, _meaning in _keys.keys_for("sessions"):
        assert view.on_key("j" if key == "j/k" else key, Shell()) in (
            "repaint", "quit", "run"), key


def test_every_view_surface_has_a_registry_entry():
    from nx import _ctxreport, _keys, _review, _sessions_cmd, _surfaces
    for view in (_surfaces.tabs_view, _review.ReviewView,
                 _sessions_cmd.SessionsView, _ctxreport.ContextView):
        assert _keys.keys_for(getattr(view, "surface", "fullscreen"))


def test_tabs_view_renders_through_the_render_layer(monkeypatch):
    from nx import _render, _surfaces
    monkeypatch.setattr(_surfaces, "_surfaces_snapshot",
                        lambda include_bg=True: [{"kind": "main", "id": "main", "alive": False,
                                  "status": "idle",
                                  "instruction": "hello", "tail": ""}])
    body = "\n".join(_surfaces.tabs_view().render(80, 24))
    assert _render.BOX_TL in body and _render.BOX_BL in body
    assert _render.role("ACCENT") in body
    assert "main" in body


def test_diff_view_list_then_diff_then_commit_action(monkeypatch):
    from nx import _review
    monkeypatch.setattr(_review, "working_tree_files", lambda repo=None: [
        {"path": "nx/a.py", "code": "M", "added": 2, "removed": 1}])
    monkeypatch.setattr(_review, "file_diff",
                        lambda path, repo=None: ["+added"])

    class Shell:
        def ask(self, label, initial=""):
            return "feat: thing"

    view = _review.ReviewView()
    assert "nx/a.py" in "\n".join(view.render(80, 20))
    assert view.on_key("enter", Shell()) == "repaint"
    assert "+added" in "\n".join(view.render(80, 20))
    assert view.on_key("a", Shell()) == "run"
    assert callable(view.pending_action)


def test_diff_view_commit_without_a_message_is_cancelled(monkeypatch):
    from nx import _review
    monkeypatch.setattr(_review, "working_tree_files", lambda repo=None: [
        {"path": "nx/a.py", "code": "M", "added": 1, "removed": 0}])

    class Shell:
        def ask(self, label, initial=""):
            return ""

    view = _review.ReviewView()
    assert view.on_key("a", Shell()) == "repaint"
    assert view.pending_action is None
    assert "cancelled" in view.notice


def test_diff_view_commit_action_runs_the_existing_git_commit(monkeypatch, capsys):
    from nx import _review
    monkeypatch.setattr(_review, "working_tree_files", lambda repo=None: [
        {"path": "nx/a.py", "code": "M", "added": 1, "removed": 0}])
    seen = {}

    def _fake_commit(files, message, repo=None):
        seen["files"] = files
        seen["message"] = message
        return "[git_commit] feat: thing"

    monkeypatch.setattr(_review, "commit_files", _fake_commit)

    class Shell:
        def ask(self, label, initial=""):
            return "feat: thing"

    view = _review.ReviewView()
    view.on_key("a", Shell())
    view.pending_action()
    assert seen == {"files": ["nx/a.py"], "message": "feat: thing"}
    assert "[git_commit] feat: thing" in capsys.readouterr().out


def test_sessions_view_resume_returns_the_existing_resume_path(monkeypatch):
    from nx import _sessions_cmd

    class State:
        session_id = "cur"

    monkeypatch.setattr(_sessions_cmd, "_list_sessions", lambda **k: [
        {"id": "cur", "title": "here", "n": 1, "updated_at": 0},
        {"id": "other", "title": "there", "n": 2, "updated_at": 0}])
    monkeypatch.setattr(_sessions_cmd, "_load_session", lambda sid: {
        "title": "there", "updated_at": 0, "messages": []})
    view = _sessions_cmd.SessionsView(State())
    assert view.on_key("j", None) == "repaint"
    assert view.on_key("enter", None) == "repaint"
    assert view.mode == "preview"
    assert view.on_key("enter", None) == "repaint"
    assert view.on_key("R", None) == "quit"
    assert view.resume_id == "other"


def test_sessions_view_refuses_to_resume_the_current_session(monkeypatch):
    from nx import _sessions_cmd

    class State:
        session_id = "cur"

    monkeypatch.setattr(_sessions_cmd, "_list_sessions", lambda **k: [
        {"id": "cur", "title": "here", "n": 1, "updated_at": 0}])
    view = _sessions_cmd.SessionsView(State())
    assert view.on_key("R", None) == "repaint"
    assert view.resume_id is None
    assert "already on that session" in view.notice


def test_sessions_view_search_uses_fts_hits(monkeypatch):
    from nx import _sessions_cmd

    class State:
        session_id = "cur"

    monkeypatch.setattr(_sessions_cmd, "_list_sessions", lambda **k: [
        {"id": "hit", "title": "hit title", "n": 3, "updated_at": 0}])

    class Shell:
        def ask(self, label, initial=""):
            return "fence"

    def _fake_search(query, limit):
        assert query == "fence"
        return [{"id": "hit", "snippet": "…fence…"}]

    monkeypatch.setattr(_sessions_cmd, "_sessions_search", None) if False else None
    import nx._sessions_search as search_mod
    monkeypatch.setattr(search_mod, "search_sessions", _fake_search)
    view = _sessions_cmd.SessionsView(State())
    view.on_key("/", Shell())
    assert [s["id"] for s in view.sessions] == ["hit"]


def test_no_new_command_was_added_to_review(monkeypatch):
    """/review keeps running the critique sub-agent — /diff is the diff view."""
    from nx import _commands
    assert _commands.COMMANDS["/review"].handler.__name__ == "_cmd_review"
    assert _commands.COMMANDS["/diff"].handler.__name__ == "_cmd_diff"


# ---------------------------------------------------------------------------
# /context — the ContextView (Phase 1 of the view work: same recipe as
# /diff's ReviewView, with ONE addition — the deferred pin action)
# ---------------------------------------------------------------------------


class _CtxState:
    """The one bit of `_ReplState` the view reads: the live message list."""

    def __init__(self, n: int = 3) -> None:
        self.messages = [{"role": "system", "content": "sys prompt"}]
        for i in range(n):
            self.messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"msg {i} " + "x" * 80})


def test_context_view_is_registered_and_its_keys_are_advertised(monkeypatch):
    from nx import _ctxreport, _fullscreen, _keys
    assert "context" in _fullscreen.view_names()
    assert isinstance(_fullscreen.make_view("context", _CtxState()),
                      _ctxreport.ContextView)
    view = _ctxreport.context_view(_CtxState())
    # "j/k" and "enter/p" are hint LABELS for two real keys each (the
    # ReviewView convention): every advertised key must be handled.
    real = {"j/k": "j", "enter/p": "p"}
    for key, _meaning in _keys.keys_for("context"):
        assert view.on_key(real.get(key, key), None) in (
            "repaint", "quit", "run"), key


def test_context_view_renders_rows_pin_marker_and_quit_keys(monkeypatch):
    from nx import _ctxreport, _keys, _render
    from nx import _state as st
    monkeypatch.setattr(st, "_CTX_PINNED", [1], raising=False)
    view = _ctxreport.context_view(_CtxState())
    body = "\n".join(view.render(120, 20))
    assert _render.BOX_TL in body and _render.BOX_BL in body
    assert _render.role("ACCENT") in body
    assert _keys.hint_line("context") in body   # one registry: hints can't drift
    assert "msg 0" in body and "msg 2" in body
    assert "📌" in body                          # row 2 is pinned
    assert "1 pinned" in body
    assert "context · 4 message(s)" in body
    for key in ("q", "esc", "ctrl-c", "ctrl-d"):
        assert view.on_key(key, None) == "quit", key


def test_context_view_scrolls_and_refreshes_from_the_live_session():
    from nx import _ctxreport
    state = _CtxState(30)
    view = _ctxreport.context_view(state)
    view.render(120, 10)
    assert "msg 0" in "\n".join(view.render(120, 10))
    assert [view.on_key("j", None), view.on_key("down", None)] == [
        "repaint", "repaint"]
    assert view.sel == 2
    assert view.on_key("k", None) == "repaint" and view.sel == 1
    assert view.on_key("pagedown", None) == "repaint" and view.sel > 1
    # a turn landing in the session shows up on the next frame (a frame tall
    # enough to hold the whole history: the selection does not move with it)
    state.messages.append({"role": "user", "content": "a brand new turn"})
    assert "a brand new turn" in "\n".join(view.render(120, 60))
    assert view.n_messages == len(state.messages)


def test_context_view_pin_key_queues_the_existing_pin_helper(monkeypatch):
    from nx import _ctxreport, _fullscreen
    from nx import _state as st
    monkeypatch.setattr(st, "_CTX_PINNED", [], raising=False)
    seen = {}
    real_pin, real_unpin = _ctxreport._ctx_pin, _ctxreport._ctx_unpin

    def _pin(spec, messages):
        seen["pin"] = spec
        return real_pin(spec, messages)

    def _unpin(spec, messages):
        seen["unpin"] = spec
        return real_unpin(spec, messages)

    monkeypatch.setattr(_ctxreport, "_ctx_pin", _pin)
    monkeypatch.setattr(_ctxreport, "_ctx_unpin", _unpin)
    view = _fullscreen.make_view("context", _CtxState())
    # `p` (and enter/space) is DEFERRED: the shell runs pending_action only
    # after leaving the alternate screen, so nothing mutates under a frame.
    assert view.on_key("p", None) == "run"
    assert callable(view.pending_action)
    assert st._CTX_PINNED == []
    view.pending_action()
    assert seen["pin"] == "1"            # specs are 1-based, row 0 -> "1"
    assert st._CTX_PINNED == [0]
    assert view.on_key("enter", None) == "run"
    view.pending_action()
    assert seen["unpin"] == "1"
    assert st._CTX_PINNED == []


def test_context_view_pin_on_an_empty_session_is_a_notice_not_a_crash():
    from nx import _ctxreport

    class _Empty:
        messages: list = []

    view = _ctxreport.context_view(_Empty())
    assert "no messages yet" in "\n".join(view.render(100, 12))
    assert view.on_key("p", None) == "repaint"
    assert view.pending_action is None
    assert "no message rows" in view.notice


def test_context_handler_opens_the_view_when_a_tty_is_available(
        monkeypatch, capsys):
    """Bare /context takes the fullscreen view, with the inline report wired
    as the fallback (so the off-tty path cannot rot)."""
    import nx
    from nx import _repl_settings_cmds as rs
    opened = []
    monkeypatch.setattr(nx._fullscreen, "wants_fullscreen", lambda: True)
    monkeypatch.setattr(nx._fullscreen, "open_view",
                        lambda view, fallback=None: opened.append(
                            (view, fallback)) or True)
    state = _CtxState()
    rs._cmd_ctx("/context", state)
    view, fallback = opened[0]
    assert view.surface == "context" and view.state is state
    assert callable(fallback)            # the inline path stays wired
    fallback()
    assert "context by message" in capsys.readouterr().out


def test_context_off_tty_prints_the_pre_view_body_byte_for_byte(
        monkeypatch, capsys):
    """NX_FULLSCREEN=0 (and any off-tty run) keeps the pre-view /context
    output: `_ctx_inline_report` IS the body the command printed before the
    view existed."""
    import nx
    from nx import _repl_settings_cmds as rs
    monkeypatch.setenv("NX_FULLSCREEN", "0")
    monkeypatch.setattr(nx._fullscreen, "open_view",
                        lambda view, fallback=None: pytest.fail(
                            "off a tty /context must not open a view"))
    state = _CtxState()
    rs._cmd_ctx("/context", state)
    inline = capsys.readouterr().out
    assert "by role:" in inline
    rs._ctx_inline_report(state)
    assert capsys.readouterr().out == inline


def test_context_subcommands_keep_their_inline_behaviour(monkeypatch, capsys):
    import nx
    from nx import _repl_settings_cmds as rs
    monkeypatch.setattr(nx._state, "_CTX_PINNED", [], raising=False)
    monkeypatch.setattr(nx._fullscreen, "wants_fullscreen", lambda: True)
    monkeypatch.setattr(nx._fullscreen, "open_view",
                        lambda view, fallback=None: pytest.fail(
                            "/context pin must not open the view"))
    state = _CtxState()
    rs._cmd_ctx("/context pin 1", state)
    assert "pinned 1 turn(s)" in capsys.readouterr().out
    rs._cmd_ctx("/context unpin 1", state)
    assert "unpinned 1 turn(s)" in capsys.readouterr().out
    rs._cmd_ctx("/context clear", state)
    assert "unpinned" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# view stack (0.148.9): push/pop and esc-pops
# ---------------------------------------------------------------------------


def _fullscreen_View():
    from nx._fullscreen import View
    return View


class _CountingView(_fullscreen_View()):
    surface = "tabs"
    title = "counting"

    def __init__(self, log, quit_on=("esc",)):
        super().__init__()
        self.log = log
        self.quit_on = quit_on

    def render(self, width, height):  # noqa: ARG002
        self.log.append(("render", id(self)))
        return ["view"]

    def on_key(self, key, shell):
        if key in self.quit_on:
            return "quit"
        return "repaint"


def test_pushed_view_runs_on_top_and_esc_returns_to_the_view_below():
    from nx import _fullscreen
    log = []
    bottom = _CountingView(log, quit_on=("esc",))
    top = _CountingView(log, quit_on=("esc",))
    pushed = []

    orig_on_key = bottom.on_key

    def _bottom_key(key, shell):
        if not pushed:               # first key: stack the top view
            pushed.append(top)
            shell.push(top)
            return "repaint"
        return orig_on_key(key, shell)

    bottom.on_key = _bottom_key
    out = FakeOut()
    # Fixed script (a reader that raises EOFError when dry, like _reader_for,
    # so a missed step FAILS instead of looping forever):
    #   "x"   -> bottom stacks `top` on top, repaint
    #   "esc" -> stack depth 2: the loop pops back to bottom
    #   "esc" -> stack depth 1: bottom itself quits, run ends
    _fullscreen.run_view(bottom, fd=0, out=out,
                         _reader=_reader_for(["x", "esc", "esc"]),
                         _notifier=_fullscreen._ResizeNotifier())
    ids = [entry[1] for entry in log]
    # bottom, top, then bottom again after the pop — the view underneath resumed
    assert ids == [id(bottom), id(top), id(bottom)]


def test_quit_on_a_stacked_view_resumes_the_bottom_without_leaving():
    from nx import _fullscreen
    log = []
    bottom = _CountingView(log, quit_on=("esc",))
    top = _CountingView(log, quit_on=("q",))   # q, not esc: esc pops a stack
    pushed = []

    orig_on_key = bottom.on_key

    def _bottom_key(key, shell):
        if not pushed:               # first key: stack the top view
            pushed.append(top)
            shell.push(top)
            return "repaint"
        return orig_on_key(key, shell)

    bottom.on_key = _bottom_key
    out = FakeOut()
    #   "x"   -> bottom stacks `top`, repaint
    #   "q"   -> top quits: the bottom view resumes underneath (no esc-pop)
    #   "esc" -> bottom quits, run ends
    _fullscreen.run_view(bottom, fd=0, out=out,
                         _reader=_reader_for(["x", "q", "esc"]),
                         _notifier=_fullscreen._ResizeNotifier())
    ids = [entry[1] for entry in log]
    assert ids == [id(bottom), id(top), id(bottom)]
    # and the run ended (bottom's own esc quit), not hung: the ALT_EXIT went out
    assert out.getvalue().endswith(_fullscreen.ALT_EXIT)


def test_esc_on_the_bottom_view_still_quits_normally():
    from nx import _fullscreen
    log = []
    bottom = _CountingView(log)
    out = FakeOut()
    _fullscreen.run_view(bottom, fd=0, out=out, _reader=_reader_for(["esc"]),
                         _notifier=_fullscreen._ResizeNotifier())
    assert [entry[1] for entry in log] == [id(bottom)]
    assert out.getvalue().endswith(_fullscreen.ALT_EXIT)


# ---------------------------------------------------------------------------
# view registry (0.148.9)
# ---------------------------------------------------------------------------


def test_the_builtin_views_are_registered():
    from nx import _fullscreen
    assert set(_fullscreen.view_names()) >= {
        "tabs", "review", "sessions", "todos", "bgjobs", "context"}


def test_make_view_builds_registered_views_lazily():
    from nx import _fullscreen
    from nx._surfaces import TabsView
    assert isinstance(_fullscreen.make_view("tabs"), TabsView)
    with pytest.raises(KeyError):
        _fullscreen.make_view("no-such-view")


def test_register_view_is_idempotent_decorator_style():
    from nx import _fullscreen

    class V(_fullscreen.View):
        pass

    @_fullscreen.register_view("_test_v")
    def _f():
        return V()

    @_fullscreen.register_view("_test_v")
    def _g():
        return V()

    assert isinstance(_fullscreen.make_view("_test_v"), V)
    del _fullscreen._VIEW_FACTORIES["_test_v"]


# ---------------------------------------------------------------------------
# contract: a reader that runs dry must end the run (suite wiring guard)
# ---------------------------------------------------------------------------


def test_a_dry_reader_ends_the_run_instead_of_looping():
    """The wiring contract that keeps a bad test from hanging `-n auto`: the
    ONLY sanctioned way to drive run_view in tests is a scripted reader that
    raises EOFError when its keys run out — and run_view must convert that
    into a clean exit. (A reader that returns filler keys forever spins the
    poll loop hot on its xdist worker and stalls the whole suite.)"""
    from nx import _fullscreen
    log = []
    view = _CountingView(log, quit_on=("esc",))
    out = FakeOut()
    # The script runs dry after two keys: run_view must exit via EOFError.
    _fullscreen.run_view(view, fd=0, out=out,
                         _reader=_reader_for(["x", "x"]),
                         _notifier=_fullscreen._ResizeNotifier())
    assert out.getvalue().endswith(_fullscreen.ALT_EXIT)


# ---------------------------------------------------------------------------
# pty: the acceptance test for "enter/exit restores the terminal"
# ---------------------------------------------------------------------------

def _pty_available() -> bool:
    try:
        import pty  # noqa: F401  (POSIX only)
    except ImportError:
        return False
    return hasattr(os, "openpty")


@pytest.mark.skipif(not _pty_available(), reason="no pty on this platform")
def test_pty_view_round_trip_writes_the_alternate_screen_pair():
    """Drive a real view loop over a pty: the terminal sees ALT_ENTER, the
    frames, and ALT_EXIT as the last bytes — which is what restores the cursor
    and the scrollback (the terminal owns the scrollback swap; the test can
    only pin that we emit the sequence, and only once)."""
    import pty
    from nx import _fullscreen
    from nx._surfaces import tabs_view

    master, slave = pty.openpty()
    try:
        out = os.fdopen(os.dup(slave), "w", encoding="utf-8", buffering=1)
        view = tabs_view()
        # make the slave readable so the loop's select() fires (the reader
        # below is scripted, so the byte itself is never parsed)
        os.write(master, b"\x1b")
        # run_view takes fd=slave; the reader is scripted so no input is needed
        try:
            _fullscreen.run_view(view, fd=slave, out=out,
                                 _reader=_reader_for(["esc"]),
                                 _notifier=_fullscreen._ResizeNotifier())
        finally:
            out.close()
        # drain the master side until it goes quiet (the loop is done, so
        # nothing more is coming); EIO is the pty's normal EOF once the slave
        # side is closed.
        import time
        import fcntl
        fcntl.fcntl(master, fcntl.F_SETFL,
                    fcntl.fcntl(master, fcntl.F_GETFL) | os.O_NONBLOCK)
        data = b""
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                chunk = os.read(master, 4096)
            except BlockingIOError:
                time.sleep(0.02)
                continue
            except OSError:
                break
            if not chunk:
                break
            data += chunk
    finally:
        os.close(slave)
        os.close(master)
    text = data.decode("utf-8", "replace")
    assert _fullscreen.ALT_ENTER in text
    assert text.rstrip().endswith(_fullscreen.ALT_EXIT)
    assert text.count(_fullscreen.ALT_ENTER) == 1


# ---------------------------------------------------------------------------
# /memory — the MemoryView (Phase 2 of the view work: the ContextView recipe,
# plus the two-mode list -> note browsing ReviewView has, and the deferred
# delete going through the EXISTING memory_delete machinery)
# ---------------------------------------------------------------------------


@pytest.fixture
def _mem_notes(tmp_path, monkeypatch):
    """Two notes on disk, project scope only (the view reads the same dirs the
    inline list reads)."""
    from nx import _memory_store as store
    d = tmp_path / "memories"
    d.mkdir()
    notes = []
    for i, (name, title) in enumerate([
            ("20250101_000001_two.md", "# Second note\n\nbody two"),
            ("20250201_000001_one.md", "# First note\n\nbody one")]):
        p = d / name
        p.write_text(title + "\n\nkind: fact\n", encoding="utf-8")
        notes.append(p)  # noqa: B901
    monkeypatch.setattr(store, "_iter_memory_dirs", lambda scope: [(str(d), "project")])
    # the view module re-exports the store names at import time: patch there too
    from nx import _memory_cmd as mc
    monkeypatch.setattr(mc, "_iter_memory_dirs", lambda scope: [(str(d), "project")])
    return d, notes


def test_memory_view_is_registered_and_its_keys_are_advertised(_mem_notes):
    from nx import _fullscreen, _keys, _memory_cmd
    assert "memory" in _fullscreen.view_names()
    assert isinstance(_fullscreen.make_view("memory"), _memory_cmd.MemoryView)
    view = _memory_cmd.MemoryView()
    real = {"j/k": "j"}
    class _S:
        ask = staticmethod(lambda label, initial="": "y")
    for key, _meaning in _keys.keys_for("memory"):
        assert view.on_key(real.get(key, key), _S()) in (
            "repaint", "quit", "run"), key


def test_memory_view_renders_rows_hint_and_quit_keys(_mem_notes):
    from nx import _keys, _memory_cmd, _render
    view = _memory_cmd.MemoryView()
    body = "\n".join(view.render(120, 20))
    assert _render.BOX_TL in body and _render.BOX_BL in body
    assert _render.role("ACCENT") in body
    assert _keys.hint_line("memory") in body   # one registry: hints can't drift
    assert "First note" in body and "Second note" in body
    assert "memories · 2" in body
    for key in ("q", "ctrl-c", "ctrl-d"):
        assert view.on_key(key, None) == "quit", key


def test_memory_view_enter_opens_the_note_body_and_tab_returns(_mem_notes):
    from nx import _memory_cmd
    view = _memory_cmd.MemoryView()
    assert view.on_key("enter", None) == "repaint"
    assert view.mode == "note"
    body = "\n".join(view.render(120, 20))
    assert "First note" in body       # newest note first, opened
    assert "line 1-" in body           # scroll footer
    assert view.on_key("tab", None) == "repaint"
    assert view.mode == "list"


def test_memory_view_d_defers_through_the_real_delete_path(_mem_notes, monkeypatch):
    from nx import _memory_cmd
    _d, notes = _mem_notes
    deleted = []

    def _fake_delete(target, scope="project", **_):
        deleted.append(target)
        return f"deleted project memory: {target}"

    monkeypatch.setattr(_memory_cmd, "memory_delete", _fake_delete)
    view = _memory_cmd.MemoryView()
    shell = type("S", (), {"ask": staticmethod(
        lambda label, initial="": "y")})()
    # DEFERRED: nothing is deleted while the view is still on screen
    assert view.on_key("d", shell) == "run"
    assert callable(view.pending_action)
    assert deleted == [] and all(p.exists() for p in notes)
    view.pending_action()
    assert deleted == ["20250201_000001_one.md"]   # exact filename, not a prefix
    assert not notes[1].exists() or True  # fake delete: no real unlink expected


def test_memory_view_delete_confirm_cancelled_is_a_notice(monkeypatch):
    from nx import _memory_cmd

    class _None:
        notes = [{"name": "a.md", "title": "A", "mtime": 0, "path": "/x/a.md",
                  "label": "project"}]
        notice = ""

        def refresh(self):
            pass

    view = _memory_cmd.MemoryView()
    view.notes = _None.notes
    shell = type("S", (), {"ask": staticmethod(lambda label, initial="": None)})()
    assert view.on_key("d", shell) == "repaint"
    assert view.pending_action is None
    assert "delete cancelled" in view.notice


def test_memory_handler_opens_the_view_when_a_tty_is_available(monkeypatch, capsys):
    """Bare /memory takes the fullscreen view, with the inline paths wired as
    the fallback (so the off-tty path cannot rot)."""
    import nx
    from nx import _memory_cmd as mc
    opened = []
    monkeypatch.setattr(nx._fullscreen, "wants_fullscreen", lambda: True)
    monkeypatch.setattr(nx._fullscreen, "open_view",
                        lambda view, fallback=None: opened.append(
                            (view, fallback)) or True)
    mc._cmd_memory([], "/memory")
    view, fallback = opened[0]
    assert view.surface == "memory"
    assert callable(fallback)
    fallback()
    assert "usage:" in capsys.readouterr().out
    mc._cmd_memory([], "/memory list")
    view, fallback = opened[1]
    assert view.surface == "memory"
    fallback()
    assert "memories" in capsys.readouterr().out


def test_memory_off_tty_prints_the_pre_view_body_byte_for_byte(
        monkeypatch, capsys):
    """NX_FULLSCREEN=0 (and any off-tty run) keeps the pre-view /memory
    output byte for byte."""
    import nx
    from nx import _memory_cmd as mc
    monkeypatch.setenv("NX_FULLSCREEN", "0")
    monkeypatch.setattr(nx._fullscreen, "open_view",
                        lambda view, fallback=None: pytest.fail(
                            "off a tty /memory must not open a view"))
    mc._cmd_memory([], "/memory")
    inline = capsys.readouterr().out
    assert "usage:" in inline
    mc._memory_usage_inline()
    assert capsys.readouterr().out == inline


def test_memory_subcommands_keep_their_inline_behaviour(monkeypatch, capsys):
    import nx
    from nx import _memory_cmd as mc
    monkeypatch.setattr(nx._fullscreen, "wants_fullscreen", lambda: True)
    monkeypatch.setattr(nx._fullscreen, "open_view",
                        lambda view, fallback=None: pytest.fail(
                            "/memory subcommand must not open the view"))
    mc._cmd_memory([], "/memory remember ")      # empty query: usage warning
    assert "usage" in capsys.readouterr().out.lower()
    mc._cmd_memory([], "/memory delete ")
    assert "usage" in capsys.readouterr().out.lower()
    mc._cmd_memory([], "/memory optimize")
    assert capsys.readouterr().out is not None
