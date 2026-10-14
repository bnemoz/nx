"""Full-screen views on the alternate screen (UIUX_SPEC Phase D, 0.137.0).

ONE owner of the alternate screen — `\\033[?1049h` / `\\033[?1049l` — plus the
view dispatch, the raw-key loop, the SIGWINCH repaint, and the "stop the world"
contract the spec spells out:

  * a view is only ever entered while the TURN LOOP IS IDLE (the prompt owns
    stdin; `wants_fullscreen()` refuses mid-turn), so nothing else is draining
    the terminal;
  * the mid-turn live dashboard (`_surfaces` live region) is CLOSED on entry
    and REOPENED on exit when it had been open — paused, not lost;
  * the alternate screen itself guarantees the acceptance criterion: entering
    and leaving it restores the cursor position and the scrollback exactly as
    they were, because a view's frames are drawn in the alternate buffer and
    never touch the normal one.

Views are plain `View` objects implemented in the module that owns their data
(registered by name in `_VIEW_FACTORIES` here so the surface set is enumerable
in one place) and they render EXCLUSIVELY through `_render.py` components: no raw
palette reads, one glyph set, semantic roles (the Phase B grep-guard test
covers every module in the package). Key lists come from `_keys.py` — the same
registry behind the chatbox border hints and the `?` overlay — so a fullscreen
view cannot advertise a key it does not handle, and `?` inside a view paints
the same overlay the chatbox paints.

State is read-only from a view: the only mutations a view can cause are
*existing* commands (`/reset`-free resume, the git_commit confirmation flow),
which a view hands back to the shell as a deferred action — the shell leaves the
alternate screen FIRST and then runs the action, so an approval prompt is never
painted into the alternate buffer.

Off a tty, and when `NX_FULLSCREEN=0`, `wants_fullscreen()` is False and every
command keeps its pre-Phase-D inline output byte for byte (pinned by tests
against the pre-existing command functions).
"""
from __future__ import annotations

import contextlib
import os
import signal
import sys
from collections.abc import Iterator
from typing import Any, Callable, Optional

# --- the contract with the terminal ------------------------------------------
# Enter/exit are the ONLY cursor-and-scrollback-relevant bytes this module
# emits outside a frame: the alternate screen saves the cursor and swaps the
# scrollback, so pair-for-pair they restore the pre-entry state exactly.
ALT_ENTER = "\033[?1049h"
ALT_EXIT = "\033[?1049l"
# The first frame of a run (and every frame after a resize, a prompt, or the
# overlay) is drawn with home + erase below, so no cursor arithmetic has to be
# kept in sync with it; later frames are line-diffed against the previous one.
FRAME_HOME = "\033[H"
FRAME_ERASE = "\033[2J"
# Erase from the cursor to the end of the screen: how a repainted frame sheds
# the rows it gave up, and (with FRAME_HOME) the full-repaint clear.
FRAME_ERASE_BELOW = "\033[J"

# How long to wait after a lone ESC for the rest of a CSI/SS3 sequence before
# calling it a bare Esc (the same disambiguation the chatbox uses).
ESC_WAIT = 0.05
# Key-poll slice: also the resize-repaint latency ceiling on platforms where a
# SIGWINCH wake pipe is unavailable (Windows).
POLL = 0.25

OVERLAY_KEY = "?"

# --- key decoding -------------------------------------------------------------
_SIMPLE = {
    "\r": "enter", "\n": "enter", "\t": "tab",
    "\x03": "ctrl-c", "\x04": "ctrl-d", "\x7f": "backspace", "\x08": "backspace",
    " ": "space",
}
_ESC_SEQ = {
    "[A": "up", "[B": "down", "[C": "right", "[D": "left",
    "[H": "home", "[F": "end", "[Z": "shift+tab",
    "[5~": "pageup", "[6~": "pagedown",
    "OA": "up", "OB": "down", "OC": "right", "OD": "left",
}
_FINALS = "ABCDHFZ~"


def read_key(fd: int, *, wait: Optional[Callable[[int, float], bool]] = None) -> str:
    """One keypress as a symbolic name ("esc", "enter", "up", "ctrl-c") or the
    literal character for anything printable.

    Built on the SAME raw reader the chatbox uses (`_chatbox._raw_read_key`),
    so there is one byte source per process: a bare Esc is told apart from an
    arrow/Ctrl-arrow sequence by a short `wait_ready` probe. Raises EOFError
    when stdin closes (the caller's exit signal).
    """
    from ._chatbox import _raw_read_key
    if wait is None:
        from ._tty import wait_ready as wait
    ch = _raw_read_key(fd)
    if ch != "\x1b":
        return _SIMPLE.get(ch, ch)
    if not wait(fd, ESC_WAIT):
        return "esc"
    tail = _raw_read_key(fd)
    if tail == "O":
        if wait(fd, ESC_WAIT):
            tail += _raw_read_key(fd)
    elif tail == "[":
        while len(tail) < 3 and wait(fd, ESC_WAIT):
            tail += _raw_read_key(fd)
            if tail[-1] in _FINALS:
                break
    return _ESC_SEQ.get(tail, "esc")


def read_line(fd: int, out, label: str, *, width: int = 0, initial: str = "") -> Optional[str]:
    """A one-line raw-mode editor drawn on `out`'s bottom row: printable keys,
    backspace, Enter commits, Esc/Ctrl-C cancels (returns None).

    Deliberately minimal, exactly like the pre-Phase-D /tabs steer prompt: a
    one-line nudge (a steer note, a search query, a commit subject) is not a
    chat box, and reusing the chatbox editor inside the alternate screen would
    duplicate its frame arithmetic for no gain.
    """
    from ._tty import wait_ready
    buf = initial
    while True:
        out.write(f"\r\x1b[K{label} {buf}")
        out.flush()
        if not wait_ready(fd, POLL):
            continue
        try:
            key = read_key(fd)
        except EOFError:
            return None
        if key == "enter":
            return buf.strip()
        if key in ("esc", "ctrl-c", "ctrl-d"):
            return None
        if key == "backspace":
            buf = buf[:-1]
            continue
        if len(key) == 1 and key >= " ":
            buf += key


# --- resize (SIGWINCH) --------------------------------------------------------
class _ResizeNotifier:
    """SIGWINCH -> a readable fd, so the key loop's select() wakes immediately
    instead of waiting out its poll slice.

    POSIX only (Windows has no SIGWINCH; there the loop's POLL timeout is the
    repaint trigger). `trigger()` is the handler body, exposed so tests can
    simulate a resize without a real signal.
    """

    def __init__(self) -> None:
        self._fds: tuple[int, int] = (-1, -1)
        self._prev = None
        self._drain = lambda: None

    def open(self) -> None:
        if os.name != "posix":
            return
        try:
            self._fds = os.pipe()
            os.set_blocking(self._fds[0], False)
            os.set_blocking(self._fds[1], False)

            def _drain() -> None:
                try:
                    while os.read(self._fds[0], 64):
                        pass
                except (BlockingIOError, OSError):
                    pass
            self._drain = _drain
            self._prev = signal.signal(signal.SIGWINCH, self._on_signal)
        except (AttributeError, OSError, ValueError, RuntimeError):
            self._fds = (-1, -1)
            self._prev = None

    def _on_signal(self, signum, frame) -> None:  # pragma: no cover - signal path
        self.trigger()

    def trigger(self) -> None:
        """Post a resize wake-up (the SIGWINCH handler's whole body)."""
        if self._fds[1] >= 0:
            try:
                os.write(self._fds[1], b"w")
            except OSError:
                pass

    def fds(self) -> list[int]:
        return [f for f in self._fds if f >= 0]

    def drain(self) -> None:
        self._drain()

    def close(self) -> None:
        if self._prev is not None:
            try:
                signal.signal(signal.SIGWINCH, self._prev)
            except (OSError, ValueError):
                pass
            self._prev = None
        for f in self._fds:
            if f >= 0:
                try:
                    os.close(f)
                except OSError:
                    pass
        self._fds = (-1, -1)
        self._drain = lambda: None


# --- the alternate screen + stop-the-world -----------------------------------
_SCREEN_OWNER = False


def owns_screen() -> bool:
    """Is a fullscreen view painting right now? The live region asks this
    before repainting (a 4 fps dashboard frame must never land inside a view's
    alternate screen)."""
    return _SCREEN_OWNER


def _term_lines(rows: tuple[int, int]) -> tuple[int, int]:
    import shutil
    from ._layout import TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS, TERM_MIN_COLS
    size = shutil.get_terminal_size((TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS))
    return max(TERM_MIN_COLS, size.columns), max(3, size.lines)


@contextlib.contextmanager
def alternate_screen(*, out=None, pause: bool = True) -> Iterator[None]:
    """Own the alternate screen for the duration of the block.

    Enter emits `ALT_ENTER` and exit emits `ALT_EXIT` (plus an SGR reset so a
    colored last frame cannot bleed into the restored prompt) — that pair is
    what restores the cursor and the scrollback, and the exit line is emitted
    from a `finally`, so an exception inside a view (or Ctrl-C) still gives the
    terminal back.

    `pause=True` implements the stop-the-world contract: the mid-turn live
    dashboard is closed on entry and reopened on exit if it *was* open.
    """
    global _SCREEN_OWNER
    from ._render import RESET
    out = out if out is not None else sys.stdout
    was_open = False
    if pause:
        was_open = _pause_dashboard()
    out.write(ALT_ENTER)
    out.flush()
    _SCREEN_OWNER = True
    try:
        yield
    finally:
        _SCREEN_OWNER = False
        try:
            out.write(RESET + ALT_EXIT)
            out.flush()
        except Exception:
            pass
        if pause and was_open:
            _resume_dashboard()


def _pause_dashboard() -> bool:
    """Close the live region for the duration of a view; True if it was open."""
    try:
        from ._surfaces import close_dashboard, dashboard_open
        if not dashboard_open():
            return False
        close_dashboard()
        return True
    except Exception:
        return False


def _resume_dashboard() -> None:
    """Reopen the live region a view paused (only if something is still live —
    the region's own termination rule, `maybe_close_if_idle`)."""
    try:
        from ._surfaces import close_dashboard, open_dashboard
        if not open_dashboard():
            return
        from ._surfaces import _surfaces_snapshot
        if not any(r.get("alive") for r in _surfaces_snapshot()
                   if r.get("kind") != "main"):
            close_dashboard()
    except Exception:
        pass


# --- view protocol ------------------------------------------------------------
class View:
    """One alternate-screen view.

    A view gathers its own data and renders lines; it never writes to the
    terminal (the shell does), never mutates state, and never handles keys the
    shell reserves (`?` overlay, Esc/quitter). A view that needs a mutation
    returns "run" from `on_key` after setting `pending_action` to a zero-arg
    callable; the shell leaves the alternate screen and then runs it.
    """

    #: `_keys.py` surface name — the key list both the hint line and the `?`
    #: overlay inside this view are rendered from.
    surface = "fullscreen"
    #: The view's own box title.
    title = "view"

    def __init__(self) -> None:
        self.pending_action: Optional[Callable[[], None]] = None
        #: transient one-line feedback drawn under the view
        self.notice = ""

    def render(self, width: int, height: int) -> list[str]:
        """The frame as ready-to-write lines (already colored, clipped to
        `width`). MUST go through `_render.py` components."""
        raise NotImplementedError

    def on_key(self, key: str, shell: "Shell") -> str:
        """Handle one key: return "repaint", "quit", or "run" (then the shell
        runs `pending_action` after leaving the alternate screen)."""
        return "repaint"

    def take_action(self) -> Optional[Callable[[], None]]:
        act, self.pending_action = self.pending_action, None
        return act


class Shell:
    """The bridge a view uses for the two things the shell owns: drawing, and a
    one-line prompt on the bottom row (`ask`) — the view's data stays its own.

    It also carries the view stack: `shell.push(view)` stacks a new view (the
    caller's `on_key` then returns "repaint" to hand control over), and
    `shell.pop()` returns to the view underneath without consuming a key.
    """

    def __init__(self, fd: int, out, width: int, height: int,
                 _stack: Optional[list[View]] = None) -> None:
        self.fd = fd
        self.out = out
        self.width = width
        self.height = height
        self._stack = _stack if _stack is not None else []
        #: `ask` sets this: the prompt drew onto the frame outside `paint`'s
        #: knowledge, so the shell's row cache no longer describes the screen
        #: and the next frame must be a full repaint.
        self.screen_dirty = False

    def ask(self, label: str, initial: str = "") -> Optional[str]:
        self.screen_dirty = True
        return read_line(self.fd, self.out, label, initial=initial)

    def push(self, view: "View") -> None:
        """Stack `view` on top of the running one (Esc or its `quit` pops)."""
        self._stack.append(view)

    def pop(self) -> None:
        """Return to the view underneath (no-op on the bottom view)."""
        if len(self._stack) > 1:
            self._stack.pop()


def run_view(view: View, *, fd: Optional[int] = None, out=None,
             _reader: Optional[Callable[[int], str]] = None,
             _notifier: Optional[_ResizeNotifier] = None,
             _stack: Optional[list[View]] = None) -> Optional[Callable[[], None]]:
    """Run one view until it quits; returns the deferred action (or None).

    The loop is the chatbox's discipline, not cursor arithmetic: the first
    frame (and any frame after a resize, a prompt, or the `?` overlay) is
    `FRAME_HOME + FRAME_ERASE + lines`; later frames are line-diffed against
    the previous one — identical frames write nothing, changed rows are
    rewritten in place. That kills the flicker the wholesale erase caused
    below content-height frames (the views repaint on every 250 ms poll tick
    and mostly re-render the same bytes). Resizes repaint immediately
    (SIGWINCH wake pipe on POSIX, the poll slice elsewhere). `?` paints the
    `_keys` overlay for one frame.

    `_stack` (normally not passed; `push_view`/`pop_view` manage it) makes the
    loop a view STACK: a view's `on_key` may call `push_view(shell, next)` and
    then return "repaint" to hand control to `next` — the previous view stays
    alive underneath and resumes when the top view returns "quit". `pop_view`
    (or a plain Esc, when the stack is deeper than one) returns to the view
    below; quitting the BOTTOM view still ends the whole run and returns its
    deferred action.
    """
    from ._tty import raw_mode, wait_ready, wait_ready_any
    out = out if out is not None else sys.stdout
    fd = fd if fd is not None else sys.stdin.fileno()
    reader = _reader or read_key
    if _stack is None:
        _stack = [view]
    elif not _stack:
        _stack.append(view)

    notifier = _notifier
    own_notifier = notifier is None
    if own_notifier:
        notifier = _ResizeNotifier()
        notifier.open()

    # The last painted frame, for the diff paint below.
    prev: list[str] = []

    def paint(lines: list[str]) -> None:
        # Enforce the render contract ("clipped to width") centrally: a view
        # line wider than the terminal autowraps into an extra physical row,
        # desyncing the frame. clip_display is ANSI-aware, so SGR survives.
        nonlocal prev
        from ._ansi import CLEAR_LINE, clip_display
        rows = [clip_display(ln, width) for ln in lines][:max(1, height)]
        changed = [i for i, r in enumerate(rows)
                   if i >= len(prev) or prev[i] != r]
        if not changed and len(rows) >= len(prev):
            # The poll tick repaints 4x/s and views re-snapshot live state, so
            # most frames are byte-identical: an erase+redraw of those is pure
            # terminal traffic + flicker (the _surfaces live region suppresses
            # the same thing; the alternate screen is no cure — a full-screen
            # erase flashes the blank rows below a content-height frame).
            return
        if not prev:
            # first frame, resize, or a prompt drew over the rows: full clear
            out.write(FRAME_HOME + FRAME_ERASE + "\n".join(rows))
        else:
            # Line-diff paint: rewrite only the rows that changed, each at its
            # absolute position (CUP cannot scroll, unlike a relative move
            # from the bottom row), cleared first so a shorter row leaves no
            # tail. Rows the frame gave up are erased below the last one.
            buf = [f"\033[{i + 1}H{CLEAR_LINE}{rows[i]}" for i in changed]
            if len(rows) < len(prev):
                buf.append(f"\033[{len(rows) + 1}H{FRAME_ERASE_BELOW}")
            if rows:
                # Park the cursor at the end of the frame's last row (what a
                # full paint leaves): `ask` draws its prompt at the cursor, so
                # it must not land mid-frame on whichever row changed last.
                buf.append(f"\033[{len(rows)}H{CLEAR_LINE}{rows[-1]}")
            out.write("".join(buf))
        prev = rows
        out.flush()

    overlay = False
    action: Optional[Callable[[], None]] = None
    try:
        with alternate_screen(out=out):
            with (raw_mode(fd, icrnl=True) if _is_tty(fd)
                  else contextlib.nullcontext()):
                width, height = _term_lines(_term_size())
                shell = Shell(fd, out, width, height, _stack)
                while True:
                    if not _stack:                 # every view popped: done
                        break
                    view = _stack[-1]
                    paint(_overlay_lines(view.surface) if overlay
                          else view.render(width, height))
                    if overlay:
                        # the overlay is persistent: a poll timeout repaints it
                        # (loop top), only a key dismisses it
                        if not wait_ready(fd, POLL):
                            continue
                        try:
                            reader(fd)
                        except EOFError:
                            break
                        overlay = False
                        prev = []   # the overlay drew over the cached rows
                        continue
                    extra = notifier.fds()
                    ready = (wait_ready_any([fd, *extra], POLL) if extra
                             else (fd if wait_ready(fd, POLL) else None))
                    if extra and ready in extra:
                        notifier.drain()
                        width, height = _term_lines(_term_size())
                        prev = []          # rows were clipped for the old size
                        continue
                    if ready is None:
                        # poll timeout: repaint (clock/live rows). Re-probe the
                        # size here too — on a platform with no SIGWINCH wake
                        # pipe (Windows) this is the ONLY resize trigger, and a
                        # stale `prev` after a real resize would let the
                        # line-diff skip the repaint the new size needs (the
                        # notifier branch above is the POSIX fast path).
                        size = _term_lines(_term_size())
                        if size != (width, height):
                            width, height = size
                            prev = []
                        continue
                    try:
                        key = reader(fd)
                    except EOFError:
                        break
                    if key == OVERLAY_KEY:
                        overlay = True
                        prev = []   # the overlay starts on a cleared screen
                        continue
                    if key == "esc" and len(_stack) > 1:
                        _stack.pop()               # a stacked view: pop to below
                        continue
                    verdict = view.on_key(key, shell)
                    if shell.screen_dirty:   # a prompt drew over the frame
                        shell.screen_dirty = False
                        prev = []            # next paint must clear the screen
                    if verdict == "quit":
                        _stack.pop()
                        if _stack:
                            continue               # resume the view underneath
                        break                      # bottom view quit: leave
                    if verdict == "run":
                        action = view.take_action()
                        break
    finally:
        if own_notifier and notifier is not None:
            notifier.close()
    return action


def _is_tty(fd: int) -> bool:
    try:
        return os.isatty(fd)
    except OSError:
        return False


def _term_size() -> tuple[int, int]:
    import shutil
    from ._layout import TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS
    size = shutil.get_terminal_size((TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS))
    return size.columns, size.lines


def _overlay_lines(surface: str) -> list[str]:
    """The `?` overlay for a fullscreen view: the SAME `_keys.overlay_lines`
    the chatbox renders (one registry, so hints and overlay cannot drift)."""
    from ._keys import overlay_lines
    return overlay_lines(surface, title=(
        f"keys · {surface} (any key closes)"))


# --- view registry ------------------------------------------------------------
# Views live in the module that owns their data, but their *identity* is
# registered here (the same tiered-registry idiom as `_tool_gates._TOOL_GATES`):
# one place enumerates the fullscreen surfaces, so dispatch, docs and tests can
# walk them without importing each owner module eagerly (factories import
# lazily, which keeps the `_surfaces` <-> `_fullscreen` cycle broken).
_VIEW_FACTORIES: dict[str, Callable[..., View]] = {}


def register_view(name: Optional[str] = None,
                  factory: Optional[Callable[..., View]] = None) -> Callable[..., Any]:
    """Register a zero-or-more-arg view factory under `name` (e.g. "tabs").

    Usable as a decorator on a factory function, or called directly. Returns
    the factory unchanged.
    """
    if factory is None:                       # bare decorator form
        # A decorator, not a finished factory: it returns register_view's own
        # result (the factory), so the shape is Callable -> factory.
        return lambda f: register_view(name, f)
    _VIEW_FACTORIES[name] = factory           # type: ignore[index]
    return factory


def view_names() -> list[str]:
    """Registered fullscreen surface names, in registration order."""
    return list(_VIEW_FACTORIES)


def make_view(name: str, *args, **kwargs) -> View:
    """Build the view registered under `name`, passing through arguments.

    Raises KeyError for an unknown name — callers dispatch on names they own.
    """
    try:
        factory = _VIEW_FACTORIES[name]
    except KeyError:
        raise KeyError(f"no fullscreen view registered as {name!r} "
                       f"(known: {', '.join(_VIEW_FACTORIES) or 'none'})") from None
    return factory(*args, **kwargs)


# --- dispatch -----------------------------------------------------------------
def wants_fullscreen() -> bool:
    """Should a view take the alternate screen right now?

    False off a tty, with `NX_FULLSCREEN=0` (or false/no/off), or while a turn
    is running — the views are prompt-time surfaces, and refusing mid-turn is
    the "turn loop idle" half of the stop-the-world contract. Callers fall
    through to their inline (pre-Phase-D) output when this is False.
    """
    if os.environ.get("NX_FULLSCREEN", "").strip().lower() in (
            "0", "false", "no", "off"):
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError, OSError):
        return False
    from . import _state as st
    return not st._TURN_ACTIVE


@register_view("tabs")
def tabs():
    from ._surfaces import TabsView
    return TabsView()


@register_view("review")
def review(state=None):
    from ._review import ReviewView
    return ReviewView(state)


@register_view("context")
def context(state=None):
    from ._ctxreport import ContextView
    return ContextView(state)


@register_view("memory")
def memory_view():
    from ._memory_cmd import MemoryView
    return MemoryView()


@register_view("undo")
def undo_view():
    from ._undo import UndoView
    return UndoView()


@register_view("sessions")
def sessions(state):
    from ._sessions_cmd import SessionsView
    return SessionsView(state)


@register_view("todos")
def todos():
    from ._todo import TodoView
    return TodoView()


@register_view("bgjobs")
def bgjobs():
    """The /bg view (background run_bash jobs). Lives in `_bg_completions`:
    the view module must not be `_surfaces`, which imports `_fullscreen` at
    module level — `_fullscreen` importing it back would close the cycle."""
    from ._bg_completions import BackgroundJobsView
    return BackgroundJobsView()


def _drain_pending_input(fd: int) -> None:
    """Discard any bytes already queued on `fd`.

    The key loop leaves raw mode before the deferred action runs, and an
    action can take real time (a session delete GCs image sidecars). Bytes
    typed during that window would otherwise be echoed by the kernel in
    cooked mode and leak into the prompt as literal `^[[A` garbage.
    """
    from ._tty import wait_ready
    try:
        while wait_ready(fd, 0):
            if not os.read(fd, 4096):
                break                      # EOF: readable forever, never drains
    except OSError:
        pass


def open_view(view: View, *, fallback: Optional[Callable[[], None]] = None) -> bool:
    """Run `view` in the alternate screen when `wants_fullscreen()`, else call
    `fallback` (the command's pre-Phase-D inline output) and return False.

    Returns True when the view actually ran. A failure to reach a usable tty
    fd falls back too, so a /tabs in a weird terminal still prints something.
    """
    if not wants_fullscreen():
        if fallback is not None:
            fallback()
        return False
    try:
        action = run_view(view)
    except (OSError, ValueError):
        if fallback is not None:
            fallback()
        return False
    if action is not None:
        try:
            _drain_pending_input(sys.stdin.fileno())
        except (OSError, ValueError):    # no usable stdin (tests, pipes)
            pass
        action()
    return True


__all__ = [
    "ALT_ENTER", "ALT_EXIT", "OVERLAY_KEY",
    "FRAME_HOME", "FRAME_ERASE", "FRAME_ERASE_BELOW",
    "View", "Shell", "read_key", "read_line",
    "alternate_screen", "run_view", "open_view",
    "wants_fullscreen", "owns_screen", "tabs", "review", "sessions", "todos",
    "bgjobs", "register_view", "view_names", "make_view",
]
