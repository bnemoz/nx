"""Lets the user press Esc during model generation to stop the stream and drop
back to the prompt. A SINGLE daemon thread, started once and parked between
turns, watches stdin for bare Esc. The main loop checks _USER_INTERRUPTED
between chunks and closes the stream on interrupt. Tools are never killed;
run_bash and wait_background arm the watcher while polling, so Esc ends only their wait;
hard cancellation remains a separate follow-up.

Esc-then-type redirect: once a bare Esc fires, the watcher doesn't just park —
it keeps draining stdin (`_drain_redirect_text`) for a couple seconds so a user
who immediately starts typing "actually, do X instead" isn't cut off mid-word.
If they hit Enter before the drain window closes, `st._INTERRUPT_REDIRECT_TEXT`
+ `_SUBMITTED` are read by `model_turn()` (`_turn.py`), which injects the text
straight in as the next user turn and loops again (`TURN_REDIRECT`) instead of
returning to the REPL prompt. A partial (unterminated) line is instead handed
to the REPL as a prefill for its next `read_multiline()` call.

Why a single persistent thread instead of spawn-per-turn:

  The watcher reads stdin byte-by-byte and discards anything that isn't Esc
  (line 1175ish). That makes it a stdin *thief* while alive. An earlier
  implementation spawned a fresh thread per turn and stopped it with
  `watcher.join(timeout=0.5)`. A timed join is not a guaranteed kill: if the
  thread was mid-read when the timeout elapsed, it kept running — and the
  `finally` then `_INTERRUPT_EVENT.clear()`d the very flag the thread needs
  to notice it should stop. The leaked thread was then immortal (its exit
  signal had already been lowered) and sat in a tight `select`+`os.read`
  loop, eating 50-75% of keystrokes that the chatbox was supposed to get.
  Worse, termios is global to the tty (not per-thread), so leaked watchers
  fighting the chatbox over `VMIN` (watcher sets 0, chatbox sets 1) could
  end up in a *blocking* `os.read` from which they could never re-check the
  exit flag — so the next turn's `_INTERRUPT_EVENT.set()` couldn't reach
  them and the leaks stacked. Result: a session got progressively more
  laggy the longer it ran, and only Ctrl+C / restart (which tears down all
  threads) cleared it. A persistent singleton thread with a clean
  arm/disarm protocol has no spawn/join window to leak through.

Four events:
  _INTERRUPT_ARMED    - "watcher should actively read stdin right now"
                        main sets it for a generation or interruptible tool wait; clears it to park.
  _INTERRUPT_EXIT     - "watcher should die entirely" (set once at process exit)
  _USER_INTERRUPTED   - "the user actually pressed Esc" - only the watcher sets
                        this; main reads it after disarming. Separate from the
                        arm flag so disarming to park != "you pressed Esc".
  _INTERRUPT_RESTORED - "watcher has restored termios to cooked mode" - the
                        watcher sets this in its inner finally after tcsetattr.
                        main clears it before arming and waits on it after
                        disarming so the chatbox never captures the watcher's
                        raw-mode termios as its "old" state. Without this
                        handshake the disarm is asynchronous (the watcher may
                        still be in select(0.1) when the chatbox runs) and the
                        chatbox can inherit VMIN=0, breaking Enter
                        and leaving the terminal stuck in raw mode.
"""
from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
from contextlib import contextmanager
from typing import Any

from . import _state as st

_INTERRUPT_ARMED = threading.Event()
_INTERRUPT_EXIT = threading.Event()
_USER_INTERRUPTED = threading.Event()
_INTERRUPT_RESTORED = threading.Event()
_INTERRUPT_RESTORED.set()  # initially: nothing to restore (OK to proceed)
# Model-generation Esc supports an immediate typed redirect; tool-wait Esc does
# not, so its watcher scope can park promptly before the next model turn.
_INTERRUPT_CAPTURE_REDIRECT = threading.Event()
_INTERRUPT_CAPTURE_REDIRECT.set()
st._INTERRUPT_THREAD = None


def _interrupt_watcher() -> None:
    """Persistent daemon: park while idle, read stdin for bare Esc while armed.

    When armed (during model generation) stdin is put into raw mode while
    leaving ISIG enabled, so Ctrl+C still kills the process. A bare Esc (not
    the start of an arrow-key / bracketed-paste / etc. CSI sequence) sets
    _USER_INTERRUPTED. When disarmed (between turns, while the chatbox owns
    stdin) we restore termios and park on _INTERRUPT_ARMED — doing NOTHING with
    stdin, so the chatbox / approval prompt get every keystroke.

    Single-reader invariant: this is the only thread that ever calls os.read on
    stdin, and only while armed. Parked = hands off stdin entirely. That's what
    stops the keystroke-eating leaks the spawn-per-turn design suffered from.
    """
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return  # stdin has no fileno (pytest capture, redirect, etc.)
    if not os.isatty(fd):
        return
    try:
        old: list[Any] = termios.tcgetattr(fd)
    except Exception:
        return
    new: list[Any] = old[:]
    # Keep ISIG enabled: clearing it turns Ctrl+C into an ordinary \x03 byte,
    # which this Esc-only watcher would discard instead of letting the tty
    # generate SIGINT.
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[0] &= ~termios.ICRNL
    new[6][termios.VMIN] = 0
    new[6][termios.VTIME] = 0
    last_fire = 0.0
    while not _INTERRUPT_EXIT.is_set():
        # Park until a generation arms us. While parked we touch stdin NOT AT
        # ALL — the chatbox owns it. This is the whole point of the redesign.
        _INTERRUPT_ARMED.wait(0.25)
        if not _INTERRUPT_ARMED.is_set() or _INTERRUPT_EXIT.is_set():
            continue
        # We're armed: take stdin into raw mode for the duration of generation.
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
        except Exception:
            _INTERRUPT_RESTORED.set()  # never armed — nothing to restore
            return
        try:
            while _INTERRUPT_ARMED.is_set() and not _INTERRUPT_EXIT.is_set():
                r, _, _ = select.select([fd], [], [], 0.1)
                if not r:
                    continue
                try:
                    c = os.read(fd, 1)
                except OSError:
                    _INTERRUPT_RESTORED.set()  # dying — unblock main's wait
                    return
                if c != b"\x1b":
                    continue  # discard anything that isn't Esc
                # Could be bare Esc OR the lead byte of an escape sequence
                # (arrow keys, Home/End, bracketed paste, etc.). Wait up to 50ms
                # for more bytes; if none arrive, it's a bare Esc.
                r2, _, _ = select.select([fd], [], [], 0.05)
                if r2:
                    try:
                        os.read(fd, 1)  # swallow the rest of the sequence
                    except OSError:
                        pass
                    continue
                now = time.time()
                if now - last_fire < 0.25:  # debounce — don't fire twice in a row
                    continue
                last_fire = now
                _USER_INTERRUPTED.set()
                # Leave _INTERRUPT_ARMED set: main will clear it on disarm,
                # which exits the inner loop and parks us. In the meantime
                # (main is closing the stream / breaking out of its chunk
                # loop), keep reading stdin so a user who immediately starts
                # typing a redirect ("actually, do X instead") doesn't have
                # those keystrokes silently discarded — capture them below
                # instead of just discarding non-Esc bytes.
                if _INTERRUPT_CAPTURE_REDIRECT.is_set():
                    st._INTERRUPT_REDIRECT_TEXT = _drain_redirect_text(fd, new)
                    st._INTERRUPT_REDIRECT_SUBMITTED = st._INTERRUPT_REDIRECT_TEXT.endswith("\n")
                    st._INTERRUPT_REDIRECT_TEXT = st._INTERRUPT_REDIRECT_TEXT.rstrip("\n")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass
            # Signal main that termios has been restored to cooked mode.
            # main waits on this after clearing _INTERRUPT_ARMED so the chatbox
            # never captures our raw-mode termios as its "old" state.
            _INTERRUPT_RESTORED.set()


def _drain_redirect_text(fd: int, raw_termios: list[Any]) -> str:
    """After a bare Esc fires, keep reading stdin (still in raw mode) for a
    short grace window so a user typing a redirect right away isn't cut off
    mid-word. Returns the typed text, ending in "\n" if Enter was pressed
    before the window closed (a full line, ready to inject immediately) or
    without it if the window simply ran out (a partial line, used as a
    prefill for the next chatbox prompt instead).

    Deliberately dumb: no arrow-key/history support, backspace only. This is
    a stopgap capture buffer for the handoff window, not a second input
    widget — full editing happens in the chatbox once the redirect is either
    injected or handed over as a prefill.
    """
    buf: list[str] = []
    deadline = time.time() + 2.0  # generous — covers stream-close + tool teardown
    # Deliberately NOT gated on _INTERRUPT_ARMED: main clears that the instant
    # it notices the interrupt (next chunk / stream-close), which is almost
    # immediately — long before this deadline. Gating on it here used to
    # collapse the 2s typing window down to near-zero, silently dropping
    # whatever the user typed. Only the deadline or process exit should stop us.
    while not _INTERRUPT_EXIT.is_set() and time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.05)
        if not r:
            continue
        try:
            c = os.read(fd, 1)
        except OSError:
            break
        if c in (b"\r", b"\n"):
            buf.append("\n")
            break
        if c in (b"\x7f", b"\x08"):  # backspace
            if buf:
                buf.pop()
            continue
        if c == b"\x1b":  # a second Esc while draining: abandon the redirect
            buf = []
            break
        try:
            ch = c.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if ch >= " ":
            buf.append(ch)
    return "".join(buf)


def _ensure_interrupt_watcher() -> None:
    """Start the singleton interrupt watcher thread if it isn't already alive.

    Called lazily before the first model turn (and defensively on every turn —
    cheap because a living thread short-circuits the check). Guaranteed exactly
    one reader of stdin across the whole process lifetime.
    """
    if st._INTERRUPT_THREAD is not None and st._INTERRUPT_THREAD.is_alive():
        return
    _INTERRUPT_EXIT.clear()
    _INTERRUPT_ARMED.clear()
    _INTERRUPT_RESTORED.set()  # fresh thread: nothing to restore yet
    _USER_INTERRUPTED.clear()
    st._INTERRUPT_REDIRECT_TEXT = ""
    st._INTERRUPT_REDIRECT_SUBMITTED = False
    # Go through the top-level package (not the bare name in this module's
    # own globals) so a test that monkeypatches `nx._interrupt_watcher`
    # (e.g. to make the watcher body a no-op) actually takes effect — a bare
    # reference here would always resolve to this module's own function,
    # never picking up the patched one.
    import nx as _m
    st._INTERRUPT_THREAD = threading.Thread(
        target=_m._interrupt_watcher, daemon=True, name="nx-interrupt")
    st._INTERRUPT_THREAD.start()


@contextmanager
def _watch_for_interrupts():
    """Arm the singleton watcher for an interruptible operation.

    Model streaming and PID polling both need exclusive ownership of stdin
    while Esc is meaningful. Keeping the arm/disarm and termios-restore
    handshake here makes those windows identical, and guarantees the watcher
    is parked again before the chatbox or another operation can read input.
    """
    _ensure_interrupt_watcher()
    capture_redirect = _INTERRUPT_CAPTURE_REDIRECT.is_set()
    _INTERRUPT_CAPTURE_REDIRECT.clear()
    _USER_INTERRUPTED.clear()
    _INTERRUPT_RESTORED.clear()
    _INTERRUPT_ARMED.set()
    try:
        yield
    finally:
        _INTERRUPT_ARMED.clear()
        # _drain_redirect_text's own deadline is 2.0s; wait past it so we never
        # proceed while the watcher still holds raw termios mid-drain.
        if not _INTERRUPT_RESTORED.wait(timeout=2.5):
            print(
                "nx: warning: interrupt watcher did not confirm termios "
                "restore within 2.5s; terminal state may be stale",
                file=sys.stderr,
            )
        # Do not let an Esc from this operation interrupt a later model turn.
        _USER_INTERRUPTED.clear()
        if capture_redirect:
            _INTERRUPT_CAPTURE_REDIRECT.set()
