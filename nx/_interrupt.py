"""Lets the user press Esc during model generation to stop the stream and drop
back to the prompt. A SINGLE daemon thread, started once and parked between
turns, watches stdin for bare Esc. The main loop checks _USER_INTERRUPTED
between chunks and closes the stream on interrupt. Esc ends only the current
wait; the ESCALATION LADDER (UIUX_SPEC C1, 0.136.0) adds hard cancellation:

  Esc (1x)        interrupt the current wait (existing: stream chunk loop,
                  _poll_pid -> "interrupted", command keeps running).
  Esc (2x within  NX_ESC_LADDER_SECONDS, default 0.8s) -> _HARD_CANCEL fires:
      Esc)        the running tool's registered cancel hook runs (run_bash/
                  wait_background: SIGTERM the process group, SIGKILL after
                  2s; job marked killed in /bg), and every tool-dispatch
                  boundary in _turn_toolcalls skips the rest of the group
                  with a CANCELLED result (the model is always told).
  Ctrl+C          same hard cancel, one motion: the KeyboardInterrupt
                  landing in a tool call converts to request_hard_cancel().

Threads cannot be killed, so sub-agent (and other cooperative) tools get a
CHECK-AT-BOUNDARY contract instead: poll loops check cancel_requested()
(~every 100ms) and tool dispatch checks it before every call — that is the
honest, bounded form of "cooperative cancellation". register_tool_cancel()
holds AT MOST one live hook (the current tool clears it in its finally), so
the ladder can never aim at a stale pid. Everything is event-based: no new
threads live between cancellations, and the watcher stays the persistent
singleton described below (the thread-leak history further down is the
regression this design must not revive).

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
  _INTERRUPT_RESTORED - "watcher has restored input mode to cooked mode" - the
                        watcher sets this in its inner finally after restoring.
                        main clears it before arming and waits on it after
                        disarming so the chatbox never captures the watcher's
                        raw-mode input mode as its "old" state. Without this
                        handshake the disarm is asynchronous (the watcher may
                        still be mid-poll when the chatbox runs) and the
                        chatbox can inherit VMIN=0, breaking Enter
                        and leaving the terminal stuck in raw mode.

plus a POSIX wake pipe (PF-06, 0.56.0): the armed window blocks on stdin AND the
pipe, and main pokes it on disarm so the watcher restores cooked mode in
single-digit ms instead of after its next 0.1s poll.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any

from . import _state as st
from ._ansi import DIM, RESET
from ._env import _env_float
from ._subagent_tls import _subagent_ctx
from ._tty import (
    IS_WINDOWS, apply_interrupt_mode, capture_input_mode, read_byte,
    restore_input_mode, wait_ready, wait_ready_any,
)

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

# ------------------------------------------------- hard-cancel ladder (C1) ---
# Set by a second Esc inside the ladder window, by Ctrl+C landing in a tool
# call, or by request_hard_cancel() directly. Checked by tool-wait poll loops
# (_poll_pid -> "cancelled") and by the tool-dispatch loop in
# _turn_toolcalls (skip the rest of the group). Cleared at the start of every
# interactive model turn (model_turn) — never by _watch_for_interrupts, which
# arms/disarms between tools of the SAME group and must let the flag survive
# from one tool of the group to the next.
_HARD_CANCEL = threading.Event()
# At most one tool is running on the owner thread at a time, so a single
# slot + lock is the whole registry. The hook must be non-blocking (spawn a
# bounded daemon thread if it needs a grace period — see _bash_jobs).
_CANCEL_FN = None
_CANCEL_LOCK = threading.Lock()
# Wall-clock of the last watcher Esc fire (module-level, survives the disarm
# gap BETWEEN tools — a second Esc that lands during the next tool's armed
# window still escalates). Drives the ladder window below.
_LAST_ESC_AT = [0.0]


def _ladder_window() -> float:
    """Esc-escalation window in seconds (NX_ESC_LADDER_SECONDS, default 0.8)."""
    return _env_float("NX_ESC_LADDER_SECONDS", 0.8)


def cancel_requested() -> bool:
    """Cooperative cancellation check for long waits and dispatch boundaries."""
    return _HARD_CANCEL.is_set()


def register_tool_cancel(fn) -> None:
    """Publish the current tool's kill hook (called by run_bash/wait_background
    around their poll window; cleared in their finally)."""
    global _CANCEL_FN
    with _CANCEL_LOCK:
        _CANCEL_FN = fn


def clear_tool_cancel() -> None:
    global _CANCEL_FN
    with _CANCEL_LOCK:
        _CANCEL_FN = None


def request_hard_cancel(reason: str = "") -> bool:
    """Escalate: mark the turn hard-cancelled and fire the live tool hook.
    Returns whether a tool hook ran. Idempotent-safe (the event is sticky
    until the next interactive turn starts); never raises."""
    _HARD_CANCEL.set()
    with _CANCEL_LOCK:
        fn = _CANCEL_FN
    if fn is not None:
        try:
            fn()
        except Exception:
            pass  # a kill hook must never take down the caller
    return fn is not None

# ---------------------------------------------------------------- wake pipe ---
# PF-06 (0.56.0): disarming used to leave the watcher blocked in a 0.1s
# wait_ready() poll before it noticed the cleared arm flag and restored cooked
# input mode — so every turn paid up to 100ms of dead time *after* its last
# token (2.5s if the watcher was mid-drain). Main now pokes this pipe on disarm
# so the watcher's select() returns at once. POSIX-only; on Windows (or if the
# pipe can't be created) _wake_watcher() is a no-op and the watcher falls back
# to its poll interval, exactly as before.
if IS_WINDOWS:
    _WAKE_R: int | None = None
    _WAKE_W: int | None = None
else:
    try:
        _WAKE_R, _WAKE_W = os.pipe()
        # Both ends non-blocking: _WAKE_W so a full pipe can never block main,
        # _WAKE_R so _drain_wake()'s read loop returns instead of blocking on an
        # empty pipe (select still works fine on a non-blocking fd).
        os.set_blocking(_WAKE_R, False)
        os.set_blocking(_WAKE_W, False)
    except OSError:  # pragma: no cover - fd exhaustion / restricted env
        _WAKE_R = _WAKE_W = None


def _wake_watcher() -> None:
    """Poke the watcher out of its stdin select. Best-effort, never raises."""
    if _WAKE_W is None:
        return
    try:
        os.write(_WAKE_W, b"\x00")
    except (OSError, BlockingIOError, ValueError):
        pass


def _drain_wake() -> None:
    """Consume any pending wake bytes so a stale poke can't make the next
    select() return immediately and spin."""
    if _WAKE_R is None:
        return
    try:
        while os.read(_WAKE_R, 4096):
            pass
    except (OSError, BlockingIOError, ValueError):
        pass


def _disarm_interrupt_armed() -> None:
    """Clear the arm flag AND wake the watcher so it restores cooked input mode
    immediately instead of on its next poll tick. The disarm→restore wait in
    model_turn/_watch_for_interrupts then returns in single-digit ms in the
    common case, which is what PF-06 (0.56.0) is about."""
    _INTERRUPT_ARMED.clear()
    _wake_watcher()


def _interrupt_watcher() -> None:
    """Persistent daemon: park while idle, read stdin for bare Esc while armed.

    When armed (during model generation) stdin is put into raw mode while
    leaving ISIG enabled, so Ctrl+C still kills the process. A bare Esc (not
    the start of an arrow-key / bracketed-paste / etc. CSI sequence) sets
    _USER_INTERRUPTED. When disarmed (between turns, while the chatbox owns
    stdin) we restore the input mode and park on _INTERRUPT_ARMED — doing
    NOTHING with stdin, so the chatbox / approval prompt get every keystroke.

    Single-reader invariant: this is the only thread that ever calls os.read on
    stdin, and only while armed. Parked = hands off stdin entirely. That's what
    stops the keystroke-eating leaks the spawn-per-turn design suffered from.
    """
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return  # stdin has no fileno (pytest capture, redirect, etc.)
    # Snapshot the cooked input mode once (a POSIX termios attr list or a
    # Windows console-mode token, opaque to this module). None => no
    # controlling terminal (pytest capture, redirect) => the watcher can't
    # run, matching the old early-return.
    old = capture_input_mode(fd)
    if old is None:
        return
    while not _INTERRUPT_EXIT.is_set():
        # Park until a generation arms us. While parked we touch stdin NOT AT
        # ALL — the chatbox owns it. This is the whole point of the redesign.
        # A 25ms slice bounds how long an _INTERRUPT_EXIT (process shutdown)
        # can be ignored without paying a wake poke for that path too.
        _INTERRUPT_ARMED.wait(0.025)
        if not _INTERRUPT_ARMED.is_set() or _INTERRUPT_EXIT.is_set():
            continue
        # We're armed: take stdin into raw mode for the duration of generation.
        # ISIG (POSIX) / ENABLE_PROCESSED_INPUT (Windows) stays ON so Ctrl+C
        # still generates SIGINT instead of arriving as a byte this Esc-only
        # watcher would discard.
        if not _run_armed_session(fd, old):
            return  # stdin died / could not arm — watcher can't run


def _run_armed_session(fd: int, old: Any) -> bool:
    """Run one armed window: raw input mode, Esc watching, then restore.

    Returns True to stay alive (park again), False to exit the watcher thread
    entirely (stdin died / apply failed)."""
    last_fire = 0.0
    _dash_last = [0.0]  # last live-region repaint (watcher-tick driver)
    try:
        apply_interrupt_mode(fd, old)
    except Exception:
        _INTERRUPT_RESTORED.set()  # never armed — nothing to restore
        return False
    try:
        _drain_wake()  # drop any stale poke so the first select doesn't spin
        while _INTERRUPT_ARMED.is_set() and not _INTERRUPT_EXIT.is_set():
            # Block on stdin AND the wake pipe (PF-06): main pokes the pipe
            # when it disarms, so we notice immediately instead of after up
            # to 100ms of poll. Re-check the arm flag even when the wake pipe
            # (not stdin) fired.
            # Phase 3 live-region driver (EVENTSURFACE_SPEC §2/§3), turn-loop-
            # owned terminal: the watcher IS the tick — each ~100ms poll
            # iteration that finds nothing to read also repaints the live
            # region if it is open (4 fps cap via interval). While the chatbox
            # owns the terminal the watcher is PARKED (hands stdin off), so
            # this hook cannot fire — the chatbox's own read loop does it.
            ready_fd = _select_stdin_or_wake(fd, 0.1)
            if ready_fd is None:
                try:
                    from ._surfaces import repaint_live_tick, dashboard_open, \
                        LIVE_FPS_INTERVAL
                    if dashboard_open():
                        now = time.time()
                        if now - _dash_last[0] >= LIVE_FPS_INTERVAL:
                            _dash_last[0] = now
                            repaint_live_tick()
                except Exception:
                    pass
                continue
            if ready_fd != fd:
                _drain_wake()
                continue  # woken to disarm — loop re-checks _INTERRUPT_ARMED
            try:
                c = read_byte(fd)
            except OSError:
                _INTERRUPT_RESTORED.set()  # dying — unblock main's wait
                return False
            if c != b"\x1b":
                # 0.105.0 steering: while a turn is active, printable input is
                # no longer discarded — it accumulates as a capture buffer and
                # Enter queues it as a steering note (nx/_surfaces.py). The
                # turn loop injects it at its next step boundary. Esc keeps
                # its interrupt meaning below; other bytes (Ctrl+C etc.) are
                # left to the terminal/ISIG as before.
                if st._TURN_ACTIVE and _INTERRUPT_CAPTURE_REDIRECT.is_set():
                    _capture_steering_byte(fd, c)
                continue
            # Could be bare Esc OR the lead byte of an escape sequence
            # (arrow keys, Home/End, bracketed paste, etc.). Wait up to 50ms
            # for more bytes; if none arrive, it's a bare Esc.
            if wait_ready(fd, 0.05):
                try:
                    read_byte(fd)  # swallow the rest of the sequence
                except OSError:
                    pass
                continue
            now = time.time()
            gap = now - last_fire
            if gap < 0.08:  # key-repeat debounce — a held key is not a ladder
                continue
            # C1 ladder: a second Esc inside NX_ESC_LADDER_SECONDS escalates
            # to a hard cancel of the running tool (see _HARD_CANCEL above).
            if 0.08 <= gap <= _ladder_window():
                request_hard_cancel("double Esc")
            last_fire = now
            _LAST_ESC_AT[0] = now
            _USER_INTERRUPTED.set()
            # Leave _INTERRUPT_ARMED set: main will clear it on disarm,
            # which exits the inner loop and parks us. In the meantime
            # (main is closing the stream / breaking out of its chunk
            # loop), keep reading stdin so a user who immediately starts
            # typing a redirect ("actually, do X instead") doesn't have
            # those keystrokes silently discarded — capture them below
            # instead of just discarding non-Esc bytes.
            if _INTERRUPT_CAPTURE_REDIRECT.is_set():
                st._INTERRUPT_REDIRECT_TEXT = _drain_redirect_text(fd, old)
                st._INTERRUPT_REDIRECT_SUBMITTED = st._INTERRUPT_REDIRECT_TEXT.endswith("\n")
                st._INTERRUPT_REDIRECT_TEXT = st._INTERRUPT_REDIRECT_TEXT.rstrip("\n")
    finally:
        restore_input_mode(fd, old)
        # Signal main that input mode has been restored to cooked. main
        # waits on this after clearing _INTERRUPT_ARMED so the chatbox
        # never captures our raw-mode attrs as its "old" state.
        _INTERRUPT_RESTORED.set()
    return True


def _capture_steering_byte(fd: int, c: bytes) -> None:
    """Fold one printable (or backspace/Enter) byte into the steering capture.

    Ctrl-T (\x14) is intercepted FIRST (EVENTSURFACE_SPEC §3): it toggles the
    /tabs live dashboard instead of becoming steering text — today it used to
    be silently dropped by the printable check below. The byte is consumed
    either way. Enter (\r) commits the buffer as a queued steering note;
    backspace edits it; printable bytes append. Echoes minimally (dim) so the
    user sees the line they are building without disturbing the streamed
    output above. Kept tiny and exception-safe: it runs on the watcher thread
    between chunks and must never break generation."""
    from ._surfaces import queue_steering, set_steer_partial, steer_partial
    try:
        if c == b"\x14":
            from ._surfaces import toggle_dashboard, repaint_live_tick
            toggle_dashboard()
            repaint_live_tick()  # paint the just-opened region on this tick
            return
        if c in (b"\r", b"\n"):
            line = steer_partial()
            if line.strip():
                queue_steering(line)
                sys.stdout.write(f"\r{DIM}  ▸ steered: {line.strip()[:60]}{RESET}\n")
                sys.stdout.flush()
            set_steer_partial("")
            return
        if c in (b"\x7f", b"\x08"):
            set_steer_partial(steer_partial()[:-1])
            return
        try:
            ch = c.decode("utf-8")
        except UnicodeDecodeError:
            return
        if ch >= " ":
            set_steer_partial(steer_partial() + ch)
    except Exception:
        pass


def _select_stdin_or_wake(fd: int, timeout: float) -> int | None:
    """Return the fd that became readable (stdin `fd` or the wake pipe), or
    None on timeout. Falls back to a plain stdin poll when there is no wake
    pipe (Windows, or the pipe could not be created)."""
    if _WAKE_R is None:
        return fd if wait_ready(fd, timeout) else None
    return wait_ready_any([fd, _WAKE_R], timeout)


# Exposed for the interrupt-watcher tests (PF-06, T-8): the direct watcher-loop
# entry point so a test can drive one armed window over a real pipe without a
# tty, and measure the disarm latency.
_armed_loop = _run_armed_session


def _drain_redirect_text(fd: int, old_mode: Any) -> str:
    """After a bare Esc fires, keep reading stdin (still in raw mode) for a
    short grace window so a user typing a redirect right away isn't cut off
    mid-word. Returns the typed text, ending in "\n" if Enter was pressed
    before the window closed (a full line, ready to inject immediately) or
    without it if the window simply ran out (a partial line, used as a prefill
    for the next chatbox prompt instead).

    Deliberately dumb: no arrow-key/history support, backspace only. This is
    a stopgap capture buffer for the handoff window, not a second input
    widget — full editing happens in the chatbox once the redirect is either
    injected or handed over as a prefill. `old_mode` (the captured input
    mode) is accepted for call-site symmetry but unused — the drain assumes
    the caller already put the fd into raw mode and only reads bytes.
    """
    buf: list[str] = []
    deadline = time.time() + 2.0  # generous — covers stream-close + tool teardown
    # Deliberately NOT gated on _INTERRUPT_ARMED: main clears that the instant
    # it notices the interrupt (next chunk / stream-close), which is almost
    # immediately — long before this deadline. Gating on it here used to
    # collapse the 2s typing window down to near-zero, silently dropping
    # whatever the user typed. Only the deadline or process exit should stop us.
    while not _INTERRUPT_EXIT.is_set() and time.time() < deadline:
        if not wait_ready(fd, 0.05):
            continue
        try:
            c = read_byte(fd)
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
            # C1 ladder: the drain runs right after the first Esc fired, so
            # an Esc inside the window here is the 2x-press escalation too.
            if time.time() - _LAST_ESC_AT[0] <= _ladder_window():
                request_hard_cancel("double Esc")
            break
        try:
            ch = c.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if ch >= " ":
            buf.append(ch)
    return "".join(buf)


def _watcher_disabled() -> bool:
    """NX_INTERRUPT_WATCHER=0 opts out of the stdin watcher entirely (swarm
    P12): a headless run (`nx run -f`, --prompt-file, --parallel) has no tty
    to steal and no user to interrupt, so the thread (and its one-time
    termios snapshot) is pure per-process overhead there. A disabled watcher
    never arms: model_turn's non-quiet arm path checks this too, so an
    interactive session cannot silently lose Esc by a stale env value read
    at import — the flag is read at call time."""
    from ._env import _env_int
    return _env_int("NX_INTERRUPT_WATCHER", 1) == 0


def _ensure_interrupt_watcher() -> None:
    """Start the singleton interrupt watcher thread if it isn't already alive.

    Called lazily before the first model turn (and defensively on every turn —
    cheap because a living thread short-circuits the check). Guaranteed exactly
    one reader of stdin across the whole process lifetime.

    Swarm P12: a QUIET loop (async sub-agent, batch worker) must never START
    the thread — it will never arm it, so the start is pure overhead (one
    thread + a termios snapshot) multiplied by every agent in an in-process
    swarm; the owner of stdin starts it when its first non-quiet turn runs.
    NX_INTERRUPT_WATCHER=0 disables it for the whole process (headless
    deployments)."""
    if st._INTERRUPT_THREAD is not None and st._INTERRUPT_THREAD.is_alive():
        return
    if _watcher_disabled() or _subagent_ctx() is not None:
        return
    _INTERRUPT_EXIT.clear()
    _INTERRUPT_ARMED.clear()
    _INTERRUPT_RESTORED.set()  # fresh thread: nothing to restore yet
    _USER_INTERRUPTED.clear()
    _HARD_CANCEL.clear()
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
    while Esc is meaningful. Keeping the arm/disarm and input-mode-restore
    handshake here makes those windows identical, and guarantees the watcher
    is parked again before the chatbox or another operation can read input.

    Swarm P12: NX_INTERRUPT_WATCHER=0 disables the whole handshake — a
    no-op yield (nothing ever arms, _USER_INTERRUPTED stays clear, and the
    disarm path's restore-wait is skipped)."""
    if _watcher_disabled():
        yield
        return
    _ensure_interrupt_watcher()
    capture_redirect = _INTERRUPT_CAPTURE_REDIRECT.is_set()
    _INTERRUPT_CAPTURE_REDIRECT.clear()
    _USER_INTERRUPTED.clear()
    _INTERRUPT_RESTORED.clear()
    _INTERRUPT_ARMED.set()
    # _drain_redirect_text's own deadline is 2.0s; the default 2.5s wait covers
    # it with slack so we never proceed while the watcher still holds raw
    # input mode mid-drain. NX_INTERRUPT_RESTORE_WAIT tunes it — tests shrink
    # it so they don't pay the full drain window 6+ times per run.
    restore_wait = _env_float("NX_INTERRUPT_RESTORE_WAIT", 2.5)
    try:
        yield
    finally:
        # Clear + wake so the watcher restores cooked mode immediately, not on
        # its next poll tick (PF-06).
        _disarm_interrupt_armed()
        # _drain_redirect_text's own deadline is 2.0s; wait past it so we never
        # proceed while the watcher still holds raw input mode mid-drain.
        if not _INTERRUPT_RESTORED.wait(timeout=restore_wait):
            print(
                f"nx: warning: interrupt watcher did not confirm input-mode "
                f"restore within {restore_wait:g}s; terminal state may be stale",
                file=sys.stderr,
            )
        # Do not let an Esc from this operation interrupt a later model turn.
        _USER_INTERRUPTED.clear()
        if capture_redirect:
            _INTERRUPT_CAPTURE_REDIRECT.set()
