#!/usr/bin/env python3
"""Regression test: Shift+Tab toggles YOLO at the approval prompt.

Before the fix, a user who started the session in manual (non-yolo) mode
could only toggle YOLO with Shift+Tab while the chatbox owned stdin — i.e.
between turns. Once a model turn was in flight and a write/bash approval
prompt appeared, Shift+Tab was swallowed (the prompt only accepted
Y/n/esc), so the user was stuck in manual mode until the turn finished
and the chatbox came back. The fix lets Shift+Tab toggle YOLO right at
the approval prompt, mirroring the chatbox editor, with a visible
confirmation and a re-prompt of the SAME action (the toggle affects
future approvals, not the one being decided).

No real TTY is needed: the approval prompt's raw-mode branch is driven by
monkeypatching os.read (scripted keystroke bytes), select.select (so the
escape-sequence drain sees the trailing bytes of the current burst, then
goes empty — exactly like a real terminal where the rest of an escape
sequence is already in the kernel buffer but the user's NEXT keypress
hasn't happened yet), raw_mode (no-op), and the tty checks.
"""
import builtins
import os
import sys
import tempfile
from contextlib import contextmanager

_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402  (path must be set on sys.path first, like every other test)

_REAL_PRINT = builtins.print


def setup_function(_function):
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    # Each test starts from a clean, non-yolo, prompt-everything state.
    m.YOLO = False
    m.APPROVE_LEVEL = None
    m.DEFAULT_APPROVE_LEVEL = None


def _drive_approval(bursts: list[list[bytes]]):
    """Drive the real `_ask_approval` raw-mode branch with scripted bytes.

    `bursts` is a list of byte-bursts, each one a sequence of bytes that
    arrive atomically (an escape sequence like Shift+Tab, or a single
    keypress like Enter/'n'). Inside _ask_approval the escape-sequence
    drain polls select for more bytes; bytes within the *current* burst
    are readable, then select returns empty when the burst is exhausted —
    mirroring a real terminal where the trailing bytes of an escape
    sequence are already queued but the user's next keypress hasn't
    happened yet. The next burst only becomes readable once a fresh
    os.read starts it (i.e. after the prompt re-renders).

    Returns (result, printed_text).
    """
    apmod = sys.modules["nx._approval"]
    saved = {
        "read": apmod.os.read,
        "select": apmod.select.select,
        "raw_mode": apmod.raw_mode,
        "stdin": apmod.sys.stdin,
        "stdout": apmod.sys.stdout,
        "print": builtins.print,
    }
    printed: list[str] = []
    state = {"gi": 0, "pos": 0}

    def fake_read(fd, n):
        gi, pos = state["gi"], state["pos"]
        if gi >= len(bursts):
            raise AssertionError(f"_ask_approval read past all scripted bursts (gi={gi})")
        if pos >= len(bursts[gi]):
            # current burst exhausted — a fresh keypress starts the next one
            gi += 1
            pos = 0
            state["gi"] = gi
            state["pos"] = pos
            if gi >= len(bursts):
                raise AssertionError(f"_ask_approval read past all scripted bursts (gi={gi})")
        b = bursts[gi][pos]
        state["pos"] = pos + 1
        return b

    def fake_select(r, w, x, timeout):
        # readable iff the current burst still has unconsumed bytes
        gi, pos = state["gi"], state["pos"]
        if gi < len(bursts) and pos < len(bursts[gi]):
            return r, [], []
        return [], [], []

    @contextmanager
    def fake_raw_mode(fd, **kw):
        yield [0, 0, 0, 0, 0, 0, [0] * 16]

    class FakeStdin:
        def isatty(self):
            return True
        def fileno(self):
            return 0

    class FakeStdout:
        def isatty(self):
            return True
        def write(self, s):
            printed.append(s)
        def flush(self):
            pass

    apmod.os.read = fake_read
    apmod.select.select = fake_select
    apmod.raw_mode = fake_raw_mode
    apmod.sys.stdin = FakeStdin()
    apmod.sys.stdout = FakeStdout()
    builtins.print = lambda *a, **k: None
    try:
        result = apmod._ask_approval("allow write foo.py? [Y/n/esc] ")
        return result, "".join(printed)
    finally:
        apmod.os.read = saved["read"]
        apmod.select.select = saved["select"]
        apmod.raw_mode = saved["raw_mode"]
        apmod.sys.stdin = saved["stdin"]
        apmod.sys.stdout = saved["stdout"]
        builtins.print = saved["print"]


def test_shift_tab_toggles_yolo_on_at_approval_prompt():
    """Shift+Tab (CSI [Z) at an approval prompt flips YOLO on, then
    re-prompts; the following Enter approves the action (returns 'y')."""
    assert m.YOLO is False
    # burst 1: Shift+Tab  (\x1b[Z)   burst 2: Enter (approve, the default)
    result, printed = _drive_approval([[b"\x1b", b"[", b"Z"], [b"\r"]])
    assert result == "y", f"expected 'y' after Shift+Tab then Enter, got {result!r}"
    assert m.YOLO is True, "YOLO should be ON after Shift+Tab at the approval prompt"
    # A visible confirmation must be printed — never a silent flip.
    assert "yolo" in printed.lower() and "ON" in printed, \
        f"expected a visible yolo-ON confirmation, got: {printed!r}"
    print("PASS — Shift+Tab [Z toggles YOLO on at the approval prompt, then re-prompts")


def test_shift_tab_csi_1_2z_variant_toggles_yolo():
    """The \x1b[1;2Z Shift+Tab encoding (some terminals) also toggles YOLO."""
    assert m.YOLO is False
    result, printed = _drive_approval([
        [b"\x1b", b"[", b"1", b";", b"2", b"Z"], [b"\r"]])
    assert result == "y", f"expected 'y', got {result!r}"
    assert m.YOLO is True, "YOLO should be ON after \\x1b[1;2Z at the approval prompt"
    assert "ON" in printed, f"expected a visible yolo-ON confirmation, got: {printed!r}"
    print("PASS — Shift+Tab [1;2Z variant toggles YOLO on at the approval prompt")


def test_shift_tab_then_toggles_yolo_back_off():
    """Two Shift+Tabs at the prompt cancel out: YOLO ends back off, and the
    final 'n' still denies the action."""
    assert m.YOLO is False
    result, printed = _drive_approval([
        [b"\x1b", b"[", b"Z"],     # first Shift+Tab -> yolo ON
        [b"\x1b", b"[", b"Z"],     # second Shift+Tab -> yolo OFF
        [b"n"],
    ])
    assert result == "n", f"expected 'n', got {result!r}"
    assert m.YOLO is False, "YOLO should be OFF after two Shift+Tabs"
    # Both confirmations printed: ON then OFF.
    assert "ON" in printed and "OFF" in printed, \
        f"expected both yolo-ON and yolo-OFF confirmations, got: {printed!r}"
    print("PASS — two Shift+Tabs cancel out; YOLO back to off, prompt returns 'n'")


def test_bare_esc_still_returns_esc():
    """The Shift+Tab handling must not break the bare-Esc path: a lone Esc
    (no trailing bytes within the poll window) still returns 'esc' and does
    NOT toggle YOLO."""
    assert m.YOLO is False
    # A single-byte burst: just \x1b, nothing follows -> bare Esc.
    result, printed = _drive_approval([[b"\x1b"]])
    assert result == "esc", f"bare Esc should return 'esc', got {result!r}"
    assert m.YOLO is False, "bare Esc must not toggle YOLO"
    print("PASS — bare Esc still returns 'esc' (Shift+Tab handling didn't break it)")


def test_unrecognized_sequence_reprompts_not_toggle():
    """A non-Shift+Tab escape sequence (e.g. an arrow key \x1b[A) must NOT
    toggle YOLO; it falls through to the 'unrecognized key sequence'
    reprompt, after which a normal Y/n key still works."""
    assert m.YOLO is False
    # burst 1: Up arrow (\x1b[A)   burst 2: 'n'
    result, printed = _drive_approval([[b"\x1b", b"[", b"A"], [b"n"]])
    assert result == "n", f"expected 'n' after an arrow key + n, got {result!r}"
    assert m.YOLO is False, "an arrow key must not toggle YOLO"
    assert "unrecognized" in printed.lower(), \
        f"expected an 'unrecognized key sequence' notice, got: {printed!r}"
    print("PASS — non-Shift+Tab escape sequences reprompt, don't toggle YOLO")