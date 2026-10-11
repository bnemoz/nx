#!/usr/bin/env python3
"""UIUX_SPEC C3 (0.136.0): approval prompt polish.

Driven through the real raw-mode `_ask_approval` with the same scripted-byte
harness as tests/test_approval_shifttab.py (no TTY needed):
  * +/- widen/narrow the diff context (default _layout.DIFF_CONTEXT_LINES);
  * `?` renders the registry overlay above the prompt, any key dismisses;
  * Shift+Tab proceeds with the pending action (pinned in
    test_approval_shifttab.py too);
  * the prompt shows the resolved reason (risk level) on one line and its key
    tail comes from the nx/_keys registry.
"""
import builtins
import os
import sys
import tempfile
from contextlib import contextmanager

_tmp = tempfile.mkdtemp(prefix="nx-c3-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _layout  # noqa: E402

BEFORE = "\n".join(f"line {i}" for i in range(1, 21))
AFTER = BEFORE.replace("line 10", "line 10 changed")


def setup_function(_f):
    os.environ["NX_SESSIONS_DIR"] = _tmp
    os.environ["NX_HOME"] = _tmp
    m.YOLO = False
    m.APPROVE_LEVEL = None
    m.DEFAULT_APPROVE_LEVEL = None


def _drive(bursts, diff_src=None):
    """Scripted keystrokes into the real _ask_approval; returns (result, printed)."""
    apmod = sys.modules["nx._approval"]
    saved = {"read_byte": apmod.read_byte, "wait_ready": apmod.wait_ready,
             "raw_mode": apmod.raw_mode, "stdin": apmod.sys.stdin,
             "stdout": apmod.sys.stdout, "print": builtins.print}
    printed = []
    state = {"gi": 0, "pos": 0}

    def fake_read_byte(fd):
        gi, pos = state["gi"], state["pos"]
        if gi >= len(bursts):
            raise AssertionError("read past scripted bursts")
        if pos >= len(bursts[gi]):
            gi, pos = gi + 1, 0
            state["gi"], state["pos"] = gi, pos
            if gi >= len(bursts):
                raise AssertionError("read past scripted bursts")
        b = bursts[gi][pos]
        state["pos"] = pos + 1
        return b

    def fake_wait_ready(fd, timeout):
        gi, pos = state["gi"], state["pos"]
        return gi < len(bursts) and pos < len(bursts[gi])

    @contextmanager
    def fake_raw_mode(fd, **kw):
        yield [0, 0, 0, 0, 0, 0, [0] * 16]

    class FakeStdin:
        def isatty(self): return True
        def fileno(self): return 0

    class FakeStdout:
        def isatty(self): return True
        def write(self, s): printed.append(s)
        def flush(self): pass

    apmod.read_byte = fake_read_byte
    apmod.wait_ready = fake_wait_ready
    apmod.raw_mode = fake_raw_mode
    apmod.sys.stdin = FakeStdin()
    apmod.sys.stdout = FakeStdout()
    builtins.print = lambda *a, **k: None
    try:
        result = apmod._ask_approval("allow edit foo.py? [Y/n/a/esc] ",
                                     allow_hint="edit foo.py",
                                     diff_src=diff_src)
        return result, "".join(printed)
    finally:
        apmod.read_byte = saved["read_byte"]
        apmod.wait_ready = saved["wait_ready"]
        apmod.raw_mode = saved["raw_mode"]
        apmod.sys.stdin = saved["stdin"]
        apmod.sys.stdout = saved["stdout"]
        builtins.print = saved["print"]


# ------------------------------------------------------------------ +/- keys

def test_plus_widens_diff_context_from_layout_default():
    """Default context is _layout.DIFF_CONTEXT_LINES; `+` re-renders with one
    more context line and keeps the prompt open."""
    result, printed = _drive([[b"+"], [b"y"]], diff_src=(BEFORE, AFTER))
    assert result == "y"
    assert f"diff context {_layout.DIFF_CONTEXT_LINES + 1}" in printed, printed
    # A wider diff must show MORE untouched context than the default render:
    # count context lines (unchanged lines ride through the diff component).
    from nx._render import diff as rdiff
    wide = rdiff(BEFORE, AFTER, _layout.DIFF_CONTEXT_LINES + 1)
    narrow = rdiff(BEFORE, AFTER, _layout.DIFF_CONTEXT_LINES)
    assert wide.count("line ") > narrow.count("line ")
    assert wide in printed  # the widened diff was actually printed

def test_minus_narrows_then_zero_floor_holds():
    result, printed = _drive([[b"-"], [b"-"], [b"-"], [b"n"]],
                             diff_src=(BEFORE, AFTER))
    assert result == "n"
    ctx2, ctx1, ctx0 = (_layout.DIFF_CONTEXT_LINES,
                        _layout.DIFF_CONTEXT_LINES - 1,
                        max(0, _layout.DIFF_CONTEXT_LINES - 2))
    assert f"diff context {ctx1}" in printed
    assert f"diff context {ctx0}" in printed
    # never below 0
    assert "diff context -1" not in printed

def test_plus_minus_inert_without_structured_diff():
    """Bash/commit previews have no before/after pair: +/- re-prompt silently
    (like any unrecognized key at those prompts) instead of erroring."""
    result, _ = _drive([[b"+"], [b"y"]], diff_src=None)
    assert result == "y"


# ----------------------------------------------------------------- ? overlay

def test_question_mark_renders_registry_overlay_then_dismisses(monkeypatch):
    from nx import _keys
    from nx import _render
    apmod = sys.modules["nx._approval"]
    overlays = []
    monkeypatch.setattr(apmod, "print_overlay",
                        lambda surface, **kw: overlays.append(
                            (surface, _keys.overlay_lines(surface, **kw))))
    result, printed = _drive([[b"?"], [b"y"]], diff_src=(BEFORE, AFTER))
    assert result == "y", "any key dismisses the overlay and the prompt continues"
    assert overlays and overlays[0][0] == "approval"
    rendered = "\n".join(overlays[0][1])
    # the overlay shows the approval-surface keys...
    for key, _meaning in _keys.keys_for("approval"):
        assert key in rendered, f"{key} missing from overlay render"
    # ...as a _render panel (one glyph set)
    assert _render.BOX_TL in rendered and _render.BOX_BL in rendered


# --------------------------------------------------- reason line + hint tail

def test_confirm_prompt_shows_reason_and_registry_tail(monkeypatch):
    """The prompt line carries the resolved risk level + reason on one line,
    and its key tail is generated from the nx/_keys registry."""
    from nx import _keys
    monkeypatch.setattr(m, "_assess_risk", lambda a: ("high", "writes over user data"))
    prompts = []
    monkeypatch.setattr(m, "_ask_approval",
                        lambda prompt, allow_hint=None, **kw:
                        prompts.append(prompt) or "y")
    # an action _local_risk_hint does NOT pre-classify, so the (patched)
    # classifier's level+reason are what the prompt line must show
    assert m._confirm("run: customtool --flag", preview=None) is True
    assert len(prompts) == 1
    p = prompts[0]
    assert p.count("\n") == 0, "reason must be on ONE line"
    assert "HIGH" in p and "writes over user data" in p
    assert _keys.prompt_tail("approval") in p


def test_confirm_passes_diff_src_through_to_ask_approval(monkeypatch):
    """_confirm(write ...) forwards the structured (before, after) pair so the
    prompt can widen/narrow, alongside the rendered preview."""
    monkeypatch.setattr(m, "_assess_risk", lambda a: ("medium", "write"))
    seen = {}

    def ask(prompt, allow_hint=None, diff_src=None, context=None):
        seen["diff_src"] = diff_src
        seen["context"] = context
        return "y"

    monkeypatch.setattr(m, "_ask_approval", ask)
    assert m._confirm("write foo.py (3 bytes)", preview="pretend",
                      diff_src=("a\n", "b\n")) is True
    assert seen["diff_src"] == ("a\n", "b\n")
    # default context comes from _layout (via the _ask_approval default; here
    # the patched lambda sees None because _confirm only forwards diff_src)
    assert seen["context"] is None
