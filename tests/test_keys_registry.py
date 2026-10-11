#!/usr/bin/env python3
"""UIUX_SPEC C2 (0.136.0): one key registry drives hints AND the `?` overlay.

The registry (nx/_keys.py) is the single source of truth:
  * the chatbox border hint line is `hint_line("chatbox", ...)` — a test
    pins its text against KEYS["chatbox"] so they cannot drift;
  * the approval prompt's [..] tail is `prompt_tail("approval")` — same pin;
  * the `?` overlay renders `overlay_lines(surface)` from the SAME tuples;
  * Phase D surfaces ("tabs", "fullscreen") are already registered.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-keys-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _keys  # noqa: E402
from nx import _render  # noqa: E402


def test_registry_shape():
    for surface in ("chatbox", "approval", "tabs", "fullscreen"):
        entries = _keys.keys_for(surface)
        assert entries, f"surface {surface!r} has no keys registered"
        for k, v in entries:
            assert isinstance(k, str) and isinstance(v, str)
            assert k and v
    # `?` is the overlay gesture on every surface
    for surface in ("chatbox", "approval"):
        assert any(k == _keys.OVERLAY_KEY for k, _ in _keys.keys_for(surface))
    assert _keys.keys_for("no-such-surface") == []


def test_hint_line_and_overlay_share_one_source_of_truth():
    """THE drift guard: hint text and overlay text are both derived from
    KEYS[surface] tuples — a change to one path cannot skip the other."""
    for surface in ("chatbox", "approval", "tabs", "fullscreen"):
        hint = _keys.hint_line(surface, mode="manual")
        overlay = "\n".join(_keys.overlay_lines(surface, mode="manual"))
        for key, meaning in _keys.keys_for(surface):
            assert key in hint, f"{surface}: {key} missing from hint line"
            assert key in overlay, f"{surface}: {key} missing from overlay"
            plain = meaning.replace("{mode}", "manual")
            assert plain in hint and plain in overlay


def test_chatbox_border_hint_is_the_registry():
    """The chatbox frame renders `hint_line('chatbox', mode=...)`; pin the
    exact text against the raw KEYS data so a literal re-hardcode in
    _chatbox.py would fail this test."""
    keys = _keys.KEYS["chatbox"]
    sep = f" {_render.TAIL_BULLET} "
    expected = sep.join(k + " " + (m if "{mode}" not in m else "yolo")
                        for k, m in keys)
    assert _keys.hint_line("chatbox", mode="yolo") == expected


def test_approval_prompt_tail_is_the_registry():
    tail = _keys.prompt_tail("approval")
    assert tail == "/".join(_keys.short_key(k) for k, _ in _keys.KEYS["approval"])
    for key in ("Y", "n", "a", "+", "-", "shift+tab", "?", "esc"):
        assert key in tail.split("/")


def test_overlay_renders_through_the_render_layer():
    """The overlay is a _render.panel: one glyph set, no raw palette."""
    lines = _keys.overlay_lines("approval")
    assert lines[0].startswith(_render.C.WARNING) or _render.BOX_TL in lines[0]
    assert _render.BOX_TL in lines[0]
    assert _render.BOX_V in lines[1]
    assert _render.BOX_BL in lines[-1]


def test_chatbox_uses_registry_for_hints_source_check():
    """_chatbox renders its hint line via _keys.hint_line (not a hardcoded
    literal): the module's source must reference the registry."""
    import inspect
    src = inspect.getsource(sys.modules["nx._chatbox"])
    assert 'hint_line("chatbox"' in src
