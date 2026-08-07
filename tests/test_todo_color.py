"""Per-status coloring of the todo_write list (render_todos_display).

render_todos() returns plain text for the model (the todo_write tool result
body that enters message history); render_todos_display() returns the same
list colored by status for the screen — done=green, in_progress=yellow,
pending=orange — so the three states are visually distinct. Item text is
sanitized of ANSI/control sequences before nx's own color codes are applied,
so model-supplied text cannot smuggle a terminal escape through.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-todocolor-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
for _k in [k for k in os.environ if k.startswith("NX_")]:
    del os.environ[_k]
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _state as st  # noqa: E402
from nx._ansi import C, RESET  # noqa: E402
from nx._todo import render_todos, render_todos_display  # noqa: E402


def test_display_colors_each_status_distinctly():
    st._TODOS = [
        {"text": "done item", "status": "done"},
        {"text": "wip item", "status": "in_progress"},
        {"text": "pending item", "status": "pending"},
    ]
    lines = render_todos_display().splitlines()
    assert len(lines) == 3
    # each status maps to its semantic hue
    assert lines[0].startswith(C.SUCCESS), "done should be green (SUCCESS)"
    assert lines[1].startswith(C.WARNING), "in_progress should be yellow (WARNING)"
    assert lines[2].startswith(C.TODO), "pending should be orange (TODO)"
    # the three hues are genuinely distinct
    assert len({C.SUCCESS, C.WARNING, C.TODO}) == 3
    # status marks and item text survive
    assert "[x]" in lines[0] and "done item" in lines[0]
    assert "[~]" in lines[1] and "wip item" in lines[1]
    assert "[ ]" in lines[2] and "pending item" in lines[2]
    # every line is RESET-terminated (no bleed)
    assert all(ln.endswith(RESET) for ln in lines)


def test_plain_render_has_no_color_codes():
    st._TODOS = [
        {"text": "done item", "status": "done"},
        {"text": "wip item", "status": "in_progress"},
        {"text": "pending item", "status": "pending"},
    ]
    out = render_todos()
    # the model-facing copy must stay plain — no SGR escapes at all
    assert "\033[" not in out, f"render_todos leaked color codes: {out!r}"
    assert "[x]" in out and "[~]" in out and "[ ]" in out


def test_display_strips_embedded_ansi_from_item_text():
    # a malicious/careless model could put a terminal escape in item text;
    # render_todos_display must strip it before applying its own color.
    esc = "\033[?25l"  # cursor-hide
    st._TODOS = [{"text": f"evil{esc}task", "status": "pending"}]
    out = render_todos_display()
    assert esc not in out, f"embedded escape survived sanitization: {out!r}"
    assert "eviltask" in out


def test_empty_list_display():
    st._TODOS = []
    assert render_todos_display() == "(todo list empty)"


def test_unknown_status_falls_back_to_pending_color():
    st._TODOS = [{"text": "weird", "status": "frobnicated"}]
    out = render_todos_display()
    # unknown status -> "[ ]" mark and TODO (orange) hue, same as pending
    assert out.startswith(C.TODO)
    assert "[ ]" in out
    assert "weird" in out


if __name__ == "__main__":
    for fn in [
        test_display_colors_each_status_distinctly,
        test_plain_render_has_no_color_codes,
        test_display_strips_embedded_ansi_from_item_text,
        test_empty_list_display,
        test_unknown_status_falls_back_to_pending_color,
    ]:
        fn()
        print(f"OK — {fn.__name__}")
    print("OK — todo display coloring")