#!/usr/bin/env python3
"""The calm-sweep waiting animation in `nx/_ansi.py` (LifeSpinner).

Regression tests for the v0.6.2 rework: the old spinner animated a 24-cell
Game-of-Life row at ~11 Hz — every cell on the line changed between frames in
a high-contrast glyph pattern, which is both noisy and a photosensitive-
epilepsy risk (large flashing areas above ~3 Hz are the classic trigger).
It also raced with main-thread prints (approval diff previews smeared across
the bar — see the `@@ -0,0 +1,19 @@` overlap) and joined its render thread
with a 0.5s timeout, abandoning live threads mid-render.

The replacement must keep:
  * a static-width line — consecutive frames differ in ≤ 2 cells;
  * a slow, floored tick — never faster than _MIN_TICK;
  * a sanitized, clamped label — a live command's log tail must not inject
    ANSI/OSC or break the line width;
  * deterministic teardown — stop() returns only after the render thread has
    wiped its line;
  * NX_SPINNER=0 — no animation at all (one static draw, cleared on stop).

No model/network round-trip.
"""
import io
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-test-spinner-")
os.environ["NX_HOME"] = _tmp
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nx import _ansi  # noqa: E402
from nx._ansi import CLEAR_LINE, LifeSpinner  # noqa: E402


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:  # noqa: N802 — mirror the stream API
        return True


def _capture_frames(spinner, frames: int) -> list[str]:
    """Drive the render math directly (no thread) for `frames` steps."""
    out = []
    pos = 0
    for _ in range(frames):
        out.append(spinner._render(pos))
        pos = (pos + 1) % spinner._WIDTH
    return out


def test_consecutive_frames_change_at_most_two_cells():
    # The safety property: between two frames at most the runner's head
    # appears and its tail disappears — the rest of the bar never blinks.
    sp = LifeSpinner(label="running write_file")
    frames = _capture_frames(sp, 3 * sp._WIDTH)
    for a, b in zip(frames, frames[1:]):
        cells_a = a.split("[")[1].split("]")[0]
        cells_b = b.split("[")[1].split("]")[0]
        diff = sum(1 for x, y in zip(cells_a, cells_b) if x != y)
        assert diff <= 2, f"{diff} cells changed between frames: {cells_a!r} -> {cells_b!r}"


def test_rendered_line_has_constant_width():
    # A static-width line never wraps or shifts as it animates — a wrap is
    # what left droppings over the tool box.
    sp = LifeSpinner(label="thinking · esc to interrupt")
    frames = _capture_frames(sp, 2 * sp._WIDTH)
    widths = {len(_ansi.strip_ansi_for_display(f)) for f in frames}
    assert len(widths) == 1, f"line width varied across frames: {widths}"


def test_label_is_sanitized_and_clamped():
    # _bash_jobs.py streams a live command's log tail into spinner.label —
    # untrusted bytes. The render must strip ANSI/OSC (no terminal-control
    # injection, no width breakage) and clamp overlong labels.
    sp = LifeSpinner(label="\033]52;c;pawned\033\\ \033[2K boom")
    text = _ansi.strip_ansi_for_display(sp._render(0))
    assert "pawned" not in text and "\033" not in text, text

    sp.label = "x" * 200
    line = sp._render(0)
    label_part = _ansi.strip_ansi_for_display(line).split(" [")[0].strip()
    assert len(label_part) <= 48 + 3, f"overlong label not clamped: {label_part!r}"
    assert label_part.endswith("...")

    # A raw newline in a log tail must not push the bar onto a second line.
    sp.label = "line one\nline two"
    assert "\n" not in sp._render(0).replace(CLEAR_LINE, "")


def test_tick_has_floor():
    # The caller can't make the animation strobe: anything faster than
    # _MIN_TICK is clamped to it.
    sp = LifeSpinner(tick=0.001)
    assert sp.tick >= LifeSpinner._MIN_TICK


def test_stop_joins_render_thread_and_clears_line():
    # The old stop() used join(timeout=0.5) and could abandon a live render
    # thread, smearing the bar over the tool box. stop() must return only
    # after the thread is dead and its line wiped.
    out = _FakeTTY()
    real_stdout = sys.stdout
    sys.stdout = out
    try:
        sp = LifeSpinner(label="running run_bash", tick=LifeSpinner._MIN_TICK)
        sp.start()
        assert sp._t is not None, "no render thread started on a tty"
        t = sp._t
        sp.stop()
        assert not t.is_alive(), "stop() returned while the render thread was alive"
        assert sp._t is None
        assert out.getvalue().endswith(CLEAR_LINE), "spinner line not wiped on stop"
    finally:
        sys.stdout = real_stdout


def test_nx_spinner_zero_draws_static_line_without_thread():
    # NX_SPINNER=0 (photosensitivity / screen-reader opt-out): exactly one
    # draw carrying the label, no render thread, and stop() clears it.
    out = _FakeTTY()
    real_stdout, real_env = sys.stdout, os.environ.get("NX_SPINNER")
    os.environ["NX_SPINNER"] = "0"
    sys.stdout = out
    try:
        sp = LifeSpinner(label="running write_file")
        sp.start()
        assert sp._t is None, "NX_SPINNER=0 still spawned a render thread"
        drawn = out.getvalue()
        assert "running write_file" in drawn, drawn
        # one label draw + the clear on stop — nothing animates in between
        sp.stop()
        after = out.getvalue()
        assert after.count("running write_file") == 1, after
        assert after.endswith(CLEAR_LINE)
    finally:
        sys.stdout = real_stdout
        if real_env is None:
            os.environ.pop("NX_SPINNER", None)
        else:
            os.environ["NX_SPINNER"] = real_env


def test_headless_start_is_noop():
    # Piped/redirected stdout (tests, CI, nx | tee): no animation, no thread,
    # no output — the plain StringIO here is not a tty.
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        sp = LifeSpinner(label="thinking")
        sp.start()
        assert sp._t is None
        assert sys.stdout.getvalue() == ""
        sp.stop()  # must not raise either
    finally:
        sys.stdout = real_stdout


def test_concurrent_threads_clear_cleanly():
    # Many short-lived spinners (one per tool call) must each wipe their own
    # line before the next print lands — no cross-thread smearing.
    out = _FakeTTY()
    real_stdout = sys.stdout
    sys.stdout = out
    try:
        for _ in range(5):
            sp = LifeSpinner(label="running tool", tick=LifeSpinner._MIN_TICK)
            sp.start()
            sp.stop()
        text = out.getvalue()
        # every animated draw is followed by a wipe; nothing left dangling
        assert text.endswith(CLEAR_LINE)
        assert text.count("running tool") == 5
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
