"""Live progress for an in-flight tool call (0.35.0).

A tool that took 40 seconds looked exactly like one that took 0.1 seconds: the
activity line was a static `running read_file [....]` bar with no clock, no
count, and no way for a tool to say what it was doing. The user sat in the dark
for the whole call.

Three channels, all drawn by `nx._ansi.LifeSpinner._tail()`, all OFF by default
so every pre-existing spinner call site renders a byte-identical line:

  elapsed  `run_tool()` and `_turn._run_read_only_parallel()` opt in with
           `show_elapsed=True`, so the line carries seconds in flight.
  counter  a parallel fan-out shares a `ProgressCounter`, bumped as each call
           lands, so the group line reads like 2/3 instead of nothing.
  note     a long tool with no other feedback channel calls `note_progress()`,
           which appends a short message. `run_bash` needs none of this:
           `_bash_jobs._poll_pid` already streams the job's log tail.

`note_progress` writes onto `st._ACTIVE_SPINNER`, NOT a thread-local, and that
choice is load-bearing: the line is drawn by the spinner's own render thread, so
a per-thread slot would be invisible to it. Writing to the spinner object is what
the bash log tail already does (`spinner.label = tail`). A GIL-atomic attribute
swap is all the synchronization this needs; a frame shows the note from before or
after the write, never a torn one.

Notes are DISPLAY-ONLY: never in the message history, so a tool can report
freely without spending context, and one cannot lie about a later call because
every tool call builds its own fresh spinner, which starts with an empty note.

During a parallel fan-out there is no `st._ACTIVE_SPINNER` at all (workers run
quiet, which is why the group line replaced N spinners), so `note_progress()`
from a fan-out worker is a no-op by design; that line reports n/m instead.
"""
from __future__ import annotations

# Longest note a tool may SET, longest note SHOWN, and longest whole tail. The
# activity line is one row and must never wrap: a wrap leaves droppings over the
# tool box. These are the budget's only knobs; with a full tail the plain line is
# about 65 cells, under the pre-0.35 worst case (51-cell label plus the bar).
MAX_NOTE = 60
MAX_NOTE_SHOWN = 24
MAX_TAIL = 34
# The label and the tail share one budget so the row can never wrap. The label
# wins it first — the tool name is the most useful thing on the line — and the
# tail clips into the remainder, down to MIN_TAIL so the clock still reads. Worst
# case is TAIL_BUDGET plus the bar, narrower than the pre-0.35 line could get.
TAIL_BUDGET = 48
LABEL_CELLS = 34
MIN_TAIL = 12
_TRUNC = "".join(["."] * 3)


def note_progress(message: str) -> None:
    """Report what the current tool call is doing, live, on the activity line
    (`running run_bash`, 12s in, plus this message).

    Use it for a long call with no other feedback channel — a hook, a
    multi-file transform, a sub-agent's own turn loop. Whitespace is collapsed
    and the text clipped; empty or None clears it. Display-only, and harmless
    when nothing is animating (headless, NX_SPINNER=0, or inside a parallel
    fan-out), so a tool can call it unconditionally without guarding on tty
    state. Never visible to the model and never in the transcript.
    """
    from . import _state as st  # late import: _state pulls in _ansi
    from ._render import stream_tail  # UIUX_SPEC B1: the one tail component

    # Sanitize + clamp through the shared component (it strips ANSI/control
    # bytes and collapses whitespace — a note can come from a tool that quoted
    # arbitrary output). MAX_NOTE is the budget a tool may SET.
    text = stream_tail(str(message), MAX_NOTE, plain=True) if message else ""
    sink = st._PROGRESS_SINK
    if sink is not None:
        try:
            sink(text)
        except Exception:
            pass
    spinner = st._ACTIVE_SPINNER
    if spinner is not None:
        spinner.note = text


class ProgressCounter:
    """How many of a parallel fan-out's calls have landed, for its line.

    One object shared by the dispatching thread (which draws it) and the pool
    workers (which bump it). `done` is a plain int and `+= 1` on it is atomic
    under the GIL, so this stays lock-free — which matters because the render
    thread reads it ~4x/second while up to 8 workers write it.
    """

    __slots__ = ("total", "done")

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0

    def bump(self) -> int:
        """Mark one more call finished; returns the new count."""
        self.done += 1
        return self.done

    def __str__(self) -> str:
        return f"{self.done}/{self.total}"


def clip_line(text: str, width: int) -> str:
    """Clamp a one-line display segment to `width` cells, marking a cut."""
    if width <= 3 or len(text) <= width:
        return text
    return text[:width - 3] + _TRUNC


def fmt_secs(secs: float) -> str:
    """Elapsed seconds for the tail: one decimal under 10, whole under a minute,
    then minutes plus seconds (`2m03`). Fixed shapes hold the line width.

    Clamped at 999m: the stamp comes from time.monotonic(), whose zero point is
    machine boot, so a caller that forgot to stamp (or a test seeding 0.0) would
    otherwise print a five-digit minute count where a duration belongs."""
    if secs < 0:
        secs = 0.0
    if secs >= 999 * 60:
        return "999m"
    if secs < 10:
        return f"{secs:.1f}"
    if secs < 60:
        return f"{int(secs)}"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}"
