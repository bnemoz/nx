"""Human steering + the /tabs live dashboard (0.105.0).

Two halves share this module:

STEERING — while a turn runs (model streaming or an interruptible tool wait),
the interrupt watcher no longer discards printable stdin: printable characters
accumulate in a raw capture buffer and Enter queues the line as a steering
note. The turn loop drains the queue at its step boundaries and injects each
line as a `[Steering note]` user message — mid-task redirection without killing
the turn (the Esc path stays the hard interrupt). Because the watcher runs on
the singleton stdin thread, capture is cheap and race-free: only it ever reads
stdin while armed.

TABS — `/tabs` (or Ctrl-T at the prompt) opens a live dashboard of everything
running in this process: the main turn, background run_bash jobs, and async
sub-agents. Each row shows a status glyph, the instruction/command, and a live
tail. It is a read surface: watching requires no lock, no flag, and no change
to any job. Steering from the dashboard is the same queue the watcher fills.

Both halves degrade to no-ops off a tty (headless, tests): the watcher never
captures there, and /tabs falls back to a one-shot print via /bg's data.
"""
from __future__ import annotations

import sys
import threading
from collections import deque

from . import _state as st
from . import _fullscreen
from ._ansi import (DIM, RESET, C, clip_display, pad_display,
                    strip_ansi_for_display)
from ._tty import IS_WINDOWS

# --------------------------------------------------------------------------
# steering queue
# --------------------------------------------------------------------------

_STEER_LOCK = threading.Lock()
# FIFO of queued steering lines; deque for O(1) popleft from the drain.
_STEER_QUEUE: deque[str] = deque()
# Per-agent queues, keyed by job id (0.106.0): /tabs routes a steered line to
# one running agent; the agent's turn loop drains its own queue at step
# boundaries via its TLS context (which carries job_id). Main-turn steering
# (typed while the model streams) keeps using the module-global deque above.
_AGENT_QUEUES: dict[str, deque[str]] = {}
# Monotonic count of lines ever queued — lets the chatbox hint cheaply.
_STEER_TOTAL = 0
# The watcher's in-progress capture (chars typed, no Enter yet). Kept here so
# a queue drain can't lose the partial when the turn loop reads between turns.
_STEER_PARTIAL = ""


def queue_steering(text: str, agent: str | None = None) -> None:
    """Queue one line of user steering. agent=None targets the main turn;
    otherwise it targets the running sub-agent with that job id."""
    global _STEER_TOTAL
    text = text.strip()
    if not text:
        return
    with _STEER_LOCK:
        if agent:
            _AGENT_QUEUES.setdefault(agent, deque()).append(text)
        else:
            _STEER_QUEUE.append(text)
        _STEER_TOTAL += 1


def queue_agent_steering(agent: str, text: str) -> bool:
    """Steer one running agent. Returns False (and queues nothing) when the
    job id is unknown or the agent is no longer running — the /tabs UI turns
    that into an inline message instead of silently eating the line."""
    from ._subagent import _ASYNC_AGENTS, _ASYNC_LOCK
    with _ASYNC_LOCK:
        rec = _ASYNC_AGENTS.get(agent)
        alive = rec is not None and rec.get("status") == "running"
    if not alive:
        return False
    queue_steering(text, agent=agent)
    return True


def drain_agent_steering(agent: str) -> list[str]:
    """Take all queued steering lines for one agent (oldest first)."""
    with _STEER_LOCK:
        q = _AGENT_QUEUES.get(agent)
        if not q:
            return []
        out = list(q)
        q.clear()
    return out


def agent_steering_pending(agent: str) -> int:
    with _STEER_LOCK:
        q = _AGENT_QUEUES.get(agent)
        return len(q) if q else 0


def set_steer_partial(text: str) -> None:
    """Record the watcher's not-yet-Enter capture (empty = cleared)."""
    global _STEER_PARTIAL
    with _STEER_LOCK:
        _STEER_PARTIAL = text


def steer_partial() -> str:
    with _STEER_LOCK:
        return _STEER_PARTIAL


def drain_steering() -> list[str]:
    """Take all queued steering lines (oldest first). Non-blocking, atomic."""
    with _STEER_LOCK:
        out = list(_STEER_QUEUE)
        _STEER_QUEUE.clear()
    return out


def steer_pending() -> int:
    with _STEER_LOCK:
        return len(_STEER_QUEUE)


STEER_WRAP = ("[Steering note: the user typed this while your turn was "
              "running — fold it into the current task. It is a mid-course "
              "adjustment, not a new task]")


def inject_steering(messages: list, agent: str | None = None) -> bool:
    """Drain steering into `messages` as user turns. True if anything landed.

    Called from _run_model_turn_loop's step boundaries. With agent=None the
    main turn's queue drains; with a job id the named agent's queue drains
    (the turn loop passes its TLS context's job id, so an agent can never
    steal a parent steering line — the pre-0.106.0 race). Each line becomes
    one user message so the model sees the adjustment as a distinct
    instruction with its own position in the history (same shape the
    Esc-redirect path appends in model_turn)."""
    lines = drain_agent_steering(agent) if agent else drain_steering()
    for line in lines:
        messages.append({"role": "user",
                         "content": f"{STEER_WRAP}\n{line}"})
    return bool(lines)


def steer_status_line() -> str:
    """One-line hint for the status bar / prompt, e.g. 'steer: 2 queued'."""
    n = steer_pending()
    partial = steer_partial()
    bits = []
    if n:
        bits.append(f"{n} queued")
    if partial:
        bits.append(f"typing: {partial[-24:]!r}")
    return "steering: " + (" · ".join(bits) if bits else "—")


# --------------------------------------------------------------------------
# /tabs — live dashboard
# --------------------------------------------------------------------------

_DONE_KEEP = 4  # newest finished bg jobs shown below the live ones


def _process_status(job: dict) -> str:
    """The ONE status word for a background job, shared by the live /tabs
    frame, the /tabs fullscreen view and the /bg background-jobs view.

    C1: a hard-cancelled job (the Esc ladder / Ctrl+C path) shows as `killed`,
    never a bare `finished`, so the ladder's outcome stays visible. Lifted out
    of `_surfaces_snapshot` (0.148.6) so `_bg_completions.BackgroundJobsView`
    can report the same word instead of re-deriving it."""
    if job.get("alive", False):
        return "running"
    if job.get("killed"):
        return "killed"
    return str(job.get("status") or "finished")


def _surfaces_snapshot(*, include_bg: bool = True) -> list[dict]:
    """Unified rows for every running surface: main turn + bg jobs + agents.

    Shape mirrors list_bg_jobs()/list_async_agents() plus the synthetic main
    row; kind distinguishes them ("main"|"bg"|"agent").

    `include_bg=False` drops the bg-job rows: since 0.148.6 those belong to the
    dedicated `/bg` view (`_bg_completions.BackgroundJobsView`) and the /tabs
    surface shows live-status rows only (main turn + async agents), so the two
    dashboards no longer duplicate each other. The live-region frame and
    `maybe_close_if_idle` keep the full snapshot."""
    import nx as _m
    rows: list[dict] = [{
        "kind": "main", "id": "main",
        "alive": bool(st._TURN_ACTIVE), "status": "running" if st._TURN_ACTIVE else "idle",
        "instruction": "main conversation",
        "tail": steer_status_line(),
        "obj": None,
    }]
    if include_bg:
        try:
            # mine_only default: the shared bg-logs dir holds every session's
            # jobs on the machine; /tabs is a per-session dashboard.
            for job in _m.list_bg_jobs():
                alive = job.get("alive", False)
                rows.append({"instruction": job.get("command") or job.get("instruction") or "",
                             "status": _process_status(job),
                             "tail": job.get("tail") or "", **job,
                             "kind": "bg", "id": f"bg-{job['pid']}", "alive": alive})
        except Exception:
            pass
    try:
        for ag in _m.list_async_agents():
            # EVENTSURFACE_SPEC §2 agent-row tails: a RUNNING agent has no
            # answer yet, which is why its row used to render an empty tail.
            # Fall back to the event sink's live ring (6 display lines), and
            # once the worker's finally drops the sink, `sink_tail` transparently
            # picks up the retained copy — so the row keeps its last activity
            # after the agent finishes rather than blanking.
            # Finished agents with an answer keep the answer tail unchanged
            # (dense: the report head the user actually reads), so this changes
            # only what was previously empty. Read-only: tails are pulled from
            # the sink here; no dashboard/live-mode state is involved.
            tail = ag.get("tail") or ""
            if not tail and ag.get("status") == "running":
                from ._subagent_tls import sink_tail
                tail = "\n".join(sink_tail(ag.get("id"), 6))
            rows.append({"instruction": "", **ag, "tail": tail,
                         "kind": "agent", "id": ag["id"],
                         "alive": ag.get("alive", False)})
    except Exception:
        pass
    # finished surfaces accumulate across the session; keep the newest few so
    # the dashboard stays a live view (a job's full history, finished rows
    # included, is the /bg view's own windowed list).
    done = [r for r in rows if not r.get("alive") and r["kind"] != "main"]
    rows = [r for r in rows if r.get("alive") or r["kind"] == "main"] + done[:_DONE_KEEP]
    return rows


def _glyph(alive: bool, status: str) -> str:
    if alive:
        return f"{C.SUCCESS}●{RESET}"
    if status in ("error", "partial", "killed"):
        return f"{C.ERROR}✗{RESET}"
    if status == "done":
        return f"{C.ACCENT}✔{RESET}"
    return f"{DIM}○{RESET}"


def _tail_lines(tail: str, n: int = 6) -> list[str]:
    lines = [ln for ln in (tail or "").splitlines() if ln.strip()]
    return lines[-n:]


def run_tabs_dashboard() -> None:
    """Interactive /tabs view (tty only): raw-key loop.

    Keys: ↑/↓ (or j/k) select a row · Enter opens the steer prompt on a
    selected live agent · r repaints · q/Esc/Ctrl-C back to the prompt.
    Inline (no alternate screen), same repaint discipline as the chatbox:
    clear exactly the rows drawn last frame. A steer prompt is a raw-mode
    line editor (printable + backspace, Enter commits) — arrows/paste are
    not supported here by design; this is a one-line nudge, not a chat."""
    fd = sys.stdin.fileno()
    from ._tty import read_byte, wait_ready
    from ._ansi import raw_mode

    drawn = 0
    sel = 0        # selected row index into the last snapshot
    rows: list[dict] = []
    notice = ""    # transient inline feedback (steer accepted/refused)

    def render(first: bool = False, extra: list[str] | None = None) -> None:
        nonlocal drawn, rows, sel
        rows = _surfaces_snapshot()
        sel = min(sel, max(0, len(rows) - 1))
        width = _term_cols()
        lines = [f"{C.ACCENT}tabs{RESET}{DIM} — live surfaces · ↑/↓ select · "
                 f"Enter steer agent · r refresh · q back{RESET}"]
        for i, r in enumerate(rows):
            status = r.get("status") or "—"
            marker = "›" if i == sel else " "
            head = f"{marker} {_glyph(r.get('alive', False), status)} {r['kind']:<5} " \
                   f"{r['id']:<18} {status:<8}"
            lines.append(head)
            instr = " ".join(str(r.get("instruction", "")).split())
            if instr:
                lines.append(f"{DIM}    {instr[:width - 6]}{RESET}")
            for tl in _tail_lines(r.get("tail") or ""):
                lines.append(f"{DIM}    │ {tl[:width - 8]}{RESET}")
            pend = agent_steering_pending(r["id"]) if r["kind"] == "agent" else 0
            if pend:
                lines.append(f"{C.WARNING}    ▸ {pend} steering line"
                             f"{'s' if pend != 1 else ''} queued{RESET}")
        if notice:
            lines.append(f"{C.WARNING}{notice}{RESET}")
        lines.extend(extra or [])
        out = ""
        if not first:
            out = f"\x1b[{drawn}A\r" if drawn else ""
            out += "\x1b[J"
        out += "\n".join(lines) + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()
        drawn = len(lines)

    def steer_prompt(agent_id: str) -> None:
        """Raw-mode one-line editor; Enter queues the note for the agent."""
        nonlocal notice
        buf = ""
        cursor = f"{DIM}steer {agent_id} ›{RESET} "
        while True:
            frame = f"\r\x1b[K{cursor}{buf}"
            sys.stdout.write(frame)
            sys.stdout.flush()
            if not wait_ready(fd, 0.5):
                continue
            c = read_byte(fd)
            if c in (b"\r", b"\n"):
                if buf.strip():
                    ok = queue_agent_steering(agent_id, buf)
                    notice = (f"  ▸ steered {agent_id}: {buf.strip()[:50]}"
                              if ok else
                              f"  ✗ {agent_id} is not running — line dropped")
                else:
                    notice = ""
                return
            if c in (b"\x03", b"\x04", b"\x1b", b"q"):
                notice = ""
                return
            if c in (b"\x7f", b"\x08"):
                buf = buf[:-1]
                continue
            try:
                ch = c.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if ch >= " ":
                buf += ch

    with raw_mode(fd):
        render(first=True)
        try:
            while True:
                if not wait_ready(fd, 0.5):
                    render()
                    continue
                c = read_byte(fd)
                if c in (b"q", b"\x03", b"\x04"):
                    break
                if c == b"\x1b":
                    # bare Esc quits; arrow sequences move the selection
                    if not wait_ready(fd, 0.05):
                        break
                    seq = read_byte(fd)
                    if seq == b"[":
                        arrow = read_byte(fd)
                        if arrow == b"A":
                            sel = max(0, sel - 1)
                        elif arrow == b"B":
                            sel = min(max(0, len(rows) - 1), sel + 1)
                    render()
                    continue
                if c in (b"j",):
                    sel = min(max(0, len(rows) - 1), sel + 1)
                    render()
                    continue
                if c in (b"k",):
                    sel = max(0, sel - 1)
                    render()
                    continue
                if c == b"\r" or c == b"\n":
                    if rows and sel < len(rows) and rows[sel]["kind"] == "agent":
                        if rows[sel].get("alive"):
                            steer_prompt(rows[sel]["id"])
                        else:
                            notice = f"  ✗ {rows[sel]['id']} is not running"
                    elif rows and sel < len(rows):
                        notice = "  ✗ only sub-agents accept steering (select an agent row)"
                    render()
                    continue
                # r or any key: fall through to repaint
                notice = ""
                render()
        finally:
            sys.stdout.write(f"\x1b[{drawn}A\x1b[J" if drawn else "")
            sys.stdout.flush()


def _term_cols() -> int:
    import shutil
    from ._layout import TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS, TERM_MIN_COLS
    return max(TERM_MIN_COLS,
               shutil.get_terminal_size(
                   (TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS)).columns)


# --------------------------------------------------------------------------
# live region (EVENTSURFACE_SPEC §2/§3, Phase 3; SPEC_LIVEENGINE.md 0.128.0)
# --------------------------------------------------------------------------
# A bounded overlay painted BELOW the current render line while a turn runs,
# redrawn at ~4 fps from the sink tails + _surfaces_snapshot(). NOT an alternate
# screen — the main turn's output continues above it.
#
# 0.128.0: the paint is a rich Live on the SHARED console, not hand-rolled
# cursor arithmetic (\x1b[{n}A\r\x1b[J). The old arithmetic assumed the region was
# the last thing written; every mid-turn print below it (stats footer, tool
# blocks, spinner) invalidated the anchor, so frames stacked and tore. rich
# instead owns line discipline: mid-turn `console.print` lines are composed
# ABOVE the frame, and `transient=True` erases the frame on stop().

_WRITE_LOCK = threading.RLock()
# EVENTSURFACE_SPEC §2 Phase 4 (v1-required): this lock is now also taken by
# every raw console write in _turn/_stream_parse/_dispatch, so a 4 fps repaint
# can never interleave with (tear) a parent-turn line while the dashboard is
# open. Exported through write_lock() so writers import one cheap name instead
# of reaching for the private global.


def write_lock() -> "threading.RLock":
    """The console write lock: take it around ANY raw terminal write you want
    serialized against the /tabs live repaint. An RLock, so nested sites in one
    thread (e.g. a grouped message that wraps its own multi-print block and also
    uses the per-write helper) re-enter instead of deadlocking.

    Return value, not a contextmanager generator: `with write_lock():` acquires
    the lock itself, so the helper costs one plain call + one acquire and needs
    no contextlib machinery on the per-chunk streaming path."""
    return _WRITE_LOCK

_DASH_LOCK = threading.Lock()
_DASH_OPEN = False
_DASH_DISABLED = False   # a repaint crash disables live mode for the session
# The region's rich Live (created on first open, dropped on close).
# live_region() is the accessor tests use; the raw global stays private.
_DASH_LIVE = None

LIVE_FPS_INTERVAL = 0.25  # 4 fps

# The last painted frame as `(live, plain lines)`, for repaint change
# suppression (0.140.0). rich Live rewrites the whole frame per refresh, so an
# unchanged frame is pure terminal traffic + flicker; repaint_live compares
# against this and skips update/refresh entirely. The Live INSTANCE is part of
# the key, so a reopened region (or a fresh Live) always paints once even if the
# frame text is unchanged. Only the terminal's owner thread writes it
# (repaint_live's contract).
_LAST_FRAME: "tuple[object, tuple[str, ...]] | None" = None


def _tabs_height() -> int:
    """NX_TABS_HEIGHT: the live region's max rows. Non-int/absent -> 8,
    clamped to 1..20."""
    from ._env import _env_int
    return max(1, min(20, _env_int("NX_TABS_HEIGHT", 8)))


def _frame_height() -> int:
    """The render-time frame height: NX_TABS_HEIGHT clamped against the
    terminal's ROWS too — min(NX_TABS_HEIGHT, rows - 3). A tall frame on a
    short terminal pushes the prompt (and the next committed line) off the
    screen, leaving the user typing blind; the 3-row reserve is the prompt, the
    line being composed, and one row of slack. Never below 1, so an absurdly
    small terminal still renders one row. `_tabs_height()` keeps its own 1..20
    contract (its callers and tests read the env semantics, not the screen)."""
    import shutil
    from ._layout import TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS
    # os.terminal_size names its fields columns/lines (there is no `.rows`).
    lines = shutil.get_terminal_size(
        (TERM_FALLBACK_COLS, TERM_FALLBACK_ROWS)).lines
    return max(1, min(_tabs_height(), lines - 3))


def _vt_supported() -> bool:
    """Does this terminal handle the ANSI cursor/erase idioms the live region
    needs? POSIX terminals do when they are real ttys; the Windows console
    only with VT mode enabled (reuse of the _tty detection: nx enables
    ENABLE_VIRTUAL_TERMINAL_INPUT for its editor, so a console the editor
    runs in reports VT-capable; anything else refuses live mode)."""
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
    except (AttributeError, ValueError, OSError):
        return False
    if IS_WINDOWS:
        # stdout VT output enabled? (the editor enables VT *input* via _tty;
        # a console that renders our cursor/erase idioms needs the output bit)
        try:
            import ctypes
            mode = ctypes.wintypes.DWORD()
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            h = k32.GetStdHandle(-11)
            if not k32.GetConsoleMode(h, ctypes.byref(mode)):
                return False
            return bool(mode.value & 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            return False
    return True


def _agent_styles(rows: list[dict]) -> list[bool]:
    """Per-row 'run as done' decision over the rows, enforcing the
    tear-avoidance rule (EVENTSURFACE_SPEC §2): a contiguous block of agent
    rows — in _ASYNC_AGENTS dict insertion order = spawn order, which is the
    order _surfaces_snapshot() already emits (no separate sort key) — only
    flips to done styling when the WHOLE block is terminal, so a finishing
    agent cannot visually reorder against still-running siblings."""
    n = len(rows)
    out = [False] * n
    i = 0
    while i < n:
        if rows[i]["kind"] != "agent":
            out[i] = True
            i += 1
            continue
        j = i
        while j < n and rows[j]["kind"] == "agent":
            j += 1
        block_terminal = all(not rows[k].get("alive") for k in range(i, j))
        for k in range(i, j):
            out[k] = block_terminal
        i = j
    return out


def _live_frame() -> list[str]:
    """The ≤ NX_TABS_HEIGHT lines of the live region (no ANSI erase prefix).
    Agent-row tails come from the event sink (Phase 2's sink_tail)."""
    from ._subagent_tls import sink_tail
    rows = _surfaces_snapshot()
    styles = _agent_styles(rows)
    height = _frame_height()
    lines: list[str] = [f"{C.ACCENT}tabs{RESET}{DIM} — live · ctrl-t closes{RESET}"]
    width = _term_cols()
    for i, r in enumerate(rows):
        if len(lines) >= height:
            break
        status = r.get("status") or "—"
        done_style = styles[i] if r["kind"] == "agent" else True
        mark = _glyph(r.get("alive", False), status)
        if not r.get("alive", False) and done_style and status == "done":
            mark = f"{DIM}○{RESET}"
        head = f" {mark} {r['kind']:<5} {r['id']:<18} {status:<8}"
        lines.append(clip_display(head, width))
        instr = " ".join(str(r.get("instruction", "")).split())
        if instr and len(lines) < height:
            lines.append(f"{DIM}    {clip_display(instr, width - 6)}{RESET}")
        if len(lines) < height and r["kind"] == "agent":
            for tl in _tail_lines("\n".join(sink_tail(r["id"], 2)), 2):
                if len(lines) >= height:
                    break
                lines.append(f"{DIM}    │ {clip_display(tl, width - 8)}{RESET}")
    lines = lines[:height]
    # right status rail (SPEC_PANEL.md): compose `<left pad>│ <cell>` per line
    # when NX_PANEL/`/panel` is on. Not enabled -> the untouched lines (the
    # panel-off frame is byte-identical to the pre-panel build), or — when the
    # user asked for the rail and the terminal is merely too narrow for it —
    # the same cells as ONE bottom status line (rail-off, compact-on; see
    # _panel.degraded_line). Both exits below run only when the rail is WANTED
    # (degraded_line re-checks), so panel-off stays byte-identical.
    from ._panel import degraded_line, panel_cells, panel_enabled, panel_width
    if not panel_enabled():
        line = degraded_line(rows, width)
        if not line:
            return lines
        # the compact line is a frame ROW like any other: it must live inside
        # the same height clamp, so drop the last content row instead of
        # pushing the frame one row past `rows - 3`.
        return lines[:max(1, height - 1)] + [line]
    pw = panel_width()
    # Rail geometry in DISPLAY CELLS, with the terminal's LAST column reserved
    # (lw + 2 + pw <= width - 1): writing the last cell auto-wraps on most
    # terminals, which adds a row the Live's erase count never accounted for.
    # Every slice is the shared ANSI-aware, wcwidth-accurate `_ansi` trio, so a
    # wide char or an SGR sequence can neither shift the `│` nor wrap the row.
    lw = max(1, width - pw - 3)
    cells = panel_cells(pw, rows)
    out: list[str] = []
    for i, ln in enumerate(lines):
        cell = cells[i] if i < len(cells) else ""
        out.append(f"{pad_display(ln, lw)}{DIM}│{RESET} "
                   f"{clip_display(cell, pw)}{RESET}")
    return out


def live_region():
    """The region's rich Live, or None while the region is closed."""
    return _DASH_LIVE


def _fullscreen_owns_screen() -> bool:
    """Is a Phase D fullscreen view painting the alternate screen right now?

    Lazy import: `_fullscreen` imports this module's view classes at call time,
    so a module-level import here would be a cycle. Failure to import (a
    stripped build) means "no view", which is the pre-Phase-D behaviour.
    """
    try:
        from ._fullscreen import owns_screen
        return owns_screen()
    except Exception:
        return False


def _region_renderable(lines: "list[str] | None" = None):
    """The Live's renderable: the frame lines joined and parsed to a `Text`.

    `lines` lets repaint_live reuse the frame it already composed for the
    change check instead of recomposing (the frame is expensive enough to
    matter at 4 fps: it walks the surfaces snapshot + the rail's cells).

    MUST NOT be a plain `str`: `Live.update(str)` runs it through
    `Console.render_str`, which parses rich MARKUP — and a frame row routinely
    contains square brackets (a tool instruction like `[Steering note]`, a
    clipped `[/etc/hosts]` tail), which raises MarkupError and would trip the
    crash handler, permanently disabling the region for the session. Passing a
    `Text` bypasses render_str entirely.

    `Text.from_ansi` is the safe form (SPEC_LIVEENGINE.md §2): it keeps the
    frame's own SGR. (Where the console has no color system at all the Live
    does not render its frame anyway, so the style pruning there is
    unobservable.) `no_wrap=True` because the frame is already clipped to the
    terminal width — a wrap would add a row and desync the Live's erase count."""
    from rich.text import Text
    lines = _live_frame() if lines is None else lines
    return Text.from_ansi("\n".join(lines), no_wrap=True)


def repaint_live(force: bool = False) -> None:
    """One dashboard repaint (the 4 fps tick). No-op when closed or disabled;
    a crash inside is caught and disables live mode for the session
    (belt-and-braces vs termios corruption).

    The Live is built with auto_refresh=False, so nothing repaints from an
    external thread; update+refresh run here, under write_lock(), triggered by
    the terminal's owner hooks (see repaint_live_tick)."""
    global _DASH_DISABLED, _LAST_FRAME
    if _fullscreen_owns_screen():
        return
    with _DASH_LOCK:
        if not _DASH_OPEN or _DASH_DISABLED:
            return
        live = _DASH_LIVE
    if live is None or not _vt_supported():
        close_dashboard()
        return
    try:
        lines = _live_frame()
        # Frame-level change suppression (0.140.0): rich Live rewrites the WHOLE
        # frame per refresh, so an unchanged frame is pure terminal traffic. The
        # comparison is on the PLAIN lines (ANSI stripped) — a repaint that
        # differs only in SGR bytes is not a change the user can see, and a
        # color-only difference must not buy a full redraw. `force` (the
        # open-time first paint) always repaints.
        plain = tuple(strip_ansi_for_display(ln) for ln in lines)
        cached = _LAST_FRAME
        if not force and cached is not None and cached[0] is live \
                and cached[1] == plain:
            return
        renderable = _region_renderable(lines)
        with _WRITE_LOCK:
            live.update(renderable)
            live.refresh()
        _LAST_FRAME = (live, plain)
    except Exception:
        _DASH_DISABLED = True
        try:
            close_dashboard()
        except Exception:
            pass


def repaint_live_tick() -> None:
    """Ownership-driven repaint hook (EVENTSURFACE_SPEC §2 'Editor
    ownership'). Turn-loop-owned terminal: called from the armed interrupt
    watcher's ~100ms poll tick (nx/_interrupt.py). Chatbox-owned: from the
    chatbox read loop's wait_ready timeout (nx/_chatbox.py). NEVER from an
    external thread — the caller is always the terminal's single stdin
    reader/owner."""
    repaint_live()


def _make_region_live():
    """Construct the region's Live (lazy, once per open).

    `Live.__init__` owns the cursor: it WRITES the first frame with surrounding
    control bytes (cursor hide, a preceding blank line, cursor moves). That
    write is the open-time paint the old code did by hand, so the Live is built
    (not merely update()d) under write_lock() in open_dashboard.

    redirect_stdout/stderr MUST stay False: rich's FileProxy AnsiDecoder DROPS
    non-SGR control bytes, which is exactly what nx's own pre-formatted writes
    carry. auto_refresh=False keeps refresh on the owner hooks only.
    transient=True gives the close semantics _erase_live used to implement.
    vertical_overflow="visible" so a taller-than-terminal frame is never
    cropped (the frame is already clamped to NX_TABS_HEIGHT anyway)."""
    from rich.live import Live
    from ._rich import shared_console
    return Live(_region_renderable(),
                console=shared_console(),
                auto_refresh=False,
                transient=True,
                refresh_per_second=4,      # LIVE_FPS_INTERVAL = 0.25s = 4 fps
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="visible")


def open_dashboard() -> bool:
    """Open the live region. False when refused (off-tty / VT missing /
    disabled for the session). Idempotent: a second open is a no-op success."""
    global _DASH_OPEN, _DASH_LIVE
    with _WRITE_LOCK:
        with _DASH_LOCK:
            if _DASH_DISABLED or not _vt_supported():
                return False
            if _DASH_OPEN:
                return True
            _DASH_LIVE = _make_region_live()
            try:
                _DASH_LIVE.start(refresh=True)
            except Exception:
                _DASH_LIVE = None
                raise
            _DASH_OPEN = True
    return True


def close_dashboard() -> None:
    """Erase the live region and mark it closed. Idempotent; safe anywhere.

    transient=True makes the Live erase its own frame on stop(), which is what
    the old _erase_live() did with cursor arithmetic — no _DASH_DRAWN counter to
    keep in sync any more."""
    global _DASH_OPEN, _DASH_LIVE, _LAST_FRAME
    live = None
    with _DASH_LOCK:
        was_open = _DASH_OPEN
        _DASH_OPEN = False
        _LAST_FRAME = None   # a reopened region must paint at least once
        live, _DASH_LIVE = _DASH_LIVE, None
    if not was_open and live is None:
        return
    if live is None:
        return
    try:
        with _WRITE_LOCK:
            live.stop()
    except Exception:
        pass


def dashboard_open() -> bool:
    with _DASH_LOCK:
        return _DASH_OPEN


def dashboard_disabled() -> bool:
    return _DASH_DISABLED


def toggle_dashboard() -> bool:
    """Ctrl-T: open the live region if closed, close it if open. Returns
    whether the dashboard is now open."""
    if dashboard_open():
        close_dashboard()
        return False
    return open_dashboard()


def maybe_close_if_idle() -> None:
    """Termination rule: when the agent set empties (no live bg/agent rows),
    the dashboard closes itself — nothing left to show."""
    if not dashboard_open():
        return
    rows = _surfaces_snapshot()
    if not any(r["alive"] for r in rows if r["kind"] != "main"):
        close_dashboard()


def tabs_print_fallback() -> None:
    """Non-tty / one-shot view: same data, plain print."""
    rows = _surfaces_snapshot()
    for r in rows:
        mark = "●" if r["alive"] else "○"
        print(f" {mark} [{r['kind']}] {r['id']}: {r.get('status', '')} — "
              f"{' '.join(str(r.get('instruction', '')).split())[:100]}")
        for tl in _tail_lines(r.get("tail") or "", 2):
            print(f"     │ {tl[:100]}")


# --------------------------------------------------------------------------
# /tabs — alternate-screen view (UIUX_SPEC Phase D)
# --------------------------------------------------------------------------
# The pre-0.137.0 inline raw-key dashboard (`run_tabs_dashboard`) is kept below
# for callers that enter it directly; the /tabs COMMAND now goes through the
# fullscreen shell (`_fullscreen.open_view(tabs_view(), fallback=...)`), which
# owns the alternate screen, the resize repaint and the `?` overlay. The data,
# the steering queue and the inline fallback are unchanged.

class TabsView(_fullscreen.View):
    """The Phase D /tabs view: one row per LIVE surface (the main turn and the
    async sub-agents), j/k focus, Enter steers the focused agent row through
    the SAME `_surfaces` steering queue the mid-turn watcher fills.

    Background jobs are NOT here (0.148.6): they are process history rather
    than a live surface, and they have their own view — `/bg`
    (`_bg_completions.BackgroundJobsView`, Ctrl+O). The live region below the
    prompt still paints them; only this alternate-screen dashboard split.

    Renders through `_render.py` (`panel`, `role`, `hint_line`) — read-only:
    steering a row is the queue the pre-existing dashboard already used, and
    the view touches no job state itself.
    """

    surface = "tabs"
    title = "tabs"

    def __init__(self) -> None:
        super().__init__()
        self.sel = 0
        self.rows: list[dict] = []
        self.job_rows: list[dict] = []   # not painted here; /bg owns these rows
        self.notice = ""

    def refresh(self) -> None:
        """Live-status rows only: main turn + async agents. `job_rows` keeps
        the bg rows for the notice's "see /bg" hint; the live-region frame
        reads the unfiltered snapshot and still shows them."""
        self.rows = _surfaces_snapshot(include_bg=False)
        self.job_rows = [r for r in _surfaces_snapshot() if r["kind"] == "bg"]
        self.sel = min(self.sel, max(0, len(self.rows) - 1))

    def _row_lines(self, f: dict, selected: bool, width: int) -> list[str]:
        from ._render import role
        status = f.get("status") or "—"
        marker = "›" if selected else " "
        tier = f.get("tier")
        head = (f"{role('ACCENT')}{marker}{RESET} "
                f"{_glyph(f.get('alive', False), status)} "
                f"{f['kind']:<5} {f['id']:<18} "
                f"{role('MUTED')}{status:<8}{RESET}")
        if tier:
            head += f"{role('MUTED')} tier: {tier:<5}{RESET}"
        out = [clip_display(head, width - 4)]
        instr = " ".join(str(f.get("instruction", "")).split())
        if instr:
            out.append(f"{role('MUTED')}    "
                       f"{clip_display(instr, width - 10)}{RESET}")
        for tl in _tail_lines(f.get("tail") or ""):
            out.append(f"{role('MUTED')}    │ "
                       f"{clip_display(tl, width - 12)}{RESET}")
        pend = (agent_steering_pending(f["id"]) if f.get("kind") == "agent"
                else 0)
        if pend:
            out.append(f"{role('WARNING')}    ▸ {pend} steering line"
                       f"{'s' if pend != 1 else ''} queued{RESET}")
        return out

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        self.refresh()
        lines = [f"{role('ACCENT')}tabs{RESET}{role('MUTED')} — "
                 f"{hint_line('tabs')}{RESET}"]
        # Allocate body space per row instead of blind-truncating at the end:
        # finished rows sort after the live ones (the snapshot's live/done
        # split), so a plain lines[:height] clips exactly the row the user is
        # looking for. Every row keeps at least its head line; deep tails are
        # what give way, and overflow says so.
        chrome = 1 + 2 + (1 if self.notice else 0)  # title + panel top/bottom + notice
        body: list[str] = []
        budget = max(height - chrome, len(self.rows))  # each row keeps >= 1 line
        row_lines = [self._row_lines(f, i == self.sel, width)
                     for i, f in enumerate(self.rows)]
        if sum(len(r) for r in row_lines) <= budget:
            for r in row_lines:
                body.extend(r)
        else:
            # Progressive trim: rows keep head (+instruction) while budget
            # allows; tails are dropped first, rows beyond keep only heads.
            for i, r in enumerate(row_lines):
                per = max(1, (budget - len(body)) // (len(row_lines) - i))
                body.extend(r[:per])
            body.append(f"{role('MUTED')}  … some tails hidden "
                        f"(terminal too short){RESET}")
        extra = (f" · {len(self.job_rows)} bg job(s) in /bg"
                 if self.notice and self.job_rows else "")
        lines.extend(panel(
            f"tabs · live surfaces · {len(self.rows)} row(s){extra}",
            body, footer="q/esc back to chat"))
        if self.notice:
            lines.append(f"{role('WARNING')}{self.notice[:width]}{RESET}")
        return lines[:height]

    def _steer(self, shell) -> None:
        row = self.rows[self.sel] if self.sel < len(self.rows) else None
        if row is None or row.get("kind") != "agent":
            self.notice = "only sub-agent rows accept steering (select one)"
            return
        if not row.get("alive"):
            self.notice = f"{row['id']} is not running"
            return
        text = shell.ask(f"steer {row['id']} ›")
        if not text:
            self.notice = "steer cancelled"
            return
        ok = queue_agent_steering(row["id"], text)
        self.notice = (f"steered {row['id']}: {text[:50]}" if ok
                       else f"{row['id']} is not running — line dropped")

    def on_key(self, key: str, shell) -> str:
        self.notice = ""
        if key == "j/k":   # the hint line's "j/k move" entry
            return "repaint"
        if key in ("j", "down"):
            self.sel = min(max(0, len(self.rows) - 1), self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            self.sel = max(0, self.sel - 1)
            return "repaint"
        if key == "enter":
            self._steer(shell)
            return "repaint"
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def tabs_view() -> "TabsView":
    """The /tabs view object the fullscreen shell runs (`_fullscreen.tabs()`
    is the thin alias the command path uses)."""
    return TabsView()
