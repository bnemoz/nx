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
from ._ansi import DIM, RESET, C

# --------------------------------------------------------------------------
# steering queue
# --------------------------------------------------------------------------

_STEER_LOCK = threading.Lock()
# FIFO of queued steering lines; deque for O(1) popleft from the drain.
_STEER_QUEUE: deque[str] = deque()
# Monotonic count of lines ever queued — lets the chatbox hint cheaply.
_STEER_TOTAL = 0
# The watcher's in-progress capture (chars typed, no Enter yet). Kept here so
# a queue drain can't lose the partial when the turn loop reads between turns.
_STEER_PARTIAL = ""


def queue_steering(text: str) -> None:
    """Queue one line of user steering for the running/next turn."""
    global _STEER_TOTAL
    text = text.strip()
    if not text:
        return
    with _STEER_LOCK:
        _STEER_QUEUE.append(text)
        _STEER_TOTAL += 1


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


def inject_steering(messages: list) -> bool:
    """Drain the queue into `messages` as user turns. True if anything landed.

    Called from _run_model_turn_loop's step boundaries. Each line becomes one
    user message so the model sees the adjustment as a distinct instruction
    with its own position in the history (same shape the Esc-redirect path
    appends in model_turn)."""
    lines = drain_steering()
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


def _surfaces_snapshot() -> list[dict]:
    """Unified rows for every running surface: main turn + bg jobs + agents.

    Shape mirrors list_bg_jobs()/list_async_agents() plus the synthetic main
    row; kind distinguishes them ("main"|"bg"|"agent")."""
    import nx as _m
    rows: list[dict] = [{
        "kind": "main", "id": "main",
        "alive": bool(st._TURN_ACTIVE), "status": "running" if st._TURN_ACTIVE else "idle",
        "instruction": "main conversation",
        "tail": steer_status_line(),
        "obj": None,
    }]
    try:
        for job in _m.list_bg_jobs():
            alive = job.get("alive", False)
            rows.append({"instruction": job.get("command") or job.get("instruction") or "",
                         "status": "running" if alive else "finished",
                         "tail": job.get("tail") or "", **job,
                         "kind": "bg", "id": f"bg-{job['pid']}", "alive": alive})
    except Exception:
        pass
    try:
        for ag in _m.list_async_agents():
            rows.append({"instruction": "", "tail": "", **ag,
                         "kind": "agent", "id": ag["id"],
                         "alive": ag.get("alive", False)})
    except Exception:
        pass
    # finished bg jobs accumulate across the session; keep the newest few so
    # the dashboard stays a live view (full history remains under /bg).
    done = [r for r in rows if not r.get("alive") and r["kind"] != "main"]
    rows = [r for r in rows if r.get("alive") or r["kind"] == "main"] + done[:_DONE_KEEP]
    return rows


def _glyph(alive: bool, status: str) -> str:
    if alive:
        return f"{C.SUCCESS}●{RESET}"
    if status in ("error", "partial"):
        return f"{C.ERROR}✗{RESET}"
    if status == "done":
        return f"{C.ACCENT}✔{RESET}"
    return f"{DIM}○{RESET}"


def _tail_lines(tail: str, n: int = 6) -> list[str]:
    lines = [ln for ln in (tail or "").splitlines() if ln.strip()]
    return lines[-n:]


def run_tabs_dashboard() -> None:
    """Interactive /tabs view (tty only): raw-key loop, r repaints, q quits.

    Inline (no alternate screen), same repaint discipline as the chatbox:
    clear exactly the rows drawn last frame. Ctrl-C/EOF propagate to the REPL
    handler, which just returns to the prompt."""
    fd = sys.stdin.fileno()
    from ._tty import read_byte, wait_ready
    from ._ansi import raw_mode

    drawn = 0

    def render(first: bool = False) -> None:
        nonlocal drawn
        rows = _surfaces_snapshot()
        width = _term_cols()
        lines = [f"{C.ACCENT}tabs{RESET}{DIM} — live surfaces · r refresh · q back{RESET}"]
        for r in rows:
            status = r.get("status") or "—"
            head = f"  {_glyph(r.get('alive', False), status)} {r['kind']:<5} {r['id']:<18} " \
                   f"{status:<8}"
            head += _pad(head, width)
            lines.append(head)
            instr = " ".join(str(r.get("instruction", "")).split())
            if instr:
                lines.append(f"{DIM}    {instr[:width - 6]}{RESET}")
            for tl in _tail_lines(r.get("tail") or ""):
                lines.append(f"{DIM}    │ {tl[:width - 8]}{RESET}")
        # clear previous frame, then draw
        out = ""
        if not first:
            out = f"\x1b[{drawn}A\r" if drawn else ""
            out += "\x1b[J"
        out += "\n".join(lines) + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()
        drawn = len(lines)

    with raw_mode(fd):
        render(first=True)
        try:
            while True:
                if not wait_ready(fd, 0.5):
                    render()
                    continue
                c = read_byte(fd)
                if c in (b"q", b"\x03", b"\x04", b"\x1b"):
                    break
                # r or any key: fall through to repaint
                render()
        finally:
            sys.stdout.write(f"\x1b[{drawn}A\x1b[J" if drawn else "")
            sys.stdout.flush()


def _term_cols() -> int:
    import shutil
    return max(40, shutil.get_terminal_size((80, 24)).columns)


def _pad(s: str, width: int) -> str:
    return ""  # rows are left-aligned; trailing pad unneeded on tty


def tabs_print_fallback() -> None:
    """Non-tty / one-shot view: same data, plain print."""
    rows = _surfaces_snapshot()
    for r in rows:
        mark = "●" if r["alive"] else "○"
        print(f" {mark} [{r['kind']}] {r['id']}: {r.get('status', '')} — "
              f"{' '.join(str(r.get('instruction', '')).split())[:100]}")
        for tl in _tail_lines(r.get("tail") or "", 2):
            print(f"     │ {tl[:100]}")
