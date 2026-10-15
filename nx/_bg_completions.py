"""Background-job completion scanning: the one-time [Runtime note] for tracked
bg jobs that finished since the model last looked.

Split out of `_dispatch.py` (which keeps run_tool — the first-tool-of-turn
hook that calls the sweep). Shared by the wake path (`_autowake.wake_due`),
whose text-only turns otherwise never run it."""
from __future__ import annotations

import os
import threading
import time


from . import _state as st
from . import _notices
from . import _fullscreen

# The scan runs from both the model turn (run_tool's first-tool hook) and the
# wake poller thread; the seen-set is read-modify-rewritten, so both sides
# serialize on this (also held by _bash_jobs._mark_bg_notice_seen).
_BG_SCAN_LOCK = threading.Lock()


# --- per-task job scope (0.163.0, AUTOWAKE_SPEC Phase E prerequisite 2) -------
# The sweep attributes a job to a session by `owner_pid != os.getpid()`
# — one bucket per PROCESS. A `nx run --parallel N` process runs N tasks in N
# threads that share that pid, so every task saw every other task's jobs: task
# A's turn announced task B's exit line, and a chained task re-swept a job it
# had already consumed. The batch task driver installs a SCOPE (the pids this
# task launched) on its own thread; the sweep intersects with it when one is
# installed, exactly like `jobs=` threads one list_bg_jobs() result through the
# whole sweep instead of re-scanning. `None` (the interactive REPL, a sub-agent
# loop, the poller) means "no narrowing" — the pre-0.163 behavior.
_BG_SCOPE_TLS = threading.local()


def bg_scope_install() -> set:
    """Install an empty job scope on the calling thread and return it (the
    batch task driver calls this once per task; jobs launched on that thread
    join it via `bg_scope_add`)."""
    pids: set[str] = set()
    _BG_SCOPE_TLS.pids = pids
    return pids


def bg_scope() -> "set[str] | None":
    """This thread's job scope, or None when none is installed."""
    return getattr(_BG_SCOPE_TLS, "pids", None)


def bg_scope_add(pid: "int | str | None") -> None:
    """Record a job this thread's task launched. No-op outside a scope, so the
    interactive path pays one attribute lookup and changes nothing."""
    if pid is None or pid == "":
        return
    pids = getattr(_BG_SCOPE_TLS, "pids", None)
    if pids is not None:
        pids.add(str(pid))


def bg_scope_clear() -> "set[str] | None":
    """Remove this thread's scope and return it (the driver reaps it at task
    end, so the set survives its own owner by one step)."""
    pids = getattr(_BG_SCOPE_TLS, "pids", None)
    _BG_SCOPE_TLS.pids = None
    return pids


def bg_scope_adopt(pids: "set[str] | None") -> "set[str] | None":
    """Install an EXISTING scope on the calling thread, returning what it
    replaced (0.163.1, review fix 2).

    `bg_scope_install` makes a NEW set, which is right for the owner (the task
    driver) and wrong for anyone who has to share one: an async sub-agent runs
    its whole turn loop on its own worker thread, where this thread-local is
    None, so a `run_bash` it backgrounded called `bg_scope_add` into nothing —
    the job escaped the parent task's sweep narrowing AND its task-end reap
    (an orphaned live process and a completion note that reached nobody).
    The spawn site hands the worker the PARENT'S set object (not a copy), so
    a pid added on the worker thread is immediately visible to the parent
    task's sweeps and to `_reap_task_bg_jobs`.

    Pair with `bg_scope_restore` in the adopting thread's finally. None means
    "no scope" (the interactive path) and restores the pre-0.163 behavior."""
    prev = getattr(_BG_SCOPE_TLS, "pids", None)
    _BG_SCOPE_TLS.pids = pids
    return prev


def bg_scope_restore(prev: "set[str] | None") -> None:
    """Put back what `bg_scope_adopt` replaced (the worker's finally)."""
    _BG_SCOPE_TLS.pids = prev


def _in_scope(pid: str) -> bool:
    """True when `pid` may be announced to this thread's task."""
    pids = getattr(_BG_SCOPE_TLS, "pids", None)
    return pids is None or pid in pids


def _scan_background_completions(jobs=None) -> str | None:
    """One-time notice for tracked background jobs that finished since the
    model last looked (checked on the first tool call of each turn).

    The model must otherwise REMEMBER to wait_background/read_file a job it
    backgrounded turns ago; a build that failed 20 minutes ago doesn't
    announce itself. Returns the note text (or None): lists finished jobs with
    exit codes once, then marks them seen — no repeated noise.

    *jobs* is an already-computed list_bg_jobs() result a caller can share
    (wake_due scans the map once and threads it through the whole sweep);
    None scans it here."""
    from ._bash_jobs import list_bg_jobs, _read_bg_exit_status
    from ._verify import _is_verify_job  # verify jobs report via _stage_verify_outcome instead
    with _BG_SCAN_LOCK:
        try:
            if jobs is None:
                jobs = list_bg_jobs()
        except Exception:
            return None
        mine = os.getpid()
        seen = _notices.get("_BG_NOTICE_SEEN")
        if seen is None:
            seen = set(getattr(st, "_BG_NOTICE_BASELINE", set()) or set())
        notes = []
        for j in jobs:
            try:
                pid = str(j.get("pid", ""))
                if not pid:
                    continue
                # only jobs this nx process launched — never another session's strays
                if j.get("owner_pid") != mine:
                    continue
                if _is_verify_job(j):
                    continue  # NX_VERIFY_CMD outcome has its own dedicated note path
                if not _in_scope(pid):
                    # 0.163.0 (Phase E prereq 2): a batch task only ever hears
                    # about jobs IT launched. Deliberately NOT marked seen —
                    # the owning task still has to announce it, and this thread
                    # must not consume its sibling's note out of the shared
                    # pid->log map.
                    continue
                if j.get("alive"):
                    continue  # still running — the existing [background] flow covers it
                if pid in seen:
                    continue
                # A dead job whose log is already gone has nothing left to
                # announce: its output reached the model another way (run_bash
                # finished inside its own poll window, wait_background, or a
                # read_file of the log) and _finish_bg_job deleted the log.
                # "pid N exited (code unknown)" with no tail is pure noise —
                # and each such orphan woke a pointless auto-continuation turn
                # — so mark it consumed silently (0.37.4).
                if not j.get("log_path") or not os.path.exists(j["log_path"]):
                    seen.add(pid)
                    continue
                seen.add(pid)
                code = _read_bg_exit_status(pid)[1]
                tail = (j.get("tail") or "").strip()[:80]
                code_s = "unknown" if code is None else str(code)
                notes.append(f"  pid {pid} exited (code {code_s})" + (f" — tail: {tail}" if tail else ""))
            except Exception:
                # One malformed job must not discard the whole sweep: the
                # marks made so far are committed below either way, so a probe
                # that raises can never make the same pid re-announce forever.
                continue
        _notices.set_("_BG_NOTICE_SEEN", seen)
    if not notes:
        return None
    return ("[Runtime note: background job(s) you started have finished:\n"
            + "\n".join(notes)
            + "\nCheck their output with read_file(log) or wait_background before relying on them.]")


def _stage_bg_note(note: str) -> None:
    """Stage the sweep's note with the merge+dedup discipline of the old
    monolithic _run_bg_completion_scan (a staged note can still be unflushed
    here — the poller's own sweep may have staged one before this turn's
    first tool call; assigning would drop the earlier job's exit line)."""
    prev = _notices.get("_PENDING_BG_COMPLETION_NOTICE")
    if prev:
        # Dedup (0.37.4): the poller thread and this turn can both sweep
        # the same job in a race — _BG_SCAN_LOCK marks it seen once, but
        # each sweep may have already staged its own copy of the note.
        fresh = [ln for ln in note.splitlines()
                 if ln.strip().startswith("pid ") and ln not in prev]
        if fresh:
            note = f"{prev}\n" + "\n".join(fresh)
        # else: every line already staged — keep prev untouched
        else:
            note = prev
    _notices.set_("_PENDING_BG_COMPLETION_NOTICE", note)


def _stage_verify_outcome_safe(jobs=None) -> None:
    """The verify half of the sweep, split out so run_tool()'s inline call and
    _run_bg_completion_scan share one implementation (NX_VERIFY_CMD job just
    finished? Its outcome is staged as its own pending notice)."""
    from ._verify import _stage_verify_outcome
    try:
        _stage_verify_outcome(jobs=jobs)
    except Exception:
        pass


def _stage_subagent_completions_safe() -> None:
    """The sub-agent half of the sweep: async agents finished since the last
    look get their [Runtime note] staged (idempotent per job id)."""
    from ._subagent import _stage_subagent_completions
    try:
        _stage_subagent_completions()
    except Exception:
        pass


def _run_bg_completion_scan(jobs=None) -> None:
    """The completion sweep run_tool() runs on the first tool call of a turn:
    scan finished bg jobs, NX_VERIFY_CMD outcomes, and finished async agents,
    staging each as its one-time [Runtime note]. The wake path shares it —
    see _autowake.wake_due for why a wake turn must not fire on an empty
    sweep.

    *jobs* lets a caller (wake_due) pass the list_bg_jobs() result it already
    paid for, so one keystroke scans the pid->log map once, not three times."""
    note = _scan_background_completions(jobs=jobs)
    if note:
        _stage_bg_note(note)
    _stage_verify_outcome_safe(jobs=jobs)
    _stage_subagent_completions_safe()
    _stage_orphan_agent_intents_safe()


def _stage_orphan_agent_intents_safe() -> None:
    """The crash-residue half of the sweep (0.147.2, effect sandwich): intent
    records for async agents whose owner process died mid-run are announced
    once as a [Runtime note], then deleted. A live process's own agents are
    skipped (their intents are cleared at finalize, and liveness is checked
    before announcing anything)."""
    from ._subagent import _scan_orphan_agent_intents
    try:
        lines = _scan_orphan_agent_intents()
    except Exception:  # noqa: BLE001 — the sweep must never break the turn
        return
    if lines:
        _stage_bg_note(
            "[Runtime note: sub-agent(s) from a crashed nx session were killed "
            "mid-run and never finished:\n" + "\n".join(lines) + "\n"
            "Re-spawn deliberately if the work still matters.]")


# ---------------------------------------------------------------------------
# /bg — alternate-screen view (UIUX_SPEC Phase D pattern, 0.148.6)
# ---------------------------------------------------------------------------
# The bg-job HALF of the /tabs dashboard, split out so /tabs is a live-status
# surface again and history has its own view. Entered by the `/bg` command
# (existing: same command, same inline print as the fallback) and by Ctrl+O at
# the prompt.
#
# OWNERSHIP NOTE: the view lives in `_bg_completions` — the in-repo module
# whose job it is to walk the bg-job registry — because `_bash_jobs` (where
# `list_bg_jobs` is defined) is the module a cycle forbids reaching for a
# `_fullscreen` import: `_fullscreen` lazy-imports `_surfaces.TabsView`, and
# `_surfaces` imports `_fullscreen` at module level, so a VIEW in `_surfaces`
# would close that cycle at import time. The data accessors are imported
# lazily here for the same reason.

_BG_DONE_KEEP = 4  # newest finished jobs shown (mirrors _surfaces._DONE_KEEP)

# status -> _render role / glyph: the SAME mapping /tabs' `_glyph` and /bg's
# inline output already apply (running=SUCCESS, killed/error=ERROR,
# done/finished=ACCENT).
_BG_STATUS_ROLE = {"running": "SUCCESS", "killed": "ERROR", "error": "ERROR",
                   "partial": "WARNING", "finished": "ACCENT", "done": "ACCENT"}
_BG_GLYPH = {"running": "●", "killed": "✗", "error": "✗", "partial": "◐",
             "finished": "✔", "done": "✔"}


def _bg_status(job: dict) -> str:
    """The one status word for a job row.

    Delegates to `_surfaces._process_status` — the pre-Phase-D /tabs
    classifier (alive -> running; killed -> killed; else the recorded exit
    status or finished) — so /bg and /tabs can never disagree about what a job
    is doing. Falls back to the local rules when that helper is unavailable (a
    stripped build), which is exactly what the old inline /bg print did."""
    try:
        from ._surfaces import _process_status
        return str(_process_status(job))
    except Exception:  # noqa: BLE001 — a row must always render a status
        if job.get("alive"):
            return "running"
        if job.get("killed"):
            return "killed"
        return str(job.get("status") or "finished")


def _bg_age(job: dict) -> str:
    """Elapsed time since `job["started"]` (unix seconds), e.g. "2m14s";
    empty when the map carries no start time."""
    started = job.get("started")
    if not isinstance(started, (int, float)):
        return ""
    total = max(0, int(time.time() - started))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


class BackgroundJobsView(_fullscreen.View):
    """The /bg view: this session's background run_bash jobs, full screen.

    Read-only: a status glyph, pid, status word, age and command per row, with
    the log tail `list_bg_jobs()` already returns; killing stays `/bg kill
    <pid>` (announced in the panel footer). j/k scroll the selection, `?` is the
    shell's key overlay, Esc/q back to chat. An empty registry gets an explicit
    empty state instead of a blank frame.
    """

    surface = "bgjobs"
    title = "bg"

    def __init__(self) -> None:
        super().__init__()
        self.sel = 0
        self.jobs: list[dict] = []
        self.rows: list[dict] = []   # live jobs first, then the newest finished

    def refresh(self) -> None:
        """Re-read the registry (the poll-time repaint calls this, so a job that
        finishes while the view is open lands without reopening it)."""
        try:
            from ._bash_jobs import list_bg_jobs
            jobs = list_bg_jobs()
        except Exception:  # noqa: BLE001 — a broken registry renders empty
            jobs = []
        for job in jobs:
            job["status"] = _bg_status(job)
        done = [j for j in jobs if not j.get("alive")]
        self.rows = ([j for j in jobs if j.get("alive")]
                     + done[:_BG_DONE_KEEP])
        self.jobs = jobs
        self.sel = min(self.sel, max(0, len(self.rows) - 1))

    def _heading(self) -> str:
        n = len(self.rows)
        running = sum(1 for j in self.rows if j.get("alive"))
        if not self.jobs:
            return "bg jobs · none"
        return (f"bg jobs · {n} shown · {running} running · "
                f"{len(self.jobs) - running} finished")

    def _row_lines(self, job: dict, selected: bool, width: int) -> list[str]:
        from ._ansi import clip_display
        from ._render import RESET, role
        status = job.get("status") or "finished"
        marker = "›" if selected else " "
        glyph = (f"{role(_BG_STATUS_ROLE.get(status, 'MUTED'))}"
                 f"{_BG_GLYPH.get(status, '○')}{RESET}")
        age = _bg_age(job)
        head = (f"{role('ACCENT')}{marker}{RESET} {glyph} "
                f"{str(job.get('pid', '?')):>7} "
                f"{role('MUTED')}{status:<9}{age:<8}{RESET}")
        out = [clip_display(head, width - 4)]
        cmd = " ".join(str(job.get("command") or "").split())
        if cmd:
            out.append(f"{role('MUTED')}    {clip_display(cmd, width - 10)}"
                       f"{RESET}")
        tail = job.get("tail") or ""
        if tail:
            out.append(f"{role('MUTED')}    │ {clip_display(tail, width - 12)}"
                       f"{RESET}")
        return out

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        self.refresh()
        lines = [f"{role('ACCENT')}bg{RESET}{role('MUTED')} — "
                 f"{hint_line('bgjobs')}{RESET}"]
        body: list[str] = []
        if not self.rows:
            body.append(f"{role('MUTED')}  (no background jobs — run_bash with "
                        f"a trailing `&`, or /bg all for every session's){RESET}")
        else:
            budget = max(1, height - (1 + 2))  # title line + panel top/bottom
            row_lines = [self._row_lines(j, i == self.sel, width)
                         for i, j in enumerate(self.rows)]
            if sum(len(r) for r in row_lines) <= budget:
                for r in row_lines:
                    body.extend(r)
            else:
                # Progressive trim (TabsView's rule): every row keeps at least
                # its head line, deep tails give way first, and the trim says so.
                for i, r in enumerate(row_lines):
                    per = max(1, (budget - len(body)) // (len(row_lines) - i))
                    body.extend(r[:per])
                body.append(f"{role('MUTED')}  … some tails hidden "
                            f"(terminal too short){RESET}")
        lines.extend(panel(
            self._heading(), body,
            footer="q/esc back to chat · /bg kill <pid> stops a job"))
        return lines[:height]

    def on_key(self, key: str, shell) -> str:
        self.notice = ""
        if key == "j/k":
            return "repaint"
        if key in ("j", "down"):
            self.sel = min(max(0, len(self.rows) - 1), self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            self.sel = max(0, self.sel - 1)
            return "repaint"
        if key in ("r", "enter"):
            return "repaint"   # r: the repaint itself re-reads the registry
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def bg_jobs_view() -> "BackgroundJobsView":
    """The /bg view object the fullscreen shell runs (`_fullscreen.bgjobs()` is
    the thin alias the command path uses)."""
    return BackgroundJobsView()
