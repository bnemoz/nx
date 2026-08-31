"""Background bash job lifecycle: launch, poll, log, and reap.

run_bash always launches detached (setsid) and polls briefly before
backgrounding; wait_background resumes polling a job started earlier. Logs
live under st.SESSION_HOME/bg-logs, auto-trimmed by _delete_old_bg_logs.

`_confirm` is monkeypatched by tests via `m._confirm = fake` (the top-level
`nx` module attribute), so it's called through a lazy `import nx as _m;
_m._confirm(...)` rather than bound directly at import time — see the note
at the top of _repl.py for the same pattern.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import threading
import time
from contextlib import nullcontext
from typing import Any

from . import _state as st
from ._env import _env_int
from ._interrupt import _USER_INTERRUPTED, _watch_for_interrupts
from ._logging import _log_event


def _confirm(action: str, preview: str | None = None) -> bool:
    import nx as _m
    return _m._confirm(action, preview=preview)


def _run_bash(command: str) -> tuple[str, str]:
    """Launch `command` detached in its own session and return (pid, log_path).

    The command is started with ``setsid`` so it runs in a **new session / new
    process group**, fully detached from nx's controlling terminal. This is
    important: with plain ``nohup`` the background job stays in nx's
    process group, so a terminal Ctrl+C (SIGINT to the foreground group) kills
    it — exactly the failure mode that kept canceling long ``docker pull``s
    mid-extraction. ``nohup`` only ignores SIGHUP, not SIGINT; ``setsid`` puts
    the job beyond the reach of terminal-generated signals entirely.

    Exit-code capture is inlined into a single ``sh -c`` invocation so no
    wrapper script is needed — avoids a race where the launcher starts after
    the temp file is already deleted.

    Stores a ``bg-pid-{pid}.map`` file so ``wait_background`` can locate logs
    by PID later. On Linux the map also records the process start time, which
    distinguishes the launched process from a later process that reuses its PID.
    """
    _delete_old_bg_logs()
    bg_dir = os.path.join(st.SESSION_HOME, "bg-logs")
    os.makedirs(bg_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"{stamp}-{secrets.token_hex(3)}"
    log_path = os.path.join(bg_dir, f"bg-{suffix}.log")
    # Inline: run the command in a subshell, then echo its exit status.
    # setsid detaches into a new session so terminal Ctrl+C / SIGINT can't
    # reach the background job. Redirect to the log file ourselves.
    inner = f"({command}); echo \"[exit: $?]\""
    wrapped = (f"setsid sh -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 < /dev/null &\n"
               f"echo $!")
    r = subprocess.run(wrapped, shell=True, capture_output=True, text=True)
    pid = r.stdout.strip()
    # Write a tiny pid→logpath map file so wait_background can find it.
    # Also stash the command + start time so /bg can render a dashboard
    # without needing to re-derive anything from the log itself. `owner` is
    # this process's pid: the completion scan only announces jobs this nx
    # process started, never another session's strays (or jobs from an old
    # ~/.nx shared by every session on the machine).
    try:
        map_path = os.path.join(bg_dir, f"bg-pid-{pid}.map")
        with open(map_path, "w") as f:
            json.dump({"log_path": log_path, "command": command,
                       "started": time.time(),
                       "pid_start_time": _pid_start_time(pid),
                       "owner_pid": os.getpid()}, f)
    except OSError:
        pass
    return pid, log_path


def _pid_start_time(pid: int | str) -> str | None:
    """Return Linux's immutable process start-time tick for *pid*, if known."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            stat = f.read()
        # ``comm`` (field 2) can contain spaces and parentheses; split after
        # its final closing parenthesis before indexing fields 3 onward.
        fields = stat.rsplit(") ", 1)[1].split()
        return fields[19]  # field 22 (starttime), after fields 1 and 2
    except (OSError, ValueError, IndexError):
        return None


def _pid_is_alive(pid: int, expected_start_time: str | None = None) -> bool:
    """Return whether *pid* is alive and, when known, is the original job."""
    if expected_start_time is not None:
        current_start_time = _pid_start_time(pid)
        if current_start_time is not None and current_start_time != expected_start_time:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # exists but owned by another uid
    except (ProcessLookupError, OSError):
        return False


def list_bg_jobs() -> list[dict[str, Any]]:
    """Scan ``bg-logs/bg-pid-*.map`` files and return a dashboard-ready list.

    Each entry: {"pid", "alive", "command", "started", "log_path", "tail"}.
    Sorted newest-first. Used by the ``/bg`` REPL command.
    """
    bg_dir = os.path.join(st.SESSION_HOME, "bg-logs")
    jobs = []
    try:
        names = [n for n in os.listdir(bg_dir) if n.startswith("bg-pid-") and n.endswith(".map")]
    except OSError:
        return jobs
    for name in names:
        p = os.path.join(bg_dir, name)
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        pid = name[len("bg-pid-"):-len(".map")]
        try:
            pid_i = int(pid)
            alive = _pid_is_alive(pid_i, data.get("pid_start_time"))
        except ValueError:
            alive = True  # bad pid — retain the historical dashboard behavior
        log_path = data.get("log_path", "")
        jobs.append({
            "pid": pid,
            "alive": alive,
            "command": data.get("command", ""),
            "started": data.get("started"),
            "log_path": log_path,
            "owner_pid": data.get("owner_pid"),
            "tail": _tail_last_line(log_path) if log_path else "",
        })
    jobs.sort(key=lambda j: j.get("started") or 0, reverse=True)
    return jobs


def _delete_old_bg_logs(max_logs: int = 50, max_age_days: int = 7) -> None:
    """Trim background logs so they don't pile up forever.

    Also removes stale ``bg-pid-*.map`` files whose target log no longer exists.
    """
    bg_dir = os.path.join(st.SESSION_HOME, "bg-logs")
    try:
        log_entries = []
        map_entries = []
        for name in os.listdir(bg_dir):
            if not name.startswith("bg-"):
                continue
            p = os.path.join(bg_dir, name)
            try:
                if name.endswith(".map"):
                    map_entries.append((os.path.getmtime(p), p, name))
                else:
                    log_entries.append((os.path.getmtime(p), p, name))
            except OSError:
                continue
        now = time.time()
        # remove logs by age
        for mtime, p, _ in log_entries:
            if (now - mtime) > max_age_days * 86400:
                try:
                    os.remove(p)
                except OSError:
                    pass
        # then cap log count (keep newest)
        log_entries.sort(key=lambda x: x[0], reverse=True)
        for _, p, _ in log_entries[max_logs:]:
            try:
                os.remove(p)
            except OSError:
                pass
        # Clean stale map files (no corresponding log) and age-prune maps too
        for mtime, p, _ in map_entries:
            try:
                with open(p) as f:
                    data = json.load(f)
                target = data.get("log_path", "")
                should_del = not os.path.exists(target)
            except (OSError, json.JSONDecodeError):
                should_del = True
            if not should_del and (now - mtime) > max_age_days * 86400:
                should_del = True
            if should_del:
                try:
                    os.remove(p)
                except OSError:
                    pass
    except OSError:
        pass


def _tail_last_line(log_path: str, maxlen: int = 70) -> str:
    """Return the last non-empty line of *log_path*, trimmed to *maxlen* chars.

    Reads only the trailing chunk of the file (no full-file read), so this is
    cheap to call on every poll tick even for chatty commands.
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if line and not re.match(r"^\[exit:\s*-?\d+]$", line):
            return line if len(line) <= maxlen else line[:maxlen - 1] + "…"
    return ""


def _poll_pid(pid: int | str, timeout: float, check_esc: bool = False,
              log_path: str | None = None, expected_start_time: str | None = None) -> str:
    """Poll *pid* up to *timeout* seconds.

    Returns:
        "exited"      — process finished
        "timeout"     — poll window expired, process still alive
        "interrupted" — Esc was pressed (only if *check_esc* is True)

    Uses ``os.kill(pid, 0)`` (a kernel-level signal-0 probe) instead of
    shelling out to the external ``kill`` binary.  Some minimal Docker
    images (Alpine/busybox without ``procps``) don't ship a ``kill``
    executable in PATH, which caused *every* ``run_bash`` call to raise
    ``FileNotFoundError`` — the agent could run no shell commands at all.
    ``os.kill`` talks to the kernel directly, so it works in any
    environment with a POSIX layer, no external binary required.
    When a job map provides ``expected_start_time``, Linux also compares
    ``/proc/<pid>/stat`` before signal-0 so a reused PID is treated as exited.


    Signal-0 semantics:
      • returns normally    → process exists and is signalable → alive
      • ProcessLookupError  → ESRCH, no such process            → exited
      • PermissionError     → EPERM, process exists but not ours → alive
      • ValueError          → pid wasn't a valid int             → exited

    If *log_path* is given, the active spinner's label (``st._ACTIVE_SPINNER``)
    is updated each tick with the log's last line, so a long-running command's
    progress streams live into the chatbox instead of staying silent until it
    finishes or gets backgrounded.
    """
    deadline = time.time() + timeout
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return "exited"  # no valid pid → the command never started
    # model_turn disarms the watcher before dispatching a tool. Re-arm it for
    # the actual polling window; otherwise this check could only observe a
    # stale event and Esc could never interrupt run_bash/wait_background.
    watch = _watch_for_interrupts() if check_esc else nullcontext()
    with watch:
        last_tail = None
        _last_label_at = time.monotonic()
        while time.time() < deadline:
            if not _pid_is_alive(pid_i, expected_start_time):
                return "exited"
            if check_esc and _USER_INTERRUPTED.is_set():
                return "interrupted"
            if log_path is not None:
                tail = _tail_last_line(log_path)
                if tail and tail != last_tail:
                    spinner = st._ACTIVE_SPINNER
                    if spinner is not None:
                        # throttle label updates to ~1/s: the spinner redraws
                        # every tick, and a chatty command emitting lines every
                        # 100ms must not turn the calm sweep back into a
                        # flicker source (photosensitivity — see _ansi.py).
                        now = time.monotonic()
                        if now - _last_label_at >= 1.0 and tail != spinner.label:
                            spinner.label = tail[:60]
                            _last_label_at = now
                    last_tail = tail
            time.sleep(0.1)
    return "timeout"


def _read_log(log_path: str) -> tuple[str, int | None]:
    """Read *log_path*, parse the trailing ``[exit: N]`` marker, return (content, exit_code)."""
    try:
        with open(log_path, "rb") as f:
            out = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "", None
    exit_code = None
    last_nl = out.rstrip("\n").rfind("\n")
    trailer = out[last_nl + 1:].strip() if last_nl != -1 else out.strip()
    m = re.match(r"\[exit:\s*(-?\d+)]", trailer)
    if m:
        exit_code = int(m.group(1))
        out = out[:last_nl] if last_nl != -1 else ""
    return out, exit_code


def _infer_timeout_from_sleep(command: str, default: int) -> int:
    """If *command* contains ``sleep N`` (with N in seconds, fractional seconds,
    or a time suffix — ``sleep 0.5``, ``sleep 2m``, ``sleep 1h30m``), return a poll
    timeout that comfortably covers it, so the command finishes synchronously
    without the model having to remember ``timeout=N``.

    This compensates for the natural habit of writing
    ``run_bash("sleep 30 && cat …")`` — a pattern that *seems* like it should
    block (the shell waits, then cats) but is itself launched detached (setsid), so
    it gets backgrounded after ~3 s just like anything else.  By recognising
    the sleep we make the "obvious" pattern work again.
    """
    max_sleep = 0.0
    # `sleep` accepts an integer number of seconds, a fractional one (GNU/BSD
    # both take `sleep 0.5`), or a suffixed form (GNU coreutils ≥8.4: `s`, `m`,
    # `h`, `d`, chained as in `sleep 1h30m`). A suffixed or fractional sleep is
    # still a wall-clock wait, so it must extend the poll window just like
    # `sleep 30` does — previously anything non-integer silently fell through
    # to the 3 s default and backgrounded.
    # Accept only values GNU/BSD sleep would (a bare number or a suffixed
    # duration chain, validated by _sleep_seconds); a bare `sleep $VAR` is
    # left alone — its value is unknowable here. The capture grabs sleep's
    # first argument up to whitespace/shell-boundary; junk like `2M` or `1x`
    # parses to 0.0 and falls back to the default poll window. (GNU sums
    # multiple `sleep 30 5` args — we take the first, which covers the
    # overwhelming case of a single wait.)
    for m in re.finditer(r'(?:^|[;&|]\s*)sleep\s+([^\s;&|]+)', command):
        n = _sleep_seconds(m.group(1))
        if n > max_sleep:
            max_sleep = n
    if max_sleep > 0:
        return int(max_sleep) + 10
    return default


def _sleep_seconds(token: str) -> float:
    """Parse one ``sleep`` argument — a bare number (``30``, ``0.5``) or a
    suffixed duration chain (``2m``, ``1h30m``, ``1h30``) — into seconds,
    following GNU sleep semantics: each number carries an optional unit
    (default seconds) and units may repeat. Returns 0.0 for anything
    unparseable so the caller falls back to the default poll window. Units are
    case-sensitive lowercase — GNU sleep rejects ``2M`` (it is not case-
    insensitive), so an uppercase unit must not extend the poll window."""
    unit_mult = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    part_re = re.compile(r"(\d+(?:\.\d+)?)([smhd]?)", re.ASCII)
    total = 0.0
    pos = 0
    for m in part_re.finditer(token):
        if m.start() != pos:
            return 0.0  # gap between number/unit pairs → not a clean chain
        total += float(m.group(1)) * unit_mult[m.group(2) or "s"]
        pos = m.end()
    if pos != len(token) or pos == 0:
        return 0.0
    return total


# Secret-looking patterns in a run_bash command string. When NX_BASH_PROMPT_SECRETS=1,
# a command matching any of these forces the approval prompt even in YOLO / high
# auto-approve mode — a guard against the model echoing a fetched secret into
# another command (curl … | sh), pasting a key into a file write, etc.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),                    # OpenAI-style keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),               # GitHub tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),                          # AWS access key ids
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),             # Slack tokens
    re.compile(r"glpat-[A-Za-z0-9_-]{16,}"),                 # GitLab PATs
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),       # PEM blocks
)


def _contains_secret(command: str) -> str | None:
    """Return the name of the first secret pattern the command matches, or None."""
    for rx in _SECRET_PATTERNS:
        m = rx.search(command)
        if m:
            return m.group(0)[:8] + "…"
    return None


def run_bash(command: str, timeout: int | None = None, **_: Any) -> str:
    """Run a shell command.

    Commands always run detached via setsid — they never block the agent loop.
    If the command finishes within a short poll window (~3 s by default),
    output is returned directly and the log file is deleted.  If it's still
    running after the poll window, it continues in the background and you
    get a PID + log path to check later with ``read_file``.

    **Do NOT use** ``sleep N && …`` inside the command string as a way to
    wait — the entire command is launched detached (setsid) so *it* will also
    be backgrounded after a few seconds.  Instead, pass ``timeout=N`` to tell
    nx to wait synchronously (e.g. ``timeout=35`` for a ``sleep 30``
    command).  As a convenience, if your command *does* start with
    ``sleep N …`` the tool auto-extends its poll window to ``N+10`` s
    so the "obvious" pattern still works.

    Use ``timeout=0`` to wait *indefinitely* (the tool blocks until the
    command completes).  To wait for an **already-backgrounded** command,
    use ``wait_background(pid=…)``.

    Long-running commands (servers, watchers, builds) produce background logs
    at ``~/.nx/bg-logs/``, automatically cleaned up (max ~50 files /
    7 days).  Pressing Esc during the poll window immediately backgrounds
    the command.

    NX_BASH_MAX_SECONDS (default 0 = off) is a watchdog on *backgrounded*
    commands: after the poll window expires and the process keeps running, nx
    reaps it at N seconds (SIGTERM, then SIGKILL) instead of leaving it to run
    forever. Long-running servers/builds that SHOULD outlive the tool call are
    the normal case — the default is off; set it when you want a hard ceiling.

    With NX_BASH_PROMPT_SECRETS=1, a command that looks like it carries a
    credential (sk-…, ghp_…, AKIA…, a PEM block, …) always prompts, even in
    YOLO mode — the prompt is the last chance to catch a secret before it
    lands in a log, a file, or a remote host.
    """
    secret = _contains_secret(command) if _env_int("NX_BASH_PROMPT_SECRETS", 0) else None
    if secret is not None:
        from ._subagent import _subagent_quiet
        if _subagent_quiet():
            # A detached agent can't show the secret-guard prompt — fail the
            # call instead of prompting from a background thread.
            return (f"ERROR: command looks like it carries a credential ({secret!r}); "
                    "refused in a background sub-agent — let the parent run it")
        # force the prompt even under YOLO: bypass the level/YOLO short-circuit
        # with a direct ask (Esc still drops to chat)
        from ._approval import _ask_approval
        import nx as _m
        choice = _m._ask_approval(
            f"  run (contains possible secret {secret!r}): {command[:120]}? "
            f"[Y/n/esc] ", allow_hint=None)
        if choice == "esc":
            from ._approval import _EscToChat
            raise _EscToChat(f"run: {command}")
        if choice != "y":
            return "DENIED by user"
    elif not _confirm(f"run: {command}"):
        return "DENIED by user"
    if timeout is None:
        timeout = _env_int("NX_BASH_POLL_SECONDS", 3)
        timeout = _infer_timeout_from_sleep(command, timeout)
    if timeout <= 0:
        timeout = 999999999  # ∞ — wait indefinitely
    max_seconds = _env_int("NX_BASH_MAX_SECONDS", 0)
    pid, log_path = _run_bash(command)
    status = _poll_pid(pid, timeout, check_esc=True, log_path=log_path,
                       expected_start_time=_find_bg_start_time_for_pid(pid))
    if status == "interrupted":
        _log_event("tool", f"run_bash interrupted by user while polling (pid {pid})")
        return (f"[background] PID {pid} | log: {log_path}\n"
                "The command was interrupted (Esc pressed) and is now running "
                "in the background. "
                f"Check output with read_file('{os.path.abspath(log_path)}')")
    if status == "timeout":
        _maybe_start_watchdog(pid, command, max_seconds)
        return (f"[background] PID {pid} | log: {log_path}\n"
                "The command is still running in the background. "
                f"Check output with read_file('{os.path.abspath(log_path)}')")
    return _finish_bg_job(log_path)


# PIDs this process has handed a watchdog thread (avoid duplicates when the
# same pid is reported backgrounded twice — shouldn't happen, but cheap).
_WATCHDOG_PIDS: set[int] = set()


def _maybe_start_watchdog(pid: int | str, command: str, max_seconds: int) -> None:
    """Arm the NX_BASH_MAX_SECONDS reaper for a just-backgrounded job.

    A daemon thread sleeps out the remaining budget (poll-window seconds are
    already spent by the time we get here — the watchdog counts from process
    start as closely as we can observe), then SIGTERMs and, after a grace
    period, SIGKILLs the process group. Setsid put the job in its own session,
    so killing the group takes down any children it spawned — which is what
    you want from a ceiling: nothing survives to keep consuming the machine.
    """
    if max_seconds <= 0:
        return
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return
    start_time = _find_bg_start_time_for_pid(pid)
    if pid_i in _WATCHDOG_PIDS:
        return
    _WATCHDOG_PIDS.add(pid_i)

    def _reap() -> None:
        try:
            time.sleep(max(1, max_seconds))
            if not _pid_is_alive(pid_i, start_time):
                return
            try:
                os.killpg(os.getpgid(pid_i), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                return
            time.sleep(5)
            if _pid_is_alive(pid_i, start_time):
                try:
                    os.killpg(os.getpgid(pid_i), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            _log_event("tool", f"watchdog killed bg job pid {pid_i} after {max_seconds}s: {command[:120]}")
        except Exception:
            pass  # a watchdog must never take the agent down
        finally:
            _WATCHDOG_PIDS.discard(pid_i)

    threading.Thread(target=_reap, daemon=True,
                     name=f"nx-bash-watchdog-{pid_i}").start()


def _finish_bg_job(log_path: str) -> str:
    """Read+format the output of a finished background job, deleting its log."""
    time.sleep(0.2)  # brief settle so the log is flushed
    out, exit_code = _read_log(log_path)
    try:
        os.remove(log_path)
    except OSError:
        pass
    if exit_code is not None:
        return f"[exit {exit_code}]\n{out}"
    return f"[exited]\n{out}"


def wait_background(pid: int | str, log_path: str | None = None, timeout: int = 0, **_: Any) -> str:
    """Wait for a backgrounded command (started by ``run_bash``) to finish.

    Polls *pid* until it exits (default: wait indefinitely).  When the command
    exits the output is returned and the log file is cleaned up.  Esc can be
    pressed at any time to interrupt the wait — the command keeps running and
    the log path is returned so you can check back later.

    Pass ``timeout=N`` only if you specifically want to bail after N seconds
    (rarely useful — you called this because you want to *wait*, not guess a
    ceiling).

    Parameters
    ----------
    pid : int or str
        The PID returned by the prior ``run_bash`` call.
    log_path : str, optional
        Ignored; kept for backward compatibility with older tool-call args.
        The log path is always resolved server-side from *pid* via the
        pid->log map written by ``run_bash``.
    timeout : int
        Max seconds to wait (default 0 = wait indefinitely until the command
        finishes or Esc is pressed).
    """
    pid = int(pid)
    # Never trust a caller-supplied log_path: always resolve it ourselves from
    # the pid->log map written by _run_bash, so a model-supplied path/pid pair
    # can't be used to read arbitrary files outside the run_bash pipeline.
    log_path = _find_bg_log_for_pid(pid)
    if not log_path:
        return (f"[error] No background log found for PID {pid}. "
                "The log may have been cleaned up already.")
    timeout_s = 999999999 if timeout <= 0 else timeout
    status = _poll_pid(pid, timeout_s, check_esc=True, log_path=log_path,
                       expected_start_time=_find_bg_start_time_for_pid(pid))
    if status == "interrupted":
        return (f"[interrupted] PID {pid} still running. "
                f"Check later with read_file('{os.path.abspath(log_path)}')")
    if status == "timeout":
        return (f"[timeout after {timeout}s] PID {pid} still running. "
                f"Check output so far with read_file('{os.path.abspath(log_path)}')")
    return _finish_bg_job(log_path)


def _read_bg_exit_status(pid: int | str) -> tuple[bool, int | None, str]:
    """Non-destructive status probe for a background job's completion notice.

    Returns (exited, exit_code, log_path). Unlike _finish_bg_job it does NOT
    delete the log or consume the output — the model is still free to call
    wait_background/read_file on it. exit_code is None when the job is alive
    or the [exit: N] marker hasn't been written yet.
    """
    log_path = _find_bg_log_for_pid(pid)
    if not log_path:
        return True, None, ""
    if _pid_is_alive(int(pid), _find_bg_start_time_for_pid(pid)):
        return False, None, log_path
    out, exit_code = _read_log(log_path)
    return True, exit_code, log_path


def _find_bg_start_time_for_pid(pid: int | str) -> str | None:
    """Return the recorded process identity for a background PID, if any."""
    map_path = os.path.join(st.SESSION_HOME, "bg-logs", f"bg-pid-{pid}.map")
    try:
        with open(map_path) as f:
            return json.load(f).get("pid_start_time")
    except (OSError, IOError, json.JSONDecodeError):
        return None


def _find_bg_log_for_pid(pid: int | str) -> str | None:
    """Look up the log path for a background PID via the pid→log map files
    written by ``_run_bash``.  Returns the log path, or None if not found."""
    pid = str(pid)
    bg_dir = os.path.join(st.SESSION_HOME, "bg-logs")
    map_path = os.path.join(bg_dir, f"bg-pid-{pid}.map")
    try:
        with open(map_path) as f:
            data = json.load(f)
        return data.get("log_path")
    except (OSError, IOError, json.JSONDecodeError):
        pass
    # Fallback: try to find a log file that mentions this PID in the first 1K
    try:
        for name in os.listdir(bg_dir):
            if not name.startswith("bg-") or ".map" in name:
                continue
            p = os.path.join(bg_dir, name)
            try:
                with open(p, "rb") as f:
                    head = f.read(1024).decode("utf-8", errors="replace")
                if pid in head:
                    return p
            except OSError:
                continue
    except OSError:
        pass
    return None
