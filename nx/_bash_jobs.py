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
import subprocess
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
    # without needing to re-derive anything from the log itself.
    try:
        map_path = os.path.join(bg_dir, f"bg-pid-{pid}.map")
        with open(map_path, "w") as f:
            json.dump({"log_path": log_path, "command": command,
                       "started": time.time(),
                       "pid_start_time": _pid_start_time(pid)}, f)
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
                        spinner.label = f"running: {tail}"
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
    """If *command* contains ``sleep N`` (with N in seconds), return a poll
    timeout that's ``max(N) + 10``, so the command finishes synchronously
    without the model having to remember ``timeout=N``.

    This compensates for the natural habit of writing
    ``run_bash("sleep 30 && cat …")`` — a pattern that *seems* like it should
    block (the shell waits, then cats) but is itself launched via nohup, so
    it gets backgrounded after ~3 s just like anything else.  By recognising
    the sleep we make the "obvious" pattern work again.
    """
    max_sleep = 0
    for m in re.finditer(r'(?:^|[;&|]\s*)sleep\s+(\d+)', command):
        n = int(m.group(1))
        if n > max_sleep:
            max_sleep = n
    if max_sleep > 0:
        return max_sleep + 10
    return default


def run_bash(command: str, timeout: int | None = None, **_: Any) -> str:
    """Run a shell command.

    Commands always run via nohup — they never block the agent loop.
    If the command finishes within a short poll window (~3 s by default),
    output is returned directly and the log file is deleted.  If it's still
    running after the poll window, it continues in the background and you
    get a PID + log path to check later with ``read_file``.

    **Do NOT use** ``sleep N && …`` inside the command string as a way to
    wait — the entire command is launched via nohup so *it* will also be
    backgrounded after a few seconds.  Instead, pass ``timeout=N`` to tell
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
    """
    if not _confirm(f"run: {command}"):
        return "DENIED by user"
    if timeout is None:
        timeout = _env_int("NX_BASH_POLL_SECONDS", 3)
        timeout = _infer_timeout_from_sleep(command, timeout)
    if timeout <= 0:
        timeout = 999999999  # ∞ — wait indefinitely
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
        return (f"[background] PID {pid} | log: {log_path}\n"
                "The command is still running in the background. "
                f"Check output with read_file('{os.path.abspath(log_path)}')")
    return _finish_bg_job(log_path)


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
