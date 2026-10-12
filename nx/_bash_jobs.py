"""Background bash job lifecycle: launch, poll, log, and reap.

run_bash always launches detached (own session) and polls briefly before
backgrounding; wait_background resumes polling a job started earlier. Logs
live under st.SESSION_HOME/bg-logs/<project-slug>/ (per-project since
0.145.2; the flat parent is the legacy machine-global store), auto-trimmed
by _delete_old_bg_logs.

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
from ._env import _env_float, _env_int
from ._interrupt import (
    _USER_INTERRUPTED, _watch_for_interrupts, cancel_requested,
    clear_tool_cancel, register_tool_cancel,
)
from ._logging import _log_event
from ._ripgrep import prefer_ripgrep
from ._private import open_private, private_dir, touch_private


def _confirm(action: str, preview: str | None = None) -> bool:
    import nx as _m
    return _m._confirm(action, preview=preview)


def _wrap_exit_trailer(command: str) -> str:
    """Wrap *command* in a subshell that echoes its exit status.

    The trailing newline before the closing ``)`` is load-bearing, not
    cosmetic: a model-authored command routinely arrives with no final newline,
    and gluing ``)`` onto its last line corrupts every shape that must end on a
    line of its own. The common case is a heredoc — the terminator has to be
    *alone* on its line, so ``EOF); echo "[exit: 0]"`` never closes the heredoc,
    the ``(`` never closes, and the shell dies at parse time with
    ``Syntax error: end of file unexpected (expecting ")")`` before running
    anything. A trailing ``# comment`` has the same requirement. A trailing
    ``\\`` is unaffected: backslash-newline splices the lines back together.

    A genuinely broken command (dangling ``&&``, unbalanced quote) still fails
    loudly, which is the correct outcome.
    """
    return f"({command}\n); echo \"[exit: $?]\""


def _heredoc_quote_trap(command: str) -> str | None:
    """Detect a heredoc whose delimiter is a quoting idiom, not a plain word.

    Models fuse two shell idioms that never belong together: the ssh
    quote-embedding dance (each single quote respelled as an alternating
    double/single-quote pair) and a heredoc delimiter. The shell decodes the
    word after << with normal quoting rules, so the fused spelling yields a
    7-character delimiter of quote characters around EOF, which a bare EOF
    terminator can never match. The heredoc then swallows everything after it
    (including the exit-trailer wrapper's closing paren) and sh dies at parse
    time with "Syntax error: end of file unexpected" - the exact signature of
    the pre-0.36.2 wrapper bug, which is why the failure got misread as "the
    heredoc wrapper broke again". nx executes the command verbatim; this
    pre-flight check names the culprit instead of leaving the model to
    misdiagnose a parse error it cannot see the source of.

    A delimiter word that carries shell quoting is decoded the way the shell
    would: single-quoted, double-quoted, backslash-escaped and doubled-empty-
    quote spellings of a plain word all pass. A word that still carries quote
    characters after decoding (the fused-idiom shape) is reported when the
    body closes with the bare token its author clearly meant; if the body
    instead closes with that same quote-riddled token, the heredoc is
    self-consistent and passes.

    Returns a human-readable explanation, or None when no trap is present.
    Diagnosis only - never rewrites the command.
    """
    sq, dq = chr(39), chr(34)
    fused = sq + dq + sq + sq + "EOF" + sq + sq + dq + sq
    for m in re.finditer(r"<<(-?)([ \t]*)(\S+)", command):
        delim = m.group(3)
        if not ("'" in delim or '"' in delim or "\\" in delim):
            continue  # bare word delimiter (EOF, PY, END) - nothing to decode
        # Decode the delimiter word the way the shell would: strip quoting.
        decoded_chars: list[str] = []
        in_sq = in_dq = False
        for ch in delim:
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == "\\" and not in_sq and not in_dq:
                continue  # backslash-quoted delimiter char (<<\\EOF)
            else:
                decoded_chars.append(ch)
        decoded = "".join(decoded_chars)
        if decoded == "EOF" and not in_sq and not in_dq:
            continue  # a legal quoting spelling of the word EOF
        # The word still carries quote characters after decoding, so the
        # delimiter is not the plain token its author intended. Report when
        # the body closes with that bare token -- such a heredoc can never
        # terminate, because its real delimiter is the quote-riddled word.
        core = decoded.strip("'\"\\")
        if (
            core
            and core != decoded
            and re.search(r"(?m)^\s*" + re.escape(core) + r"\s*$", command[m.end():])
        ):
            return (
                "ERROR: heredoc delimiter is a quoting idiom, not a word - the "
                "shell decodes what follows << to the delimiter " + repr(delim)
                + " (decoded: " + repr(decoded) + "), so the body's bare "
                "terminator line never matches and the whole command dies at "
                "parse time "
                "(Syntax error: end of file unexpected). The " + fused
                + " spelling is the ssh quote-embedding idiom; it must never "
                "decorate a heredoc delimiter. Write << 'EOF' and close with "
                "EOF alone on its line."
            )
    return None


def _heredoc_terminator_suffix_trap(command: str) -> str | None:
    """Detect a heredoc whose terminator line carries trailing text.

    The other half of the fused-quote trap: a model appends a redirection or
    chain onto the terminator line (a trailing "2>&1", "EOF > /tmp/x", or an
    "&&" chain). The shell matches the terminator against the WHOLE line, so
    that line is no longer the delimiter, the heredoc swallows everything
    after it (including the exit-trailer wrapper's closing paren) and sh dies
    at parse time. The model sees a nonsense parse error plus its own stray
    redirection echoed back and misdiagnoses "the shell is garbling my
    commands" — a belief it then carries for the whole session. This check
    names the real culprit and states that the shell ran the command verbatim.

    Fires only when no bare-terminator line exists after the operator AND a
    line that starts with the delimiter carries trailing content — the
    intended terminator with junk glued on. A delimiter line with trailing
    text elsewhere in a well-formed heredoc (an unlikely but legal body line)
    stays silent, as does a redirection on the OPERATOR line (a "<<EOF 2>&1"
    operator), which is legal shell.

    Returns a human-readable explanation, or None when no trap is present.
    Diagnosis only — never rewrites the command.
    """
    for m in re.finditer(r"<<(-?)([ \t]*)(\S+)", command):
        _t = _heredoc_terminator_scan(m, command)
        if _t:
            return _t
    return None


def _bg_dir() -> str:
    """The bg-logs dir for THIS project: bg-logs/<project-slug>/ (0.145.2).
    The flat parent bg-logs/ is the legacy machine-global store: readers that
    accept foreign entries scan it too (owner_pid filtering stays the gate),
    and _delete_old_bg_logs drains it."""
    import nx as _m
    return os.path.join(st.SESSION_HOME, "bg-logs", _m.project_store_slug())


def _bg_pid_map_path(pid: int | str) -> str:
    """Path of the pid→map file: per-project dir first, legacy flat parent
    as fallback (a job launched before the 0.145.2 split keeps working)."""
    name = f"bg-pid-{pid}.map"
    p = os.path.join(_bg_dir(), name)
    if os.path.exists(p):
        return p
    return os.path.join(st.SESSION_HOME, "bg-logs", name)


def _run_bash(command: str, _cwd: str | None = None) -> tuple[str, str]:
    """Launch `command` detached in its own session and return (pid, log_path).

    The command is started in a **new session / new process group**, fully
    detached from nx's controlling terminal. This is important: with plain
    ``nohup`` the background job stays in nx's process group, so a terminal
    Ctrl+C (SIGINT to the foreground group) kills it — exactly the failure
    mode that kept canceling long ``docker pull``s mid-extraction. ``nohup``
    only ignores SIGHUP, not SIGINT; a new session puts the job beyond the
    reach of terminal-generated signals entirely.

    The new session is created by the launcher subprocess itself via the
    ``start_new_session`` option of ``subprocess.run`` — the C library's
    setsid syscall, invoked directly. This replaces the earlier external
    ``setsid`` binary: macOS does not ship one (it is a util-linux tool), so
    the old shape failed there with "setsid: command not found" and blocked
    every command. The syscall path works anywhere Python does, Linux and
    macOS alike, with identical semantics: the wrapper shell becomes session
    leader and the backgrounded job inherits that session.

    Exit-code capture is inlined into a single ``sh -c`` invocation so no
    wrapper script is needed — avoids a race where the launcher starts after
    the temp file is already deleted.

    Stores a ``bg-pid-{pid}.map`` file so ``wait_background`` can locate logs
    by PID later. On Linux the map also records the process start time, which
    distinguishes the launched process from a later process that reuses its PID.
    """
    _delete_old_bg_logs()
    bg_dir = _bg_dir()
    private_dir(bg_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"{stamp}-{secrets.token_hex(3)}"
    log_path = os.path.join(bg_dir, f"bg-{suffix}.log")
    # The log is created HERE, at 0600, before the shell redirection opens it:
    # `>` honours the umask, so a group-writable default would put every
    # backgrounded command's full output where other local users can read it
    # (and this directory is shared by every nx session on the machine).
    touch_private(log_path)
    # Inline: run the command in a subshell, then echo its exit status.
    # start_new_session detaches into a new session so terminal Ctrl+C /
    # SIGINT can't reach the background job — no external setsid binary
    # needed (macOS does not ship one). Redirect to the log file ourselves.
    # Prefer bash: on Debian/Ubuntu /bin/sh is dash, which rejects common
    # bash-isms (&>, |&, process substitution) with "Syntax error:
    # redirection unexpected".  Fall back to sh on shells-free systems.
    import shutil as _shutil
    shell = _shutil.which("bash") or "sh"
    inner = _wrap_exit_trailer(command)
    wrapped = (f"{shell} -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 < /dev/null &\n"
               f"echo $!")
    r = subprocess.run(wrapped, shell=True, capture_output=True, text=True,
                       start_new_session=True, cwd=_cwd or None)
    pid = r.stdout.strip()
    # The launcher must echo exactly the PID of the detached job ($!) — an
    # integer. Anything else (empty output when sh died before the
    # echo, a banner, a multi-line error) means there is no real PID: refuse
    # to mint a ``bg-pid-<garbage>.map`` file for it, so pid→log resolution
    # can never be pointed at a log this process didn't start.
    if not pid.isdigit() or int(pid) <= 1:
        pid = ""
    # Write a tiny pid→logpath map file so wait_background can find it.
    # Also stash the command + start time so /bg can render a dashboard
    # without needing to re-derive anything from the log itself. `owner` is
    # this process's pid: the completion scan only announces jobs this nx
    # process started, never another session's strays (or jobs from an old
    # ~/.nx shared by every session on the machine).
    if pid:
        try:
            map_path = os.path.join(bg_dir, f"bg-pid-{pid}.map")
            with open_private(map_path) as f:
                json.dump({"log_path": log_path, "command": command,
                           "started": time.time(),
                           "pid_start_time": _pid_start_time(pid),
                           "owner_pid": os.getpid(),
                           # the launch directory: bg-logs is shared by every
                           # session on the machine, so a dashboard that ever
                           # lists foreign jobs needs to say where they run
                           "cwd": os.getcwd()}, f)
        except OSError:
            pass
    return pid, log_path


def _suffix_trap_msg(term_line: str, core: str) -> str:
    """The diagnosis for a terminator line that carries trailing text."""
    fix = "cmd <<'{}' ... {} 2>&1".format(core, core)
    tail = ("Put the terminator {!r} alone on its line and put the "
            "redirection on the command line: {}").format(core, fix)
    mid = ("The terminator line reads {!r}: the shell matches the delimiter "
           "against the WHOLE line, so any trailing text (a redirection such "
           "as 2>&1, a && chain) glued onto {!r} makes that line part of the "
           "heredoc body.").format(term_line, core)
    return ("ERROR: this heredoc never terminated, so the command died at "
            "parse time — the shell ran it verbatim, nothing was garbled. "
            + mid + " " + tail)


def _heredoc_terminator_scan(m, command: str) -> str | None:
    """Per-operator scan behind _heredoc_terminator_suffix_trap."""
    delim = m.group(3)
    in_sq = in_dq = False
    if "'" in delim or '"' in delim or "\\" in delim:
        # Decode a quoted delimiter the way the shell would: the closing
        # terminator is spelled with the decoded word, quotes stripped.
        decoded_chars: list[str] = []
        for ch in delim:
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == "\\" and not in_sq and not in_dq:
                continue
            else:
                decoded_chars.append(ch)
        core = "".join(decoded_chars).strip("'\"\\")
    else:
        core = delim
    if not core or in_sq or in_dq:
        return None
    rest = command[m.end():]
    # <<-'EOF' allows leading TABS on the terminator; without the dash no
    # leading whitespace ever matches, so require the line to start with the
    # delimiter itself (tabs allowed only when the operator was <<-).
    lead = r"[ \t]*" if m.group(1) else ""
    bare = re.compile(r"(?m)^" + lead + re.escape(core) + r"[ \t]*$")
    if bare.search(rest):
        return None  # the heredoc closes properly somewhere — not this trap
    suffixed = re.search(r"(?m)^" + lead + re.escape(core) + r"[ \t]+(\S.*)$",
                         rest)
    if not suffixed:
        return None
    return _suffix_trap_msg(suffixed.group(0), core)


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


def list_bg_jobs(mine_only: bool = True) -> list[dict[str, Any]]:
    """Scan ``bg-logs/bg-pid-*.map`` files and return a dashboard-ready list.

    Each entry: {"pid", "alive", "command", "started", "log_path", "tail"}.
    Sorted newest-first. Used by the ``/bg`` REPL command.

    ``bg-logs`` is shared by EVERY nx session on the machine, so the default
    ``mine_only=True`` keeps a session's view (and ``/bg kill``'s pid set)
    scoped to the jobs this process launched — the same ``owner_pid`` filter
    the completion sweep applies (_bg_completions). A map written before the
    owner was recorded has no owner to match and is therefore foreign: with
    ``mine_only`` it is skipped, deliberately with no liveness fallback — the
    filter is exactly process-based, so it can never resurrect another
    session's job. Callers that genuinely want the machine-wide view (``/bg
    all``) pass ``mine_only=False``.
    """
    bg_dir = _bg_dir()
    legacy_dir = os.path.dirname(bg_dir)
    jobs = []
    mine = os.getpid()
    # per-project dir first; the legacy flat parent is scanned too (foreign
    # maps are still gated by owner_pid below, so this adds nothing foreign
    # to a mine_only view — it only keeps this session's pre-split jobs
    # visible until the legacy store drains)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for d in (bg_dir, legacy_dir):
        try:
            for n in os.listdir(d):
                if n.startswith("bg-pid-") and n.endswith(".map") and n not in seen:
                    seen.add(n)
                    found.append((n, os.path.join(d, n)))
        except OSError:
            continue
    for name, p in found:
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if mine_only and data.get("owner_pid") != mine:
            continue  # another session's job (or an owner-less legacy map)
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
            "cwd": data.get("cwd", ""),
            # C1: true once the hard-cancel ladder killed this job (/bg and
            # the live panel render it as `killed` instead of `exited`).
            "killed": bool(data.get("killed")),
            "tail": _tail_last_line(log_path) if log_path else "",
        })
    jobs.sort(key=lambda j: j.get("started") or 0, reverse=True)
    return jobs


def _delete_old_bg_logs(max_logs: int = 50, max_age_days: int = 7) -> None:
    """Trim background logs so they don't pile up forever.

    Also removes stale ``bg-pid-*.map`` files whose target log no longer exists.

    Trims the per-project dir and drains the legacy flat parent (0.145.2).
    """
    try:
        log_entries = []
        map_entries = []
        for bg_dir in (_bg_dir(), os.path.join(st.SESSION_HOME, "bg-logs")):
            try:
                names = os.listdir(bg_dir)
            except OSError:
                continue
            for name in names:
                if not name.startswith("bg-"):
                    continue
                p = os.path.join(bg_dir, name)
                try:
                    if os.path.isdir(p):
                        continue  # a sibling project's slug dir, never a log
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


# Grace between SIGTERM and SIGKILL on a hard cancel (UIUX_SPEC C1).
CANCEL_KILL_GRACE = _env_float("NX_CANCEL_KILL_GRACE", 2.0)


def _mark_job_killed(pid: int | str) -> None:
    """Stamp `"killed": true` into the job's bg-pid map so /bg lists it as
    killed (UIUX_SPEC C1: the ladder's kill is visible in the dashboard, not
    a vanished process). Best-effort: bookkeeping must never break the tool."""
    pid = str(pid)
    if not pid.isdigit():
        return
    try:
        p = _bg_pid_map_path(pid)
        with open(p) as f:
            data = json.load(f)
        data["killed"] = True
        with open_private(p) as f:
            json.dump(data, f)
    except (OSError, IOError, json.JSONDecodeError):
        pass


def cancel_kill_group(pid: int | str, command: str = "") -> None:
    """Hard-cancel a detached job: SIGTERM its process group, SIGKILL after
    CANCEL_KILL_GRACE seconds if it survives (UIUX_SPEC C1).

    Non-blocking: the escalation runs on the interrupt-watcher thread (double
    Esc) or inside a tool wait (Ctrl+C), and neither may stall on the grace
    period — so the SIGKILL follow-up rides a short-lived daemon thread that
    exits as soon as the group is gone (bounded: at most one grace period).
    The job runs in its own session (start_new_session), so killing the group
    takes its children down too — same reasoning as the NX_BASH_MAX_SECONDS
    watchdog. The job is marked killed in its map file for /bg."""
    _mark_job_killed(pid)
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return
    try:
        os.killpg(os.getpgid(pid_i), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # Group already gone (or not ours) — still log the cancellation.
        _log_event("tool", f"hard cancel: job pid {pid} already gone: {command[:120]}")
        return

    def _escalate() -> None:
        deadline = time.time() + CANCEL_KILL_GRACE
        while time.time() < deadline:
            if not _pid_is_alive(pid_i):
                break
            time.sleep(0.1)
        else:
            if _pid_is_alive(pid_i):
                try:
                    os.killpg(os.getpgid(pid_i), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        _log_event("tool", f"hard cancel killed bg job pid {pid}: {command[:120]}")

    threading.Thread(target=_escalate, daemon=True, name="nx-cancel-kill").start()


def _poll_pid(pid: int | str, timeout: float, check_esc: bool = False,
              log_path: str | None = None, expected_start_time: str | None = None) -> str:
    """Poll *pid* up to *timeout* seconds.

    Returns:
        "exited"      — process finished
        "timeout"     — poll window expired, process still alive
        "interrupted" — Esc was pressed (only if *check_esc* is True)
        "cancelled"   — the C1 hard-cancel ladder fired (sticky; checked on
                        every tick whether Esc armed us or not, so a Ctrl+C
                        from outside still ends an unwatched wait)

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
    # Esc is only meaningful on an interactive parent thread: a quiet thread
    # (async sub-agent, batch task) never owns the singleton stdin watcher, so
    # arming it here would fight the parent's arm/disarm handshake and emit
    # spurious "did not confirm input-mode restore" warnings. Poll without it.
    from ._subagent import _subagent_quiet
    if check_esc and _subagent_quiet():
        check_esc = False
    watch = _watch_for_interrupts() if check_esc else nullcontext()
    with watch:
        last_tail = None
        _last_label_at = time.monotonic()
        while time.time() < deadline:
            # C1: a hard cancel outranks the plain-Esc backgrounding — the
            # wait ends and the CALLER kills (it knows pid + command).
            if cancel_requested():
                return "cancelled"
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
                            # UIUX_SPEC B1: the log tail rides the spinner label
                            # through the one tail component (ANSI/control bytes
                            # stripped, whitespace collapsed, clamped — a raw
                            # command line would otherwise smear the row it
                            # lands on).
                            from ._render import stream_tail
                            spinner.label = stream_tail(tail, 60, plain=True)
                            _last_label_at = now
                    last_tail = tail
            # Wait between liveness probes on the interrupt event rather than
            # a blind sleep (PF-08): the loop must notice an Esc within one
            # poll interval, and this makes that wait event-driven (and still
            # correct when there is no watcher armed to fire it).
            if check_esc:
                _USER_INTERRUPTED.wait(0.1)
            else:
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


# Fail-fast ceiling (0.130.0): an explicit timeout= the model passes is capped
# here — values like 240 or 900 were seen to stall a whole turn on a command
# that was never going to finish. 0 disables the cap.
MAX_EXPLICIT_TIMEOUT = _env_int("NX_BASH_TIMEOUT_CAP", 120)


def _infer_timeout_from_sleep(command: str, default: int) -> int:
    """If *command* contains ``sleep N`` (with N in seconds, fractional seconds,
    or a time suffix — ``sleep 0.5``, ``sleep 2m``, ``sleep 1h30m``), return a poll
    timeout that comfortably covers it, so the command finishes synchronously
    without the model having to remember ``timeout=N``.

    This compensates for the natural habit of writing
    ``run_bash("sleep 30 && cat …")`` — a pattern that *seems* like it should
    block (the shell waits, then cats) but is itself launched detached, so
    it gets backgrounded after ~3 s just like anything else.  By recognising
    the sleep we make the "obvious" pattern work again.

    The inferred window is capped at NX_BASH_SLEEP_CAP seconds (default 120):
    a sleep beyond the cap no longer blocks the turn — the command backgrounds
    after the capped window and the caller gets a PID + log path, which is the
    correct shape for an hour-long wait anyway.
    """
    cap = _env_int("NX_BASH_SLEEP_CAP", 120)
    if cap <= 0:
        # 0/negative = off: restore the old unbounded window (escape hatch for
        # someone who really wants `sleep 1d && …` to block for a day).
        cap = 999999999
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
    # A `sleep` can also legitimately start a line after a newline or open a
    # subshell — `cmd\nsleep 30 && x`, `(sleep 30 && x)` — and those used to
    # fall through to the 3 s default and background. Treat newline and `(` as
    # boundaries too; `\s*` after them spans the following blank lines.
    for m in re.finditer(r'(?:^|[;&|(]|\n)\s*sleep\s+([^\s;&|]+)', command):
        n = _sleep_seconds(m.group(1))
        if n > max_sleep:
            max_sleep = n
    if max_sleep > 0:
        return int(min(max_sleep, cap)) + 10
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

# 0.147.0: commands that DUMP the environment or credential files. Matched
# only when NX_BASH_PROMPT_SECRETS is on (same opt-in as _SECRET_PATTERNS —
# the model legitimately needs env dumps for debugging, so this is an
# approval gate, not a refusal).
_SECRET_ENV_DUMPS = (
    re.compile(r"(?:^|[;&|]\s*)(?:printenv|env|set|export)(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)(?:cat|less|more|head|tail|grep)\s+(?:[^;&|]*\s)?\.env\b"),
    re.compile(r"(?:^|[;&|]\s*)(?:cat|less|more|head|tail)\s+[^;&|]*(?:credentials|\.netrc|\.aws/)"),
)


def _contains_secret(command: str, env_dumps: bool = True) -> str | None:
    """Return the name of the first secret pattern the command matches, or None.

    env_dumps=False restricts the scan to content patterns (tokens, PEM
    blocks, ...) — used by the commit guard, which scans DIFF TEXT, not a
    shell command: a diff line reading "env" or "| set " is data, not an
    environment dump (0.152.1).
    """
    for rx in _SECRET_PATTERNS:
        m = rx.search(command)
        if m:
            return m.group(0)[:8] + "…"
    if env_dumps:
        for rx in _SECRET_ENV_DUMPS:
            if rx.search(command):
                return "environment/credential dump"
    return None


def run_bash(command: str, timeout: int | None = None,
             _cwd: str | None = None, **_: Any) -> str:
    """Run a shell command.

    Commands always run detached in their own session — they never block the agent loop.
    If the command finishes within a short poll window (~3 s by default),
    output is returned directly and the log file is deleted.  If it's still
    running after the poll window, it continues in the background and you
    get a PID + log path to check later with ``read_file``.

    **Do NOT use** ``sleep N && …`` inside the command string as a way to
    wait — the entire command is launched detached in its own session so *it* will also
    be backgrounded after a few seconds.  Instead, pass ``timeout=N`` to tell
    nx to wait synchronously (e.g. ``timeout=35`` for a ``sleep 30``
    command).  As a convenience, if your command *does* start with
    ``sleep N …`` the tool auto-extends its poll window to ``N+10`` s
    so the "obvious" pattern still works (capped at NX_BASH_SLEEP_CAP,
    default 120 s — a longer sleep just backgrounds with a log path).

    Use ``timeout=0`` to wait *indefinitely* (the tool blocks until the
    command completes).  To wait for an **already-backgrounded** command,
    use ``wait_background(pid=…)``.

    Long-running commands (servers, watchers, builds) produce background logs
    at ``~/.nx/bg-logs/``, automatically cleaned up (max ~50 files /
    7 days).  Pressing Esc during the poll window immediately backgrounds
    the command.

    NX_BASH_MAX_SECONDS (default 1800) is a watchdog on *backgrounded*
    commands: after the poll window expires and the process keeps running, nx
    reaps it at N seconds (SIGTERM, then SIGKILL) instead of leaving it to run
    forever. Set NX_BASH_MAX_SECONDS=0 to allow
    servers/builds that SHOULD outlive the tool call to run unbounded.

    With NX_BASH_PROMPT_SECRETS=1, a command that looks like it carries a
    credential (sk-…, ghp_…, AKIA…, a PEM block, …) or that dumps the
    environment / a credential file (printenv, env, cat .env, …) always
    prompts, even in YOLO mode -- the prompt is the last chance to catch a
    secret before it lands in a log, a file, or a remote host.

    NX_BASH_SLEEP_CAP (default 120) bounds the sleep-inferred poll window so
    "sleep 10m && cmd" cannot block the turn; 0 disables the cap.
    """
    trap = _heredoc_quote_trap(command) or _heredoc_terminator_suffix_trap(command)
    if trap:
        return trap
    # Harness-side upgrade (0.64.2): the model sometimes writes a grep-family
    # command inside run_bash even when ripgrep is installed. nx knows whether
    # rg is on PATH, so rewrite before approval, timeout inference, or the
    # secret scan see it — the user approves (and the log records) what will
    # actually run.
    command = prefer_ripgrep(command)
    secret = _contains_secret(command) if _env_int("NX_BASH_PROMPT_SECRETS", 0) else None
    staged = None
    if secret is not None:
        from ._subagent import _subagent_quiet
        if _subagent_quiet():
            # A detached agent can't show the secret-guard prompt — fail the
            # call instead of prompting from a background thread.
            return (f"ERROR: command looks like it carries a credential ({secret!r}); "
                    "refused in a background sub-agent — let the parent run it")
        # force the prompt even under YOLO: bypass the level/YOLO short-circuit
        # with a direct ask (Esc still drops to chat)
        import nx as _m
        from ._approval import _paused_spinner
        with _paused_spinner():
            choice = _m._ask_approval(
                f"  run (contains possible secret {secret!r}): {command[:120]}? "
                f"[Y/n/esc] ", allow_hint=None)
        if choice == "esc":
            from ._approval import _EscToChat
            raise _EscToChat(f"run: {command}")
        if choice != "y":
            return "DENIED by user"
    else:
        # Clobber guard (0.104.2): a command that would TRUNCATE an existing,
        # non-empty file forces its own prompt even in YOLO — a truncated
        # dotfile is unrecoverable in a way a mere failed command is not.
        # Precedent: the secret guard above. NX_CLOBBER_GUARD=0 disables.
        from ._clobber import clobber_targets, clobber_guard_message
        clobber = clobber_targets(command) or None
        if clobber:
            from ._subagent import _subagent_quiet
            if _subagent_quiet():
                return (f"ERROR: command would truncate existing file(s) "
                        f"{', '.join(clobber[:3])}; refused in a background "
                        "sub-agent — let the parent run it")
            import nx as _m
            from ._approval import _paused_spinner
            with _paused_spinner():
                choice = _m._ask_approval(
                    clobber_guard_message(clobber), allow_hint=None)
            if choice == "esc":
                from ._approval import _EscToChat
                raise _EscToChat(f"run: {command}")
            if choice != "y":
                return "DENIED by user"
            from ._clobberclaim import mark_clobbered
            mark_clobbered(clobber)  # one Y clears this session's re-asks (0.125.1)
        # Git-index guard: a raw `git add`/`git commit`/`git reset`/… while the
        # SHARED index holds staged entries sweeps whatever is staged — maybe
        # ANOTHER session's work — into its own result (the incident a private
        # GIT_INDEX_FILE in git_commit prevents). Same shape as the clobber
        # guard above: its own prompt even in YOLO, refuse in a quiet
        # sub-agent. NX_GIT_INDEX_GUARD=0 disables.
        from ._clobber import git_index_guard, git_index_guard_message
        staged = git_index_guard(command, cwd=_cwd) or None
        if staged:
            from ._subagent import _subagent_quiet
            if _subagent_quiet():
                return (f"ERROR: command mutates the git index while "
                        f"{', '.join(staged[:3])} "
                        f"{'is' if len(staged) == 1 else 'are'} staged in the "
                        "shared index (possibly another session's work); "
                        "refused in a background sub-agent — let the parent "
                        "run it")
            import nx as _m
            from ._approval import _paused_spinner
            with _paused_spinner():
                choice = _m._ask_approval(
                    git_index_guard_message(command, staged), allow_hint=None)
            if choice == "esc":
                from ._approval import _EscToChat
                raise _EscToChat(f"run: {command}")
            if choice != "y":
                return "DENIED by user"
    if (not secret and clobber is None and staged is None
            and not _confirm(f"run: {command}")):
        return "DENIED by user"
    if timeout is None:
        timeout = _env_int("NX_BASH_POLL_SECONDS", 3)
        timeout = _infer_timeout_from_sleep(command, timeout)
    if timeout > 0 and MAX_EXPLICIT_TIMEOUT > 0:
        timeout = min(timeout, MAX_EXPLICIT_TIMEOUT)
    if timeout <= 0:
        timeout = 999999999  # ∞ — wait indefinitely
    max_seconds = _env_int("NX_BASH_MAX_SECONDS", 1800)
    # Staged-path attribution (0.125.0): a raw `git add`/`git commit -a` here
    # stages work this session OWNS, and every later git op of this session
    # would otherwise prompt on it. Snapshot the shared index pre-run and claim
    # whatever the command newly staged, so git_index_guard subtracts it from
    # then on. Only a mutating op pays the extra `git status`; a command that
    # gets backgrounded never reaches the claim (conservative: a prompt).
    claim_cwd: str | None = None
    pre_staged: set[str] | None = None
    try:
        from ._clobber import git_mutating_index_op, staged_paths_abs
        hit = git_mutating_index_op(command)
        if hit is not None:
            claim_cwd = hit[1] or _cwd
            pre_staged = set(staged_paths_abs(claim_cwd))
    except Exception:
        pre_staged = None  # attribution is best-effort; the guard is not
    pid, log_path = _run_bash(command, _cwd=_cwd)
    # C1: publish the kill hook for the hard-cancel ladder for exactly the
    # duration of the wait (cleared in the finally — the slot can never point
    # at a stale pid between tools).
    register_tool_cancel(lambda: cancel_kill_group(pid, command))
    try:
        status = _poll_pid(pid, timeout, check_esc=True, log_path=log_path,
                           expected_start_time=_find_bg_start_time_for_pid(pid))
    except KeyboardInterrupt:
        # C1: Ctrl+C during the poll window is a hard cancel, not a turn
        # unwind — convert it HERE so the hook (and the kill below) runs while
        # the pid is still ours to kill. The model gets the cancelled result,
        # and the sticky flag makes the dispatch boundary skip the rest of
        # the tool group.
        from ._interrupt import request_hard_cancel
        request_hard_cancel("Ctrl+C in run_bash")
        status = "cancelled"
    finally:
        clear_tool_cancel()
    if status == "cancelled":
        cancel_kill_group(pid, command)
        _mark_bg_notice_seen(pid)
        _log_event("tool", f"run_bash hard-cancelled by user (pid {pid}): {command[:120]}")
        return (f"[cancelled] PID {pid} was killed (SIGTERM, then SIGKILL after "
                f"{CANCEL_KILL_GRACE:g}s). Partial output: {log_path}")
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
    # Finished inside the poll window: the result below carries the command's
    # FULL output + exit code. Mark it seen so the completion scan never
    # announces "pid N exited (code unknown)" for work the model just saw —
    # the log is about to be deleted, leaving a dead pid + map file that the
    # scan (and the wake poller) would otherwise report once per turn forever.
    _mark_bg_notice_seen(pid)
    if pre_staged is not None:
        _claim_newly_staged(pre_staged, claim_cwd)
    return _finish_bg_job(log_path)


def _claim_newly_staged(pre: set[str], cwd: str | None) -> None:
    """Claim the paths a just-finished raw git op NEWLY staged into the shared
    index (post-run minus pre-run) as this session's own staged work, so
    `_clobber.git_index_guard` stops prompting about them — and still prompts
    for any path this session never staged. The diff is what keeps it
    conservative: staged work that predates the command is never claimed by
    it, so another session's entry stays attributable. Best-effort."""
    try:
        from ._clobber import staged_paths_abs
        from ._stagedclaim import mark_staged
        fresh = set(staged_paths_abs(cwd)) - pre
        if fresh:
            mark_staged(fresh)
    except Exception:
        pass  # bookkeeping must never break the tool result


# PIDs this process has handed a watchdog thread (avoid duplicates when the
# same pid is reported backgrounded twice — shouldn't happen, but cheap).
_WATCHDOG_PIDS: set[int] = set()


def _maybe_start_watchdog(pid: int | str, command: str, max_seconds: int) -> None:
    """Arm the NX_BASH_MAX_SECONDS reaper for a just-backgrounded job.

    A daemon thread sleeps out the remaining budget (poll-window seconds are
    already spent by the time we get here — the watchdog counts from process
    start as closely as we can observe), then SIGTERMs and, after a grace
    period, SIGKILLs the process group. The job runs in its own session
    (start_new_session), so killing the group takes down any children it
    spawned — which is what you want from a ceiling: nothing survives to keep
    consuming the machine.

    NX_WATCHDOG_POLL_SECONDS (default 0.5) is the _reap poll granularity: how
    often the reaper checks whether the job died on its own before the budget
    elapses, and how often it re-checks after SIGTERM (bounded by
    NX_WATCHDOG_KILL_GRACE, default 5s, before escalating to SIGKILL).
    Finer = a shorter tail after early-exit; coarser = fewer wakeups.
    Tests shrink it so background-job tests don't pay the default tail.
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
            remaining = max(1, max_seconds)
            poll = _env_float("NX_WATCHDOG_POLL_SECONDS", 0.5)
            # Poll instead of one long sleep so a job that exits early doesn't
            # leave its pid parked in _WATCHDOG_PIDS until the full budget
            # elapses (tests poll that set).
            while remaining > 0 and _pid_is_alive(pid_i, start_time):
                step = min(poll, remaining)
                time.sleep(step)
                remaining -= step
            if not _pid_is_alive(pid_i, start_time):
                return
            try:
                os.killpg(os.getpgid(pid_i), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                return
            grace = _env_float("NX_WATCHDOG_KILL_GRACE", 5.0)
            deadline = time.time() + grace
            while time.time() < deadline:
                if not _pid_is_alive(pid_i, start_time):
                    break
                time.sleep(min(grace, poll))
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


def _mark_bg_notice_seen(pid: int | str | None) -> None:
    """Record a background job's pid as already delivered to the model.

    The completion scan (nx/_dispatch.py) announces finished jobs the model
    hasn't heard about yet; but when nx itself handed the job's output to the
    model in a tool result — run_bash finishing inside its own poll window
    (inline `[exit N]` result), or wait_background returning the full log —
    the transcript already contains that work, and announcing "pid N exited"
    afterwards only produces a pointless wake turn ("Nothing to do — that was
    the sweep script from my last turn exiting").

    Without this, every short-lived bash command leaks exactly such an
    orphan: _finish_bg_job deletes the LOG but leaves the pid->log MAP file,
    so the next scan (or wake_due's sweep) finds a dead pid, no log, and no
    seen-entry — "exited (code unknown)" — once per turn, forever.

    Locks the same mutex as the scan so a concurrent poller sweep can't
    drop one side's marking (the scan rewrites st._BG_NOTICE_SEEN in full).
    """
    if not pid:
        return
    pid = str(pid)
    if not pid.isdigit():
        return
    try:
        from ._dispatch import _BG_SCAN_LOCK
        with _BG_SCAN_LOCK:
            seen = getattr(st, "_BG_NOTICE_SEEN", None)
            if seen is None:
                seen = set(getattr(st, "_BG_NOTICE_BASELINE", set()) or set())
            seen.add(pid)
            st._BG_NOTICE_SEEN = seen
    except Exception:
        pass  # a seen-marking hint must never break the tool result


def _mark_log_consumed(log_path: str | None) -> None:
    """read_file() just returned *log_path* in full: if it is a background
    job's log and the job has already exited, its output now sits entirely in
    the transcript — mark the job seen so the completion scan never repeats
    it. A still-running job stays unmarked: its eventual exit is still news
    worth announcing."""
    if not log_path:
        return
    bg_dir = os.path.dirname(os.path.abspath(log_path))
    if bg_dir not in (_bg_dir(), os.path.join(st.SESSION_HOME, "bg-logs")):
        return
    try:
        for name in os.listdir(bg_dir):
            if not (name.startswith("bg-pid-") and name.endswith(".map")):
                continue
            pid = name[len("bg-pid-"):-len(".map")]
            try:
                with open(_bg_pid_map_path(pid)) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("log_path") != log_path:
                continue
            if _pid_is_alive(int(pid), data.get("pid_start_time")):
                continue  # still running — the exit will be the real notice
            _mark_bg_notice_seen(pid)
            return
    except Exception:
        pass  # a seen-marking hint must never break the tool result


def _wait_for_log_trailer(
    log_path: str, timeout: float = 0.2, interval: float = 0.005,
) -> int | None:
    """Poll *log_path* until the trailing [exit: N] marker the launcher's
    subshell writes as its LAST action is visible (so the output before it is
    flushed too). Returns the exit code once seen, or None if the marker never
    appeared within *timeout* (truncated/corrupt log, or a kill that skipped
    the trailer).

    PF-08 (0.56.0): this replaces a blind fixed settle sleep in
    ``_finish_bg_job``. The subshell writes the trailer after all command
    output and the redirection flushes in order, so "marker present on disk"
    IS the real completion signal: waiting on it lands the common case in a few
    ms instead of a flat 200ms, while never returning before the output is
    actually there.
    """
    deadline = time.monotonic() + timeout
    while True:
        _out, exit_code = _read_log(log_path)
        if exit_code is not None:
            return exit_code
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def _finish_bg_job(log_path: str) -> str:
    """Read+format the output of a finished background job, deleting its log."""
    # Wait on the actual artifact (the flushed exit trailer) instead of a fixed
    # settle sleep (PF-08). If the trailer never lands, _read_log still returns
    # whatever is on disk, so a truncated log degrades to prior behaviour
    # rather than hanging.
    _wait_for_log_trailer(log_path)
    out, exit_code = _read_log(log_path)
    try:
        os.remove(log_path)
    except OSError:
        pass
    if exit_code is not None:
        return f"[exit {exit_code}]\n{out}"
    return f"[exited]\n{out}"


def kill_background(pid: int | str, **_: Any) -> str:
    """SIGTERM a background job THIS session started (model-facing /bg kill).

    The killable set is deliberately narrow: only PIDs this process launched
    (via list_bg_jobs(mine_only=True)). bg-logs is shared by every nx session
    on the machine, so an unfiltered pid would let one session terminate
    another session's work."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return "[error] kill_background: pid must be the integer PID run_bash returned."
    known = {j["pid"] for j in list_bg_jobs()}
    if str(pid) not in known:
        if str(pid) in {j["pid"] for j in list_bg_jobs(mine_only=False)}:
            return (f"[error] PID {pid} is another session's job — only jobs "
                    "this session started can be killed.")
        return f"[error] PID {pid} is not a tracked background job of this session."
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return f"[error] no such process {pid} (already exited?)."
    except (PermissionError, OSError) as e:
        return f"[error] SIGTERM to {pid} failed: {e}"
    return f"Sent SIGTERM to PID {pid}. Its log remains in bg-logs for the reaper."


# Fail-fast ceiling (0.130.0) for wait_background's indefinite default.
WAIT_CEILING = _env_int("NX_WAIT_BACKGROUND_CEILING", 1800)
# Cap on an *explicit* wait_background timeout= (0.144.x): mirrors
# MAX_EXPLICIT_TIMEOUT for run_bash. Models that pass timeout=600 to a wait
# turn a poll into a ten-minute blind block; the whole point of a wait is to
# come back quickly with either output or a log path to read_file.
WAIT_TIMEOUT_CAP = _env_int("NX_WAIT_TIMEOUT_CAP", 120)


def wait_background(pid: int | str, log_path: str | None = None, timeout: int = 0, **_: Any) -> str:
    """Wait for a backgrounded command (started by ``run_bash``) to finish.

    Polls *pid* until it exits (default: wait indefinitely).  When the command
    exits the output is returned and the log file is cleaned up.  Esc can be
    pressed at any time to interrupt the wait — the command keeps running and
    the log path is returned so you can check back later.

    Pass ``timeout=N`` only if you specifically want to bail after N seconds
    (rarely useful — you called this because you want to *wait*, not guess a
    ceiling). An explicit N is capped (NX_WAIT_TIMEOUT_CAP, default 120s) so
    a wait can never become a long blind block; polling briefly and coming
    back later is always cheaper.

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
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        # Don't let a malformed pid turn into an unhandled ValueError; a
        # garbage pid resolves to no map file below anyway, but say why.
        return "[error] wait_background: pid must be the integer PID run_bash returned."
    # Never trust a caller-supplied log_path: always resolve it ourselves from
    # the pid->log map written by _run_bash, so a model-supplied path/pid pair
    # can't be used to read arbitrary files outside the run_bash pipeline.
    log_path = _find_bg_log_for_pid(pid)
    if not log_path:
        return (f"[error] No background log found for PID {pid}. "
                "The log may have been cleaned up already.")
    timeout_s = timeout if timeout > 0 else WAIT_CEILING
    # Explicit timeout= gets the fail-fast cap, exactly like run_bash's
    # timeout= (the indefinite default is still bounded by WAIT_CEILING).
    if timeout > 0 and WAIT_TIMEOUT_CAP > 0:
        timeout_s = min(timeout_s, WAIT_TIMEOUT_CAP)
    if WAIT_CEILING > 0:
        timeout_s = min(timeout_s, WAIT_CEILING)
    # C1: same cancel-hook contract as run_bash — waiting on a bg job is a
    # cancellable wait, and cancelling KILLS the job (the user asked for it).
    _wcmd = ""
    try:
        with open(_bg_pid_map_path(pid)) as f:
            _wcmd = json.load(f).get("command", "")
    except (OSError, IOError, json.JSONDecodeError):
        pass
    register_tool_cancel(lambda: cancel_kill_group(pid, _wcmd))
    try:
        status = _poll_pid(pid, timeout_s, check_esc=True, log_path=log_path,
                           expected_start_time=_find_bg_start_time_for_pid(pid))
    except KeyboardInterrupt:
        from ._interrupt import request_hard_cancel
        request_hard_cancel("Ctrl+C in wait_background")
        status = "cancelled"  # C1: Ctrl+C hard-cancels the wait (see run_bash)
    finally:
        clear_tool_cancel()
    if status == "cancelled":
        cancel_kill_group(pid, _wcmd)
        _mark_bg_notice_seen(pid)
        _log_event("tool", f"wait_background hard-cancelled (pid {pid})")
        return (f"[cancelled] PID {pid} was killed (SIGTERM, then SIGKILL after "
                f"{CANCEL_KILL_GRACE:g}s). Partial output: {log_path}")
    if status == "interrupted":
        return (f"[interrupted] PID {pid} still running. "
                f"Check later with read_file('{os.path.abspath(log_path)}')")
    if status == "timeout":
        return (f"[timeout after {timeout_s:g}s] PID {pid} still running. "
                f"Check output so far with read_file('{os.path.abspath(log_path)}')")
    # The full output is about to land in the transcript — same as an inline
    # run_bash finish, so the completion scan must never announce this pid.
    _mark_bg_notice_seen(pid)
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
    pid = str(pid)
    if not pid.isdigit():
        return None
    try:
        with open(_bg_pid_map_path(pid)) as f:
            return json.load(f).get("pid_start_time")
    except (OSError, IOError, json.JSONDecodeError):
        return None


def _find_bg_log_for_pid(pid: int | str) -> str | None:
    """Look up the log path for a background PID via the pid→log map files
    written by ``_run_bash``.  Returns the log path, or None if not found.

    Deliberately no content-scanning fallback ("does some log mention this
    pid in its first 1K?"): a pid substring matches plenty of unrelated
    output, so a model-supplied pid could be resolved to *another* job's log
    and wait_background would then return that other command's output as
    authoritative. A pid without a map file simply has no log.
    """
    pid = str(pid)
    if not pid.isdigit():
        return None
    try:
        with open(_bg_pid_map_path(pid)) as f:
            return json.load(f).get("log_path")
    except (OSError, IOError, json.JSONDecodeError):
        return None
