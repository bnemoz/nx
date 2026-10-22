"""PTC Phase 2 (docs/ptc-migration-spec.md §Phase 2 + Audit): `run_tools(code)` —
the execution engine that turns the Phase 1 stub module into a live call path.

Design (settled; the Audit section is authoritative where the spec body is not):

* The model's Python runs in a **subprocess** (`sys.executable -I -X utf8`,
  argv-only, `start_new_session=True` per the `_bash_jobs.py:235` precedent so
  Esc can kill the whole process group without touching the terminal). The
  rendered stub module is written to a temp dir next to a generated launcher
  and is importable as `stubs`. `-I` implies `-E -s`: no PYTHONPATH, no user
  site, and the script dir is inserted explicitly by the launcher (isolated
  mode does not prepend it).

* **Pipe protocol** (NDJSON, harness-side reader loop): the launcher's
  `_capture(name, args)` sends `{id, name, args}` frames on the child's
  stdout and blocks reading a response line; the harness answers
  `{id, result}` or `{id, error}`. Every routed call goes through
  `run_tool(name, args)` (`nx/_dispatch.py:201`) **unmodified** — plan gate,
  withheld refusals, breaker check, arg repair, write fence, hooks, cache,
  undo snapshot and sanitize all apply per-call exactly as in JSON mode.
  Results therefore pass through `run_tool`'s sanitize exactly once (the
  harness never re-sanitizes what `run_tool` returned).

* **Allowlist** (Audit correction #2): stub-callable tools are the explicit
  INTERSECTION `_dispatch._CACHEABLE_READ_TOOLS ∩ _plan.PLAN_READONLY_TOOLS`
  — the six pure reads. Not the union: PLAN_READONLY_TOOLS contains
  spawn_agent/todo_write/plan_edit which must NOT be script-callable. Any
  other name over the pipe gets a standard refusal string (no dispatch).
  Approval-needing calls are refused by the allowlist itself; `_EscToChat`
  escaping a deeper prompt is additionally handled as a kill+skip-fill
  (turn-layer parity below).

* **Sandbox honesty** (Audit, corrected 0.168.x — the previous revision of
  this bullet claimed "no unshare/seccomp is added", which has been false
  since 0.165.x; `_ptc_preexec` below is the source of truth). The
  subprocess is untrusted but capability-poor, and the boundary is TWO
  layers deep:

  1. **Logical** (always present): the stub allowlist (six pure reads) plus
     the write fence inside `run_tool`, plus a scrubbed env at the `Popen`
     site (only PATH/HOME/LANG/PYTHONIOENCODING cross over, so API keys are
     not in the child's `os.environ`).
  2. **OS-level, best-effort** in `_ptc_preexec`, installed as `preexec_fn`:
     seccomp deny for the socket/connect family (libseccomp when present),
     Landlock filesystem lockdown (read+execute under `/`, read+write only
     under the script's workdir, /dev writable + /proc read), and prctl
     NO_NEW_PRIVS — see that function's docstring for the exact grant set.

  Every hardening layer is `try/except Exception: pass` and `# pragma: no
  cover`: on macOS, on kernels without Landlock (< 5.13 or LSM disabled),
  with no libseccomp, or in a container that blocks those syscalls, **the
  layer is silently absent** and the child degrades to the 0.165.2 posture
  (layer 1 only). No status line reports which layers landed, so OS
  isolation is opportunistic defense-in-depth, never the guarantee — the
  guarantee remains layer 1: the allowlist plus the write fence inside
  `run_tool`.

* **Turn-layer parity**: per-call failures come back as `"ERROR: ..."`
  result strings inside the script (never raise across the pipe); a
  script-level exception, a timeout, or a protocol violation folds into ONE
  error result for `run_tools` plus the same malformed-args breaker
  increment `_turn_toolcalls.py:200-216` applies to a malformed JSON call.
  Esc/`KeyboardInterrupt` kills the process group and every unanswered
  in-flight `_capture` is skip-filled with an error frame before the kill
  (the launcher unblocks and exits; the transcript for the `run_tools` call
  itself is written by the JSON-mode skip-fill loop around it — parity with
  `_turn_toolcalls.py:269-327`).

* **Phase 3 shape**: the reader thread only moves frames into a queue; the
  serial dispatch loop is one function (`_dispatch_pipe_call`). A
  `ThreadPoolExecutor` can replace the serial loop later without touching the
  protocol or the dispatcher (writes are already lock-guarded).
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Any

from . import _state as st
from ._env import _env_int


def _ptc_preexec(workdir: str = "") -> None:  # pragma: no cover — forked child
    """Best-effort OS hardening for the PTC child, run between fork and exec
    (`preexec_fn`). Three layers, all optional and silent on failure:

    1. seccomp deny for the socket/connect family (libseccomp when present):
       network egress is the exfiltration channel the stub allowlist cannot
       govern, since the script may import socket/urllib directly. SCMP_ACT_ERRNO
       (EACCES) rather than SIGKILL so a stray call raises a catchable Python
       error instead of nuking the interpreter.
    2. Landlock filesystem lockdown (kernel >= 5.13 with the LSM enabled):
       a ruleset that handles EVERY fs right, grants read+execute beneath `/`
       and read+write beneath the script's own workdir (and /dev, which the
       interpreter may need for /dev/urandom) — every write outside the
       workdir (project tree, home, /etc) returns EACCES. This closes the
       documented 0.165.2 limit: the child could previously write anywhere
       its user could.
    3. prctl NO_NEW_PRIVS — also required by landlock_restrict_self; the
       child never gains privileges even if something execs a setuid binary.

    The child therefore cannot: reach the network, write outside its tempdir,
    or gain privileges. It CAN: read (project files via the stub reads, its
    own interpreter), execute, and talk the pipe protocol over inherited fds.
    Failures are tolerated (macOS, old kernels, containers that block
    seccomp/landlock syscalls) so the feature degrades to the 0.165.2
    posture rather than breaking."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        libc.syscall.argtypes = [ctypes.c_long] * 7
        PR_SET_NO_NEW_PRIVS = 38
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        _ptc_landlock(libc, workdir)
        _ptc_seccomp_net_deny(libc, ctypes.util)
    except Exception:  # noqa: BLE001 — hardening is best-effort, never fatal
        pass


def _ptc_landlock(libc, workdir: str) -> None:  # pragma: no cover
    """Layer 2: every fs right handled; read+execute beneath `/`, read+write
    beneath `workdir` (and /dev for /dev/urandom, /proc read for runtime).
    Unprivileged; silently skipped when the kernel lacks Landlock (ABI probe
    returns < 0) or a rule cannot be added."""
    import struct
    import ctypes  # noqa: F401 — _addr uses it; keep the layer self-contained
    SYS_CREATE_RULESET, SYS_ADD_RULE, SYS_RESTRICT_SELF = 444, 445, 446
    RULE_PATH_BENEATH = 1
    (EXECUTE, WRITE_FILE, READ_FILE, READ_DIR, REMOVE_DIR, REMOVE_FILE,
     MAKE_CHAR, MAKE_DIR, MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK,
     MAKE_SYM, REFER, TRUNCATE, IOCTL_DEV) = (1 << i for i in range(16))
    READX = READ_FILE | READ_DIR | EXECUTE
    WRITE_ALL = (WRITE_FILE | MAKE_REG | MAKE_DIR | REMOVE_DIR | REMOVE_FILE
                 | MAKE_CHAR | MAKE_SOCK | MAKE_FIFO | MAKE_BLOCK | MAKE_SYM
                 | REFER | TRUNCATE | IOCTL_DEV)
    attr = struct.pack("<Q", 0xFFFF)  # handle every fs right (ABI 5+)
    rs_fd = libc.syscall(SYS_CREATE_RULESET, _addr(attr),
                         len(attr), 0, 0, 0, 0)
    if not isinstance(rs_fd, int) or rs_fd < 0:
        return  # pre-5.13 kernel or LSM disabled — degrade silently

    def grant(path: str, access: int) -> bool:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = struct.pack("<Qq", access, fd)
            return libc.syscall(SYS_ADD_RULE, rs_fd, RULE_PATH_BENEATH,
                                _addr(rule), 0, 0, 0) == 0
        finally:
            os.close(fd)

    ok = grant("/", READX)
    if workdir:
        ok = grant(workdir, READX | WRITE_ALL) and ok
    # /dev (urandom) and /proc (runtime self-stats): read, plus write under
    # /dev since some environments create pipes/ptys there at startup.
    ok = grant("/dev", READX | WRITE_ALL) and ok
    ok = grant("/proc", READX | WRITE_FILE | TRUNCATE | MAKE_REG) and ok
    if not ok:
        return  # partial ruleset is worse than none — degrade to no Landlock
    libc.syscall(SYS_RESTRICT_SELF, rs_fd, 0, 0, 0, 0, 0)
    # fd is closed by exec or process exit; no leak either way.


def _addr(b: bytes) -> int:  # pragma: no cover — trivial helper
    import ctypes
    return ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value


def _ptc_seccomp_net_deny(libc, ctypes_util) -> None:  # pragma: no cover
    """Layer 1: SCMP_ACT_ERRNO(EACCES) for the socket/connect family via
    libseccomp (when present). Kept separate from the Landlock layer so each
    degrades independently."""
    import ctypes
    seccomp_name = ctypes_util.find_library("seccomp")
    if not seccomp_name:
        return
    sc = ctypes.CDLL(seccomp_name, use_errno=True)
    SCMP_ACT_ALLOW = 0x7FFF0000
    SCMP_ACT_ERRNO = 0x00050000
    EACCES = 13
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_rule_add.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint)
    sc.seccomp_syscall_resolve_name.restype = ctypes.c_int
    ctx = sc.seccomp_init(SCMP_ACT_ALLOW)
    if not ctx:
        return
    # Anything that can create a network endpoint or ship bytes out. The pipe
    # protocol uses os.read/os.write on inherited fds, not socketpair, so
    # denying socket() entirely is protocol-safe.
    for call in ("socket", "socketpair", "connect", "accept", "accept4",
                 "bind", "listen", "sendto", "sendmsg", "sendmmsg",
                 "recvfrom", "recvmsg", "getsockopt", "setsockopt"):
        nr = sc.seccomp_syscall_resolve_name(call.encode())
        if nr < 0:
            continue  # not on this kernel
        sc.seccomp_rule_add(ctx, SCMP_ACT_ERRNO | EACCES, nr, 0)
    sc.seccomp_load(ctypes.c_void_p(ctx))

RUN_TOOLS_TOOL = {
    "type": "function",
    "function": {
        "name": "run_tools",
        "description": "PTC: run a Python `code` script in an isolated subprocess with the session's read tools importable as `import stubs` (stubs.read_file(path), stubs.search_files(pattern), ... — see the [Runtime note] carrying the stub module source). Read-only pipe: any other tool name (writes, run_bash, spawn_agent) is refused over the pipe; single tool calls stay as normal JSON tool calls. The script is unsandboxed at the OS level — the allowlist governs the stub pipe, not the Python language. Returns the script's captured stdout; a script exception or timeout is one error result.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source; `import stubs`, call stubs.<tool>(...) for >=2 independent READS in one turn (loops/filters over many files; results come back as strings inside your script — print what you want back in context). The script is untrusted code run with -I; the read-only allowlist is the pipe boundary (the script itself is unsandboxed at the OS level, so its own prints — not tool results — are what to review)"},
                "timeout": {"type": "integer", "description": "Wall-clock seconds (default 0 = session default NX_PTC_TIMEOUT, 30; hard cap NX_PTC_TIMEOUT_MAX, 120)"},
            },
            "required": ["code"],
        },
    },
}


def _ptc_enabled() -> bool:
    """True when the run_tools schema is being offered this session. Delegates
    to the Phase 4 resolver (nx/_ptc_rollout._ptc_enabled): env hard override,
    then the fallback ladder, then the nx.NX_PTC attribute (tests/REPL flip
    it) — so a mid-session os.environ change cannot desync the gate and the
    prompt surface."""
    from ._ptc_rollout import _ptc_enabled as _resolve
    return _resolve()


# --- prompt surface (spec §Prompt wiring; Audit-corrected) ---------------------
# Session-start PTC note, `_delegation_note.py` style: static text (so riding
# the pinned prefix is safe — pin_prefix only breaks when head BYTES change,
# and these bytes are per-session constants), gate-gated like the delegation
# note. The DYNAMIC part (the stub-module source) never rides the prefix: it
# ships as a one-time [Runtime note] tail (see _stage_stub_source).
PTC_NOTE_MARKER = "[PTC note:"
PTC_NOTE = (
    PTC_NOTE_MARKER + " programmatic tool calling is enabled this session: "
    "the run_tools tool runs a Python script with the session's read tools "
    "importable as `import stubs` (stubs.read_file(...), "
    "stubs.search_files(...), stubs.memory_search(...), "
    "stubs.recall_lessons(...) — each returns the tool result string; your "
    "script's printed output is what re-enters context).\n"
    "- Use it for \u2265\u0032 independent reads in one turn (loops, filters, cross-file "
    "aggregation); keep single calls as ordinary JSON tool calls.\n"
    "- The pipe is read-only: writes, run_bash, spawn_agent and every other "
    "tool stay normal JSON tool calls; a script that calls one gets a refusal "
    "string.\n"
    "- The exact stub-module source arrives once as a [Runtime note] on your "
    "first tool call — read it before writing a script.]"
)


def _load_ptc_note() -> dict[str, str] | None:
    """User-role session-start PTC note, or None when run_tools is not offered
    (the _load_delegation_note() contract, consumed by _prefix.prefix_notes())."""
    if not _ptc_enabled():
        return None
    return {"role": "user", "content": PTC_NOTE}


def _append_ptc_note(text: str) -> None:
    """Stage a PTC runtime note (nudge and/or stub source) on the shared
    _PENDING_PTC_NOTICE slot; a nudge and a source staged in the same turn
    ride ONE note instead of overwriting each other."""
    from . import _notices
    cur = _notices.get("_PENDING_PTC_NOTICE")
    _notices.set_("_PENDING_PTC_NOTICE", (cur + "\n" + text) if cur else text)


def _stage_stub_source() -> None:
    """Ship the generated stub-module source ONCE as a [Runtime note] tail —
    the Audit's hard rule: the source is dynamic (it follows the gate vector)
    so it must NEVER enter the pinned prefix; the runtime-note tail is
    append-only and cache-safe. Fires on the first parent-loop tool call of a
    PTC session (see _dispatch.run_tool); sub-agent/batch loops skip — they
    don't own the parent's notice channel (their scripts still get the right
    render: run_tools re-renders per call via render_stub_module_for_turn())."""
    if st._PTC_SOURCE_SHIPPED or not _ptc_enabled():
        return
    from ._subagent import _subagent_ctx
    from ._batch import _batch_active
    if _subagent_ctx() is not None or _batch_active():
        return
    st._PTC_SOURCE_SHIPPED = True
    from ._ptc_stubs import render_stub_module_for_turn
    try:
        src = render_stub_module_for_turn()
    except Exception:  # noqa: BLE001 — a failed render must not break a call
        st._PTC_SOURCE_SHIPPED = False
        return
    _append_ptc_note(
        "[Runtime note: PTC stub module for run_tools(code) — generated from "
        "this turn's offered tool set (re-shipped whenever the gate set "
        "changes). Inside run_tools, `import stubs` and call these; each "
        "returns the tool result string, and only your script's printed "
        "output re-enters the conversation.\n" + src + "\n]")


def ptc_allowlist() -> frozenset[str]:
    """Stub-callable tool names: the explicit intersection of the dispatch
    read-cache tuple and the plan-mode read-only set (Audit correction #2 —
    the union would expose spawn_agent/todo_write/plan_edit to scripts).
    Resolved at call time so both source tables stay the single truth; lazy
    import keeps this module off the early import path."""
    from ._dispatch import _CACHEABLE_READ_TOOLS
    from ._plan import PLAN_READONLY_TOOLS
    return frozenset(_CACHEABLE_READ_TOOLS) & frozenset(PLAN_READONLY_TOOLS)


def _ptc_refusal(name: str) -> str:
    """Standard refusal for a non-allowlisted name over the pipe — the same
    'not available + what to do instead' shape as _GATE_REFUSALS."""
    # Only names actually rendered as stubs (Phase 1 exclusions dropped, and
    # retired-from-the-floor names like list_dir never appear in a render).
    from ._ptc_stubs import EXCLUDED_TOOLS
    allowed = ", ".join(sorted(ptc_allowlist() - EXCLUDED_TOOLS))
    return (f"{name} is not callable from a run_tools script — the PTC pipe is "
            f"read-only ({allowed}). Every other tool (writes, run_bash, "
            "spawn_agent, ...) must be a normal JSON tool call this turn; make "
            "the read you need here and the change outside the script.")


def _breaker_increment() -> None:
    """One malformed-args breaker increment for `run_tools` itself, mirroring
    `_turn_toolcalls.py:200-216` (streak count, trip at the limit, the
    _PTC_BREAKER_TRIPPED baseline counter). A script-level failure is PTC's
    version of a malformed call: same accounting, so the fallback ladder sees
    PTC and JSON failures alike."""
    from ._breaker import _breaker_views
    tool_name = "run_tools"
    if st.MALFORMED_TOOL_BREAKER_LIMIT <= 0:
        return
    failures, broken = _breaker_views()
    count = failures.get(tool_name, 0) + 1
    failures[tool_name] = count
    if count >= st.MALFORMED_TOOL_BREAKER_LIMIT:
        broken.add(tool_name)
        st._PTC_BREAKER_TRIPPED += 1


# --- the generated child side ---------------------------------------------------
# The launcher runs under -I beside the rendered stubs.py and the model's
# user_script.py. It owns the ONLY capability the child has: the frame pipe on
# its real stdin/stdout (fd 0/1). Real stdout/stderr are captured into a
# StringIO so script printing can never corrupt the frame channel; the script's
# exception (if any) rides the terminal `done` frame.

_PTC_LAUNCHER_SRC = '''"""nx run_tools launcher — generated per call by nx/_ptc_exec.py. Do not edit."""
import io
import json
import os
import sys
import threading
import traceback

_MAX_CAPTURE = 200000

_here = os.path.dirname(os.path.realpath(sys.argv[0]))
# -I does not prepend the script dir; insert it so `import stubs` resolves to
# the module rendered next to this launcher and nothing else.
sys.path.insert(0, _here)

_out = sys.stdout   # the frame channel stays on the real stdout (fd 1)
_in = sys.stdin     # response lines come from the real stdin (fd 0)
_cap = io.StringIO()
sys.stdout = _cap   # script prints never touch the frame channel
sys.stderr = _cap
_lock = threading.Lock()   # guards _seq only (frame writes are atomic-enough
_seq = [0]                 # via GIL line buffering, but stay tidy)
_waiters = {}              # capture id -> ([response], threading.Event)
_waiters_lock = threading.Lock()


def _send(obj):
    _out.write(json.dumps(obj) + "\\n")
    _out.flush()


def _reader():
    """Owns _in exclusively: one thread reads response frames and wakes the
    per-call waiter, so script threads may stub-call concurrently. EOF clears
    every pending waiter with a None response (the pipe-closed string)."""
    while True:
        line = _in.readline()
        if not line:
            break
        try:
            resp = json.loads(line)
            if not isinstance(resp, dict):
                raise ValueError
        except ValueError:
            resp = {"error": "malformed harness response frame"}
        cid = resp.get("id")
        with _waiters_lock:
            waiter = _waiters.pop(cid, None) if isinstance(cid, int) else None
        if waiter is not None:
            waiter[0].append(resp)
            waiter[1].set()
    with _waiters_lock:
        pending = list(_waiters.values())
        _waiters.clear()
    for holder, ev in pending:
        holder.append(None)
        ev.set()


def _capture(name, args):
    """Marshal one stub call to the harness and block for its answer.
    Failures arrive as "ERROR: ..." strings (the run_tool convention), never
    as raises across the pipe — a refused or failed call is the script's to
    handle, exactly as an ERROR tool result is JSON mode's. Thread-safe:
    several script threads may capture concurrently (Phase 3)."""
    global _seq
    with _lock:
        _seq[0] += 1
        cid = _seq[0]
    holder: list = []
    ev = threading.Event()
    with _waiters_lock:
        _waiters[cid] = (holder, ev)
    try:
        _send({"id": cid, "name": name, "args": args})
    except Exception:
        with _waiters_lock:
            _waiters.pop(cid, None)
        return "ERROR: run_tools pipe closed by the harness"
    ev.wait()
    if not holder or holder[0] is None:
        return ("ERROR: run_tools pipe closed by the harness (the script was "
                "killed or the turn ended; results are not available)")
    resp = holder[0]
    if "error" in resp:
        return "ERROR: " + str(resp["error"])
    return str(resp.get("result", ""))


import stubs  # noqa: E402  (rendered by render_stub_module_for_turn())
stubs._capture = _capture
threading.Thread(target=_reader, daemon=True).start()

_err = None
try:
    with open(os.path.join(_here, "user_script.py"), encoding="utf-8") as _f:
        _code = _f.read()
    exec(compile(_code, "run_tools_script", "exec"), {"__name__": "__main__"})
except SystemExit as _e:
    if _e.code not in (0, None):
        _err = "SystemExit(%r)" % (_e.code,)
except BaseException:
    _err = traceback.format_exc(20)
finally:
    _text = _cap.getvalue()
    if len(_text) > _MAX_CAPTURE:
        _text = _text[:_MAX_CAPTURE] + "\\n[script output truncated]"
    try:
        _send({"type": "done", "stdout": _text, "error": _err})
    except Exception:
        pass
'''


def _skip_fill_reason(exc: BaseException | None) -> str:
    """Skip-fill wording for in-flight captures, mirroring the JSON-mode
    skip-fill strings at _turn_toolcalls.py:269-327 (Esc vs hard-cancel)."""
    if isinstance(exc, KeyboardInterrupt):
        return "CANCELLED by user (Ctrl+C hard cancel) — script killed"
    return "CANCELLED by user (Esc) — script killed, returned to chat input"


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the script's whole process group (it owns one via
    start_new_session), SIGKILL after a short grace. Never raises; a proc that
    already exited is a no-op."""
    if proc.poll() is not None:
        return
    try:
        try:
            os.killpg(proc.pid, 15)  # SIGTERM the group
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)  # SIGKILL the group
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:  # reap — a SIGKILLed child left unwaited stays a zombie
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001 — teardown must never raise
                pass
    except Exception:  # noqa: BLE001 — teardown must never mask the real path
        pass


class _PtcProtocolError(Exception):
    """A frame the harness cannot parse, or a child that died mid-protocol."""


def _dispatch_pipe_call(name: Any, args: Any, respond) -> None:
    """Route ONE pipe request and answer it with `respond(payload)`.

    This is the Phase 3 seam: the function is self-contained (allowlist check
    + run_tool + reply), so a ThreadPoolExecutor can map frames through it
    concurrently once fan-out lands — writes already go through the caller's
    write lock. Every refusal/failure is reported as `{id, error}`; the
    launcher converts those into "ERROR: ..." return values, so per-call
    failures never kill the script (parity: an ERROR tool result in JSON
    mode doesn't kill the turn either)."""
    allow = ptc_allowlist()
    if not isinstance(name, str) or name not in allow:
        respond({"error": _ptc_refusal(str(name))})
        return
    if not isinstance(args, dict):
        respond({"error": f"script sent non-object args ({type(args).__name__})"})
        return
    # Module-attribute lookup so tests can monkeypatch nx._dispatch.run_tool
    # (the historical fake-dispatch seam) and have it land here.
    from . import _dispatch as _disp
    from ._approval import _EscToChat
    try:
        result = _disp.run_tool(name, dict(args))
    except _EscToChat:
        raise  # control-flow signal, not an error: must reach the turn layer
                 # so the JSON-mode skip-fill writes the CANCELLED result
    except Exception as e:  # noqa: BLE001 — run_tool folds its own errors;
        # anything escaping it (a hook exploding, say) still gets a frame.
        respond({"error": f"{type(e).__name__}: {e}"})
        return
    if isinstance(result, list):  # content-parts (read_image) can't ride the pipe
        result = "[non-text tool result]"
    respond({"result": result if isinstance(result, str) else str(result)})


def _dispatch_pipe_batch(batch: list[tuple[int, Any, Any]], respond,
                         deadline: float | None = None) -> tuple[int, bool]:
    """Dispatch a BATCH of pending pipe frames (Phase 3): >1 call goes through
    a ThreadPoolExecutor (bound of _run_read_only_parallel), 1 call stays on
    the serial fast path — no pool. Replies are emitted in frame-id order once
    the batch completes (the protocol carries ids either way); run_tool is
    resolved per-call via the module attribute, so a monkeypatched
    nx._dispatch.run_tool applies from pool threads. Returns (peak concurrent
    in-flight calls, timed_out) — timed_out means the wall-clock deadline
    expired while the batch was still running; unanswered calls then get
    skip-filled by the run_tools finally (kill + error frames), so a hung
    dispatch (a blocked hook, a FIFO read) can never outlive the script's
    budget."""
    if len(batch) <= 1:
        cid, name, args = batch[0]
        _dispatch_pipe_call(name, args,
                            lambda payload: respond(dict(payload, id=cid)))
        return 1, False
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        return len(batch), True
    replies: dict[int, dict] = {}

    def _one(item: tuple[int, Any, Any]) -> None:
        cid, name, args = item
        _dispatch_pipe_call(name, args,
                            lambda payload, _cid=cid:
                            replies.__setitem__(_cid, dict(payload, id=_cid)))

    with ThreadPoolExecutor(max_workers=min(8, len(batch))) as pool:
        futures = [pool.submit(_one, item) for item in batch]
        if deadline is None:
            for f in futures:
                f.result()
            timed_out = False
        else:
            budget = max(0.0, deadline - time.monotonic())
            done, not_done = concurrent.futures.wait(
                futures, timeout=budget)
            timed_out = bool(not_done)
            if timed_out:
                # Cancel what can still be cancelled and leave the rest to
                # the skip-fill + kill path (pool threads are daemonic-ish:
                # shutdown(wait=False) lets the with-block return while a
                # wedged run_tool thread dies with the process group kill).
                for f in not_done:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                for cid, _n, _a in batch:
                    replies.setdefault(
                        cid, {"id": cid,
                              "error": _skip_fill_reason(None)})
    for cid in sorted(replies):
        respond(replies[cid])
    return len(batch), timed_out


def run_tools(code: str = "", timeout: int = 0) -> str:
    """The `run_tools` tool body: run the model's Python with `stubs`
    importable, dispatching stub calls back through run_tool() over a NDJSON
    frame pipe. Returns the script's captured stdout (sanitized ONCE here,
    like every tool result by run_tool's tail) — per-call results live inside
    the script, exactly as tool results live inside a JSON-mode turn.

    Isolation posture: best-effort OS hardening only. `_ptc_preexec` (the
    `preexec_fn` below) tries seccomp net-deny + Landlock + NO_NEW_PRIVS,
    and every layer is silently skipped where the kernel/library doesn't
    support it — so on macOS/old kernels the child runs as this user with
    filesystem/network reach, and even on a fully hardened Linux box the
    read-only allowlist governs the stub pipe, not the Python language.
    The boundary that is *always* in force: the scrubbed environment (no
    secrets via os.environ), the allowlist + write fence inside run_tool
    for pipe-routed calls, and the process-group kill on timeout/cancel."""
    from ._approval import _EscToChat
    from ._interrupt import cancel_requested

    code = str(code or "")
    if not code.strip():
        return "ERROR: run_tools: `code` is empty — pass a Python script that imports `stubs`."
    try:
        limit = int(timeout or 0)
    except (TypeError, ValueError):
        limit = 0
    default_limit = _env_int("NX_PTC_TIMEOUT", 30)
    hard_max = _env_int("NX_PTC_TIMEOUT_MAX", 120)
    limit = limit if limit > 0 else default_limit
    limit = max(1, min(limit, hard_max))

    # The stub module for THIS turn (gate-aware: a sub-agent's tiered set gets
    # its own render, Phase 1 memo rules apply). Rendered once per call,
    # memoized across calls by _ptc_stubs.
    from ._ptc_stubs import render_stub_module_for_turn
    stub_src = render_stub_module_for_turn()

    workdir = tempfile.mkdtemp(prefix="nx-ptc-")
    proc = None
    try:
        with open(os.path.join(workdir, "stubs.py"), "w", encoding="utf-8") as f:
            f.write(stub_src)
        with open(os.path.join(workdir, "run_tools_launcher.py"), "w",
                  encoding="utf-8") as f:
            f.write(_PTC_LAUNCHER_SRC)
        with open(os.path.join(workdir, "user_script.py"), "w",
                  encoding="utf-8") as f:
            f.write(code)

        # Scrubbed env: the child is unsandboxed (`-I` isolates Python, not
        # the OS), so an inherited environment would hand API keys/tokens to
        # the script via os.environ. Only what the protocol needs is passed.
        proc = subprocess.Popen(
            [sys.executable, "-I", "-X", "utf8",
             os.path.join(workdir, "run_tools_launcher.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=True,  # own process group: Esc kills the script's
            # whole group, never the terminal (the _bash_jobs.py:235 precedent)
            cwd=os.getcwd(),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "HOME": os.environ.get("HOME", "/tmp"),
                 "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"},
            preexec_fn=lambda: _ptc_preexec(workdir),  # seccomp+landlock+NNP
        )
        st._ACTIVE_PTC_PROC = proc  # visible to tests/teardown; cleared below

        frames: "queue.Queue[str | None]" = queue.Queue()

        def _reader() -> None:  # daemon: moves frames, owns nothing else.
            try:
                for line in proc.stdout:
                    frames.put(line)
            except Exception:  # noqa: BLE001 — EOF/pipe teardown race
                pass
            finally:
                frames.put(None)  # EOF sentinel

        threading.Thread(target=_reader, daemon=True).start()

        write_lock = threading.Lock()

        def _respond(payload: dict) -> None:
            try:
                with write_lock:
                    proc.stdin.write(json.dumps(payload) + "\n")
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass  # child already gone; the loop notices via EOF

        script_out: str | None = None
        script_err: str | None = None
        pending_ids: set[int] = set()  # captures awaiting a reply (batch-wide:
        # the Phase 3 skip-fill answers EVERY id in the batch before the kill)
        n_calls = 0
        n_failed = 0
        n_concurrent = 0  # peak in-flight stub calls (summary + deprecation base)
        timed_out = False
        interrupted: BaseException | None = None
        deadline = time.monotonic() + limit
        # Phase 3 replaces this serial loop with a ThreadPoolExecutor mapping
        # frames through _dispatch_pipe_call (index-ordered like
        # _run_read_only_parallel); the protocol and that function stay as-is.
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = frames.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    if cancel_requested():
                        interrupted = KeyboardInterrupt("hard cancel")
                        break
                    continue
                if line is None:  # EOF without a done frame
                    raise _PtcProtocolError(
                        "script exited without a completion frame")
                try:
                    frame = json.loads(line)
                except ValueError:
                    raise _PtcProtocolError("malformed frame from the script")
                if frame.get("type") == "done":
                    script_out = frame.get("stdout") or ""
                    script_err = frame.get("error")
                    break
                cid = frame.get("id")
                if not isinstance(cid, int):
                    raise _PtcProtocolError("frame without an id")
                # Phase 3: drain every ALREADY-pending frame into one batch and
                # map it through the pool; frames arriving mid-batch wait for
                # the next iteration (the reader thread keeps buffering).
                batch = [(cid, frame.get("name"), frame.get("args"))]
                held_done = None
                while True:
                    try:
                        extra_line = frames.get_nowait()
                    except queue.Empty:
                        break
                    if extra_line is None:  # EOF mid-batch: handled next loop
                        frames.put(None)
                        break
                    try:
                        extra = json.loads(extra_line)
                    except ValueError:
                        raise _PtcProtocolError(
                            "malformed frame from the script")
                    if extra.get("type") == "done":
                        # Hold — do NOT re-queue: putting it back at the tail
                        # inverts the queue if the reader's EOF sentinel is
                        # already in it ([None, done] → the sentinel gets
                        # consumed first and a completed script reports a
                        # false protocol error + breaker increment).
                        held_done = extra_line
                        break
                    extra_id = extra.get("id")
                    if not isinstance(extra_id, int):
                        raise _PtcProtocolError("frame without an id")
                    batch.append((extra_id, extra.get("name"),
                                  extra.get("args")))
                n_calls += len(batch)
                n_failed += sum(1 for _n, _a in
                                ((f[1], f[2]) for f in batch)
                                if not _call_will_dispatch(_n, _a))
                pending_ids = {f[0] for f in batch}
                peak, batch_timed_out = _dispatch_pipe_batch(
                    batch, _respond, deadline=deadline)
                n_concurrent = max(n_concurrent, peak)
                if batch_timed_out:
                    timed_out = True
                    break
                pending_ids = set()
                if held_done is not None:
                    # The completion frame was already in the queue during the
                    # drain — the script finished; consume it now instead of
                    # looping back around (where the EOF sentinel would be
                    # dequeued first).
                    extra = json.loads(held_done)
                    script_out = extra.get("stdout") or ""
                    script_err = extra.get("error")
                    break
        except (_EscToChat, KeyboardInterrupt) as e:
            interrupted = e
        except _PtcProtocolError as e:
            script_err = f"PTC protocol error: {e}"
        finally:
            if interrupted is not None or timed_out or script_err is not None:
                # Skip-fill BEFORE the kill: every in-flight `_capture` gets a
                # well-formed `{id, error}` response so the launcher unblocks,
                # its script sees an ERROR result (not a hang), and the kill
                # lands on a child that is already exiting — the transcript
                # entry for the run_tools call itself is written by the JSON
                # skip-fill loop wrapped around this call (parity with
                # _turn_toolcalls.py:269-327, which still sees the raise).
                for _cid in sorted(pending_ids):
                    _respond({"id": _cid,
                              "error": _skip_fill_reason(interrupted)})
                _kill_group(proc)
            else:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_group(proc)
        if interrupted is not None:
            raise interrupted

        out_text = script_out or ""
        if timed_out:
            # NOT a breaker event: a correct-but-slow script is not malformed
            # output (JSON mode trips the breaker only on parse failures), so
            # a legit long job must never withdraw run_tools via the fallback
            # ladder — fold the error result and let the model retry smaller.
            return _fold(f"ERROR: run_tools timed out after {limit}s — the "
                         f"script's process group was killed. Split the work "
                         f"into smaller scripts or raise the per-call "
                         f"`timeout` (cap {hard_max}s).", out_text)
        if script_err:
            _breaker_increment()
            head = script_err.strip().splitlines()
            return _fold("ERROR: run_tools script failed: "
                         + ("\n".join(head[-6:]) if head else "unknown error"),
                         out_text)

        if n_calls > 1:
            st._PTC_FANOUT_SCRIPTS += 1  # deprecation baseline (Phase 3)
        summary = (f"[run_tools: {n_calls} stub call(s)"
                   + (f", {n_failed} refused/failed" if n_failed else "")
                   + (f", peak {n_concurrent} concurrent" if n_concurrent > 1
                      else "")
                   + f", {limit}s cap]")
        body = out_text.strip()
        if not body:
            body = "(script printed nothing — print the summary you want back)"
        from ._text_fallback import _sanitize_tool_result
        return _sanitize_tool_result(f"{summary}\n--- script stdout ---\n{body}")
    finally:
        # Single teardown: kill the child no matter which window raised, then
        # clear the test-visible handle, then drop the tempdir. Covers the
        # Popen→reader/queue setup span that previously had no kill coverage.
        if proc is not None and proc.poll() is None:
            _kill_group(proc)
        st._ACTIVE_PTC_PROC = None
        shutil.rmtree(workdir, ignore_errors=True)


def _call_will_dispatch(name: Any, args: Any) -> bool:
    """True when this frame's call will NOT be refused by the allowlist (the
    failure count for the run_tools summary; mirrors how the JSON turn counts
    ERROR/DENIED results without touching the breaker)."""
    return isinstance(name, str) and name in ptc_allowlist() and isinstance(args, dict)


def _fold(head: str, script_out: str) -> str:
    """One error result for run_tools with whatever the script printed so far
    attached (the same ERROR-fold shape every tool failure uses), sanitized
    exactly once here."""
    from ._text_fallback import _sanitize_tool_result
    text = head
    if script_out.strip():
        text += "\n--- script stdout before the failure ---\n" + script_out.strip()[:2000]
    return _sanitize_tool_result(text)
