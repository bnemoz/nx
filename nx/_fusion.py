"""Action fusion (SoL-Pi port, 0.67.0): an edit/write can carry its follow-up
validation command in the SAME tool call (`then_run`), eliminating one model
round-trip for the edit -> run tests cadence.

nx is already detached-and-polled for bash, so the fused command reuses
_bash_jobs.run_bash wholesale: it goes through the same approval prompt, the
same ripgrep rewrite, the same secret scan, the same backgrounding semantics.
If the fused command's output lands whole (it finished within the poll
window), the raw bash output is replaced by an observation-pack handle via
_obs.pack_output() so a large build/test log doesn't bill full weight twice
(once inline in the fused result, once in history).

Markers (stable, parsed by the reducer and by tests):
  [then_run:succeeded]  the fused command ran and exited
  [then_run:failed]     the fused command ran but errored / was denied
  [then_run:skipped]    the mutation itself failed, so the command never ran

Per-file serialization (with_fused_file_queue): two fused calls on the same
file must not interleave mutation+command phases. A module-level
path-keyed promise chain (a lock + the file's own threading.Lock tail) orders
them; a plain threading.Lock per canonical path is enough since each fused
call holds its file's lock for the whole mutation+command sequence.

Gate: NX_FUSION=1 enables then_run on the edit tools. The extra `then_run`
property costs ~40 schema tokens per edit tool while enabled; disabled by
default like the other opt-in behavior changers (NX_SUBAGENTS, NX_VISION).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any, Callable

from ._env import _env_int

THEN_RUN_SUCCEEDED = "[then_run:succeeded]"
THEN_RUN_FAILED = "[then_run:failed]"
THEN_RUN_SKIPPED = "[then_run:skipped]"

# ON by default (0.67.1): fused edit+validate is the agent's most common
# two-step cadence and the schema property itself is the only cost (~40
# tokens per edit tool, inside the recorded 4K budget). NX_FUSION=0 restores
# the old separate round-trip.
FUSION_ENABLED = _env_int("NX_FUSION", 1)

# Seconds nx waits for the fused command before backgrounding it. A fused
# validation command is usually a test/build; the model asked to see its
# result, so give it a longer poll window than run_bash's interactive 3s
# default (the model can always pass then_run.timeout for longer).
FUSION_POLL_SECONDS = _env_int("NX_FUSION_POLL_SECONDS", 30)

# Schema fragment appended to write_file/edit_file/multi_edit when enabled.
THEN_RUN_PROP = {
    "type": "object",
    "description": (
        "Optional follow-up command run in the same call after the edit "
        "succeeds (saves a round-trip). Skipped entirely if the edit fails."
    ),
    "properties": {
        "command": {"type": "string",
                    "description": "Shell command, e.g. the test command"},
        "timeout": {"type": "integer",
                    "description": "Seconds to wait before backgrounding (default 30; 0 = wait until it exits)"},
    },
    "required": ["command"],
}

_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def fusion_enabled() -> bool:
    return FUSION_ENABLED == 1


def _canonical_key(path: str) -> str:
    """Realpath of the deepest existing ancestor + the missing tail, so two
    spellings of the same not-yet-created file share one queue slot."""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    missing: list[str] = []
    probe = p
    while probe and not os.path.exists(probe):
        nxt, tail = os.path.split(probe)
        if nxt == probe:
            break
        missing.append(tail)
        probe = nxt
    try:
        base = os.path.realpath(probe or p)
    except OSError:
        base = probe or p
    return os.path.join(base, *reversed(missing))


class _FileQueue:
    """Per-path serialization for fused mutation+command sequences.

    Bounded (0.108.4): one lock per distinct path, and a swarm touching many
    paths could grow the table without limit. Stale locks are safe to drop —
    an eviction while another thread holds the lock just means the next caller
    for that path gets a fresh lock and serializes against it separately, so
    the cap only trims never-currently-contended paths (checked before insert).
    """

    _CAP = 256

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def __call__(self, path: str) -> threading.Lock:
        key = _canonical_key(path)
        with self._guard:
            lk = self._locks.get(key)
            if lk is None:
                if len(self._locks) >= self._CAP:
                    self._locks.pop(next(iter(self._locks)))
                lk = threading.Lock()
                self._locks[key] = lk
        return lk


with_fused_file_queue = _FileQueue()


def _hash_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def assert_unchanged_before_command(path: str, before: str | None,
                                    reason: str) -> str | None:
    """Hash guard: hash the target, yield briefly, hash again. A mismatch
    means something else wrote the file between our mutation and the fused
    command — run neither's assumptions forward. Returns a skip reason or
    None (also None when the file vanished, which is a legitimate outcome)."""
    time.sleep(0)  # one scheduler yield, same spirit as SoL-Pi's setImmediate
    after = _hash_file(path)
    if after is not None and before is not None and after != before:
        return (f"{reason} (file changed under us between the edit and the "
                "fused command)")
    return None


def append_then_run(result: str, ok: bool, command: str, output: str,
                    skipped: str | None = None) -> str:
    """Attach the fused-command outcome to the mutation result string."""
    if skipped:
        return f"{result}\n\n{THEN_RUN_SKIPPED} {skipped} The command was not run."
    marker = THEN_RUN_SUCCEEDED if ok else THEN_RUN_FAILED
    body = output.strip() or "(no output)"
    return f"{result}\n\n{marker}\n{body}"


def run_fused(args: dict[str, Any], mutate: Callable[[], str],
              result_prefix: str = "") -> str:
    """Shared executor for write_file/edit_file/multi_edit then_run.

    `mutate` runs the underlying tool body (already past approval — run_tool
    calls this after `fn(**args)` returned, or the wrapper body calls it
    directly). Returns the combined result string with markers."""
    path = str(args.get("path") or "")
    then_run = args.get("then_run") or {}
    command = str(then_run.get("command") or "").strip()
    want_fused = fusion_enabled() and bool(command) and bool(path)

    lock = with_fused_file_queue(path) if want_fused else None
    if lock is not None:
        lock.acquire()
    try:
        before = _hash_file(path) if want_fused else None
        try:
            result = mutate()
        except Exception as e:  # noqa: BLE001 — mirror run_tool's error fold
            result = f"ERROR: {type(e).__name__}: {e}"
        if not want_fused:
            return result
        if result.startswith(("ERROR", "DENIED")):
            return append_then_run(
                result, ok=False, command=command, output="",
                skipped="The file mutation did not complete successfully.")
        skip = assert_unchanged_before_command(path, before, THEN_RUN_SKIPPED)
        if skip:
            return append_then_run(result, ok=False, command=command,
                                   output="", skipped=skip[len(THEN_RUN_SKIPPED) + 1:])
        from ._bash_jobs import run_bash
        timeout = then_run.get("timeout")
        timeout = FUSION_POLL_SECONDS if timeout is None else max(0, int(timeout))
        out = run_bash(command, timeout=timeout)
        ok = not out.startswith(("ERROR", "DENIED"))
        from ._obs import maybe_pack_fused_output
        out = maybe_pack_fused_output(out, "then_run", f"{path}:{command}")
        return append_then_run(result, ok=ok, command=command, output=out)
    finally:
        if lock is not None:
            lock.release()
