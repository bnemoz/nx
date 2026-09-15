"""Tool dispatch: the DISPATCH table, run_tool() (approval/spinner/cache/undo
wrapping around each call), the read-only result cache, the per-session
one-time nudge messages (backgrounding, multi_edit batching), and the
fresh-session message list.

TOOLS/FINAL_ANSWER_TOOL/SYSTEM themselves stay in __init__.py (see
CLAUDE.md: they're load-bearing on the per-turn token floor and meant to be
easy to find) — this module is the glue around them, not the schemas.

`TOOLS` and `SYSTEM` are monkeypatched/read from the top-level `nx` module in
a couple of tests, so they're looked up lazily via `import nx as _m` rather
than bound at import time — same pattern as _repl.py.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from . import _state as st
from ._env import _env_int
from ._ansi import DIM, RESET, C, LifeSpinner, strip_ansi_for_display
from ._tool_cache import (
    cache_valid, drop_changed_entries,
    drop_vcs_entries, store_cacheable_result,
)
from ._approval import _EscToChat
from ._bash_jobs import run_bash, wait_background
from ._edit_tools import write_file, edit_file, multi_edit, sed_edit, apply_patch, delete_file
from ._memory_cmd import memory_write, memory_search, memory_delete
from ._metrics import _abbr
from ._text_fallback import _sanitize_tool_result
from ._todo import todo_write, render_todos_display
from ._plan import plan_edit, plan_active, check_gate
from ._subagent import spawn_agent, _delegation_enabled
from ._images import read_image
from ._tools import read_file, list_dir, search_files, git_status, git_diff, git_log
from ._testrunner import run_tests
from ._gitcommit import git_commit
from ._toolblocks import ToolBlock, current_recorder, tool_display
from ._undo import snapshot as _undo_snapshot, push_checkpoint as _undo_push, push_if_absent as _undo_push_once

DISPATCH = {
    "read_file": read_file, "write_file": write_file, "edit_file": edit_file,
    "list_dir": list_dir, "run_bash": run_bash, "wait_background": wait_background,
    "search_files": search_files, "multi_edit": multi_edit, "sed_edit": sed_edit,
    "git_status": git_status, "git_diff": git_diff, "git_log": git_log,
    "todo_write": todo_write, "apply_patch": apply_patch, "delete_file": delete_file,
    "memory_write": memory_write, "memory_search": memory_search,
    "memory_delete": memory_delete,
    "spawn_agent": spawn_agent, "read_image": read_image,
    "plan_edit": plan_edit, "run_tests": run_tests, "git_commit": git_commit,
}

# spawn_agent() runs its sub-loop with this table installed in st._SUBAGENT_DISPATCH
# (plus a per-thread copy in _subagent._SUBAGENT_TLS — concurrent agents each need
# their own): read-only entries ONLY (run_bash wrapped in a read-only allowlist
# gate), so a sub-agent can never mutate state or run an arbitrary command even if
# it tries. run_tool() consults the TLS table before DISPATCH. None = normal dispatch.
st._SUBAGENT_DISPATCH = None
st._SUBAGENT_ACTIVE = False

st._SESSION_HAS_BACKGROUNDED = False  # flips True the first time a run_bash call in this session actually backgrounds
st._PENDING_BACKGROUND_NOTICE = None  # set by run_tool() on that first flip; flushed once by model_turn (see _flush_pending_background_notice)
st._BG_SCAN_DONE = False  # per-turn: scan for finished bg jobs once, on the first tool call
st._PENDING_BG_COMPLETION_NOTICE = None  # set by that scan; flushed with the other runtime notes
# [Runtime note: ...] staged when a background (async) sub-agent finishes —
# worker-side via _subagent._stage_subagent_answer, swept by the same
# first-tool-of-turn scan below, flushed once with the other runtime notes.
st._PENDING_SUBAGENT_NOTICE = None
st._BG_NOTICE_SEEN = None  # pids already announced (None → seed from _BG_NOTICE_BASELINE)
# The scan runs from both the model turn (run_tool's first-tool hook) and the
# wake poller thread; the seen-set is read-modify-rewritten, so both sides
# serialize on this (also held by _bash_jobs._mark_bg_notice_seen).
_BG_SCAN_LOCK = threading.Lock()
st._CONSECUTIVE_EDITS = 0  # count of back-to-back successful edit_file calls (any other tool call resets it)
st._MULTI_EDIT_NUDGED = False  # flips True once the batching nudge has fired this session — see run_tool()
st._PENDING_MULTI_EDIT_NOTICE = None  # set by run_tool() on that nudge; flushed once by model_turn (see _flush_pending_background_notice)

_MULTI_EDIT_NUDGE_THRESHOLD = 2  # ≥2 single edits in a row before the batching nudge fires

# Delegation nudge (see DELEGATION_STREAK_NOTICE below): streak counter, once
# flag and staged notice — same three-piece shape as the multi_edit nudge.
st._READ_STREAK = 0  # how many exploration reads in a row with no spawn_agent
st._DELEGATION_NUDGED = False  # flips True once the delegation nudge has fired this session
st._PENDING_DELEGATION_NOTICE = None  # staged notice, flushed with the other runtime notes

# ≥ this many such reads before the nudge fires. Deliberately above any
# legitimate single-task read count (an edit usually touches 1-3 files) and far
# below the 20-40-read sweeps models grind through when they never think to
# delegate. NX_DELEGATION_NUDGE_READS=0 disables the nudge.
_DELEGATION_NUDGE_THRESHOLD = _env_int("NX_DELEGATION_NUDGE_READS", 8)


def _breaker_endpoint_key() -> tuple[str | None, str | None]:
    """Identity of the endpoint the malformed-args evidence was gathered
    against: (active source name, active model). A per-model key because
    `/source <name> <model>` can point one Source at any model on a
    multi-model host, and truncation is a property of the weights + server
    pair, not of the Source row."""
    return (st.ACTIVE.name if st.ACTIVE is not None else None, st.MODEL)


def _breaker_views() -> tuple[dict[str, int], set[str]]:
    """(failures, broken) — the breaker's live containers, after re-keying.

    0.40: the breaker is evidence about ONE endpoint. It used to be
    process-global, so switching to a healthy source left the model permanently
    missing tools that endpoint handles fine. Every reader goes through here:
    if the (source, model) identity changed since the evidence was gathered,
    the streaks are dropped in place (clear, not reassign — the conftest state
    rewind and `nx.<name>` proxy both rely on container identity).

    Lazy rather than a hook in switch_source(): tests (and future callers) may
    reassign st.ACTIVE/st.MODEL directly, and a read-time check cannot be
    bypassed by forgetting to call a notifier.

    The first observation just adopts the key without clearing — there is no
    prior-endpoint evidence to drop yet, and the containers may legitimately
    hold state installed by a caller (a test, or /toolcalls reset) rather than
    by an observed truncation."""
    key = _breaker_endpoint_key()
    observed = st._BREAKER_ENDPOINT_KEY
    if observed != key:
        if observed != st.BREAKER_UNOBSERVED:
            st._MALFORMED_TOOL_FAILURES.clear()
            st._MALFORMED_TOOL_BROKEN.clear()
        st._BREAKER_ENDPOINT_KEY = key
    return st._MALFORMED_TOOL_FAILURES, st._MALFORMED_TOOL_BROKEN


def _breaker_broken(name: str) -> bool:
    """True if `name` is currently dropped by the malformed-args breaker."""
    return name in _breaker_views()[1]


def _active_tools() -> list[dict[str, Any]]:
    """The tool schema list to send this turn. wait_background is included
    only once run_bash has actually backgrounded this session — see run_tool();
    spawn_agent (read-only sub-agent) only when NX_SUBAGENTS=1; read_image
    only when NX_VISION=1 AND native tool_calls have been confirmed this
    session (a text-protocol tool result can't carry a content-parts array);
    plan_edit only while a plan is active (/plan <goal> — see _plan.py)."""
    import nx as _m
    tools = _m.TOOLS
    if not st._SESSION_HAS_BACKGROUNDED:
        tools = [t for t in tools if t["function"]["name"] != "wait_background"]
    if not _m.NX_SUBAGENTS:
        tools = [t for t in tools if t["function"]["name"] != "spawn_agent"]
    # Malformed-args circuit breaker: an endpoint that keeps truncating one
    # tool's JSON args stops being offered that tool for the session (see
    # st._MALFORMED_TOOL_BROKEN — armed in _turn.py's native parse path).
    broken = _breaker_views()[1]
    if broken:
        tools = [t for t in tools if t["function"]["name"] not in broken]
    if not _m.NX_VISION or not st._NATIVE_TOOL_CALLS_CONFIRMED:
        tools = [t for t in tools if t["function"]["name"] != "read_image"]
    if not plan_active():
        tools = [t for t in tools if t["function"]["name"] != "plan_edit"]
    return tools


def _messages_show_prior_backgrounding(messages: list[dict[str, Any]]) -> bool:
    """True if a loaded transcript already contains a backgrounded run_bash
    result, so resuming a session doesn't lose wait_background access it
    already had (see call sites in main())."""
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content")
            if isinstance(content, str) and content.startswith("[background]"):
                return True
    return False


def _messages_show_prior_multi_edit_nudge(messages: list[dict[str, Any]]) -> bool:
    """True if a loaded transcript already used multi_edit, or made ≥2
    edit_file calls, so resuming doesn't re-fire a nudge run_tool() would
    already have sent by now (see call sites in _repl.py)."""
    edit_calls = 0
    for msg in messages:
        for call in (msg.get("tool_calls") or []):
            fn_name = (call.get("function") or {}).get("name")
            if fn_name == "multi_edit":
                return True
            if fn_name == "edit_file":
                edit_calls += 1
                if edit_calls >= _MULTI_EDIT_NUDGE_THRESHOLD:
                    return True
    return False


def _messages_show_prior_delegation_nudge(messages: list[dict[str, Any]]) -> bool:
    """True if a loaded transcript already spawned an agent or already received
    the delegation nudge, so resuming doesn't re-nudge a model that has already
    shown it delegates (or was already told once this session)."""
    for msg in messages:
        for call in (msg.get("tool_calls") or []):
            if (call.get("function") or {}).get("name") == "spawn_agent":
                return True
        content = msg.get("content")
        if isinstance(content, str) and ("[Delegation note:" in content
                                         or "without using spawn_agent" in content):
            return True
    return False


def _initial_messages() -> list[dict[str, Any]]:
    """Fresh-session message list: SYSTEM followed by the canonical prefix block
    (AGENTS.md note, memory index, orientation, delegation guidance). The block
    itself is built by _prefix.prefix_notes() — the single source of truth shared
    with the resume path (PERF.md PF-15) — and each note is kept out of SYSTEM so
    the per-turn token floor doesn't grow with project size (see _agents.py).

    Also (re)pins the block: a fresh session defines the bytes every later
    request must still open with (see _prefix.pin_prefix)."""
    from ._prefix import pin_prefix, prefix_notes
    import nx as _m
    messages = [{"role": "system", "content": _m.SYSTEM}]
    messages.extend(prefix_notes())
    pin_prefix(messages)
    return messages


# Paid once, only in sessions where it's actually relevant: a run_bash command
# backgrounding is the exception, not the norm, so the detailed operating
# guidance for it (and the wait_background tool schema — see _active_tools())
# don't belong in every single turn's SYSTEM/TOOLS floor. Instead this is
# injected as a one-time "[Runtime note: …]" user-role message the first time
# `run_tool()` sees a run_bash result start with "[background]" (see
# st._SESSION_HAS_BACKGROUNDED / st._PENDING_BACKGROUND_NOTICE below and the flush
# in model_turn). Appending it at the tail keeps every earlier turn's bytes
# untouched, so it never invalidates a server's prompt-prefix cache.
BACKGROUND_GUIDANCE = (
    "[Runtime note: a command you ran just backgrounded (didn't finish within "
    "the short poll window) — wait_background is now available as a tool. "
    "Guidance for backgrounded commands:\n"
    "- Do NOT use `sleep N && …` as a chaining trick — the whole command is "
    "detached (setsid) so it backgrounds too. Only pass `timeout=N` to run_bash when "
    "you KNOW the command finishes in ~N seconds (e.g. `timeout=35` for "
    "`sleep 30`).\n"
    "- For commands of unknown duration (docker pull, builds, servers), let "
    "them background, then call `wait_background(pid=...)` — it waits "
    "indefinitely until the command finishes; Esc cancels the WAIT, never the "
    "process.\n"
    "- A process with no new log output is NOT necessarily dead. Docker's "
    "layer-extraction, AOT compilation, and CUDA/graph-cache builds are "
    "legitimately silent for minutes. Check `/proc/<pid>/io`, `du -sh`, or "
    "`ps` CPU% BEFORE declaring a hang.\n"
    "- NEVER `pkill -f '<the thing we just started>'` as a \"clean restart\" "
    "— that destroys a job that may be 99% done. Prove it's stuck first (ps, "
    "/proc/<pid>/io, du), and only kill it if genuinely hung.\n"
    "- Treat exit 130 (SIGINT) / `context canceled` as a signal-of-a-signal, "
    "not a command failure — check the command's own logs before retrying.]"
)

# Same pattern as BACKGROUND_GUIDANCE above: paid once, only in sessions where
# it's relevant. Fires the first time _CONSECUTIVE_EDITS hits
# _MULTI_EDIT_NUDGE_THRESHOLD, to steer away from the observed failure mode of
# grinding through several more single edit_file calls on the same file
# instead of switching to one multi_edit batch (multi_edit's schema is sent
# from the start — see _active_tools — this is a usage nudge, not an unlock).
MULTI_EDIT_GUIDANCE = (
    "[Runtime note: you've made several edit_file calls in a row. If you have "
    "more literal find/replace changes left to make in a file (even just 2-3 "
    "more), batch them into ONE multi_edit call (edits=[{old,new},...]) "
    "instead of continuing with separate edit_file calls — it's one approval "
    "instead of several, and applies atomically.]"
)

# Third instance of the same paid-once pattern: staged when the model has made
# _DELEGATION_NUDGE_THRESHOLD exploration calls itself without ever spawning a
# sub-agent. Only staged while the spawn_agent schema is actually offered
# (NX_SUBAGENTS=1) — see run_tool()'s counter and _subagent._delegation_enabled.
DELEGATION_STREAK_NOTICE = (
    "[Runtime note: you've read or searched many files yourself without using "
    "spawn_agent. The remaining investigation is exactly what sub-agents are "
    "for: issue one spawn_agent call per remaining independent question in a "
    "single turn (they run concurrently, return only their findings, and keep "
    "their intermediate reads out of this context), then work from their "
    "answers. If the files you're reading are the exact ones an edit already "
    "in hand needs, ignore this and continue.]"
)


def _scan_background_completions() -> str | None:
    """One-time notice for tracked background jobs that finished since the
    model last looked (checked on the first tool call of each turn).

    The model must otherwise REMEMBER to wait_background/read_file a job it
    backgrounded turns ago; a build that failed 20 minutes ago doesn't
    announce itself. Returns the note text (or None): lists finished jobs with
    exit codes once, then marks them seen — no repeated noise."""
    from ._bash_jobs import list_bg_jobs, _read_bg_exit_status
    from ._verify import _is_verify_job  # verify jobs report via _stage_verify_outcome instead
    with _BG_SCAN_LOCK:
        try:
            jobs = list_bg_jobs()
        except Exception:
            return None
        mine = os.getpid()
        seen = getattr(st, "_BG_NOTICE_SEEN", None)
        if seen is None:
            seen = set(getattr(st, "_BG_NOTICE_BASELINE", set()) or set())
        notes = []
        finished = []
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
                finished.append(j)
                code = _read_bg_exit_status(pid)[1]
                tail = (j.get("tail") or "").strip()[:80]
                code_s = "unknown" if code is None else str(code)
                notes.append(f"  pid {pid} exited (code {code_s})" + (f" — tail: {tail}" if tail else ""))
            except Exception:
                # One malformed job must not discard the whole sweep: the
                # marks made so far are committed below either way, so a probe
                # that raises can never make the same pid re-announce forever.
                continue
        st._BG_NOTICE_SEEN = seen
    if not notes:
        return None
    return ("[Runtime note: background job(s) you started have finished:\n"
            + "\n".join(notes)
            + "\nCheck their output with read_file(log) or wait_background before relying on them.]")

_CACHEABLE_READ_TOOLS = ("read_file", "list_dir", "search_files", "git_status", "git_diff", "git_log", "memory_search")


def _run_bg_completion_scan() -> None:
    """The completion sweep run_tool() runs on the first tool call of a turn:
    scan finished bg jobs, NX_VERIFY_CMD outcomes, and finished async agents,
    staging each as its one-time [Runtime note]. The wake path shares it —
    see _autowake.wake_due for why a wake turn must not fire on an empty
    sweep."""
    note = _scan_background_completions()
    if note:
        # Merge, don't overwrite: a staged note can still be unflushed here
        # (the poller's own sweep staged one before this turn's first tool
        # call). Assigning would drop the earlier job's exit line — it is
        # already marked seen in _BG_NOTICE_SEEN, so no later scan would
        # ever re-announce it. _stage_subagent_answer merges the same way.
        prev = getattr(st, "_PENDING_BG_COMPLETION_NOTICE", None)
        if prev:
            # Dedup (0.37.4): the poller thread and this turn can both sweep
            # the same job in a race — _BG_SCAN_LOCK marks it seen once, but
            # each sweep may have already staged its own copy of the note.
            fresh = [ln for ln in note.splitlines()
                     if ln.strip().startswith("pid ") and ln not in prev]
            if fresh:
                note = f"{prev}\n" + "\n".join(fresh)
            # else: every line already staged — keep prev untouched
        st._PENDING_BG_COMPLETION_NOTICE = note
    # NX_VERIFY_CMD job just finished? Its outcome is staged as its own
    # pending notice (independent of the generic bg-completion note).
    from ._verify import _stage_verify_outcome
    try:
        _stage_verify_outcome()
    except Exception:
        pass
    # Async sub-agent finished since the last look? Its answer is normally
    # staged by the worker itself; this sweep catches any finished-but-unstaged
    # stragglers (idempotent per job id).
    from ._subagent import _stage_subagent_completions
    try:
        _stage_subagent_completions()
    except Exception:
        pass



def _tool_cache_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    return (name, json.dumps(args, sort_keys=True))




def _invalidate_tool_cache_path(path: str | None) -> None:
    if not path:
        return
    try:
        norm = os.path.abspath(path)
    except (TypeError, ValueError):
        return
    for key, entry in list(st._TOOL_RESULT_CACHE.items()):
        if entry.path == norm:
            del st._TOOL_RESULT_CACHE[key]


def _withheld_tool_error(name: str) -> str:
    """Refusal for a tool that exists but is NOT offered this turn: a session
    gate (sub-agents off, no backgrounded job yet, no plan active) or the
    malformed-args breaker. run_tool() applies it to RESOLVED names (0.40) —
    a repaired spelling like `multi_edits` -> `multi_edit` used to bypass both
    the early breaker check (which saw only the emitted name) and the gate
    checks, and the fall-through 'unknown tool' text then sent the model off to
    invent a substitute instead of fixing its spelling or its formatting.
    Empty for a tool that is in fact offered, so callers fall through."""
    if _breaker_broken(name):
        from ._turn import _malformed_args_nudge  # call-time: _turn imports this module
        return _malformed_args_nudge(name, broken_now=True)
    if name in {t["function"]["name"] for t in _active_tools()}:
        return ""
    return _unknown_tool_error(name)


def _unknown_tool_error(name: str) -> str:
    """Error text for a tool name that resolved to nothing in the dispatch table.

    Three distinct situations hide behind a failed lookup, and a flat "unknown
    tool" answers only the first: (a) the name really is made up (suggest the
    closest real names); (b) the tool exists but is withheld this session by a
    gate — the model never saw its schema this turn, so a bare "unknown" reads
    as "this capability does not exist at all" and the model abandons the task
    (small models do this; seen live with wait_background/read_image/spawn_agent);
    (c) the malformed-args circuit breaker dropped it — the model must know the
    tool died for cause, and that the failure is about ITS formatting, not the
    tool's existence. Never raises; unknown names fall to the generic wording."""
    import nx as _m
    known = {t["function"]["name"] for t in _m.TOOLS}
    known.update(t["function"]["name"] for t in getattr(_m, "_EXTRA_TOOL_SCHEMAS", ()) or ())
    if name not in known:
        import difflib
        suggestions = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.7)
        if suggestions:
            return (f"ERROR: unknown tool {name} — did you mean {', '.join(suggestions)}? "
                    f"Available tools: {', '.join(sorted(known))}")
        return f"ERROR: unknown tool {name} — Available tools: {', '.join(sorted(known))}"
    # Withheld this session: name the gate and the alternative, so the model
    # doesn't conclude the capability doesn't exist.
    if name == "wait_background":
        return (f"ERROR: {name} is not available in this session (no command has backgrounded "
                "yet — nothing to wait on). run_bash returns a '[background] PID' line when a "
                "command needs waiting on; read_file that log path in the meantime.")
    if name == "read_image":
        return (f"ERROR: {name} is not available in this session (NX_VISION is off, or native "
                "tool calls are not confirmed on this source). Describe images by other means "
                "or ask the user to paste the relevant details.")
    if name == "spawn_agent":
        return (f"ERROR: {name} is not available in this session (NX_SUBAGENTS is off). Do the "
                "investigation yourself with search_files/read_file/list_dir.")
    if name == "plan_edit":
        return (f"ERROR: {name} is not available in this session (no plan is active). Start one "
                "with /plan <goal>, or just proceed with the normal edit tools.")
    return (f"ERROR: {name} is not available in this session (withheld by the current session "
            "configuration). Use one of: " + ", ".join(sorted(known)))


def _breaker_decay(name: str) -> None:
    """0.29.9: one clean execution of `name` clears its malformed-args streak
    (and re-arms the tool if the breaker had dropped it). Two consecutive
    truncations on a flaky local server were routine; losing a tool for the
    whole session on that evidence was too heavy. A deliberately BROKEN tool
    (the model can't reach it anyway) only decays when the user re-offers it
    via /toolcalls reset — direct run_tool() calls (tests, hooks) count.

    0.40: reads through _breaker_views() first, so a clean dispatch on a NEW
    endpoint can never inherit a stale `broken` mark from the previous one."""
    failures, broken = _breaker_views()
    if failures.pop(name, None) is not None:
        broken.discard(name)


def run_tool(name: str, args: dict[str, Any]) -> str | list:
    # Plan-mode write gate: refuse mutating tools while a plan is active, with
    # the gate message the model needs to act on (plan_edit, or ask for
    # /plan done). Checked before every other gate so a gated call costs no
    # classifier round-trip, no approval prompt, and no sub-agent dispatch.
    import nx as _m
    gated = check_gate(name)
    if gated:
        return gated
    # Gated/withheld tools (wait_background / read_image / spawn_agent /
    # plan_edit) stay in DISPATCH so direct callers (tests, hooks, admin
    # scripts) can still exercise their bodies, but a MODEL that emits one
    # before the gate opened it must get actionable refusal — the schema was
    # never offered this turn, so its very next move after a bare "unknown
    # tool" (or worse, after it actually ran) would be to conclude the
    # capability doesn't exist. Returned before the pre-hook on purpose: these
    # tools are not being offered, so a policy veto would be the wrong story.
    # A sub-agent loop is exempt: its restricted table has none of these
    # entries, so the lookup below already returns the sub-agent-specific
    # refusal (and the parent's gate state says nothing about what an agent
    # thread may call).
    from ._subagent import _subagent_ctx as _gate_ctx
    if _gate_ctx() is None:
        if name == "wait_background" and not st._SESSION_HAS_BACKGROUNDED:
            return _unknown_tool_error(name)
        if name == "read_image" and not (_m.NX_VISION and st._NATIVE_TOOL_CALLS_CONFIRMED):
            return _unknown_tool_error(name)
        if name == "spawn_agent" and not _m.NX_SUBAGENTS:
            return _unknown_tool_error(name)
        if name == "plan_edit" and not plan_active():
            return _unknown_tool_error(name)
    # Malformed-args circuit breaker: the tool was dropped for cause (this
    # endpoint keeps truncating its args), so refuse with the same wording the
    # retry nudge uses — "unknown tool" would read as the tool never existed
    # and invite the model to invent a substitute action instead of changing
    # how it formats the call.
    if _breaker_broken(name):
        from ._turn import _malformed_args_nudge  # call-time: _turn imports this module
        return _malformed_args_nudge(name, broken_now=True)
    # A spawn_agent() sub-loop installs a read-only table (per-thread, so 2+
    # concurrent sub-agents each keep their own); the sub-agent's tool calls
    # resolve against it (so "write_file" inside a subagent is simply an
    # unknown tool), and nested spawn_agent calls are refused by _subagent.py.
    from ._subagent import _subagent_table, _subagent_active
    table = _subagent_table() or DISPATCH
    fn = table.get(name)
    name_note = None  # set by the tool-name resolution branch below
    if not fn:
        if _subagent_active():
            return f"ERROR: {name} is not available to the read-only sub-agent (parent must do it)"
        # Tool-NAME repair (0.29.0, nx/_argrepair.resolve_tool_name): one alias
        # or one high-similarity match gets executed under its real name with
        # the note riding the result; ambiguity falls through to the
        # unknown-tool error, which lists the close names instead of guessing.
        from ._argrepair import resolve_tool_name
        # The gates the top of run_tool() applies to an EXACT name must be
        # re-applied to the RESOLVED one (0.40): the early breaker check sees
        # only the emitted spelling, so a breaker-dropped `multi_edit` used to
        # run when the model typed `multi_edits`, and a typo could repair its
        # way onto a tool the session deliberately withholds.
        resolved, name_note = resolve_tool_name(name)
        if resolved and table.get(resolved):
            refused = _withheld_tool_error(resolved)
            if refused:
                return refused
            name, fn = resolved, table[resolved]
        else:
            return _unknown_tool_error(name)
    # Schema-level arg repair (NX_ARG_REPAIR, nx/_argrepair.py) runs BEFORE
    # the pre-hook (0.29.5 reorder, was after): a policy keyed on `path` must
    # see the repaired `path`, not the model's `pth`. Coerces wrong-typed args
    # and renames near-miss keys against the tool's declared schema BEFORE
    # anything downstream (hook, cache key, args preview, body) sees them; the
    # note rides the result so the model learns what was normalized. Disabled /
    # unknown tool / malformed args = pass-through.
    from ._argrepair import repair_args
    args, arg_notes = repair_args(name, args)
    if name_note:
        arg_notes = [name_note] + arg_notes
    # Required-args pre-check (0.29.0): after repair (a renamed key counts as
    # present), before hook/cache/dispatch — the body would otherwise die with
    # a TypeError the model can't act on. Never fabricates the missing value.
    from ._argrepair import check_required
    req_err = check_required(name, args)
    if req_err:
        return f"[arg repair: {', '.join(arg_notes)}] {req_err}" if arg_notes else req_err
    # Hard write fence (2.1): a deterministic boundary, not a judgement —
    # before hook, cache, undo snapshot, approval and body. The classifier and
    # the Y/n prompt stay downstream for everything the fence allows through
    # (see nx/_fence.py's docstring for why a fence is not a risk level).
    from ._fence import check_write_fence
    fence = check_write_fence(name, args)
    if fence:
        return fence
    # User/org pre-hook (NX_PRE_TOOL_HOOK): JSON {tool,args} on stdin; non-zero
    # exit vetoes the call. Checked before any cache/approval machinery so a
    # policy can block even cached reads — and (since 0.29.5) AFTER arg repair,
    # so the payload carries the same normalized args the body will receive.
    from ._hooks import pre_tool_hook
    veto = pre_tool_hook(name, args)
    if veto:
        return f"ERROR: {veto}"
    # Background-completion notice: on the first tool call of a turn, check
    # whether any tracked bg job finished since the model last looked, and
    # stage a one-time runtime note (flushed with the other pending notices).
    # The scan body lives in _run_bg_completion_scan(), shared with the wake
    # path (nx/_autowake.py), whose text-only turns otherwise never run it.
    # Skipped inside a sub-agent loop (run_tool is shared machinery): the
    # notices belong to the PARENT's transcript — an agent thread must never
    # consume/stage them into its own scoped messages, and async agents run
    # turns long after the parent's per-turn reset, so the flag alone can't
    # keep them out anymore.
    from ._subagent import _subagent_ctx
    from ._batch import _batch_active
    if not st._BG_SCAN_DONE and _subagent_ctx() is None and not _batch_active():
        st._BG_SCAN_DONE = True
        _run_bg_completion_scan()
    # Delegation streak counter (runs BEFORE the cache lookup so a cache-hit
    # read still counts): exploration calls accumulate, a spawn_agent resets,
    # anything else (an edit, a bash run, a todo) breaks the streak — those
    # mean the model is acting on what it read, not surveying. Skipped inside
    # a sub-agent/batch loop: those threads don't own the parent's notice
    # channel and a sub-agent can't fan out anyway.
    if name == "spawn_agent":
        st._READ_STREAK = 0
    else:
        from ._subagent import _subagent_ctx as _dc_ctx
        from ._batch import _batch_active as _dc_batch
        if name in ("read_file", "list_dir", "search_files"):
            if _dc_ctx() is None and not _dc_batch() and _DELEGATION_NUDGE_THRESHOLD > 0:
                st._READ_STREAK += 1
                if (st._READ_STREAK >= _DELEGATION_NUDGE_THRESHOLD
                        and not st._DELEGATION_NUDGED and _delegation_enabled()):
                    st._DELEGATION_NUDGED = True
                    st._PENDING_DELEGATION_NOTICE = DELEGATION_STREAK_NOTICE
        else:
            st._READ_STREAK = 0
    if name in _CACHEABLE_READ_TOOLS:
        cache_key = _tool_cache_key(name, args)
        cached = st._TOOL_RESULT_CACHE.get(cache_key)
        if cached is not None and cache_valid(cached):
            # A cache hit is still a read that happened: fire the post-hook
            # (with "cached": true) so an audit trail sees every read, not
            # just the miss that populated the entry. Pre-hook already ran
            # above, so a vetoed call can never serve from the cache.
            from ._hooks import post_tool_hook
            post_tool_hook(name, args, cached.result, cached=True)
            # Breaker decay (0.29.9): a cache hit is still a successful
            # dispatch of exactly the tool the endpoint kept truncating — the
            # endpoint's failure was in the ARGS stream, not the tool itself.
            _breaker_decay(name)
            # A parallel fan-out still shows a cache-hit read: until now a
            # served-from-cache call printed NOTHING at all, so it looked like
            # the call had been dropped. The sequential box keeps its existing
            # (silent) cache-hit behavior — only the group box gains the line.
            from ._subagent import _subagent_ctx as _cache_ctx
            _rec = current_recorder() if _cache_ctx() is None else None
            if _rec is not None:
                _h, _, _d = tool_display(name, args)
                _rec.blocks.append(ToolBlock(name=name, header=_h, detail=_d,
                                             cached=True))
            return cached.result
    # sed_edit snapshots its own (potentially multi-file) checkpoint internally,
    # since a glob path isn't a real file to snapshot here.
    undo_snapshot = _undo_snapshot(args.get("path")) if name in ("write_file", "edit_file", "multi_edit", "delete_file") else None
    # newline so the tool arrow gets its own line — streamed text uses end=""
    # and would otherwise run straight into the indicator. The header and args
    # preview come from tool_display(), shared with the grouped renderer.
    header, arg_preview, detail = tool_display(name, args)
    # Tag write-tier sub-agent calls in the box so a diff prompt that appears
    # mid-sub-agent (the model's own loop output is suspended) is visibly the
    # sub-agent's edit, not the parent's. The tag rides the header line only —
    # the action string inside _confirm stays the standard shape, so allow
    # rules / risk hints match sub-agent and parent writes alike.
    from ._subagent import _subagent_ctx, _subagent_quiet
    ctx = _subagent_ctx()
    if ctx is not None and ctx.write and name in ("write_file", "edit_file", "multi_edit", "sed_edit"):
        header = f"{header} (sub-agent)"
    # A quiet (async-agent) loop renders NO tool box: the box, its spinner,
    # and st._ACTIVE_SPINNER all belong to the parent's terminal while this
    # agent runs detached. Results still flow to the agent's own transcript.
    quiet = _subagent_quiet()
    # A parallel fan-out worker owns neither the terminal nor a spinner: it
    # records a ToolBlock (nx/_toolblocks) and the dispatching thread draws the
    # group. Without this, N pool threads each printed half a box while N
    # spinner threads erased one another's lines mid-print. Two exclusions: a
    # thread with a recorder but a live sub-agent context (a sync agent's reads,
    # which run ON the parent's fan-out thread) records nothing, and neither
    # does anything inside a sub-agent loop — those belong to the agent's own
    # transcript, not the parent's group box. `quiet` deliberately plays no
    # part here: a fan-out worker sets it precisely to mean "draw nothing",
    # yet its call must still be recorded for the group.
    recorder = current_recorder() if ctx is None else None
    buffered = recorder is not None
    if not quiet and not buffered:
        print(f"\n{C.ACCENT}  ┌─ {header}{RESET}")
        print(f"{C.ACCENT}  │ {RESET}{DIM}{arg_preview}{RESET}")
    # Animate the gap between "cyan args line" and "cyan result line". Tool
    # bodies can take a while — _confirm makes a network round-trip to the
    # risk classifier, run_bash can run for tens of seconds, write_file on a
    # big payload takes a beat — and without this the user just sees a frozen
    # screen after the green model output finishes. _confirm pauses/resumes
    # us around its own I/O so the auto-allow line / Y/n prompt aren't clobbered.
    spinner = LifeSpinner(label=f"running {name}")
    # 0.35.0: the line also counts the seconds the call has been in flight. A
    # 40s run_bash and a 0.1s one used to be indistinguishable, so a slow call
    # read as a hang and people Esc'd out of work that was fine. A tool can add
    # its own progress with nx._progress.note_progress("compiling 3/40").
    spinner.show_elapsed = True
    if not quiet and not buffered:
        spinner.start()
        st._ACTIVE_SPINNER = spinner
    try:
        result = fn(**args)
    except _EscToChat:
        # Close the tool box before propagating so it isn't left visually open.
        if not quiet and not buffered:
            print(f"{C.ACCENT}  └─{RESET} {DIM}(escaped){RESET}")
        raise  # not an error — control-flow signal back to model_turn/REPL
    except Exception as e:  # noqa: BLE001 — surface any tool error back to the model
        result = f"ERROR: {type(e).__name__}: {e}"
    finally:
        if not quiet and not buffered:
            st._ACTIVE_SPINNER = None
            spinner.stop()
    # Breaker decay (0.29.9): the tool executed cleanly — one success clears
    # its malformed-args streak (and re-arms a broken tool). Kept after the
    # error fold so an ERROR result does NOT count as the clean dispatch that
    # earns the re-arm (the endpoint may still be truncating).
    if isinstance(result, str) and not result.startswith(("ERROR", "DENIED")):
        _breaker_decay(name)
    # Arg-repair note rides the result, the way edit_file's whitespace-
    # tolerant note does: the model sees what was normalized, never silently.
    if arg_notes and isinstance(result, str):
        result = f"[arg repair: {', '.join(arg_notes)}] {result}"
    # Post-edit structural check: for a successful .py write/edit, parse the
    # new contents in-process (~1ms) and append the verdict to the result, so
    # a broken edit is surfaced on the turn that caused it. Also records the
    # edit for the NX_VERIFY_CMD settle log. Best-effort — never raises.
    if (name in ("write_file", "edit_file", "multi_edit")
            and isinstance(result, str) and not result.startswith(("ERROR", "DENIED"))):
        from ._verify import record_py_edit, _py_path
        vp = _py_path(args.get("path"))
        if vp:
            problem = None
            checked = False
            try:
                problem = record_py_edit(vp)
                checked = True
            except Exception:
                pass  # verification is best-effort — never fail the edit
            if checked:
                result = f"{result}\n{problem or os.path.basename(vp) + ' — py_compile OK'}"
    # Dedup + cap the result BEFORE it enters the message history (the model and
    # the on-screen preview both get this sanitized form). Bounding repetitive
    # tool output is the cheapest defense against context-copying collapse.
    # A content-parts result (read_image) is not a string — it bypasses text
    # sanitization entirely and gets a one-line display hint instead. The
    # hint is built by _parts_preview so the text parts ("[image: path, N
    # bytes]") stay visible and the base64 payload never reaches a terminal,
    # /tooloutput, or the post-hook JSON (any of which would have to carry
    # megabytes it cannot trim).
    if isinstance(result, list):
        from ._images import _parts_preview
        preview_lines = [_parts_preview(result)]
    else:
        result = _sanitize_tool_result(result)
        # box the result; the preview is a further display-only trim of the (already
        # context-bounded) result
        preview_lines = result.splitlines()
        if len(preview_lines) > 8:
            hidden = len(preview_lines) - 8
            preview_lines = preview_lines[:8] + [f"... [{hidden} more lines]"]
    # User/org post-hook (NX_POST_TOOL_HOOK): observer only — failures are
    # logged, never raised, and the result is unaffected. Parts results are
    # serialized for the hook's JSON stdin — via _parts_preview, so the hook
    # sees the same one-line hint the terminal does, not a base64 dump.
    from ._hooks import post_tool_hook
    post_tool_hook(name, args, result if isinstance(result, str) else preview_lines[0])
    # Stash the sanitized result of the most recent tool call regardless of
    # visibility, so /tooloutput last can replay it after the fact even when the
    # result body was collapsed live (mirrors _LAST_REASONING for reasoning).
    st._LAST_TOOL_NAME = name
    st._LAST_TOOL_OUTPUT = result if isinstance(result, str) else preview_lines[0]
    # todo_write colors each item by status itself (render_todos_display:
    # done=green, in_progress=yellow, pending=orange) and sanitizes the item
    # text, so it goes to the screen verbatim inside the box — not the generic
    # strip + uniform-recolor path used for other tool/bash output.
    if buffered:
        # Recorded for the group box: one header line always, the result body
        # only when tool output is visible (the group footer carries the single
        # collapse hint instead of one per call). A failure also carries its
        # first line as `note`, so a collapsed group still says why it failed.
        body: list[str] = []
        if st.TOOL_OUTPUT_VISIBLE:
            if name == "todo_write":
                body = render_todos_display().splitlines()
            else:
                body = [f"{C.TOOL_OUTPUT}{strip_ansi_for_display(line)}{RESET}"
                        for line in preview_lines]
        failed = isinstance(result, str) and result.startswith(("ERROR", "DENIED"))
        note = ""
        if failed:
            first = strip_ansi_for_display(result.splitlines()[0])
            note = first[:100] + "..." if len(first) > 100 else first
        recorder.blocks.append(ToolBlock(
            name=name, header=header, detail=detail,
            body=body, note=note, failed=failed))
    elif quiet:
        pass  # quiet (async-agent) loop: no box, no result echo
    elif name == "todo_write":
        disp = render_todos_display().splitlines()
        if len(disp) > 8:
            disp = disp[:8] + [f"{DIM}... [{len(disp) - 8} more lines]{RESET}"]
        for line in disp:
            print(f"{C.ACCENT}  │ {RESET}{line}{RESET}")
    elif st.TOOL_OUTPUT_VISIBLE:
        for line in preview_lines:
            print(f"{C.ACCENT}  │ {RESET}{C.TOOL_OUTPUT}{strip_ansi_for_display(line)}{RESET}")
    else:
        # Collapsed: the header/args lines already named the call; only the
        # result body is hidden. Keep the box closed with a one-line hint.
        print(f"{C.ACCENT}  │ {RESET}{DIM}… (output collapsed — /tooloutput to expand){RESET}")
    if not quiet and not buffered:
        print(f"{C.ACCENT}  └─{RESET}")
    if name in _CACHEABLE_READ_TOOLS:
        # git_* results reflect .git/ state, which the root dir's mtime never
        # tracks (commits write under .git/ only) — store_cacheable_result
        # stamps them so a hit can detect a commit/checkout that happened
        # behind the cache's back.
        store_cacheable_result(st._TOOL_RESULT_CACHE, cache_key, name, args, result)
    elif name in ("write_file", "edit_file", "multi_edit", "delete_file") and not result.startswith("ERROR"):
        _invalidate_tool_cache_path(args.get("path"))
        # an edit changes git state (status/diff at least) — git_* entries are
        # keyed on the repo root dir, which the path-based drop never matches.
        drop_vcs_entries(st._TOOL_RESULT_CACHE)
        if result != "DENIED by user":
            # push_if_absent (0.47.0, ROADMAP 2.2): a body that pushes its own
            # checkpoint (delete_file, sed_edit, apply_patch) is not stacked
            # twice — the dispatcher only fills the gap for the single-path
            # write tools whose bodies don't push. One approved call, one
            # /undo press, one depth slot.
            _undo_push_once(name, [undo_snapshot])
    elif name == "run_bash":
        # bash can touch any path; entries whose backing file is unchanged
        # (re-stat) stay valid — only mtime-changed/vanished entries are dropped.
        # Git state can change (commit/reset/status via worktree edit, or .git
        # removed) in ways no stamp or mtime sees, so git_* entries always go.
        drop_changed_entries(st._TOOL_RESULT_CACHE)
        drop_vcs_entries(st._TOOL_RESULT_CACHE)
    elif name in ("memory_write", "memory_delete") and not result.startswith("ERROR") and result != "DENIED by user":
        # A new (or removed) memory invalidates any cached memory_search
        # index/snippets — regardless of the scope the write/delete targeted.
        for k in [k for k in st._TOOL_RESULT_CACHE if k[0] == "memory_search"]:
            del st._TOOL_RESULT_CACHE[k]
    if name == "run_bash" and result.startswith("[background]") and not st._SESSION_HAS_BACKGROUNDED:
        st._SESSION_HAS_BACKGROUNDED = True
        st._PENDING_BACKGROUND_NOTICE = BACKGROUND_GUIDANCE
        # This session now has background work whose completion only a LATER
        # turn can observe — arm the auto-continuation poller (0.36.0) so the
        # finish notice reaches the model without the user typing first.
        from ._autowake import arm_wake
        arm_wake()
    # (No seen-marking here: a just-backgrounded job is still alive, so the
    # completion scan's alive-check already skips it — and if it exits before
    # the next scan, announcing "exited (code N)" is exactly what the model
    # needs to hear, since it was told the job was backgrounded.)
    if name == "edit_file" and not result.startswith("ERROR"):
        st._CONSECUTIVE_EDITS += 1
        if st._CONSECUTIVE_EDITS >= _MULTI_EDIT_NUDGE_THRESHOLD and not st._MULTI_EDIT_NUDGED:
            st._MULTI_EDIT_NUDGED = True
            st._PENDING_MULTI_EDIT_NOTICE = MULTI_EDIT_GUIDANCE
    elif name != "edit_file":
        st._CONSECUTIVE_EDITS = 0
    # read_file just returned a bg job's log in full? If the job has already
    # exited, its output now sits in the transcript — mark it seen so the
    # completion scan never wakes the model to say what it just read (0.37.4).
    # (run_bash/wait_background mark their own inline deliveries inside
    # _bash_jobs; this covers the model fetching the log by path instead.)
    if name == "read_file" and isinstance(result, str) and not result.startswith("ERROR"):
        try:
            from ._bash_jobs import _mark_log_consumed
            for _p in (args.get("path") if isinstance(args.get("path"), list) else [args.get("path")]):
                _mark_log_consumed(_p)
        except Exception:
            pass
    return result


def _tool_args_parse_error(c: dict[str, Any], err: json.JSONDecodeError, finish_reasons: list[str] | None = None) -> str:
    name = c.get("name") or "tool"
    args_text = c.get("args") or ""
    detail = f"{name} arguments are not valid JSON"
    if args_text:
        detail += f" ({_abbr(len(args_text))} chars"
        if finish_reasons:
            detail += f"; finish_reason={','.join(finish_reasons)}"
        detail += f"; {err.msg} at char {err.pos})"
    else:
        detail += f" ({err.msg} at char {err.pos})"
    return detail
