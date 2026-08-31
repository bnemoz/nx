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
from typing import Any

from . import _state as st
from ._ansi import DIM, RESET, C, LifeSpinner, strip_ansi_for_display
from ._tool_cache import cache_store_entry, cache_valid, drop_changed_entries
from ._approval import _EscToChat
from ._bash_jobs import run_bash, wait_background
from ._edit_tools import write_file, edit_file, multi_edit, sed_edit, apply_patch, delete_file
from ._memory_cmd import memory_write, memory_search, memory_delete, _load_memory_note
from ._metrics import _abbr
from ._orientation import _load_orientation_note
from ._text_fallback import _sanitize_tool_result
from ._todo import todo_write, render_todos_display
from ._plan import plan_edit, plan_active, check_gate
from ._subagent import spawn_agent
from ._images import read_image
from ._tools import read_file, list_dir, search_files, git_status, git_diff, git_log
from ._undo import snapshot as _undo_snapshot, push_checkpoint as _undo_push

DISPATCH = {
    "read_file": read_file, "write_file": write_file, "edit_file": edit_file,
    "list_dir": list_dir, "run_bash": run_bash, "wait_background": wait_background,
    "search_files": search_files, "multi_edit": multi_edit, "sed_edit": sed_edit,
    "git_status": git_status, "git_diff": git_diff, "git_log": git_log,
    "todo_write": todo_write, "apply_patch": apply_patch, "delete_file": delete_file,
    "memory_write": memory_write, "memory_search": memory_search,
    "memory_delete": memory_delete,
    "spawn_agent": spawn_agent, "read_image": read_image,
    "plan_edit": plan_edit,
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
st._CONSECUTIVE_EDITS = 0  # count of back-to-back successful edit_file calls (any other tool call resets it)
st._MULTI_EDIT_NUDGED = False  # flips True once the batching nudge has fired this session — see run_tool()
st._PENDING_MULTI_EDIT_NOTICE = None  # set by run_tool() on that nudge; flushed once by model_turn (see _flush_pending_background_notice)

_MULTI_EDIT_NUDGE_THRESHOLD = 2  # ≥2 single edits in a row before the batching nudge fires


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


def _initial_messages() -> list[dict[str, Any]]:
    """Fresh-session message list: SYSTEM, plus a one-time AGENTS.md note
    (project-root convention, same idea as CLAUDE.md) if one exists. Kept out
    of SYSTEM itself so the per-turn token floor doesn't grow with project
    size — see _agents.py."""
    from ._agents import _load_agents_md_note
    import nx as _m
    messages = [{"role": "system", "content": _m.SYSTEM}]
    note = _load_agents_md_note(getattr(st, "PROJECT_ROOT", None))
    if note is not None:
        messages.append(note)
    # Memory-awareness note: titles only (capped) so the model knows what's
    # already remembered without paying for the bodies — see _memory_cmd.py.
    # Paid once per fresh session, not folded into SYSTEM so the per-turn
    # token floor stays flat when no memories exist (see test_initial_context).
    mem_note = _load_memory_note(getattr(st, "PROJECT_ROOT", None))
    if mem_note is not None:
        messages.append(mem_note)
    # Orientation note: the real cwd (SYSTEM only says "cwd = user's current
    # directory" without the actual path) plus a pointer at the approval-free
    # read-only tools. Paid once per fresh session, not folded into SYSTEM so
    # the per-turn token floor stays flat. Suppressed by NX_ORIENTATION=0.
    orient_note = _load_orientation_note()
    if orient_note is not None:
        messages.append(orient_note)
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


def _scan_background_completions() -> str | None:
    """One-time notice for tracked background jobs that finished since the
    model last looked (checked on the first tool call of each turn).

    The model must otherwise REMEMBER to wait_background/read_file a job it
    backgrounded turns ago; a build that failed 20 minutes ago doesn't
    announce itself. Returns the note text (or None): lists finished jobs with
    exit codes once, then marks them seen — no repeated noise."""
    from ._bash_jobs import list_bg_jobs, _read_bg_exit_status
    from ._verify import _is_verify_job  # verify jobs report via _stage_verify_outcome instead
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
        seen.add(pid)
        finished.append(j)
        code = _read_bg_exit_status(pid)[1]
        tail = (j.get("tail") or "").strip()[:80]
        code_s = "unknown" if code is None else str(code)
        notes.append(f"  pid {pid} exited (code {code_s})" + (f" — tail: {tail}" if tail else ""))
    st._BG_NOTICE_SEEN = seen
    if not notes:
        return None
    return ("[Runtime note: background job(s) you started have finished:\n"
            + "\n".join(notes)
            + "\nCheck their output with read_file(log) or wait_background before relying on them.]")

_CACHEABLE_READ_TOOLS = ("read_file", "list_dir", "search_files", "git_status", "git_diff", "git_log", "memory_search")


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


def run_tool(name: str, args: dict[str, Any]) -> str:
    # Plan-mode write gate: refuse mutating tools while a plan is active, with
    # the gate message the model needs to act on (plan_edit, or ask for
    # /plan done). Checked before every other gate so a gated call costs no
    # classifier round-trip, no approval prompt, and no sub-agent dispatch.
    gated = check_gate(name)
    if gated:
        return gated
    # A spawn_agent() sub-loop installs a read-only table (per-thread, so 2+
    # concurrent sub-agents each keep their own); the sub-agent's tool calls
    # resolve against it (so "write_file" inside a subagent is simply an
    # unknown tool), and nested spawn_agent calls are refused by _subagent.py.
    from ._subagent import _subagent_table, _subagent_active
    table = _subagent_table() or DISPATCH
    fn = table.get(name)
    if not fn:
        if _subagent_active():
            return f"ERROR: {name} is not available to the read-only sub-agent (parent must do it)"
        return f"ERROR: unknown tool {name}"
    # User/org pre-hook (NX_PRE_TOOL_HOOK): JSON {tool,args} on stdin; non-zero
    # exit vetoes the call. Checked before any cache/approval machinery so a
    # policy can block even cached reads.
    from ._hooks import pre_tool_hook
    veto = pre_tool_hook(name, args)
    if veto:
        return f"ERROR: {veto}"
    # Background-completion notice: on the first tool call of a turn, check
    # whether any tracked bg job finished since the model last looked, and
    # stage a one-time runtime note (flushed with the other pending notices).
    # Skipped inside a sub-agent loop (run_tool is shared machinery): the
    # notices belong to the PARENT's transcript — an agent thread must never
    # consume/stage them into its own scoped messages, and async agents run
    # turns long after the parent's per-turn reset, so the flag alone can't
    # keep them out anymore.
    from ._subagent import _subagent_ctx
    if not st._BG_SCAN_DONE and _subagent_ctx() is None:
        st._BG_SCAN_DONE = True
        note = _scan_background_completions()
        if note:
            st._PENDING_BG_COMPLETION_NOTICE = note
        # NX_VERIFY_CMD job just finished? Its outcome is staged as its own
        # pending notice (independent of the generic bg-completion note).
        from ._verify import _stage_verify_outcome
        try:
            _stage_verify_outcome()
        except Exception:
            pass
        # Async sub-agent finished since the last look? Its answer is
        # normally staged by the worker itself; this sweep catches any
        # finished-but-unstaged stragglers (idempotent per job id).
        from ._subagent import _stage_subagent_completions
        try:
            _stage_subagent_completions()
        except Exception:
            pass
    if name in _CACHEABLE_READ_TOOLS:
        cache_key = _tool_cache_key(name, args)
        cached = st._TOOL_RESULT_CACHE.get(cache_key)
        if cached is not None and cache_valid(cached):
            return cached.result
    # sed_edit snapshots its own (potentially multi-file) checkpoint internally,
    # since a glob path isn't a real file to snapshot here.
    undo_snapshot = _undo_snapshot(args.get("path")) if name in ("write_file", "edit_file", "multi_edit", "delete_file") else None
    # newline so the tool arrow gets its own line — streamed text uses end=""
    # and would otherwise run straight into the indicator
    arg_preview = json.dumps(args)
    if len(arg_preview) > 120:
        arg_preview = arg_preview[:117] + "..."
    # Surface the target path in the header line for path-taking tools, so the
    # file a write_file/edit_file/multi_edit touches is obvious at a glance even
    # when the args preview (usually a big old/new diff body) gets truncated and
    # buries the path. Falls back to just the name when there's no usable path.
    header = name
    path_arg = args.get("path")
    if isinstance(path_arg, list):
        path_arg = path_arg[0] if path_arg else None
    if isinstance(path_arg, str) and path_arg not in (".", ""):
        header = f"{name} {path_arg}"
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
    if not quiet:
        print(f"\n{C.ACCENT}  ┌─ {header}{RESET}")
        print(f"{C.ACCENT}  │ {RESET}{DIM}{arg_preview}{RESET}")
    # Animate the gap between "cyan args line" and "cyan result line". Tool
    # bodies can take a while — _confirm makes a network round-trip to the
    # risk classifier, run_bash can run for tens of seconds, write_file on a
    # big payload takes a beat — and without this the user just sees a frozen
    # screen after the green model output finishes. _confirm pauses/resumes
    # us around its own I/O so the auto-allow line / Y/n prompt aren't clobbered.
    spinner = LifeSpinner(label=f"running {name}")
    if not quiet:
        spinner.start()
        st._ACTIVE_SPINNER = spinner
    try:
        result = fn(**args)
    except _EscToChat:
        # Close the tool box before propagating so it isn't left visually open.
        if not quiet:
            print(f"{C.ACCENT}  └─{RESET} {DIM}(escaped){RESET}")
        raise  # not an error — control-flow signal back to model_turn/REPL
    except Exception as e:  # noqa: BLE001 — surface any tool error back to the model
        result = f"ERROR: {type(e).__name__}: {e}"
    finally:
        if not quiet:
            st._ACTIVE_SPINNER = None
            spinner.stop()
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
    # sanitization entirely and gets a one-line display hint instead.
    if isinstance(result, list):
        preview_lines = [f"[image attached — {len(result)} part(s), base64 payload not shown]"]
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
    # serialized for the hook's JSON stdin.
    from ._hooks import post_tool_hook
    post_tool_hook(name, args, result if isinstance(result, str) else json.dumps(result)[:2000])
    # Stash the sanitized result of the most recent tool call regardless of
    # visibility, so /tooloutput last can replay it after the fact even when the
    # result body was collapsed live (mirrors _LAST_REASONING for reasoning).
    st._LAST_TOOL_NAME = name
    st._LAST_TOOL_OUTPUT = result if isinstance(result, str) else preview_lines[0]
    # todo_write colors each item by status itself (render_todos_display:
    # done=green, in_progress=yellow, pending=orange) and sanitizes the item
    # text, so it goes to the screen verbatim inside the box — not the generic
    # strip + uniform-recolor path used for other tool/bash output.
    if quiet:
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
    if not quiet:
        print(f"{C.ACCENT}  └─{RESET}")
    if name in _CACHEABLE_READ_TOOLS:
        cache_path = args.get("path", ".")
        if isinstance(cache_path, list):
            cache_path = cache_path[0] if cache_path else "."
        st._TOOL_RESULT_CACHE[cache_key] = cache_store_entry(result, cache_path)
    elif name in ("write_file", "edit_file", "multi_edit", "delete_file") and not result.startswith("ERROR"):
        _invalidate_tool_cache_path(args.get("path"))
        if result != "DENIED by user":
            _undo_push(name, [undo_snapshot])
    elif name == "run_bash":
        # bash can touch any path; entries whose backing file is unchanged
        # (re-stat) stay valid — only mtime-changed/vanished entries are dropped.
        drop_changed_entries(st._TOOL_RESULT_CACHE)
    elif name in ("memory_write", "memory_delete") and not result.startswith("ERROR") and result != "DENIED by user":
        # A new (or removed) memory invalidates any cached memory_search
        # index/snippets — regardless of the scope the write/delete targeted.
        for k in [k for k in st._TOOL_RESULT_CACHE if k[0] == "memory_search"]:
            del st._TOOL_RESULT_CACHE[k]
    if name == "run_bash" and result.startswith("[background]") and not st._SESSION_HAS_BACKGROUNDED:
        st._SESSION_HAS_BACKGROUNDED = True
        st._PENDING_BACKGROUND_NOTICE = BACKGROUND_GUIDANCE
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
