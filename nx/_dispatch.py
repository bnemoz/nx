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
from ._approval import _EscToChat
from ._bash_jobs import run_bash, wait_background
from ._edit_tools import write_file, edit_file, multi_edit, sed_edit, apply_patch
from ._metrics import _abbr
from ._text_fallback import _sanitize_tool_result
from ._todo import todo_write
from ._tools import read_file, list_dir, search_files, git_status, git_diff, git_log
from ._undo import snapshot as _undo_snapshot, push_checkpoint as _undo_push

DISPATCH = {
    "read_file": read_file, "write_file": write_file, "edit_file": edit_file,
    "list_dir": list_dir, "run_bash": run_bash, "wait_background": wait_background,
    "search_files": search_files, "multi_edit": multi_edit, "sed_edit": sed_edit,
    "git_status": git_status, "git_diff": git_diff, "git_log": git_log,
    "todo_write": todo_write, "apply_patch": apply_patch,
}

st._SESSION_HAS_BACKGROUNDED = False  # flips True the first time a run_bash call in this session actually backgrounds
st._PENDING_BACKGROUND_NOTICE = None  # set by run_tool() on that first flip; flushed once by model_turn (see _flush_pending_background_notice)
st._CONSECUTIVE_EDITS = 0  # count of back-to-back successful edit_file calls (any other tool call resets it)
st._MULTI_EDIT_NUDGED = False  # flips True once the batching nudge has fired this session — see run_tool()
st._PENDING_MULTI_EDIT_NOTICE = None  # set by run_tool() on that nudge; flushed once by model_turn (see _flush_pending_background_notice)

_MULTI_EDIT_NUDGE_THRESHOLD = 2  # ≥2 single edits in a row before the batching nudge fires


def _active_tools() -> list[dict[str, Any]]:
    """The tool schema list to send this turn. wait_background is included
    only once run_bash has actually backgrounded this session — see run_tool()."""
    import nx as _m
    tools = _m.TOOLS
    if not st._SESSION_HAS_BACKGROUNDED:
        tools = [t for t in tools if t["function"]["name"] != "wait_background"]
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
    "detached so it backgrounds too. Only pass `timeout=N` to run_bash when "
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

_CACHEABLE_READ_TOOLS = ("read_file", "list_dir", "search_files", "git_status", "git_diff", "git_log")


def _tool_cache_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    return (name, json.dumps(args, sort_keys=True))


def _invalidate_tool_cache_path(path: str | None) -> None:
    if not path:
        return
    try:
        norm = os.path.abspath(path)
    except (TypeError, ValueError):
        return
    for key, (_result, cached_path) in list(st._TOOL_RESULT_CACHE.items()):
        if cached_path == norm:
            del st._TOOL_RESULT_CACHE[key]


def run_tool(name: str, args: dict[str, Any]) -> str:
    fn = DISPATCH.get(name)
    if not fn:
        return f"ERROR: unknown tool {name}"
    if name in _CACHEABLE_READ_TOOLS:
        cache_key = _tool_cache_key(name, args)
        cached = st._TOOL_RESULT_CACHE.get(cache_key)
        if cached is not None:
            return cached[0]
    # sed_edit snapshots its own (potentially multi-file) checkpoint internally,
    # since a glob path isn't a real file to snapshot here.
    undo_snapshot = _undo_snapshot(args.get("path")) if name in ("write_file", "edit_file", "multi_edit") else None
    # newline so the tool arrow gets its own line — streamed text uses end=""
    # and would otherwise run straight into the indicator
    arg_preview = json.dumps(args)
    if len(arg_preview) > 120:
        arg_preview = arg_preview[:117] + "..."
    print(f"\n{C.ACCENT}  ┌─ {name}{RESET}")
    print(f"{C.ACCENT}  │ {RESET}{DIM}{arg_preview}{RESET}")
    # Animate the gap between "cyan args line" and "cyan result line". Tool
    # bodies can take a while — _confirm makes a network round-trip to the
    # risk classifier, run_bash can run for tens of seconds, write_file on a
    # big payload takes a beat — and without this the user just sees a frozen
    # screen after the green model output finishes. _confirm pauses/resumes
    # us around its own I/O so the auto-allow line / Y/n prompt aren't clobbered.
    spinner = LifeSpinner(label=f"running {name}")
    spinner.start()
    st._ACTIVE_SPINNER = spinner
    try:
        result = fn(**args)
    except _EscToChat:
        # Close the tool box before propagating so it isn't left visually open.
        print(f"{C.ACCENT}  └─{RESET} {DIM}(escaped){RESET}")
        raise  # not an error — control-flow signal back to model_turn/REPL
    except Exception as e:  # noqa: BLE001 — surface any tool error back to the model
        result = f"ERROR: {type(e).__name__}: {e}"
    finally:
        st._ACTIVE_SPINNER = None
        spinner.stop()
    # Dedup + cap the result BEFORE it enters the message history (the model and
    # the on-screen preview both get this sanitized form). Bounding repetitive
    # tool output is the cheapest defense against context-copying collapse.
    result = _sanitize_tool_result(result)
    # box the result; the preview is a further display-only trim of the (already
    # context-bounded) result
    preview_lines = result.splitlines()
    if len(preview_lines) > 8:
        hidden = len(preview_lines) - 8
        preview_lines = preview_lines[:8] + [f"... [{hidden} more lines]"]
    for line in preview_lines:
        print(f"{C.ACCENT}  │ {RESET}{C.TOOL_OUTPUT}{strip_ansi_for_display(line)}{RESET}")
    print(f"{C.ACCENT}  └─{RESET}")
    if name in _CACHEABLE_READ_TOOLS:
        cache_path = args.get("path", ".")
        if isinstance(cache_path, list):
            cache_path = cache_path[0] if cache_path else "."
        st._TOOL_RESULT_CACHE[cache_key] = (result, os.path.abspath(cache_path))
    elif name in ("write_file", "edit_file", "multi_edit") and not result.startswith("ERROR"):
        _invalidate_tool_cache_path(args.get("path"))
        if result != "DENIED by user":
            _undo_push(name, [undo_snapshot])
    elif name == "run_bash":
        # bash can touch any path; cheaper to drop the whole cache than to parse the command.
        st._TOOL_RESULT_CACHE.clear()
    if name == "run_bash" and result.startswith("[background]") and not st._SESSION_HAS_BACKGROUNDED:
        st._SESSION_HAS_BACKGROUNDED = True
        st._PENDING_BACKGROUND_NOTICE = BACKGROUND_GUIDANCE
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
