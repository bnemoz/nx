"""The fresh-session message list and the per-session runtime nudge strings.

Split out of `_dispatch.py` (which keeps run_tool — the site that fires the
nudges). Three runtime notes (backgrounding guidance, multi_edit
batching — both paid once — and the delegation streak, which RE-ARMS) plus the
resume-time "already shown" checks and `_initial_messages()`, the canonical
fresh-session head built from _prefix.prefix_notes() (PERF.md PF-15)."""
from __future__ import annotations

from typing import Any

from ._env import _env_int

_MULTI_EDIT_NUDGE_THRESHOLD = 2  # ≥2 single edits in a row before the batching nudge fires

# ≥ this many such reads before the delegation nudge fires. The streak resets on
# each fire, so the nudge re-arms every 2x threshold additional consecutive
# reads (5, then 15, then 25 cumulative). Lowered from 8 to 5 to make steering
# more aggressive: an edit usually touches 1-3 files, so 5 solo exploration reads
# already suggests a sweep that should have been a fan-out. Re-firing keeps a
# model that ignored the first nudge from grinding a long sweep undisturbed.
# NX_DELEGATION_NUDGE_READS=0 disables the nudge.
_DELEGATION_NUDGE_THRESHOLD = _env_int("NX_DELEGATION_NUDGE_READS", 5)


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
# st._SESSION_HAS_BACKGROUNDED / st._PENDING_BACKGROUND_NOTICE in _dispatch and
# the flush in _runtime_notes._flush_pending_background_notice). Appending it
# at the tail keeps every earlier turn's bytes untouched, so it never
# invalidates a server's prompt-prefix cache.
BACKGROUND_GUIDANCE = (
    "[Runtime note: a command you ran just backgrounded (didn't finish within "
    "the short poll window) — wait_background is now available as a tool. "
    "Guidance for backgrounded commands:\n"
    "- Do NOT use `sleep N && …` as a chaining trick — the whole command is "
    "detached (own session) so it backgrounds too. Only pass `timeout=N` to run_bash when "
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

# Fourth instance of the same one-shot pattern: a YOLO session that passed
# async:false on a write-tier spawn. The session-start delegation note is
# built once, so a session that STARTED in yolo already heard the right
# contract — this catches the model that passes explicit foreground anyway
# (stale training prior) and, since it fires on the act itself, also covers a
# mid-session /yolo toggle whose note is stale. NX_SUBAGENT_ASYNC=0 (a
# legitimate user default) never fires it: only an explicit async:false does.
YOLO_ASYNC_GUIDANCE = (
    "[Runtime note: this session runs in yolo mode — background (async) "
    "write-tier sub-agents are permitted and run concurrently; async:false is "
    "only needed when you must have the answer this turn. Prefer omitting "
    "async and spawning independent write agents in one turn.]"
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

# Third instance of the same pattern, with one difference: it RE-ARMS. Staged
# when the model has made _DELEGATION_NUDGE_THRESHOLD exploration calls itself
# without spawning a sub-agent; the fire resets the streak, so ignoring it once
# only buys the model 2x threshold more reads before it lands again (see
# run_tool()'s counter and st._DELEGATION_NUDGE_COUNT). Only staged while the
# spawn_agent schema is actually offered (NX_SUBAGENTS=1) — see
# _delegation_note._delegation_enabled.
DELEGATION_STREAK_NOTICE = (
    "[Runtime note: you've read or searched many files yourself without using "
    "spawn_agent. The remaining investigation is exactly what sub-agents are "
    "for: issue one spawn_agent call per remaining independent question in a "
    "single turn (they run concurrently, return only their findings, and keep "
    "their intermediate reads out of this context), then work from their "
    "answers. If the files you're reading are the exact ones an edit already "
    "in hand needs, ignore this and continue.]"
)
