"""The runtime-note contract: nx's one-time injections into the transcript.

Every one-time injection into the model's transcript (recovery nudges,
backgrounded-job guidance, async sub-agent answers, verification results) is
a user-role message whose content starts with RUNTIME_NOTE_PREFIX and ends
with `]`.

Append-only on purpose (0.40, ROADMAP 1.1): injected notes are appended as
fresh tail messages and never rewrite an earlier one — a rewritten message is
always one the model has ALREADY been sent, so rewriting busts the server's
prompt-prefix cache on exactly the turns with the largest prompt. Consumers:
_turn (nudges, note flushing), _dispatch (staging), _bash_jobs, _occompact,
_compress (keep-window sizing)."""
from __future__ import annotations

from typing import Any

from . import _state as st
from . import _notices

# Every one-time injection into the model's transcript (recovery nudges,
# backgrounded-job guidance, async sub-agent answers, verification results) is
# a user-role message whose content starts with this prefix and ends with `]`.
RUNTIME_NOTE_PREFIX = "[Runtime note: "


def _is_runtime_note(msg: Any) -> bool:
    """True if `msg` is one of nx's own injected notes rather than something the
    user or the model wrote. Several places need to know whether a message
    counts as a real turn boundary (see _last_is_dangling_tool)."""
    content = msg.get("content")
    return (msg.get("role") == "user" and isinstance(content, str)
            and content.lstrip().startswith(RUNTIME_NOTE_PREFIX))


def _nudge_current_user_turn(messages: list[dict[str, Any]], nudge: str) -> None:
    """Inject a recovery nudge as a fresh trailing user message.

    APPEND-ONLY ON PURPOSE (0.40, ROADMAP 1.1). This function used to rewrite
    the last user turn in place, stripping the previous note with a regex and
    re-attaching a new one. That is a prompt-cache bullet: the rewritten message
    is always one the model has ALREADY been sent, so every recovery rung —
    malformed args, empty turn, reasoning stall, length continuation —
    invalidated the server's prefix cache from that point on, on exactly the
    turns with the largest prompt. nx bills itself on `cached:` in the footer
    and documents append-only staging in _flush_pending_background_notice for
    precisely this reason; recovery is now consistent with it. A repeated nudge
    stacks (bounded by each failure family's own retry limit) instead of
    replacing, which also gives the transcript an honest record of how many
    rescues fired.

    Notes are still deduped when the identical one is already the pending tail,
    so a loop that re-enters the same rung with the same text can't double it."""
    note = f"{RUNTIME_NOTE_PREFIX}{nudge}]"
    if messages and messages[-1].get("content") == note:
        return
    messages.append({"role": "user", "content": note})


def _flush_pending_background_notice(messages: list[dict[str, Any]]) -> None:
    """Append BACKGROUND_GUIDANCE once, if run_tool() just saw the first
    backgrounded run_bash result this session. Appends a brand-new tail
    message (never rewrites an earlier one) so it can't invalidate a
    server's prompt-prefix cache. Also drains worker-staged notes that landed
    in the module store outside this loop's notice frame (see
    _flush_module_store_notes).

    Ownership gate (0.156.1): a no-op inside a sub-agent sub-loop. run_tool
    already refuses to STAGE the parent's notices there (_dispatch.py's
    first-tool-of-turn gate); this closes the matching flush hole — a sibling
    async agent's (or a sync agent's) tool-call flush used to drain the
    parent's staged notes, including worker-staged module-store notes, into
    the sub-loop's doomed messages list. The note's `announced`/seen marks
    were already burned at staging, so the parent never re-received it and
    wake_due() saw nothing pending: /tabs showed every agent done while the
    main loop sat un-woken. Notes staged by the sub-loop itself fold back
    into the parent frame at pop (_notices), and parent-owned notes simply
    wait for the parent's own flush — the same rule, both ends."""
    from ._subagent import _subagent_ctx
    if _subagent_ctx() is not None:
        return
    _delivered: set[str] = set()
    if st._PENDING_BACKGROUND_NOTICE:
        messages.append({"role": "user", "content": st._PENDING_BACKGROUND_NOTICE})
        _delivered.add(st._PENDING_BACKGROUND_NOTICE)
        st._PENDING_BACKGROUND_NOTICE = None
    if _notices.get("_PENDING_BG_COMPLETION_NOTICE"):
        messages.append({"role": "user",
                         "content": _notices.get("_PENDING_BG_COMPLETION_NOTICE")})
        _delivered.add(_notices.get("_PENDING_BG_COMPLETION_NOTICE"))
        _notices.set_("_PENDING_BG_COMPLETION_NOTICE", None)
    # Finished async sub-agent answer(s): staged by the agent worker itself
    # (nx/_subagent.py) the moment the sub-loop completes, swept by run_tool()'s
    # first-tool-of-turn scan, and delivered here as the parent's runtime note.
    subagent_note = _notices.get("_PENDING_SUBAGENT_NOTICE")
    if subagent_note:
        messages.append({"role": "user", "content": subagent_note})
        _delivered.add(subagent_note)
        _notices.set_("_PENDING_SUBAGENT_NOTICE", None)
    # Online Context Compact (0.67.0): the post-compaction continuation note
    # (re-issue the plan) staged by _occompact.maybe_compact_after_boundary.
    occ_note = getattr(st, "_PENDING_OCC_REMINDER", None)
    if occ_note:
        messages.append({"role": "user", "content": occ_note})
        st._PENDING_OCC_REMINDER = None
    # The remaining staged notices all share one shape (a runtime-note string
    # on an st slot, consumed once), so they flush in a loop. The delegation
    # streak notice is staged by run_tool(); NX_VERIFY_CMD's outcome by
    # _dispatch._scan_background_completions.
    for _slot in ("_PENDING_MULTI_EDIT_NOTICE", "_PENDING_VERIFY_NOTICE",
                  "_PENDING_DELEGATION_NOTICE", "_PENDING_YOLO_ASYNC_NOTICE",
                  "_PENDING_WAKE_HOLD_NOTICE", "_PENDING_PTC_NOTICE"):
        _text = _notices.get(_slot)
        if _text:
            messages.append({"role": "user", "content": _text})
            _delivered.add(_text)
            _notices.set_(_slot, None)
    _flush_module_store_notes(messages, _delivered)


# Slots whose notes may be staged into the MODULE store (vars(st)) while a turn
# loop's per-thread notice frame is installed: an async sub-agent's worker
# thread, the wake poller and the verify sweep all run OUTSIDE the parent's
# loop, and _notices.set_() from a frameless thread writes the module store —
# which the parent's frame-scoped reads and the frame-scoped flush above can
# never see. Without this drain the note sits in the module store until the
# turn ends text-only (undelivered), and the wake poller then fires a pointless
# "background job finished — auto-continuing" turn for work the parent could
# have consumed on its own last turn (seen live: every chat using an async
# agent ends with a banner + "Nothing needs doing").
_MODULE_NOTE_SLOTS = ("_PENDING_BG_COMPLETION_NOTICE", "_PENDING_SUBAGENT_NOTICE",
                      "_PENDING_VERIFY_NOTICE", "_PENDING_MULTI_EDIT_NOTICE",
                      "_PENDING_DELEGATION_NOTICE", "_PENDING_YOLO_ASYNC_NOTICE",
                      # 0.160.0: staged framelessly (poller thread / idle REPL),
                      # so the module-store drain is its normal delivery path.
                      "_PENDING_WAKE_HOLD_NOTICE",
                      # 0.162.0 (PTC): the run_tools nudge / stub-source note.
                      "_PENDING_PTC_NOTICE")


def _flush_module_store_notes(messages: list[dict[str, Any]],
                              delivered: set[str]) -> None:
    """Deliver worker-staged notes the frame-based flush above cannot see.

    For each pending-note slot, when the module store holds a non-empty string
    the frame-based flush did NOT already deliver, it is appended and cleared
    in place. `delivered` carries the strings the frame drain appended (the
    frame seeds a pre-existing module-store value by reference, so the
    frame-based flush may have delivered exactly that string while clearing
    only the frame slot — without the guard it would be appended twice).
    Clearing the module slot directly is safe: the frame's slot stays at its
    seed value (None), so the pop fold-back skips it as unchanged and cannot
    resurrect or clobber the worker's note."""
    for _slot in _MODULE_NOTE_SLOTS:
        _mod = getattr(st, _slot, None)
        if not _mod or not isinstance(_mod, str):
            continue
        if _mod in delivered:
            # The frame seeded this string by reference and already delivered
            # it — clear the module copy so it can't re-announce later.
            setattr(st, _slot, None)
            continue
        messages.append({"role": "user", "content": _mod})
        setattr(st, _slot, None)


def _keep_past_notes(body: list[dict[str, Any]], keep: int) -> int:
    """Smallest keep-count whose verbatim tail still holds `keep` real turns.

    Injected runtime-note messages (recovery nudges, background-job and
    sub-agent announcements) are not conversation turns, and since 0.40 the
    recovery stack appends one per rescue. `keep` counts MESSAGES, so a couple
    of trailing notes would otherwise eat the whole verbatim window of a manual
    /compress (COMPRESS_KEEP=2) and fold away the last real tool result."""
    if keep <= 0 or not body:
        return keep
    real = sum(1 for m in body[-keep:] if not _is_runtime_note(m))
    size = keep
    while real < keep and size < len(body):
        size += 1
        real = sum(1 for m in body[-size:] if not _is_runtime_note(m))
    return size


def _last_is_dangling_tool(messages: list[dict[str, Any]]) -> bool:
    """True if the last real turn is a `tool` result with no assistant turn after it.

    This is the layout right after a tool ran and before the model follows up: an
    empty assistant turn here is most suspicious — the model produced nothing in
    response to a fresh tool result, so the user sees the agent "finish" with no
    answer. Auto-continue is most warranted in exactly this case.

    Trailing runtime notes are skipped (0.40): nudges now arrive as their own
    user message, and a note is not a turn boundary — the tool result underneath
    it is still the thing awaiting a follow-up.
    """
    for msg in reversed(messages):
        if _is_runtime_note(msg):
            continue
        return msg.get("role") == "tool"
    return False
