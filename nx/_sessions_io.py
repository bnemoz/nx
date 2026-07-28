"""Pure session-data helpers with no filesystem/env dependency: id generation,
title/text sanitizing, message-list pruning, and listing-summary shaping.

The filesystem side (session directory resolution, read/write/index) stays in
__init__.py because it's keyed off SESSION_HOME, a module global that /reload
in tests reassigns from NX_HOME/NX_SESSIONS_DIR — see the reload
semantics note in __init__.py's sessions section.
"""
from __future__ import annotations

import secrets
import time
from typing import Any

Message = dict[str, Any]


def _flatten_message_text(
    m: Message, per_msg: int | None = None, tool_call_arg_chars: int | None = None
) -> str:
    """Collapse one message to a single display string: tool calls render as
    `→ name(...)`, list-content (multi-part) is joined, newlines become
    spaces. Shared by transcript printing, memory rendering, and session
    description summarization, which previously each reimplemented this.

    `per_msg` truncates the result (with an ellipsis) if given. When
    `tool_call_arg_chars` is given, tool-call arguments are inlined up to
    that length instead of collapsed to `(...)`.
    """
    content = m.get("content")
    if content is None and m.get("tool_calls"):
        if tool_call_arg_chars is not None:
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments'][:tool_call_arg_chars]})"
                for tc in m["tool_calls"])
        else:
            calls = ", ".join(
                f"{tc['function']['name']}(...)" for tc in m["tool_calls"])
        content = f"→ {calls}"
    elif isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    content = (content or "").replace("\n", " ").strip()
    if per_msg is not None and len(content) > per_msg:
        content = content[:per_msg - 1] + "…"
    return content


def _new_session_id() -> str:
    """Short, unguessable, sortable-ish: YYYYMMDD-HHMMSS-<6 hex>."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _safe_title(text: Any, maxlen: int = 60) -> str | None:
    """Turn the first user message into a filesystem-safe-ish title.

    Collapses whitespace, strips control chars, clamps length. We don't
    scrub for path-separators beyond replacing them with spaces — the id
    (not the title) is the filename, so a weird title can't break lookup.
    """
    if not text:
        return None
    text = " ".join(str(text).split())
    text = "".join(c for c in text if c.isprintable())
    if len(text) > maxlen:
        text = text[:maxlen - 1] + "…"
    return text or None


def _is_empty_assistant_message(msg: Message) -> bool:
    if msg.get("role") != "assistant" or msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.strip():
                return False
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return False
        return True
    return False


def _drop_leading_dangling_tool_turns(msgs: list[Message]) -> tuple[list[Message], int]:
    """Trim leading turns that would break a chat template's tool_call/result
    boundary if `msgs` were sent as-is (e.g. `open_stream` on the full history,
    or `compress()` sizing its verbatim tail).

    A `tool` turn with no preceding assistant(tool_calls) parent — or an
    assistant(tool_calls) turn whose result got cut off elsewhere — makes
    llama.cpp's Jinja template raise "Message has tool role, but there was no
    previous assistant message with a tool call!". Walk from the front and
    drop any leading tool/half-tool-call turns until landing on something
    safe (user, plain assistant, or system). Returns (trimmed_list, n_dropped).
    """
    n_dropped = 0
    while msgs and msgs[0].get("role") in ("tool", "assistant"):
        first = msgs[0]
        if first.get("role") == "tool":
            msgs = msgs[1:]
            n_dropped += 1
            continue
        # assistant: only safe if it has NO tool_calls, OR every tool_call has
        # its matching tool result later in msgs
        if first.get("tool_calls"):
            ids = {tc["id"] for tc in first["tool_calls"]}
            seen = set()
            for m in msgs[1:]:
                tcid = m.get("tool_call_id")
                if m.get("role") == "tool" and tcid:
                    seen.add(tcid)
            if ids - seen:
                msgs = msgs[1:]
                n_dropped += 1
                continue
        break
    return msgs, n_dropped


def _prune_empty_assistant_messages(messages: list[Message]) -> int:
    """Drop assistant turns that have neither visible content nor tool calls."""
    if not isinstance(messages, list):
        return 0
    kept = [m for m in messages if not _is_empty_assistant_message(m)]
    removed = len(messages) - len(kept)
    if removed:
        messages[:] = kept
    return removed


def _short_id(session_id: str | None) -> str:
    """The scannable tail of a session id (the 6-hex suffix), for listings.

    The full id is `YYYYMMDD-HHMMSS-XXXXXX`; in a list of recent sessions the
    date+time prefix is shared/redundant, so we show just the 6 hex chars as a
    quick tag the user can grep or pass to --resume."""
    if session_id and "-" in session_id:
        return session_id.rsplit("-", 1)[-1]
    return session_id or ""


def _summarize_session_data(session_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """The small listing summary (id, title, preview, …) pulled out of a
    full session dict. Shared by the sidecar-index writer and the
    full-parse fallback in _session_summary_from_file, so both paths
    produce identical fields."""
    msgs = data.get("messages", [])
    preview = ""
    for msg in msgs:
        if msg.get("role") == "user" and msg.get("content"):
            preview = _safe_title(msg["content"]) or ""
            break
    return {
        "id": session_id,
        "short": _short_id(session_id),
        "title": data.get("title") or preview or "(empty)",
        "description": data.get("description"),
        "preview": preview,
        "updated_at": data.get("updated_at", 0),
        "n": len([m for m in msgs if m.get("role") != "system"]),
        "model": data.get("model"),
        "source": data.get("source"),
        "cwd": data.get("cwd"),
    }


def _session_matches_query(summary: dict[str, Any], query: str | None) -> bool:
    if not query:
        return True
    q = query.lower()
    return (
        q in (summary.get("title") or "").lower()
        or q in (summary.get("description") or "").lower()
        or q in (summary.get("preview") or "").lower()
        or q in (summary.get("id") or "").lower()
        or q in (summary.get("short") or "").lower()
    )
