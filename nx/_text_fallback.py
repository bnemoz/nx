"""Text-protocol tool-call fallback: parse <tool_call>/[nx_tool_call] tags
out of assistant text for servers without native tool-calling, and sanitize
tool results before they re-enter message history."""
import json
import re

from ._env import _env_int

TOOL_TAG_PATTERN = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
NX_TOOL_TAG_PATTERN = r"\[nx_tool_call\]\s*(\{.*?\})\s*\[/nx_tool_call\]"
TOOL_TAG = re.compile(TOOL_TAG_PATTERN, re.DOTALL)
NX_TOOL_TAG = re.compile(NX_TOOL_TAG_PATTERN, re.DOTALL)
# <tool_call> is what SYSTEM now asks for (the convention most open-weight
# models are actually fine-tuned on, e.g. Qwen/Hermes chat templates); keep
# the old bracketed form recognized too for sessions/models already using it.
TOOL_TAGS = (TOOL_TAG, NX_TOOL_TAG)
TOOL_PROTOCOL_TAG_RE = re.compile(r"</?tool_call\b[^>]*>")
TOOL_RESULT_PROTOCOL_NOTE = (
    "[nx note: escaped tool-call protocol delimiters from this tool result "
    "before sending it back to the model]\n"
)


TOOL_RESULT_CHARS = _env_int("NX_TOOL_RESULT_CHARS", 20000)


def parse_text_calls(content):
    """Pull standalone text-protocol tool-call messages.

    Literal protocol tags can appear in files, docs, and code examples. Treat
    them as executable only when the whole assistant message is protocol text.
    """
    text = content or ""
    pos = 0
    calls = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        match = None
        for tag in TOOL_TAGS:
            match = tag.match(text, pos)
            if match:
                break
        if not match:
            return []
        try:
            obj = json.loads(match.group(1))
            calls.append((obj["name"], obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            return []
        pos = match.end()
    return calls


def _escape_tool_protocol_delimiters(text):
    """Neutralize active tool-call delimiters in untrusted tool output."""
    if not isinstance(text, str):
        return text
    safe = TOOL_PROTOCOL_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text)
    safe = safe.replace("[nx_tool_call]", "&#91;nx_tool_call&#93;")
    safe = safe.replace("[/nx_tool_call]", "&#91;/nx_tool_call&#93;")
    if safe != text:
        return TOOL_RESULT_PROTOCOL_NOTE + safe
    return text


def _sanitize_tool_result(text):
    """Neutralize active protocol delimiters, dedup, and cap tool results.

    Large, near-duplicate tool outputs (find/ls/grep path dumps) are the fuel
    for context-copying repetition collapse, so we collapse runs of >=3
    identical consecutive lines, then head/tail-cap to TOOL_RESULT_CHARS with a
    visible marker. Short, non-repetitive results pass through except for
    protocol-delimiter escaping.
    """
    if not isinstance(text, str):
        return text
    text = _escape_tool_protocol_delimiters(text)
    if len(text) <= 1000:
        return text
    lines = text.split("\n")
    out_lines = []
    i, n = 0, len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        run = j - i
        if run >= 3:
            out_lines.append(lines[i])
            out_lines.append(f"... [+{run - 1} identical lines elided]")
        else:
            out_lines.extend(lines[i:j])
        i = j
    result = "\n".join(out_lines)
    budget = TOOL_RESULT_CHARS
    if budget > 0 and len(result) > budget:
        head = budget * 2 // 3
        tail = budget - head
        elided = len(result) - head - tail
        result = (result[:head]
                  + f"\n... [{elided} chars elided to bound context; "
                    f"re-run more narrowly if you need the rest]\n"
                  + result[-tail:])
    return result
