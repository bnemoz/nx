"""Text-protocol tool-call fallback: parse <tool_call>/[nx_tool_call] tags
out of assistant text for servers without native tool-calling, and sanitize
tool results before they re-enter message history."""
import json
import re

from ._env import _env_int

TOOL_TAG_PATTERN = (
    r"<tool_call\s*>\s*(\{.*?\})\s*"
    r"(?:</tool_call\s*>|(?=<tool_call\s*>)|(?=\[nx_tool_call\])|$)")
NX_TOOL_TAG_PATTERN = (
    r"\[nx_tool_call\]\s*(\{.*?\})\s*"
    r"(?:\[/nx_tool_call\]|(?=\[nx_tool_call\])|(?=<tool_call\s*>)|$)")
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


def _inside_code_fence(text, index):
    """True if `index` falls inside a ``` fenced block (odd fence count before it)."""
    return text.count("```", 0, index) % 2 == 1


def _escape_bare_control_chars(raw):
    """Escape literal newline/tab/carriage-return characters that appear
    inside JSON string literals, without touching anything outside strings.

    Small models routinely emit a multi-line string value as raw newlines
    (e.g. `"content": "line one\nline two"` written with an actual newline
    byte instead of the two-character `\n` escape) — strict JSON forbids
    unescaped control characters in strings, so json.loads rejects the whole
    call with "Invalid control character". This is a mechanical, unambiguous
    fix-up (not a parse of arbitrary malformed JSON), so it's tried as a
    fallback before giving up and asking the model to retry the whole call.
    """
    out = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if in_string and c == "\\" and i + 1 < n:
            out.append(c)
            out.append(raw[i + 1])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if in_string and c in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_text_calls_with_errors(content):
    """Like parse_text_calls(), but also returns per-tag parse error strings.

    Returns (calls, errors). `errors` has one short description per tag that
    matched but failed to parse (bad JSON or missing "name") — used to build a
    retry-with-correction nudge instead of silently dropping the model's intent.
    """
    text = content or ""
    pos = 0
    calls = []
    errors = []
    found_tag = False
    n = len(text)
    while pos < n:
        match = None
        for tag in TOOL_TAGS:
            m = tag.search(text, pos)
            if m and (match is None or m.start() < match.start()):
                match = m
        if not match:
            break
        if _inside_code_fence(text, match.start()):
            pos = match.end()
            continue
        found_tag = True
        raw_body = match.group(1)
        obj = None
        parse_exc = None
        for candidate in (raw_body, _escape_bare_control_chars(raw_body)):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError as e:
                parse_exc = e
        if obj is None:
            errors.append(f"invalid JSON ({parse_exc}): {raw_body[:120]!r}")
        else:
            try:
                calls.append((obj["name"], obj.get("arguments", {})))
            except KeyError:
                errors.append(f"missing \"name\" field: {raw_body[:120]!r}")
        pos = match.end()
    return (calls, errors) if found_tag else ([], [])


def parse_text_calls(content):
    """Pull text-protocol tool-call tags out of an assistant message.

    Small models routinely add a sentence of chatter before/after (or between)
    tags despite SYSTEM asking for the tag alone, so this scans for tags
    anywhere in the text rather than anchoring to the start — any prose
    preamble/postamble/interstitial text is just ignored, not treated as
    disqualifying. Tags inside a ``` fenced code block are still treated as
    documentation/examples, not executable calls, since that's the common
    false-positive case (docs, code samples) this protocol has to avoid.
    Individual calls with malformed JSON are skipped rather than discarding
    the whole message, since small models often emit one good call alongside
    a malformed one.
    """
    return parse_text_calls_with_errors(content)[0]


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
