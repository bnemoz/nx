"""Text-protocol tool-call fallback: parse <tool_call>/[nx_tool_call] tags
out of assistant text for servers without native tool-calling, and sanitize
tool results before they re-enter message history."""
import json
import re
from typing import Any, List, Tuple

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


def _inside_code_fence(text: str, index: int) -> bool:
    """True if `index` falls inside a ``` fenced block (odd fence count before it)."""
    return text.count("```", 0, index) % 2 == 1


_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape_bare_control_chars(raw: str) -> str:
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
            out.append(raw[i:i + 2])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if in_string and c in _CONTROL_ESCAPES:
            out.append(_CONTROL_ESCAPES[c])
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_trailing_commas(raw: str) -> str:
    """Drop a comma immediately before a closing `}`/`]` — small models often
    leave one behind after reordering or dropping the last argument."""
    return TRAILING_COMMA_RE.sub(r"\1", raw)


def _single_quotes_to_double(raw: str) -> str:
    """Swap `'` for `"` when the body has no double quotes at all — i.e. the
    model wrote the whole object with Python-style single-quoted strings.
    Skipped whenever a `"` is already present, since blindly swapping would
    then also corrupt apostrophes inside otherwise-valid string literals."""
    if '"' in raw:
        return raw
    return raw.replace("'", '"')


def repair_json_candidates(raw: str) -> List[str]:
    """Cheap, mechanical JSON repair heuristics, worth trying before treating
    a parse failure as fatal (see `_escape_bare_control_chars` docstring for
    the rationale: small models produce fixable-not-fatal JSON far more often
    than genuinely unparseable output). Returns candidate strings to try with
    `json.loads`, original first, each subsequent one a incrementally more
    aggressive repair.
    """
    candidates = [raw]
    step = _escape_bare_control_chars(raw)
    if step != raw:
        candidates.append(step)
    step2 = _strip_trailing_commas(step)
    if step2 != step:
        candidates.append(step2)
    step3 = _single_quotes_to_double(step2)
    if step3 != step2:
        candidates.append(step3)
    return candidates


def parse_text_calls_with_errors(
        content: str) -> Tuple[List[Tuple[str, Any]], List[str]]:
    """Like parse_text_calls(), but also returns per-tag parse error strings.

    Returns (calls, errors). `errors` has one short description per tag that
    matched but failed to parse (bad JSON or missing "name") — used to build a
    retry-with-correction nudge instead of silently dropping the model's intent.
    """
    text = content or ""
    pos = 0
    calls: List[Tuple[str, Any]] = []
    errors: List[str] = []
    found_tag = False
    n = len(text)
    while pos < n:
        matches = (m for m in (tag.search(text, pos) for tag in TOOL_TAGS) if m)
        match = min(matches, key=lambda m: m.start(), default=None)
        if not match:
            break
        if _inside_code_fence(text, match.start()):
            pos = match.end()
            continue
        found_tag = True
        raw_body = match.group(1)
        obj = None
        parse_exc = None
        for candidate in repair_json_candidates(raw_body):
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


def parse_text_calls(content: str) -> List[Tuple[str, Any]]:
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


def _escape_tool_protocol_delimiters(text: Any) -> Any:
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


def _sanitize_tool_result(text: Any) -> Any:
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
