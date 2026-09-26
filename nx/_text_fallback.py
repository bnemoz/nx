"""Text-protocol tool-call fallback: parse <tool_call>/[nx_tool_call] tags
out of assistant text for servers without native tool-calling, and sanitize
tool results before they re-enter message history."""
import json
import re
from typing import Any, List, Tuple

from ._env import _env_int
from ._overflow import (
    OVERFLOW_READ_MARKER as _OVERFLOW_READ_MARKER,
    build_overflow_result,
    cap_overflow_read as _cap_overflow_read,
    spill_to_disk,
)

TOOL_OPEN_TAG = re.compile(r"<tool_call\s*>\s*", re.DOTALL)
NX_OPEN_TAG = re.compile(r"\[nx_tool_call\]\s*", re.DOTALL)
TOOL_TAGS = (TOOL_OPEN_TAG, NX_OPEN_TAG)


def _extract_json_object(text: str, start: int):
    """Find the `{...}` object beginning at `start`, honoring JSON string/escape
    rules so a literal `}` (or a tag-like substring) inside a string value
    doesn't end the match early. Returns (raw_body, end_index) or None if no
    balanced object is found (e.g. the model got cut off mid-call).

    A naive non-greedy `\\{.*?\\}` regex stops at the first `}` followed by a
    closing/opening tag, with no quote-awareness — if a string argument
    contains a literal ``</tool_call>``-shaped example (plausible: this
    project's own README documents that exact tag), the match truncates
    mid-JSON-string and the real call fails to parse.
    """
    n = len(text)
    while start < n and text[start] != "{":
        return None
    depth = 0
    in_string = False
    i = start
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], i + 1
        i += 1
    return None
# <tool_call> is what SYSTEM now asks for (the convention most open-weight
# models are actually fine-tuned on, e.g. Qwen/Hermes chat templates); keep
# the old bracketed form recognized too for sessions/models already using it.
TOOL_PROTOCOL_TAG_RE = re.compile(r"</?tool_call\b[^>]*>")
TOOL_RESULT_PROTOCOL_NOTE = (
    "[nx note: escaped tool-call protocol delimiters from this tool result "
    "before sending it back to the model]\n"
)


TOOL_RESULT_CHARS = _env_int("NX_TOOL_RESULT_CHARS", 102400)

# Oversized results are additionally spilled to disk (see _overflow.py): the
# model gets head + tail + the path of the full text instead of a blind cut.
# 0 disables spilling (plain elision marker, as before).
TOOL_RESULT_SPILL = _env_int("NX_TOOL_RESULT_SPILL", 1)

# PF-14: the floor at which a tool result is spilled to disk the moment it is
# produced, BEFORE it is ever sent to the model. Held equal to the hard cap
# above so any whole-file read within the cap passes whole with no spill at
# all (0.66.2's 4000 default stubbed even a ~25KB markdown read_file). It is
# no longer tied to the evictors' thresholds
# for a context overrun already carries its on-disk stub by the time it enters
# history. That matters for the prompt cache: a retroactive spill rewrites a
# mid-transcript message's bytes, invalidating the server prefix cache for the
# whole remaining prompt on exactly the turns it is largest. 0 restores the
# old arrival-time cap only (>TOOL_RESULT_CHARS), leaving the evictors as the
# sole (rewriting) spiller.
APPEND_SPILL_MIN_CHARS = _env_int("NX_TOOL_RESULT_SPILL_MIN_CHARS", 102400)

# First-line secrets hygiene (0.40, roadmap 1.6): mask known token SHAPES in
# tool results before the result is shown, persisted (sessions, spills,
# sub-agent reports, bg logs) or cached. Best-effort by construction — it
# catches credentials that announce themselves, not arbitrary strings — but
# every pattern below is a shape no legitimate tool output needs verbatim.
# 0 restores pass-through. The approval-side counterpart (input commands) is
# NX_BASH_PROMPT_SECRETS in _bash_jobs.py.
TOOL_RESULT_REDACT = _env_int("NX_TOOL_RESULT_REDACT", 1)

# (compiled pattern, placeholder label). Patterns with a second group keep
# that prefix (a header name, `Bearer `) and redact only the value after it.
_REDACT_PATTERNS = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private-key"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "api-key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "github-token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws-access-key"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "slack-token"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}\b"), "gitlab-token"),
    (re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
     "bearer-token"),
    (re.compile(r"(?i)\b((?:x-api-key|api-key|api_key)\s*[:=]\s*)[A-Za-z0-9._~+/=-]{16,}"),
     "api-key"),
)


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


TOOL_CLOSE_TAG = re.compile(r"</tool_call\s*>", re.DOTALL)
NX_CLOSE_TAG = re.compile(r"\[/nx_tool_call\]", re.DOTALL)
TOOL_CLOSE_TAGS = (TOOL_CLOSE_TAG, NX_CLOSE_TAG)

XML_NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)
XML_ARGUMENTS_RE = re.compile(r"<arguments\s*>(.*?)</arguments\s*>", re.DOTALL)
XML_ARG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
XML_BARE_NAME_RE = re.compile(r"^\s*(\w+)")
# GLM's own chat template writes tool calls as `name<arg_key>k</arg_key>
# <arg_value>v</arg_value>...` inside the <tool_call> tag -- no <name>/
# <arguments> wrapper at all, just the bare name then repeated key/value
# pairs. Recognize this alongside our own <name>/<arguments> convention
# rather than demanding the model conform to one specific dialect.
XML_ARG_KV_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

_XML_ENTITIES = {
    "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'", "&amp;": "&",
}


def _xml_unescape(raw: str) -> str:
    for entity, char in _XML_ENTITIES.items():
        raw = raw.replace(entity, char)
    return raw


def normalize_tool_args(obj: Any) -> Any:
    """Fold degenerate `arguments` payloads into a plain mapping, or return the
    input unchanged when it isn't recoverable.

    Servers and small models occasionally wrap the whole arguments object one
    more time than the schema wants: a JSON-encoded STRING (double-encoded),
    or a LIST of name/value pair objects. Left as-is, run_tool dies with
    "TypeError: argument after ** must be a mapping", which reads to the model
    as a harness bug rather than its own fixable formatting. One extra
    json.loads for strings, one fold for pair-lists; anything else (scalars,
    nested junk) is NOT recoverable and is returned untouched so the caller
    treats it as a parse failure. Never raises."""
    if isinstance(obj, str):
        try:
            decoded = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            return obj  # not JSON: a parse failure for the caller to report
        if isinstance(decoded, dict):
            return decoded
        return obj  # decoded to a scalar/list: still not a mapping
    if isinstance(obj, list):
        out: dict[str, Any] = {}
        for item in obj:
            if not isinstance(item, dict):
                return obj  # heterogeneous list: not recoverable, don't guess
            name = item.get("name") or item.get("key") or item.get("arg")
            if not isinstance(name, str) or ("value" not in item and "val" not in item):
                return obj
            out[name] = item.get("value", item.get("val"))
        return out  # [] folds to {} (an arguments-less call)
    return obj


def _parse_xml_arg_value(raw: str) -> Any:
    """A bare XML tag body has no type info -- try JSON first (covers numbers,
    booleans, and models that still emit a list/object for a single arg),
    falling back to the unescaped text itself."""
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return _xml_unescape(stripped)


def _parse_xml_tool_call(body: str):
    """Parse a `<tool_call>`/`[nx_tool_call]` body in either of two dialects:

    - ours: `<name>...</name><arguments><k>v</k>...</arguments>`
    - GLM's own chat-template convention: a bare name followed by repeated
      `<arg_key>k</arg_key><arg_value>v</arg_value>` pairs, no wrapper tags.

    Returns (name, arguments dict), or None if no name can be found at all
    (caller reports a "missing name" error the same way the JSON path does).
    """
    name_m = XML_NAME_RE.search(body)
    arguments: dict = {}
    if name_m:
        name = name_m.group(1).strip()
        args_m = XML_ARGUMENTS_RE.search(body)
        if args_m:
            for arg_m in XML_ARG_RE.finditer(args_m.group(1)):
                arguments[arg_m.group(1)] = _parse_xml_arg_value(arg_m.group(2))
        # 0.29.5: a value that is itself a JSON-encoded mapping (double-encoded
        # args) or a list of {name,value} pairs folds to a plain mapping.
        folded = normalize_tool_args(arguments)
        if isinstance(folded, dict):
            arguments = folded
        return name, arguments
    bare_m = XML_BARE_NAME_RE.match(body)
    if not bare_m:
        return None
    name = bare_m.group(1).strip()
    for kv_m in XML_ARG_KV_RE.finditer(body):
        arguments[kv_m.group(1).strip()] = _parse_xml_arg_value(kv_m.group(2))
    folded = normalize_tool_args(arguments)
    if isinstance(folded, dict):
        arguments = folded
    return name, arguments


def _find_close_tag(text: str, start: int):
    matches = (m for m in (tag.search(text, start) for tag in TOOL_CLOSE_TAGS) if m)
    return min(matches, key=lambda m: m.start(), default=None)


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
        rest = text[match.end():].lstrip()
        if rest.startswith("{"):
            # Legacy JSON-in-tag body (still recognized for older sessions/
            # sources that already learned this convention).
            extracted = _extract_json_object(text, match.end())
            if extracted is None:
                pos = match.end()
                continue
            raw_body, body_end = extracted
            found_tag = True
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
                    calls.append((obj["name"], normalize_tool_args(obj.get("arguments", {}))))
                except KeyError:
                    errors.append(f"missing \"name\" field: {raw_body[:120]!r}")
            pos = body_end
            continue
        close_match = _find_close_tag(text, match.end())
        if close_match is None:
            pos = match.end()
            continue
        raw_body = text[match.end():close_match.start()]
        body_end = close_match.end()
        found_tag = True
        parsed = _parse_xml_tool_call(raw_body)
        if parsed is None:
            errors.append(f"missing <name> tag: {raw_body[:120]!r}")
        else:
            calls.append(parsed)
        pos = body_end
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


def _redact_tool_secrets(text: str) -> str:
    """Mask known secret shapes (see _REDACT_PATTERNS). Runs BEFORE anything
    else in _sanitize_tool_result — including the disk spill — so a spill holds
    no more than the model was sent (that was the one sink that persisted the
    COMPLETE untruncated text). The placeholder keeps the token's kind so the
    model can still reason about what it was ("the key was rejected"), just not
    the value."""
    if not TOOL_RESULT_REDACT:
        return text
    for rx, label in _REDACT_PATTERNS:
        text = rx.sub(
            lambda m, _label=label: (
                (m.group(1) + f"[redacted:{_label}]")
                if m.lastindex else f"[redacted:{_label}]"),
            text)
    return text


def _sanitize_tool_result(text: Any) -> Any:
    """Redact secret shapes, neutralize protocol delimiters, dedup, and cap
    tool results.

    Large, near-duplicate tool outputs (find/ls/grep path dumps) are the fuel
    for context-copying repetition collapse, so we collapse runs of >=3
    identical consecutive lines, then head/tail-cap to TOOL_RESULT_CHARS with a
    visible marker. Short, non-repetitive results pass through except for
    secret redaction and protocol-delimiter escaping.
    """
    if not isinstance(text, str):
        return text
    text = _redact_tool_secrets(text)
    text = _escape_tool_protocol_delimiters(text)
    if len(text) <= 1000:
        return text.replace(_OVERFLOW_READ_MARKER, "")
    # Never re-spill a read of an overflow spill file: the spill stub invites
    # the model to read_file that path, so spilling the read would chase
    # pointers forever (each stub points at a fresh file, and an offset/limit
    # read of the middle gets spilled instead of served — observed as a ~100K
    # token burn across dozens of spill-file reads). Serve it verbatim; the
    # only bound is the hard cap (no pointer), since a full spill-file read is
    # at most one previous TOOL_RESULT_CHARS anyway.
    if text.lstrip().startswith(_OVERFLOW_READ_MARKER):
        return _cap_overflow_read(text)
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
    over_budget = budget > 0 and len(result) > budget
    over_floor = APPEND_SPILL_MIN_CHARS > 0 and len(result) > APPEND_SPILL_MIN_CHARS
    # Spill BEFORE the result is ever sent (PERF.md PF-14): once a message is on
    # the wire its bytes must never change, so the decision to keep only head +
    # tail + a pointer is made here rather than retroactively by a context-overrun
    # evictor. It triggers on the arrival cap OR the lower append floor, which
    # pre-empts those evictors. The stub is bounded (MAX_SPILL_ECHO lines).
    if TOOL_RESULT_SPILL and (over_budget or over_floor):
        stub = _spill_with_echo(result)
        # The stub is bounded (MAX_SPILL_ECHO lines), so at the default cap it
        # is far smaller than the result it replaces. (A user who lowers
        # NX_TOOL_RESULT_CHARS below the echo budget still gets the path — the
        # pointer matters more than the cap, as the arrival-time spill always
        # did.)
        if stub is not None:
            return stub
    if over_budget:
        head = budget * 2 // 3
        tail = budget - head
        elided = len(result) - head - tail
        marker = (f"\n... [{elided} chars elided to bound context; "
                  f"re-run more narrowly if you need the rest]\n")
        result = result[:head] + marker + result[-tail:]
    return result


def _spill_with_echo(full: str) -> str | None:
    """Write `full` to the overflow dir and return the bounded stub that
    replaces it in the message (head lines + tail lines + the path), or None
    when spilling isn't viable (I/O error). Never raises — spilling must never
    lose a result."""
    try:
        path = spill_to_disk(full)
        return build_overflow_result(full, path) if path else None
    except Exception:  # noqa: BLE001 — spilling must never lose a result
        return None
