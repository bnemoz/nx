"""Observation Pack (SoL-Pi port, 0.67.0): the full text of large tool
results is sent for the first N provider requests, then replaced — in the
PROJECTED context only, never in the stored transcript — by a short handle
with exact paged recall via the `obs_recall` tool.

Why a projection and not a rewrite: nx's transcript bytes are the prompt-cache
contract (`_prefix.check_prefix`) and the session-save format. Rewriting a
mid-history tool result (as the spill-evictors do under pressure) invalidates
the whole prefix. Packing at request time instead keeps stored history intact
across compress/resume and still bills less on every later request.

Port deltas from SoL-Pi:
  - No per-request hook exists in nx, so packing runs inside
    `_sanitize_tool_result`'s caller path: results ARE stored full in history
    and the pack replaces the string only at the moment run_tool returns it
    to the turn loop for messages-history append (see _dispatch.run_tool).
    Recall sends count via st._OBS_SEND_COUNTS keyed on (tool, callKey).
  - Receipt protection: a tool result containing a line exactly equal to
    sol_pi_evidence_receipt_v1 is never packed (the reducer's receipt IS the
    compact form; packing it would hide verified evidence).
  - Overflow-spill stubs ("saved to:") are never packed — they are already
    handles.

Env: NX_OBS=1 enables (off by default); NX_OBS_THRESHOLD (default 10240
bytes, strictly greater to participate); NX_OBS_FULL_SENDS (default 2).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from ._env import _env_int

OBS_ENABLED = _env_int("NX_OBS", 0)
THRESHOLD_BYTES = _env_int("NX_OBS_THRESHOLD", 10 * 1024)
FULL_SENDS = _env_int("NX_OBS_FULL_SENDS", 2)
PLACEHOLDER_EXCERPT_BYTES = 1024
CHARS_PER_TOKEN = 4

RECALL_MAX_BYTES = 16384
RECALL_MAX_LINES = 400

RECEIPT_LINE = "sol_pi_evidence_receipt_v1"

# ids look like obs_<24 hex>; objects live under SESSION_HOME/observations/
_ID_PREFIX = "obs_"
_OBS_DIR_NAME = "observations"
_LEDGER_NAME = "ledger.jsonl"

# In-memory per-session counters (conftest rewinds module attrs).
_SEND_COUNTS: dict[str, int] = {}


def obs_enabled() -> bool:
    return OBS_ENABLED == 1


def obs_dir() -> str:
    from . import _state as st
    return os.path.join(st.SESSION_HOME, _OBS_DIR_NAME)


def objects_dir() -> str:
    return os.path.join(obs_dir(), "objects")


def _ledger_path() -> str:
    return os.path.join(obs_dir(), _LEDGER_NAME)


def _env_int_guard(v: int, lo: int) -> int:
    return v if v >= lo else lo


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def make_observation_id(tool_name: str, call_key: str, content_hash: str) -> str:
    h = hashlib.sha256(
        (tool_name + "\0" + call_key + "\0" + content_hash).encode("utf-8")
    ).hexdigest()
    return _ID_PREFIX + h[:24]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def packable(tool_name: str, result: str) -> bool:
    """Eligibility: enabled, string, big enough, all-text, not a spill stub,
    not a reducer receipt."""
    if not obs_enabled() or not isinstance(result, str):
        return False
    if len(result.encode("utf-8", "replace")) <= THRESHOLD_BYTES:
        return False
    if "saved to:" in result and OVERFLOW_STUB_HINT in result:
        return False
    if any(line.strip() == RECEIPT_LINE for line in result.splitlines()):
        return False
    return True


OVERFLOW_STUB_HINT = "[large output truncated"


def _complete_line_excerpt(text: str, budget: int, tail: bool = False) -> str:
    """Greedily take whole lines from the head (or tail) that fit `budget`
    bytes."""
    lines = text.splitlines(keepends=True)
    if tail:
        lines = lines[::-1]
    out: list[str] = []
    used = 0
    for line in lines:
        b = len(line.encode("utf-8", "replace"))
        if used + b > budget:
            break
        out.append(line)
        used += b
    return "".join(out if not tail else out[::-1])


def build_placeholder(text: str, obs_id: str, tool_name: str) -> str:
    raw = text.encode("utf-8", "replace")
    head = _complete_line_excerpt(text, PLACEHOLDER_EXCERPT_BYTES // 2)
    tail = _complete_line_excerpt(text, PLACEHOLDER_EXCERPT_BYTES // 2,
                                  tail=True)
    omitted = max(0, len(raw) - len(head.encode("utf-8", "replace"))
                  - len(tail.encode("utf-8", "replace")))
    return "\n".join([
        f"[large tool result replaced after its first {FULL_SENDS} provider requests]",
        f"id: {obs_id}",
        f"tool: {tool_name}",
        f"original_bytes: {len(raw)}",
        f"original_lines: {text.count(chr(10)) + (0 if text.endswith(chr(10)) else 1)}",
        f"estimated_tokens: {estimate_tokens(text)}",
        f'retrieve: call obs_recall with {{"id":"{obs_id}","offset":0}}; continue with returned next_offset',
        "[first complete lines, up to 512 bytes]",
        head,
        "[middle omitted; last complete lines, up to 512 bytes]",
        tail,
        f"[{omitted} original bytes omitted]",
    ])


def _append_ledger(entry: dict[str, Any]) -> None:
    try:
        os.makedirs(obs_dir(), mode=0o700, exist_ok=True)
        rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **entry}
        with open(_ledger_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # ledger is best-effort; packing itself must never fail


def ensure_stored(obs_id: str, text: str) -> str | None:
    """Write objects/<id>.txt (0600, exclusive); return None on success, an
    error string on integrity failure (fail-open callers leave the original)."""
    try:
        odir = objects_dir()
        os.makedirs(odir, mode=0o700, exist_ok=True)
        path = os.path.join(odir, obs_id + ".txt")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            return None
        except FileExistsError:
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
            if existing != text:
                return f"observation integrity mismatch for {obs_id}"
            return None
    except OSError as e:
        return f"{type(e).__name__}: {e}"


def pack_result(tool_name: str, result: str, call_key: str = "") -> str:
    """The run_tool seam: full text for the first FULL_SENDS sends of this
    (tool, result) identity, then the placeholder. Never raises."""
    try:
        if not packable(tool_name, result):
            return result
        chash = content_hash(result)
        obs_id = make_observation_id(tool_name, call_key or chash, chash)
        key = f"{tool_name}\0{obs_id}"
        sends = _SEND_COUNTS.get(key, 0)
        err = ensure_stored(obs_id, result)
        if err is not None:
            return result  # fail open
        if sends < FULL_SENDS:
            _SEND_COUNTS[key] = sends + 1
            _append_ledger({"kind": "full", "id": obs_id, "tool": tool_name,
                            "sends": sends + 1,
                            "bytes": len(result.encode("utf-8", "replace"))})
            return result
        placeholder = build_placeholder(result, obs_id, tool_name)
        _append_ledger({"kind": "placeholder", "id": obs_id,
                        "tool": tool_name, "sends": sends + 1,
                        "placeholder_bytes": len(placeholder),
                        "removed_tokens": max(
                            0, estimate_tokens(result)
                            - estimate_tokens(placeholder))})
        _SEND_COUNTS[key] = sends + 1
        return placeholder
    except Exception:  # noqa: BLE001 — packing must never break a tool result
        return result


def maybe_pack_fused_output(output: str, tool_name: str, call_key: str) -> str:
    """Hook used by _fusion.run_fused: a fused validation command's whole
    output is replaced by the pack handle immediately (the edit result is
    what the model reads first; the log stays recallable verbatim)."""
    if not obs_enabled():
        return output
    return pack_result(tool_name, output, call_key)


# --- obs_recall tool ---------------------------------------------------------

OBS_RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "obs_recall",
        "description": (
            "Page back the archived original of a large tool result that was "
            "replaced by an observation handle (a result whose header said "
            "`id: obs_…`). Returns whole lines plus a next_offset; pass "
            "next_offset back in until eof=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string",
                       "description": "The obs_… id from the handle stub"},
                "offset": {"type": "integer",
                           "description": "Byte offset into the original (default 0; use next_offset from the previous chunk)"},
            },
            "required": ["id"],
        },
    },
}


def _safe_obs_path(obs_id: str) -> str | None:
    if not obs_id.startswith(_ID_PREFIX) or not obs_id[len(_ID_PREFIX):].isalnum():
        return None
    path = os.path.join(objects_dir(), obs_id + ".txt")
    if os.path.realpath(os.path.dirname(path)) != objects_dir():
        return None
    return path


def obs_recall(id: str, offset: int = 0) -> str:  # noqa: A002 — schema name
    offset = max(0, int(offset))
    path = _safe_obs_path(id)
    if path is None or not os.path.isfile(path):
        return f"ERROR: unknown observation id {id!r}"
    size = os.path.getsize(path)
    if offset > size:
        return f"ERROR: offset {offset} beyond size {size}"
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read(RECALL_MAX_BYTES + 4)
    chunk = chunk[:RECALL_MAX_BYTES]
    # cut at line boundary if over the line cap
    lines = chunk.split(b"\n")
    if len(lines) > RECALL_MAX_LINES:
        chunk = b"\n".join(lines[:RECALL_MAX_LINES]) + b"\n"
    # trim trailing UTF-8 continuation bytes
    while chunk and (chunk[-1] & 0xC0) == 0x80:
        chunk = chunk[:-1]
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        text = chunk.decode("utf-8", "replace")
    next_offset = offset + len(chunk)
    eof = next_offset >= size
    header = (f"[obs_recall id={id} offset={offset} "
              f"next_offset={next_offset} eof={str(eof).lower()}]\n"
              f"[chunk_bytes={len(chunk)} chunk_lines={text.count(chr(10))}; "
              "use next_offset to continue]")
    _append_ledger({"kind": "recall", "id": id, "offset": offset,
                    "bytes": len(chunk), "next_offset": next_offset,
                    "eof": eof})
    return f"{header}\n{text}"
