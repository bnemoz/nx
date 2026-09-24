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
    Recall sends count via _SEND_COUNTS keyed on (tool, content).
  - Receipt protection: a tool result containing a line exactly equal to
    sol_pi_evidence_receipt_v1 is never packed (the reducer's receipt IS the
    compact form; packing it would hide verified evidence).
  - Overflow-spill stubs ("saved to:") are never packed — they are already
    handles.

Env: NX_OBS=1 enables (off by default); NX_OBS_THRESHOLD (default 10240
bytes, strictly greater to participate); NX_OBS_FULL_SENDS (default 2).

Swarm P10 (0.98.1): storage identity + lifecycle moved to `nx/_obsstore.py`
— the cross-process content-addressed store. Ids are now obs_ + the first
24 hex of the CONTENT sha (previously tool+call_key+content, which made
two agents reading the same file write two objects and defeated dedup).
The env reads also moved to call time (CONTRIBUTING's rule): tests set the
NX_* env vars instead of patching module attrs. `_SEND_COUNTS` is bounded
(NX_OBS_MAX_TRACKED, default 4096, drop-oldest) so a long swarm session
can't grow it without bound.
"""
from __future__ import annotations

import hashlib
import os

from ._env import _env_int

PLACEHOLDER_EXCERPT_BYTES = 1024
CHARS_PER_TOKEN = 4

RECALL_MAX_BYTES = 16384
RECALL_MAX_LINES = 400

RECEIPT_LINE = "sol_pi_evidence_receipt_v1"

# ids look like obs_<24 hex>; objects live in the shared content-addressed
# store (NX_HOME/obsstore/, nx/_obsstore.py) — P10.
_ID_PREFIX = "obs_"

# In-memory per-session counters (conftest rewinds module attrs). Bounded:
# a swarm session that packs thousands of results used to grow this dict
# without bound (swarm audit item 3). Drop-oldest past the cap.
_MAX_TRACKED = _env_int("NX_OBS_MAX_TRACKED", 4096)
_SEND_COUNTS: dict[str, int] = {}


def obs_enabled() -> bool:
    return bool(_env_int("NX_OBS", 0))


def _threshold_bytes() -> int:
    return _env_int("NX_OBS_THRESHOLD", 10 * 1024)


def _full_sends() -> int:
    return _env_int("NX_OBS_FULL_SENDS", 2)


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def packable(tool_name: str, result: str) -> bool:
    """Eligibility: enabled, string, big enough, all-text, not a spill stub,
    not a reducer receipt."""
    if not obs_enabled() or not isinstance(result, str):
        return False
    if len(result.encode("utf-8", "replace")) <= _threshold_bytes():
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
    full = _full_sends()
    head = _complete_line_excerpt(text, PLACEHOLDER_EXCERPT_BYTES // 2)
    tail = _complete_line_excerpt(text, PLACEHOLDER_EXCERPT_BYTES // 2,
                                  tail=True)
    omitted = max(0, len(raw) - len(head.encode("utf-8", "replace"))
                  - len(tail.encode("utf-8", "replace")))
    return "\n".join([
        f"[large tool result replaced after its first {full} provider requests]",
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


def _bump_send_count(key: str) -> int:
    """Read-and-increment this process's send counter for `key`, evicting the
    oldest tracked key past _MAX_TRACKED (bounded RAM, P10)."""
    sends = _SEND_COUNTS.get(key, 0)
    if len(_SEND_COUNTS) >= _MAX_TRACKED and key not in _SEND_COUNTS:
        oldest = next(iter(_SEND_COUNTS), None)
        if oldest is not None:
            _SEND_COUNTS.pop(oldest, None)
    _SEND_COUNTS[key] = sends + 1
    return sends


def ensure_stored(obs_id: str, text: str) -> str | None:
    """Store `text` in the shared content-addressed store (P10); return None
    on success, an error string on failure (fail-open callers leave the
    original). The handle's hex must match the content sha prefix — a
    mismatch means the caller's id was not derived from this text."""
    from . import _obsstore
    resolved = _obsstore.resolve_handle(obs_id)
    if resolved is None:
        return f"invalid observation id {obs_id!r}"
    sha, _path = resolved
    if _obsstore.content_sha(text)[:len(sha)] != sha:
        return f"observation integrity mismatch for {obs_id}"
    return None if _obsstore.put(text) is not None else "obsstore write failed"


def pack_result(tool_name: str, result: str, call_key: str = "") -> str:
    """The run_tool seam: full text for the first FULL_SENDS sends of this
    (tool, result) identity, then the placeholder. Never raises.

    P10: the identity that counts sends is (tool, content) — the handle is
    content-keyed — so a re-read of identical bytes still counts as the
    same identity, and two processes packing the same content share the
    stored object (each keeps its own send counter; FULL_SENDS is a
    per-process grace, which is the cheap-correct approximation for a
    swarm)."""
    try:
        if not packable(tool_name, result):
            return result
        chash = content_hash(result)
        obs_id = _ID_PREFIX + chash[:24]
        key = f"{tool_name}\0{obs_id}"
        sends = _bump_send_count(key)
        err = ensure_stored(obs_id, result)
        if err is not None:
            return result  # fail open
        if sends < _full_sends():
            return result
        return build_placeholder(result, obs_id, tool_name)
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
    """Resolve an obs_ handle to its object path via the shared store (P10).
    Handles carry the first 24 hex of the content sha; the object file is
    the full 64-hex name, so a truncated handle globs its fanout dir.
    Falls back to the legacy per-session layout so objects written by an
    older nx (or by a session with a different NX_HOME) still recall."""
    from . import _obsstore
    resolved = _obsstore.resolve_handle(obs_id)
    if resolved is None:
        return None
    sha, path = resolved
    if os.path.isfile(path):
        return path
    if len(sha) < 64:
        # truncated handle: look up the full-sha object in the fanout dir
        full = _obsstore.expand_prefix(sha)
        if full is not None:
            return _obsstore.object_path(full)
    # legacy layout (pre-P10): SESSION_HOME/observations/objects/<id>.txt
    legacy = os.path.join(obs_legacy_objects_dir(), obs_id + ".txt")
    if os.path.realpath(os.path.dirname(legacy)) == obs_legacy_objects_dir() \
            and os.path.isfile(legacy):
        return legacy
    return None


def obs_legacy_objects_dir() -> str:
    from . import _state as st
    return os.path.join(st.SESSION_HOME, "observations", "objects")


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
    return f"{header}\n{text}"
