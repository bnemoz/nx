"""Per-message token accounting and pinned-turn protection for /ctx + compress.

Two pieces:

1. `_estimate_tokens` / `_messages_token_report` — a char-based estimator
   (chars/4, the standard heuristic; within ~15% for code-heavy English) with
   a per-role breakdown. Used by /ctx to show where the context actually goes.
   When the active source's /tokenize endpoint works, /ctx uses it for the
   total (exact); per-message numbers stay estimated (tokenizing each message
   separately would round-trip the server N times).

2. Per-turn overhead — `_turn_overhead_est()` / `_overhead_lines()`: the
   tool schemas sent on every model call (`tools=`) live outside the message
   list, so without this /ctx's total quietly misses them and undercounts
   the real per-turn floor. Counted the same way `nx --tokens` counts its
   floor: JSON-serialized schemas, chars/4. Rendered as a footer line, not a
   message row (they aren't messages — pinning a tool schema makes no sense).

3. Pinned turns — `_ctx_pin(index_range)` marks message indexes that
   compress() must keep verbatim (pinned turns move into the kept tail even
   when they're old). State is a list of indexes on st (session lifetime,
   persisted by the session save like todos — see _repl_state).
"""
from __future__ import annotations

from typing import Any

from . import _state as st

CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """char/4 estimate — the standard rough heuristic; good enough to rank
    messages by size, which is what /ctx exists for."""
    if not text:
        return 0
    if isinstance(text, list):  # content-parts form
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return max(1, len(str(text)) // CHARS_PER_TOKEN)


def _message_content_text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _message_tokens(msg: dict[str, Any]) -> int:
    """Estimated tokens for one message, including native tool_calls args."""
    n = _estimate_tokens(_message_content_text(msg))
    tcs = msg.get("tool_calls") or []
    for tc in tcs:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        n += _estimate_tokens(str(fn.get("name") or "")) + _estimate_tokens(str(fn.get("arguments") or ""))
    return n


def _tokenize_total(messages: list[dict[str, Any]]) -> int | None:
    """Exact total via the server's /tokenize endpoint when available, else None.
    Never raises, never blocks long — this is a UX nicety, not a correctness path."""
    try:
        if st.ACTIVE is None or not getattr(st.ACTIVE, "_tokenize_ok", None):
            return None
        resp = st.ACTIVE.client._client.post(
            "/tokenize", json={"messages": messages})
        resp.raise_for_status()
        n = resp.json().get("tokens")
        return n if isinstance(n, int) and n > 0 else None
    except Exception:
        return None


def _cache_hits_by_turn(n_messages: int) -> list[float | None]:
    """Per-model-call prompt-cache hit fractions for /ctx's cache column.

    Source of truth is the metrics ring buffer (st._METRICS.turns — one entry
    per settled model call, newest last, capped at 64). Alignment with /ctx's
    message rows is approximate by nature: several calls can serve one logical
    turn (tool loops, reasoning retries), and a resumed session replays with
    no per-call history at all. So the raw series is left-padded with None
    (turns that predate the buffer → no data) then right-truncated to
    n_messages. Each /ctx row renders its fraction as `N%` or `—` when the
    server reported no cache numbers for that call (or there was none).
    """
    m = st._METRICS
    if m is None:
        return []
    hits: list[float | None] = []
    for t in getattr(m, "turns", []) or []:
        p = int(t.get("prompt_tokens") or 0)
        # cache_read_tokens < 0 is the _normalize_usage sentinel for "the
        # server sent no cache breakdown at all" — distinct from a real 0
        # (breakdown present, cache missed), which /ctx should render as 0%.
        c = int(t.get("cache_read_tokens") or 0)
        hits.append((min(1.0, c / p) if p else None) if c >= 0 else None)
    if not hits:
        return []
    if len(hits) < n_messages:
        hits = [None] * (n_messages - len(hits)) + hits
    return hits[-n_messages:] if n_messages else []


# --- per-turn overhead (tool schemas live outside `messages`) ---------------

def _turn_overhead_est() -> int:
    """Estimated tokens of the per-turn `tools=` payload — the function schemas
    the server renders into every prompt but which never appear in the message
    list, so a messages-only sum undercounts the real context floor. Estimated
    chars/4 over the JSON serialization, exactly how `nx --tokens` counts its
    TOOLS line. Never raises: /ctx is a UX nicety, not a correctness path."""
    try:
        from ._dispatch import _active_tools
        import json as _json
        return _estimate_tokens(_json.dumps(_active_tools()))
    except Exception:
        return 0


def _overhead_lines(total_est: int, overhead: int) -> list[str]:
    """Footer lines for /ctx naming the per-turn overhead the message rows
    can't show. Suppressed when there's nothing to name."""
    if overhead <= 0:
        return []
    pct = 100 * overhead // total_est if total_est > 0 else 0
    return [f"  + tools schema (sent every turn, not shown as a row): ~{overhead} tok ({pct}% of total est)"]


def _messages_token_report(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-message token estimates + role subtotals, for /ctx.

    Returns {"rows": [(idx, role, tokens, preview, pinned)], "total_est": int,
             "total_exact": int|None, "by_role": {role: tokens},
             "overhead_est": int, "cache": [float|None per row]}.
    Indexes are into `messages` itself (what /ctx pin N takes). `total_est`
    includes `overhead_est` (the per-turn tools= schemas — see
    _turn_overhead_est), which by_role does NOT. `cache` is the per-call
    prompt-cache hit fraction aligned to rows (see _cache_hits_by_turn) —
    empty when nothing has run this session.
    """
    pinned = set(getattr(st, "_CTX_PINNED", []) or [])
    rows = []
    by_role: dict[str, int] = {}
    total = 0
    for i, m in enumerate(messages):
        t = _message_tokens(m)
        role = m.get("role", "?")
        preview = _message_content_text(m).strip().replace("\n", " ")[:64]
        if not preview and m.get("tool_calls"):
            preview = "→ " + ", ".join(
                ((c.get("function") or {}).get("name") or "?")
                for c in m["tool_calls"])
        rows.append((i, role, t, preview, i in pinned))
        by_role[role] = by_role.get(role, 0) + t
        total += t
    overhead = _turn_overhead_est()
    return {"rows": rows, "total_est": total + overhead,
            "total_exact": _tokenize_total(messages), "by_role": by_role,
            "overhead_est": overhead,
            "cache": _cache_hits_by_turn(len(messages))}


# --- pinned turns -----------------------------------------------------------

def _parse_pin_spec(spec: str, n_messages: int) -> list[int]:
    """'12', '12-15', '4,8,12-15' (1-based, inclusive) → message indexes.
    Raises ValueError on anything unparseable or out of range."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
        else:
            lo = hi = int(part)
        if lo < 1 or hi > n_messages:
            raise ValueError(f"range {lo}-{hi} outside 1..{n_messages}")
        out.update(range(lo - 1, hi))
    return sorted(out)


def _ctx_pin(spec: str, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Pin message indexes (1-based spec). Returns (newly_pinned, all_pinned)."""
    idxs = _parse_pin_spec(spec, len(messages))
    pinned = set(getattr(st, "_CTX_PINNED", []) or [])
    added = [i for i in idxs if i not in pinned]
    pinned.update(idxs)
    st._CTX_PINNED = sorted(pinned)
    return added, st._CTX_PINNED


def _ctx_unpin(spec: str | None, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Unpin all (spec None) or the given 1-based spec. Returns (removed, remaining)."""
    pinned = set(getattr(st, "_CTX_PINNED", []) or [])
    if spec is None or spec.strip().lower() in ("all", "*"):
        removed = sorted(pinned)
        st._CTX_PINNED = []
        return removed, []
    idxs = set(_parse_pin_spec(spec, len(messages)))
    removed = sorted(pinned & idxs)
    st._CTX_PINNED = sorted(pinned - idxs)
    return removed, st._CTX_PINNED


def _pinned_indexes() -> set[int]:
    return set(getattr(st, "_CTX_PINNED", []) or [])
