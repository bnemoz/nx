"""Online Context Compact (SoL-Pi port, 0.67.0): completed plan steps become
candidate points for nx's native compaction (compress), subject to economic
and window-pressure checks; after a successful compaction the task continues
in a new turn with a plan reminder.

Mapping from SoL-Pi's pi-harness events to nx:
  update_plan boundary  -> a todo_write call that newly completes >= 1 item
                           (nx's todo list IS the plan; _todo.todo_write is
                           the boundary event)
  turn_end decision     -> after a settled model turn carrying that boundary
  native compaction     -> nx's compress(messages, auto=True) — the same
                           append-only summary machinery, which already
                           preserves the pinned prefix and the latest
                           todo_write turn
  post-compaction turn  -> a [Runtime note] staged for the next turn (nx's
                           continuation mechanism), carrying the plan reminder

Economics (port of economics.ts, same formulas and defaults):
  keepRecentTokens = NX_OCC_KEEP_RECENT_TOKENS (20000)
  cacheWriteReadRatio = NX_OCC_CACHE_RATIO (12.5; 0 disables pricing)
  windowReserveTokens = 16384
  firstCompactionRequestScale = 2, subsequentCompactionMargin = 1.5
  memoTokens = 1000 (estimate of the summary turn's own cost)
  text tokens = ceil(bytes / 4)

State (st.* module globals, conftest-rewound):
  _OCC_PLAN              the todo list as last recorded [{text,status}]
  _OCC_REQUEST_COUNT     settled provider requests this session
  _OCC_LAST_CONTEXT_TOKENS
  _OCC_POSITIVE_DELTA_TOTAL / _COUNT   for averageContextTokenIncrement
  _OCC_COMPLETED_BOUNDARY_INTERVALS    list of requests-between-boundaries
  _OCC_LAST_BOUNDARY_REQUEST_COUNT
  _OCC_NATIVE_COMPACTIONS
  _OCC_PENDING_BOUNDARY  set on a boundary-carrying turn, consumed at settle
  _OCC_COMPACTING        True while the fold call is in flight
"""
from __future__ import annotations

from typing import Any

from ._env import _env_int, _env_float

OCC_ENABLED = _env_int("NX_OCC", 0)
KEEP_RECENT_TOKENS = _env_int("NX_OCC_KEEP_RECENT_TOKENS", 20_000)
CACHE_WRITE_READ_RATIO = _env_float("NX_OCC_CACHE_RATIO", 12.5)
WINDOW_RESERVE_TOKENS = 16_384
FIRST_COMPACTION_REQUEST_SCALE = 2
SUBSEQUENT_COMPACTION_MARGIN = 1.5
SMALL_SAMPLE_SCALE = 0.5
MINIMUM_VARIANCE_SAMPLES = 3
MEMO_TOKENS = 1_000

POST_COMPACTION_PLAN_REMINDER = (
    "Online context compaction finished. The parent task is still active. "
    "Before continuing work, re-issue todo_write with a fresh plan for the "
    "remaining work."
)

# st-level state (declared lazily via getattr with defaults so conftest's
# rewind of nx.* module attributes simply restores the pre-import values).
_DEFAULTS: dict[str, Any] = {
    "_OCC_PLAN": [],
    "_OCC_REQUEST_COUNT": 0,
    "_OCC_LAST_CONTEXT_TOKENS": 0,
    "_OCC_POSITIVE_DELTA_TOTAL": 0,
    "_OCC_POSITIVE_DELTA_COUNT": 0,
    "_OCC_COMPLETED_BOUNDARY_INTERVALS": [],
    "_OCC_LAST_BOUNDARY_REQUEST_COUNT": 0,
    "_OCC_NATIVE_COMPACTIONS": 0,
    "_OCC_PENDING_BOUNDARY": False,
    "_OCC_COMPACTING": False,
    "_OCC_CACHE_DEBT_TOKENS": 0,
    "_OCC_REMAINING_BOUNDARIES": 0,
}


def occ_enabled() -> bool:
    return OCC_ENABLED == 1


def _get(name: str) -> Any:
    from . import _state as st
    if name in _DEFAULTS and not hasattr(st, name):
        setattr(st, name, _DEFAULTS[name])
    return getattr(st, name, _DEFAULTS.get(name))


def _set(name: str, value: Any) -> None:
    from . import _state as st
    setattr(st, name, value)


def record_provider_request(context_tokens: int) -> None:
    count = _get("_OCC_REQUEST_COUNT") + 1
    _set("_OCC_REQUEST_COUNT", count)
    prev = _get("_OCC_LAST_CONTEXT_TOKENS")
    if prev and context_tokens > prev:
        _set("_OCC_POSITIVE_DELTA_TOTAL",
             _get("_OCC_POSITIVE_DELTA_TOTAL") + context_tokens - prev)
        _set("_OCC_POSITIVE_DELTA_COUNT",
             _get("_OCC_POSITIVE_DELTA_COUNT") + 1)
    _set("_OCC_LAST_CONTEXT_TOKENS", context_tokens)


def record_boundary(completed_count: int, remaining_boundaries: int) -> None:
    """A boundary = a todo_write that newly completed >= 1 item."""
    count = _get("_OCC_REQUEST_COUNT")
    intervals = list(_get("_OCC_COMPLETED_BOUNDARY_INTERVALS"))
    intervals.append(count - _get("_OCC_LAST_BOUNDARY_REQUEST_COUNT"))
    _set("_OCC_COMPLETED_BOUNDARY_INTERVALS", intervals)
    _set("_OCC_LAST_BOUNDARY_REQUEST_COUNT", count)
    _set("_OCC_PENDING_BOUNDARY", True)
    _set("_OCC_REMAINING_BOUNDARIES", max(0, remaining_boundaries))


def _estimate_tokens(text: str) -> int:
    return -(-len(text.encode("utf-8", "replace")) // 4)


def _context_window() -> int | None:
    from . import _state as st
    if st.ACTIVE is None:
        return None
    try:
        window = st.ACTIVE.resolve_context_window()
    except Exception:  # noqa: BLE001
        return None
    return window if isinstance(window, int) and window > 0 else None


def decide_compact(write_tokens: int, prior_compactions: int,
                   carried_debt_tokens: int) -> dict[str, Any]:
    """Port of economics.ts decideCompaction. Returns
    {compact: bool, reason: str, archiveTokens, memoTokens, ...}."""
    return _decide(write_tokens, prior_compactions, carried_debt_tokens)


def _decide(write_tokens: int, prior_compactions: int,
            carried_debt_tokens: int) -> dict[str, Any]:
    fixed_tokens = _fixed_tokens()
    archive_tokens = max(0, write_tokens - fixed_tokens - KEEP_RECENT_TOKENS)
    memo_tokens = MEMO_TOKENS
    saving_tokens = archive_tokens - memo_tokens
    if saving_tokens <= 0:
        return {"compact": False, "reason": "non_positive_saving",
                "archiveTokens": archive_tokens, "memoTokens": memo_tokens}

    ratio = CACHE_WRITE_READ_RATIO if CACHE_WRITE_READ_RATIO >= 0 else None
    if ratio is None:
        return {"compact": False, "reason": "cache_ratio_unavailable",
                "archiveTokens": archive_tokens, "memoTokens": memo_tokens}
    incremental = max(0.0, ratio - 1.0)
    breakeven = (write_tokens * incremental) / saving_tokens
    combined_breakeven = (carried_debt_tokens
                          + write_tokens * incremental) / saving_tokens

    intervals = list(_get("_OCC_COMPLETED_BOUNDARY_INTERVALS"))
    mean = (sum(intervals) / len(intervals)) if intervals else 0.0
    if len(intervals) < MINIMUM_VARIANCE_SAMPLES:
        lower_bound = mean * SMALL_SAMPLE_SCALE
    else:
        if len(intervals) > 1:
            var = sum((x - mean) ** 2 for x in intervals) / (len(intervals) - 1)
            lower_bound = max(0.0, mean - (var ** 0.5))
        else:
            lower_bound = mean
    remaining = _get("_OCC_REMAINING_BOUNDARIES")
    unbounded = 1 + int(lower_bound * max(0, remaining))

    total_delta = _get("_OCC_POSITIVE_DELTA_TOTAL")
    delta_count = _get("_OCC_POSITIVE_DELTA_COUNT")
    avg_increment = (total_delta / delta_count) if delta_count > 0 else None
    window = _context_window()
    ctx_tokens = _get("_OCC_LAST_CONTEXT_TOKENS") or write_tokens
    window_upper = None
    if avg_increment and avg_increment > 0 and window:
        window_upper = max(0, int((window - ctx_tokens) / avg_increment))
    expected = min(unbounded, window_upper) if window_upper is not None else unbounded

    window_protection = bool(window and ctx_tokens >= window - WINDOW_RESERVE_TOKENS)
    first = prior_compactions == 0
    if first:
        effective = min(expected * FIRST_COMPACTION_REQUEST_SCALE,
                        window_upper if window_upper is not None else expected * FIRST_COMPACTION_REQUEST_SCALE)
        economic = effective > 0 and breakeven <= effective
    else:
        economic = (expected > 0 and breakeven <= expected
                    and breakeven * SUBSEQUENT_COMPACTION_MARGIN <= expected
                    and combined_breakeven <= expected)
    if window_protection:
        reason = "window_protection"
    elif economic:
        reason = "economic"
    elif expected <= 0:
        reason = "horizon_unavailable"
    elif not first and breakeven * SUBSEQUENT_COMPACTION_MARGIN > expected:
        reason = "deferred_subsequent_margin"
    elif not first and combined_breakeven > expected:
        reason = "deferred_carried_debt"
    else:
        reason = "deferred_economic"
    return {"compact": window_protection or economic, "reason": reason,
            "archiveTokens": archive_tokens, "memoTokens": memo_tokens,
            "breakeven": breakeven, "expected": expected,
            "windowProtection": window_protection}


def _fixed_tokens() -> int:
    """Fixed head cost (system prompt + pinned prefix notes) estimate."""
    import nx as _m
    total = _estimate_tokens(_m.SYSTEM)
    from ._prefix import prefix_notes
    try:
        for msg in prefix_notes():
            total += _estimate_tokens(str(msg.get("content") or ""))
    except Exception:  # noqa: BLE001
        pass
    return total


def maybe_compact_after_boundary(messages: list[dict[str, Any]]) -> bool:
    """The turn_end seam: called after a settled turn that carried a boundary.
    Runs the native fold (nx.compress) when the decision says so. Never
    raises; returns True when a compaction happened."""
    if not occ_enabled() or not _get("_OCC_PENDING_BOUNDARY"):
        return False
    _set("_OCC_PENDING_BOUNDARY", False)
    if _get("_OCC_COMPACTING"):
        return False
    write_tokens = _get("_OCC_LAST_CONTEXT_TOKENS")
    if not write_tokens or write_tokens <= 0:
        return False
    prior = _get("_OCC_NATIVE_COMPACTIONS")
    debt = _get("_OCC_CACHE_DEBT_TOKENS")
    decision = _decide(write_tokens, prior, debt)
    if not decision["compact"]:
        return False
    import nx as _m
    _set("_OCC_COMPACTING", True)
    try:
        result = _m.compress(messages, auto=True)
    except Exception:  # noqa: BLE001
        result = None
    finally:
        _set("_OCC_COMPACTING", False)
    if result is None:
        return False
    _set("_OCC_NATIVE_COMPACTIONS", prior + 1)
    incremental = max(0.0, (CACHE_WRITE_READ_RATIO if CACHE_WRITE_READ_RATIO >= 0 else 0.0) - 1.0)
    _set("_OCC_CACHE_DEBT_TOKENS", int(write_tokens * incremental))
    # Reset the plan/boundary bookkeeping the fold absorbed; the reminder note
    # asks the model to re-issue todo_write, which re-seeds it.
    _set("_OCC_COMPLETED_BOUNDARY_INTERVALS", [])
    _set("_OCC_LAST_BOUNDARY_REQUEST_COUNT", _get("_OCC_REQUEST_COUNT"))
    _set("_OCC_POSITIVE_DELTA_TOTAL", 0)
    _set("_OCC_POSITIVE_DELTA_COUNT", 0)
    _stage_reminder()
    return True


def _stage_reminder() -> None:
    """Stage the post-compaction continuation note (nx's runtime-note channel
    replaces SoL-Pi's invisible continuation turn — the note flushes on the
    next user turn and the model continues the same task)."""
    from ._compress import RUNTIME_NOTE_PREFIX
    from . import _state as st
    st._PENDING_OCC_REMINDER = (f"{RUNTIME_NOTE_PREFIX}{POST_COMPACTION_PLAN_REMINDER}]")


# --- todo_write boundary detection -------------------------------------------

def inspect_todo_write(todos: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Given the NEW todo list (already validated by todo_write), compare with
    the recorded plan: returns (newly_completed, remaining) when >= 1 item
    newly completed, else None. Also records the new plan."""
    prev = {t.get("text"): t.get("status") for t in _get("_OCC_PLAN")}
    newly = sum(1 for t in todos
                if t.get("status") == "done" and prev.get(t.get("text")) != "done")
    remaining = sum(1 for t in todos if t.get("status") != "done")
    _set("_OCC_PLAN", [{"text": t.get("text", ""), "status": t.get("status", "pending")}
                       for t in todos])
    if newly <= 0:
        return None
    return newly, remaining
