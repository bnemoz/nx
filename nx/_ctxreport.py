"""Per-message token accounting and pinned-turn protection for /ctx + compress.

Two pieces:

1. `_estimate_tokens` / `_messages_token_report` — per-message estimates via
   the shared estimator in `nx/_tokens.py` (chars/3.5) with a per-role
   breakdown. Used by /ctx to show
   where the context actually goes.
   When the active source's /tokenize endpoint works, /ctx uses it for the
   total (exact); per-message numbers stay estimated (tokenizing each message
   separately would round-trip the server N times).

2. Per-turn overhead — `_turn_overhead_est()` / `_overhead_lines()`: the
   tool schemas sent on every model call (`tools=`) live outside the message
   list, so without this /ctx's total quietly misses them and undercounts
   the real per-turn floor. Counted the same way `nx --tokens` counts its
   floor: JSON-serialized schemas, same estimator. Rendered as a footer line,
   not a message row (they aren't messages — pinning a tool schema makes no
   sense).

3. Pinned turns — `_ctx_pin(index_range)` marks message indexes that
   compress() must keep verbatim (pinned turns move into the kept tail even
   when they're old). State is a list of indexes on st (session lifetime,
   persisted by the session save like todos — see _repl_state).

4. `ContextView` — the full-screen /context view (the Phase-D pattern: the view
   lives in the module that owns the data). It renders the report through
   `_render.py`/`_keys.py`, and is the one view allowed a mutation — pinning —
   which it hands back to the fullscreen shell as a deferred `pending_action`
   calling the EXISTING `_ctx_pin`/`_ctx_unpin` helpers above (no new pin
   logic).
"""
from __future__ import annotations

from typing import Any

from . import _state as st
from . import _tokens as tokens
from ._fullscreen import View


def _estimate_tokens(text: str) -> int:
    """Thin wrapper over nx._tokens.estimate_tokens (kept as the patchable
    module attribute /ctx's callers know)."""
    return tokens.estimate_tokens(text)


def _message_content_text(msg: dict[str, Any]) -> str:
    """The message's plain text (used for previews as well as token math)."""
    return tokens.message_content_text(msg)


def _message_tokens(msg: dict[str, Any]) -> int:
    """Estimated tokens for one message, including native tool_calls args."""
    return tokens.estimate_message_tokens(msg)


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
    with the shared estimator over the JSON serialization, exactly how
    `nx --tokens` counts its TOOLS line. Never raises: /ctx is a UX nicety, not a correctness path."""
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


# ---------------------------------------------------------------------------
# /context — alternate-screen view (Phase 1 of the 3-phase view work: same
# Phase-D pattern as /tabs, /diff, /sessions, /todo — the view lives in the
# module that owns the data). One deliberate deviation from those views: a
# view is otherwise read-only, but /context's whole point is pinning, so `p`
# hands the mutation back to the shell as a DEFERRED pending_action that calls
# the EXISTING `_ctx_pin`/`_ctx_unpin` helpers (no new pin logic here): the
# shell leaves the alternate screen first, then the action runs, so no state
# changes underneath a live frame. The inline one-shot print stays as the
# command's fallback, byte-identical to the pre-view `/context` output.
# ---------------------------------------------------------------------------

# message role -> semantic render role (never a raw palette hue: the Phase B
# grep guard enforces it). Unknown roles degrade to MUTED.
_MSG_ROLES = {
    "user": "ACCENT",
    "assistant": "SUCCESS",
    "system": "MUTED",
    "tool": "TODO",
}


def _context_row(row: tuple[Any, ...], selected: bool, pinned: bool,
                 hit: float | None, width: int) -> str:
    """One message row: selection marker, pin marker, the same columns the
    inline report prints (index, role, est tokens, cache hit, preview)."""
    from ._ansi import clip_display
    from ._render import RESET, role
    i, role_name, t, preview = row[0], row[1], row[2], row[3]
    marker = "›" if selected else " "
    pin = "📌" if pinned else " "
    hit_s = "—" if hit is None else f"{hit * 100:.0f}%"
    text = clip_display(f"{pin}{i + 1:>4}  {role_name:<9} {t:>6}  "
                        f"{hit_s:>4}  {preview}", width - 5)
    return (f"{role('ACCENT')}{marker}{RESET} "
            f"{role(_MSG_ROLES.get(role_name, 'MUTED'))}{text}{RESET}")


class ContextView(View):
    """The `/context` view: one row per message, selected with j/k, `p` (or
    enter/space) toggles the pin that protects the turn from /compress.

    The report is re-read whenever the message count changed (a background turn
    appending to the session shows up without reopening the view) and on `r`;
    between those the cached report is reused, because `_messages_token_report`
    may call the server's /tokenize endpoint for its exact total and a repaint
    every 0.25s must not cost a round trip. Pin markers are read live off
    `st._CTX_PINNED` on every frame, so the marker never lags the state.
    """

    surface = "context"
    title = "context"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.state = state
        self.sel = 0
        self.report: dict[str, Any] = {}
        self.n_messages = -1
        self.refresh()

    def _messages(self) -> list[dict[str, Any]]:
        return list(getattr(self.state, "messages", None) or [])

    def refresh(self) -> None:
        """Re-read the report from the session's live message list."""
        messages = self._messages()
        self.report = _messages_token_report(messages)
        self.n_messages = len(messages)
        self.sel = min(self.sel, max(0, len(self.report.get("rows") or []) - 1))

    def _heading(self) -> str:
        """Panel title: totals, the exact count when the server gave one, and
        the window share — all of it already in the report / a live attribute
        read, nothing recomputed."""
        from ._metrics import _abbr
        rep = self.report
        total = rep.get("total_est", 0)
        head = f"context · {len(rep.get('rows') or [])} message(s) · ~{total} est"
        exact = rep.get("total_exact")
        if isinstance(exact, int) and exact > 0:
            head += f" · {exact} exact"
        mx = st.ACTIVE._context_window if st.ACTIVE else None
        if isinstance(mx, int) and mx > 0:
            head += f" · ≈{100 * total // mx}% of {_abbr(mx)} window"
        return head

    def _footer(self) -> str:
        """Selection position, pin count and the per-role subtotals (by_role
        excludes the per-turn tools= overhead, which gets its own line above)."""
        from ._metrics import _abbr
        by_role = self.report.get("by_role") or {}
        role_s = "  ".join(f"{r}:{_abbr(t)}"
                           for r, t in sorted(by_role.items(), key=lambda kv: -kv[1]))
        n = len(self.report.get("rows") or [])
        pos = f"{self.sel + 1}/{n}" if n else "0/0"
        pins = len(_pinned_indexes())
        pin_s = f"{pins} pinned · " if pins else ""
        return f"{pos} · {pin_s}{role_s}"

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        if len(self._messages()) != self.n_messages:
            self.refresh()          # a turn landed since the last frame
        rep = self.report
        rows = list(rep.get("rows") or [])
        cache = rep.get("cache") or []
        pinned = _pinned_indexes()
        n = len(rows)
        self.sel = min(self.sel, max(0, n - 1))
        lines = [f"{role('ACCENT')}context{RESET}{role('MUTED')} — "
                 f"{hint_line('context')}{RESET}"]
        # The tools= schemas are in total_est but have no message row: name them
        # above the box (the same sentence the inline report prints below it).
        lines.extend(f"{role('MUTED')}{ln[:width]}{RESET}"
                     for ln in _overhead_lines(rep.get("total_est", 0),
                                               rep.get("overhead_est", 0)))
        budget = max(1, height - len(lines) - 2)   # header line(s) + box borders
        body: list[str] = []
        if not rows:
            body.append(f"{role('MUTED')}  (no messages yet — /context prints "
                        f"the same report inline){RESET}")
        else:
            # Windowed on the selection (the TodoView rule): a long history
            # scrolls instead of clipping the row the cursor sits on; the
            # overflow note costs one body line only when something is hidden.
            shown = budget if n <= budget else max(1, budget - 1)
            start = max(0, min(self.sel - shown + 1, n - shown))
            stop = min(n, start + shown)
            for i in range(start, stop):
                hit = cache[i] if i < len(cache) else None
                body.append(_context_row(rows[i], i == self.sel,
                                         rows[i][0] in pinned, hit, width))
            hidden = n - (stop - start)
            if hidden > 0:
                body.append(f"{role('MUTED')}  … {hidden} more row"
                            f"{'s' if hidden != 1 else ''} (j/k scroll){RESET}")
        lines.extend(panel(self._heading(),
                           [ln[:width] for ln in body],
                           footer=self._footer()))
        if self.notice:
            lines.append(f"{role('WARNING')}{self.notice[:width]}{RESET}")
        return lines[:height]

    def _toggle_pin(self) -> str:
        """Queue the pin/unpin of the selected row as the deferred action; the
        shell runs it AFTER leaving the alternate screen (the view's only
        sanctioned mutation path). The mutation itself is the existing
        `_ctx_pin`/`_ctx_unpin` pair — this view adds no pin logic."""
        rows = list(self.report.get("rows") or [])
        if not rows:
            self.notice = "no message rows to pin"
            return "repaint"
        idx = rows[min(self.sel, len(rows) - 1)][0]
        spec = str(idx + 1)                 # pin specs are 1-based
        was_pinned = idx in _pinned_indexes()

        def _act() -> None:
            from ._render import commit_line
            messages = self._messages()
            if was_pinned:
                removed, remaining = _ctx_unpin(spec, messages)
                commit_line(f"  unpinned {len(removed)} turn(s) "
                            f"({len(remaining)} pinned remain)")
            else:
                added, pinned = _ctx_pin(spec, messages)
                commit_line(f"  pinned {len(added)} turn(s) — compress will "
                            f"keep them verbatim ({len(pinned)} pinned)")

        self.pending_action = _act
        return "run"

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        rows = list(self.report.get("rows") or [])
        last = max(0, len(rows) - 1)
        if key == "j/k":
            # The registry advertises the pair as ONE hint label ("j/k move");
            # the two real keys are handled below (the ReviewView convention).
            return "repaint"
        if key in ("j", "down"):
            self.sel = min(last, self.sel + 1)
            return "repaint"
        if key in ("k", "up"):
            self.sel = max(0, self.sel - 1)
            return "repaint"
        if key == "pagedown":
            self.sel = min(last, self.sel + 10)
            return "repaint"
        if key == "pageup":
            self.sel = max(0, self.sel - 10)
            return "repaint"
        if key in ("enter", "p", "space"):
            return self._toggle_pin()
        if key == "r":
            self.refresh()
            return "repaint"
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def context_view(state: Any = None) -> "ContextView":
    """The /context view object the fullscreen shell runs (`_fullscreen.context`
    is the thin alias the command path uses)."""
    return ContextView(state)
