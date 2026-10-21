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

# --- Phase 1: transcript detail + as-sent (last-prompt tap) ------------------

def _message_digest(msg: dict[str, Any]) -> str:
    """sha256 hex of utf-8 role + text — the SAME function both the live view
    and the as-sent tap apply, so identical inputs align by construction."""
    import hashlib
    role = str(msg.get("role") or "")
    return hashlib.sha256(
        role.encode("utf-8") + b"\x00"
        + tokens.message_content_text(msg).encode("utf-8")).hexdigest()


def _sent_message_text(msg: dict[str, Any]) -> str:
    """One message rendered as plain text: text content, then tool-call
    name+args, then tool results — everything the model actually saw."""
    out: list[str] = []
    text = tokens.message_content_text(msg).strip()
    if text:
        out.append(text)
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        out.append(f"→ tool call: {fn.get('name') or '?'}({fn.get('arguments') or ''})")
    if msg.get("role") == "tool":
        out.append(f"← tool result: {text}")
    return "\n".join(out) if out else "(empty)"


def _transcript_detail_lines(msg: dict[str, Any], index: int,
                             sent: dict[str, Any] | None,
                             width: int) -> list[str]:
    """The detail body for one message: full text (live), tool-call name+args,
    tool results; in as-sent mode a role/index/digest header and whether the
    message reached the model identical to live. Plain text, no ANSI, so the
    inline `/context t <n>` path can reuse it byte-for-byte."""
    from ._metrics import _abbr
    lines = [f"[{index + 1}] role={msg.get('role', '?')}  "
             f"~{_abbr(_message_tokens(msg))} est"]
    if sent is not None:
        d = _message_digest(msg)
        match = (sent.get("sha256") or sent.get("digest")) == d
        lines.append(f"as-sent index={sent.get('index', index)}  "
                     f"digest={d[:12]}")
        lines.append("identical to live" if match
                     else "differs from live (elided/summarized)")
    lines.append("")
    text = _sent_message_text(msg)
    if not text.strip():
        text = "(empty)"
    import textwrap
    for ln in text.splitlines() or ["(empty)"]:
        # wrap to the panel's own row budget (panel_row clips at
        # _term_width()-6; 72 is the safe floor on the narrowest terminal)
        lines.extend(textwrap.wrap(ln, max(24, min(72, width - 8))) or [""])
    return lines


def _transcript_detail(msg: dict[str, Any], index: int,
                       sent: dict[str, Any] | None = None) -> str:
    """Full transcript of one message as plain text (the view's detail pane
    and `/context t <n>` print the same content; the inline path passes a
    wide width so long lines are not wrapped)."""
    return "\n".join(_transcript_detail_lines(msg, index, sent, 10_000))


def _sent_snapshot_for(index: int) -> dict[str, Any] | None:
    """The as-sent record for message `index` from the last-prompt tap
    (st._LAST_PROMPT: list of {role, index, digest} dicts, newest call last),
    or None when no settled call has been recorded yet."""
    for e in reversed(getattr(st, "_LAST_PROMPT", None) or []):
        if isinstance(e, dict) and e.get("index") == index:
            return e
    return None


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
        self.detail = False          # in the transcript detail pane (esc backs out)
        self.as_sent = False         # live vs as-sent toggle ('a')
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
        if self.detail:
            return (f"{self.sel + 1} · transcript"
                    f"{' · as-sent' if self.as_sent else ''} · esc back")
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
        if self.detail:
            return self._render_detail(width, height)
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

    def _render_detail(self, width: int, height: int) -> list[str]:
        """The transcript detail pane: the FULL selected message (text, tool
        calls, tool results). Read-only subview — no new mutation; as-sent
        mode marks the message identical-to / differing-from live."""
        from ._keys import hint_line
        from ._render import RESET, panel, role
        messages = self._messages()
        idx = min(self.sel, max(0, len(messages) - 1))
        sent = _sent_snapshot_for(idx) if self.as_sent else None
        lines = [f"{role('ACCENT')}transcript{RESET}{role('MUTED')} — "
                 f"{hint_line('context')}{RESET}"]
        if self.as_sent and sent is None:
            body = [f"{role('MUTED')}  — no settled call recorded —{RESET}"]
        else:
            body = [ln[:width] for ln in _transcript_detail_lines(
                messages[idx], idx, sent, width)]
        lines.extend(panel(f"message {idx + 1}/{len(messages)}",
                           body, footer=self._footer()))
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
        if self.detail:
            if key in ("esc", "q"):
                self.detail = False
                return "repaint"
            if key == "a":
                self.as_sent = not self.as_sent
                return "repaint"
            return "repaint"
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
        if key in ("p", "space"):
            return self._toggle_pin()
        if key == "enter":
            if not rows:
                self.notice = "no message rows to open"
                return "repaint"
            self.detail = True
            return "repaint"
        if key == "a":
            self.as_sent = not self.as_sent
            return "repaint"
        if key == "c":
            # Phase 2: jump to the last-compression ledger. The shell's own
            # view stack does the navigation (push + "repaint" hands control
            # over, esc pops back here); nothing is recomputed or mutated.
            if shell is None:
                return "repaint"
            shell.push(compression_view(self.state))
            return "repaint"
        if key == "t":
            # Phase 3: jump to the per-call event log (same push pattern as
            # the `c` jump above; esc pops back here).
            if shell is None:
                return "repaint"
            from ._turnlog import turns_view
            shell.push(turns_view(self.state))
            return "repaint"
        if key == "S":
            # Phase 5: jump to the annotated system-prompt floor (same push
            # pattern as `c`/`t`; esc pops back here). Capital because every
            # lowercase letter on this surface is taken.
            if shell is None:
                return "repaint"
            from ._promptview import prompt_view
            shell.push(prompt_view(self.state))
            return "repaint"
        if key == "g":
            # Phase 4: jump to the tool-gate view (same push pattern as
            # `c`/`t`/`S`; esc pops back here).
            if shell is None:
                return "repaint"
            from ._toolview import tools_view
            shell.push(tools_view(self.state))
            return "repaint"
        if key == "r":
            self.refresh()
            return "repaint"
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


# --- Phase 7: the cumulative session token ledger ----------------------------

def _ledger_lines(metrics: Any = None) -> list[str]:
    """Cumulative session totals over st._METRICS as plain text (no ANSI).

    What the per-call /turns rows show one call at a time, summed over the
    whole session: tokens sent (input + cache_read), the cache-hit share and
    hit %, tokens generated (output + reasoning), how much of that the server
    MEASURED vs what the estimator guessed (the /cost provenance fields), and
    the per-turn tools= overhead actually paid (per-turn floor x settled
    calls). Defensive reads throughout — an unknown/garbage field falls back
    to 0 and nothing here ever raises. No calls this session → one dim
    placeholder line, the /turns convention.
    """
    m = metrics if metrics is not None else getattr(st, "_METRICS", None)
    calls = _int_or(getattr(m, "api_calls", 0)) if m is not None else 0
    if calls <= 0:
        return ["  no calls this session"]

    def _f(name: str) -> int:
        return _int_or(getattr(m, name, 0))

    sent = _f("input_tokens") + max(0, _f("cache_read_tokens"))
    cache_hit = max(0, _f("cache_read_tokens"))
    hit_pct = (100 * cache_hit // sent) if sent > 0 else 0
    generated = _f("output_tokens") + _f("reasoning_tokens")
    est_calls = _f("estimated_calls")
    measured_calls = max(0, calls - est_calls)
    overhead = _turn_overhead_est() * calls
    return [
        f"  ledger · {calls} call(s) this session",
        f"  sent ~{sent} tok · cache-hit ~{cache_hit} ({hit_pct}% hit)",
        f"  generated ~{generated} tok "
        f"(output ~{_f('output_tokens')} · reasoning ~{_f('reasoning_tokens')})",
        f"  measured {measured_calls} call(s) · "
        f"estimated {est_calls} call(s) "
        f"(~{_f('estimated_input_tokens')} in / "
        f"~{_f('estimated_output_tokens')} out est)",
        f"  overhead ~{overhead} tok paid this session "
        f"(~{_turn_overhead_est()} tools schema x {calls} call(s))",
    ]


def _int_or(value: Any) -> int:
    """Ledger counts render as ints; a missing/garbage count renders 0."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def context_view(state: Any = None) -> "ContextView":
    """The /context view object the fullscreen shell runs (`_fullscreen.context`
    is the thin alias the command path uses)."""
    return ContextView(state)


# ---------------------------------------------------------------------------
# Phase 2 — the last-compression ledger, read side
#
# The WRITE side lives in `nx/_compress.py` (it publishes `st._LAST_COMPRESSION`,
# session-lifetime, drained by the test fixture). This module only renders it:
# `/compress --show` prints these lines inline and the `CompressionView` below
# paints the SAME list in the alternate screen, so the two surfaces cannot drift
# (the /context rule).
#
# Ledger schema (defensive reads only — unknown keys ignored, anything missing
# falls back to 0, nothing here ever raises):
#   {"kept": int, "dropped": int, "summarized": int, "summary_tokens_est": int,
#    "pinned": int, "before_messages": int, "after_messages": int, "ts": float,
#    "ranges": [{"start": int, "end": int, "action": str}, ...]}
# `ranges` are HALF-OPEN 0-based indexes into the PRE-compression list, so the
# display form is the 1-based INCLUSIVE `[start+1..end]`.
# ---------------------------------------------------------------------------

#: action -> what it means to the user. An unknown action renders verbatim
#: (a new write-side action must not break an old render).
_COMPRESSION_ACTIONS = {
    # "kept" includes the system + prefix block, so it is not just "kept":
    # those messages were never even considered for folding.
    "kept": "kept — verbatim, untouched",
    "pinned-kept": "kept — pinned, protected from folding",
    "summarized": "summarized — folded into the summary",
    "dropped": "dropped — evicted, not summarized",
}


def _compression_action_label(action: Any) -> str:
    return _COMPRESSION_ACTIONS.get(str(action or "?"), str(action or "?"))


def _compression_ts(ts: Any) -> str:
    """Ledger timestamp as local time; anything unparseable renders as-is."""
    try:
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError, OverflowError):
        return "?" if ts is None else str(ts)


def _compression_range_line(rng: Any) -> str | None:
    """One range as `[start+1..end] <action>` (half-open 0-based → inclusive
    1-based). None for an entry with no usable indexes."""
    if not isinstance(rng, dict):
        return None
    try:
        lo, hi = int(rng.get("start")) + 1, int(rng.get("end"))
    except (TypeError, ValueError):
        return None
    if hi < lo:
        return None
    return f"  [{lo}..{hi}] {_compression_action_label(rng.get('action'))}"


def _compression_report_lines(ledger: dict | None = None) -> list[str]:
    """The last-compression ledger as plain text lines (no ANSI).

    `ledger` defaults to `st._LAST_COMPRESSION`; the None/absent case renders
    the placeholder line the command shows dim. The fullscreen view paints the
    exact same list, so both surfaces agree byte-for-byte.
    """
    led = getattr(st, "_LAST_COMPRESSION", None) if ledger is None else ledger
    if not isinstance(led, dict):
        return ["  no compression recorded this session"]
    head = f"  last compression · {_compression_ts(led.get('ts'))}"
    if led.get("before_messages") is not None \
            or led.get("after_messages") is not None:
        head += (f" · {_int_or(led.get('before_messages'))} → "
                 f"{_int_or(led.get('after_messages'))} messages")
    lines = [head,
             f"  kept {_int_or(led.get('kept'))} · "
             f"dropped {_int_or(led.get('dropped'))} · "
             f"summarized {_int_or(led.get('summarized'))} · "
             f"pinned {_int_or(led.get('pinned'))} · "
             f"summary ~{_int_or(led.get('summary_tokens_est'))} tok est"]
    ranges = [ln for ln in (_compression_range_line(r)
                            for r in (led.get("ranges") or [])) if ln]
    if ranges:
        lines.append("  what happened to the turns "
                     "(1-based, pre-compression numbering):")
        lines.extend(ranges)
    return lines


class CompressionView(View):
    """The `/compress --show` view: the last compression's ledger, read-only.

    The same Phase-D pattern as `ContextView` minus the one deviation
    ContextView was granted — this view mutates nothing at all. Its body IS
    `_compression_report_lines()`, so the inline command output and the view
    agree by construction; `r` re-reads the ledger, esc/q leave.
    """

    surface = "compress"
    title = "last compression"

    def __init__(self, state: Any = None) -> None:
        super().__init__()
        self.state = state
        self.lines: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self.lines = _compression_report_lines()

    def render(self, width: int, height: int) -> list[str]:
        from ._keys import hint_line
        from ._render import RESET, panel, role
        lines = [f"{role('ACCENT')}compress{RESET}{role('MUTED')} — "
                 f"{hint_line('compress')}{RESET}"]
        budget = max(1, height - len(lines) - 2)   # hint line + box borders
        body = [f"{role('MUTED')}{ln[:width]}{RESET}"
                for ln in self.lines[:budget]]
        lines.extend(panel(self.title, body,
                           footer=f"{len(self.lines)} line(s) · the ledger "
                                  f"covers this session only"))
        return lines[:height]

    def on_key(self, key: str, shell: Any) -> str:
        self.notice = ""
        if key == "r":
            self.refresh()
        if key in ("q", "esc", "ctrl-c", "ctrl-d"):
            return "quit"
        return "repaint"


def compression_view(state: Any = None) -> "CompressionView":
    """The /compress --show view object the fullscreen shell runs."""
    return CompressionView(state)
