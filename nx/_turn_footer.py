"""The per-turn stats footer and the context-window probe.

Split out of `_turn.py` (which keeps model_turn / the loop). `_print_stats_footer`
prefers llama.cpp `timings` (server-measured, real tok/s); `_capture_last_prompt_tokens`
runs on EVERY settled stream — including give-up paths that return before the
footer is ever printed — so `_maybe_autocompress()` never checks a stale token
count. `_ensure_ctx_probe` resolves a Source's max context window on a
background thread, sharing `_capability.STARTUP_PROBE_LOCK` so the two probes
don't both queue ahead of the user's first prompt."""
from __future__ import annotations

import threading
from typing import Any, Optional

from ._ansi import DIM, RESET, C
from ._layout import REASONING_DIVIDER_CELLS
from ._metrics import _abbr
from ._render import BOX_BL, BOX_H, commit_line as _commit_line, kv_line
from . import _state as st

# renders the assistant's final-answer markdown live — themed like the rest
# of the terminal (raw-ANSI `C` roles elsewhere) instead of rich's default
# palette; retheme_console() restyles it on a live /theme switch.
#
# SPEC_LIVEENGINE.md: this is the process-wide SHARED console (one object, owned
# by `_rich.shared_console()`), not a module-private one built at import. The
# /tabs region Live and every mid-turn writer go through it too, because rich
# composes same-console Lives (and Console.print during a Live) into one block,
# while a second console on stdout fights for the cursor.
from ._rich import shared_console as _shared_console
from ._rich import themed_console  # noqa: F401  (re-exported, as before)

_say_console = _shared_console()


def _ctx_field(prompt_n: int) -> str:
    """The colored `ctx` field for the stats footer: `<cur>/<max> ctx`.
    The max context window is cyan; the current context is colored by
    utilization — green under 30%, yellow under 60%, red at or above.
    When the max isn't known yet (probe still pending or unreachable)
    the field degrades to `<cur> ctx` with no color, and a background
    probe is kicked off so a later turn can fill in the max."""
    from ._turn import _ensure_ctx_probe
    cur = _abbr(prompt_n)
    if st.ACTIVE is None:
        return f"{cur} ctx"
    mx = st.ACTIVE._context_window
    if not isinstance(mx, int) or mx <= 0:
        _ensure_ctx_probe(st.ACTIVE)
        return f"{cur} ctx"
    mx_s = _abbr(mx)
    ratio = prompt_n / mx if mx > 0 else 0.0
    if ratio < 0.30:
        color = C.SUCCESS
    elif ratio < 0.60:
        color = C.WARNING
    else:
        color = C.ERROR
    # RESET after each colored run, then re-apply DIM so the rest of the
    # footer (joined around this field) stays dim.
    return f"{color}{cur}{RESET}{DIM}/{C.STATUS_BAR}{mx_s}{RESET}{DIM} ctx"


_CTX_PROBE_STARTED = {}  # id(Source) -> PROBE_GENERATION being probed (no dupes)


def _ensure_ctx_probe(src: Any) -> None:
    """Start a background thread to resolve a Source's max context window if
    it isn't already known, it has not already missed, and no probe is in
    flight for it. The result lands in src._context_window; the footer picks
    it up on a later turn. Idempotent via _CTX_PROBE_STARTED (cleared when the
    thread finishes).

    A local server's metadata endpoints can 503 "Loading model" for tens of
    seconds at cold start (the weights are still mmapping) even while a chat
    slot is already serving. A retry must be explicitly requested after that
    state changes: retrying automatically is unsafe because some servers
    silently clamp the oversized chat probe and may generate until their
    context limit instead of returning a 400.

    Shares _capability.STARTUP_PROBE_LOCK with the capability probe so the two
    background probes don't both queue a request against a single-slot local
    server ahead of the user's actual first prompt."""
    from ._openai_client import probe_scope
    if getattr(src, "_context_window_probe_attempted", False):
        return
    sid = id(src)
    gen = st.PROBE_GENERATION
    if _CTX_PROBE_STARTED.get(sid) == gen:
        return
    _CTX_PROBE_STARTED[sid] = gen

    def _work() -> None:
        try:
            if isinstance(src._context_window, int) and src._context_window > 0:
                return  # resolved by another path (e.g. a /source switch)
            from ._capability import STARTUP_PROBE_LOCK
            with STARTUP_PROBE_LOCK:
                if isinstance(src._context_window, int) and src._context_window > 0:
                    return
                with probe_scope(lambda: st.PROBE_GENERATION != gen):
                    src.resolve_context_window()
        except Exception:
            pass  # never raise from a background probe
        finally:
            if _CTX_PROBE_STARTED.get(sid) == gen:
                _CTX_PROBE_STARTED.pop(sid, None)

    t = threading.Thread(target=_work, daemon=True, name="nx-ctx-probe")
    t.start()


def _capture_last_prompt_tokens(usage: Any, timings: Optional[dict]) -> None:
    """Cache the just-seen prompt-token count in st._LAST_PROMPT_TOKENS and the
    prompt-cache hit fraction in st._LAST_CACHE_HIT.

    Split out of _print_stats_footer so it can run on EVERY settled stream —
    including the loop_cut give-up paths (collapse / exhausted reasoning-stall
    retries) that return before the footer is ever printed. Those turns still
    grow `messages` (captured reasoning, runtime-note nudges), so skipping this
    left _maybe_autocompress() checking a stale, smaller token count right when
    autocompress was most needed.
    """
    prompt_n = 0
    cache_n = 0
    reported = False  # server actually sent a cache breakdown this call?
    if timings and timings.get("predicted_n"):
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        reported = True
    elif usage and (usage.prompt_tokens or 0):
        prompt_n = int(usage.prompt_tokens or 0)
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None and getattr(ptd, "cached_tokens", None) is not None:
            cache_n = int(ptd.cached_tokens)
            reported = True
    if prompt_n:
        st._LAST_PROMPT_TOKENS = prompt_n
        # Fraction of the prompt served from the server's KV/prompt cache.
        # A steady high value here is the regression signal for append-only
        # discipline (memory_write cache keys, /undo-style message edits): a
        # silent prefix bust shows up as this collapsing to ~0.
        #
        # "Server reported 0" and "server reported nothing" are different
        # facts, so they render differently: 0.0 (red, a real signal) vs None
        # (no `cached` field at all). Several OpenAI-compatible backends
        # (vLLM, some gateways) omit prompt_tokens_details entirely whenever
        # nothing was served from cache — or always — and rendering that as
        # 0% cached turns the footer into a permanent false alarm.
        st._LAST_CACHE_HIT = (min(1.0, cache_n / prompt_n)
                              if reported else None)


def _cache_hit_field(hit: Optional[float]) -> str:
    """The colored `cached: N%` field for the stats footer. Hit-rate color:
    green at/above 75%, yellow at/above 40%, red below — a cold (first-turn)
    or busted prefix is exactly what you want to notice. None (server doesn't
    report cache numbers) renders nothing."""
    if hit is None:
        return ""
    pct = hit * 100.0
    if pct >= 75:
        color = C.SUCCESS
    elif pct >= 40:
        color = C.WARNING
    else:
        color = C.ERROR
    return f"{color}{pct:.0f}%{RESET}{DIM} cached"


def _tps_field(tps: float, trusted: bool) -> str:
    """The colored `tok/s` footer field. Server-reported paces (llama.cpp
    timings) render bare; client-side estimates render with a `~` whenever the
    decode window was too short to trust, because five decimals of precision
    on a 0.1s span is a lie about noise, not a measurement."""
    num = f"{C.STATUS_BAR}~{tps:5.1f}" if not trusted else f"{C.STATUS_BAR}{tps:5.1f}"
    return f"{num}{RESET}{DIM} tok/s"


def _window_tps(gen_n: float, window: float) -> float:
    """gen_n tokens over a decode window, 0.0-safe."""
    return gen_n / window if window > 0 else 0.0


# --- per-user-turn footer accumulation ---------------------------------------
# (0.148.2) model_turn() runs once per step of the user turn's loop, and every
# step used to print its own stats row — a staircase of footers whenever a
# turn spanned several model calls. Now the loop brackets accumulation
# (begin/end_user_turn_footer in _run_model_turn_loop): every step RECORDS its
# numbers instead of rendering, and ONE aggregated footer flushes when the
# loop settles — covering every exit status through the loop's single finally.
# Keyed per-thread: concurrent sub-agent loops call model_turn via their own
# threads; they are quiet=True so they record nothing, but thread-locality
# guarantees their tallies can never mix with the parent's. Outside a
# bracketed loop (tests, model_turn called directly) the footer renders
# exactly as before — the accumulator is simply absent.
_FOOTER_ACC = threading.local()


class _StepAcc:
    """One user turn's accumulated step numbers."""

    __slots__ = ("gen_n", "chars", "wall", "t_first", "ttft_ms", "prompt_n",
                 "steps", "last")

    def __init__(self) -> None:
        self.gen_n = 0       # sum of token counts from real token sources
        self.chars = 0       # streamed chars from steps with no token source
        self.wall = 0.0      # sum of per-step elapsed
        self.t_first = None  # FIRST token-bearing step's first-token time
        self.ttft_ms = 0.0   # FIRST server-reported ttft (0 = none reported)
        self.prompt_n = 0    # LAST token-bearing step's prompt size
        self.steps = 0
        self.last = None     # last step's raw footer args, for the flush


def _acc() -> Optional[_StepAcc]:
    return getattr(_FOOTER_ACC, "acc", None)


def begin_user_turn_footer() -> None:
    """Start accumulating this thread's user turn (called by the turn loop)."""
    if _acc() is None:
        _FOOTER_ACC.acc = _StepAcc()


def end_user_turn_footer() -> None:
    """Flush the accumulated user-turn footer (the loop's finally calls it).

    Renders nothing when the turn was quiet (sub-agents) or recorded no steps
    (the turn died before the first stream settled — the give-up paths that
    never reached a footer). A single-step turn re-renders exactly that
    step's footer — byte-identical to the pre-0.148.2 per-step row. A
    multi-step turn aggregates: gen tokens sum over the steps that had a real
    token source (steps mixing sources still sum — a conservative
    approximation), wall is the summed step elapsed, tok/s is recomputed over
    the total and marked untrusted (`~` — a sum of decode windows is not a
    server measurement), ttft keeps the FIRST step's value, and ctx keeps the
    LAST step's prompt size (the most recent reading). The cache-hit field
    already reads st._LAST_CACHE_HIT, which _capture_last_prompt_tokens keeps
    at the last settled step."""
    acc = _acc()
    if acc is None:
        return
    _FOOTER_ACC.acc = None
    if acc.steps == 0:
        return
    if acc.steps == 1:
        _render_footer(*acc.last)
        return
    usage, timings, chars, t_first, elapsed, text, tcs, parser = acc.last
    if acc.gen_n:
        timings = {"prompt_n": acc.prompt_n, "predicted_n": acc.gen_n,
                   "predicted_per_second": _window_tps(acc.gen_n, acc.wall),
                   "prompt_ms": acc.ttft_ms * 1000.0,
                   "predicted_per_second_trusted": False}
        usage, chars = None, 0
    elif acc.chars:
        usage, timings, chars = None, None, acc.chars
    _render_footer(usage, timings, chars, acc.t_first, acc.wall, text, tcs,
                   parser, aggregated=True)


def _acc_record(acc: _StepAcc, usage: Any, timings: Optional[dict],
                streamed_chars: int, t_first: Optional[float], elapsed: float,
                text: str, tcs: dict, parser: Optional[Any]) -> None:
    """Fold one settled step's numbers into the user-turn accumulator."""
    acc.steps += 1
    acc.wall += max(0.0, elapsed)
    if acc.t_first is None and t_first is not None:
        acc.t_first = t_first
    if timings and timings.get("predicted_n"):
        acc.gen_n += int(timings["predicted_n"])
        if timings.get("prompt_ms") and acc.ttft_ms == 0.0:
            acc.ttft_ms = timings["prompt_ms"] / 1000.0
        acc.prompt_n = int(timings.get("prompt_n") or 0)
    elif usage and (usage.completion_tokens or 0):
        acc.gen_n += int(usage.completion_tokens or 0)
        acc.prompt_n = int(usage.prompt_tokens or 0)
    else:
        acc.chars += streamed_chars
    acc.last = (usage, timings, streamed_chars, t_first, elapsed, text, tcs,
                parser)


def _footer_row(parts: list[str], tally: str) -> None:
    """Emit the per-turn stats footer row (UIUX_SPEC B1).

    `parts` are pre-rendered fields, each already carrying its own role color
    and its `{RESET}{DIM} label` tail — the per-field hues (ctx utilization,
    cache hit rate) are per-value decisions the component cannot make for the
    caller, so `kv_line`'s role machinery is bypassed and only its `╰ ` lead
    shape is kept (the glyph itself comes from `_render.BOX_BL`). `tally` is
    the self-healing work note (already ` · `-joined).
    """
    _commit_line(f"{DIM}  {BOX_BL} {(f'{RESET}{DIM} · ').join(parts)}{RESET}" + tally)


def _print_stats_footer(usage: Any, timings: Optional[dict], streamed_chars: int, t_first: float | None,
                         elapsed: float, text: str, tcs: dict, messages: Optional[list] = None,
                         parser: Optional[Any] = None, quiet: bool = False) -> None:
    """Render the per-turn stats footer.

    Inside a bracketed turn loop (one user turn), each step RECORDS its
    numbers and the loop's finally prints ONE aggregated footer — a
    multi-step turn no longer stacks a stats row per step. A step whose only
    visible output was collapsed reasoning (no prose, no tool calls) still
    renders a dim divider + `(step: reasoning)` label above the final footer
    so that step stays visible.

    Prefers llama.cpp `timings` (server-measured, real tok/s) when present;
    falls back to the standard streaming `usage` object (OpenAI/Z.ai/vLLM)
    divided by the client-side generation window (parser.gen_window(): the
    first→last token span, NOT wall-clock — wall-clock includes the prefill
    head and the render/teardown tail, which is what made short turns and
    just-read-a-file turns swing wildly); then to a char-based estimate; then
    to wall-clock only. Untrustworthy windows (below _stream_parse's
    _TPS_TRUST_WINDOW) render with a `~` so a burst-delivered short generation
    doesn't masquerade as a precise measurement.
    Prompt-token caching itself lives in _capture_last_prompt_tokens(), called
    separately so it also runs on turns that never reach this footer.
    """
    if quiet:
        return
    acc = _acc()
    if acc is not None:
        # Bracketed turn loop: record, never render — the loop's finally
        # flushes one aggregated footer (end_user_turn_footer).
        _acc_record(acc, usage, timings, streamed_chars, t_first, elapsed,
                    text, tcs, parser)
        return
    _render_footer(usage, timings, streamed_chars, t_first, elapsed, text,
                   tcs, parser)


def _render_footer(usage: Any, timings: Optional[dict], streamed_chars: int,
                   t_first: Optional[float], elapsed: float, text: str, tcs: dict,
                   parser: Optional[Any], aggregated: bool = False) -> None:
    """The footer's render ladder (usage → timings → char estimate → bare
    wall), unchanged since 0.30.0 except where noted. `aggregated` marks a
    multi-step user-turn flush: skip the trivial-turn short-circuit (a cheap
    final step still rides on real earlier work) and read the timings trust
    flag, which the aggregated synthesis sets to False (see
    end_user_turn_footer — a sum of decode windows is not a server
    measurement)."""
    # Trivial turn: a bare few tokens of tool-call args (a quick router step)
    # earns no full stats ladder — the wall time is the only useful number.
    if (not aggregated and not text.strip() and not getattr(parser, "reasoning_only_chars", 0)
            and usage is not None and (usage.completion_tokens or 0) < 100
            and not st._TALLY_ARG_REPAIRS and not st._TALLY_RETRIES):
        _commit_line(kv_line([("", f"{DIM}{elapsed:4.1f}s wall")]))
        return
    # Step with no prose (collapsed reasoning, possibly a tool call whose
    # block renders after this footer, or whose output is collapsed): a dim
    # divider + label so consecutive bare footers read as distinct steps.
    # Only prose disqualifies — `streamed_chars` counts tool-call args and
    # `tcs` is set by any tool-call delta, so a think→tool-call step with
    # collapsed reasoning used to slip past both guards and render as a bare
    # footer with nothing above it.
    if (parser is not None and getattr(parser, "reasoning_only_chars", 0) > 0
            and not getattr(parser, "tcs", None) and not text.strip()):
        _commit_line(f"{DIM}  {BOX_H * REASONING_DIVIDER_CELLS}{RESET}")
        _commit_line(f"{DIM}  (step: reasoning · /reasoning to expand){RESET}")
    # 0.30.0: one footer field for the self-healing work this turn — args
    # repairs applied + corrective retries — so the knobs are tunable in
    # practice ("2 args repaired · 1 retry"). Folded into every footer branch.
    # (st._TALLY_* route through nx/_recovery.py's descriptors: the running
    # loop's private tally, or the module globals outside any loop.)
    tally = ""
    if st._TALLY_ARG_REPAIRS or st._TALLY_RETRIES:
        bits = []
        if st._TALLY_ARG_REPAIRS:
            bits.append(f"{st._TALLY_ARG_REPAIRS} arg{'s' if st._TALLY_ARG_REPAIRS != 1 else ''} repaired")
        if st._TALLY_RETRIES:
            bits.append(f"{st._TALLY_RETRIES} retr{'ies' if st._TALLY_RETRIES != 1 else 'y'}")
        tally = " · " + " · ".join(bits)
    if timings and timings.get("predicted_n"):
        prompt_n = timings.get("prompt_n", 0)
        gen_n = timings["predicted_n"]
        tps = timings.get("predicted_per_second", 0)
        trusted = timings.get("predicted_per_second_trusted", True)
        ctx = _ctx_field(prompt_n)
        prompt_ms = timings.get("prompt_ms", 0)
        ttft = (prompt_ms / 1000.0) if prompt_ms else t_first
        parts = [f"{C.STATUS_BAR}{gen_n}{RESET}{DIM} tok", _tps_field(tps, trusted), ctx]
        if st.MODEL:
            parts.append(f"{C.STATUS_BAR}{st.MODEL}{RESET}")
        cached = _cache_hit_field(st._LAST_CACHE_HIT)
        if cached:
            parts.append(cached)
        if ttft:
            parts.append(f"{C.STATUS_BAR}{ttft*1000:4.0f}{RESET}{DIM}ms ttft")
        parts.append(f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall")
        _footer_row(parts, tally)
    elif usage and (usage.completion_tokens or 0):
        gen_n = usage.completion_tokens or 0
        prompt_n = usage.prompt_tokens or 0
        window, trusted = (parser.gen_window(elapsed) if parser is not None
                           else (max(0.0, (elapsed - t_first) if t_first else elapsed), False))
        ctx = _ctx_field(prompt_n)
        parts = [f"{C.STATUS_BAR}{gen_n}{RESET}{DIM} tok",
                 _tps_field(_window_tps(gen_n, window), trusted), ctx]
        if st.MODEL:
            parts.append(f"{C.STATUS_BAR}{st.MODEL}{RESET}")
        cached = _cache_hit_field(st._LAST_CACHE_HIT)
        if cached:
            parts.append(cached)
        if t_first:
            parts.append(f"{C.STATUS_BAR}{t_first*1000:4.0f}{RESET}{DIM}ms ttft")
        parts.append(f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall")
        _footer_row(parts, tally)
    elif streamed_chars > 0:
        gen_n = max(1, streamed_chars // 4)
        window, trusted = (parser.gen_window(elapsed) if parser is not None
                           else (max(0.0, (elapsed - t_first) if t_first else elapsed), False))
        parts = [f"≈{C.STATUS_BAR}{_abbr(gen_n)}{RESET}{DIM} tok",
                 _tps_field(_window_tps(gen_n, window), trusted),
                 f"{C.STATUS_BAR}{elapsed:4.1f}{RESET}{DIM}s wall"]
        _footer_row(parts, tally)
    elif text or tcs:
        # One field, no label: kv_line's bare `╰ <value>` row.
        _commit_line(kv_line([("", f"{DIM}{elapsed:4.1f}s wall")]) + tally)
