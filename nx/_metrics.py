"""Usage metrics (default off).

nx already captures every model call's token usage (the streaming
`usage` object from OpenAI/Z.ai, plus llama.cpp's `timings`) to print the
stats footer — and then throws it away. This module retains those numbers
so they can be written into the session JSON (always — it's free, the data
is already in hand) and optionally pushed to an external metrics endpoint
(only when NX_METRICS_URL is set). The endpoint is generic: nx
posts a small JSON blob of cumulative session totals after each turn. One
such consumer is a personal dashboard; the format is dashboard-agnostic.

Accounting matches the OpenAI usage convention (and Hermes's canonical
usage): `input_tokens` EXCLUDES cached tokens (cache is broken out into
`cache_read_tokens`), and `output_tokens` EXCLUDES reasoning tokens
(reasoning is broken out into `reasoning_tokens`). Dashboards that want a
"total in" sum input + cache_read; a "total out" sum output + reasoning.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from os import environ
from collections.abc import Callable
from typing import Any, TypedDict

from . import _state as st
from . import _tokens as tokens


class _AsyncPusher:
    """PF-07 (0.56.0): the optional dashboard push used to run
    ``urllib.request.urlopen(req, timeout=1.5)`` INLINE on the REPL thread,
    1-3 times per turn after the answer rendered — up to ~4.5s of dead time
    before it self-disabled on an unreachable endpoint. Same queue+thread
    pattern as `_logging._AsyncLogWriter`: `submit()` is a `queue.put()` and
    never blocks the caller, so a slow/absent dashboard can never add latency
    to the prompt.

    """

    def __init__(self) -> None:
        self._q: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._disabled = False  # latched off after the first failure

    @property
    def disabled(self) -> bool:
        return self._disabled

    def submit(self, url: str, payload: dict[str, Any], owner: Any) -> None:
        """Queue a cumulative-totals payload. Drops it silently if the push
        has been latched off or the queue is saturated."""
        if self._disabled or not url:
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="nx-metrics-push")
                self._thread.start()
        try:
            self._q.put_nowait((url, payload, owner))
        except queue.Full:  # unbounded by default — defensive only
            pass

    def _run(self) -> None:
        while True:
            url, payload, owner = self._q.get()
            try:
                if not self._disabled:
                    data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        url, data=data,
                        headers={"Content-Type": "application/json"})
                    _urlopen(req, timeout=1.5)
            except Exception:
                # Dashboard down / wrong port / not running — stop trying so we
                # never add latency to the REPL on every turn.
                self._disabled = True
                try:
                    owner._push_disabled = True
                except Exception:
                    pass
            finally:
                self._q.task_done()


_pusher = _AsyncPusher()


def _urlopen(req, timeout=None):
    """Module-local indirection over urllib.request.urlopen so tests (and the
    disabled-latch path) can stub the network without mutating the shared
    stdlib module object — a process-global any other thread can read."""
    return urllib.request.urlopen(req, timeout=timeout)


class UsageDelta(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    reasoning_tokens: int
    estimated: int  # 1 = nx/_tokens fallback estimate (no server usage), 0 = measured


def _normalize_usage(
    usage: Any,
    timings: Any,
    fallback_chars: int | Callable[[], int] = 0,
    fallback_prompt_chars: int | Callable[[], int] = 0,
) -> UsageDelta | None:
    """Pull a {input, output, cache_read, reasoning} token dict out of a
    finished model call, from whichever the server gave us: the standard
    streaming `usage` object (OpenAI/Z.ai) or the llama.cpp `timings` extra
    (where `usage` is null). Returns None if neither has anything to say.

    Numbers are the per-call totals reported by the server, not deltas —
    callers accumulate them. The split is:

      input_tokens        prompt tokens, MINUS cached (matches OpenAI's
                          prompt_tokens minus prompt_tokens_details.cached)
      cache_read_tokens   cached prompt tokens (prompt_tokens_details.cached
                          on OpenAI; cache_n on llama.cpp)
      output_tokens       completion tokens, MINUS reasoning
      reasoning_tokens    reasoning split, if the server reports one — read
                          from output_tokens_details.reasoning_tokens (OpenAI
                          compat / vLLM / Z.ai) or, failing that,
                          completion_tokens_details.reasoning_tokens (OpenAI's
                          own schema); 0 when neither is present

    When neither `usage` nor `timings` is available, a rough estimate is
    derived from `fallback_chars` (client-side character count of streamed
    content + tool-call arguments) for output_tokens, and from
    `fallback_prompt_chars` (character count of the outgoing messages array)
    for input_tokens. This ensures backends like vLLM without
    `stream_options` support still report token counts to agentdash instead
    of silently reporting zero input tokens every turn.

    Swarm P8 (PERF.md PF-23): either fallback may be a zero-arg callable,
    computed ONLY when the estimate path actually runs. Callers used to build
    sum(len(str(m.get("content"))) for messages) eagerly on every settled
    stream - a full-prompt str()+len() scan per turn per agent, wasted on any
    server that reports usage.
    """
    def _resolve(v: int | Callable[[], int]) -> int:
        return v() if callable(v) else v
    # llama.cpp: no `usage`, but a `timings` object on the final chunk with
    # prompt_n / predicted_n / cache_n. There's no reasoning split here —
    # predicted_n is all generated tokens.
    if (not usage) and timings and timings.get("predicted_n"):
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        gen_n = int(timings.get("predicted_n") or 0)
        return {
            "input_tokens": max(0, prompt_n - cache_n),
            "output_tokens": gen_n,
            "cache_read_tokens": cache_n,
            "reasoning_tokens": 0,
            "estimated": 0,
        }
    if not usage:
        # Fallback: estimate from client-side character counts via the shared
        # estimator (nx/_tokens.py, chars/3.5). Flagged `estimated: 1` so
        # accounting (and /cost) can
        # show the provenance instead of billing an estimate as a measured
        # number.
        fallback_chars = _resolve(fallback_chars)
        fallback_prompt_chars = _resolve(fallback_prompt_chars)
        if fallback_chars > 0 or fallback_prompt_chars > 0:
            estimated_out = (max(1, tokens.estimate_tokens_from_chars(fallback_chars))
                             if fallback_chars > 0 else 0)
            estimated_in = (max(1, tokens.estimate_tokens_from_chars(fallback_prompt_chars))
                            if fallback_prompt_chars > 0 else 0)
            return {
                "input_tokens": estimated_in,
                "output_tokens": estimated_out,
                "cache_read_tokens": 0,
                "reasoning_tokens": 0,
                "estimated": 1,
            }
        return None
    prompt_total = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_total = int(getattr(usage, "completion_tokens", 0) or 0)
    # OpenAI/Z.ai: prompt_tokens_details.cached_tokens is the cache portion.
    # Absent breakdown (ptd None, or cached_tokens absent — vLLM omits the
    # whole object when nothing was served from cache) is NOT the same fact
    # as a reported 0. Sentinel -1 keeps the distinction through the ring
    # buffer so /ctx renders "—", not a lying 0%. Cost math treats unknown
    # as uncached (cache_read_tokens clamped at 0 in _source_cost's inputs).
    ptd = getattr(usage, "prompt_tokens_details", None)
    cached_raw = getattr(ptd, "cached_tokens", None) if ptd is not None else None
    cache_n = -1 if cached_raw is None else int(cached_raw)
    # Reasoning models break completion into content + reasoning. Two shapes
    # exist in the wild: output_tokens_details (OpenAI-compat servers, vLLM,
    # Z.ai) and completion_tokens_details (OpenAI's own field name — the
    # docstring always claimed that one; pre-0.43.1 the code only read the
    # former, so the reasoning column sat at 0 on OpenAI-shaped backends).
    otd = getattr(usage, "output_tokens_details", None)
    reasoning_n = int(getattr(otd, "reasoning_tokens", 0) or 0) if otd else 0
    if not reasoning_n:
        ctd = getattr(usage, "completion_tokens_details", None)
        reasoning_n = int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
    return {
        "input_tokens": max(0, prompt_total - max(0, cache_n)),
        "output_tokens": max(0, completion_total - reasoning_n),
        "cache_read_tokens": cache_n,
        "reasoning_tokens": reasoning_n,
        "estimated": 0,
    }


def _record_usage(
    usage: Any,
    timings: Any = None,
    fallback_chars: int | Callable[[], int] = 0,
    fallback_prompt_chars: int | Callable[[], int] = 0,
) -> None:
    """Add one completed API call to the active session's totals.

    Keep this separate from presentation code: streams can end through a
    recovery/error path that intentionally skips the stats footer, and a few
    inexpensive maintenance calls are non-streaming altogether.
    """
    m = st._active_metrics()
    if m is not None:
        m.add(_normalize_usage(
            usage, timings,
            fallback_chars=fallback_chars,
            fallback_prompt_chars=fallback_prompt_chars,
        ))


def _record_api_call(
    resp: Any,
    fallback_chars: int | Callable[[], int] = 0,
    fallback_prompt_chars: int | Callable[[], int] = 0,
) -> None:
    """_record_usage for a non-streaming response object: takes the usage off
    `resp` (falling back to the shared-estimator estimate, flagged
    `estimated`, when
    the server sent none). Exists so auxiliary model calls — the risk
    classifier, the capability/thinking probes, a ctx-window probe that a
    clamping server actually executes — land in the same totals /cost prints
    instead of vanishing from the bill."""
    _record_usage(
        getattr(resp, "usage", None),
        fallback_chars=fallback_chars,
        fallback_prompt_chars=fallback_prompt_chars,
    )


# Cap on the per-turn ring buffer _Metrics keeps for /ctx's cache column.
# /ctx shows at most a dozen rows anyway; 64 turns of history is plenty.
_TURN_RING_MAX = 64


class _Metrics:
    """Cumulative token usage for the current session, plus an optional push.

    A single module-global instance, `st._METRICS`, is created in main() when a
    session starts (and recreated on /reset, reloaded on /resume) — or a
    per-task instance rides the thread-local slot in batch mode (see
    _state.py's _active_metrics()). When
    NX_METRICS_URL is unset it's just an in-memory accumulator that feeds
    the session-JSON totals — no network, no threads, no dependency. When the
    URL is set, each flush() also fires off a fire-and-forget POST so a live
    dashboard can plot tok/min. The push is best-effort: it disables itself
    after the first failure so a missing dashboard never slows the REPL.
    """

    def __init__(
        self,
        session_id: str,
        model: str | None = None,
        source: str | None = None,
        started_at: float | None = None,
        push_url: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.source = source
        self.started_at = started_at or time.time()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.reasoning_tokens = 0
        self.api_calls = 0
        # Provenance (2.5): how many of those api_calls were ESTIMATED by the
        # nx/_tokens fallback instead of measured from a server usage object.
        # /cost prints the split so an estimate never masquerades as a bill.
        self.estimated_calls = 0
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0
        # Per-call ring buffer (newest last, capped) for /ctx's per-turn
        # cache-read column: each entry is one settled model call with its
        # prompt total (input + cache_read, per the OpenAI convention where
        # input EXCLUDES cache) and the cached portion the server reported.
        self.turns: list[dict[str, int]] = []
        # Harness-event ring (explainable-harness Phase 3, newest last, same
        # cap): notable interventions attributable to a settled call —
        # {"ts": float, "kind": str, "detail": str, "call": int | None}.
        # Recorded from record-side code that already touches _Metrics
        # (add()/event()); _turn.py and _dispatch.py stay untouched, so
        # retry/salvage kinds land when their recording points move here.
        self.events: list[dict[str, Any]] = []
        # Per-source scoreboard for /cost: source name → totals dict. A /source
        # switch mid-session is exactly the comparison /cost exists for (local
        # vs hosted on the same task), so usage is bucketed by the ACTIVE
        # source at the time of each call.
        self.by_source: dict[str, dict[str, int]] = {}
        self.push_url = push_url
        self._push_disabled = False  # latched off after the first failure

    def add(self, delta: UsageDelta | None) -> None:
        """Accumulate a per-call usage dict from _normalize_usage()."""
        if not delta:
            return
        self.input_tokens += delta.get("input_tokens") or 0
        self.output_tokens += delta.get("output_tokens") or 0
        # -1 (server sent no cache breakdown) contributes 0 here but is kept
        # verbatim in the ring buffer below for /ctx's "—" rendering.
        self.cache_read_tokens += max(0, delta.get("cache_read_tokens") or 0)
        self.reasoning_tokens += delta.get("reasoning_tokens") or 0
        self.api_calls += 1
        if delta.get("estimated"):
            self.estimated_calls += 1
            self.estimated_input_tokens += delta.get("input_tokens") or 0
            self.estimated_output_tokens += delta.get("output_tokens") or 0
        self.turns.append({
            "prompt_tokens": (delta.get("input_tokens") or 0)
                              + max(0, delta.get("cache_read_tokens") or 0),
            "cache_read_tokens": delta.get("cache_read_tokens") or 0,
        })
        del self.turns[:-_TURN_RING_MAX]
        # Derivable harness events (Phase 3): a settled call that paid full
        # prompt price on a big context is a cache miss the /turns log should
        # name. Threshold avoids flagging every cold first call.
        if prompt_tokens := ((delta.get("input_tokens") or 0)
                             + max(0, delta.get("cache_read_tokens") or 0)):
            cached = delta.get("cache_read_tokens") or 0
            if prompt_tokens > 2048 and cached == 0:
                self.event("cache-miss",
                           f"~{prompt_tokens} prompt tokens, 0 cached",
                           call=len(self.turns) - 1)
        src = st.ACTIVE.name if st.ACTIVE is not None else (self.source or "unknown")
        bucket = self.by_source.setdefault(src, {
            "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
            "reasoning_tokens": 0, "api_calls": 0, "estimated_calls": 0})
        bucket["input_tokens"] += delta.get("input_tokens") or 0
        bucket["output_tokens"] += delta.get("output_tokens") or 0
        bucket["cache_read_tokens"] += max(0, delta.get("cache_read_tokens") or 0)
        bucket["reasoning_tokens"] += delta.get("reasoning_tokens") or 0
        bucket["api_calls"] += 1
        if delta.get("estimated"):
            bucket["estimated_calls"] += 1

    def event(self, kind: str, detail: str = "",
              call: int | None = None) -> None:
        """Record one harness event into the /turns ring (Phase 3). Cheap and
        best-effort: a ring append, nothing more — never raises."""
        try:
            self.events.append({"ts": time.time(), "kind": kind,
                                "detail": str(detail)[:200], "call": call})
            del self.events[:-_TURN_RING_MAX]
        except Exception:
            pass

    def as_meta(self) -> dict[str, Any]:
        """The fields merged into the session JSON on save. by_source rides
        along since 0.43.1 (2.5): a resume used to wipe the /cost scoreboard's
        per-source attribution while keeping the header totals — half a bill."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
            "by_source": self.by_source,
        }

    def flush(self) -> None:
        """Queue the optional HTTP push. Cumulative totals, so the endpoint
        can compute its own deltas (agentdash's /api/tokens/push does).

        PF-07 (0.56.0): this is now a queue put — the actual POST runs on the
        `_pusher` background thread, never on the REPL thread."""
        if not self.push_url or self._push_disabled:
            return
        payload = {
            "session_id": self.session_id,
            "model": self.model,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
            "ended_at": time.time(),
        }
        _pusher.submit(self.push_url, payload, self)


# Recorder binding. The REPL's recorder is the module-global `st._METRICS`
# (None until main() builds one for the session); it's threaded down into the
# model-turn path so each API call's usage is accumulated, and the REPL merges
# its .as_meta() into the session JSON on save. Batch workers install a
# per-task recorder on the thread-local `st._METRICS_TLS` slot instead (see
# _state.py's _active_metrics) so parallel workers never race the global.
st._METRICS = None


def _env_metrics_url() -> str | None:
    """NX_METRICS_URL — optional endpoint to receive cumulative token
    totals after each turn. Default unset (metrics stay local to the session
    JSON, nothing leaves the machine). No scheme/host validation beyond
    "looks like a URL" — a bad value just latches the push off on first try."""
    url = environ.get("NX_METRICS_URL", "").strip()
    return url or None


def _abbr(n: int) -> str:
    """Compact token count for the stats footer: 832 → '832', 1500 → '1.5K',
    78825 → '78K', 1234567 → '1.2M'. Keeps the footer tidy once context grows."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000:
        return f"{n // 1000}K"
    if n < 10_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n // 1_000_000}M"


def _precise_abbr(n: int) -> str:
    """Like _abbr but keeps two decimals in the K/M/B ranges.

    Used for the live tool-call-args counter ("generating tool call write_file
    args 25.15K chars"), where watching the number tick up as args stream in
    is part of the feedback — so we don't want to round 25,152 down to '25K'.
    832 → '832', 25152 → '25.15K', 1234567 → '1.23M'."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.2f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1_000_000_000:.2f}B"
