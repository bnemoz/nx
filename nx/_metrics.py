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
import time
import urllib.request
from os import environ
from typing import Any, TypedDict

from . import _state as st


class UsageDelta(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    reasoning_tokens: int


def _normalize_usage(
    usage: Any,
    timings: Any,
    fallback_chars: int = 0,
    fallback_prompt_chars: int = 0,
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
      reasoning_tokens    completion_tokens_details.reasoning_tokens, if any
                          (reasoning models: MiniMax-M3, DeepSeek-R1, …)

    When neither `usage` nor `timings` is available, a rough estimate is
    derived from `fallback_chars` (client-side character count of streamed
    content + tool-call arguments) for output_tokens, and from
    `fallback_prompt_chars` (character count of the outgoing messages array)
    for input_tokens. This ensures backends like vLLM without
    `stream_options` support still report token counts to agentdash instead
    of silently reporting zero input tokens every turn.
    """
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
        }
    if not usage:
        # Fallback: estimate from client-side character count (~4 chars/token)
        if fallback_chars > 0 or fallback_prompt_chars > 0:
            estimated_out = max(1, fallback_chars // 4) if fallback_chars > 0 else 0
            estimated_in = max(1, fallback_prompt_chars // 4) if fallback_prompt_chars > 0 else 0
            return {
                "input_tokens": estimated_in,
                "output_tokens": estimated_out,
                "cache_read_tokens": 0,
                "reasoning_tokens": 0,
            }
        return None
    prompt_total = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_total = int(getattr(usage, "completion_tokens", 0) or 0)
    # OpenAI/Z.ai: prompt_tokens_details.cached_tokens is the cache portion.
    ptd = getattr(usage, "prompt_tokens_details", None)
    cache_n = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
    # Reasoning models break completion into content + reasoning.
    otd = getattr(usage, "output_tokens_details", None)
    reasoning_n = int(getattr(otd, "reasoning_tokens", 0) or 0) if otd else 0
    return {
        "input_tokens": max(0, prompt_total - cache_n),
        "output_tokens": max(0, completion_total - reasoning_n),
        "cache_read_tokens": cache_n,
        "reasoning_tokens": reasoning_n,
    }


class _Metrics:
    """Cumulative token usage for the current session, plus an optional push.

    A single module-global instance, `st._METRICS`, is created in main() when a
    session starts (and recreated on /reset, reloaded on /resume). When
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
        self.push_url = push_url
        self._push_disabled = False  # latched off after the first failure

    def add(self, delta: UsageDelta | None) -> None:
        """Accumulate a per-call usage dict from _normalize_usage()."""
        if not delta:
            return
        self.input_tokens += delta.get("input_tokens") or 0
        self.output_tokens += delta.get("output_tokens") or 0
        self.cache_read_tokens += delta.get("cache_read_tokens") or 0
        self.reasoning_tokens += delta.get("reasoning_tokens") or 0
        self.api_calls += 1

    def as_meta(self) -> dict[str, Any]:
        """The fields merged into the session JSON on save."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
        }

    def flush(self) -> None:
        """Fire the optional HTTP push. Cumulative totals, so the endpoint
        can compute its own deltas (agentdash's /api/tokens/push does)."""
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
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.push_url, data=data,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1.5)
        except Exception:
            # Dashboard down / wrong port / not running — stop trying so we
            # never add latency to the REPL on every turn.
            self._push_disabled = True


# Module-global recorder. None until main() builds one for the session.
# It's threaded down into the model-turn path so each API call's usage is
# accumulated; the REPL merges its .as_meta() into the session JSON on save.
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
