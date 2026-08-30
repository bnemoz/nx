"""Per-source usage/cost scoreboard for /cost.

Prices come from env vars, because that's nx's only config surface:
  NX_PRICE_<SOURCE>_INPUT   = $/1M input tokens  (uncached)
  NX_PRICE_<SOURCE>_OUTPUT  = $/1M output tokens
  NX_PRICE_<SOURCE>_CACHE   = $/1M cached input tokens (optional; default = input price)
Unpriced sources still show token totals — only the $ column is blank.
"""
from __future__ import annotations

import os
from typing import Any

from ._ansi import DIM, RESET, C
from ._metrics import _abbr


def _price_for(source: str, kind: str) -> float | None:
    """$/1M tokens for source+kind (input|output|cache), or None."""
    v = os.environ.get(f"NX_PRICE_{source.upper()}_{kind.upper()}", "")
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _source_cost(bucket: dict[str, int], source: str) -> float | None:
    """Total $ for one source's usage bucket, or None when unpriced."""
    in_p = _price_for(source, "input")
    out_p = _price_for(source, "output")
    if in_p is None or out_p is None:
        return None
    cache_p = _price_for(source, "cache")
    if cache_p is None:
        cache_p = in_p
    cost = 0.0
    cost += (bucket.get("input_tokens", 0) / 1_000_000) * in_p
    cost += (bucket.get("cache_read_tokens", 0) / 1_000_000) * cache_p
    cost += ((bucket.get("output_tokens", 0) + bucket.get("reasoning_tokens", 0)) / 1_000_000) * out_p
    return cost


def _fmt_cost(c: float | None) -> str:
    if c is None:
        return "—"
    if c < 0.01:
        return f"${c:.4f}"
    if c < 1.0:
        return f"${c:.3f}"
    return f"${c:.2f}"


def print_cost_report(metrics: Any) -> None:
    """/cost: session totals, then the per-source scoreboard ($ when priced)."""
    if metrics is None:
        print(f"{DIM}  no usage recorded yet this session{RESET}")
        return
    total_in = metrics.input_tokens + metrics.cache_read_tokens
    total_out = metrics.output_tokens + metrics.reasoning_tokens
    print(f"{DIM}  session usage: {metrics.api_calls} model calls · "
          f"in {_abbr(total_in)} (cache {_abbr(metrics.cache_read_tokens)}) · "
          f"out {_abbr(total_out)} (reasoning {_abbr(metrics.reasoning_tokens)}){RESET}")
    buckets = getattr(metrics, "by_source", None) or {}
    if not buckets:
        print(f"{DIM}  (no per-source data — nothing has run yet){RESET}")
        return
    print(f"{DIM}  per source:{RESET}")
    header = f"{DIM}    {'source':<14} {'calls':>5} {'in':>8} {'out':>8} {'cost':>9}{RESET}"
    print(header)
    rows = sorted(buckets.items(), key=lambda kv: -(kv[1].get("input_tokens", 0) + kv[1].get("output_tokens", 0)))
    any_price = False
    any_unpriced = False
    for name, b in rows:
        cost = _source_cost(b, name)
        if cost is not None:
            any_price = True
        else:
            any_unpriced = True
        tin = b.get("input_tokens", 0) + b.get("cache_read_tokens", 0)
        tout = b.get("output_tokens", 0) + b.get("reasoning_tokens", 0)
        model_note = ""
        if st_active_model(name):
            model_note = f"  {DIM}({st_active_model(name)}){RESET}"
        print(f"{DIM}    {name:<14} {b.get('api_calls', 0):>5} "
              f"{_abbr(tin):>8} {_abbr(tout):>8} {C.STATUS_BAR}{_fmt_cost(cost)}{RESET}{DIM}{model_note}{RESET}")
    if not any_price:
        print(f"{DIM}    (no $ — set NX_PRICE_<SOURCE>_INPUT/_OUTPUT per 1M tokens){RESET}")
    elif any_unpriced:
        print(f"{DIM}    (— = unpriced; set NX_PRICE_<SOURCE>_INPUT/_OUTPUT per 1M tokens){RESET}")


def st_active_model(name: str) -> str | None:
    """The model most recently used on `source` (from the live registry when
    it's the active one, else None — a session JSON record is future work)."""
    from . import _state as st
    if st.ACTIVE is not None and st.ACTIVE.name == name:
        return st.MODEL
    src = (st.SOURCES or {}).get(name)
    return getattr(src, "model", None) if src else None


def price_doc_line() -> str:
    """The README-facing summary of the price env vars."""
    return ("NX_PRICE_<SOURCE>_INPUT / _OUTPUT / _CACHE — $ per 1M tokens, "
            "feeds /cost's $ column; unset sources show token totals only.")
