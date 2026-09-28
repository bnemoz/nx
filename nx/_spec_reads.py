"""Speculative read-only tool dispatch during model streaming.

A tool call's arguments stream in token by token, then the stream closes and
dispatch starts from zero — for a `read_file` of a big file that's the model's
decode time PLUS the tool's I/O time, serialized. But once a call's accumulated
arguments are BALANCED JSON, the call is already fully determined: for the
cacheable read-only tools (`_CACHEABLE_READ_TOOLS`: no approval gate, no state
mutation, no ordering requirement — the same set `run_tool()` result-caches)
there is nothing to lose by running it right now, on a background pool, while
the model keeps generating.

The speculative run goes through `run_tool()` itself — exactly like the
post-stream mixed-turn prefetch (`_tool_prefetch._prefetch_cacheable_tools`),
so it crosses the same gates (plan gate, pre-hook veto, both hooks) and, on
success, `run_tool` stores the result in `st._TOOL_RESULT_CACHE` itself. When
the stream closes, the real dispatch finds the entry and is a plain cache hit,
with the same hits accounting a prefetch-induced hit has today (the insert
carries `hits=0`; only `cache_touch` on a real dispatch counts a hit, and the
repeat-call stub at `hits>=2` therefore fires on exactly the same model-level
repeat it fires on behind a prefetch).

Safety rules (all enforced in `maybe_speculate`):
  * only `_CACHEABLE_READ_TOOLS` — the approval-free read-only set; a name
    outside it is never dispatched, however parseable its args are;
  * args must `json.loads` to a `dict` — anything else (truncated JSON,
    a list, a scalar) is skipped, and the real dispatch path applies every
    argument-level guard anyway;
  * never inside a sub-agent context (`_subagent_ctx() is not None`): a
    sub-agent is a separate process of thought and warming the PARENT's cache
    from its stream is pointless (and its restricted table could differ);
  * the kill switch `NX_SPECULATIVE_READS=0` disables everything, read
    lazily from the flag this module's names mirror in `_compress`.

A speculative run that fails, vetoes, or races the stream close is simply a
no-op for the cache: the real dispatch re-runs the tool from scratch and
surfaces the error itself. Speculation may waste one read; it may never change
one.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import _state as st

# In-flight speculative keys: (name, args-json). A key enters before the job is
# submitted and leaves when the job finishes, so the same balanced args
# arriving in five more deltas do not queue five runs. Mutated only from the
# streaming thread on the check side (add) and pool threads on the release
# side (discard); set add/discard are atomic under the GIL, and the worst
# possible race (a discard between check and add) costs one duplicate read,
# never a correctness problem.
_inflight: set = set()

# The pool is created lazily on first use and lives in a one-entry dict so
# tests (and teardown) can shut it down via `shutdown_pool()` without racing a
# module-attribute rebinding. max_workers=2 on purpose: speculation is a
# nicety, not a fan-out — two concurrent warm reads already cover the common
# 'read 2 files' turn, and a deeper pool would let speculative work queue
# behind itself on a multi-call turn the post-stream fan-out is about to run
# concurrently anyway.
_pool_holder: dict = {}
_pool_lock = None  # lazy threading.Lock (import kept cheap)


def _enabled() -> bool:
    """Kill switch, read lazily: tests reload `_compress` and patch the name
    (same pattern as `_request_opts._parallel_tool_calls_opts` reading
    NX_PARALLEL_TOOL_CALLS). Absent flag reads as OFF."""
    from . import _compress as _c
    return bool(getattr(_c, "NX_SPECULATIVE_READS", False))


def _submit(job) -> None:
    """Run `job` on the speculative pool. Seam for tests: monkeypatch this to
    execute inline for deterministic assertions."""
    pool = _pool_holder.get("pool")
    if pool is None:
        import threading
        global _pool_lock
        if _pool_lock is None:
            _pool_lock = threading.Lock()
        with _pool_lock:
            pool = _pool_holder.get("pool")
            if pool is None:
                pool = ThreadPoolExecutor(max_workers=2,
                                          thread_name_prefix="nx-spec-read")
                _pool_holder["pool"] = pool
    pool.submit(job)


def shutdown_pool() -> None:
    """Tear the pool down (test cleanup / process hygiene). Idempotent; jobs
    already running are not cancelled — a warm read finishing after shutdown
    is harmless (it only ever writes a cache entry)."""
    pool = _pool_holder.pop("pool", None)
    if pool is not None:
        pool.shutdown(wait=False)


def _loads_balanced(raw: str) -> Any:
    """json.loads on the accumulated args string, or None. 'Balanced' IS the
    check: args arrive as a growing prefix, so a successful parse means the
    model finished the object. Cheap by construction — only called for
    cacheable read-only tools, whose argument payloads are paths/patterns,
    not file contents."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def maybe_speculate(name: str, args: Any) -> None:
    """Speculatively dispatch a cacheable read-only call whose args just became
    complete. `args` is the raw accumulated JSON string (native deltas) or an
    already-parsed dict (a completed text-protocol tag). Every safety rule of
    this module lives here; anything questionable returns without dispatching.
    Never raises: this runs inline in the SSE loop."""
    try:
        if not name:
            return
        if not _enabled():
            return
        from ._tool_prefetch import _CACHEABLE_READ_TOOLS
        if name not in _CACHEABLE_READ_TOOLS:
            return
        # A sub-agent's stream must not warm the parent's cache: different
        # table, different purpose, and its parent never dispatches this call.
        from ._subagent import _subagent_ctx
        if _subagent_ctx() is not None:
            return
        if isinstance(args, str):
            args = _loads_balanced(args)
        elif not isinstance(args, dict):
            args = None
        if not isinstance(args, dict):
            return  # unbalanced JSON / non-object payload: never speculate
        from ._tool_prefetch import _tool_cache_key
        key = _tool_cache_key(name, args)
        if key in st._TOOL_RESULT_CACHE or key in _inflight:
            return
        _inflight.add(key)
        try:
            _submit(_make_job(key, name, args))
        except Exception:
            _inflight.discard(key)  # no pool (shutdown race): not an event
    except Exception:
        return  # the stream loop must not see speculation fail


def _make_job(key, name: str, args: dict):
    def _job() -> None:
        # A pool worker never owns the terminal: force the quiet slot exactly
        # as _prefetch_cacheable_tools' workers do (thread-local quiet is not
        # inherited). Everything the speculative run would print, the real
        # post-stream dispatch prints anyway (cache-hit box included).
        from ._subagent import _ASYNC_QUIET_SLOTS
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            from ._dispatch import run_tool
            # run_tool() applies the plan gate, the pre-hook veto and both
            # hooks, and stores the cache entry itself (hits=0) on success —
            # mirror of the prefetch's insertion path by construction.
            run_tool(name, args)
        except Exception:
            pass  # the real dispatch re-runs and surfaces any error itself
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False
            _inflight.discard(key)
    return _job
