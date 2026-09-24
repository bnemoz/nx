"""Read-only parallel fan-out and cache prefetch for multi-call turns.

Split out of `_turn.py` (which keeps the tool-call handlers that call these).
The common agent turn is 'read 3 files + grep' — strictly sequential dispatch
burns wall-clock on I/O that could overlap. Read-only cacheable tools have no
approval gate, no state mutation, and no output ordering requirement beyond
the Observation order, so a thread pool is safe here; mutating/prompting calls
keep the sequential path (approval UX depends on it)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import _state as st
from ._approval import _EscToChat

_CACHEABLE_READ_TOOLS = ("read_file", "list_dir", "search_files", "git_status",
                         "git_diff", "git_log", "memory_search")


def _tool_cache_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    import json
    return (name, json.dumps(args, sort_keys=True))


def _run_read_only_parallel(calls: list[tuple[str, dict[str, Any]]]) -> dict[int, str]:
    """Execute 2+ read-only, approval-free tool calls concurrently.

    Mutating/prompting calls keep the sequential path (approval UX depends on
    it). Results print in original order after all complete; observations map
    back by index so the model sees the same transcript shape as before.

    'Print in original order' is what this function did NOT do until 0.31.0:
    each worker called run_tool() non-quiet, so N threads interleaved their own
    tool boxes while N spinner threads cleared lines one another had just
    drawn. Workers now run quiet and record a ToolBlock instead (nx/_toolblocks);
    the dispatching thread — the only one owning the terminal — draws the whole
    turn as ONE group box, in call order, with a single spinner and one elapsed
    time. A quiet caller (async agent / batch task) records but never renders."""
    results: dict[int, str] = {}
    jobs = [(i, n, a) for i, (n, a) in enumerate(calls)]
    workers = min(8, len(jobs))
    blocks_by_index: dict[int, list] = {}
    from ._dispatch import run_tool

    # Pool threads inherit nothing from the dispatching thread's quiet state
    # (thread-local by design), which is why a quiet loop's fan-out had to
    # re-apply quiet in every worker. Since 0.31.0 the worker is ALWAYS quiet,
    # interactive parent included: "quiet" is how a thread says "I do not own
    # the terminal", which is exactly true of a pool worker — its output
    # reaches the user through the recorder instead of straight to stdout.
    from ._progress import ProgressCounter
    from ._subagent import _subagent_quiet, _ASYNC_QUIET_SLOTS
    from ._toolblocks import (attach_recorder, detach_recorder, group_spinner,
                              render_group)
    render = not _subagent_quiet()
    started = time.time()
    # One activity line for the whole group, in place of one spinner thread per
    # call erasing the lines its siblings had just printed. Since 0.35.0 the
    # line is live (elapsed + n/m landed): a group of 3 that had finished 2 of
    # its calls looked exactly like one that had finished none.
    counter = ProgressCounter(len(jobs))
    spinner = group_spinner(f"running {len(jobs)} tool calls", counter=counter) if render else None
    if spinner is not None:
        spinner.start()

    def _one(job: tuple[int, str, dict[str, Any]]) -> tuple[int, str]:
        i, n, a = job
        _ASYNC_QUIET_SLOTS.quiet = True
        rec = attach_recorder()
        try:
            return i, run_tool(n, a)
        except _EscToChat:
            raise  # read-only tools can't prompt, but propagate defensively
        except Exception as e:  # noqa: BLE001 — mirror run_tool's error surfacing
            return i, f"ERROR: {type(e).__name__}: {e}"
        finally:
            # Distinct keys per worker; the parent only reads this after the
            # pool has joined, so no lock is needed.
            blocks_by_index[i] = rec.blocks
            detach_recorder()
            _ASYNC_QUIET_SLOTS.quiet = False

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # as_completed, not ex.map: map hands back results in SUBMISSION
            # order, so with a slow first call the live counter would sit at
            # 0/3 while two calls had already landed. as_completed reports each
            # call as it actually finishes, which is the number being shown.
            # Results are keyed by index, so consumption order is free here.
            futs = [ex.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                i, res = fut.result()
                counter.bump()
                results[i] = res
    finally:
        if spinner is not None:
            spinner.stop()
        # Render even on the exception path (an Esc mid-fan-out, a tool that
        # raised past run_tool's own folding): the calls that DID happen have
        # already happened, and dropping their lines would misreport the turn.
        if render:
            render_group(
                [b for i in range(len(jobs)) for b in blocks_by_index.get(i, [])],
                time.time() - started, st.TOOL_OUTPUT_VISIBLE)
            # `/tooloutput last` replays st._LAST_TOOL_*, which each worker set
            # from whichever thread finished last. Re-point it at the group's
            # LAST call so "last" means the last thing the user just saw, not
            # an artifact of thread scheduling.
            _last = results.get(len(jobs) - 1)
            if isinstance(_last, str):
                st._LAST_TOOL_NAME = calls[-1][0]
                st._LAST_TOOL_OUTPUT = _last
    return results


def _all_read_only(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    """True when every call in the turn is a cacheable read-only tool (the
    same set run_tool() result-caches — no approval gate, no mutation)."""
    import nx as _m
    return bool(calls) and all(n in _m._CACHEABLE_READ_TOOLS for n, _ in calls)


def _write_capable_agent(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    """True when any call in the turn is a spawn_agent that will run with the
    write tier (NX_SUBAGENT_WRITE=1). Those agents' tool bodies hit the real
    _confirm() prompt, so they must stay on the sequential path — overlapping
    them would interleave two agents' approval prompts on one terminal."""
    if not any(n == "spawn_agent" for n, _ in calls):
        return False
    from ._subagent import _write_tier_enabled
    return _write_tier_enabled()


def _should_parallelize(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    """True when this turn's 2+ calls may run concurrently on the thread pool.

    Two shapes qualify:
      • the original all-cacheable-read turn ('read 3 files + grep'), and
      • a fan-out turn with spawn_agent calls — each agent is an independent
        messages list, its toolset is restricted, and nothing it runs prompts,
        so overlapping them is the whole point of delegating. Reads may ride
        along in the same turn (all read-only, no ordering requirements).

    Single calls and mutating/prompting tools stay sequential. A turn while
    already inside a sub-agent stays sequential too (no nested fan-out) -
    unless NX_SUBAGENT_MAX_NESTED=1 opts in (P5: two-level swarms multiply
    threads/streams; default off).

    WRITE-TIER fan-outs stay sequential when an approval prompt could appear:
    a write-capable sub-agent's bodies run the parent's real _confirm(), and
    two of them in flight would interleave approval prompts. Two conditions
    relax this (both must hold - P5):
      - the session is YOLO (st.YOLO) - nothing can prompt, so there is
        nothing to serialize; and
      - NX_SUBAGENT_WORKTREE=1 - each write agent edits in its own git
        worktree, so siblings cannot trample the shared tree.

    Since 0.31.0 this gate carries a second load-bearing meaning: a fan-out
    worker runs quiet (it must not own the terminal), and _confirm() auto-DENIES
    on a quiet thread. So anything eligible here must also be incapable of
    needing an approval prompt — true for _CACHEABLE_READ_TOOLS and read-only
    spawn_agent, and exactly why the write tier stays sequential."""
    if len(calls) < 2:
        return False
    import nx as _m
    from ._subagent import _subagent_active
    from ._env import _env_int
    if _subagent_active() and not _env_int("NX_SUBAGENT_MAX_NESTED", 0):
        return False  # a sub-agent never fans out further (default)
    spawn = sum(1 for n, _ in calls if n == "spawn_agent")
    if spawn:
        if _write_capable_agent(calls):
            from ._worktree import _worktree_enabled
            from . import _state as _st
            if not (_st.YOLO and _worktree_enabled()):
                return False  # prompts could interleave / trees would trample
        return all(n in _m._CACHEABLE_READ_TOOLS or n == "spawn_agent" for n, _ in calls)
    return _all_read_only(calls)


def _prefetch_cacheable_tools(ordered: list[dict], parsed_args: list[dict]) -> None:
    """Warm the read_file/list_dir cache concurrently for this turn's calls.

    Kept for the MIXED-call case: when a turn combines mutating and read-only
    calls, reads still run sequentially inside run_tool(), so warming the
    cache in parallel here keeps those hits instant. When EVERY call is a
    cacheable read, _run_read_only_parallel() handles dispatch directly and
    this prefetch is skipped (it would double-execute the reads).

    Each job dispatches through run_tool() — NOT the tool body directly — so a
    mixed turn's prefetched reads cross the same gates as every other call:
    the plan-mode gate, the NX_PRE_TOOL_HOOK veto (a blocked read is never
    opened, and its vetoed args never enter st._TOOL_RESULT_CACHE), and the
    NX_POST_TOOL_HOOK audit record. run_tool() populates the cache itself for
    cacheable reads, so this helper no longer stores entries by hand; the
    sequential run_tool() later in the turn is then a cache hit."""
    from ._dispatch import run_tool
    jobs = []
    seen_keys = set()
    for c, args in zip(ordered, parsed_args):
        if c["name"] not in _CACHEABLE_READ_TOOLS:
            continue
        cache_key = _tool_cache_key(c["name"], args)
        if cache_key in seen_keys or cache_key in st._TOOL_RESULT_CACHE:
            continue
        seen_keys.add(cache_key)
        jobs.append((c["name"], args))
    if len(jobs) < 2:
        return

    # The sequential run_tool() pass later in this turn renders each call's tool
    # box itself (cache hits included), so the prefetch stays invisible: force
    # the quiet slot in every pool worker, exactly as _run_read_only_parallel
    # does for a quiet loop, so a pool thread never prints a box / spinner /
    # pre-hook veto line onto the shared terminal (thread-local quiet is not
    # inherited — it must be re-applied here).
    from ._subagent import _ASYNC_QUIET_SLOTS

    def _run(job: tuple[str, dict[str, Any]]) -> None:
        name, args = job
        _ASYNC_QUIET_SLOTS.quiet = True
        try:
            # Applies the plan gate, the pre-hook veto, both hooks, and stores
            # the cache entry itself for cacheable reads. A veto/error simply
            # returns the message string; the sequential run_tool() call later
            # in the turn re-evaluates from the same state and surfaces it.
            run_tool(name, args)
        except _EscToChat:
            raise  # read-only tools can't prompt, but propagate defensively
        except Exception:
            return  # let the normal run_tool() call surface/report the error
        finally:
            _ASYNC_QUIET_SLOTS.quiet = False

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        list(ex.map(_run, jobs))
