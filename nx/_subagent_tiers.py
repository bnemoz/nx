"""Sub-agent tool tiers: the read-only toolset, the write tier (NX_SUBAGENT_
WRITE=1), and the per-thread restricted dispatch tables run_tool() resolves
through.

Split out of `_subagent.py` (which keeps the spawn machinery). A sub-agent is
run through the same `_run_model_turn_loop` machinery as the parent but with a
RESTRICTED dispatch table: read-only by default; the write tier binds the four
file-mutation tools to confirm-through wrappers that tear down the sub-agent
context around the parent's real body, so every write crosses the same
risk-classified Y/n gate the parent shows."""
from __future__ import annotations

import threading
from typing import Any

from . import _state as st
from ._env import _env_int
from ._subagent_tls import _SUBAGENT_TLS, _subagent_ctx  # noqa: F401 (re-exports)

# Tools the sub-agent may call: the read-only dispatch entries (derived from
# the same tuple run_tool() result-caches, plus list_dir) + run_bash, which is
# gated to the read-only allowlist by _subagent_bash(). Never mutating tools,
# never wait_background.
_READ_ONLY_TOOL_NAMES = (
    "read_file", "list_dir", "search_files", "git_status", "git_diff", "git_log",
    "memory_search", "run_bash",
)

# Tools offered only when the write tier is on (NX_SUBAGENT_WRITE=1): the
# schemas come from the parent TOOLS list, but inside a tier-enabled sub-loop
# they resolve to the confirm-through wrappers in _write_tier_dispatch().
# delete_file is deliberately excluded (removal, not creation — a delete is
# one edit away from a recreate, but its undo snapshot belongs to the parent);
# so are apply_patch (one call already covers multi-file restructures),
# memory_write, and run_bash (the read-only allowlist tier covers inspection).
# run_tests IS included (0.49.0): a write-tier agent that can edit but not
# verify can only guess. It runs through the same user-gated _confirm() the
# parent's run_tests hits (the wrapper tears down the sub-agent context), so
# it is exactly as approved as a parent run — just tagged (sub-agent). It
# stays refused for READ-ONLY agents: their table never contains it.
# git_commit IS included (0.51.0): an agent that edits but cannot land its
# work strands its changes. Same gate: the parent's _confirm shows the staged
# diff (and the secret scan runs before it) exactly as a parent call would.
_WRITE_TIER_TOOL_NAMES = ("write_file", "edit_file", "multi_edit", "sed_edit",
                          "run_tests", "git_commit")

# Per-agent sub-loop context. A global flag/table can't express two agents
# running at once (the second's finally would clear the first's table), so
# spawn_agent() installs its restricted table on a threading.local slot and
# run_tool() resolves tool calls through it. Falls back to the plain globals
# for code (tests, REPL) that sets them directly.
_SUBAGENT_TLS = threading.local()


class _SubagentContext:
    """One running sub-agent's restricted-dispatch state (thread-local)."""

    __slots__ = ("table", "active", "write", "quiet", "model_target")

    def __init__(self, table: dict[str, Any], write: bool = False,
                 quiet: bool = False) -> None:
        self.table = table
        self.active = True
        self.write = write  # write tier (NX_SUBAGENT_WRITE=1): write tools resolve
        self.quiet = quiet  # no terminal writes (async worker threads)
        # (client, model, extra_request_kwargs) this agent's model calls use,
        # or None to inherit the parent's active source/model. Pinned by
        # _subagent_model.install() at agent start (NX_SUBAGENT_SOURCE/_MODEL).
        self.model_target = None


def _subagent_ctx() -> "_SubagentContext | None":
    return getattr(_SUBAGENT_TLS, "ctx", None)


def _subagent_table() -> dict[str, Any] | None:
    """The restricted dispatch table for THIS thread, or None outside a
    sub-agent loop. TLS wins over the global flag (concurrent agents)."""
    ctx = _subagent_ctx()
    if ctx is not None:
        return ctx.table
    return st._SUBAGENT_DISPATCH


def _subagent_active() -> bool:
    if _subagent_ctx() is not None:
        return True
    # global fallback is for serial callers/tests that set it directly; under
    # concurrent agents it's racy (a sibling's True is visible here), so TLS
    # is the authoritative signal and the global is never read on agent
    # threads' entry paths.
    return bool(st._SUBAGENT_ACTIVE)


def _write_tier_enabled() -> bool:
    """NX_SUBAGENT_WRITE=1 opts into write-capable sub-agents. Read from the
    top-level module attribute first (resolved from the env at import, like
    NX_SUBAGENTS) so tests/REPL code can flip it live; the env is the
    fallback when the attribute isn't there (partially-initialized module)."""
    import nx as _m
    try:
        return bool(int(_m.NX_SUBAGENT_WRITE))
    except (AttributeError, TypeError, ValueError):
        return bool(_env_int("NX_SUBAGENT_WRITE", 0))


def _subagent_tools() -> list[dict[str, Any]]:
    """Tool schemas for the sub-loop: read-only entries, plus the write tier
    when NX_SUBAGENT_WRITE=1 (only when a write-capable sub-agent is actually
    being spawned — see spawn_agent)."""
    import nx as _m
    wanted = set(_READ_ONLY_TOOL_NAMES)
    if _write_tier_enabled():
        wanted.update(_WRITE_TIER_TOOL_NAMES)
    return [t for t in _m.TOOLS if t["function"]["name"] in wanted]


def _read_only_dispatch() -> dict[str, Any]:
    from ._subagent_bash import _subagent_bash
    from ._dispatch import DISPATCH
    table = {name: DISPATCH[name] for name in _READ_ONLY_TOOL_NAMES if name in DISPATCH}
    table["run_bash"] = _subagent_bash  # ty: ignore[invalid-assignment]  # allowlist gate in front of run_bash
    return table


def _write_tier_dispatch() -> dict[str, Any]:
    """Dispatch table for a write-tier sub-agent: the read-only entries (bash
    still allowlist-gated — git diff/grep for verifying an edit) plus the four
    file-mutation tools bound to _write_tier_wrapper, which runs the parent's
    real tool body OUTSIDE the sub-agent context. The body's own _confirm()
    then hits the normal gate — risk classification, the Y/n prompt with the
    same colored diff preview the parent's writes show — so a sub-agent edit
    is exactly as approved as a parent edit, just tagged (sub-agent) in the
    tool box."""
    table = _read_only_dispatch()
    for name in _WRITE_TIER_TOOL_NAMES:
        table[name] = _write_tier_wrapper(name)
    return table


def _write_tier_wrapper(name: str):
    """Build the confirm-through binding for one write-tier tool.

    Closures over the parent's real body (resolved at spawn time) so a test
    or hook that swaps nx.write_file still governs what actually runs. The
    wrapper tears down this thread's sub-agent context around the body so
    _confirm() sees the parent's real approval state (and a nested
    spawn_agent from inside a hook would be refused, not looped)."""
    from ._dispatch import DISPATCH

    def _run(**args: Any) -> str:
        body = DISPATCH.get(name)
        if body is None:
            return f"ERROR: {name} is unavailable"
        ctx = _subagent_ctx()
        if ctx is None:  # defensive: a wrapper escaped its sub-agent context
            return f"ERROR: {name} write-tier wrapper used outside a sub-agent"
        prev_ctx = _SUBAGENT_TLS.ctx
        _SUBAGENT_TLS.ctx = None  # body + _confirm() run as the parent
        try:
            return body(**args)
        finally:
            _SUBAGENT_TLS.ctx = prev_ctx

    _run.__name__ = f"_subagent_write_{name}"
    return _run
