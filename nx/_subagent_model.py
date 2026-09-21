"""Per-subagent model routing (0.86.0).

spawn_agent() sub-agents inherit the parent's active source and model. That is
the right default — but a cheap/fast model (glm-5.3-flash, qwen3.8-flash-next)
often answers broad read-only sweeps just as well as the expensive one driving
the parent, and never sees the conversation history, so a smaller context
window is not the constraint it is for the main loop.

NX_SUBAGENT_SOURCE / NX_SUBAGENT_MODEL route the sub-agent's model calls to a
different endpoint, exactly the way NX_COMPRESS_SOURCE / NX_COMPRESS_MODEL
route the context-fold call (nx/_compress._compress_call_target):

- ``NX_SUBAGENT_MODEL`` alone: same client, different model id.
- ``NX_SUBAGENT_SOURCE`` alone: that source's client and default model.
- Both: that source's client + the named model.
- Neither (default): inherit the parent, as before.

The target is used as-is — no capability probes, no ``switch_source()``. An
unknown source name falls back to the parent's active source with a one-time
warning, so a typo degrades to the old behavior instead of killing agents.

Threading: the resolved target is attached to the sub-agent's thread-local
``_SubagentContext`` and consumed by ``model_turn()`` -> ``open_stream()``.
No process-global is swapped, so concurrent agents (sync fan-out and async
workers) can each ride a different model without racing the parent's state —
the same reason the restricted dispatch table is TLS and not a global.
"""

from __future__ import annotations

import os
from typing import Any

from . import _state as st


def _subagent_model() -> str:
    """NX_SUBAGENT_MODEL, via the top-level module attribute first (tests/REPL
    can flip it live; same pattern as NX_SUBAGENT_MAX_TURNS)."""
    import nx as _m
    try:
        return str(_m.NX_SUBAGENT_MODEL or "").strip()
    except AttributeError:
        return os.environ.get("NX_SUBAGENT_MODEL", "").strip()


def _subagent_source() -> str:
    import nx as _m
    try:
        return str(_m.NX_SUBAGENT_SOURCE or "").strip()
    except AttributeError:
        return os.environ.get("NX_SUBAGENT_SOURCE", "").strip()


def subagent_call_target() -> tuple[Any, str, dict[str, Any], Any] | None:
    """(client, model, extra_request_kwargs, source) a sub-agent loop should
    call, or None to inherit the parent's active source/model (the default).

    Resolved lazily per agent start, so a mid-session /source switch or a
    REPL flip of nx.NX_SUBAGENT_MODEL is picked up by the next agent."""
    model = _subagent_model()
    name = _subagent_source()
    if not model and not name:
        return None
    if name:
        src = st.SOURCES.get(name)
        if src is None or getattr(src, "client", None) is None:
            if not st._SUBAGENT_SOURCE_WARNED:
                st._SUBAGENT_SOURCE_WARNED = True
                from ._ansi import C, RESET
                print(f"{C.WARNING}  ↳ NX_SUBAGENT_SOURCE={name!r} unusable — "
                      f"sub-agents run on {getattr(st.ACTIVE, 'name', st.MODEL)}{RESET}")
            return None
        use_model = model or getattr(src, "model", None)
        if not use_model and hasattr(src, "resolve_model"):
            try:
                use_model = src.resolve_model()
            except Exception:  # noqa: BLE001 — a probe failure must not kill the agent
                use_model = None
        client = src.client
        pin_src = src if hasattr(src, "_request_shapes") else None
    else:
        # model id alone: ride the parent's active source (client, extra body,
        # capability-ladder memory) under the new model name.
        src = st.ACTIVE
        if src is None or st.client is None:
            client, pin_src = st.client, None
        else:
            client, pin_src = st.client, (src if hasattr(src, "_request_shapes") else None)
        use_model = model
    if not client or not use_model:
        return None
    extra = src.extra_request_kwargs() if src is not None and hasattr(src, "extra_request_kwargs") else {}
    # The 4th element is the Source (None for a duck-typed stand-in): the
    # capability ladder pins request shapes per (Source, model) in
    # _manifest.py, so the sub-agent's endpoint keeps its own learned rung
    # instead of inheriting the parent's.
    return client, use_model, extra, pin_src


def install(context: Any) -> None:
    """Resolve the routing knobs once per agent start and pin the target on
    the thread's sub-agent context (or clear it — None — when unset/unusable)."""
    context.model_target = subagent_call_target()
