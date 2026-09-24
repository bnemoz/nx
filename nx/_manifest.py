"""One cached capability manifest per `(source, model)` — PERF.md 0.59.0.

Three findings in PERF.md share the same fix, so they share this module:

- **PF-11** — `_compress.open_stream()` used to re-derive the *accepted request
  shape* by failing: up to 3 fresh POSTs of the full prompt (tools + tool_choice
  + extra_body, then minus tool_choice, then minus extra_body) against an
  endpoint that had already rejected the first one. The ladder is correct the
  first time; paying it every turn is not. Once a rung succeeds we **pin** it
  for that `(source, model)`, so the next turn opens with the reduced shape and
  makes exactly one request.

- **PF-10** — nx displays `cached:` and documents prefix-cache discipline, but
  never sent a stable session identity, so a vLLM / SGLang / llama.cpp router
  could not pin a session to a replica or KV block: every turn was free to
  cold-prefill on a different worker. `routing_identity()` supplies that stable
  per-session value; it is **never** derived from anything that varies within a
  session (a moving key would itself be a cache-buster). Opt out per source with
  `NX_SOURCE_<NAME>_CACHE_ROUTING=0`, or globally with `NX_CACHE_ROUTING=0`.

- **PF-21** — `Source.resolve_model()` is memoized on the Source (see
  `_sources.py`); the manifest is the natural home for the shape half.

Nothing here performs I/O on the request path: `shape_for()` is a dict read and
`routing_identity()` reads the already-built session recorder.
"""
from __future__ import annotations

import os
from typing import Any

from . import _state as st

# The three request shapes `open_stream()` can send, most to least complete.
# Each is (tools, extra_body): a rung drops `tools` (and `tool_choice`, which is
# meaningless without it) and then `extra_body` (grammar / source body /
# recovery samplers) after a 400/422 that named those fields. This mirrors
# exactly what the pre-0.59.0 ladder sent, so pinning a rung is behavior-
# preserving — it only stops re-deriving the shape from scratch every turn.
SHAPE_FULL = (True, True)
SHAPE_NO_TOOLS = (False, True)
SHAPE_NO_TOOLS_NO_EXTRA = (False, False)
LADDER: tuple[tuple[bool, bool], ...] = (
    SHAPE_FULL, SHAPE_NO_TOOLS, SHAPE_NO_TOOLS_NO_EXTRA)

# `_log_event`'s "_fallback" tag per rung, matching the pre-0.59.0 ladder so
# log consumers see the same strings.
RUNG_TAG: tuple[str | None, ...] = (None, "no-tools", "no-tools-no-extra")


def shape_index(shape: tuple[bool, bool]) -> int:
    """Position of `shape` in LADDER (0 = full). Unknown shapes pin at 0."""
    try:
        return LADDER.index(shape)
    except ValueError:
        return 0


def _shapes_of(src: Any) -> dict[str, tuple[bool, bool]] | None:
    """The per-Source `{model: shape}` dict, created lazily, or None when
    `src` cannot carry one (a duck-typed source in a test). Kept on the Source
    instance (not a module global) so a `/source` switch back to a source keeps
    its answer, while conftest's per-test module-state rewind stays untouched.

    Returns None rather than an empty dict sentinel: `{}` is a perfectly
    normal value here (a Source that has learned nothing), and callers must
    not confuse "no store" with "nothing learned".
    """
    if src is None:
        return None
    shapes = getattr(src, "_request_shapes", None)
    if shapes is None:
        shapes = {}
        try:
            src._request_shapes = shapes
        except Exception:  # a duck-typed source — no pinning, no crash
            return None
    return shapes


def pinned_index(src: Any, model: str | None) -> int:
    """The ladder rung already known to work for `(src, model)`, or 0 (full)
    when nothing has been learned yet. A pure dict read — never probes."""
    if not model:
        return 0
    shapes = _shapes_of(src)
    if not shapes:
        return 0
    return shape_index(shapes.get(model, SHAPE_FULL))


def pin(src: Any, model: str | None, index: int) -> None:
    """Record that rung `index` is what this `(src, model)` accepts. Idempotent,
    and only ever *lowers* the shape (a higher index = fewer fields): a later
    success on a more complete rung must not un-learn a rejection the server
    already gave us."""
    if src is None or not model or index <= 0:
        return
    index = min(index, len(LADDER) - 1)
    shapes = _shapes_of(src)
    if shapes is None:
        return
    cur = shape_index(shapes.get(model, SHAPE_FULL))
    if index > cur:
        shapes[model] = LADDER[index]


def reset(src: Any, model: str | None = None) -> None:
    """Forget the pinned shape for `(src, model)` (or every model of `src`).
    Used when a `/source <name> <model>` override points the same Source at a
    different model — the old model's rejection says nothing about the new one."""
    if src is None:
        return
    shapes = _shapes_of(src)
    if shapes is None:
        return
    if model:
        shapes.pop(model, None)
    else:
        shapes.clear()


# --- PF-10: stable cache-routing identity ------------------------------------

def _flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def cache_routing_enabled(src: Any) -> bool:
    """Whether to attach the routing hint for `src`. Per-source opt-out wins
    (`NX_SOURCE_<NAME>_CACHE_ROUTING`), then the global `NX_CACHE_ROUTING`,
    then on."""
    name = getattr(src, "name", None)
    if name:
        per = os.environ.get(f"NX_SOURCE_{name.upper()}_CACHE_ROUTING")
        if per is not None and per.strip():
            return _flag(f"NX_SOURCE_{name.upper()}_CACHE_ROUTING")
    return _flag("NX_CACHE_ROUTING")


def routing_identity() -> str | None:
    """A stable, session-scoped value for the OpenAI `user` field (and the
    `prompt_cache_key` body field, where the backend reads one).

    Derived from the session id that already exists on the metrics recorder, so
    it is constant for the whole session and different across sessions — exactly
    what a router needs to pin a session to a replica/KV block. Returns None
    when there is no session (import-time probes, batch workers before their
    recorder is installed), in which case nothing is attached.

    Swarm P7: per-session keys spread a swarm's agents across replicas, so the
    SHARED prompt head (SYSTEM + prefix notes — byte-identical for every agent
    from the same project+model) is cold-prefilled N times instead of once.
    NX_CACHE_ROUTING_GROUP set to a non-empty value overrides the session id
    with that shared key (e.g. derived from project root + model), co-locating
    agents whose prefixes share bytes on one replica/KV block. Set it to "1"
    for an automatic project+model-derived group key.
    """
    group = os.environ.get("NX_CACHE_ROUTING_GROUP", "").strip()
    if group:
        if group == "1":
            root = getattr(st, "PROJECT_ROOT", None) or os.getcwd()
            model = getattr(st, "MODEL", None) or ""
            import hashlib
            return "nxgrp-" + hashlib.sha256(f"{root}|{model}".encode()).hexdigest()[:16]
        return f"nxgrp-{group}"
    m = st._active_metrics()
    sid = getattr(m, "session_id", None) if m is not None else None
    if not sid:
        return None
    return f"nx-{sid}"


def routing_kwargs(src: Any) -> dict[str, Any]:
    """The routing hint for `src`: `{"user": ...}` plus `prompt_cache_key` in
    extra_body where it belongs. Empty dict when disabled or no session exists.
    Pure reads — safe to call on every request."""
    if not cache_routing_enabled(src):
        return {}
    ident = routing_identity()
    if not ident:
        return {}
    return {"user": ident, "extra_body": {"prompt_cache_key": ident}}
