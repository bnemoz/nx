"""The per-thread sub-agent context slot, shared by the tiers and async
machinery.

Small dedicated home for `_SUBAGENT_TLS` / `_subagent_ctx` so `_subagent_tiers`
and `_subagent` (async workers) can share the one slot without a cycle —
the tiers module must not import the async-heavy `_subagent` at module level.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._subagent_tiers import _SubagentContext

# Per-agent sub-loop context (see _subagent_tiers for why it's thread-local).
_SUBAGENT_TLS = threading.local()


def _subagent_ctx() -> "_SubagentContext | None":
    return getattr(_SUBAGENT_TLS, "ctx", None)
