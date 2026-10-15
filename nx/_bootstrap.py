"""Deferred source wiring: pick and switch to the starting source.

Before 0.159.0 this ran at `import nx` time — which made every invocation
(`nx --help` included) pay switch_source()'s network calls, and against an
unreachable endpoint blocked for ~10-15s (resolve_model()'s /v1/models
round-trip plus the capability/thinking probes). That in turn forced
__init__.py to argv-sniff for -h/--help/--version and the offline verbs to
avoid the tax — a heuristic that silently broke for any other invocation
that never touches the model.

Now: `_cli.main()` calls `ensure_active_source()` right after its
--help/--version short-circuit, so the wiring (st.ACTIVE / st.client /
st.MODEL, model resolution, startup probes) runs exactly once, on the paths
that actually need a model connection. Paths that import nx directly without
going through main() (the test suite's reload-based tests, library callers)
get the wiring lazily via nx.__getattr__ → this function, preserving the old
"import nx leaves ACTIVE set" contract at first attribute access.
"""

from __future__ import annotations

import os
import sys
import threading

from . import _state as st

_done = False
_lock = threading.Lock()


def ensure_active_source() -> bool:
    """Wire the active source exactly once per process (idempotent).

    Resolves --source argv flag → NX_ACTIVE env → first discovered source,
    then switch_source() to populate st.ACTIVE/st.client/st.MODEL (with the
    model-id/capability/thinking network probes that switch_source() arms —
    still gated per-probe by NX_SKIP_STARTUP_PROBES).

    Safe to call multiple times (the CLI path and the lazy nx.__getattr__
    path can both reach it); subsequent calls are no-ops. A direct
    switch_source() call (e.g. /source in the REPL) still works normally —
    this only decides the *starting* source.

    Re-armable: tests that `importlib.reload(nx)` rebuild st.SOURCES from new
    env vars and expect the wiring to follow the fresh registry. On reload,
    __init__.py calls rearm() (below); a subsequent ensure_active_source()
    then re-resolves against the new source set."""
    global _done
    with _lock:
        # Nothing to do if a source is already wired — most importantly, an
        # explicit switch_source() (e.g. the test suite's, or /source in the
        # REPL) must NEVER be overridden by the startup default picked here.
        if _done or st.ACTIVE is not None:
            return st.ACTIVE is not None
        _done = True
    from ._sources import switch_source  # deferred: _sources imports heavy deps

    start = None
    for _i, _arg in enumerate(sys.argv):
        if _arg == "--source" and _i + 1 < len(sys.argv):
            start = sys.argv[_i + 1]
            break
    start = start or os.environ.get("NX_ACTIVE") or \
        (st.SOURCE_ORDER[0] if st.SOURCE_ORDER else None)
    if not start or start not in st.SOURCES:
        start = st.SOURCE_ORDER[0] if st.SOURCE_ORDER else None
    if not start:
        return False
    return switch_source(start)


def rearm() -> None:
    """Forget that ensure_active_source() ran (nx/__init__.py calls this on
    reload, after st.SOURCES is rebuilt). The next ensure_active_source()
    re-resolves --source/NX_ACTIVE/first against the fresh registry."""
    global _done
    with _lock:
        _done = False
