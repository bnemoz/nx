"""Decision breadcrumbs: dim one-line `[harness] ...` prints at the moment a
harness decision happens (Phase 6 of docs/EXPLAINABLE_HARNESS_SPEC.md).

One tiny helper owns the gate logic so the instrumented modules (which have
no render context) don't each re-implement it: prints only when the setting
is on AND the caller is on the visible (non-quiet) path. Best-effort — a
breadcrumb must never break the decision it is reporting. No formatting cost
when off: note() takes a pre-formatted message, and the caller builds it only
after checking enabled() (or simply calls note(), which returns before any
work when the gate is closed)."""
from __future__ import annotations

import os

from ._ansi import DIM, RESET

# Default ON. Seeded from the env (NX_BREADCRUMBS=0/false/off disables) and
# flipped at runtime by /breadcrumbs (nx/_repl_settings_cmds.py), which also
# persists NX_BREADCRUMBS to ~/.env the same way /reasoning persists its flag.
ENABLED: bool = os.environ.get("NX_BREADCRUMBS", "1") not in ("0", "false", "off")


def enabled() -> bool:
    return ENABLED


def set_enabled(on: bool) -> None:
    global ENABLED
    ENABLED = bool(on)


def note(msg: str) -> None:
    """Dim inline `[harness] <msg>` line, or nothing when off / quiet."""
    if not ENABLED or _quiet():
        return
    try:
        print(f"{DIM}  [harness] {msg}{RESET}")
    except Exception:  # noqa: BLE001 — never break the decision being noted
        pass


def _quiet() -> bool:
    """True on a sub-agent/batch thread: never print from a quiet context."""
    try:
        from ._subagent import _subagent_quiet
        return bool(_subagent_quiet())
    except Exception:  # noqa: BLE001
        return False
