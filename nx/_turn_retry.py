"""Turn-loop retry ladder and the batch per-task wall-clock deadline.

Split out of `_turn.py` (which keeps the loop that consumes them). The HTTP
layer (_openai_client._send) backs off 429/5xx/broken connections before the
FIRST byte; a timeout after the stream has begun is structurally unreplayable
there (a partially-consumed stream can't be re-sent), so the turn loop owns
it: model_turn() returns TURN_TIMED_OUT_RETRY / TURN_TIMED_OUT and the loop
applies the same ladder at the step boundary."""
from __future__ import annotations

import threading
import time

from ._env import _env_int


def _step_retry_limit() -> int:
    from ._openai_client import _HTTP_RETRIES_DEFAULT
    return max(0, _env_int("NX_TURN_RETRIES", _HTTP_RETRIES_DEFAULT))


def _step_backoff_delay(attempt: int, retry_after: float | None) -> float:
    """Same curve as the HTTP layer's pre-first-byte backoff (Retry-After wins,
    else base * 2**attempt, same cap) so a stream timeout and a connection
    retry behave identically."""
    from ._openai_client import _backoff_delay
    return _backoff_delay(attempt, retry_after)


def _retry_log(attempt: int, error_class: str) -> None:
    """v1 usage metrics: one `retry` event per retry-ladder attempt (never
    per chunk). `error_class` names the turn-level failure the ladder is
    rescuing. Best-effort — never raises, never delays the ladder."""
    try:
        from ._logging import usage_retry
        usage_retry(attempt, error_class)
    except Exception:  # noqa: BLE001 — telemetry must never break a retry
        pass
    try:
        from ._breadcrumbs import note
        note(f"retrying after {error_class} (ladder step {attempt})")
    except Exception:  # noqa: BLE001 — a breadcrumb must never break a retry
        pass


def _plural_ies(n: int) -> str:
    return "y" if n == 1 else "ies"


# --- per-task wall-clock deadline (batch) -------------------------------------
# The batch harness arms this per worker thread just before its turn loop and
# clears it after. It must be a thread-local, not a module global -- with
# --parallel > 1 sibling workers would share and clobber each other's
# deadlines. The loop checks it at the on_step seam (every model-turn
# boundary) and aborts with a status-"error" summary rather than let one slow
# task hold a whole CI run open. A mid-tool-call cancel would be much messier
# and buys little: the deadline still fires at the next step boundary.
_TASK_DEADLINE_TLS = threading.local()


def _task_deadline_at(deadline):
    """Arm the calling worker's wall-clock deadline (None disarms it)."""
    _TASK_DEADLINE_TLS.deadline = deadline


def _task_deadline():
    """Return the armed deadline for this worker, or None if not armed."""
    return getattr(_TASK_DEADLINE_TLS, "deadline", None)


def _task_deadline_exceeded():
    d = getattr(_TASK_DEADLINE_TLS, "deadline", None)
    return d is not None and time.time() > d


def _task_timeout_summary(turns: int) -> dict:
    """The summary _run_model_turn_loop returns when a batch worker's
    wall-clock deadline lapses: status "error" so the task gets exit_code 4
    and the run stops scheduling (unless --continue-on-error), with
    "timed_out" so a harness can tell a timeout from a crash."""
    return {"status": "error",
            "error": (f"task timeout: exceeded the per-task wall-clock budget "
                      f"at turn {turns}"),
            "turns": turns,
            "timed_out": True}
