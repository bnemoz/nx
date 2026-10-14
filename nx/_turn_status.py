"""TURN_* status constants for the turn loop.

Split out of `_turn.py` (0.90.0 decomposition): these are the vocabulary
every turn-layer module (the loop, the tool-call handlers, the retry ladder)
speaks, but they had no home of their own — `_turn.py` defined them alongside
the loop, so even leaf modules had to import from the loop module. `_turn.py`
re-exports everything here, so the import surface (`nx._turn.TURN_DONE` etc.,
including test attribute access) is unchanged.
"""
from __future__ import annotations

TURN_DONE = "done"
TURN_TOOL = "tool"
TURN_STREAM_CUT = "stream_cut"
TURN_FORCE_FINAL = "force_final"
TURN_STALL = "stall"  # reasoning-only stall → nudge to act (full tools retained)
TURN_EMPTY = "empty"  # assistant emitted no content and no tool calls → silent stall
TURN_ESC = "esc"  # user pressed Esc at an approval prompt → drop to chat input
TURN_REDIRECT = "redirect"  # user typed+submitted a new instruction right after Esc
TURN_MALFORMED_CALL = "malformed_call"  # text-protocol tag(s) present but unparseable → one corrective retry
TURN_API_ERROR = "api_error"  # request hard-failed before any chunk (endpoint down, 4xx/5xx) — never retried
TURN_CONTEXT_OVERRUN = "context_overrun"  # 400: prompt exceeded the context window — recoverable via one compress+retry
TURN_LENGTH_CUT = "length_cut"  # finish_reason=length: the model's response budget ran out mid-stream — auto-continue it
TURN_TIMED_OUT = "timed_out"  # mid-stream timeout after the step-retry ladder was exhausted — a real failure, scored as one
# 0.154.6: a NON-timeout mid-stream cut (transport abort, SSE error frame,
# degenerated special-token/repetition collapse, reasoning-only stall) whose
# recovery ladder is exhausted with nothing usable produced. Distinct from
# TURN_STREAM_CUT, which is RETRYABLE (the loop re-opens the request on it) —
# reusing it here would spin forever. The loop maps this to summary status
# "error" exactly like TURN_TIMED_OUT: headless callers (nx run, nx eval,
# sub-agents) must never score a truncated, never-completed task as "done".
TURN_GAVE_UP = "gave_up"
TURN_TIMED_OUT_RETRY = "timed_out_retry"  # mid-stream timeout with ladder left — the loop backs off and re-opens the same request
TURN_STEER = "steer"  # 0.105.0: the user typed a steering note mid-turn — inject it and keep the turn alive
TRUNCATION_MARKER = "\n\n[Truncated by token limit before completion.]"
