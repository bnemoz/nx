"""The spawn_agent tool: run a focused read-only sub-agent turn loop.

A sub-agent is a fresh `messages` list (scoped SYSTEM + the instruction) run
through the same `_run_model_turn_loop` machinery, but with a READ-ONLY toolset
(derived from _CACHEABLE_READ_TOOLS — the tools that never touch state) and
YOLO semantics inside the loop, so a read-only subtask never prompts and never
mutates anything. The sub-agent's final answer comes back as the tool result —
the parent's context stays clean of the exploration (reads, greps, dir walks)
the subagent did, which is the point: delegation without context bloat.

Deliberately NOT offered to the sub-agent: final_answer's forced-final mode
(we reuse it), wait_background (no bash at all), and every mutating tool
(write/edit/sed/patch/bash/memory_write). If the model needs to change
something, it must report back and let the parent do it — one actor per
capability keeps the approval story simple.
"""
from __future__ import annotations

import json
from typing import Any

from . import _state as st

SUBAGENT_SYSTEM = (
    "You are a focused read-only sub-agent answering ONE question about this "
    "codebase for a parent agent. Read files, search, and inspect; then reply "
    "in plain prose with exactly what was asked (key facts, paths, line "
    "numbers) and nothing else — no preamble, no plan narration, no "
    "next-steps. You cannot write, edit, or run commands."
)

SUBAGENT_TOOL = {"type": "function", "function": {
    "name": "spawn_agent",
    "description": "Delegate a focused READ-ONLY investigation (find where X is implemented, summarize module Y, locate all callers of Z) to a sub-agent that returns only its findings. Read/search tools only — it cannot write or run commands. Use for exploratory reads whose intermediate output would bloat this conversation; do the work yourself when the answer is one read_file away.",
    "parameters": {"type": "object", "properties": {
        "instruction": {"type": "string", "description": "The exact question or task for the sub-agent, with any needed paths/context."},
        "max_turns": {"type": "integer", "description": "Cap on sub-agent tool turns (default 12; 1-40)."},
    }, "required": ["instruction"]},
}}

# Tools the sub-agent may call: exactly the read-only dispatch entries. Derived
# from the same tuple run_tool() uses for the result cache, plus list_dir
# (also read-only) — never mutating tools, never run_bash.
_READ_ONLY_TOOL_NAMES = (
    "read_file", "list_dir", "search_files", "git_status", "git_diff", "git_log",
    "memory_search",
)

# Stashed on st by model_turn? No — set here directly when a subagent runs, so
# run_tool() inside the sub-loop routes calls to the restricted table.
_ACTIVE = False


def _subagent_tools() -> list[dict[str, Any]]:
    """Tool schemas for the sub-loop: restricted to read-only entries."""
    import nx as _m
    wanted = set(_READ_ONLY_TOOL_NAMES)
    return [t for t in _m.TOOLS if t["function"]["name"] in wanted]


def _read_only_dispatch() -> dict[str, Any]:
    from ._dispatch import DISPATCH
    return {name: DISPATCH[name] for name in _READ_ONLY_TOOL_NAMES if name in DISPATCH}


def spawn_agent(instruction: str, max_turns: int = 12, **_: Any) -> str:
    """Run a read-only sub-agent on `instruction`; return its final answer."""
    global _ACTIVE
    if not isinstance(instruction, str) or not instruction.strip():
        return "ERROR: instruction is empty"
    try:
        max_turns = max(1, min(40, int(max_turns)))
    except (TypeError, ValueError):
        max_turns = 12
    if st.client is None or st.MODEL is None:
        return "ERROR: no active model source"
    if st.YOLO is None:  # pragma: no cover — defensive; YOLO is always a bool
        return "ERROR: approval state unavailable"

    from ._turn import _run_model_turn_loop  # import-check only (circular-import canary)
    import nx as _m  # lazy: monkeypatch-friendly, same pattern as _repl.py
    messages = [{"role": "system", "content": SUBAGENT_SYSTEM},
                {"role": "user", "content": instruction.strip()}]
    # Force the sub-loop through the read-only dispatch table. run_tool() and
    # model_turn() consult _SUBAGENT_DISPATCH before DISPATCH (checked lazily
    # so tests can swap DISPATCH freely).
    st._SUBAGENT_DISPATCH = _read_only_dispatch()
    st._SUBAGENT_ACTIVE = True
    prev_yolo, prev_level = st.YOLO, st.APPROVE_LEVEL
    st.YOLO, st.APPROVE_LEVEL = True, "high"  # read-only tools never prompt anyway
    _ACTIVE = True
    try:
        summary = _m._run_model_turn_loop(
            messages,
            max_turns_override=max_turns,
            on_step=None,
        )
    except Exception as e:  # noqa: BLE001 — a failed subagent is a tool error, not a crash
        return f"ERROR: sub-agent failed: {type(e).__name__}: {e}"
    finally:
        st._SUBAGENT_ACTIVE = False
        st._SUBAGENT_DISPATCH = None
        st.YOLO, st.APPROVE_LEVEL = prev_yolo, prev_level
        _ACTIVE = False
    # Extract the last assistant text as the answer (the loop's final turn).
    answer = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and msg["content"].strip():
            answer = msg["content"].strip()
            break
    status = summary.get("status") if isinstance(summary, dict) else "done"
    turns = summary.get("turns", 0) if isinstance(summary, dict) else 0
    if not answer:
        return (f"[sub-agent produced no answer after {turns} turn(s); "
                f"status={status} — try a narrower instruction]")
    if status == "cap":
        return (f"[sub-agent hit its {max_turns}-turn cap; partial answer]\n{answer}")
    return answer
