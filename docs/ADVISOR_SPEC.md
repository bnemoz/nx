# Spec: Advisor strategy for sub-agents

Status: shipped (0.112.0); tier merged into implement per docs/SIMPLIFY_SPEC.md (0.158.0)
Version target: 0.111.0 (semver: new feature → minor bump)

## Summary

An `advisor` sub-agent tier in which the sub-agent acts as an **executor**:
it runs a task end-to-end, calling tools and iterating, but may consult the
**main model as an advisor** when it hits a decision it cannot reasonably
resolve alone. The advisor returns guidance only — a plan, a correction, or a
stop signal — never tool calls, never user-facing output. The executor
resumes with the guidance.

The executor maintains an `implementation-notes.md` in its worktree with the
plan and a `## Deviations` section for any conservative fallback or plan
drift; the notes path is included in the final report for audit.

## Background (verified, file:line)

- Sub-agent turn loop: `spawn_agent()` `nx/_subagent.py:1031` →
  `_run_sync_agent_body()` `:846`; local `messages` list, final answer from
  last assistant message. Never touches the parent transcript.
- Tier dispatch tables: `_read_only_dispatch()` / `_write_tier_dispatch()`
  `nx/_subagent_tiers.py:136,144`, installed per-thread via
  `_SubagentContext` + `_SUBAGENT_TLS` (`_subagent_tiers.py:63-67`,
  `_subagent_tls.py:20`). Tier normalization: `_normalize_tier()` `:178-192`.
  Write/implement differ only in system prompt (`_subagent.py:876-877`,
  prompts at `nx/_prompts.py:77,91,108`).
- There is currently **no** channel from a running sub-agent to the main
  model. Outbound = final answer / report file / `[Runtime note]`
  (`_stage_subagent_answer()` `:588`). Human → agent steering queues exist
  (`nx/_surfaces.py:39-51`, drained at turn boundaries `nx/_turn.py:924-928`)
  but are fire-and-forget, not a round-trip.
- Nested `spawn_agent` is refused (`_subagent.py:864-866`).

## Design

### The consult round-trip

New tool `consult_advisor`, available **only** in the advisor tier:

```
consult_advisor(question, options, context_digest)
  question       what the executor cannot decide
  options        2-3 candidate paths with tradeoffs
  context_digest instruction, current plan, turns spent, relevant findings
```

Handler semantics (recommended: side-call, not parent-queue):

1. Handler makes a **single-shot, no-tools model call to the parent's
   model** with `ADVISOR_SYSTEM` (new constant in `nx/_prompts.py`):
   "You are the advisor for an executing sub-agent. Return JSON
   `{decision, guidance, stop}`. `decision`: which option or a correction.
   `guidance`: concise reasoning. `stop: true` only if the executor should
   halt and report. Never call tools; this reply is never shown to the user."
2. The reply is returned to the executor as the tool result.
3. Rationale for side-call over a parent-session queue: a queue only drains
   when the parent's turn loop runs (user prompt / autowake), which starves
   async sub-agents; a side-call works for both sync and async agents and
   trivially guarantees the advisor never touches the parent transcript.

Fallback (rejected): reverse steering queues via `nx/_surfaces.py`. Keep in
mind if a future design wants *human* escalation: the consult handler could
optionally enqueue to the parent's steering queue when the side-call fails,
so the question surfaces as a `[Runtime note]` instead of being lost.

### Guardrails

- **Consult budget**: default 3 per sub-agent (env override
  `NX_SUBAGENT_CONSULT_BUDGET`); past the budget the tool refuses
  breaker-style (reuse `nx/_breaker.py` refusal tone) and the executor must
  proceed conservatively.
- **Timeout**: no advisor reply within `NX_SUBAGENT_CONSULT_TIMEOUT`
  (default 60s) → executor picks the **conservative option**, logs it under
  `## Deviations`, continues.
- **Malformed advisor reply** (unparseable JSON): one retry with a
  format-repair nudge, then conservative fallback as above.
- **Routing**: advisor side-call uses the **parent's model**, not the
  `NX_SUBAGENT_MODEL_POOL` — the advisor is role-played by the main model.

### Tier wiring (small surface, per existing patterns)

1. `nx/_subagent_tiers.py`:
   - `_normalize_tier()`: accept `"advisor"`.
   - New `_advisor_tier_dispatch()`: read-only table + `consult_advisor`.
2. `nx/_prompts.py`: two new constants:
   - `ADVISOR_EXEC_SYSTEM` — executor prompt: plan first, consult on
     unrecoverable ambiguity only, maintain `implementation-notes.md`
     (Plan + Deviations sections), conservative default on conflict,
     never let a consult replace its own judgment silently.
   - `ADVISOR_SYSTEM` — the advisor side-call prompt above.
3. `nx/_subagent.py`:
   - `tier` enum in the `spawn_agent` schema (~line 151): add `"advisor"`.
   - Prompt selection at `:876-877` gains the advisor branch.
   - Consult budget/timeout bookkeeping on `_SubagentContext`.
4. `docs`: CONTRIBUTING.md tool touch points if the tool is registered in
   the main registry; if the tool exists only in the advisor dispatch table
   it is not an always-on schema and costs 0 floor tokens — verify with
   `tests/test_initial_context.py` (budget must not move).
5. Semver bump + memory note on completion.

### implementation-notes.md contract

- Created in the sub-agent's worktree (or cwd for non-worktree runs) at
  executor start, before the first tool call:
  `# Implementation notes` + `## Plan` (short numbered plan).
- Appended under `## Deviations` when any of:
  - consult answer is rejected by the executor's own judgment;
  - consult timeout / malformed-reply / exhausted-budget fallback fires;
  - an edge case forces deviation from the plan (log the conservative
    choice made and why).
- Kept going-forward: a deviation is logged, not escalated — the executor
  does not stop to renegotiate the plan.
- Final report includes the notes path; on completion the advisor's
  decisions and the deviations list are summarized in the report head so
  the main model can audit judgment calls without opening the file.

### Interaction with existing machinery

- **Worktrees**: with `NX_SUBAGENT_WORKTREE=1`, notes land in the agent's
  own worktree — no cross-agent clobbering. Parallel advisor agents are as
  safe as parallel write agents.
- **Steering**: human steering continues to work unchanged; consult replies
  are not steering and do not go through `_AGENT_QUEUES`.
- **Autowake / bg sweep**: unchanged. Async advisor agents report via the
  existing `[Runtime note]` contract; consults happen inside the agent's
  own thread and are invisible to the sweep.
- **Staleness/locks** (`_sessionlock`, `_staleness`): consult side-calls
  are model calls, not edits — they take no session lock.

## Testing plan

- `tests/test_subagent_advisor.py`:
  - tier normalization accepts `advisor`; read-tier agents refuse the tool;
    advisor-tier table exposes it.
  - consult handler returns parsed JSON to the executor (fake model).
  - budget exhaustion → refusal; timeout/malformed → conservative fallback
    + `## Deviations` entry appended.
  - `stop: true` ends the run with a structured report.
  - notes file created at start, path present in final report.
  - token floor unchanged (`test_initial_context.py` passes untouched).

## Non-goals

- No live conversation between executor and advisor (single-shot per
  consult).
- No advisor access to tools or the parent transcript.
- No change to read/write/implement tiers' behavior.
- No human-in-the-loop escalation path (noted as a future extension).
