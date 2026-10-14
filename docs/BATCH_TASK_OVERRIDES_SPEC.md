# Batch task overrides — per-task approval posture and model (spec, 0.156.0)

Status: SHIPPED 0.156.0 (phases M=19064ae, A=4501f9e; all 5 calls signed off, plus the
headless-level opt-in amendment at sign-off). Follows `docs/YOLO_POSTURE_SPEC.md`
(0.155.0), which lists this as a non-goal: "Batch per-task posture (now trivially
possible later; not part of this work)" (`YOLO_POSTURE_SPEC.md:53`).

## Problem

`nx run` applies ONE posture and ONE model to every task in the file:

- Posture is process-global: `--yolo`/`--approval` are consumed at import by a
  whole-argv scan (`nx/__init__.py:300-308`) and never touched per task
  (`_batch.py:176-178`). Headless, `_confirm` denies everything not YOLO- or
  `NX_APPROVAL_POLICY`-covered (`_approval.py:347-357`,
  `_approval_policy.py:43-50`) — so an eval corpus mixing read-only and
  write tasks forces all-or-nothing YOLO for the whole run.
- Model is process-global: batch workers never populate a call target, so
  `open_stream` takes the globals branch (`_compress.py:239-254`). There is no
  `--model` flag and no per-task `model` field.
- The task schema (`instruction`/`id`/`session`/`timeout_seconds`,
  `_batch.py:214-229`) silently COPIES unknown keys (`task = dict(parsed)`,
  `:215`) and ignores them — an author writing `{"model": ...}` today gets
  silence, which is the worst failure mode.

## Mechanism (settled — no new primitive)

Both overrides install per task in the one shared task body,
`_run_one_task` (`_batch.py:329-376`, used by serial AND parallel paths),
alongside the existing per-thread slots (`_ASYNC_QUIET_SLOTS`, `_TASK_TLS`,
`st._set_thread_metrics`):

1. **Posture:** wrap the body in `st._posture_scope(posture)` — the shipped
   0.155.0 TLS scope (`_state.py:154-186`). `_confirm` already resolves through
   `st._posture()` at decision time (`_approval.py:349,362,412`), so a scoped
   posture takes effect with zero approval-path changes.
2. **Model:** a new dedicated thread-local call target —
   `_subagent_model.call_target_scope(target)` (`_CALL_TARGET_TLS` + resolver
   `call_target()` + scope, house pattern) — consumed in `open_stream`'s
   resolution chain.
3. **Per-task threshold opt-in (amendment):** `_approval.py` gains
   `_TASK_LEVEL_TLS` (plain thread-local, batch-installed alongside the
   posture scope when the task set a LEVEL, not `yolo`). The quiet branch
   becomes: YOLO → allow; policy glob → allow; **task level set AND
   `_local_risk_hint(action)` at-or-below it → allow** (local hint ONLY — no
   model-classifier round trip headless); else deny. Unset task level keeps
   today's exact flow. `_approval_policy` gains a TLS resolver
   (`policy_scope(spec)`, `_POLICY_TLS` checked before `os.environ` in
   `_policy_globs`) for per-task `approval_policy`; sub-agents spawned inside
   the task inherit both via the same thread.

### Rejected: synthetic `_SubagentContext` for batch tasks
Setting `_SUBAGENT_TLS.ctx` on a batch worker would reuse model routing for
free, but ctx-presence is the "inside a sub-agent" test elsewhere (restricted
dispatch tables, tier gates, delegation notes). A batch task is a FULL agent,
not a sub-agent — it must keep the full tool table. Dedicated slot, and a
guard test that batch never sets `_SUBAGENT_TLS.ctx`.

### `open_stream` resolution order (new)
```
sub-agent ctx target  →  batch call-target TLS  →  globals (st.client/st.MODEL/st.ACTIVE)
```
The guard at `_compress.py:244` (`target is not None and st.client is not
None and st.MODEL is not None`) must consult the resolved target BEFORE the
global sanity check, or a batch target silently falls back when globals are
unconfigured.

### Sub-agent inheritance chain (fix folded in)
A sub-agent spawned *inside* a batch task resolves "inherit parent" as
`st.client/st.MODEL` (`_subagent_model.py:129-134`) — after this change that
must become the *active call target* (TLS) when one is set, so the chain is
run → task → agent, not run → agent. Posture already chains correctly via
`_posture()`'s ctx-first rule (`_agent_posture` reads `_posture()` at spawn).

### The `_request_opts` wart — DECISION REQUIRED (§Calls, call 3)
With a routed target, `_clamp_max_tokens` (`_request_opts.py:50-55`) and
`_grammar_extra_body` (`:113,123`) still read `st.ACTIVE`, and
`extra_request_kwargs()` reads `st.MODEL` (`_sources.py:160`) — so a task on a
different SOURCE gets the *run's* token clamp and tool-call grammar.
Sub-agents already carry this. For batch it is a correctness issue, not a
nag: a `source:` target on a smaller window would be clamped by the wrong
source.

## Schema and precedence

One grammar per key, mirroring `timeout_seconds`' ladder
(task field → run flag → env/default, `_batch.py:266-270`):

| task field   | run flag            | env                     | default          |
|--------------|---------------------|-------------------------|------------------|
| `approval`   | `--approval`/`--yolo` | `NX_APPROVAL`         | prompt-all       |
| `model`      | `--model`           | active source (`st.MODEL`) | process default |
| `approval_policy` | (none — env only today) | `NX_APPROVAL_POLICY` | deny-all   |

- **`approval`**: string, EXACTLY the `NX_APPROVAL` vocabulary normalized by
  `_normalize_approval` (`_approval.py:~90-113`) — `"yolo"`, `"low"`,
  `"medium"`, `"high"`. No new boolean key, no `yolo: true` dialect. A task
  `approval` is an ABSOLUTE posture, not a merge: `{"approval": "low"}` in a
  `--yolo` run prompts-deny above low. Task field beats run flag beats env.
- **Amended at sign-off — headless levels made real.** On quiet surfaces the
  threshold never fires today (`_confirm`'s quiet branch short-circuits before
  it, `_approval.py:347-357` — run-level `--approval medium` in batch is
  deny-all), so accepting a level per task without wiring it would be the
  silently-dead key this spec's Problem section condemns. Per-task levels get
  an explicit, cost-free opt-in (below, Mechanism 3); RUN-level headless
  behavior stays byte-identical (deny; documented, pinned by
  `test_subagent_async.py:401-415` and `test_approval_policy.py`).
- **`model`**: string; plain model id rides the active source's client;
  `"<source>:<model>"` routes to that registered source — the EXACT pool-entry
  syntax (`NX_SUBAGENT_MODEL_POOL`, `_subagent_model.py:56-69`), so one
  mental model covers pool, task, and flag. `--model` (new run flag, same
  syntax) sets the run-level default that feeds the same ladder.
- **`approval_policy`**: string, same `headless-allow:<globs>` value as the
  env var. `_policy_globs` reads `os.environ` per call
  (`_approval_policy.py:34`); add a TLS resolver (same house pattern) checked
  before env. No run flag — env covers the run level already; a per-task
  policy narrowing a `--yolo` run is exactly the "one kind of thing
  per task" case C4 documented as impossible without full YOLO.

## Validation — fail fast, names the line

New keys validate in `_parse_tasks` (`_parse_tasks`, `_batch.py:197-231`),
raising `_BatchUsageError` → stderr, process exit 2, NO JSONL line — the
established class for author mistakes. Do NOT copy `timeout_seconds`' split
behavior (per-task error line in serial, exit 2 in parallel,
`_batch.py:505-536`): a bad posture must never reach the API. This also ends
the silent-ignore trap for these three keys specifically (unknown OTHER keys
keep being ignored — that's the schema's forward-compat stance).

`model` validates harder than the sub-agent path: an unknown `source:` prefix
is a PARSE error here (operator-authored file), not the sub-agent's
warn-once-and-inherit (`_subagent_model.py:112-117`), which exists because
spawn is best-effort. Batch authors get told before tokens burn.

## Output and session truth

- Result line (`_batch.py:307-321`): add `model` (effective model id) and
  `approval` (effective posture string) — per-task provenance for eval
  forensics; a `--yolo` task in a deny run must be visible in the log.
- Session meta (`_batch.py:391`): currently stamps `st.MODEL` — MUST stamp the
  task's effective model, or resumed transcripts lie about what answered.

## Files touched (estimate)

`nx/_batch.py` (schema, flags, scope install), `nx/_subagent_model.py`
(+`call_target_scope`/resolver, inherit branch), `nx/_compress.py`
(resolution chain), `nx/_approval_policy.py` (TLS resolver), `nx/_sources.py`
+ `nx/_request_opts.py` (wart fix, if accepted), `nx/run.py`-surface =
`_parse_batch_args` + `_USAGE_TEXT` (`_batch.py:86-115,122-185`),
`nx/_approval.py` (quiet branch + `_TASK_LEVEL_TLS` ONLY — the interactive
ladder is untouched), `nx/_approval_policy.py` (TLS resolver),
`tests/_nxtest.py` + `tests/test_state_reset_contract.py` (drain
`_CALL_TARGET_TLS`/`_TASK_LEVEL_TLS`/`_POLICY_TLS` through the unified reset),
README §run + §env, docs ledger. NO change to `_state.py`, `_turn*.py`.

## Inheritance matrix (the whole feature in one table)

| surface            | posture source                  | model source                    |
|--------------------|---------------------------------|---------------------------------|
| REPL               | globals (/yolo, argv, env)      | globals (switch_source)         |
| batch run-level    | globals (unchanged --yolo path) | new --model → run default       |
| batch task         | `_posture_scope` (new)          | `call_target_scope` (new)       |
| sub-agent of task  | ctx scope over task (existing)  | inherit = active target (fixed) |
| Telegram           | globals (unchanged)             | globals (unchanged)             |

## Non-goals

- Sandbox / per-task fence (P3-1: no sandbox module exists; `NX_WRITE_FENCE`
  reads env per call, `_fence.py:179,461` — a per-task fence is its own item).
- Per-task INTERACTIVE approval answering (no protocol; headless stays deny-by-default).
- Per-task `timeout` grammar changes; unknown-key strictness in general.
- Run-level `--model` for the REPL (`--source` + `switch_source` own that;
  this flag is batch-scoped — argv is `nx run`-only).
- Eval corpus schema changes beyond the three keys (harness reuse gives them free).

## Test plan

1. **Precedence, per key** — task > run flag > env > default, all three keys,
   parse-level unit tests.
2. **Validation rejects** — bad `approval` value, bad `model` syntax, unknown
   `source:` prefix, non-string types → exit 2, stderr names the line, no
   JSONL line, no API call.
3. **Per-task yolo approves, sibling denies** — two parallel tasks, gated
   `_confirm`: task A (`"approval": "yolo"`) True, task B default → `"DENIED
   by user"`; extends `test_approval_policy.py`'s wiring test.
3b. **Task-level threshold opt-in** — task `"approval": "medium"` headless
   allows an action with a local hint ≤ medium and denies hint `high` or
   hintless; run-level `--approval medium` with no task key stays DENY-all
   (byte-identical legacy), and a `yolo` task still short-circuits first.
4. **Per-task model reaches the wire** — extend the `test_turn_loop_e2e.py`
   stub-server pattern: two parallel tasks with different `model` fields,
   assert each request body's model id (the sibling of
   `test_concurrent_agents_do_not_clobber`).
5. **Resolution order** — open_stream: ctx target > batch TLS > globals;
   and the guard fix (batch target with unconfigured globals doesn't fall back).
6. **Inherit chain** — sub-agent inside a per-model task inherits the TASK's
   target, not the run's.
7. **Session meta + result line** — effective model/approval stamped on both;
   `st.MODEL` never stamped when the task overrode it.
8. **Per-task `approval_policy`** — glob allows its tool headless; sibling
   without the key still denied; env still governs runs without the key.
9. **Guard** — batch never writes `st.YOLO`/`st.APPROVE_LEVEL`/
   `os.environ[NX_APPROVAL*]`/`_SUBAGENT_TLS.ctx` (pattern:
   `test_posture_readers_guard.py`).
10. **Parse pass-through** — `--model` accepted in the flag loop, unknown
    value still `_BatchUsageError`; usage text updated.

## Calls needing sign-off

1. **Scope: posture + model in one release** (one mechanism, one validation
   story, one ledger entry) vs posture-only-now. Recommendation: one release.
2. **`approval` is absolute, not merged** — task `"low"` inside `--yolo` run
   demotes. Recommendation: yes, absolute; merge semantics would make the run
   flag un-overridable in the only direction that matters (tightening).
3. **The `_request_opts` wart**: fix in-scope (thread `call_src`/`call_model`
   into clamp/grammar helpers — small, but touches the shared request path)
   vs document-and-defer (batch tasks on a *different model id, same source*
   are unaffected; the wart only bites `source:` targets).
4. **Echo fields on the JSONL line** (`model`, `approval`): new output fields
   are technically additive-but-visible to strict consumers.
   Recommendation: include; provenance is the point.
5. **Version: 0.156.0** (new flags + schema keys, backward compatible).
