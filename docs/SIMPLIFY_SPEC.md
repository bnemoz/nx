# Simplify spec: one harness, thin clients (0.156.3 → 0.158.0)

Status: approved for implementation. Origin: adversarial audit of the advisor
tier + delegation + `/build` trio (2026-10-13). Verdict: delegation is the
primitive; `/build` is a second harness wearing a slash command — it
re-implements fan-out/join, locking metadata, and a model side-call the layer
already has equivalents for. The advisor tier is cheap but is a tier-shaped
solution to a flag-shaped problem. Plan: converge the primitives, shrink
`/build` to a driver, then fold the advisor tier into implement.

## Baked decisions (do not re-litigate)

1. **`/build`'s ACTIVE pid file stays.** `_sessionlock` is a per-repo,
   per-session advisory gate acquired at REPL startup; it cannot express a
   per-run hard fence (same-process second `/build` would sail through), and
   the refusal message needs `{pid, run_id, branch, started}` metadata it does
   not carry. Verified against `_sessionlock.py` (full) and `_build.py:104-171`.
2. **The side-call helper covers exactly three sites**: `_risk._assess_risk`,
   `_subagent_tiers._consult_advisor` (raw-call part only; budget/deadline
   wrappers stay), `_build._ask_main_model`. The four memory/`_sessions_fs`
   sites are a different primitive (user-only prompt, `_record_usage`, no JSON
   contract) — do NOT migrate.
3. **`spawn_agent_job` sits below the policy layer** (async+write refusal,
   duplicate guard, YOLO nudge stay in `spawn_agent`); `spawn_agent` becomes a
   thin caller of it. No new model-facing tool; zero schema tokens.
4. **Phase 3 critique bug fixed in passing**: `_phase_critique` currently
   parses findings out of `_lean_result`'s 600-char head; the new path returns
   the full answer.
5. **Kept as-is** during cleanup: `_subagent._live_worker_threads()` (test
   harness drain depends on it) and `NX_SUBAGENT_MAX_NESTED` (documented knob
   that gates prefetch parallelization).

## Phase 1 (0.156.3) — shared one-shot JSON side call

New `nx/_sidecall.py`:

```python
def side_call_json(system, user, *, timeout=30.0, parse_attempts=1,
                   retry_prompt=None, purpose="side_call", conn_retries=0,
                   conn_retry_seconds=0.0) -> tuple[dict | None, str]:
    """One-shot, no-tools call to the session's model. Returns
    (parsed_or_None, raw_text). None on any failure; raw_text always set
    (possibly "")."""
```

Helper owns: `st.client is None or st.MODEL is None` guard, payload assembly,
`chat.completions.create(stream=False)` with timeout +
`extra_request_kwargs()` spread, `_log_event("req"/"resp", purpose)`,
`_record_api_call(resp, fallback_chars=..., fallback_prompt_chars=...)`
(this fixes `/build`'s missing prompt-char fallback), `json.loads` on stripped
content with `parse_attempts` re-calls (same payload) and optional
`retry_prompt` appended payload (risk's sharp follow-up only).

Caller-owned: failure mapping (risk → `("high", reason)` + its regex
level-word fallback over `raw_text` + its connection-retry ladder via
`conn_retries`; advisor → its ERROR strings, budget decrement and deadline
loop stay in `_consult_advisor` passing `timeout=max(1, deadline-now)`;
build → `None` + `_note`). Migrate all three sites; delete their duplicated
call/payload/metrics blocks.

Verify: `tests/test_risk*.py tests/test_subagent_advisor.py tests/test_build.py`
+ `ruff`. Add one test pinning `(dict, raw)` contract and metrics recording.

## Phase 2 (0.156.3, parallel with P1) — joinable sub-agent handle

Refactor `nx/_subagent.py` internals (no signature change to the
model-facing `spawn_agent`):

```python
def spawn_agent_job(instruction, max_turns, tier="read", scope=None, *,
                    worktree=None, announce=True) -> AgentJob

class AgentJob:
    job_id: str
    def join(self, timeout=None) -> str    # FULL answer, None on timeout
    def result(self) -> dict               # {answer,status,report,exit_state,turns,worktree}
    def done(self) -> bool
```

Required changes (reuse everything else as-is — ledger, `_ASYNC_LOCK`,
semaphore/queue, quiet slot, reports, intents, eviction, `_LIVE_WORKERS`):

- Extract `spawn_agent_job` from `_spawn_async`; `_spawn_async` becomes a
  formatter over the handle (it needs only `job_id`/`queued`/`queue_depth`).
  `spawn_agent` keeps its policy layer and calls through.
- `handle.done = threading.Event()`, set in `_worker`'s `finally` immediately
  after `_run_async_agent` returns (record already final), before the
  bookkeeping tail. `join(timeout)` = `done.wait(timeout)` then read the
  record under `_ASYNC_LOCK`. `join()` returns the full answer, not the lean
  string.
- `announce=False` (the orchestrator contract): at finalize set
  `rec["announced"] = True` and skip `_stage_subagent_answer` — setting the
  flag is mandatory (else the parent sweep stages a note anyway and the
  record never becomes an eviction candidate). Truncate the RAM answer the
  same way AFTER the handle captured the full answer. No `[Runtime note]`,
  no wake arm side effects for such jobs.
- `worktree=` pass-through: caller-supplied worktree skips the
  `_worktree_enabled()/create_worktree` block; outcome-baseline code already
  handles a given worktree.
- `result()` exposes `rec["status"]` so callers can see `queued` (the
  semaphore may park build packages).

Verify: `tests/test_subagent*.py tests/test_async* tests/test_worktree*
tests/test_autowake.py` — all green unchanged except new tests:
announce=False leaves no `_PENDING_SUBAGENT_NOTICE` residue and the record
still evicts; `join()` returns full answer; ledger shows build-style jobs
(`list_async_agents` visibility comes for free).

## Phase 3 (0.157.0) — `/build` becomes a driver

Depends on P1+P2. In `nx/_build.py`:

- `_run_agent`: delete the hand-rolled quiet save/restore and the direct
  `_run_sync_agent_body` call; use `spawn_agent_job(..., announce=False,
  worktree=pkg.worktree)`.
- `_run_packages`: replace raw `threading.Thread`+`join()` with N jobs +
  `join(timeout)`; `_join_package_threads` → bounded joins on the handles.
  Keep main-thread worktree creation (git lock race). The semaphore now caps
  fan-out (accepted behavior change — record status shows `queued`).
- Phase 3: critique via the joinable path so `_parse_findings` sees the full
  answer (fixes the lean-head bug).
- `_ask_main_model` body → `side_call_json(BUILD_FIX_SYSTEM, ...)` (P1).
- Merge-back (`_merge_package`/`_retry_package`), spec parsing, the critique→
  fix ladder, preflight, pid-file fence: KEEP — genuinely `/build`-specific.
- Net target: `_build.py` well under ~950 LOC; behavior diffs limited to
  package visibility in `/tabs` + semaphore-gated fan-out (document both in
  README's `/build` rows).

Verify: `tests/test_build.py` (31 tests; update the fan-out internals pins),
then one full suite run: `uv run pytest tests/ -n 8`.

## Phase 4 (0.158.0) — advisor tier → implement + flag, cleanup

- Advisor becomes `tier="implement"` + consult eligibility: add
  `consult_advisor` to the implement dispatch when `NX_SUBAGENT_CONSULT_BUDGET`
  > 0 and `NX_SUBAGENT_WRITE=1`; `ADVISOR_EXEC_SYSTEM` merges into
  `SUBAGENT_IMPLEMENT_SYSTEM` (notes/Deviations + consult discipline
  paragraphs). `tier="advisor"` kept working as a deprecated alias (same
  schema enum entry, README notes the alias). Delete
  `_advisor_tier_dispatch` once the alias routes to the merged path.
- Dead-code removal: `_worktree.in_worktree()`,
  `_worktree._session_worktree_enabled_with_fallback()`, the unreachable
  `wt_note` branch in `_run_async_agent` (`_subagent.py:635/722`).
- Docs: `docs/ADVISOR_SPEC.md` status `proposed` → `shipped (0.112.0),
  superseded by this spec (0.158.0 tier merge)`; README env/tool tables;
  `docs/` design doc line for `/build` pointing at this spec.

Verify: `tests/test_subagent_advisor.py tests/test_subagent_write.py
tests/test_build.py` + full suite; `ty check nx` clean on touched files.

## Non-goals

No changes to: worktree isolation, model pool, bash allowlist, wake/autowake,
the memory side-call sites, `todo_write`/plan-mode, `_batch.py`, and any
always-on context floor (pinned by `test_build_adds_no_tool_schema` /
`test_initial_context.py`).
