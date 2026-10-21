# PTC Migration Spec — programmatic tool calling for nx

Status: DRAFT — nothing landed yet (all phases unstarted) · original target 0.160.x → 0.164.x has
passed; retarget on first landing · Basis: arXiv 2608.06370 (BFCL v4, 14 models),
CodeAct (Wang et al. 2024), Anthropic advanced tool use.

## Problem

Every nx action crosses a JSON serialize/parse boundary: the model emits a JSON blob,
`_turn_toolcalls.py` repairs/salvages it (`repair_json_candidates`, `_salvage_truncated_args`,
`normalize_tool_args`), and failures feed `_argsalvage` + the `_breaker` streak. This is the
least reliable link in the harness, and it is pure overhead for code-capable models. PTC
replaces the boundary with Python: the model writes a script against typed stubs, one
subprocess execution replaces N round-trips, fan-out and chaining stop costing inference turns.

Non-goal: removing the JSON path. The arXiv data shows older/smaller models regress under
PTC; JSON stays as the default-off-ramp and per-model fallback.

---

## Phase 0 — Baseline & instrumentation (0.160.0)

Measure before touching anything. All counters already exist in fragments; consolidate.

- **Metrics to emit** (into the obs pack / stats footer, `nx/_turn_footer.py`):
  - malformed-args count per 100 tool calls (from `_handle_native_tool_calls` repair loop,
    `nx/_turn_toolcalls.py:158-193`) split by: repaired-by-candidates, bracket-salvaged,
    normalized, hard-failed, breaker-tripped.
  - per-turn tool-call count and fan-out size distribution (N per `_should_parallelize`
    batch) — the baseline for the "drops calls above model-specific N" claim.
  - tokens spent on tool-call envelopes vs tool results (envelope = the JSON args string
    itself, already present in `assistant.tool_calls`).
- **Acceptance:** footer shows a `ptc-baseline` line; a `tests/test_ptc_baseline.py`
  asserts the counters against synthetic streams. No behavior change.

## Phase 1 — Typed stub module from DISPATCH (0.161.0)

One generator, one source of truth, zero new schemas.

- **New module `nx/_ptc_stubs.py`:** `render_stub_module(tools: list[dict]) -> str` walks
  the `TOOLS` OpenAI envelopes (`nx/__init__.py:395-440`) and emits a Python module:

  ```python
  def read_file(path: str | list[str], offset: int = 1, limit: int | None = None) -> str:
      """<description from schema>. Returns tool result string."""
      return _capture("read_file", {"path": path, "offset": offset, "limit": limit})
  ```

  Signature generation rules: schema `type: object` flat properties → typed params
  (string→str, integer→int, boolean→bool, array→list, `anyOf`/`oneOf` → `|` union of the
  branches — `read_file`/`search_files` use `oneOf` (`nx/__init__.py:396,419`);
  defaults preserved). Array-of-object properties (`edit_file`/`multi_edit` `edits`,
  `todo_write` `todos`) map to `list[dict]` rather than a typed element type; tools whose
  handler kwargs can't be recovered from a generated signature this way are excluded
  (see exclusions below).
  Docstring = the schema property descriptions, which the
  `tests/test_tool_schema_invariants.py` invariants already guarantee exist.
- **Gate awareness:** the rendered module must reflect `_active_tools()` (`nx/_tool_gates.py:126`)
  for the *current turn* — withheld tools get no stub, so the model cannot reference them.
  On sub-agent turns `_active_tools()` bypasses the memo and returns the tiered set
  (`nx/_tool_gates.py:144-146`), so the stub module must be rendered from
  `_subagent_tools()` on that path too — never serve the parent's memoized stubs inside
  an agent turn. Render once per turn keyed on the same sorted-gate-state key as
  `_gate_vector`, with the sub-agent path skipping the shared memo.
- **Non-callable exceptions:** `spawn_agent`, `critique`, `read_image` (returns content parts,
  not str), `call_project_tool` (dict args), and `final_answer` are excluded from stub
  generation in Phase 1 and must be called via JSON: `final_answer` is not in the Phase 2
  read-only allowlist, and an end-of-turn raised from inside a script has no path to
  `_handle_forced_final_tool_call`'s TURN_DONE rendering. Script-level end-of-turn
  semantics are deliberately out of scope for this migration.
- **Tests:** golden-file test that the generated module imports and `py_compile`s; a
  signature-vs-`DISPATCH`-handler-kwarg cross-check (the handlers' `**_` tolerance makes
  over-narrow stubs safe, but a mismatch test prevents drift).
- **No runtime behavior change yet.** The module is generated and unit-tested, not exposed.

## Phase 2 — `run_tools(code)` execution tool (0.162.0)

One new tool, reusing every existing gate.

- **Schema** appended to `TOOLS` (gated: only offered when `NX_PTC=1` or auto-enabled per
  model-generation heuristic later):
  `run_tools(code: string, timeout: int = 30)` — executes the model's Python in a
  subprocess with the stub module importable (`import stubs`).
- **Execution path:** `_capture` in the stub module marshals each call back to the harness
  process over a local socket/stdio protocol, and each one goes through
  **`run_tool(name, args)` (`nx/_dispatch.py:201`) unmodified** — plan gate, withheld
  refusals, breaker check, arg repair, write fence, hooks, cache, undo snapshot, sanitize
  all apply per-call exactly as in JSON mode. This is the load-bearing design decision:
  PTC is a *syntax* migration, not a gate bypass.
- **Turn-layer integration (required, not optional):** several semantics live in the turn
  layer, not in `run_tool`, and the pipe protocol must carry them explicitly:
  - *breaker accounting*: malformed-args breaker increments happen in
    `_handle_native_tool_calls` (`nx/_turn_toolcalls.py:200-216`), not `run_tool` — the
    harness side of the pipe reports per-call failures back so the same streak/trip logic
    runs for stub calls.
  - *interrupts*: `_EscToChat` / `KeyboardInterrupt` during a script must kill the
    subprocess and skip-fill every in-flight `_capture` with an error result, mirroring
    the skip-fill loop at `nx/_turn_toolcalls.py:269-327` so every tool_call still gets
    a transcript entry.
  - *approval plumbing*: any call that would have needed interactive approval in JSON mode
    is refused over the pipe with the standard refusal string (Phase 2 is read-only
    allowlist, so this is a backstop, not the primary gate).
- **Sandbox rules (read-only phase):**
  - Allowlist: only tools in `_CACHEABLE_READ_TOOLS` ∪ `PLAN_READONLY_TOOLS`.
  - The subprocess has no network and no shell; it may only talk to the harness pipe.
  - Result protocol: each `_capture` returns the tool result string; stdout of the script
    (after results are stripped) is folded into context like a tool result, with the same
    sanitization applied by `run_tool`'s post-processing.
  - Failure mode: a stub raising inside the script marks that call failed (same
    `"ERROR: ..."` fold); a script-level exception returns one error result for `run_tools`
    itself and increments the same breaker streak as a malformed JSON call would — parity
    of failure accounting.
- **Cache interplay:** calls through the pipe hit the same `st._TOOL_RESULT_CACHE`
  (`nx/_state.py:586`; keyed `(name, sorted-args-JSON)`, validated by re-statting backing
  paths / `git_state_stamp` in `nx/_tool_cache.py`), so a repeated read inside a script is
  free, identical to sequential JSON calls today. Note the prefetch/speculate tuple in
  `_tool_prefetch.py:18` omits `recall_lessons` (present in `_dispatch._CACHEABLE_READ_TOOLS`)
  — harmless today, but stub-level fan-out should use the dispatch tuple as its source of
  truth so the sets don't diverge further.
- **Prompt surface:** one paragraph in `_session_preamble.py`'s one-time nudge string,
  plus stub-module source is injected as a system message on the first `NX_PTC` turn.
- **Tests:** subprocess harness fake (no real dispatch of write tools); a test that a
  script calling `write_file` is refused by the plan gate; breaker-parity test.

## Phase 3 — Fan-out convergence (0.163.0)

Retire hand-rolled batching in favor of stub-level batching.

- `_tool_prefetch.py` (`_run_read_only_parallel` `:27`: 2+ read-only calls on a
  `ThreadPoolExecutor`, `workers=min(8, len(jobs))`, results returned keyed by original
  call index; `_prefetch_cacheable_tools` `:184` warms `st._TOOL_RESULT_CACHE` on
  mixed turns) and `_spec_reads.py` (`maybe_speculate` `:125`: speculative dispatch of
  balanced-JSON read calls during SSE streaming, `NX_SPECULATIVE_READS` off by default)
  exist because JSON forces the model to emit N separate blobs. Under PTC, the model
  writes one `for` loop / `asyncio.gather` block. Migration:
  - `run_tools` executes concurrent `_capture`s concurrently (thread pool in the harness
    side of the pipe, same `min(8, …)` bound as `_run_read_only_parallel`), preserving
    call-index ordering for the observation group.
  - `_spec_reads` stays: speculative *streaming-time* dispatch is a different mechanism
    (it starts before the call stream completes) and remains a win in both paradigms.
  - Deprecate `_tool_prefetch`'s batching role once telemetry shows ≥80% of read fan-outs
    happen inside `run_tools` scripts; keep the cache-prefetch half (it serves JSON mode).
- **Acceptance:** Phase 0 counters show fan-out turn counts collapsing on PTC-enabled
  sessions; the "drops calls above N≈70" failure class becomes structurally impossible
  for read tools.

## Phase 4 — Rollout policy & versioning (0.164.0)

- **Per-model enablement:** default `NX_PTC=auto` — on for model IDs matching a
  generation heuristic list (the arXiv result: all current Anthropic models and
  recent GPT generations track or beat JSON; GPT-4o/4.1-class regress). `NX_PTC=0/1`
  overrides. Document in `sources.example.env` + `CONTRIBUTING.md` env table.
- **Fallback ladder:** if a session trips the breaker twice on `run_tools` scripts, the
  harness drops the tool from the gate vector for that session (same mechanism
  `_invalidate_tool_list_cache` serves) and nudges back to JSON. One knob, reversible.
- **Semver/versioning:** the repo has no CHANGELOG.md — `git commit` messages are the
  de-facto changelog (`"<semver>: <headline>; <headline>"` subject, body bullets per
  `nx/_module.py` change, per 0.158.x/0.159.0 history). Each phase lands as a version
  commit touching `pyproject.toml` (and `uv.lock`), minor bump per phase as listed
  (behavioral/scope change), with the same subject/body style.
- **Explicit non-deprecation:** JSON tool calling is never removed. `_argsalvage.py` and
  `_breaker.py` remain for the JSON path; they just stop being the hot path.

---

## Risks

| Risk | Mitigation |
|---|---|
| Stub signatures drift from handlers | Phase 1 cross-check test against handler kwargs |
| Script subprocess becomes an exfiltration channel | No-network sandbox, pipe-only protocol, read-only allowlist in Phase 2 |
| Double-parsing of results (script + context render) | Results pass through `run_tool` sanitize exactly once |
| Model writes scripts for single calls (overhead) | Nudge: "use run_tools for ≥2 independent reads"; keep JSON for singletons |
| Older models regress | Per-model default + session fallback ladder (Phase 4) |

---

## Audit (pre-implementation, code-verified)

All file:line claims in this spec were verified against the tree on 2026-10-21. Confirmed
exactly: `_turn_toolcalls.py:158-193` (repair loop), `:200-216` (breaker), `:269-327`
(skip-fill); `run_tool` at `_dispatch.py:201`; `TOOLS` at `__init__.py:395-441` with
`oneOf` for `read_file`/`search_files` at `:397`/`:420`; `_active_tools()` at
`_tool_gates.py:126` with the sub-agent bypass at `:144-146`; `_gate_vector` memo key =
`tuple(sorted((name, bool(gate())) …)) + tuple(sorted(broken))` at `_tool_gates.py:121-123`;
`_subagent_tools()` at `_subagent_tiers.py:143`; `_TOOL_RESULT_CACHE` at `_state.py:586`
(key builder `_tool_cache_key` `_dispatch.py:155-156`; validation `cache_valid()`
`_tool_cache.py:227`, `git_state_stamp` `:142`, `drop_changed_entries` `:336`);
`_run_read_only_parallel` at `_tool_prefetch.py:27` (`workers=min(8, len(jobs))` `:43`,
`dict[int, str]` keyed by call index `:41`); `_prefetch_cacheable_tools` `:184`;
`maybe_speculate` at `_spec_reads.py:125`.

### Corrections to the body

1. **`NX_SPECULATIVE_READS` defaults ON, not off** (§Phase 3, `_compress.py:152`:
   `_env_int("NX_SPECULATIVE_READS", 1) != 0`; `=0` is the kill switch). Phase 3's
   "remains a win in both paradigms" reasoning is unaffected, but any Phase 0 baseline
   counter must record speculation as the *default* path, not an opt-in.
2. **The Phase 2 read-only allowlist is wrong as written.** `PLAN_READONLY_TOOLS`
   (`nx/_plan.py:48`) is a *plan-gate* set, not a read-only set: it contains
   `plan_edit`, `todo_write`, `spawn_agent`, `wait_background`, `read_image`. Union with
   `_CACHEABLE_READ_TOOLS` (`_dispatch.py:152`) would let stub scripts call `spawn_agent`
   and mutate the todo list — both contradict §Phase 2's read-only rule and the
   "spawn_agent not callable" exclusion. The Phase 2 allowlist must be the explicit
   intersection `_CACHEABLE_READ_TOOLS ∩ PLAN_READONLY_TOOLS` (which is the 6 pure-read
   tools) plus nothing else; do not reference the plan set alone. Note `read_image` is
   excluded anyway (non-str return, §Phase 1 exclusions; `_tools.py:80` documents this).
3. **Handler-kwarg tolerance is common but not universal.** `**_: Any` is present on
   `read_file` (`_tools.py:52`), `search_files` (`:217-226`), `write_file`
   (`_edit_tools.py:144`), `run_tests` (`_testrunner.py:325`), `lint_check`
   (`_lint.py:112`) — but `duckdb_query` (`_duckdb.py:114`) and `web_fetch`
   (`_webfetch.py:862`) lack it. The Phase 1 cross-check test must treat
   missing-`**_` handlers as strict: a generated stub whose signature is narrower or
   wider than such a handler's kwargs is a hard error, not a drift warning.
4. **`recall_lessons` divergence is wider than stated** (§Phase 2 cache note): it's not
   just `_tool_prefetch.py:18` — `_spec_reads.py:136-137` imports the *prefetch* tuple,
   so speculation skips it too. Fix once: make both modules import
   `_dispatch._CACHEABLE_READ_TOOLS`.

### Implementation map for the build-out (per-phase file assignments)

- **Phase 0** — `nx/_turn_footer.py` (new `ptc-baseline` footer line); counters read from
  `_handle_native_tool_calls` state already threaded through `_turn_toolcalls.py`; new
  `tests/test_ptc_baseline.py`.
- **Phase 1** — new `nx/_ptc_stubs.py` (`render_stub_module`); golden-file +
  `py_compile` test + handler-kwarg cross-check in one new test file; renders from
  `_active_tools()` / `_subagent_tools()` per the gate rules above. Zero edits to
  existing modules.
- **Phase 2** — schema entry in `__init__.py` TOOLS + gate in `_tool_gates.py` (copy the
  `lint_check` template: gate lambda at `_tool_gates.py:80`, refusal at `_GATE_REFUSALS:99`);
  execution engine new `nx/_ptc_exec.py`; pipe protocol per §Subprocess below; stub-module
  injection and nudge per §Prompt wiring below; tests in `tests/test_ptc_exec.py`.
- **Phase 3** — concurrency lives in the harness side of `_ptc_exec.py` (mirror
  `_run_read_only_parallel`'s `min(8, …)` + index-keyed results); deprecation telemetry
  reads Phase 0 counters. `_spec_reads.py` untouched.

### Subprocess & pipe protocol details (no code has this today — net-new)

Existing subprocess precedent is uniform: argv-only `subprocess.run(..., capture_output=
True, text=True, timeout=_env_int(...))`, no `shell=True`, error folded into a string
(`_duckdb.py:160-168`, `_lint.py:145-151`, `_tools.py:312,388`). Reusable pieces:

- **Pipe framing:** closest precedent is `_hooks.py:47-51` (JSON over stdin, stdout
  capture). The `run_tools` protocol should be the inverse — harness-side reader loop
  with per-call request/response frames `{id, name, args}` → `{id, result|error}`;
  framing must be length-prefixed or NDJSON (tool results contain arbitrary text incl.
  newlines).
- **Kill/interrupt:** `_bash_jobs.py:235-236` is the precedent for
  `start_new_session=True` — required so Esc (`_EscToChat`) can kill the script's whole
  process group without touching the terminal; the skip-fill loop to mirror is
  `_turn_toolcalls.py:269-327`.
- **No-network sandbox: nothing to reuse.** No `unshare`/`seccomp`/`preexec_fn`/
  env-denylist exists anywhere in `nx/`; `_fence.py:51` explicitly disclaims being a
  sandbox. For Phase 2, "no network" is satisfied by construction (the stub module is the
  only capability surface and it only opens the harness pipe) — the spec's rule should
  read "the subprocess is untrusted but capability-poor: importable modules are the stdlib
  subset that cannot escape (no `socket`, `subprocess`, `shutil`, `os.system` reachable
  only if the model reimplements them — enforce by running with `-I` and a restricted
  audit hook, not by pretending the GIL sandbox exists)". Do not claim OS-level network
  isolation; the read-only allowlist + write fence in `run_tool` are the actual boundary.
- **Timeout:** `_env_int`-derived like every other subprocess timeout (`NX_PTC_TIMEOUT`,
  default the schema's 30); on `TimeoutExpired` kill the process group and fold one error
  result for `run_tools`, breaker-incremented.

### Prompt wiring details (§Phase 2 prompt surface)

Copy `nx/_delegation_note.py` (the best template for "note present only when a flag
gates a tool"): `DELEGATION_GUIDANCE:23`, marker constant `:108`, gate `_delegation_enabled()`
`:111-119`. A PTC note is a new `_load_ptc_note()` producer in `_prefix.prefix_notes()`
(`_prefix.py:158-180`), consumed by `_session_preamble._initial_messages():70-84` and the
resume path. **Caveat:** `pin_prefix()` (`_prefix.py:198`) pins head bytes (`_prefix.py`
is load-bearing) — the stub-module source must ride the runtime-note tail
(`_runtime_notes.py:54-57`), not the prefix, and the one-time nudge is a fifth constant
beside `BACKGROUND_GUIDANCE:97`/`YOLO_ASYNC_GUIDANCE:133`/`MULTI_EDIT_GUIDANCE:146`/
`DELEGATION_STREAK_NOTICE:161` in `_session_preamble.py`.

### Open items the phases must answer (not yet specified)

- Where `run_tools` schema sits in `TOOLS` ordering and whether it appears in the stub
  module itself (it must not recurse: no `run_tools` stub).
- Sub-agent turn execution: does the pipe respect `_subagent_tools()` gating inside the
  script too (stub render rule covers it) and does breaker state stay per-session or
  per-tier?
- Phase 4's per-model list needs a home — `_subagent_model.py` is the existing
  model-ID-classification precedent.
