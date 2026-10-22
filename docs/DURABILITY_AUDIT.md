# Codebase durability audit — `nx` 0.168.0

**Date:** 2026-10-21. **Auditor:** nx session (principal-engineer posture), assisted by four
read-only sub-agent sweeps (core execution path, persistent state, test-suite quality,
agent-safety surface) whose findings were cross-verified against direct reads.

**Verification performed by this audit:** full suite run twice — parallel `-n 8`:
**3015 passed / 25.6s**, serial: **3015 passed / 91.9s**; `ruff check .` clean;
`ty check nx`: 55 diagnostics (advisory per policy, `tests/**` rules relaxed); AST
dependency-graph analysis of all 149 `nx/` modules; direct line-level reads of every
cited location below. Everything else is labeled with its evidence source.

## A. System durability profile

**What it is.** `nx` is a terminal coding-agent CLI (61.7k LOC, 149 modules) driving
self-hosted and hosted LLM endpoints via OpenAI-compatible HTTP. One long-lived
interactive REPL process per repo, plus headless modes (`nx run` batch JSONL, `nx eval`,
sub-agents in threads/worktrees).

**Consumers.** (1) The human operator, live. (2) The model itself — `TOOLS` schemas,
`SYSTEM`, and refusal wordings are *interfaces to a non-deterministic consumer*, where a
"wrong" change produces plausible-but-bad behavior rather than a crash. (3) Headless
callers (`nx run`, batch, eval) that consume the turn-status vocabulary
(`done/cap/error/gave-up`) as machine contracts. (4) Downstream files the agent writes
into user repos.

**Persistent state, in descending replacement cost:**

- `~/.nx/sessions.sqlite` (WAL) — **sole transcript copy** since 0.101; FTS index;
  legacy JSON mirrors kept only as migration source (no longer written).
- Undo snapshots under `~/.nx/undo/` — whole-file pre-edit contents, "often secrets" by
  the repo's own comment.
- Memory store/index (`_memory_store.py`, `_memindex.py`), obsstore (content-addressed),
  bg-job logs, `_sources.py` API keys (env + `~/.nx` sources file).

**Declared invariants** (explicit, mostly test-pinned — unusually good): per-turn token
floor ≤ ~5–6K (`test_initial_context.py`); prompt-cache byte stability (`_prefix.py`
pin); honest status vocabulary — never score a truncated/non-answer turn as "done"
(0.154.6/7); write fence + approval posture; 0600/0700 private sinks; schema-on-every-
connection SQLite with fail-open stores; suite offline and ≤ ~15s parallel.

**Classification.** Durable core: turn loop + status mapping, dispatch/gates/fence,
session store, undo, credential handling. Semi-durable: tool gates/tool list cache,
compression, sub-agent tiers, docs-parity contracts (they *are* the docs' integrity
mechanism). Disposable/rebuildable: rendering (`_chatbox`, `_ansi`, `_fullscreen`),
`scripts/`, obsstore/reducer/occompact (env-gated, fail-open by contract), `_webfetch`.
Expected maintenance: extreme — 734 commits, `__init__.py` touched in 146, `_turn.py` in
102 of them; the repo self-describes as a fast-moving single-maintainer + AI-agent
codebase, which shifts the durability burden onto the parity/contract tests and the
(absent) enforcement layer discussed in §D.

**Evidence gaps:** single-machine audit; no CI/PyPI/release process visible (0 tags,
remote is GitHub, no workflows); no live provider traffic captured; flake ledger and perf
baselines taken as claimed except where re-run.

## B. Overall durability assessment

| Dimension | Assessment | Evidence |
|---|---|---|
| Understandability | **Adequate** | Excellent module map (AGENTS.md) and rationale comments; against it: `model_turn` at 529 lines with load-bearing branch ordering, 2,121 function-local imports across all 149 files, and a 152-name shared-mutable `_state.py` hub with 90 dependents — local reasoning requires knowing which of several homes a setting lives in. |
| Modification safety | **Adequate** | Honest-status contract, schema-invariant, docs-parity, and state-reset contract tests are real (verified their oracles; several confirmed line-level). Against: three confirmed defects (C-1..C-3) shipped *through* a 3015-test suite, and every gate is opt-in — no CI, no hooks. |
| Architectural sustainability | **Strong** | The 0.87–0.103 monolith decomposition works: re-export discipline holds, 1 cycle total (`_session_save↔_sessions_fs`), per-concern modules with ledger-governed growth budgets (`baselines.json`). Growth is watched, budgeted, and documented. |
| Cost of engineering rigor | **Strong (design) / Weak (enforcement)** | Rigor is genuinely cheap here — derived oracles (`dir(st)` state sweep), byte-comparison doc regeneration, 14s feedback loop. But *none* of it is automatically enforced: all gates are "run it before committing" convention. |
| AI-agent maintainability | **Strong** | The repo is ahead of the practice being audited: yolo/tiering, write fence, PTC sandbox, posture TLS, worktrees, `/turns` explainability. Caveats: agents can falsely report validation (C-4 finding), and read-only tiers are softer than their name (C-3). |

## C. Prioritized findings

### [P1] `nx sessions --prune` deletes undo snapshots of live, resumable sessions

**Category:** Modification safety (data integrity)

**Location:** `nx/_sessions_fs.py:1068-1088` (prune's undo GC); liveness check at `:1083`
uses `_session_path(sid)` defined at `:79-89` as the **legacy per-session JSON mirror
"no longer written" since 0.101**. Correct store-based GC exists at
`nx/_undo.py:283-317` and is wired from the artifacts sweep (`_sessions_fs.py:546-550`),
but prune doesn't use it.

**Observed behavior:** `os.path.exists(_session_path(sid))` is the "session dead?" test.
For every session created since 0.101 the mirror doesn't exist, so prune unlinks every
undo file except the live session's. Verified by direct read of both call sites and
`_session_path`'s own docstring. `tests/test_store_gc.py` exercises prune's image/row
paths; the undo branch is only reached with stubbed counts (sub-agent grep; test bodies
not personally re-read).

**Durability impact:** silent loss of whole-file pre-edit snapshots — the recovery layer
for exactly the edits the agent's own docs call the highest-consequence operation —
triggered by a documented user command, on sessions the user can still `/resume`.

**Failure scenario:** user runs `nx sessions --prune` to reclaim disk; next week they
`/resume` an old session, `/undo` a bad edit, and the checkpoint's snapshot file is gone
— no error, just "nothing to undo" (or partial revert of a multi-file checkpoint).

**Recommended change:** reuse `gc_orphaned_undo_files`' store-based liveness predicate
(or a batched `SELECT id FROM sessions` set) in the prune path. Keep the "current
session and just-reset predecessor are never candidates" rule (`:1081-1082`) — it's
correct and pinned.

**Acceptance criteria:** prune with a store containing sessions S1 (in store), S2 (not
in store), S3 (live session) removes only S2's undo file; runs twice (idempotent); undo
file for a resumable session survives.

**Validation:** extend `tests/test_store_gc.py` to plant undo files for store-backed
sessions **without stubbing the removal count** — the missing assertion is "the right
files survived", not "removed == N".

**Estimated effort:** Small (predicate swap + one real test). **Confidence:** High.

### [P1] Session-store corruption degrades silently to an *empty* store, and leaves WAL sidecars attached to the fresh DB

**Category:** Modification safety (data integrity)

**Location:** `nx/_sessiondb.py:84-95` (`_set_aside_corrupt` renames only
`sessions.sqlite`; the DB opens WAL at `:116`, so `sessions.sqlite-wal`/`-shm` stay in
place and re-attach to the recreated file — `_memindex.py:244` shows the codebase knows
to handle siblings, `_sessiondb` doesn't). The promised self-heal, `migrate_files`
(`:363-419`), imports only legacy JSON mirrors — no longer written
(`_sessions_fs.py:79-89`) — so for any modern store, set-aside = total transcript loss.
The only trace is `_log_event("error", …)` (`:348,353`); no user-facing notice, no
restore tooling.

**Observed behavior:** code-read verification of all four sites; busy/locked correctly
*not* treated as corrupt (pinned by `test_store_contention_p11.py`,
`test_sessiondb.py:117-127` — that part is solid). No live corrupt-DB repro constructed
(gap F-2).

**Durability impact:** the system's only copy of the user's conversation history — the
product's core asset — can vanish with a log line. The stale `.corrupt` file is the only
recovery artifact, and no command knows how to read it back.

**Failure scenario:** unclean shutdown during checkpoint leaves a malformed header; next
`nx` launch quietly creates an empty store; `/resume` and `/sessions` show nothing; the
user assumes they deleted their sessions; the WAL (which may hold the last transactions)
is corrupted by the fresh DB attaching to it.

**Recommended change:** (a) rename `-wal`/`-shm` with the main file; (b) surface the
event — a one-line notice at startup ("previous session store unreadable; archived at
`<path>.corrupt`") rather than log-only; (c) add `nx sessions --recover
<corrupt-path>` that opens the archive read-only and re-imports rows; (d) fix the
docstring in `_set_aside_corrupt` that still promises mirror-based re-import.

**Acceptance criteria:** a test that plants a corrupt main DB **plus** nonempty `-wal`,
opens the store, asserts all three siblings are gone from the active path and a
recoverable archive exists; startup notice is emitted through the user-visible channel;
`--recover` round-trips rows.

**Validation:** new tests in `tests/test_sessiondb.py`; existing contention/corruption
tests must stay green.

**Estimated effort:** Medium (a–b small; c is a new read-only CLI path).
**Confidence:** High on the mechanism, Medium on real-world corrupt+WAL frequency.

### [P1] The read-only sub-agent bash allowlist is bypassable for file writes and execution

**Category:** AI-agent readiness (guardrail)

**Location:** `nx/_subagent_bash.py:98,107-108`; allowlist `:53-64`. **Verified
directly:** `argv = parts[1:]` then the veto scans `argv[1:]` — so `find -delete …`
skips checking its first argument (compare the correct `parts[1:]` in `nx/_risk.py:271-273`,
the same veto's second home — an inconsistency, and the sibling's comment says these
lists are meant to match). Additionally not vetoed: `find -fprintf/-fprint/-fprint0/-fls`
(write files), `sort -o out.txt` / `--output=` (`sort` allowlisted), `rg --pre <cmd>`
(`rg` allowlisted; preprocessor executes an arbitrary binary), `git log
--output=<file>`.

**Observed behavior:** code-read, plus the escape shapes confirmed absent from the
table-driven test `tests/test_subagent.py:179-230,261-299` (grep for
`--pre|sort -o|--output|-fprint`: no match). The surrounding design is genuinely
fail-closed — metacharacters refused pre-`shlex` (`:90`), unknown heads refused
(`:103`) — the holes are in-argv flags of allowlisted binaries.

**Durability impact:** the read-tier contract ("can't reach the network or write files")
is the premise under which sub-agents run **forced-YOLO with no human prompt**
(`:115-118` says exactly this). A prompt-injected recon agent — say, from untrusted repo
content or a `web_fetch` result — needs one line, `rg --pre /bin/sh file.txt` style, to
write or exec inside the session's tree. Blast radius ≈ the parent's own run_bash, minus
approval.

**Failure scenario:** read-tier agent is steered by adversarial text in a scanned file
into `sort -o ~/.bashrc payload` (both paths in-tree-adjacent enough to matter once HOME
writes are reachable via git hooks/`--output`). Silent, unapproved write through a tier
documented as read-only.

**Recommended change:** fix the off-by-one (`argv` not `argv[1:]`); extend the veto to
the write-flag family per binary (`-fprint*`, `-fls` for `find`; `-o`/`--output=` for
`sort`; `--pre`/`--pre-glob` for `rg`); consider refusing any argument containing `=` on
binaries where that carries flags. Preserve the fail-closed default-on-unknown-binary
rule.

**Acceptance criteria:** each escape shape above is a table row in
`tests/test_subagent.py` asserting refusal; a meta-test asserts `_subagent_bash.py` and
`_risk.py` find-vetoes accept the same argv slice (the divergence is the root cause, not
any one flag).

**Validation:** `tests/test_subagent.py` additions; run serially with
`test_posture_readers_guard.py` neighbors per repo convention.

**Estimated effort:** Small. **Confidence:** High (code verified directly).

### [P1] Every quality gate is convention-only: no CI, no git hooks, nothing mechanically stops an unvalidated commit

**Category:** Engineering rigor (enforcement)

**Location:** absent by inspection: no `.github/`, no `.gitlab-ci.yml`, no non-sample
`.git/hooks/` entries (verified with `ls`). AGENTS.md/CONTRIBUTING state `pytest -n 8`,
`ruff check .`, `ty check nx`, and the parity-table regenerator as pre-commit
requirements; ~40 parity/contract tests exist precisely to catch silent drift (commands
table, env docs, schema invariants, floor budget) — and they only run when someone
remembers.

**Observed behavior:** all four gates were run in this audit and they pass; the
enforcement layer simply isn't there. Corroborating drift is visible even at this small
sample: `docs/test-architecture.md:5` claims "~192 files / ~2,180 tests" and that serial
collection is impossible — actuals are 222 files / **3015 tests** / serial-green in 92s
(both measured this audit). The three P1 defects above shipped *through* the test suite;
none shipped past a CI gate, because there is no second line.

**Durability impact:** in the repo's stated primary workflow — autonomous agents landing
commits in yolo mode — "tests passed" is an unverified agent claim. A single missed
`regen_commands_table.py` or a widened `_FLAKY_RERUN_NODEIDS` entry is undetectable
post-merge. The flake ledger's discipline (prune-then-re-add cycles) is exactly the
failure class an always-on runner would have caught faster.

**Failure scenario:** an agent edits `_commands.py`, skips the table regen because its
diff was "docs-adjacent," and commits; the README/REFERENCE drift persists for weeks
because the next agent trusts the docs — the repo's own AGENTS.md treats stale docs as a
defect, but nothing treats them as a *build failure*.

**Recommended change:** one minimal CI workflow: `uv sync` → `ruff check .` → `pytest -n
8` → `ty check nx` as advisory-report (matching its documented advisory status).
Optionally a pre-commit hook running the touched-file fast set. No new checks; wire the
existing ones.

**Acceptance criteria:** a PR that breaks any existing test or ruff fails a required
status check; suite wall-time in CI ≤ ~2 min; ty noise reported non-blocking.

**Validation:** self-evident; the existing suite is the payload — no test changes needed.

**Estimated effort:** Small (one YAML + uv cache). **Confidence:** High.

### [P2] Session-store schema evolution has no version gate, is forward-only and partly destructive — with a silent downgrade hazard

**Category:** Architecture (persistent-state evolution)

**Location:** `_sessiondb.py:223-324, 523-545` (shape ensured by `CREATE IF NOT EXISTS`
+ `ALTER` re-run on *every* connection, errors swallowed); `user_version` marks only the
FTS rule (`:47-58, 1274-1286`, unreadable marker = "current"); `_reclaim_stale_blobs`
(`:548-572`) runs **automatically once per process** from `_ensure_migrated` (`:496-520`)
and blanks `sessions.data` blobs the new format treats as meta-shells. No golden/fixture
DB exists in-repo (filesystem sweep, re-checked). `test_file_claims.py:237-242,622`
proves the hand-built-legacy-store pattern is understood — it just isn't applied to the
`sessions` table itself.

**Observed behavior:** code-read; old-binary → new-binary interaction reasoned from the
cited reclaim path, not executed.

**Durability impact:** `nx update` to a rollback (a documented user path — `_update.py`
exists) can destroy transcript blobs the newer binary will then auto-blank. And any
future *breaking* shape change has no mechanism to refuse an unknown-newer store — the
first such migration will be wrong for someone.

**Failure scenario:** user upgrades, the new format moves transcript authority to
`session_events`, something regresses, they run `pip install nx==0.16x`. The old binary
writes full blobs into `sessions.data`; the user upgrades again and the once-per-process
reclaim treats those as stale and blanks them. Transcripts gone.

**Recommended change:** promote `user_version` to a real schema epoch: (a) stamp it in
the current migration; (b) refuse (loudly, read-only) a store whose epoch is *newer*
than the binary's; (c) gate `_reclaim_stale_blobs` behind an epoch check + grace (e.g.
blank only rows whose `session_events` copy is verified present); (d) commit one golden
legacy `.sqlite` fixture and pin the upgrade path against it.

**Acceptance criteria:** opening a store stamped N+1 by binary N fails with an actionable
error, never mutates; reclaim test shows blobs survive when the events copy is missing;
golden-fixture migration test green.

**Validation:** `tests/test_sessiondb.py` + `tests/test_store_reclaim.py`; keep
`test_file_claims.py` migration pins green.

**Estimated effort:** Medium. **Confidence:** Medium-high (mechanism read directly; the
downgrade path is reasoned, not run).

### [P2] `/undo` rewrites user files at 0600, discarding mode; the undo stack persists through a bare exception swallow

**Category:** Modification safety

**Location:** `nx/_undo.py:413-419` — `_restore_one` uses `atomic_write_private`
(`_private.py:72-91`, forces 0600); contrast `_edit_tools.py:52-82`, which deliberately
copies `st_mode` "so as not to strip the executable bit." Persist side: `:320-334`
(dirty-flag, flush at turn boundary) + `:346-350` `except Exception: pass` — an
unwritable `~/.nx` loses the whole recovery story with zero diagnostics.

**Observed behavior:** code-read; restore *verification* (`after_sha`, refuse-on-drift,
including external deletion, `:77-117,422-465`) is real and well pinned
(`test_undo_compress_v2.py:188-300`) — the gap is fidelity of the restore write itself.

**Failure scenario:** agent edits `deploy.sh` (0755, group-readable in a shared
checkout); user hits `/undo`; file reverts to 0600 non-executable. CI step now
"permission denied" — attributed to everything except the undo that ran locally.

**Recommended change:** restore through the same mode-preserving path `_edit_tools` uses
(capture mode in the checkpoint if the original existed); emit a one-time visible warning
when `persist_undo_stack` fails, mirroring the repo's own fail-loud-vs-fail-open
distinction.

**Acceptance criteria:** test: edit a 0755 file, `/undo`, mode and exec bit intact;
test: monkeypatched failing persist surfaces exactly one warning and doesn't raise.

**Validation:** `tests/test_undo_compress_v2.py` extension.

**Estimated effort:** Small. **Confidence:** High.

### [P2] Turn-status honesty contract has a concurrency hole and an untested twin — process-global give-up cause plus a dead-branch trap in the text path

**Category:** Modification safety

**Location:** `nx/_turn.py:167-180` — `_LAST_GIVE_UP_CAUSE` is a module global written
by `_gave_up()` and read by two status summaries (`:1320,:1460`); concurrent sub-agent/
batch loops on one process interleave it (the `_recovery`/`_notices` frame machinery at
`:1022-1053` exists precisely for this hazard class and doesn't cover it — no test,
confirmed by grep). `nx/_turn_toolcalls.py:125` — `return delegated if delegated is not
None else TURN_DONE`: the call site was traced and the `None` branch is currently
unreachable (the exclusion pre-filter guarantees non-empty calls), so this is **not** a
live regression — but it re-arms the exact 0.154.7 dishonesty trap on the next edit to
the filter, it's the only remaining `else TURN_DONE` of its family, and no test drives
the text-protocol forced-final handler at all (grep: `_handle_forced_final_text_calls`
appears only in `nx/`).

**Failure scenario:** batch run with parallel tasks: task A gives up on a malformed
stream, task B gives up on a reasoning stall; B's JSONL summary reports A's cause — an
operator triages the wrong failure class. Later, someone relaxes the exclusion filter and
a headless eval scores a silent non-answer as "done" with green tests.

**Recommended change:** move the cause into the existing per-loop recovery frame (or a
TLS slot with the same reset contract as `_POSTURE_TLS`); replace the `else TURN_DONE`
with `_gave_up("forced-final text path reached no-call state")` as a tripwire; add one
`test_turn_loop_e2e.py` case driving the text forced-final handler.

**Acceptance criteria:** two interleaved give-up loops assert distinct causes; the
tripwire is unreachable by assertion in existing tests; new e2e case passes.

**Validation:** `test_turn_loop_e2e.py`, `test_forced_final_recovery.py`; state-reset
contract fixture must drain the new home (this is the repo's documented "hidden home →
fix the fixture centrally" rule).

**Estimated effort:** Small. **Confidence:** High for the global's interleaving; the
dead-branch part is a maintenance trap, stated as such.

### [P2] Provider wire fidelity is untested: fake streams are constructed from the parser's own assumptions

**Category:** Modification safety (verification independence) / AI-agent readiness

**Location:** `nx/_stream_parse.py` (815 L) and `nx/_text_fallback.py` (511 L) are
exercised by three fake tiers, none independent of the implementation: hand-built chunk
objects (`test_stream_parse_qwen_separators.py:61-72`), attribute-stub clients, and one
byte-framing test (`test_sse_framing.py:61-67`) whose template is a single-choice delta
with no `finish_reason`, no usage frame, no role delta, no legacy `function_call`. No
captured-provider corpus exists anywhere (filesystem sweep; confirmed no golden files
in-repo).

**Durability impact:** this is the audit's verification-independence question in sharp
form: a provider emitting tool-call arguments under a different key/shape passes the
entire 3015-test suite and silently truncates args in production — the *documented*
symptom that motivated `test_sse_framing.py:9-13` in the first place. Every recovery
ladder built on top of the parser inherits the blind spot.

**Failure scenario:** user points nx at a new gateway; tool args arrive complete on the
wire but the parser's delta-key assumptions drop them; the malformed-args breaker fires;
sessions "work" with degraded tooling and nobody can say whether nx or the gateway is
wrong.

**Recommended change:** a small golden corpus — a handful of raw SSE bodies captured (or
hand-written to provider spec, clearly labeled) from vLLM, llama.cpp/OSS, and one hosted
API, covering: multi-choice, finish_reason frames, usage-at-end, role-only deltas,
malformed-JSON tool args. Replay each through the real `httpx.MockTransport` path (the
`test_sse_framing.py` pattern, widened) asserting the parsed turn is byte-complete and
`dropped_lines == 0`.

**Acceptance criteria:** one test file, per-trace parse assertion; a new provider shape
can be added in one file without touching fakes.

**Validation:** runs in Tier A (no threads, no net); wall-time negligible.

**Estimated effort:** Small-medium (trace authoring is the work). **Confidence:** High on
the gap; Medium on which providers matter to the user base (gap F-3).

### [P2] Several parity guards self-certify: the env-var "actually read" scan includes `tests/`, and the allowlist-staleness check matches its own file

**Category:** Engineering rigor

**Location:** `tests/test_docs_parity.py:128-150` — "documented ⇒ actually read" walks
`nx/` **and `tests/`** (`:141`), so a REFERENCE var referenced only by a test's
`monkeypatch.setenv` passes — precisely the `NX_TEMPERATURE/NX_TOP_P` failure mode its
docstring says it exists to catch (and
`test_sampling_params_reach_the_request_payload:165` exists because exactly that shipped
before). `tests/test_env_docs_parity.py:145-150` — the "prune dead allowlist entries"
check substring-scans `.py` files including *itself*, whose `_ALLOWLIST` literal contains
every key: the assertion can never fail. **Verified directly** (both files read). Also:
README's env table is covered by no test (only REFERENCE is opened by either parity
file), and `BUDGET_TOKENS` has ~700 tokens of unbilled headroom with the near-cap signal
as a `print`, not an assert (`test_initial_context.py:235-237`).

**Failure scenario:** a refactor deletes the only `os.environ` read of a documented
tunable; the var keeps its REFERENCE row, a test still sets it, suite green — users flip
a documented switch that does nothing. This is the repo's own highest-value invariant
class (docs are machine-verified; agents trust them) with a hole cut in the verifier.

**Recommended change:** forward scan (`documented ⇒ read`) over `nx/` only; staleness
scan over `nx/` only (or exclude the parity test files); promote the SOFT floor print to
an assert with an explicit ledger-bump escape hatch; add README env-row coverage to the
existing REFERENCE regex pass (one more `re.findall` on README's table — cheap, and
`test_commands_registry.py` already models the pattern).

**Acceptance criteria:** deleting an env read from `nx/` while leaving the test stub
breaks the suite; the staleness assertion demonstrably fails on a planted dead key
(anti-vacuity pin, mirroring the good
`test_pending_slots_are_derived_from_the_store:171` pattern already in the repo).

**Validation:** `test_docs_parity.py`, `test_env_docs_parity.py`,
`test_initial_context.py`.

**Estimated effort:** Small. **Confidence:** High.

### [P2] The state hub (`_state.py`, 152 shared names, 90 dependents) plus universal lazy-import convention make local reasoning grep-dependent

**Category:** Architecture / Understandability

**Location:** measured in this audit: **2,121 function-local imports across all 149
modules** (worst: `_dispatch` 79, `__init__` 77, `_repl_settings_cmds` 61, `_turn` 51);
`_state.py` holds 152 distinct `st.<NAME>` globals with no per-name reader registry
(grep-scatter; the proxy at `__init__.py` bottom widens access to `nx.<NAME>`); one real
cycle (`_session_save ↔ _sessions_fs`, lazy-imported into silence). The mechanism is
*deliberate and documented* (module-split necessity + conftest rewinds), and the
conftest-baseline gotcha (every lazily-imported module must be registered) is a standing
per-module tax — confirmed by the growth-ledger prose in `tests/perf/baselines.json`
where each release re-justifies the module/entry budget.

**Durability impact:** the cost compounds where three conventions intersect: finding a
name's writers is grep; a new module costs an edit in `conftest.py` + a ledger bump; and
the two home-hygiene violations in this report's other findings (`_LAST_GIVE_UP_CAUSE`,
`_TOOL_LIST_CACHE` key coupling at `_tool_gates.py:141-202`) are exactly the defects this
shape breeds. `model_turn` at 529 lines with load-bearing-but-unpinned guard ordering
(`_turn.py:879-976`) is the concentration endpoint.

**Failure scenario:** a contributor adds a `_PENDING_*`-class slot to `_state.py`; the
derived-oracle sweep catches it (good — 0.154.2's fix), but nothing catches a *scalar*
like `_LAST_GIVE_UP_CAUSE`, because scalars aren't name-swept; six weeks later the
concurrency bug in the finding above is born from the same root.

**Recommended change:** no rewrite. Two cheap moves: (1) extend the state-reset contract
test's derived sweep to a declared `SCALAR_SLOTS` tuple in `_state.py` — every
module-global *mutable* scalar declared there is swept and pinned by construction,
making "hidden home" a compile-ish error; (2) per-concern state grouping in `_state.py`
section comments already exists — add the `grep` target as a machine-readable `OWNERS`
dict and a test that every `st.<CAPS>` write appears under exactly one section. Both are
~100 lines and fit the existing contract-test vocabulary.

**Acceptance criteria:** planting a new module-global scalar without declaring it fails
`test_state_reset_contract.py`.

**Validation:** existing contract test file; full suite for drain regressions.

**Estimated effort:** Small-medium. **Confidence:** Medium (the architecture is a
reasoned trade, not an accident — the finding is the *un-swept scalar class*, not the
hub itself).

### [P3] Three low-severity hygiene findings, one line each

1. **PTC sandbox is invisible and its docstring lies.** `_ptc_exec.py:33-36` claims "No
   unshare/seccomp is added"; `:643` installs seccomp+landlock+NNP via `preexec_fn`, and
   every layer is `try/except Exception: pass` (`:73-109`, `# pragma: no cover`) — on
   macOS/old kernels the sandbox silently doesn't exist and no status line says so. Fix:
   reconcile the docstring, surface sandbox state in `/toolgates`-style diagnostics.
   (Both locations read directly.)
2. **No tool-result type contract.** Nothing enumerates `DISPATCH` and asserts each body
   returns `str` (or a declared parts list); per-tool isinstance checks exist piecemeal.
   A body returning `bytes`/`dict`/`None` first surfaces deep in the loop or in a session
   row. Fix: one parametrized contract test over `m.DISPATCH` with a stubbed-call matrix
   — matches the existing schema-invariant idiom.
3. **Stale durability-doc claims.** `docs/test-architecture.md:5,148` still says ~2,180
   tests and that serial collection is impossible; serial now passes in 92s (measured).
   Update, and correct the `_set_aside_corrupt` mirror-reimport promise (finding 2).

## D. Systemic observations

1. **"Verified by convention" is the load-bearing assumption.** The repo has unusually
   good *verifiers* — derived oracles, byte-exact doc regeneration, honest-status
   contracts, adversarial fence tests — and no runner. P1-4 is upstream of most other
   P2s: the vacuous staleness check would likely have been caught by a CI rerun-on-diff;
   the parity holes get exercised every time someone edits docs under agent pressure. Fix
   the runner before adding verifiers.
2. **Legacy-layer ghosts cause live defects.** 0.101's JSON mirrors, un-written but still
   *trusted as liveness oracles*, caused the prune data loss (P1-1) and the false
   self-heal promise (P1-2). The repo's compatibility layers have no expiry mechanism —
   the same pattern CONTRIBUTING warns about for compatibility shims ("no clear
   ownership"), now with a data-loss invoice. A "retire by" stamp on compat paths would
   make the next ghost self-announcing.
3. **Fail-open is applied consistently but labeled inconsistently.** Fail-open is a
   stated contract for genuinely recoverable paths (obsstore, memory store degrade,
   worktree TTL) — correct. The three persistent-state findings share the inverse
   pattern: a fail-open swallow on a path where the lost artifact is unrecoverable
   (autosave drop `_sessions_fs.py:346-347`, undo persist `_undo.py:346-350`,
   corrupt-store `_log_event` only). The cheap general remedy: any swallowed exception
   whose loss is unrecoverable must emit exactly one user-visible notice — the repo
   already does this for the fence and give-up paths; extend the rule to stores.
4. **Single-source-of-truth lapses are argv-slice and key-set level, not module level.**
   The `_risk.py`/`_subagent_bash.py` veto divergence, the `_TOOL_LIST_CACHE` gate-vector
   coupling, and the two parity scans' file-walk scopes are all small drifts of *shared
   predicates duplicated at two homes*. This codebase's scale is safe for duplication only
   when a meta-test pins equality; three of the findings above are the interest on
   unpinned duplicates.

## E. Recommended remediation order

1. **P1-1 prune undo GC** (data loss today; smallest fix; test-first).
2. **P1-3 read-tier allowlist vetoes** (off-by-one + flag set; small).
3. **P1-4 minimal CI** (costs one workflow file; protects everything below and every
   future agent commit — do it before, not after, the durable-state work).
4. **P1-2 corrupt-store recovery + WAL siblings + user notice** (needs the notice
   mechanism P2-undo will reuse).
5. **P2 undo mode-preserving restore + fail-loud persist** (rides the same `_undo` file
   as step 4's notice pattern).
6. **P2 schema epoch + golden fixture + reclaim gate** (design work; benefits from the
   corrupt-store tests written in step 4).
7. **P2 parity-scan scope fixes + floor assert** (cheap, independent).
8. **P2 golden SSE corpus** (independent; schedule when provider breadth matters).
9. **P2 give-up-cause frame + forced-final tripwire + text-path e2e case.**
10. **P2 state-scalar sweep** (institutionalizes the rule the other fixes just taught).
11. P3s opportunistically; native/text handler dedup — fold into the next touch of
    `_turn_toolcalls.py`, since finding 9's tripwire test gives it a safety net.

No rewrite required anywhere in this report; every fix is ≤ medium and rides an existing
test seam.

## F. Verification gaps

- **F-1 Flake reality:** each suite mode was run once. The `_FLAKY_RERUN_NODEIDS` ledger
  (43 entries) and the 0.155.x open trio mean flake rate was taken on the repo's own word.
  Resolve: 10 consecutive `-n 8` runs recorded in CI (free, once E-3 lands).
- **F-2 Corrupt-store+WAL:** the P1-2 resurrection/malformation hazard is reasoned from
  SQLite WAL semantics and code reading, not reproduced. Resolve: the acceptance-criteria
  test in P1-2 (plant corrupt header + nonempty `-wal`).
- **F-3 Provider breadth:** the golden-corpus finding's priority depends on which
  gateways the user actually runs; invisible from the repo. Resolve: user input, or
  `~/.nx` source configs (not read — credential-adjacent).
- **F-4 Fence bypass class:** the interpreter/`eval` escapes of the bash fence were
  enumerated by code reading (acknowledged in `_fence.py:50-51` as "backstop, not
  sandbox" — accepted design given approval sits behind it, so not filed as a finding).
  If YOLO posture is used routinely, that accepted risk silently becomes live. Resolve:
  confirm the approval posture assumption for real usage.
- **F-5 Perf/production behavior:** RAM stress harnesses, prompt-cache byte-stability in
  real provider traffic, and obsstore quotas were inspected as code only; `ty`'s 55
  diagnostics were counted but not triaged against the "~130 pre-existing" policy.

---

Bottom line: `nx` is a durably *designed* codebase — the invariants, budgets, and
contracts are what most teams wish they had. Its durability is currently capped by three
concrete defects that the excellent test suite doesn't see, and by the absence of any
mechanism that guarantees the excellent test suite actually runs.
