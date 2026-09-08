# nx Roadmap

Working plan for the next evolution of `nx`, produced from a full-codebase audit
(six parallel area audits over `nx/*.py`, ~21.9k LOC, 213 commits, v0.39.2).
Every claim below is cited to a file/line; items marked **[v]** were re-verified
by reading the cited lines directly. Effort is S/M/L = roughly under a day / a
few days / a week-plus.

The audit's headline is not "this codebase is behind". It is: **nx has built an
unusually deep self-healing and context-frugality stack, and the next gains come
from making that stack internally consistent — cache-safe, correct-by-construction
at the boundaries (paths, secrets, resume, prompt cache) — rather than from
adding new agent features.** Feature parity with heavier CLIs (web tools, LSP,
native Anthropic) is genuinely missing, but it ranks below the correctness work
because the missing correctness actively costs tokens or trust on every run.

## Where nx stands (measured)

| Dimension | State |
| --- | --- |
| Test suite | **1023 passed in ~10 s** (`-n auto`), fully offline, xdist-safe. 90 test files, ~896 `def test_` functions. Zero skips. |
| Per-turn floor | **2886 tokens** (SYSTEM 375 + 18 active tool schemas 2510). The suite's own 2500-token tripwire now fires as a *warning* on every run — the budget has silently been exceeded. |
| Largest tools by floor cost | `spawn_agent` 396 tok, `search_files` 279, `plan_edit` 229, `edit_file` 207 |
| Dependencies | 3 runtime (`httpx`, `rich`) — still deliberately lean |
| Config surface | ~119 distinct `NX_*` vars; **30 undocumented** (all of `NX_RECOVERY_*`, `NX_BASH_POLL_SECONDS`, `NX_MAX_UNDO`, …) |
| Docs touchpoints | 4 hand-synced command lists; drift already live **[v]** |
| CI | none (no `.github/`, no CHANGELOG) |
| Eval corpus | none — `nx run` batch harness exists, but zero golden tasks/transcripts checked in |

## Guiding themes

1. **Honor the prefix cache we already measure.** nx counts `cached:` in the
   footer and documents append-only staging *precisely* to protect server KV
   caches (`_compress.py:48-52`) — and then every recovery nudge rewrites an
   earlier user message (`_compress.py:37-45`)**[v]**, and `/resume` swaps
   SYSTEM wholesale. The self-healing stack is a cache-buster on exactly the
   turns with the largest prompt. This is the single biggest lever on the
   project's own thesis.
2. **Turn opinions into invariants.** Path fences, symlink handling, breaker
   checks, doc/command parity, and resume state are all "best-effort prose + a
   model's judgement" today. Where a deterministic check exists, prefer it.
3. **One implementation per job.** Two resume paths, two list-and-print paths,
   two repair-candidate call sites, two forced-final handlers, four full-store
   memory scans, three termios save/restore implementations. Divergence is
   already user-visible (images 400 on `/resume`)**[v]**.
4. **Make the recovery stack observable and thread-safe.** It is the product's
   differentiator and it runs on process-wide globals shared with sub-agent
   threads.
5. **Keep the floor flat.** Anything added to `TOOLS`/`SYSTEM` is rent on every
   turn; budget it, don't apologize for it later.

---

## Phase 1 — Consistency and correctness (near-term minors: 0.40–0.42)

Small-to-medium, high certainty, each independently shippable.

### 1.1 Prompt-cache-safe recovery **[S/M]** — *highest ROI in the repo*
Every recovery rung injects its nudge by rewriting the last user turn in place
(`_compress.py:37-45`, called from `_turn.py:822, 1049, 1156, 1173, 1187, 1402,
1535, 1571`)**[v]**. `_flush_pending_background_notice` (`_compress.py:48-52`)
and `_capture_last_prompt_tokens` (`_turn.py:219-231`) both state the opposite
policy: append a tail message so earlier bytes — and the server's prefix cache —
survive. `tests/test_length_continuation.py:142-148` pins the rewrite, so the
cache-hostile behavior is the *tested* behavior.
**Ship as:** append-only `[Runtime note: …]` tail messages everywhere; flip the
pinning test; add a regression asserting a nudge never mutates a message older
than the turn boundary. Watch for the model's echo-of-nudge path in
`RUNTIME_NOTE_RE` cleanup logic.
**Payoff:** recovery storms stop re-billing the whole prompt; `cached:` stays
green on llama.cpp/vLLM/SGLang exactly when recovery fires.

### 1.2 Thread-isolate recovery state **[M]**
Deadlines already got a `threading.local` (`_turn.py:1594-1617`) and async-agent
quiet-ness got per-thread slots, but `_LAST_PARTIAL_TEXT`, `_TALLY_*`,
`_STALL_EPISODE_ACTED`, `_LAST_PROMPT_TOKENS`, `_NATIVE_TOOL_CALLS_CONFIRMED`,
`_MALFORMED_TOOL_*` are process globals (`_state.py:176-222`). Async sub-agents
run their own `_run_model_turn_loop` on threads and write them: a sub-agent's
prompt size can trigger the **parent's** autocompress (`_turn.py:1752-1755`),
and a sub-agent's malformed args can remove a tool from the parent's tool list.
**Ship as:** a `RecoveryContext` object owned by one loop; keep `st.*` as a thin
facade for tests. **Payoff:** unlocks `nx run --parallel` and heavier fan-out
without cross-session misfires.

### 1.3 Breaker integrity + one dead config landmine **[S]**
- `run_tool` checks `_MALFORMED_TOOL_BROKEN` against the **emitted** name
  (`_dispatch.py:476`)**[v]**; alias/similarity resolution happens *after*
  (`_dispatch.py:487-499`)**[v]** and re-dispatches without re-checking — a
  broken `multi_edit` is reachable as `multi_edits`.
- `_argrepair._schemas()` repairs against `TOOLS`, not `_active_tools()`.
- `st.MALFORMED_TOOL_BREAKER_LIMIT` is `2` in `_state.py:175` and `3` via
  `__init__.py:489`**[v]**; `test_breaker_ergonomics.py` only passes because of
  import order. Delete the `_state.py` default.
- Breaker state is per-process; switching to a healthy endpoint doesn't restore
  dropped tools. Make it per `(source, model)`.

### 1.4 One resume path **[S/M]** — *live 400-class bug*
`--resume` calls `_deserialize_image_parts` (`_repl.py:180`)**[v]**; `/resume`
loads store-form messages (`_sessions_cmd.py:35-40`)**[v]** and never expands
them — `{"type":"image_ref"}` parts go straight to the API, which the code's own
comment says 400s. Same split-brain: provider routing is re-applied only on the
CLI path; **cwd is saved but never restored** (zero `os.chdir` in `nx/`)**[v]**;
undo stack, toolcalls mode, breaker, budget, footer state are not persisted
either. `revision` is written and read only by a test.
**Ship as:** single `resume_session(id, state)` + a parity test for both entry
points; restore images, provider, cwd.

### 1.5 Docs/touchpoint parity, generated not grepped **[S/M]**
CONTRIBUTING documents four hand-synced lists. Drift is live **[v]**:
`/toolcalls` missing from `_HELP_COMMANDS`; `/resmon` missing from
`SLASH_COMMANDS` and the README; `/ctx all` is advertised by the hint at
`_repl_settings_cmds.py:337` with no branch implementing it; 7 commands absent
from the README table; `NX_TEMPERATURE`/`NX_TOP_P` documented in
`__init__.py:458-463` and read **nowhere**.
**Ship as:** one command registry (name, help row, menu entry, arg options) that
generates `_HELP_COMMANDS`/`SLASH_COMMANDS`, plus a parity test that fails on
the current drift; implement or delete `/ctx all`; implement or delete
`NX_TEMPERATURE`/`NX_TOP_P` (the former is genuinely wanted for local evals).

### 1.6 Private sinks + first-line secrets hygiene **[S]**
Sessions/images/memories are correctly `0600/0700` **[v]**, but bg-job logs are
created by shell redirection after `os.makedirs` with no mode
(`_bash_jobs.py:149`)**[v]** and `~/.nx/overflow/*.txt` spills the **complete
untruncated tool result** via plain `open()` (`_overflow.py:69-78`)**[v]** —
both are group-writable on a normal umask **[v]** while `_logging.py`'s own
comment states the intent not to persist unredacted payloads. `_sanitize_tool_result`
is size-only. The one secret check is input-side, default-**off**, and prompts
rather than redacts.
**Ship as:** `_open_private()` for bg logs + overflow spill; redaction pass in
`_sanitize_tool_result` (high-entropy + known-header patterns); flip
`NX_BASH_PROMPT_SECRETS` default after one release of warning-only mode.

### 1.7 Floor budget: reset the tripwire honestly **[S/M]**
`spawn_agent` alone costs 396 tok/turn — ~13% of the whole floor — and its
description is a multi-paragraph README embedded in a schema. Tools total 3404
tok across 21 schemas. Options, in order of preference: trim the big three
descriptions to invariant sentences + move detail to a one-time tail note
(precedent: `BACKGROUND_GUIDANCE`); consider a second-tier schema set offered
after first use (precedent: `wait_background`); make the floor test an
enforceable budget with a recorded, justified number instead of a warning.

### 1.8 Two-tier overflow recovery + window-aware sizing **[S]**
Overflow gets exactly one fold (`_turn.py:1724`, `context_retries < 1`); if the
kept tail alone exceeds the window the run dies. `max_tokens` is a fixed 16000
regardless of remaining window (`_compress.py:89`), which manufactures both
`finish_reason=length` cuts and 400s. `Source.tokenize_count()` exists
(`_sources.py:243`) and is unused for pre-flight. **Ship as:** escalate the fold
(halve keep, then evict largest tool results via spill), clamp request
`max_tokens` to `window - prompt_estimate - margin`.

---

## Phase 2 — Trust and durability (0.43–0.46)

### 2.1 Hard write fence in `run_tool()` **[M]**
There is **no path check anywhere** between arg validation
(`_dispatch.py:514`) and `fn(**args)` (`:637`). `_risk.py` is a textual
classifier; `_extract_action_path` parses only `write/edit/delete …` prefixes,
so `apply_patch` and `sed_edit` — the two multi-file writers — get
`path_scope:"unknown"` and no path judgement at all. `_atomic_write`
deliberately writes *through* symlinks (`_edit_tools.py:45-63`), so a committed
`config -> ~/.ssh/authorized_keys` converts an approved in-project edit into an
out-of-project write (classifier `realpath`s at judgement time; write resolves
fresh — TOCTOU). `delete_file`'s guard is basename-only.
**Ship as:** a `NX_WRITE_ROOTS`-style fence enforced in dispatch (default:
project root + `NX_HOME`), realpath-reject for writes outside it, deny-list
(`.git/hooks`, `~/.ssh`, `~/.aws`, dotfile rc paths), and an explicit
"following symlink out of the fence" refusal that keeps the link-farm use case
when it stays inside.

### 2.2 Undo correctness + durability **[S]**
`delete_file` snapshots twice (body at `_edit_tools.py:343` + dispatcher at
`_dispatch.py:588`)**[v]** — two `/undo` presses to pass one delete, at half
depth; its "changed during approval" guard compares two stats taken *after*
`_confirm` (tautology); `sed_edit` checkpoints before its staleness abort.
Stack is memory-only, so a crash loses the recovery story that YOLO/batch
postures assume. **Payoff:** `/undo` means what users think it means.

### 2.3 Session store: index consolidation and the per-save rewrite **[M→L]**
Every autosave deep-copies the whole transcript, re-parses the existing file,
rewrites it, and re-parses **every session file in the directory** for image refs
(`_sessions_fs.py:350-374`, `:386`) — quadratic in session length. Two indexes
(`.index/` shards + FTS5) never reconcile; stale FTS rows are never pruned, so
`/sessions <query>` can offer a deleted id. `NX_SESSION_IMAGES=0` (the
documented rollback) disables image GC entirely, orphaning every payload.
`/export` is one-way markdown that always prints "(untitled session)" —
`getattr(state, "title")` can never be non-None because `_ReplState.__slots__`
has no `title`**[v]** — and writes live-form messages (inline base64) into it.
**Ship as:** (a) ref-count or index-driven GC + GC-on-delete + size quota;
(b) fold both indexes into one SQLite updated inside `_write_session`, pruning
deleted ids; (c) then, if profiling still demands it, per-turn chunked
transcripts. Add `nx sessions --prune`, bytes-in-listings, JSON export/import.

### 2.4 Compression: from turn counts to token targets **[M/L]**
`keep` is a message count (`COMPRESS_KEEP=2`, auto = `body_len//3`); the previous
summary is just another head message, so folded history gets **re-summarized
every fold** — paid for twice, then thrice. Summaries are non-streaming on the
*same* model with no `max_tokens`. Tool-result bodies get a flat `[:2000]`.
**Ship as:** summarize-to-token-target with a reused summary prefix (cache-safe
by construction, aligning with 1.1), tool-result-aware eviction ordering (spill
before summarize), optional cheaper-model routing for the fold call.

### 2.5 Cost correctness before new price features **[S/M]**
Per-source pricing while `/source <name> <model>` pins arbitrary models; the
`/cost` scoreboard loses `by_source` on resume; `reasoning_tokens` reads
`output_tokens_details` only (its own docstring says `completion_tokens_details`)
so the column is silently 0 on OpenAI-shaped backends; risk-classifier and all
probe calls (including the `max_tokens=10_000_000` ctx-overrun POST,
`_sources.py:503-509`) are never `_record_usage`'d; `chars//4` estimates bill
into server totals with no provenance flag; unknown cache is billed as uncached.
**Ship as:** per-model price keys, account side-calls + probes, provenance tags
(`measured|estimated`) in `/cost`.

### 2.6 Memory retrieval v2 **[L]**
`memory_search` is substring, all-terms-must-match, over a full read of every
note, with the store re-scanned four separate ways across `_memory_cmd`/
`_memory_optimize`. `/memory list` reads the **global dir only**
(`_memory_cmd.py:262`)**[v]** — a project-only repo reports "no memories saved
yet". Filename creation is `os.path.exists`-then-write (TOCTOU under the merge
workers), archive stamps are second-granularity across processes. The session
note shows 5 titles/scope, so past ~10 notes discovery is only the keyword path.
`/memory optimize` reviews only the **newest 120** notes — the best retirement
candidates are never seen.
**Ship as:** one scanner + FTS5 index over both scopes (shared with the 2.3b
index), fix `/memory list` scope and the writer race first (S), embeddings only
if an embeddings-capable source is configured (provider item P3-3).

---

## Phase 3 — Capability expansion (0.47+, only after Phase 1–2)

### P3-1 Sandbox tier for `run_bash` + writes **[L]**
Zero occurrences of `bwrap/unshare/landlock/seccomp/setrlimit` repo-wide. Today's
confinement is `setsid` (terminal-signature protection, not containment), full
env inheritance (including every `~/.env` key) with no `env=`, no `cwd=`, no
rlimits, no egress policy, no output caps below `NX_BASH_MAX_SECONDS` (default
off). README's own batch recommendation (`--approval high`) means unapproved
shell on a CI runner with API keys in env and world-readable output logs.
**Design as:** optional `bwrap`/`landlock` profile (read-only repo + tmpfs,
deny `~/.ssh`/`~/.aws`, optional no-network), scrubbed `env=` with an explicit
passlist, `cwd` pinning, `RLIMIT_AS/CPU/NOFILE`. Gate per-tier: YOLO without a
sandbox should become *harder*, not louder.

### P3-2 Honest timeout + turn-level transient retry **[M]**
`httpx.TimeoutException` mid-stream prints "discarded partial output" and
returns **`TURN_DONE`** (`_turn.py:1377-1387`)**[v]** — a mid-task timeout is
reported as a completed answer, and in batch mode a `done` line with truncated
output. `TURN_API_ERROR` is terminal. **Ship as:** backoff-retry 429/5xx at the
step boundary; timeout → a distinct status that scores `error`/`timeout`, never
`done`.

### P3-3 Provider abstraction: Anthropic-native + Responses API **[L]**
Only `/chat/completions` + `/models`, Bearer-only auth, `choices[0]`,
`data:`-only SSE. This gates: explicit prompt-cache control (`cache_control` —
nx only *consumes* `cached_tokens` today), thinking budgets beyond the
OpenRouter-shaped `{"reasoning":{"max_tokens":N}}`, `tool_use` semantics,
`parallel_tool_calls`, structured outputs (`response_format`/`json_schema`, or
vLLM `guided_json` — today grammar is llama.cpp-GBNF-only and `NX_BACKEND=vllm`
merely *disables* knobs). Ranked after the correctness phases because a
translation proxy covers the feature, while cache-safety and write fences
cannot be proxied in.

### P3-4 Grammar/prompt/parser: one dialect **[M]**
The grammar constrains the legacy JSON-in-tag body (`_grammar.py:126-151`)**[v]**
while `SYSTEM` and the correction nudge teach the XML dialect
(`__init__.py:404-406`)**[v]** — on a grammar-capable source the model is
force-constrained into a shape its prompt never asked for. Also: the builder
requires required-before-optional properties, policed by a test instead of an
assertion in the builder; `_special_tokens.py` hardcodes DeepSeek's `｜`/`▁`
signatures; repetition detection fires only for chunks ≤40 bytes (tuned for
Qwen separators). **Payoff:** the repair/breaker/nudge machinery gets less to do.

### P3-5 Closed-loop verification: `run_tests` tool **[M]**
`NX_VERIFY_CMD` is one global free-text command, Python-only `ast.parse` + exit
code, 20 s settle, off in batch, no target/language parsing, no scoped re-run.
A `run_tests(scope)` tool (detect pytest/cargo/npm/go, run targeted, parse
structured failures, offer them as a tail note) is the loop every edit-heavy
session wants and the verification tier write-sub-agents need.

### P3-6 Git workflow tool **[S/M]**
`git commit/add/worktree` exist only as classifier prose. Committing is raw
`run_bash` reviewed as nothing; `apply_patch` has no stage/commit surface. A
`git_commit` with staged-diff review (and a pre-commit secret scan from 1.6)
removes the most common write-without-review path.

### P3-7 Read-back UX: output pager + `/diff` **[L]**
Tool output preview is hard-capped at 8 lines (`_dispatch.py:693-696`);
`/tooloutput last` replays only the newest result; diff context is 2 lines
printed once; no `$PAGER`/`EDITOR` anywhere; no post-write diff; no syntax
highlighting on diffs or tool bodies (rich is already a dependency, used only
for the answer panel). The biggest *trust* gap for daily drivers.

### P3-8 Chatbox editor completion **[M]**
The keymap handles 9 control bytes: no Ctrl+R search (history is a 100-entry
unsearchable file), no Ctrl+W/K, Ctrl+A/E, yank. Height is never read (only
`.columns` at `_chatbox.py:436`); no SIGWINCH; truecolor emitted
unconditionally (no `NO_COLOR`/`TERM=dumb` path); approval prompt loops on a
paste (drains ≤8 bytes then re-prompts per byte) while the chatbox handles
bracketed paste properly. Prerequisite for most of this: split `_chatbox_raw`
(400 lines, 10 `nonlocal`s) into editor model + renderer.

### P3-9 Config surface: `nx config` over the env layer **[M]**
~119 vars, four incompatible boolean idioms, no name sanitization (a source
named `my-llm` yields an unsettable `NX_SOURCE_MY-LLM_*`), and README literally
tells users to grep source. Policy stays "env vars, not config files" — add a
`nx config list|get|set|doctor` view over the same `~/.env`, plus a `/status`
snapshot (source, model, caps, floor, fence, approval, budget).

### P3-10 Capability manifest per (source, model) **[M]**
`src._capability` isn't keyed by model; vision is gated on a session flag
(`NX_VISION and _NATIVE_TOOL_CALLS_CONFIRMED`), not model ability; no probe TTL;
thinking kwargs are a 2-entry hardcoded list. Fold probes into a cached,
model-keyed manifest shown by `/source` — this deletes a third of the opt-in
archaeology (`NX_VISION`/`NX_NATIVE_TOOLS`/`NX_GRAMMAR`/`NX_BACKEND`) for
typical users.

### P3-11 Provider fallback ladder **[M]**
Five retry layers, all within one endpoint; OpenRouter `allow_fallbacks` is
deliberately off. A declared `FALLBACK=<source chain>` per source, used on
429/5xx/overrun with the switch announced in the footer, is what unattended
`nx run` needs.

### P3-12 Multi-root / worktree isolation **[L]**
One global `st.PROJECT_ROOT` computed at import; no `cwd`/`root_dir` on any
tool; `nx run --parallel N` mutates one shared tree and shares the async-note
channel. Prerequisite for safe parallel batch and write-tier sub-agents.

### P3-13 Web tools behind an egress policy **[L]**
No fetch/search tool exists; the only egress "policy" is classifier prose, and
the classifier forces `curl` to high, so web access is simultaneously loud and
unconfined. Land after P3-1 so egress has somewhere to be enforced.

### P3-14 Eval harness: golden transcripts + scored batch runs **[M]**
The `nx run` harness is benchmark-shaped (JSONL results, exit codes, per-task
token accounting) but there is no corpus: zero golden tasks, no recorded
transcripts, no regression scoring. A small corpus (10–20 tasks: refactor,
bugfix-from-issue, multi-file docs sweep, "recover from a hostile stream") with
recorded stub servers is what turns the recovery stack's heuristics from
anecdote-driven into measured. Arguably the highest-leverage item in this phase
for the *project's own* velocity — it is also where the 2500→2886 floor drift
would have been caught by a test rather than a warning.

---

## Explicit non-goals

- **A config-file format.** Env vars stay the substrate (CONTRIBUTING policy);
  P3-9 is a *viewer*, not a new format.
- **Full TUI (alternate screen, mouse everywhere, panes).** nx renders into
  scrollback deliberately (`_chatbox.py:385-389`); keep it, fix height-awareness
  and capabilities instead (P3-8).
- **Dropping the dependency-minimal posture for schema/SDK conveniences.** The
  vendored thin client exists for import-time RSS; Anthropic support (P3-3)
  extends it rather than adopting the SDKs.
- **A second memory store or note format.** Evolve retrieval (P2.6), not the
  on-disk notes.
- **Silent auto-fixing of user-visible content beyond the current garble
  guard.** The repo's philosophy is diagnosis-not-rewrite for model commands
  (`test_broken_command_still_fails_loudly`); the heredoc pre-flight is the
  model to follow.

## Suggested ordering (semver per AGENTS.md)

**STATUS (0.42): Phase 1 is DONE.** Shipped: 1.1 (db59173), 1.3 (a5f7a5b),
1.4 (3023b04), 1.6 (7b15594 — sinks *and* the redaction pass via
NX_TOOL_RESULT_REDACT; the NX_BASH_PROMPT_SECRETS flip stays deferred until
one release of warning-only mode has shipped), 1.8 (9c75278), 1.7 (5392308 —
floor enforced as recorded BUDGET_TOKENS=2900, 17-tool floor measured 2879),
1.2 (3144f5c — RecoveryContext per loop, contextvar-resolved), 1.5 (404d369 —
parity tests + the drift fixed; the full four-table registry merge was
deliberately NOT done: the parity tests buy the same correctness without
moving 60+ rows across three modules).

| Minor | Contents |
| --- | --- |
| **0.40** | 1.1 cache-safe recovery · 1.3 breaker integrity · 1.4 one resume path · 1.6 private sinks + redaction |
| **0.41** | 1.8 two-tier overflow · window-aware max_tokens |
| **0.42** | 1.2 thread-isolated recovery · 1.7 floor budget reset · 1.5 parity tests · 2.1 write fence *(next)* |
| **0.43–0.44** | 2.3a/b store index + GC · 2.5 cost correctness · P3-5 `run_tests` |
| **0.45–0.46** | 2.4 compression v2 · 2.6 memory v2 · P3-4 one dialect |
| **0.47+** | P3-1 sandbox · P3-3 provider adapters · P3-6/7/8/9/10/11 as demand dictates · P3-14 early and continuously |

**Sequencing rationale:** Phase 1 items are each <1 day–3 days, independently
releasable, and several are *tested-wrong* today (1.1, 1.3, 1.4, 1.5) — meaning
they come with their pinning tests already written, which is the cheapest
possible correctness purchase. Phase 2 removes the structural reasons those bugs
reappear. Phase 3 adds reach; it is deliberately last because each P3 item is
cheaper and safer once the fence (2.1), the sandbox substrate (P3-1), the
provider abstraction (P3-3), and an eval harness (P3-14) exist.

## Appendix — audit provenance

Findings were produced by six read-only sub-agents covering: (1) turn loop +
self-healing (`_turn`, `_dispatch`, `_argrepair`, `_grammar`, `_stream_parse`,
`_text_fallback`, `_garble`, `_compress`, `_tool_cache`); (2)
provider/source/cost (`_sources`, `_openai_client`, `_capability`, `_cost`,
`_metrics`, `_env`); (3) sessions/memory/images (`_sessions_*`, `_memory_*`,
`_images`, `_repl_state`); (4) terminal UX (`_chatbox`, `_repl*`, `_ansi`,
`_tty`, `_approval`, `_interrupt`); (5) tools + security (`_tools`,
`_edit_tools`, `_bash_jobs`, `_risk`, `_allowed`, `_hooks`, `_undo`,
`_autowake`, `_batch`, `_subagent`); (6) engineering quality/docs. Their raw
reports live in `~/.nx/agents/agent-*.md` (`agent-6-dc6f59` loop, `agent-1-8ed54b`
provider, `agent-2-58f649` sessions, `agent-3-71a3aa` UX, `agent-4-027cb6`
tools/security, `agent-5-3c471f` quality).

Claims marked **[v]** were independently re-verified line-by-line during this
audit. Everything else is a static read that was not executed; in particular no
claim here was validated against a live endpoint, and items like bg-log file
modes were verified on this machine's umask, not in the abstract. Two
sub-agent findings were checked and are accurate as described; none of the
claims in the numbered roadmap items above were rejected during verification,
though several were re-scoped (e.g. "no cache support" → "cache *read* support,
no cache *write* control"; "no tests for X" → "no coverage-mapping test,
feature-specific tests exist").
