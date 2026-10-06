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
(`_sessions_fs.py:350-374`, `:386`) — quadratic in session length. ~~Two indexes
(`.index/` shards + FTS5) never reconcile; stale FTS rows are never pruned, so
`/sessions <query>` can offer a deleted id~~ **[v] 2.3a/2.3b (0.45.0/0.46.0):
index-driven GC, GC-on-delete, quota, and one SQLite store (`.store.sqlite`)
written by every indexed save with stale-row pruning**. `NX_SESSION_IMAGES=0`
~~(the documented rollback) disables image GC entirely, orphaning every
payload~~ **[v] fixed in 2.3a**.
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

### 2.6 Memory retrieval v2 **[L]** *(shipped 0.48.0)*
`memory_search` was substring, all-terms-must-match, over a full read of every
note, with the store re-scanned five separate ways across `_memory_cmd`/
`_memory_optimize`. `/memory list` read the **global dir only** — a
project-only repo reported "no memories saved yet". Filename creation was
`os.path.exists`-then-write (TOCTOU under the merge workers), archive stamps
were second-granularity across processes. The session note showed 5
titles/scope, so past ~10 notes discovery was only the keyword path.
`/memory optimize` reviewed only the newest 120 notes — the best retirement
candidates were never seen.
**Shipped:** one scanner (`_memory_cmd._iter_notes`) under every reader; an
FTS5 (trigram) index per memories dir (`.memindex.sqlite`, mtime-reconciled
cache, `NX_MEMORY_FTS=0`/no-FTS5 falls back to the plain scan, hits re-scored
by the unchanged `_rank_hit`); `/memory list` covers both scopes; the writer
claims its filename atomically (`os.link` no-clobber) so concurrent merge
workers can't erase each other's notes; `NX_MEMORY_OPTIMIZE_MAX_NOTES` now
takes the OLDEST tail. Embeddings deferred to P3-3 (provider item) — no
embeddings-capable source is configurable today.

---

## Phase 3 — Capability expansion (0.47+, only after Phase 1–2)

### P3 priority order (set 2026-09, phase 2 complete at 0.50.0)

Re-verified against the tree at 0.50.0: P3-1/3/7/8/9/11/12/13/14 are
**absent**; P3-2 is **partial** (HTTP-layer backoff exists but only before the
first byte, `_openai_client.py:14-18`; a mid-stream timeout still prints
"discarded partial output" and returns `TURN_DONE`, `_turn.py:1393`); P3-6 is
**partial** (read-only `git_status`/`git_diff`/`git_log` shipped in `TOOLS`;
commit is still classifier prose); P3-10 is **partial** (capability cached per
`Source` instance, reset on model pin, no TTL, not shown by `/source`).
Floor ledger: the enforced budget is `BUDGET_TOKENS` in
`tests/test_initial_context.py` (5200 as of the 0.99.x era; the 3180-era
`NX_FLOOR_BUDGET` numbers elsewhere in this file are historical).
Config surface is **143** `NX_*` vars, up from ~119 at audit time.

| # | Item | Effort | Why here |
| --- | --- | --- | --- |
| 1 | **P3-2** honest timeout + step-level transient retry *(shipped 0.51.0)* | S/M | Live mis-score: a mid-task timeout returns `TURN_DONE` and batch scores `done`. Smallest pure-correctness win left; the existing HTTP retry layer is the substrate to extend to the step boundary. |
| 2 | **P3-6** `git_commit` with staged-diff review *(shipped 0.51.0)* | S/M | Half-built already (read-only git tools in `TOOLS`); routes the most common write-without-review path through review, the 2.1 fence, and the 1.6 secret scan. |
| 3 | **P3-9** `nx config` viewer + `/status` *(shipped 0.52.0)* | S/M | 141 vars and growing; every later item adds more. Land the viewer/doctor *before* the additions so each new tunable ships visible. |
| 4 | **P3-14** eval harness: seed corpus + scored batch runs *(seed shipped 0.53.0)* | M, then continuous | The project's own velocity lever (would have caught the floor drift as a failure, not a warning). Seed it early, then score every later P3 item against it. |
| 5 | **P3-1** sandbox tier for `run_bash` + writes | L | The containment substrate: env passlist, cwd pin, rlimits, optional bwrap/landlock. Makes YOLO-without-sandbox harder and — since batch/eval runs execute unapproved shell today — is what makes item 4's own batch posture safe. |
| 6 | **P3-3** provider abstraction: Anthropic-native + Responses | L | Explicit `cache_control` is the project-thesis lever (theme 1) and cannot be proxied in. Excludes structured outputs per the 2026-09 decision. |
| 7 | **P3-10** capability manifest per (source, model) | M | Natural consumer of P3-3: one cached, model-keyed manifest (probe TTL, `/source` display) deleting the opt-in archaeology. |
| 8 | **P3-11** provider fallback ladder | M | Declared source chain on 429/5xx/overrun for unattended `nx run`; wants P3-10's health/cap signals and P3-14's scoring. |
| 9 | **P3-7** read-back UX: pager, `/diff`, highlighting | L | Biggest daily-driver trust gap; independent of 5–8 — interleave by demand. |
| 10 | **P3-8** chatbox editor completion | M→L | Split `_chatbox_raw` first; independent — interleave by demand. |
| 11 | **P3-12** multi-root / worktree isolation | L | Prerequisite for parallel batch + write-tier sub-agents; low urgency until fan-out is actually wanted. |
| 12 | **P3-13** web tools behind an egress policy | L | Last: needs P3-1's sandbox so egress is enforced rather than loud. |

**Minor bands (semver):** 0.51 P3-2 + P3-6 · 0.52 P3-9 · 0.53 P3-14 seed
corpus (then continuous) · 0.54–0.55 P3-1 · 0.56–0.57 P3-3 · 0.58 P3-10 +
P3-11 · 0.59+ P3-7/P3-8 by demand · 0.60+ P3-12, then P3-13.

**Sequencing rationale:** 1–3 each close a live gap (mis-scored timeouts,
unreviewed commits, invisible config) in a few days each, with no new
subsystem surface; 4 lands before the heavy items so they ship measured; 5
precedes 6 because the sandbox also hardens the batch/eval posture that 4
introduces; 7 rides on 6 (a manifest keyed per model needs the model
abstraction underneath); 9–10 are isolated UX tracks that can interleave after
0.52; 11–12 stay last because their demand (parallel batch, web access) is
speculative until the correctness tier exists.

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

### P3-2 Honest timeout + turn-level transient retry **[M]** *(shipped 0.51.0)*
`httpx.TimeoutException` mid-stream prints "discarded partial output" and
returns **`TURN_DONE`** (`_turn.py:1377-1387`)**[v]** — a mid-task timeout is
reported as a completed answer, and in batch mode a `done` line with truncated
output. `TURN_API_ERROR` is terminal. **Shipped:** the mid-stream timeout is a
real status now — `TURN_TIMED_OUT_RETRY` while `NX_TURN_RETRIES` (default =
the HTTP layer's `NX_HTTP_RETRIES`, same `_backoff_delay` curve) has ladder
left, `TURN_TIMED_OUT` once exhausted; the loop retries the same request at
the step boundary without burning a model turn, and exhaustion maps to a
status-"error" summary (batch/one-shot exit 4, `timed_out`-adjacent scoring,
never `done`). `NX_TURN_RETRIES=0` restores fail-fast.

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

> **Decision (user, 2026-09): no structured outputs (`response_format` /
> `json_schema` / `guided_json`).** Forcing a JSON schema degrades tool calling
> on most models — the constrained shape helps only when it matches what the
> model would have produced anyway, and otherwise fights it. nx's text
> protocol + repair stack (`NX_ARG_REPAIR`, GBNF grammar, breaker/nudges) is
> the mitigation, and it degrades gracefully instead of hard-failing a turn.
> Do not re-propose this as part of P3-3 or as a standalone item; `tools=`
> stays schema-as-documentation only, with native `tool_calls` (`NX_NATIVE_TOOLS=1`)
> as the sole server-side-constraint opt-in.

### P3-4 Grammar/prompt/parser: one dialect **[M]** *(shipped 0.50.0)*
The grammar constrained the legacy JSON-in-tag body while `SYSTEM` and the
correction nudge taught the XML dialect — on a grammar-capable source the model
was force-constrained into a shape its prompt never asked for. **Shipped:** the
grammar now constrains the taught shape — a `<name>` tag carrying the tool name,
then an `<arguments>` region of one tag per property — with string values as raw
text per SYSTEM's example, enums as raw literals, array/object properties
staying strict JSON, and required properties as mandatory tags. An earlier
attempt constrained the tool name into the element name instead, which is worse
than no grammar at all: the parser finds the tool by reading the `<name>` tag, so
every grammar-constrained call came back as a missing-name parse error (pinned as
a regression test). The capability probe now validates with the real parser
instead of a probe-only regex — the old regex accepted the JSON body the grammar
refuses to emit, so it could classify a source grammar-capable on a shape a turn
could not parse. Required-before-optional moved from a test over `m.TOOLS` into
the builder, which raises and recurses into array-item schemas. A value needing a
literal `<` is emitted JSON-quoted with an escape, so the raw form is unsamplable
rather than silently corrupted. **Left alone deliberately:**
`_special_tokens.py`'s DeepSeek signatures and the <=40-byte repetition window —
both tuned to observed collapse modes, not dialect problems.

### P3-5 Closed-loop verification: `run_tests` tool **[M]** *(shipped 0.49.0)*
`NX_VERIFY_CMD` is one global free-text command, Python-only `ast.parse` + exit
code, 20 s settle, off in batch, no target/language parsing, no scoped re-run.
A `run_tests(scope)` tool (detect pytest/cargo/npm/go, run targeted, parse
structured failures, offer them as a tail note) is the loop every edit-heavy
session wants and the verification tier write-sub-agents need.

### P3-6 Git workflow tool **[S/M]** *(shipped 0.51.0)*
`git commit/add/worktree` exist only as classifier prose. Committing is raw
`run_bash` reviewed as nothing; `apply_patch` has no stage/commit surface. A
`git_commit` with staged-diff review (and a pre-commit secret scan from 1.6)
removes the most common write-without-review path. **Shipped:** `git_commit`
(`nx/_gitcommit.py`) — the model stages paths explicitly (`git add --`), the
POST-stage `git diff --staged` becomes the approval `preview` (always exactly
what the commit will contain), and the same text is secret-scanned
(`_bash_jobs._contains_secret`, the run_bash patterns) before the confirm
round-trip; a hit refuses without prompting. Fixed argv shapes only
(`add`/`commit -m`, `NX_GITCOMMIT_TIMEOUT`), no free-text shell. Plan-gated,
in the sub-agent write tier, "git commit: <subject>" has a local medium risk
hint, and `_allowlist_tool` maps it for `A`-key saves. Denial leaves the
index staged, commits nothing. `run_bash` stays the tool for every other git
operation (worktree/branch/rebase remain classifier prose).

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

### P3-9 Config surface: `nx config` over the env layer **[M]** *(shipped 0.52.0)*
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

### P3-14 Eval harness: golden transcripts + scored batch runs **[M]** *(seed shipped 0.53.0)*
The `nx run` harness is benchmark-shaped (JSONL results, exit codes, per-task
token accounting) but there is no corpus: zero golden tasks, no recorded
transcripts, no regression scoring. A small corpus (10–20 tasks: refactor,
bugfix-from-issue, multi-file docs sweep, "recover from a hostile stream") with
recorded stub servers is what turns the recovery stack's heuristics from
anecdote-driven into measured. Arguably the highest-leverage item in this phase
for the *project's own* velocity — it is also where the 2500→2886 floor drift
would have been caught by a test rather than a warning.

Shipped at 0.53.0: the seed corpus (`nx/_eval_cmd.py`, 7 scenarios) + `nx eval
list|run`. Recordings are per-scenario scripted turns (one final-answer turn,
sequences of tool calls with real schemas, empty turns, a leaked-token
collapse, a stream that stops delivering mid-answer) served as OpenAI-style
SSE by a loopback stub server keyed on the task instruction, so one server
serves a whole batch. `nx eval run` seeds a throwaway workspace, runs the real
harness inside it (--yolo, confined), and scores each scenario against its
expectation (status, final answer, error shape, workspace file state) into
`summary.json`; exit 0 only when all pass. Continuous scoring (every P3 item
lands with a corpus run) is the next band.

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

**STATUS (0.53.0): P3-14 seed corpus shipped** — `nx eval list|run`: a 7-scenario
stub-server corpus (happy path, one real write, a real multi-file read+edit,
exploration, two silent empty turns, a special-token collapse, a mid-stream
read timeout) replayed through the real `nx run` batch harness against a
loopback stub endpoint and scored per scenario (status / final answer / error
shape / workspace file effects). The seed's first catch: the vendored client
passed `timeout=None` into `build_request`, which httpx reads as *disable
timeouts* — every request ran with no read/connect timeout, `NX_HTTP_TIMEOUT_READ`
was dead for the request path, and P3-2's mid-stream ladder was unreachable.
Fixed (`timeout` kwarg attached only when set); four unquiet-gated
`_stream_parse` collapse banners that leaked into batch JSONL stdout gated.

**STATUS (0.51.0): Phase 3, priority band 1–2 shipped.** **P3-2** (honest
timeout + step-level retry: `TURN_TIMED_OUT_RETRY`/`TURN_TIMED_OUT`,
`NX_TURN_RETRIES`, exhaustion scores `error`/exit 4 in batch and one-shot —
the live mis-scored mid-task timeout is closed) and **P3-6** (`git_commit`
with post-stage staged-diff review + pre-confirm secret scan, plan-gated,
sub-agent write tier, local medium risk hint) shipped together at 0.51.0;
the P3 priority table below is updated in place per item.

**STATUS (0.47.0): Phase 2 continues.** Shipped this session: **2.2** (0.47.0)
— undo correctness + durability: one approved call = one checkpoint (dispatcher
pushes via `push_if_absent`, which dedups against body-pushed checkpoints and
updates a re-touched file's snapshot in place), the `delete_file` approval
guard now reads mtime BEFORE `_confirm` (was a post-confirm tautology),
`sed_edit` pushes only after its staleness abort passes, `undo_last()` restores
atomically per file and keeps the entry pending when every restore failed, and
the stack persists to a 0600 `undo.json` under the session home — loaded at
startup, wiped by `/reset`. **2.4** (0.47.0) — compression v2: the digest turn
is reused append-only across folds (frozen header keeps the session-merge
contract; a continuation block carries the fresh counts; the old digest is
never re-summarized, so the prefix cache survives every fold), the kept tail
can be sized by tokens (NX_COMPRESS_KEEP_TOKENS, else ~window/4 derived from
the auto probe), the foldable region's biggest tool results are spilled to
disk before summarizing (NX_COMPRESS_SPILL_MIN_CHARS/_MAX, core shared with
the 0.41 overflow evictor), the fold call routes to a cheaper endpoint
(NX_COMPRESS_SOURCE/_MODEL) and is max_tokens-bounded
(NX_COMPRESS_SUMMARY_TOKENS, window-clamped). Then **2.6 memory v2 at
0.48.0**: one scanner under every memory reader, a trigram FTS5 cache per
memories dir (NX_MEMORY_FTS), both-scope `/memory list`, a no-clobber
atomic memory writer, and an oldest-first NX_MEMORY_OPTIMIZE_MAX_NOTES.
Then **P3-5 `run_tests` at 0.49.0**: a marker-detecting, fixed-command-shape
test runner tool (`nx/_testrunner.py`, NX_TESTRUNNER_TIMEOUT) — detect
pytest/go/cargo/npm, run targeted via `targets`, fold the output to the
failures with the raw log one read_file away; local risk fast path (medium),
plan-gated, in the sub-agent write tier. NX_VERIFY_CMD stays the hands-off
post-edit loop. **Previously:** Phase 1 DONE
at 0.42; 2.1 (0.43.0), 2.5 (0.44.0), 2.3a (0.45.0), 2.3b (0.46.0) as detailed
below. **P3-4 (one dialect) closed at 0.50.0**, which empties Phase 2's open
list; what remains is Phase 3, now priority-ordered (see "P3 priority order",
the first subsection under Phase 3).

Then 2.1 (0.43.0): a hard write fence in `run_tool()` (`nx/_fence.py`, env
`NX_WRITE_FENCE`/`NX_WRITE_ROOTS`) — realpath-roots (project root + session
home + extras), every structured write candidate judged at fence time
(sed_edit glob matches, apply_patch `+++` targets), symlink-through targets
judged where the write actually lands, a denylist for credential/shell-config
destinations even from inside a root, and run_bash deliberately left to the
classifier+approval story pending P3-1. Then 2.5 cost correctness (0.44.0):
reasoning_tokens now reads BOTH field shapes (output_tokens_details and
completion_tokens_details), every side-call (risk classifier, capability /
thinking probes, the ctx-overrun POST a clamping server executes) is
_record_usage'd via _record_api_call, chars//4 estimates carry a provenance
flag surfaced by /cost (~N est), by_source survives resume (saved+reloaded
in the session JSON), and pricing gained optional per-model keys
NX_PRICE_<SOURCE>_<MODEL>_<KIND>.
Then 2.3a (0.45.0): index-driven image GC — each .index shard summary carries
`sha256s`, so the GC live-set is O(#sessions) shard reads instead of a full
transcript parse per save (missing/corrupt/pre-2.3a shards fall back to
parsing that one transcript); GC runs even when NX_SESSION_IMAGES=0 (it used
to orphan every stored payload); GC-on-delete in _delete_session; a
NX_IMAGES_QUOTA_MB cap (default 200, 0 disables) evicting oldest-unreferenced
first; and `nx sessions --prune` to reconcile by hand (drop orphaned shards,
immediate collect, quota).

| Minor | Contents |
| --- | --- |
| **0.40** | 1.1 cache-safe recovery · 1.3 breaker integrity · 1.4 one resume path · 1.6 private sinks + redaction |
| **0.41** | 1.8 two-tier overflow · window-aware max_tokens |
| **0.42** | 1.2 thread-isolated recovery · 1.7 floor budget reset · 1.5 parity tests |
| **0.43** | 2.1 hard write fence *(shipped)* · 2.5 cost correctness *(0.44.0)* |
| **0.43–0.44** | 2.3a store GC *(0.45.0)* · P3-5 `run_tests` *(0.49.0)* |
| **0.45–0.46** | 2.4 compression v2 *(0.47.0)* · 2.6 memory v2 *(0.48.0)* · P3-4 one dialect |
| **0.49** | P3-5 `run_tests` tool *(shipped)* — the Phase-2-adjacent verification tier |
| **0.47+** | P3-1 sandbox · P3-3 provider adapters · P3-4 one dialect · P3-6/7/8/9/10/11 as demand dictates · P3-14 early and continuously |

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
