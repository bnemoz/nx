# Swarm optimization audit — `nx`

> **Implementation status (0.99.0):** P1–P8 landed in 0.98.0 as one focused
> commit each (see git log: `fb34fc5` P1 … `1dd8191` P8); P10–P13 landed in
> 0.99.0 the same way (`6c45fb1` P11 store contention, `45a6a3c` P10 the
> cross-process obsstore, `b62601b` P12 headless watcher gate, `9db6971` P13
> tier-2 floor slimming). Each item's fix and its knob are noted inline where
> it applies; the table is kept as written for the original audit record.
> Remaining open items: per-process RSS floor measurement, `NX_OCC`
> benchmarking, and the pymupdf4llm extra.

**Scope:** memory/processor optimization of the `nx` harness for a move toward an
agentic-swarm deployment (many concurrent sub-agents on one host).
All findings verified against current source (v0.97.0). Baselines quoted from
`PERF.md` (§1 import-cost measurements; PF‑01…PF‑22 shipped, PF‑20 deferred,
**PF‑23 not shipped**).

---

## TL;DR — what blocks dense swarms today

| Rank | Item | Type | Fix effort |
|---|---|---|---|
| 1 | `drop_changed_entries` → O(cache × 5000) `stat()` storm per edit | CPU | small |
| 2 | Process-global coordination state (notices, prefix pin, undo) forces process-per-agent | memory ×N / correctness | medium |
| 3 | Per-process caches duplicated, zero cross-process sharing; `NX_OBS` off by default | memory ×N | medium |
| 4 | Full-prompt `json.dumps` ×2 per request into an unbounded shared log queue | CPU + RAM | small |
| 5 | Write-tier sub-agents can't run concurrently; no nested fan-out; `_ASYNC_SEM`=4 process-wide | throughput | small |
| 6 | Per-save O(store) work (`prune_stale`, image GC) on every indexed session write | CPU | small |
| 7 | Prompt-cache routing key is per-session → shared swarm prefix prefilled N times | token cost | trivial |
| 8 | Context/token management exists but swarm-relevant defaults are OFF (`NX_OCC`, `NX_OBS`) | KV pressure | config |
| 9 | Process churn ~0.12–0.18 s + `git rev-parse` subprocess per spawn | CPU | medium |
| 10 | Misc: SQLite connect-per-call, unbounded small sets, stdin-thief interrupt thread, no read deadline | small | small |

---

## P0 findings (detail)

### 1. `tree_stamp` stat storm after every mutation — largest avoidable harness CPU
- `nx/_tool_cache.py:320-325` — `drop_changed_entries` re-validates **every** cache
  entry after any mutation; `cache_valid` (`:238-240`) calls
  `tree_stamp(entry.path)` per `search_files` entry, each walking up to
  `NX_TOOL_CACHE_TREE_FILES` = **5000 entries, one `os.stat` each** (`:179-196`).
- Callers: `_dispatch.py:580` (after every successful
  `write_file/edit_file/multi_edit/sed_edit` (incl. edit_file's `diff` mode) and
  `_dispatch.py:593`
  (after every `run_bash`). With the 512-entry cache (`:277`) holding a few dozen
  search entries, **one edit costs tens of thousands of stats**. PERF.md measures
  `_walk_tree` at 35 ms / 19.7k entries ⇒ ~9 ms per `tree_stamp`.
- On a GIL-bound host with N write-tier agents this is pure harness CPU competing
  with the local server's tokenizer/sampler.
- **Race:** `drop_changed_entries` iterates a snapshot then `del cache[key]` with no
  lock while `_tool_prefetch.py:216` pool threads insert → intermittent `KeyError`.
- **Fix:** memoize the stamp with a TTL (recompute once per validity check, not per
  edit); or key invalidation off a cheap repo-level version (`.git/index` mtime /
  `git status --porcelain` hash) and demote `tree_stamp` to a lazy check on *hit*
  only. Add the missing lock.

### 2. Process-global coordination state — forces process-per-agent (memory ×N)
Load-bearing channels are single process-wide slots, not per-loop:
- `st._PENDING_SUBAGENT_NOTICE` / `_PENDING_BACKGROUND_NOTICE` /
  `_PENDING_VERIFY_NOTICE` (`_state.py:379-389`), flushed into whichever transcript
  calls `_flush_pending_background_notice` (`_runtime_notes.py:59-92`).
  `_batch.py:46-48` documents it: with `--parallel > 1` an async agent's answer can
  land in a **sibling task's turn** → in-process swarm fan-out is not usable today.
- `_prefix.pinned` is one module global (`_prefix.py:48`); concurrent sessions in one
  process clobber each other's pin, so `check_prefix` (`_turn.py:381`) warns
  spuriously and re-pins to the wrong head.
- `_UNDO_STACK` (`_state.py:399`) + `undo.json` at a **single fixed path**
  `~/.nx/undo.json` (`_undo.py:155-156,186-187`): N concurrent processes rewrite the
  same file each turn; last writer wins; `/undo` across the swarm is meaningless.
- Endpoint-evidence globals (`_MALFORMED_TOOL_FAILURES/_BROKEN`,
  `_NATIVE_TOOL_CALLS_CONFIRMED`): ROADMAP 1.2 is effectively **done** —
  `_recovery.py:39-50` routes most per-loop (`install()`/`copy_back_and_reset()` at
  `_turn.py:812,825`).
- **Fix:** move the notice slots and prefix pin onto the TLS/ContextVar pattern
  `_recovery.py` already uses; namespace `undo.json` per session id. This is the
  unlock for dense in-process swarms.

### 3. Per-process caches duplicated — the memory ×N term
Verified per-process budgets (all ×N agents):
- `st._TOOL_RESULT_CACHE`: **16 MB** result text + 512 entries (`_tool_cache.py:274-279`).
- `_UNDO_STACK`: up to `NX_MAX_UNDO`=20 checkpoints each holding **whole file
  contents** (`_undo.py:35,46-56`). PF‑05's per-entry snapshot cap is deferred
  (`PERF.md:237-238`). Biggest RSS term after the transcript in a write swarm.
- `_ASYNC_AGENTS` retains full `answer` strings for 20 finished + all running agents
  (`_subagent.py:292,311,466-472`) — redundant: answers are already on disk at
  `~/.nx/agents/<job>.md` (`_write_agent_report`, `:238-268`); drop the RAM copy once
  `report` is set.
- `_chatbox._TREE_CACHE/_TREE_FILTER_CACHE`: 20 000-entry tree listing
  (`_chatbox.py:224-245`), ~10 MB in a big repo, rebuilt per process.
- `_obs._SEND_COUNTS` unbounded (`_obs.py:54`, off by default).
- **Highest-leverage fix:** a cross-process, content-addressed tool-result store.
  Both halves already exist: `_overflow.spill_to_disk` writes sha256-named files
  (`_overflow.py:80-114`), and `NX_OBS` (`_obs.py:37`, off by default) content-addresses
  large results and swaps them for id+placeholder after 2 sends. **`NX_OBS=1` for a
  swarm gives per-agent context savings AND a shared on-disk dedup layer** — the most
  swarm-relevant existing feature, disabled by default. Also cap undo snapshots by
  bytes and spill content to the overflow store.

### 4. Double full-prompt serialization into one unbounded shared log queue
- `_log_event("req", {... "messages": messages, "tools": tools ...})`
  (`_compress.py:288-290`) inline-`json.dumps`es the **entire prompt**
  (`_logging.py:102`); httpx serializes the same payload again in
  `build_request(json=payload)` (`_openai_client.py:392-397`). ~2× ~400 KB per
  100K-token call per agent, on the agent's thread.
- Log queue is `queue.Queue()` with **no maxsize** (`_logging.py:41`); the single
  `nx-log-writer` daemon thread (line-buffered, one flush per line, `:62,81`) falling
  behind N streaming agents ⇒ **unbounded RAM backlog**.
- `LOG_PATH` is fixed `~/.nx/logs/llamacpp.log` (`:28`): every process appends to the
  same file, each independently `os.replace`s to `.1` at 10 MB (`:74-80`) → rename
  clobbering, lost diagnostics.
- **No env gate to disable request logging** (only `NX_LOG_MAX_BYTES`). PERF §3
  refuted the per-chunk cost (1.2 µs) but never measured the per-request dump.
- **Fix:** `NX_LOG_REQUESTS=0` default for headless/swarm runs; per-pid/session log
  path; bound the queue with drop-oldest; drop `tools` from the req log (byte-stable
  and pinned by `_prefix` anyway).

### 5. Fan-out deliberately serialized in exactly the shapes a swarm needs
- Write-tier sub-agents **cannot run concurrently**: `_should_parallelize` returns
  False for any write-capable spawn (`_tool_prefetch.py:162-166`) because `_confirm()`
  auto-denies on a quiet thread. `NX_SUBAGENT_WORKTREE=1` (`_worktree.py:1-9,77-103`)
  already removes file-trampling but not the approval reason — lift the gate under
  `st.YOLO` (no prompt exists to serialize).
- Nested fan-out refused outright: `if _subagent_active(): return False`
  (`_tool_prefetch.py:161`) ⇒ **no two-level swarms**.
- `_ASYNC_SEM` default **4** is process-wide (`_subagent.py:305-337`).
  `_async_duplicate_running` (`:410-420`) same-instruction dedup: keep.
- Fresh `ThreadPoolExecutor` per turn (`_tool_prefetch.py:86,224`) — thread churn;
  a shared bounded pool makes the concurrency cap one knob instead of two.
- `_SPAWN_LOCK` is **gone** (0.155.0): it only ever guarded the process-global
  YOLO/APPROVE_LEVEL snapshot-restore that a sub-agent loop used to do. Posture
  is thread-local now (`nx/_state.py: _POSTURE_TLS`, `_posture()`,
  `_posture_scope()` installed around the agent body in `nx/_subagent.py`), so
  there is no shared global left for two concurrent agents to race over and the
  lock had no remaining users.

### 6. Per-save O(store) work on every indexed session write
`_write_session` (`_sessions_fs.py:266-352`) per indexed save:
whole-file `json.load` + merge + full `json.dumps` rewrite under `fcntl.flock`
(`:288-324`); SQLite WAL upsert (`_storeindex.py:79-94`); **`prune_stale()` —
O(#sessions) `os.stat` per save** (`_storeindex.py:312-316`); image-ref scan;
`_gc_image_sidecars` + `_gc_image_quota` — **`os.stat` over the entire `_images/`
dir per save** (`_sessions_fs.py:349-352,377-391`). Linear-per-save at 1000
sessions × N agents. The cross-process flock + `busy_timeout=5000` serializes a
swarm sharing `NX_HOME` on the store DB during index writes.
- **Fix:** move `prune_stale` + image GC to `nx sessions --prune` / a lazy
  once-per-session sweep with an age/version gate. (Mid-turn autosave already passes
  `update_index=False`, `_repl.py:279`.)

### 7. Prompt-cache routing is per-session — exactly wrong for a swarm
`routing_identity()` returns `f"nx-{session_id}"` (`_manifest.py:144-158`), attached
as `user` + `prompt_cache_key` (`:170`), default-on. All swarm agents share the
identical `SYSTEM` + `prefix_notes()` head (`_prefix.py:59-81`, AGENTS.md capped at
8000 chars) — a per-agent key **spreads agents across replicas/KV blocks**, so the
shared head is cold-prefilled N times instead of once.
- **Fix:** swarm-wide routing key (e.g. `NX_CACHE_ROUTING_GROUP` derived from
  project root + model, falling back to session id) so a vLLM/SGLang router
  co-locates agents with shared prefix bytes. Free prefill savings; highest-leverage
  token-cost item on a KV-constrained host.

### 8. Context/token management: right mechanisms exist, swarm defaults are OFF
- Sub-agent answers return **head (600 chars) + report path**, not the full answer —
  both async and sync paths (`_subagent.py:229,485-519`, `_lean_result` `:620-631`;
  sync since 0.96.0). Correct swarm contract, already done. ✅
- Tool results spill on the way *in* above 102 400 chars (`_text_fallback.py:69,87`)
  — no giant result enters history (PF‑14 ✅).
- `NX_OCC` (online context compact, `_occompact.py:19-51`) defaults **0** — the
  mechanism that bounds aggregate KV pressure; benchmark on for swarms.
- Auto-compress: last-third keep target (`_compress.py:371-384,448-480`),
  `_maybe_autocompress` at 85 % (`:696`), `_keep_past_notes` protects the keep window
  (`_runtime_notes.py:95-110`). All sound.
- **Permanent token rent ×N:** 20 always-on tool schemas (`active_tools_floor`=20,
  `tests/perf/baselines.json:11-15`), `BUDGET_TOKENS` 3320. Tier-2 has **no
  occupants** (`PERF.md:711-713`) — move swarm-irrelevant tools (`plan_edit`,
  `read_image`, `duckdb`, `wait_background`) behind `search_tools` to shrink every
  agent's prompt identically.
- Token estimate `len()/3.5`; tiktoken only in `/ctx`/tests. No runtime cost ✅.
- **PF‑23 unshipped (verified):** `sum(len(str(m.get("content") or "")) for m in messages)`
  runs on every settled stream regardless of server `usage` — `_turn.py:547`, also
  `_compress.py:668`, `_sessions_fs.py:803`, `_risk.py:385`. `_normalize_usage` uses it
  only when `usage` is absent (`_metrics.py:147-160`). Trivial fix; full-prompt
  `str()`+`len()` per turn per agent.

### 9. Process churn per spawn
~0.12–0.18 s import (`PERF.md §1`: 97 ms cumulative, `httpx` 46 ms,
`rich.markdown` 15 ms) + `_detect_project_root()` ~2 ms subprocess (`_risk.py:33`).
`pymupdf4llm` is a hard core dependency (`pyproject.toml:18`) though lazily imported
(`_pdf.py:40,46`) — no RSS unless PDFs, but dominates install size; move to an extra
for headless images. If process-per-agent, prefer a long-lived supervisor forking a
warmed process, and `NX_SKIP_STARTUP_PROBES=1` (`_sources.py:539`) to kill probe
threads per process.

### 10. Lower-impact, verified
- `sqlite3.connect()` per call, no thread-local reuse (`_storeindex.py:79-94`,
  `_memindex.py:109-121`); WAL + `busy_timeout=5000` can block a shared-`NX_HOME`
  swarm up to 5 s during concurrent index saves → per-session private store dir or
  thread-local connections.
- `search_files`/git shell out per call (`_tools.py:299,375`, 30 s timeout) —
  subprocess-per-call is GIL-friendly (good for a swarm); keep.
- `_fusion._FileQueue`: `threading.Lock` per canonical path, never evicted
  (`_fusion.py:70-116`); `_bash_jobs._WATCHDOG_PIDS` (`:642`); `_ASYNC_QUEUED`
  (`_subagent.py:306`, pruned `:357-363`) — small.
- `_metrics` ring bounded at 64 turns (`_metrics.py:231-233`) ✅; `_AsyncPusher`
  off-thread (PF‑07 ✅); interrupt watcher singleton w/ wake pipe (PF‑06 ✅) but it's
  a **stdin thief** (`_interrupt.py:17-35`) and termios is tty-global ⇒ a swarm must
  run headless (`nx run --parallel` / async-quiet) or one-agent-per-tty. **This is the
  hard constraint on in-process density.**
- `httpx.Client` one per `Source`, reused, default `Limits` (`_openai_client.py:452-453`);
  default `max_connections=100` ample for ≤16 agents; `read=900 s` means a stuck agent
  pins a connection indefinitely — swarm wants a tighter read timeout or per-agent
  deadline (batch already has `_turn_retry._TASK_DEADLINE_TLS`).

---

## Landed fixes, by phase (0.98.0)
- **P1 `fb34fc5`** — `drop_changed_entries` fast-path (tree-stamped entries
  evict on one root-mtime stat; honest full walk stays on hit;
  `NX_TOOL_CACHE_SWEEP_TREE_WALK=1` restores legacy) + `pop(key, None)` race fix.
- **P2 `dde3356`** — `nx/_notices.py`: per-turn-loop notice slots (contextvar);
  per-thread prefix pin; `undo.json` namespaced per session under
  `SESSION_HOME/undo/`. Fixes the `_batch.py` cross-talk failure.
- **P3 `ab61a4d`** — `NX_MAX_UNDO_BYTES` (32 MB default) byte-caps the undo
  stack; `NX_SUBAGENT_ANSWER_KEEP` (4000 default) bounds retained async answers.
- **P4 `c8f78f4`** — `NX_LOG_REQUESTS=0` stub-logging gate; bounded (4096,
  drop-oldest) log writer queue.
- **P5 `d84da24`** — write-tier fan-out allowed under `--yolo` +
  `NX_SUBAGENT_WORKTREE=1`; `NX_SUBAGENT_MAX_NESTED=1` enables two-level swarms.
- **P6 `fe3015f`** — `prune_stale` + image GC amortized off the save path
  (`NX_SESSION_SWEEP_INTERVAL`, default 300 s).
- **P7 `d75ed08`** — `NX_CACHE_ROUTING_GROUP` shared prompt-cache routing key.
- **P8 `1dd8191`** — PF-23: lazy fallback token estimate (no eager full-prompt
  scan when the server reports usage).

## Landed fixes, by phase (0.99.0)
- **P11 `6c45fb1`** — store contention: `NX_STORE_BUSY_TIMEOUT_MS` (default
  750, was a hardcoded 5 s block) on both SQLite stores; busy-vs-corrupt
  retry split (a locked store retries with jittered backoff and is NEVER
  set aside as `.corrupt`); `prune_stale` and `search_sessions` use ONE
  connection per call (was two connects + 2 PRAGMAs each).
- **P10 `45a6a3c`** — `nx/_obsstore.py`: the cross-process content-addressed
  tool-result store. Object identity is the sha256 of the CONTENT (the old
  tool+call_key identity defeated dedup — two agents reading the same file
  wrote two objects); no-clobber `os.link` publish; TTL
  (`NX_OBS_STORE_TTL_DAYS`, 7) + byte quota (`NX_OBS_STORE_MAX_MB`, 512)
  GC amortized behind the P6 sweep gate and wired into `nx sessions
  --prune`; the unbounded `ledger.jsonl` is gone; `_SEND_COUNTS` bounded
  (`NX_OBS_MAX_TRACKED`, 4096); obs ids keyed on content so a 24-hex
  handle resolves across processes; legacy `~/.nx/observations/` objects
  still recall.
- **P12 `b62601b`** — the stdin-stealing interrupt watcher is no longer
  started (nor armed) by a quiet loop — headless/async sub-agents pay
  zero for it; `NX_INTERRUPT_WATCHER=0` disables it process-wide for
  headless runs. The "must run headless or one-agent-per-tty" density
  constraint is now: in-process density costs one watcher thread total
  (the parent's), not one per agent — and zero with the knob off.
- **P13 `9db6971`** — `NX_TOOL_FLOOR_SLIM=1` demotes `duckdb` + `read_image`
  to tier-2 (withheld behind `search_tools`, gate refusal names the knob);
  default unchanged (the 0.87.0 always-on duckdb ledger holds).

## Still open (ranked)
1. `NX_OCC=1` benchmarking for aggregate KV pressure.
2. Per-process RSS floor measurement + cache soak (commands below).
3. The pymupdf4llm extra (install-size, not runtime).
4. Tier-2 floor slimming has two occupants (`duckdb`, `read_image`);
   candidates for a further pass: nothing else is clearly swarm-irrelevant
   without hurting single-agent UX.

## Not verified (audit budget)
- No runtime measurements on this box; absolute numbers quoted from `PERF.md §1`.
  To get swarm-relevant figures:
  - `/usr/bin/time -v .venv/bin/python -c "import nx"` → per-process RSS floor.
  - A `test_perf_bounds.py`-style soak: RSS after 500 reads + 50 spawns with
    `NX_TOOL_CACHE_MAX_BYTES=16MB`.
- Files not read end-to-end: `_eval_cmd.py`, `_telegram.py`, `_webfetch.py`,
  `_gitcommit.py`, `_verify.py`, `_hooks.py` (subprocess-per-event, zero cost when
  unset — fine for a swarm), and the `_turn.py` main-loop body beyond cited seams.
- Swarm test coverage: `test_parallel_dispatch.py`, `test_perf_bounds.py`,
  `test_batch_run.py`, `test_worktree_097.py` exist but were not run; no test pinned
  cross-process `undo.json`/`llamacpp.log`/`_PENDING_*` behavior was found by name
  (files not read).
