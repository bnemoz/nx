# nx Performance Spec Sheet — the "never becomes a laggy TUI" plan

Working plan for keeping `nx` fast while it grows from a coding agent into a
personal AI assistant. Produced from a 17-agent audit swarm (one area each,
~150 tool calls of static reading plus inline benchmarking) over `nx/*.py`
(26,210 LOC, 71 modules, v0.54.1), **re-verified by the parent against the
code and against measurements**. Where a sub-agent claim failed measurement it
is recorded as refuted — that is part of the deliverable, because "we checked
and it is already fast" is what stops the plan from wasting a release.

**This document is the performance half of the plan. `ROADMAP.md` is the
correctness/consistency half.** Where they overlap (prefix-cache safety, the
token floor) PERF.md defers to ROADMAP's framing and only adds the latency and
budget numbers.

---

## 0. The thesis, in one paragraph

nx's responsiveness risk is **not** in the places people usually optimize. It
has no render loop, no retained widget tree, one reused HTTP client, an indexed
session store, bounded git caching, append-only recovery nudges, and
already-shipped concurrent read-only tool dispatch. The real risk is
**four specific accidental O(n²) loops plus a per-keystroke filesystem walk**,
all of which are invisible at demo size and brutal at real size, and **a token
floor that has 3 tokens of headroom left** so the next feature silently breaks
the suite. Everything else in this document is supporting detail.

```
Priority = (cost at realistic size) x (how often it runs) x (does it grow with features)
```

---

## 1. Measured baseline (the numbers to beat)

Every number below was measured on the audit box; `[m]` = measured by the
parent in this session, `[i]` = inferred by an auditor and not re-measured.

| Metric | Value | How measured | Notes |
| --- | --- | --- | --- |
| Process start → `import nx` done | **0.12–0.18 s** `[m]` | `/usr/bin/time -f %e python -c "import nx"` ×3 | Baseline bare interpreter: 0.01 s. `httpx`+`rich` alone: 0.08 s. |
| `nx` import (importtime, cumulative) | **~97 ms**, self ~60 ms `[m]` | `python -X importtime -c "import nx"` | `httpx` subtree 46 ms, `rich.markdown` 15 ms, `rich.console` 8 ms. |
| Per-turn token floor | **3177 tok** vs `BUDGET_TOKENS` 3180 `[m]` | `tests/test_initial_context.py` | SYSTEM 375 + TOOLS 2801 (19 active) + "hey" 1. **3 tokens of headroom.** |
| Floor composition | **1683 tok param schemas / 624 tok descriptions** `[m]` | per-tool `chars//3.5` | Schema *structure*, not prose, is the rent. |
| Test suite | **1280 tests in 11.91 s** (`-n auto`) `[m]` | `uv run pytest tests/ -n auto -q` | 114 files, 1142 `def test_`. |
| conftest per-test state rewind | **0.43 ms** ×2 per test ≈ **1.0 s** of the suite `[m]` | replayed `_restore()` inline over 56 snapshotted `nx.*` modules | Ruled out as a scaling risk (see §4). |
| Session save, 50 → 1500 msgs | **2.4 ms → 14.2 ms** (1.37 MB rewritten) `[m]` | `_write_session` timed over a temp `NX_HOME` | Flat in *session count* (~7 ms at 1000 sessions) thanks to the index. |
| Session load, 1500 msgs | **1.6 ms** `[m]` | `_load_session` | Already cheap; not a resume bottleneck. |
| Token estimate over a 308 KB prompt | **~0 ms** `[m]` | `len()/3.5` — no tiktoken at runtime | Free; not worth optimizing. |
| `json.dumps` of 100 × 2 KB messages | **0.22 ms** `[m]` | direct | Payload assembly is not a cost centre. |
| Per-chunk log serialization | **1.2 µs/chunk** (5000 chunks = 6 ms) `[m]` | replayed `_LoggingStream`'s `json.dumps` | Refutes the "CRITICAL per-token logging" claim (§4). |
| **`@`-mention completion, per keystroke** | **38–56 ms** `[m]` | `_match_file_paths` on a synthetic 19,721-entry tree | **The worst interactive number in the codebase.** |
| `_walk_tree` over 19,721 entries | **35.3 ms** `[m]` | direct call on `/tmp/bigrepo` | Re-run per keystroke, uncached. |
| `_detect_project_root()` | **1.97 ms** (subprocess) `[m]` | `git rev-parse --show-toplevel`, uncached | Called from that same per-keystroke path. |
| **Stream tool-tag scan** | per-chunk **2.4 µs → 77.2 µs**; **618 ms** total on a 288 KB answer `[m]` | `_first_tool_tag_index` over the growing buffer | 4× buffer ⇒ 4× per-chunk cost ⇒ Θ(n²). |
| rich `Markdown` render | **~1 ms/KB**, 17.4 ms at 17.5 KB `[m]` | `Console.print(Markdown(...))` | Total re-render CPU per 4000-token answer: **112 ms** `[m]`. |
| `difflib` on a 30,000-line file | 9.8 ms CPU, but **emits a 520 KB diff for a 1-line change** `[m]` | `_diff_hunk` on repetitive content | CPU fine; output size is the issue (capped downstream). |
| Idle CPU while parked at the prompt | **~0** `[i]` | 10 ms poll at `_tty.py:174` is the Windows-only branch | Linux path is `select()` — no busy poll. |

---

## 2. Bottleneck ledger

IDs are stable (`PF-nn`) — cite them in commits. Effort: S ≈ < half a day,
M ≈ a few days, L ≈ a week+.

### P0 — the four accidental quadratic loops + the keystroke walk

These are the only items that degrade *smoothly and invisibly* as sessions and
repos grow. All five are local, testable, and shippable independently.

---

**PF-01 · HIGH · `_chatbox.py:493` → `_match_file_paths` (`:255`) → `_walk_tree` (`:220`)**
*Every keystroke after typing `@` walks up to 20,000 filesystem entries.*

`_match_file_paths` is called **from inside `render()`** (`_chatbox.py:493`), and
`render()` is called once per keystroke (`:685, :712, :723, :731, :743, :780`).
With no `/` in the prefix it takes the tree-wide fuzzy branch: `_walk_tree`
(scandir BFS, `_TREE_SCAN_CAP = 20000`, `os.path.relpath` per entry) followed by
`_fuzzy_score` over every entry plus a sort. Measured **38–56 ms per keystroke**
on a 19.7k-entry repo; `relpath` and `_detect_project_root()`'s 1.97 ms
`git rev-parse` (`_risk.py:33-52`) are included in that.

This is *the* "laggy TUI" failure mode: typing `@src/auth` costs ~½ second in
file I/O before the dropdown updates, and it gets worse with repo size forever.

**Fix (three parts, all small):**
1. **Cache the tree**, invalidated by `os.stat(root).st_mtime` on a short TTL
   (~0.5 s) — files created mid-session still appear within a TTL, and `/new`
   or a `run_bash` can force-bust it. Module-level
   `_TREE_CACHE = (root, mtime, deadline, entries)`.
2. **Never do it from `render()`.** Compute matches once per key event, after
   the buffer mutation, and pass them into `render()` as a parameter. `render()`
   must be allocation-light and I/O-free by contract.
3. **Incrementally filter.** Once a prefix has been resolved against the tree,
   a longer prefix should filter the previous match set, not rescan. Also memoize
   `_detect_project_root()` per `(cwd)` — the cwd does not change mid-REPL.

*Payoff:* per-keystroke cost 38–56 ms → **< 0.5 ms**. This is the single biggest
interactive win available.

**Shipped in 0.55.0.** Module-level `_TREE_CACHE` (root + mtime + 0.5 s TTL)
rebuilt with per-entry derived data (lowercased base/path, char-presence mask);
`_PROJECT_ROOT_CACHE` memoizes `_detect_project_root()` per cwd;
`_TREE_FILTER_CACHE` + `_TREE_RESULT_CACHE` give the incremental filter and a
replay for an unchanged prefix; `render()` is now I/O-free by construction (the
completion lookup moved to a new `compute_matches()` called once per key event).
The char mask is a sound necessary-condition reject, so results are byte-identical
to a cold scan (`tests/test_chatbox_completion_cache.py`). Measured on a
20,000-entry tree: 41 ms → 0.3 ms for a selective prefix, ~27 ms only for a
1-char prefix over the whole (now mask-filtered) candidate set.

---

**PF-02 · HIGH · `_stream_parse.py:361` → `_first_tool_tag_index` (`:70-76`)**
*The tool-tag scan is Θ(n²) over every streamed answer.*

`_first_tool_tag_index(full)` runs two `str.find` passes over the **entire
accumulated** content string on **every** chunk, unthrottled. Measured: per-chunk
cost grows 2.4 µs → 77.2 µs as the buffer grows (500 → 8000 chunks), totalling
**618 ms of pure CPU on a 288 KB answer**. This sits on the SSE read thread, so
it directly delays token delivery to the terminal and delays the Esc/interrupt
path.

**Fix:** keep a scan watermark. A tag opener can only start at most
`len(opener) - 1` bytes before the previous chunk boundary, so scan from
`max(0, len(prev) - MAX_OPENER_LEN + 1)` each time. Once an opener is found the
scan stops permanently (set the watermark to the end). Independently confirmed by
three audit agents as the top CPU item.

*Payoff:* 618 ms → low single-digit ms on a long answer; makes the parse loop O(n).

**Shipped in 0.55.0.** `_first_tool_tag_index(text, start)` takes a `start`
offset and the parser keeps a `_tag_scan_from` watermark advanced to
`len(prev) - (_MAX_OPENER_LEN - 1)` after each miss, so a split opener is still
caught while the scan stays O(chunk) (`tests/test_stream_tag_scan_watermark.py`).

---

**PF-03 · MED · `_stream_parse.py:357` `self._content_str += chunk`**
*O(n²) string copies, conditional on refcount.*

CPython's in-place `+=` fast path requires the string's refcount to be 1. The
next line assigns `full = self._content_str` and hands it to
`Live.update(Markdown(...))`, so a reference survives across the throttle window
and the fast path is defeated on the renders — each `+=` then copies the whole
buffer. Two agents flagged this; it is real but secondary to PF-02 because the
render path was already the larger cost.

**Fix:** `self._content.append(chunk)` (the list already exists at `:356`) and
materialize with `"".join(...)` only where the full string is actually needed
(the throttled render and the final commit). Removes the duplicate accumulator.

---

**PF-04 · MED · `_stream_parse.py:368-371` (+ `:159`) — re-rendering the whole
accumulated answer as Markdown, twice**
*Θ(n) parse per frame at up to 20 frames/s, plus a second full parse at stop.*

`say_live.update(Markdown(strip_ansi_for_display(full)))` re-parses the entire
answer every ≥50 ms, and `stop_say_live` (`:155-160`) parses the whole thing
*again* for the final commit. Measured rich cost: ~1 ms/KB, so the whole-answer
re-render totals **~112 ms for a 4000-token answer** — real but not fatal. The
sharper problem is that rich's `Live` holds an internal lock across a whole
render (`rich/live.py:239` with its `_RefreshThread`), so a slow render on the
12 Hz refresh thread stalls the SSE thread that calls `update()` — an occasional
multi-tens-of-ms hitch whose frequency grows with answer length. Also,
`strip_ansi_for_display` (`_ansi.py:42-45`) runs 4 regex subs over the whole
buffer per frame.

**Fix (in order of value):**
1. **Strip ANSI and memoize per chunk, not per frame** — accumulate the *clean*
   string incrementally (`_clean_str` maintained alongside the raw list).
2. **Raise the throttle with size**: keep 50 ms for the first ~2 KB, then scale
   the interval with length (e.g. `max(0.05, 0.05 * len/2048)`) — no human
   reads faster than ~20 Hz and a 20 KB answer does not need 20 fps.
3. **Skip the re-parse at stop**: `Live(transient=False)` already froze the last
   frame; re-render only if the tail changed since the last update (a saved
   `len` comparison is enough).
4. Consider plain append-only writes for answers with no markdown structure
   (no `#`, `` ` ``, `-`, `|`, `*` in the tail) and reserve `Live`+`Markdown`
   for when formatting actually matters.

---

**PF-05 · MED · `_undo.py:151-160` `persist_undo_stack` — full-stack re-serialize per edit**

Every `write_file`/`edit_file`/`multi_edit`/`delete_file` re-serializes the
**entire** undo stack with `json.dumps(..., indent=1)` — up to `NX_MAX_UNDO`
(20) entries, each holding whole file contents. Editing a 200 KB file twenty
times in a row rewrites ~40 MB of JSON. CPU is survivable; write amplification
and the burst latency on the interactive path are not.

**Fix:** persist one JSONL line per checkpoint and truncate on load, or
amortize — write the stack at most once per turn (a dirty flag flushed at
turn end, same place as the session autosave). Keep `indent` off the persist
path. Cap snapshot size per entry and record that a snapshot was elided.

---

### P1 — latency tax paid on *every* turn

These are fixed costs repeated per turn; they are the difference between "feels
instant" and "feels like a tool".

**PF-06 · HIGH · `_turn.py:1397-1399` — every turn ends waiting on the interrupt
watcher.** After the stream closes, non-quiet turns do
`_INTERRUPT_RESTORED.wait(timeout=NX_INTERRUPT_RESTORE_WAIT default 2.5)`. The
watcher only notices disarm inside its own poll, so the turn pays that
detection delay **after the last token and before the prompt returns** — exactly
the gap a user perceives as "it's still thinking". Fix: have the disarming
thread set the event directly when the watcher is not mid-drain, or drop the
watcher's detect interval. Target: this wait should be < 10 ms in the common
case, and there should be a test asserting it.

**PF-07 · HIGH · `_metrics.py:287-292` via `_repl_state.py:95-96` — synchronous
dashboard push on the REPL thread.** `urllib.request.urlopen(req, timeout=1.5)`
runs inline, 1–3 times per turn after the answer renders; a unreachable
dashboard costs up to ~4.5 s of dead time before it self-disables
(`_push_disabled`). The comment claims it is fire-and-forget; it is not. Fix:
route through the existing `_AsyncLogWriter` pattern (`_logging.py:32-47`) — the
queue-and-thread already exists in the codebase.

**PF-08 · HIGH · `_bash_jobs.py:755` unconditional `time.sleep(0.2)`** (plus
`time.sleep(0.1)` at `:405` and `NX_BASH_POLL_SECONDS` default 3 at `:587`).
Two agents independently measured ~200–300 ms of pure sleep charged to every
short `run_bash`. Fix: replace the settle sleep with a bounded
wait-for-EOF/flush on the actual pipe, and make completion detection event-based
(`Event` set by the reaper) rather than poll-based.

**PF-09 · MED · `_sessions_fs.py:692-723` title/description generation is a
blocking completion on the main thread** every ~6th turn (`NX_SESSION_DESC_REFRESH`,
20 s timeout), right after the answer renders. Fix: background thread — nothing
consumes the value inline.

**PF-10 · MED · no prompt-cache routing hint is ever sent.** nx *displays*
`cached:` and documents prefix-cache discipline, but sends no stable session
identity, so a vLLM/SGLang/llama.cpp router cannot pin a session to a replica or
KV block: every turn can cold-prefill on a different worker. This is the
cheapest large prefill win available and it costs zero tokens. Fix: in
`Source.extra_request_kwargs()` (`_sources.py:122-154`) add a stable `user` (and
`prompt_cache_key`/`cache_salt` where the backend understands them) derived from
the session id that already exists at `_metrics.py:276`. Must be **opt-out-able**
per source and must never vary within a session (else it becomes a cache-buster
itself). Pair with ROADMAP P3-10 (capability manifest) so unsupported fields are
probed once, not guessed.

**PF-11 · MED · `_compress.py:575-598` field-rejection ladder re-derives the
accepted request shape by failing.** Up to 3 fresh POSTs of a full prompt per
turn against an endpoint that already rejected it — the code comments at
`:564-568` acknowledge the amplification. Fix: probe once per
`(source, model)` and pin the reduced shape (this is the same cached-manifest
work as PF-10 / ROADMAP P3-10).

---

### P2 — prefix-cache integrity (the token-billing half)

ROADMAP theme 1 already shipped the append-only recovery-nudge fix (1.1,
0.40.0) and compression v2. Four cache-killers remain, verified against current
code.

**PF-12 · HIGH · `_memory_cmd.py:756-787` `_refresh_memory_note` rewrites a
message at index 1.** `messages[at] = note` (`:786`), `del messages[at]` (`:782`),
`messages.insert(1, note)` (`:779`). Position 1 is the *worst possible* place:
it busts the server prefix cache for the entire remaining prompt, on
`/memory optimize` and on every resume where the store changed. The docstring at
`:764-773` marks this "deliberately paid", and the reasoning (a stale memory
index makes the model recite archived titles) is sound — but the placement is
not. Fix: keep the note **at the tail** as a `[Runtime note: …]` message so the
prefix survives, or accept the bust only on explicit `/memory optimize` and
never implicitly on resume. At minimum, split "correctness of the note" from
"position of the note".

**PF-13 · HIGH · `_images.py:136-139` base64 image payloads live in a `tool`
message and are re-sent on every subsequent request.** Up to
`NX_IMAGE_MAX_BYTES` (8 MB raw ⇒ ~11 MB base64) per image, per turn, until an
overflow happens to evict it. The 0.39.0 sidecar store moved the *disk* copy out
of session JSON; the *wire* copy is the expensive one.

**The hard part already exists.** `_strip_image_parts` (`_images.py:161-185`)
correctly swaps an `image_url` part for a one-line text marker, in place, and
keeps the `[image: path, N bytes]` memory for the model — but it is called only
from the overflow/compress rescue path (`_compress.py:434-435`), i.e. *after*
the payload has already cost a 400. Fix: call the same helper when a turn
completes having consumed the image (the `image_consumers` / `_verify.py` model
already knows what consumed it), instead of waiting for an overrun to force it.
Add a per-turn payload byte counter to the bench harness so this cannot regress.

**PF-14 · MED · `_compress.py:451-457` + `:685-709` `_spill_tool_results`
mutates `messages[i]["content"]` in place** (up to the 6 largest tool results)
— a mid-transcript rewrite, i.e. a cache bust on exactly the turns where the
prompt is largest and compression is already expensive. This is in tension with
compression v2's "never re-summarize the head" goal. Fix: spill at append time
(before the result is ever sent) rather than retroactively, so the wire bytes
never change once sent.

**PF-15 · MED · `_dispatch.py:209-241` `_initial_messages()` appends four notes
at indices 1–4** (AGENTS.md ≤8000 chars, memory title index, orientation, …) and
`_resume.py:50-51` rebuilds the system message wholesale on version-skipped
resumes. Each is a position/content change to the prefix. Fix: a single
canonical "prefix block" built by one function, byte-stable within a session,
with an explicit `pin_prefix()` so a regression shows up as a diff rather than a
mystery `cached: 0%`.

---

### P3 — growth that bites only after months of use

**PF-16 · HIGH · unbounded process-global caches.** `st._TOOL_RESULT_CACHE`
(`_state.py:403`) has no cap or eviction — entries hold whole file/search
results. `_ASYNC_AGENTS` (`_subagent.py:565`) never evicts finished records and
holds full sub-agent answers (line 927 collects finished ids but the ledger
keeps growing). With 8+ concurrent sub-agents this is both a memory and a GIL
concern. Fix: bounded LRU with a byte budget on the tool cache, and evict
finished agent records once their result has been delivered to the parent.

**PF-17 · MED · `_subagent.py:700-726` unbounded async fan-out** — each async
sub-agent is its own thread + HTTP stream + parser. Fix: a semaphore
(`NX_SUBAGENT_MAX_CONCURRENT`, default 4) with a queued status message.

**PF-18 · MED · tool-result cache validity is too weak for two of the seven
cached read-only tools.** The key is the whole `json.dumps(args)`
(`_dispatch.py:413-414`) but the invalidation anchor is `path_arg[0]` only
(`_tool_cache.py:122-125`), and `_invalidate_tool_cache_path` matches one exact
path — so editing file #2..N of a multi-path `read_file` never invalidates.
`search_files` anchors on *directory* mtime, which does not move when a file in
a subdirectory changes. These are correctness bugs with a performance shape
(stale hits), so fix alongside ROADMAP theme 2 (turn opinions into invariants).

**PF-19 · LOW-MED · `_approval.py:45-46, :76-77` diffs are unbounded in output
size.** CPU is fine (9.8 ms on 30k lines) but a one-line change in a repetitive
30k-line file produces a **520 KB** diff. It is capped downstream by the
overflow/`TOOL_RESULT_CHARS` path, so it costs CPU + a spill file rather than
tokens — but `_diff_hunk` output *is* fed back to the model, so a hunk cap
(~8 KB) would make the behaviour explicit instead of relying on a later filter.

**PF-20 · LOW-MED · `_argrepair.py:229, :317` run `difflib.SequenceMatcher`
over tool-name candidates on every malformed-arg repair.** Bounded (19 tools)
and off the common path, so low priority; pin it only if the breaker starts
firing often.

**PF-21 · LOW · `_sources.py:197-217` `resolve_model()` is never memoized** —
every `/source`, resume, and `resolve_context_window()` fallback pays a fresh
`/v1/models` (10 s cap) + `/props` (5 s). Fold into the PF-10/PF-11 cached
manifest.

**PF-22 · LOW · `_openai_client.py:254` `iter_lines()` + `_Obj` re-wrapping.**
One extra whole-buffer decode/split per read, and `_Obj.__getattr__`
re-wraps dicts/lists per attribute read (~5-8 allocations per token). Fix:
`iter_raw()` + split on `b"\n\n"` + `json.loads(bytes)`, and read
`obj["choices"][0]["delta"]` straight off the dict. Small, safe, per-token.

**PF-23 · LOW · `_turn.py:1411` `sum(len(str(m.get("content") or "")) for m in
messages)`** scans the whole prompt to compute a fallback estimate even when the
server reported `usage`. Fix: skip when usage is present.

---

## 3. Ruled out — do not spend a release on these

Measured or traced to conclusion. Recorded so the next audit does not re-open
them, and because several were flagged by agents and then refuted.

| Claim | Verdict | Evidence |
| --- | --- | --- |
| "Per-token `json.dumps` in `_LoggingStream` (`_turn.py:66-73`) is CRITICAL" | **Refuted.** | Measured **1.2 µs/chunk** ⇒ 6 ms per 5000 tokens. The writer is already async (`_logging.py:94` `_AsyncLogWriter`), so the queue `put` is the only inline cost. *Worth* adding an enable flag for disk growth (default log cap is `NX_LOG_MAX_BYTES` = 10 MB and it is undocumented), but it is not a latency bug. |
| "`json.dumps(messages + tools)` per request is CRITICAL" | **Refuted as critical.** | Measured 0.2 ms for 160 KB of messages + schemas. Unavoidable anyway — the payload must be serialized. |
| "Build a concurrent read-only tool dispatcher" | **Already shipped (0.31.0).** | `_turn.py:419-510` runs a bounded pool, `_turn.py:844-846` appends results in tool-call index order, `_prefetch_cacheable_tools` (`_turn.py:564-618`) warms the cache for mixed turns. Only gap: no kill switch. **Don't build.** |
| "conftest's per-test state rewind will make the suite slow as features are added" | **Refuted for now.** | Measured 0.43 ms per `_restore()` ⇒ ~1.0 s of an 11.9 s suite. Do add a budget test (§5, T-9) so it *stays* true, but do not refactor it now. |
| "Session listing/resume re-reads every session file" | **Refuted.** | The 2.3b SQLite index works: save cost is flat in session count (~7 ms at 1000 sessions), `_list_sessions()` 1.5 ms, load 1.6 ms. |
| "httpx creates a client per request / no connection reuse" | **Refuted.** | Exactly one `httpx.Client` at `_openai_client.py:399`, held as `st.client`. Timeouts correctly configured (`NX_HTTP_TIMEOUT_CONNECT` 10 s / `NX_HTTP_TIMEOUT_READ` 900 s). |
| "Idle TUI burns CPU polling" | **Refuted on Linux.** | `_tty.py:174`'s 10 ms loop is the Windows branch; POSIX uses `select()`. `_resources.py`'s interval loops run only for the on-demand `/resources` command. |
| "Token estimation re-tokenizes the prompt every turn" | **Refuted.** | Runtime estimator is `len()/3.5`; tiktoken is used only in `/ctx` and tests, behind an import guard. |
| "difflib is a pathological O(n·m) stall" | **Refuted as CPU risk** (see PF-19 for the real output-size issue). | 34 ms at 50k lines; 9.8 ms at 30k repetitive lines. |
| "Startup does blocking network calls" | **Mostly already handled.** | `NX_SKIP_STARTUP_PROBES` exists; probes are background threads (`_capability.py:217`, `_thinking.py:162`, `_turn.py:194`). Residual: `switch_source()`'s `resolve_model()` (PF-21). |

---

## 4. Performance invariants (the anti-lag contract)

These are the rules that keep feature #30 as cheap as feature #5. Each maps to
a test in §5. **A change that violates one of these needs a ledger entry
justifying the cost, exactly like the token floor already requires.**

| ID | Invariant | Why it prevents the laggy TUI |
| --- | --- | --- |
| **I-1** | `render()` in `_chatbox.py` performs **no filesystem, subprocess, or network I/O** and does no work proportional to repo size. | A keystroke must never touch disk. This is PF-01's regression fence. |
| **I-2** | Per-chunk stream handling is **O(chunk)**: no scan, regex, or render over the whole accumulated buffer except on the throttled path. | Kills the whole PF-02/PF-03/PF-04 family permanently, including future additions. |
| **I-3** | A message older than the current turn boundary is **never mutated or inserted before the end**. | Prefix cache. Extends ROADMAP 1.1 to the remaining violations (PF-12/14/15). |
| **I-4** | `SYSTEM` and the prefix block are **byte-identical across turns** within a session. | The prefix cache's precondition; catches dynamic date/cwd/branch/flag interpolation. |
| **I-5** | Per-turn token floor `<= BUDGET_TOKENS`, **with ≥ 50 tokens of headroom maintained.** | Currently 3177/3180 — the next tool breaks the suite. See §6 for the process. |
| **I-6** | **No fixed sleep on the interactive path.** Any wait is bounded by the event it waits for. | Kills the PF-06/PF-08 tax class. |
| **I-7** | Nothing on the REPL thread performs a blocking network call with a timeout > 100 ms. | PF-07/PF-09 class. |
| **I-8** | Every cache has an **explicit bound** (entries or bytes) and an eviction path. | PF-16; long sessions must have a flat RSS curve. |
| **I-9** | Every per-turn HTTP request count is **known and pinned by a test**. | Prevents "silent extra round-trip" drift as features add probes/indexes/telemetry. |
| **I-10** | Any new `NX_*` tunable is documented in the README table **in the same commit** (extends the existing docs-parity test to flag *undocumented* vars, not just wrong ones). | Currently 150 distinct vars, **35 undocumented**; the drift guard only checks the README→code direction today. |

---

## 5. Bench harness — `tests/perf/`

Optimization without a pinned number regresses within two releases. The suite
stays offline and fast: everything is a pure function, a fake chunk object, or a
temp dir. Ratio-based assertions (not absolute ms) so a loaded CI box does not
flap; `--perf-tolerance` (default 1.25) multiplies every bound; `best_of(5)` and
a recorded `tests/perf/baselines.json` ledger, mirroring the repo's existing
"recorded budget + justification ledger" convention.

| ID | Asserts | Sizes | Bound (measured today) | Pins |
| --- | --- | --- | --- | --- |
| T-1 | `_first_tool_tag_index`-equivalent per-chunk cost is flat | 500 → 8000 chunks | `last/first < 4` (**today ≈ 32**) | PF-02 |
| T-2 | `StreamParser.process()` total scales linearly | 2000 vs 8000 chunks, `quiet=True` | `t8/t2 < 5.5` (**today ≈ 13**) | PF-02/03/04 |
| T-3 | `_match_file_paths` is O(1) after warmup | 20k-entry tmp tree, 10 prefixes | warm call `< 1 ms` (**today 38–56 ms**) | PF-01 |
| T-4 | `render()` performs no I/O | monkeypatch `os.scandir`/`subprocess.run`/`os.stat` | **0 calls** during render | PF-01, I-1 |
| T-5 | Markdown re-render per frame stays sub-linear | 8 KB vs 288 KB | ratio `< 20` | PF-04 |
| T-6 | No message older than the turn boundary mutates | synthetic 30-turn transcript | **0 mutations** | PF-12/14/15, I-3 |
| T-7 | `SYSTEM` + prefix block byte-identical over 20 turns | n/a | `==` | I-4 |
| T-8 | Turn-end interrupt handshake wait | mock stream | `< 50 ms` (**today up to 2500 ms**) | PF-06 |
| T-9 | conftest snapshot budget | static: `len(_BASELINE_MODS)`, total entries | under a recorded budget | keeps §3's ruling true |
| T-10 | Tool-result cache and `_ASYNC_AGENTS` are bounded | 5000 inserts / 50 finished agents | `len <= cap` | PF-16, I-8 |
| T-11 | HTTP requests per turn, against a local mock | plain turn, turn + 1 tool call | `== N` recorded | I-9 |
| T-12 | Session save amortizes rather than rewrites per step | 50 → 300 msgs | ratio `< 2` (**today ≈ 6**) | PF-05 |
| T-13 | No blocking `urlopen`/`Client` call reachable from `_save_current` | static + monkeypatch | **0 inline** | PF-07, I-7 |
| T-14 | Floor budget with headroom (extends the existing test) | n/a | `<= BUDGET - 50` | I-5 |
| T-15 | Static: no literal-first-arg `re.<fn>()` inside functions; no duplicated pattern source | AST scan of `nx/*.py` | `== 0` (small allowlist) | regex hygiene |

Static tests (T-4, T-7, T-9, T-15) are cheap enough to run in the **normal**
suite; the rest carry a `perf` marker excluded from `pytest tests/` by default
so the 11.9 s / `-n auto` baseline is untouched, and run via `pytest tests/perf
--perf`. Keep `NX_HOME`/`NX_SESSIONS_DIR` pointed at `tmp_path` as the rest of
the suite already does; use `_Obj`-based fake chunks so `StreamParser` is
benchmarked without pydantic or a socket.

**A prototype harness already exists in the audit trail** (mock SSE server, pty
driver, TTFT/throughput/write-count/request-count probes) and should be
seeded from it rather than written from scratch.

---

## 6. Making features stay cheap (the personal-assistant part)

The stated goal is to keep stacking features. Here is the marginal cost of each
extension point, measured/counted today:

| Extension point | Per turn | Per keystroke | Per startup | Grows with N features? |
| --- | --- | --- | --- | --- |
| New slash command | ~0 (dict hit, `_repl.py:231`) | `/` dropdown: `_fuzzy_filter` over all commands, per keystroke | ~0 | Dropdown yes (trivial); dispatch no |
| **New always-on tool** | **~150 tok of permanent rent** (19 tools = 2789 tok today) | — | ~0 | **Yes — linear, forever, with no tiering mechanism** |
| New gated/tiered tool | 0 until used | — | ~0 | No |
| New `NX_*` var | ~0 (`os.environ` read) | — | ~0 | No cost, but 35/150 are undocumented today |
| New hook/callback | — | — | — | `_hooks.py` is subprocess-per-event, not a bus; keep it off the keystroke path |
| New feature module | ~0 | — | linear in import cost (~1 ms/module vs 97 ms total) | Mildly |

**The only real rent is tool schemas, and there is no mechanism to avoid it.**
`TOOLS` holds 23 schemas (`__init__.py:335-403`); `_active_tools()`
(`_dispatch.py:139-162`) withholds 4 of them (`wait_background`, `spawn_agent`,
`read_image`, `plan_edit`) via five sequential O(N) filter passes over the list
— one per gate *plus* one for the malformed-args breaker — so every gate added
adds another full copy of the schema list per turn, and there is no mechanism
for a tool to cost nothing until it is used.
ROADMAP 1.7 already names the fix, and it is the highest-leverage *structural*
item in this document:

1. **Registry with tiers.** A declarative
   `(schema, gate, tier)` table replacing the hand-written comprehensions, with
   `_active_tools()` memoized on the gate-vector.
2. **A `search_tools` meta-tool** (or `use_tool`) that lifts a tier-2 schema
   into context on demand. Precedent already exists: `wait_background` is
   withheld until a call actually backgrounds. With tiering, tool #30 costs
   ~0 tokens in sessions that never touch it instead of ~150 in all of them.
3. **Trim schema structure, not prose.** Measured split: **1683 tok params vs
   624 tok descriptions.** The fat is `parameters` JSON, so the win is in shared
   `$defs`/compact property descriptions and in the four fattest schemas
   (`search_files` 279, `sed_edit` 204, `edit_file` 201, `read_file` 181), not in
   shortening sentences.
4. **Feature-cost rule (adopt as policy):** *a new always-on tool must name the
   tool it displaces into tier 2, or ship with a ledger entry raising
   `BUDGET_TOKENS`.* Today the floor has 3 tokens of headroom, so the next tool
   simply fails the suite — which is the right forcing function, but make it a
   documented process instead of a surprise.

Secondary: single-source-of-truth for the four hand-synced command lists
(ROADMAP 1.5 shipped parity *tests*; codegen would also make adding a command
genuinely free), and extending the docs-parity test to fail on **undocumented**
`NX_*` vars (I-10).

---

## 7. Phased plan

Small minors, each independently shippable, each with its perf test landed in
the same commit. Semver per AGENTS.md: all of these are user-visible latency or
behavior changes, so minors, with patches for anything that only adds a test.

| Release | Contents | Acceptance |
| --- | --- | --- |
| **0.55.0** — *stop the keystroke hitch* | **PF-01** (tree cache + move I/O out of `render()` + memoize project root), **PF-02** (scan watermark) — **shipped** | T-1, T-3, T-4 green (`tests/test_stream_tag_scan_watermark.py`, `tests/test_chatbox_completion_cache.py`). `@`-completion < 1 ms/keystroke on a 20k-file repo; long-answer stream CPU < 50 ms |
| **0.56.0** — *per-turn tax* | **PF-06** (interrupt handshake), **PF-07** (async metrics push), **PF-08** (bash sleeps), **PF-09** (background title gen) | T-8, T-13 green. Measured gap between last token and prompt return < 50 ms; no single command charged > 20 ms of sleep |
| **0.57.0** — *render path* | **PF-03** (join instead of `+=`), **PF-04** (incremental clean-string accumulation, size-scaled throttle, no double final parse), **PF-22** (`iter_raw` + dict access) | T-2, T-5 green. `StreamParser` linear; no `_content_str` |
| **0.58.0** — *prefix-cache integrity* | **PF-12** (memory note to the tail), **PF-13** (strip image payloads after consumption), **PF-14** (spill at append time), **PF-15** (single pinned prefix block) | T-6, T-7 green. `cached:` stays non-zero across a recovery storm and across resume; per-turn payload bytes do not grow with attached images |
| **0.59.0** — *one cached capability manifest* | **PF-10** (stable cache-routing identity), **PF-11** (pin accepted request shape), **PF-21** (memoize `resolve_model`). Pairs with ROADMAP P3-10 | T-11 green (request count pinned). Zero redundant `/v1/models` per turn |
| **0.60.0** — *bounds and durability* | **PF-16** (bounded caches, agent-record eviction), **PF-17** (sub-agent semaphore), **PF-18** (cache validity anchors), **PF-05** (undo persistence amortized), **PF-19** (diff hunk cap) | T-10, T-12 green. Flat RSS over a 200-turn synthetic session |
| **0.61.0** — *make features cheap* | Tiered tool registry + `search_tools` meta-tool (§6 items 1-2), docs-parity test for undocumented `NX_*`, `tests/perf/` promoted to a supported target with `baselines.json` | T-9, T-14, T-15 green. Adding a tool costs 0 tokens unless used |

Suggested ordering rationale: P0 first because it is cheap, invisible-to-model
work that removes the two worst measured numbers; the per-turn tax next because
it is what users *feel*; render path before prefix-cache work so the bench
harness can measure the cache work's effect; the capability manifest is grouped
because PF-10/11/21 all want the same per-`(source, model)` cache; bounds and
the tiering mechanism last so they land on a stable, already-fast base.

**Do not** start any of these with a new agent-facing feature. The whole point
of doing them first is that the next 20 features inherit a flat cost curve
instead of a bent one.

---

## 8. Audit provenance and caveats

- 17 sub-agent areas: startup/import, TUI/input latency, HTTP/streaming,
  prompt-floor/prefix-cache, tool execution, disk/persistence,
  threading/polling, micro-CPU, scalability-as-features-are-added,
  long-run memory/per-turn copy, plus a prototyped benchmark harness, a
  parallel-dispatch feasibility memo, and re-runs of four areas after a
  transient gateway 502. Every HIGH finding cited here was **independently
  confirmed by at least two agents or by the parent's own measurement**.
- Several auditors could not execute code (read-only sandbox refused
  interpreters), so their numbers are analytic. Where a number mattered, the
  parent measured it; §1 marks `[m]` vs `[i]`, and §3 records the two
  "CRITICAL" claims that failed measurement.
- Not covered by any agent and still open: `_grammar.py` constrained-decoding
  cost per turn, `_batch.py`/`_eval_cmd.py` throughput, `_storeindex.py` FTS5
  accelerator hit rate under real stores, and actual TTFT against a real local
  model (the offline suite cannot measure prefill). The mock-server harness in
  §5 is the first step toward all four.
- Box-relative: absolute ms will move across machines; the **ratios** in §5 are
  the portable part, which is why the harness asserts ratios and not times.
