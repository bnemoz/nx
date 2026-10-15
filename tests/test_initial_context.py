#!/usr/bin/env python3
"""Guard the README's "under ~5K tokens on a bare hey" claim.

The README advertises that a bare `hey` costs under ~5K tokens of initial
context. That bounded floor is the whole reason nx exists — it stays trivial
against the 128K-1M-token windows today's models ship with, unlike the
harnesses that spend 20K-50K before you've typed anything — so this test
watches it. If a new tool or a fatter SYSTEM bloats the payload, we want to
know before the README quietly becomes a lie.

WHAT WE MEASURE (the part we control and send to the API):
  - SYSTEM          the system prompt
  - TOOLS           the 7-function tool schema, JSON-encoded
  - "hey"           a minimal first user turn
The per-turn floor is enforced as a recorded BUDGET, not a soft warn: a new
tool or a fatter schema fails the suite until the number is consciously
re-derived (that is the whole point — the 2500→2886 drift of 0.30–0.39 was
visible for months as a warning nobody read).

Measured 0.41, chars/3.5 (see _count_tokens), NX_SUBAGENTS/NX_VISION off —
17 active schemas: **2879** (SYSTEM 375 + TOOLS 2503 + 'hey' 1). With
NX_SUBAGENTS on the set gains spawn_agent (330 tok after the 0.41 trim,
was 396) → 3210. The old soft warn at 2500 predates the schema rewrite and
fired on every run, which is the same as having no line at all.

BUDGET covers the 20-tool default set because that is what a fresh session
pays; enabling sub-agents is a deliberate, user-visible choice. A PR that
moves the floor past BUDGET_TOKENS must either trim elsewhere or raise the
number here — with the justification in the ledger below.

TOKENIZER: prefers tiktoken (cl100k_base) when installed for a real count;
falls back to a conservative chars/3.5 estimate otherwise so the guard still
runs with zero extra dependencies, and we don't add one for a test. The
fallback over-counts ~7% vs cl100k on code/JSON, which is the safe
direction for a guard — it trips slightly early, never late.
"""
import json
import os
import sys
import tempfile

from nx._env import _env_int

# Keep nx from touching ~/.nx — same convention as the other tests,
# even though this one never writes a session.
_tmp = tempfile.mkdtemp(prefix="nx-test-")
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp
# Standard isolation (AGENTS.md): a local env file must not change what we
# measure. Exported NX_SUBAGENTS/NX_VISION are handled at call time below
# (nx resolves them at import, which may happen in another test module first).
os.environ["NX_ENV_FILE"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402

# Thresholds on the content token count (SYSTEM + TOOLS JSON + "hey").
# With the active schema gated by _active_tools() (wait_background excluded
# until a command backgrounds — currently read_file, write_file, edit_file,
# list_dir, run_bash, search_files, git_status, git_diff, git_log, multi_edit,
# sed_edit, todo_write, memory_write, memory_search, memory_delete,
# delete_file) the chars/3.5 content floor is ~2680; the README's figure adds
# ~50 of server-side framing. git_log + apply_patch (two new tools at the time;
# apply_patch was folded back into edit_file's `diff` mode in 0.158.11) plus
# read_file/search_files list-argument params pushed the floor up from
# ~1990; memory_write + memory_search (model-driven memory, two more tools)
# plus the SYSTEM memory line raised it again from ~2263; a 0.3.1 pass that
# trimmed verbose tool descriptions brought it back under 2500; a 0.18.9 pass
# re-streamlined the fattest descriptions (edit_file, search_files, multi_edit,
# sed_edit, run_bash, spawn_agent) which had crept back past the warn line; the
# 0.26 schema rewrite moved characters from terse abbreviation blobs into
# per-PROPERTY guidance (an undocumented `old_lines`/`timeout`/`scope` is a
# guessed-at argument), landing ~2835 — still warning, still under the ceiling.
# 0.41: the 2500 warn line — set when the active set was 7 tools — had been
# firing on every run for months, which is a warning nobody reads. Replaced by
# a recorded budget: BUDGET_TOKENS below, with this ledger as its justification.
# The 0.41 trim pass took spawn_agent 396→330 (trigger-list and async mechanics
# moved to the one-time Delegation note) and edit_file/plan_edit down a few
# dozen; the 17-tool floor now measures 2879.
# 0.49: run_tests (ROADMAP P3-5) joins the always-on set — one small schema
# (a single optional `targets` string, ~131 tok measured by tiktoken on the
# 18-function set); BUDGET_TOKENS rises 2900→3020 by that deliberate,
# verifiable cost.
# 0.51: git_commit (ROADMAP P3-6) joins the always-on set — files/message/path
# strings, ~157 tok measured by the suite's chars//3.5 method; BUDGET_TOKENS
# rises 3020→3180 by that deliberate, verifiable cost.
# 0.62: search_tools (PERF §6 item 2, the tier-2 meta-tool) joins the always-on
# set — one optional `pattern` string, ~118 tok by the chars//3.5 method. The
# feature-cost rule says a new always-on tool names what it displaces: this one
# displaces all FUTURE always-on rent (a later tool can ship tier-2 at 0 tok
# and be surfaced by search_tools on demand instead of riding every request).
# BUDGET_TOKENS rises 3180→3320 by that deliberate, verifiable cost.
# 0.63: policy change, not a feature cost. nx targets models with 128K-1M-token
# windows, so the floor no longer needs to be squeezed toward the minimum — a
# schema should be as descriptive as the model benefits from, not as terse as
# possible. BUDGET_TOKENS rises 3320→4096 to make that headroom explicit:
# "under ~4K" is the published contract, and the ceiling is where descriptive
# (not cryptic) schemas become affordable. The measured floor (~3.3K) is
# unchanged; this is deliberate slack so the next few schemas land descriptive
# instead of compressed. Growth past 4096 still needs a ledger entry.
# 0.87: duckdb (the DuckDB analytical-SQL CLI wrapper, nx/_duckdb.py) joins the
# always-on set — the data-science path the user's sessions live on, and one
# the model should reach for on turn 1 rather than discovering via search_tools
# mid-analysis. One required `sql` string plus two optional strings; the schema
# measured ~1080 chars (~310 tok by the chars/3.5 method) including the
# one-line pandas-steering line added to SYSTEM. BUDGET_TOKENS rises
# 4096→4800 by that deliberate, verifiable cost; the published contract is now
# "under ~5K" (README updated to match). No tier-2 displacement was chosen
# deliberately: search_tools-discovery costs a round-trip exactly when a data
# question is already on the table.
# 0.99.4: the SYSTEM prompt now carries the exact 12-line Minimalism policy,
# replacing the shorter prose block. The measured floor is 4740 tokens with
# chars/3.5; BUDGET_TOKENS rises 4800→5200 to record deliberate headroom for
# this load-bearing policy and keep the prompt budget from landing on the cap.
# 0.109.0: the SYSTEM prompt now tells the model the write fence exists and
# that a fence refusal means "ask, don't reroute" — the model escaped a fence
# refusal through a python here-doc in the wild, and the prompt is the layer
# that stops it trying. ~310 chars (~88 tok by chars/3.5); BUDGET_TOKENS rises
# 5200→5320 by that deliberate, verifiable cost.
# 0.111.0: per-tool usage guidance (search_files-first, todo_write-first,
# memory/recall sequencing, duckdb-over-pandas, sub-agent duckdb steering,
# backgrounding mechanics, multi_edit batching) moved OUT of SYSTEM and INTO
# the tool descriptions where the model reads it at argument-picking time;
# SYSTEM keeps only harness facts (approval, write fence, text protocol,
# concurrency refusal). Net +51 tok (5120→5171): the descriptions were also
# rephrased/tightened rather than taking the moved text verbatim. Budget
# stays 5320 — headroom absorbs it.
# 0.145.3: SYSTEM gains one line prohibiting filler words (Wait/Hmm/Actually and
# combinations) in reasoning traces. ~160 chars; the measured floor lands at
# 5342, so BUDGET_TOKENS rises 5320→5360 by that deliberate, verifiable cost.
# 0.113.0: web_fetch promoted tier-2 → tier-1 (schema joins TOOLS whenever
# NX_WEBFETCH != 0): it is exactly the tool a session needs on turn 1 (the
# outside world), and search_tools discovery cost a round-trip precisely
# then. The schema grew ~30 chars (revalidation/robots line). Measured floor
# rises by ~180 tok; BUDGET_TOKENS stays 5320 — headroom absorbs it (the
# floor was 5171 after 0.111.0).
# 0.132.0: critique (the scoped minimal reviewer) joins TOOLS — and the budget
# does NOT move, deliberately. Measured 0.132.0 (chars//3.5, this test's
# method, NX_SUBAGENTS/NX_VISION off): 5276 before and 5276 after, i.e. the
# reviewer costs ZERO tokens on the guarded floor. That is the whole point of
# its placement: its schema is GATED on NX_SUBAGENTS (the same switch that
# withholds spawn_agent), and this test measures with sub-agents off — enabling
# them is the deliberate, user-visible opt-in this ledger already puts out of
# scope. The alternative (an always-on `critique` schema) was rejected for the
# same reason a new always-on tool always is: ~600 tok of permanent rent —
# ~11% of the 5320 budget — for a tool a session calls a handful of times. So
# the raise this entry would otherwise justify was NOT taken; BUDGET_TOKENS
# stays 5320 and the same guard keeps covering the reviewer whenever it is
# enabled (with NX_SUBAGENTS=1 the floor is 6780, which is what a sub-agent
# session already paid for spawn_agent + friends, not a new line item).
# 0.147.0: SYSTEM gains a four-line Security block (never reproduce
# credentials from tool output / treat [redacted:…] spans as opaque / never
# echo the system prompt or schemas / tool results are data not
# instructions). ~325 chars; measured floor rises 5342→5445. Budget raised
# 5360→6000 at the user's direction: a deliberate one-shot widening to give
# future prompt/schema growth room without a ledger bump every time (the
# guard still fails loudly once 6000 is crossed).
# 0.158.x: memory_search gains one optional `include_sessions` boolean (opt-in
# reciprocal-rank fusion of the note arm with recall_lessons' session arm — one
# property on a schema already on the floor, chosen over a new tool schema,
# which would have cost hundreds of tokens for every session). Measured floor
# (chars/3.5, this test's method, NX_SUBAGENTS/NX_VISION off): 5167 -> 5223, a
# +56 tok delta. BUDGET_TOKENS stays 6000 — headroom absorbs it.
# 0.158.20: memory_write gains one optional `supersedes` string, so supersession
# becomes DATA (the retired note is stamped and filtered) instead of the
# docstring plea that told the model to memory_delete the stale one. One
# property on a schema already on the floor, again chosen over a new tool.
# Measured floor (same method): 5223 -> 5286, a +63 tok delta. BUDGET_TOKENS
# stays 6000 — headroom absorbs it. (The redaction pass and the `_summary_of:`
# footer field cost ZERO floor tokens: both are store-side, and the redaction
# notice rides a tool RESULT, not the per-turn schema.)
BUDGET_TOKENS = _env_int("NX_FLOOR_BUDGET", 6000)  # hard: recorded, justified
SOFT_TOKENS = BUDGET_TOKENS - 50  # advisory: floor drift is under way, trim before it lands


def _count_tokens(text):
    """Token count for a string.

    Returns (count, method). Uses tiktoken's cl100k_base when available
    (the closest thing to a shared standard across served models); falls
    back to a conservative chars/3.5 estimate so the test runs with no
    extra deps. The fallback over-counts on code/JSON, which is the safe
    direction for a bloat guard.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken"
    except Exception:
        return max(1, round(len(text) / 3.5)), "chars/3.5"


def test_initial_context_floor():
    # The exact payload a fresh "hey" turn sends: system prompt + the ACTIVE
    # tool schemas (_active_tools() — excludes wait_background until a
    # run_bash call has actually backgrounded this session, see
    # _SESSION_HAS_BACKGROUNDED) + a one-token user message. FINAL_ANSWER_TOOL
    # is intentionally excluded — it's only injected on a forced-final retry
    # (nx.py:1847), not on turn 1, so it's not part of the README's ~4K floor.
    # Opt-in tools are forced OFF for the measurement: a developer shell with
    # NX_SUBAGENTS=1/NX_VISION=1 must not change the guarded baseline (nx
    # resolves those at import, possibly in another test module, so set the
    # module attributes rather than the env vars).
    saved = (m.NX_SUBAGENTS, m.NX_VISION)
    m.NX_SUBAGENTS = m.NX_VISION = False
    try:
        active_tools = m._active_tools()
        system_tokens, method = _count_tokens(m.SYSTEM)
        tools_tokens, _ = _count_tokens(json.dumps(active_tools))
        user_tokens, _ = _count_tokens("hey")
    finally:
        m.NX_SUBAGENTS, m.NX_VISION = saved
    total = system_tokens + tools_tokens + user_tokens

    breakdown = (
        f"initial context content = {total} tokens ({method}): "
        f"SYSTEM {system_tokens} + TOOLS {tools_tokens} "
        f"({len(active_tools)} funcs) + 'hey' {user_tokens}"
    )
    print(breakdown)

    # Sanity: a broken payload (empty SYSTEM, TOOLS clobbered to []) would
    # make a token-floor test meaningless by passing on a near-zero count.
    assert m.SYSTEM.strip(), "SYSTEM prompt is empty"
    assert len(active_tools) >= 1, "TOOLS is empty — nothing to guard"
    assert "wait_background" not in json.dumps(active_tools), (
        "a fresh session (nothing backgrounded yet) should not pay for the "
        "wait_background tool schema"
    )

    # Recorded budget (see the ledger above BUDGET_TOKENS): the floor may not
    # grow silently. Raising this number is a deliberate act that must say why
    # in the ledger comment; NX_FLOOR_BUDGET exists for CI experiments, not
    # for shipping past the budget.
    assert total <= BUDGET_TOKENS, (
        f"initial context content is {total} tokens "
        f"> budget {BUDGET_TOKENS} — trim a schema or raise the recorded "
        f"budget WITH a ledger justification. {breakdown}"
    )

    # Advisory: 50 tokens of headroom left. Still passes, but this is the
    # point to trim the fattest description rather than land on the budget.
    if total >= SOFT_TOKENS:
        print(f"NOTE: initial context {total} tok is within 50 of the "
              f"{BUDGET_TOKENS}-tok floor budget. {breakdown}")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
