#!/usr/bin/env python3
"""I-10 (PERF §6): fail on an NX_* var READ in nx/ but not documented in
docs/REFERENCE.md (the README is intentionally thin; the complete env table
lives there). The reverse direction (documented but never read) is
test_docs_parity.test_every_documented_env_var_is_actually_read.

Deliberate non-variables (test fixtures like NX_FOO/NX_MY_TOKEN, the
NX_PRICE_* pricing table read dynamically, NX_QUOTED/NX_UNQUOTED parser
vocabulary) are allowlisted with a reason each — an allowlist entry without a
reason fails, so the list can't become a dumping ground.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

REFERENCE = open(os.path.join(REPO_ROOT, "docs", "REFERENCE.md"),
                 encoding="utf-8").read()

# name -> why it is deliberately not a README env var
_ALLOWLIST = {
    # parser vocabulary / fixture values, not tunables:
    "NX_EMPTY": "string-literal value in a test, not a variable name",
    "NX_QUOTED": "parser vocabulary in a test",
    "NX_UNQUOTED": "parser vocabulary in a test",
    "NX_FOO": "test fixture",
    "NX_MY": "test fixture",
    "NX_TEST_URLOPEN_HOLD": "test-suite only: shrinks the busy-keep sleep in test_perf_turn_tax",
    # matched inside the private module global _NX_STATE_DIR_RE in
    # nx/_clobber.py — a regex object, not an env var (0.145.4 residue)
    "NX_STATE_DIR_RE": "private identifier substring, not an env var",
    "NX_MY_TOKEN": "test fixture",
    "NX_OTHER": "test fixture",
    "NX_MISSING_API_KEY": "test fixture",
    "NX_NOPE_XYZ": "test fixture",
    # pricing keys are looked up dynamically (NX_PRICE_<SOURCE>_<IO> family
    # shape); the family, not the instances, is what the README documents:
    "NX_PRICE_HOSTED": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_BIG_INPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_BIG_MODEL_INPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_BIG_MODEL_OUTPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_BIG_OUTPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_INPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_OUTPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_HOSTED_QWEN_QWEN3_32B_INPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_LOCAL_CACHE": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_LOCAL_INPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_LOCAL_OUTPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_ZAI_CACHE": "NX_PRICE_<SOURCE>_<IO> family instance",
    "NX_PRICE_ZAI_INPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    # NX_KEYS: a comment mention only, not an env read (the real config keys
    # are NX_API_KEY/NX_SOURCE_*; /config lists provider keys by their real
    # names) — nothing to document.
    "NX_KEYS": "comment mention in /config, not a read variable",
    # NX_APPROVAL_POLICY: documented in REFERENCE since 0.135.1.
    # protocol tag regex NAMES in _text_fallback/_stream_parse — module-level
    # compiled-pattern identifiers, not environment variables:
    "NX_OPEN_TAG": "compiled regex identifier, not an env var",
    "NX_CLOSE_TAG": "compiled regex identifier, not an env var",
    "NX_TOOL_TAG": "comment mention of a compiled tag constant, not an env var",
    # deliberate-internal tunables (power users; keep out of the README table
    # on purpose — each has a safe default and a code comment):
    "NX_REASONING_BUDGET_TOKENS": "state attr name; the env var is NX_REASONING_BUDGET",
    "NX_REASONING_CAPTURE_CHARS": "internal: reasoning capture window (default 8000)",
    "NX_REASONING_ONLY_CHARS": "internal: reasoning-stall char threshold (default 36000)",
    "NX_REASONING_ONLY_TIME": "internal: reasoning-stall time threshold s (default 120)",
    "NX_FORCED_FINAL_MAX_TOKENS": "internal: forced-final answer cap (default 2048)",
    "NX_RECOVERY_MIN_P": "internal: recovery sampling min_p (default 0.02)",
    "NX_RECOVERY_REPEAT_PENALTY": "internal: recovery repeat penalty (default 1.2)",
    "NX_RECOVERY_REPEAT_LAST_N": "internal: recovery repeat window (default 512)",
    "NX_RECOVERY_DRY_MULTIPLIER": "internal: recovery dry multiplier (default 0.8)",
    "NX_RECOVERY_DRY_BASE": "internal: recovery dry base (default 1.75)",
    "NX_RECOVERY_DRY_ALLOWED_LENGTH": "internal: recovery dry length (default 2)",
    "NX_LEAKED_TOKEN_RUN_LIMIT": "internal: leaked-token degeneration threshold (default 3)",
    "NX_REPEAT_CHUNK_LIMIT": "internal: repeat-collapse chunk threshold (default 6)",
    "NX_RISK_RETRIES": "internal: risk-classifier connection retries (default 3)",
    "NX_RISK_RETRY_SECONDS": "internal: risk-classifier retry backoff s (default 1)",
    "NX_PRICE_ZAI_OUTPUT": "NX_PRICE_<SOURCE>_<IO> family instance",
    # swarm-internal sweep knob (P1): restores the legacy eager tree walk in
    # drop_changed_entries; default fast-path needs no README entry:
    "NX_TOOL_CACHE_SWEEP_TREE_WALK": "internal: eager tree-walk in the cache sweep (default off)",
    # swarm P4: gate for the full request/response JSONL log (default on):
    "NX_LOG_REQUESTS": "internal: 0 logs stubs instead of full req/resp bodies (default 1)",
    # swarm P5: nested sub-agent fan-out opt-in (default off):
    "NX_SUBAGENT_MAX_NESTED": "internal: 1 lets a sub-agent fan out (default 0)",
    # swarm P6: amortized session-store maintenance (prune + image GC):
    "NX_SESSION_SWEEP_INTERVAL": "internal: min seconds between session-store sweeps (default 300)",
    # swarm P7: shared prompt-cache routing key for a swarm (co-locates
    # agents that share prefix bytes on one replica/KV block):
    "NX_CACHE_ROUTING_GROUP": "internal: shared prompt_cache_key group ('1' = derive from project+model)",
    # swarm P3 knobs (per-process memory): undo stack byte cap and the
    # retained async-answer body in a finished job record:
    "NX_MAX_UNDO_BYTES": "internal: undo checkpoint stack byte cap (default 32MB)",
    "NX_SUBAGENT_ANSWER_KEEP": "internal: retained async answer head+tail (default 4000)",
    # swarm P11: per-call SQLite busy wait on the two stores (was a hardcoded
    # 5 s block per contending save/search under a shared NX_HOME):
    "NX_STORE_BUSY_TIMEOUT_MS": "internal: store busy_timeout ms (default 750)",
    # swarm P12: stdin interrupt-watcher kill switch for headless runs
    # (no tty to steal — the thread + termios snapshot is per-process rent):
    "NX_INTERRUPT_WATCHER": "internal: 0 disables the stdin Esc watcher (default 1)",
    # swarm P13: tier-2 tool-floor slimming for swarms (demotes duckdb +
    # read_image behind search_tools; default keeps the 0.87 ledger set):
    "NX_TOOL_FLOOR_SLIM": "internal: 1 withholds slim-set schemas behind search_tools (default 0)",
    # swarm P10: the shared content-addressed tool-result store (obsstore):
    "NX_OBS_MAX_TRACKED": "internal: bounded in-process obs send counters (default 4096)",
    "NX_OBS_STORE_TTL_DAYS": "internal: obsstore object TTL days (default 7)",
    "NX_OBS_STORE_MAX_MB": "internal: obsstore byte quota MB (default 512, 0=off)",
    # churn nudge (0.117.0): turns-without-a-write before a write-tier agent
    # gets the "place your first edit" steering note (default 8):
    "NX_SUBAGENT_CHURN_NUDGE_TURN": "internal: write-tier churn nudge trigger turns (default 8)",
    # floor-guard test-only escape hatch (durability audit P2): widens the
    # near-cap window in test_initial_context for CI experiments; the budget
    # itself (NX_FLOOR_BUDGET) is still hard:
    "NX_FLOOR_SOFT_WINDOW": "test-suite only: widens the initial-context near-cap assert window",
    # test_mcp's env-passthrough probe (asserted NOT forwarded to MCP servers):
    "NX_SECRET_PASSTHROUGH_PROBE": "test fixture: asserted absent from a server's env in tests/test_mcp.py",
}


_PARITY_TEST_FILES = {"test_docs_parity.py", "test_env_docs_parity.py"}


def _repo_py_files():
    """Every repo .py file EXCEPT this parity test's own file. The staleness
    check below reads name occurrences; the parity test files themselves
    mention every allowlisted name (docstrings, the _ALLOWLIST literal), so
    including them made the staleness check self-certifying — it could never
    fail. Durability audit P2."""
    for top in ("nx", "tests"):
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, top)):
            for f in files:
                if f.endswith(".py") and f not in _PARITY_TEST_FILES:
                    yield os.path.join(root, f)


def test_every_read_nx_var_is_documented():
    names = set()
    for path in _repo_py_files():
        names |= set(re.findall(
            r"NX_[A-Z][A-Z0-9_]*",
            open(path, encoding="utf-8").read()))
    names = {n for n in names
             if "<" not in n and not n.endswith("_")
             and not n.startswith("NX_SOURCE_") and not n.endswith("_COLOR")}
    documented = set(re.findall(r"NX_[A-Z][A-Z0-9_]*", REFERENCE))
    documented = {n for n in documented
                  if "<" not in n and not n.endswith("_")
                  and not n.startswith("NX_SOURCE_") and not n.endswith("_COLOR")}
    undocumented = sorted(names - documented - set(_ALLOWLIST))
    assert not undocumented, (
        f"NX_* vars read in nx/ but not in the REFERENCE env table: "
        f"{undocumented} — document each (AGENTS.md: every NX_* tunable gets a "
        f"docs/REFERENCE.md entry) or add it to _ALLOWLIST with a reason if it is not a "
        f"tunable")


def test_allowlist_entries_have_reasons_and_still_exist():
    for reason in _ALLOWLIST.values():
        assert reason and reason.strip()
    # an entry that stops being read anywhere should be pruned. The walk
    # excludes these parity test files (see _repo_py_files): this file's own
    # _ALLOWLIST literal would otherwise satisfy the containment check and
    # make the staleness assertion vacuous (durability audit P2).
    # NX_PRICE_ family instances are resolved dynamically by prefix in
    # nx/_cost.py (no literal occurrence) — their reader is pinned by the
    # _price_for mechanism, not by name, so they're exempt from staleness.
    stale = [k for k in _ALLOWLIST
             if not k.startswith("NX_PRICE_") and not any(
        k in open(path, encoding="utf-8").read()
        for path in _repo_py_files())]
    assert not stale, f"allowlist entries no longer read anywhere: {stale}"


def test_staleness_check_catches_a_dead_allowlist_key():
    """Anti-vacuity pin: a planted dead allowlist key (nothing reads it) must
    be flagged by the same staleness logic the live test uses — and must NOT
    be flagged when the parity test files are (wrongly) included in the walk,
    pinning the exclusion that makes the check able to fail."""
    dead = "NX_PLANTED_DEAD_ALLOWLIST_KEY"
    planted = dict(_ALLOWLIST)
    planted[dead] = "planted: nothing reads this anywhere"

    def _stale(allowlist, files):
        return [k for k in allowlist if not any(
            k in open(p, encoding="utf-8").read() for p in files)]

    assert dead in _stale(planted, _repo_py_files())
    all_files = list(_repo_py_files()) + [__file__]
    assert dead not in _stale(planted, all_files)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
