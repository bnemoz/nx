#!/usr/bin/env python3
"""I-10 (PERF §6): fail on an NX_* var READ in nx/ but not documented in the
README. The reverse direction (documented but never read) is
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

README = open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8").read()

# name -> why it is deliberately not a README env var
_ALLOWLIST = {
    # parser vocabulary / fixture values, not tunables:
    "NX_EMPTY": "string-literal value in a test, not a variable name",
    "NX_QUOTED": "parser vocabulary in a test",
    "NX_UNQUOTED": "parser vocabulary in a test",
    "NX_FOO": "test fixture",
    "NX_MY": "test fixture",
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
}


def test_every_read_nx_var_is_documented():
    names = set()
    for top in ("nx", "tests"):
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, top)):
            for f in files:
                if f.endswith(".py"):
                    names |= set(re.findall(
                        r"NX_[A-Z][A-Z0-9_]*",
                        open(os.path.join(root, f), encoding="utf-8").read()))
    names = {n for n in names
             if "<" not in n and not n.endswith("_")
             and not n.startswith("NX_SOURCE_") and not n.endswith("_COLOR")}
    documented = set(re.findall(r"NX_[A-Z][A-Z0-9_]*", README))
    documented = {n for n in documented
                  if "<" not in n and not n.endswith("_")
                  and not n.startswith("NX_SOURCE_") and not n.endswith("_COLOR")}
    undocumented = sorted(names - documented - set(_ALLOWLIST))
    assert not undocumented, (
        f"NX_* vars read in nx/ but not in the README's env table: "
        f"{undocumented} — document each (AGENTS.md: every NX_* tunable gets a "
        f"README entry) or add it to _ALLOWLIST with a reason if it is not a "
        f"tunable")


def test_allowlist_entries_have_reasons_and_still_exist():
    for reason in _ALLOWLIST.values():
        assert reason and reason.strip()
    # an entry that stops being read anywhere should be pruned
    stale = [k for k in _ALLOWLIST if not any(
        k in open(os.path.join(root, f), encoding="utf-8").read()
        for top in ("nx", "tests")
        for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, top))
        for f in files if f.endswith(".py"))]
    assert not stale, f"allowlist entries no longer read anywhere: {stale}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
