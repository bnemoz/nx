# tests/perf — the pinned-number harness (PERF §5)

Optimization without a pinned number regresses within two releases. Everything
in this directory is static or ratio-based (no absolute-ms assertions), offline,
and fast enough to run in the **normal** suite (`pytest tests/` picks it up).

- `baselines.json` — the recorded ledger: each pinned number, the release that
  measured it, and its budget. Re-recording a number is a deliberate act:
  update `measured_in` and say why in the entry's `note`.
- `test_static_budgets.py` — **T-9** (conftest snapshot budget: the per-test
  `_restore()` cost scales with `len(_BASELINE_MODS)` and its entry count) and
  **T-15** (no literal-first `re.<fn>()` inside a function body — hoist to a
  module-level `re.compile`; the allowlist is empty and every entry needs a
  reason).

Ratio/bench tests that need a live-ish harness (mock SSE server, pty driver)
are added here with a `perf` marker and excluded from the default suite; see
PERF §5 for the T-1..T-15 table and which are still open.
