#!/usr/bin/env python3
"""T-9 + T-15 (PERF §5): static budget checks cheap enough for the normal suite.

T-9 — conftest snapshot budget: len(_BASELINE_MODS) and its total entry count
      stay under recorded numbers in tests/perf/baselines.json, so the per-test
      _restore() cost (0.43 ms/test at 57 modules) can't silently grow.

T-15 — regex hygiene: no literal-first-argument re.<fn>() calls inside a
      function body (recompilation per call); module-level compiled patterns
      are the expected shape. Small allowlist for deliberate one-shot literals.
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

_BASELINES = json.load(open(os.path.join(HERE, "baselines.json")))

# file:lineno — literal-first re.* calls that are deliberately one-shot
# (cold path, called at most a handful of times per session). A new entry
# needs a reason in the comment it points at.
_REGEX_ALLOWLIST = set()


def test_t9_conftest_snapshot_budget():
    import tests.conftest as c
    b = _BASELINES["conftest_snapshot"]
    n_mods = len(c._BASELINE_MODS)
    n_entries = sum(len(v) for v in c._BASELINE_MODS.values())
    assert n_mods <= b["budget_modules"], (
        f"conftest snapshot grew to {n_mods} nx modules "
        f"(budget {b['budget_modules']}, measured {b['baseline_modules']} in "
        f"{b['measured_in']}) — _restore() cost scales with this; trim module "
        f"state or raise the recorded budget with a justification")
    assert n_entries <= b["budget_entries"], (
        f"conftest snapshot grew to {n_entries} module entries "
        f"(budget {b['budget_entries']}, measured {b['baseline_entries']} in "
        f"{b['measured_in']}) — same story: trim or re-record with a reason")


def _literal_first_re_calls(tree, filename):
    """(line, repr) for re.<fn>(<string literal>, ...) nested inside any
    function body. Module-level literal calls are fine (compiled once at
    import)."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "re"
                    and n.func.attr in ("match", "search", "sub", "findall",
                                        "fullmatch", "split", "finditer",
                                        "template")
                    and n.args and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)):
                hits.append((f"{filename}:{n.lineno}",
                             f"re.{n.func.attr}({n.args[0].value!r})"))
    return hits


def test_t15_no_literal_first_re_calls_in_functions():
    offenders = []
    for root, _dirs, files in os.walk(os.path.join(REPO_ROOT, "nx")):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            tree = ast.parse(open(path, encoding="utf-8").read())
            for loc, reprn in _literal_first_re_calls(tree, os.path.relpath(path, REPO_ROOT)):
                if loc not in _REGEX_ALLOWLIST:
                    offenders.append(f"{loc}: {reprn}")
    assert not offenders, (
        "literal-first re.<fn>() inside a function body recompiles the pattern "
        "every call (T-15); hoist it to a module-level re.compile, or allowlist "
        f"the exact location in {__file__} with a reason:\n" + "\n".join(offenders))


def test_baselines_ledger_entries_are_current():
    """The ledger's 'measured' columns must not drift from reality without a
    deliberate re-record — keep the forcing function honest."""
    import tests.conftest as c
    b = _BASELINES["conftest_snapshot"]
    assert abs(len(c._BASELINE_MODS) - b["baseline_modules"]) <= 5, (
        "measured baseline_modules drifted >5 from the ledger — re-record "
        "tests/perf/baselines.json (update measured_in too)")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
