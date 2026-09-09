#!/usr/bin/env python3
"""Invariants the model-facing tool schemas must keep.

These descriptions are the model's only manual: it picks tools and fills
arguments from these strings alone. Beyond taste, a few properties are
mechanically checkable and are load-bearing:

  1. REQUIRED-BEFORE-OPTIONAL. nx/_grammar.py builds a GBNF grammar in which
     only a *trailing* run of optional properties can be dropped. A required
     property declared after an optional one makes a legal call
     unrepresentable -- the grammar would refuse what the body accepts.
  2. FLAT PROPERTIES. The grammar understands string/integer/boolean/enum
     values plus an array-of-objects; anything else silently degrades to the
     permissive `value` rule (no constraint). Legitimate use, but it must be a
     deliberate list, so this test asserts the exception set.
  3. EVERY PROPERTY DOCUMENTED. A model filling in one argument reads that
     property's description and nothing else; an undocumented property gets
     guessed at. (Properties whose name is already the whole contract, like a
     bare `path`, may skip it -- hence the explicit exemption list.)
  4. DESCRIPTIONS MATCH THE BODIES. Argument names in the schema must be
     parameters the body actually accepts, and every `required` name must
     exist in `properties` (a typo there is invisible until a call fails).
  5. THE STEERING STRINGS some tests/README copy depend on still exist.

No model or network round-trip.
"""
import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
from nx._grammar import build_tool_call_grammar  # noqa: E402

# Properties deliberately left undescribed: the name is the contract, and a
# description would restate it ("path" -> "the path"). Adding a property here
# is a statement that it needs no prose -- think before adding.
UNDOCUMENTED_OK = {
    ("write_file", "path"),
    ("edit_file", "path"),
    ("multi_edit", "path"),
    ("sed_edit", "pattern"),
    ("git_status", "path"),
    ("git_diff", "path"),
    ("git_log", "path"),
    ("run_bash", "command"),
    ("multi_edit-item", "old"),
    ("multi_edit-item", "new"),
}

# Tools whose argument names differ from the python body's parameter names on
# purpose (`async` is a keyword, so the body takes it via **kwargs).
ARG_NAME_EXCEPTIONS = {"spawn_agent": {"async"}}


def _reads_as_prose(d):
    """True if a description reads as guidance rather than an abbreviation
    trail: it opens with a capital / backtick / quoted literal, or it is a
    `value = meaning` definition (how enum members and booleans are best
    documented -- against the literal the model is picking)."""
    return bool(re.match(r"[A-Z`'\"$\[]", d)) or "= " in d[:24]


def _all_tools():
    return m.TOOLS


def _fn(tool):
    return tool["function"]


def test_every_tool_appears_once():
    names = [_fn(t)["name"] for t in _all_tools()]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"
    # test_background_gating pins the exact count when a tool is added/removed.
    assert len(names) == 23


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: _fn(t)["name"])
def test_required_properties_are_declared_before_optional(tool):
    params = _fn(tool)["parameters"]
    required = set(params.get("required") or [])
    props = list(params.get("properties") or {})
    seen_optional = None
    for name in props:
        if name not in required:
            seen_optional = name
        elif seen_optional is not None:
            pytest.fail(
                f"{_fn(tool)['name']}: required property {name!r} is declared after "
                f"optional {seen_optional!r}. nx/_grammar.py can only drop a trailing "
                "optional run, so reorder the properties (or make it optional)."
            )


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: _fn(t)["name"])
def test_grammar_still_builds_for_every_tool(tool):
    """A schema shape the grammar can't express would degrade silently; the
    builder must still produce a rule naming this tool."""
    grammar, _ = build_tool_call_grammar([tool])
    assert grammar is not None
    assert f"args-{_fn(tool)['name']}" in grammar


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: _fn(t)["name"])
def test_property_shapes_are_the_flat_subset(tool):
    """Values are scalars/enums/str-lists, or an array-of-objects with a
    documented item schema. `oneOf` (str | [str]) is the other accepted
    escape hatch, and it is limited to the two tools that read it."""
    name = _fn(tool)["name"]
    for prop, schema in (_fn(tool)["parameters"].get("properties") or {}).items():
        if "oneOf" in schema:
            assert name in ("read_file", "search_files"), (
                f"{name}.{prop} uses oneOf, which falls through to the grammar's "
                "permissive rule; only the two list-or-string args opt into that")
            continue
        if schema.get("type") == "array":
            items = schema["items"]
            assert items["type"] == "object", f"{name}.{prop}: unexpected array item"
            assert items.get("required"), f"{name}.{prop}: item schema should require its keys"
            continue
        assert set(schema) <= {"type", "enum", "description"}, \
            f"{name}.{prop}: unexpected schema keys {set(schema)}"
        assert schema.get("type") in ("string", "integer", "boolean") or "enum" in schema, \
            f"{name}.{prop}: {schema} is not a scalar/enum -- the grammar won't constrain it"


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: _fn(t)["name"])
def test_properties_are_documented(tool):
    name = _fn(tool)["name"]
    for prop, schema in (_fn(tool)["parameters"].get("properties") or {}).items():
        if not schema.get("description"):
            assert (name, prop) in UNDOCUMENTED_OK, (
                f"{name}.{prop} has no description; either document it or list it in "
                "UNDOCUMENTED_OK with a reason")
        else:
            d = schema["description"]
            assert d.strip() == d and len(d) > 8, f"{name}.{prop}: weak description {d!r}"
            assert _reads_as_prose(d), f"{name}.{prop}: {d!r}"


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: _fn(t)["name"])
def test_required_names_exist_and_tool_description_is_self_sufficient(tool):
    fn = _fn(tool)
    params = fn["parameters"]
    props = set(params.get("properties") or {})
    for req in params.get("required") or []:
        assert req in props, f"{fn['name']}: required {req!r} is not a property"
    d = fn["description"]
    assert d.strip() and d.endswith((".", "!", "?")), \
        f"{fn['name']}: description should be complete prose, not an abbreviation trail"
    assert "\n" not in d, f"{fn['name']}: keep the tool description single-line"


@pytest.mark.parametrize("tool", _all_tools(), ids=lambda t: _fn(t)["name"])
def test_schema_arguments_exist_on_the_body(tool):
    """A schema argument the body ignores is a silent no-op -- the model
    believes it set something. Bodies take **kwargs, so introspection is the
    only check that catches the typo."""
    name = _fn(tool)["name"]
    body = m.DISPATCH.get(name)
    if body is None:  # final_answer is terminal-dispatched, not in DISPATCH
        pytest.skip(f"{name} has no dispatch body")
    sig = inspect.signature(body)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        accepted = {p.name for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        # **kwargs bodies still name their real args; only the exception set
        # (python keywords) may be missing.
        missing = (set(_fn(tool)["parameters"].get("properties") or {})
                   - accepted - ARG_NAME_EXCEPTIONS.get(name, set()))
        assert not missing, f"{name}: schema args the body never reads: {sorted(missing)}"


def test_mutating_and_read_only_sets_stay_in_agreement_with_plan_gating():
    """The approval story is stated once in SYSTEM; plan mode gates the same
    mutating set. If a new mutating tool joins one list and not the other,
    the model gets inconsistent affordances -- so cross-check against
    PLAN_GATE_TOOLS, which is the explicit mutating list."""
    from nx._plan import PLAN_GATE_TOOLS, PLAN_READONLY_TOOLS
    names = {_fn(t)["name"] for t in _all_tools()} | {"final_answer"}
    assert PLAN_GATE_TOOLS <= names, PLAN_GATE_TOOLS - names
    assert PLAN_READONLY_TOOLS <= names, PLAN_READONLY_TOOLS - names
    assert not (PLAN_GATE_TOOLS & PLAN_READONLY_TOOLS)


def test_steering_phrases_tests_and_readme_copy_depend_on():
    """Small/quantized models left spawn_agent unused and set max_turns to the
    old 12-token-scale default; both were fixed with specific wording that
    test_delegation_steering / test_subagent_budget assert. Re-asserted here so
    a schema-wide rewrite can't quietly drop them."""
    by_name = {_fn(t)["name"]: _fn(t) for t in _all_tools()}
    desc = by_name["spawn_agent"]["description"]
    assert "PROACTIVELY" in desc
    assert "ONE CALL PER QUESTION" in desc
    assert "callers" in desc
    assert "default 100" in \
        by_name["spawn_agent"]["parameters"]["properties"]["max_turns"]["description"]
    # edit_file must stay requirable without `old` (old_lines is an alternative).
    assert by_name["edit_file"]["parameters"]["required"] == ["path", "new"]


def test_active_tool_schemas_are_json_clean_and_within_budget():
    """The floor claim is measured in test_initial_context; this one catches
    the cheaper failure: a schema that doesn't serialize, or a per-tool
    description so long it alone would pay for a whole new tool."""
    blob = json.dumps(m._active_tools())
    assert blob
    for tool in _all_tools():
        fn = _fn(tool)
        assert len(fn["description"]) < 700, \
            f"{fn['name']}: {len(fn['description'])} chars of description -- split the " \
            "guidance onto properties or into SYSTEM instead"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
