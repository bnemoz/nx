#!/usr/bin/env python3
"""Smoke tests for the GBNF grammar builder (nx/_grammar.py). No live server
or llama.cpp binary needed — checks structural properties of the generated
grammar text and the open_stream() opt-in wiring."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NX_ENV_FILE"] = "/dev/null"

from nx._grammar import build_tool_call_grammar
from nx._text_fallback import parse_text_calls_with_errors

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "multi_edit",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "edits": {"type": "array", "items": {"type": "object",
                "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                "required": ["old", "new"]}}},
            "required": ["path", "edits"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "status": {"type": "string", "enum": ["answered", "blocked"]}},
            "required": ["answer"]}}},
]

# --- grammar builds without error and references every rule it defines ---
grammar, trigger = build_tool_call_grammar(TOOLS)
assert trigger == "<tool_call>"
assert grammar is not None

rule_names = set()
for line in grammar.splitlines():
    name = line.split("::=", 1)[0].strip()
    if name:
        rule_names.add(name)
assert "root" in rule_names
for expected in ("args-read_file", "args-multi_edit", "args-multi_edit-edits-item", "args-final_answer"):
    assert expected in rule_names, expected

# every rule reference used on the right-hand side of some rule must be defined
import re
refs = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", grammar))
GBNF_KEYWORDS_OR_LITERALS = {"ws", "true", "false", "null"}
for r in rule_names:
    assert r in refs  # sanity: every defined rule is referenced somewhere (root or nested)

# --- ONE DIALECT (P3-4): the body the grammar forces is the body SYSTEM
# teaches and the parser reads. A grammar that constrains a shape the prompt
# never asked for is worse than no grammar: every constrained call comes back
# as "missing <name>". Asserted as literal tag text, since that is the contract.
assert '"<name>"' in grammar and '"</name>"' in grammar
assert '"<arguments>"' in grammar and '"</arguments>"' in grammar
# the tool name is constrained INSIDE <name>, not as the element name
assert '"read_file"' in grammar and '"<read_file>"' not in grammar

# --- required args are mandatory tags; optional ones a droppable suffix ---
assert '"<path>"' in grammar
assert '("<offset>"' in grammar
# enums render as RAW literals inside a tag body, not quoted JSON forms
assert '"answered" ws' in grammar and '"\\"answered\\""' not in grammar

# --- the whole point: a grammar-legal call parses with the real parser ---
for body in (
    "<tool_call><name>read_file</name><arguments><path>foo.py</path></arguments></tool_call>",
    '<tool_call><name>multi_edit</name><arguments><path>a.py</path>'
    '<edits>[{"old": "x", "new": "y"}]</edits></arguments></tool_call>',
    # optional property dropped entirely (allowed: only a trailing run is optional)
    "<tool_call><name>final_answer</name><arguments><answer>done</answer></arguments></tool_call>",
):
    calls, errs = parse_text_calls_with_errors(body)
    assert not errs, (body, errs)
    assert len(calls) == 1, (body, calls)
assert parse_text_calls_with_errors(
    "<tool_call><name>final_answer</name><arguments><answer>ok</answer>"
    "<status>blocked</status></arguments></tool_call>")[0][0][1]["status"] == "blocked"

# --- the shape the PREVIOUS grammar forced must NOT parse (regression pin) ---
assert parse_text_calls_with_errors(
    "<tool_call><read_file>x</read_file><arguments><path>foo.py</path>"
    "</arguments></tool_call>")[1], "tool-name-as-element body must not parse"

# --- a value needing a literal '<' round-trips as a JSON-quoted escape ---
assert parse_text_calls_with_errors(
    '<tool_call><name>edit_file</name><arguments><path>a</path>'
    '<new>"\\u003cdiv\\u003e"</new></arguments></tool_call>'
)[0][0][1]["new"] == "<div>"

# --- empty tool list degrades to "no grammar" rather than raising ---
empty_grammar, _ = build_tool_call_grammar([])
assert empty_grammar is None

# --- required-before-optional is policed IN THE BUILDER now, not only by a
# test over m.TOOLS: a violation silently yields a grammar that refuses legal
# calls, so every schema source (probe, eval, future) must trip it. ---
_bad = {"type": "function", "function": {
    "name": "bad", "parameters": {"type": "object",
        "properties": {"opt": {"type": "string"}, "req": {"type": "string"}},
        "required": ["req"]}}}
try:
    build_tool_call_grammar([_bad])
except ValueError as e:
    assert "required property 'req'" in str(e), e
else:
    raise AssertionError("required-after-optional schema built without raising")

# ...and the same check recurses into array-item object schemas
_bad_item = {"type": "function", "function": {
    "name": "bad2", "parameters": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object",
            "properties": {"opt": {"type": "string"}, "req": {"type": "string"}},
            "required": ["req"]}}}}}}
try:
    build_tool_call_grammar([_bad_item])
except ValueError as e:
    assert "bad2.items" in str(e), e
else:
    raise AssertionError("required-after-optional item schema built without raising")

# --- open_stream() wiring: opt-in via NX_GRAMMAR, off by default, off for vLLM ---
os.environ.pop("NX_GRAMMAR", None)
os.environ.pop("NX_BACKEND", None)
import importlib
import nx._compress as _compress
importlib.reload(_compress)
assert _compress.NX_GRAMMAR is False
assert _compress._grammar_extra_body(TOOLS) is None

os.environ["NX_GRAMMAR"] = "1"
importlib.reload(_compress)
assert _compress.NX_GRAMMAR is True
extra = _compress._grammar_extra_body(TOOLS)
assert set(extra.keys()) == {"grammar", "grammar_triggers"}
assert extra["grammar_triggers"] == [{"type": "word", "value": "<tool_call>"}]
# cached by tool-name tuple
assert _compress._grammar_extra_body(TOOLS) is extra

os.environ["NX_BACKEND"] = "vllm"
importlib.reload(_compress)
assert _compress.NX_GRAMMAR is False
assert _compress._grammar_extra_body(TOOLS) is None

os.environ.pop("NX_GRAMMAR", None)
os.environ.pop("NX_BACKEND", None)
importlib.reload(_compress)

# --- once native tool-calling is confirmed, stop attaching our own grammar ---
# (llama.cpp already derives its own argument-constraining grammar from
# `tools=` for native-capable templates; ours would only duplicate/fight it —
# see the docstrings in nx/_grammar.py and nx/_compress.py.)
from nx import _state as st

os.environ["NX_GRAMMAR"] = "1"
importlib.reload(_compress)
assert _compress.NX_GRAMMAR is True
st._NATIVE_TOOL_CALLS_CONFIRMED = False
assert _compress._grammar_extra_body(TOOLS) is not None
st._NATIVE_TOOL_CALLS_CONFIRMED = True
assert _compress._grammar_extra_body(TOOLS) is None
st._NATIVE_TOOL_CALLS_CONFIRMED = False  # restore for any later test in this process

# --- _handle_native_tool_calls() sets the confirmation flag ---
import nx._turn as _turn
st._NATIVE_TOOL_CALLS_CONFIRMED = False
tcs = {0: {"id": "call_1", "name": "read_file", "args": '{"path": "x"}'}}
_orig_run_tool = _turn.run_tool
_turn.run_tool = lambda name, args: "ok"
try:
    messages = []
    _turn._handle_native_tool_calls(messages, tcs, None, [], 0)
finally:
    _turn.run_tool = _orig_run_tool
assert st._NATIVE_TOOL_CALLS_CONFIRMED is True
st._NATIVE_TOOL_CALLS_CONFIRMED = False  # reset so this test doesn't leak state

os.environ.pop("NX_GRAMMAR", None)
os.environ.pop("NX_BACKEND", None)
importlib.reload(_compress)

print("test_grammar.py: all assertions passed")
