#!/usr/bin/env python3
"""Smoke tests for the GBNF grammar builder (nx/_grammar.py). No live server
or llama.cpp binary needed — checks structural properties of the generated
grammar text and the open_stream() opt-in wiring."""
import json
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NX_ENV_FILE"] = "/dev/null"

from nx._grammar import build_tool_call_grammar

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

# --- required-before-optional (read_file: path required, offset optional) ---
assert '"\\"path\\""' in grammar
assert "(\",\" ws \"\\\"offset\\\"\"" in grammar or '"," ws "\\"offset\\""' in grammar

# --- enum renders as a literal alternation, not a bare string rule ---
assert '"\\"answered\\""' in grammar and '"\\"blocked\\""' in grammar

# --- empty tool list degrades to "no grammar" rather than raising ---
empty_grammar, _ = build_tool_call_grammar([])
assert empty_grammar is None

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

print("test_grammar.py: all assertions passed")
