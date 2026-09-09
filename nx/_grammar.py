"""Optional GBNF grammar-constrained decoding for text-protocol tool calls.

_text_fallback.py exists because free-form generation of the tool-call tag can
come out truncated or missing pieces — parse_text_calls_with_errors() copes
with that after the fact. llama.cpp's server can instead take a GBNF grammar
and a lazy trigger string so that once the model emits the trigger token
sequence, subsequent tokens are sampled only from strings the grammar accepts
— a malformed call becomes structurally unsamplable rather than something to
parse around.

One dialect (0.50.0, ROADMAP P3-4): the grammar constrains the SAME shape
SYSTEM teaches and the parser prefers —
`<tool_call><name>read_file</name><arguments><path>foo.py</path></arguments></tool_call>` —
not the legacy JSON-in-tag body this grammar used to constrain. A model was
previously force-constrained into a shape its prompt never asked for, which is
worse than no grammar at all: the parser finds the tool by reading `<name>`, so
a grammar-constrained call could never parse and every turn came back as
"missing <name>". The parser still accepts the legacy JSON body for old
sessions; the grammar only steers generation to the taught shape.

Value encoding inside `<k>...</k>` bodies: string-typed properties are raw text
(`xraw` — anything up to the next `<`), so a path is samplable exactly as
SYSTEM's example shows; integer/boolean/enum properties constrain their raw
literals; array/object-typed properties (the only ones whose value cannot be
flat text) stay strict JSON, parsed back by json.loads on the parser side. A
raw body cannot contain a literal `<` (that is the region terminator, and GBNF
has no lookahead to exclude only the closing tag) — a value needing one is
written as a JSON-quoted string with the `<` escaped as `<`, which xraw
permits (no `<` ever appears in the emitted bytes) and which the parser's
json.loads decodes back to `<`. A raw unquoted `<` is unsamplable instead, which
is the point: the alternative was shipping a silently corrupted value.

This only covers the JSON-schema subset TOOLS actually uses (flat objects of
string/integer/boolean/enum properties, plus one array-of-objects case) —
see build_tool_call_grammar()'s docstring. Any schema shape outside that
subset falls back to the permissive raw-text rule rather than raising, per
the repo's "degrade, don't demand" convention.

This grammar is only wired up (see nx/_compress.py's _grammar_extra_body())
while a session is in the text-protocol fallback, or hasn't yet been
confirmed to be using native tool-calling. Once a turn actually comes back
with native `tool_calls` deltas, llama.cpp is already constraining argument
generation with its own internally-derived grammar for that request (it
builds one automatically from `tools=` for chat templates it recognizes as
tool-call-capable) — there's no separate, client-observable trigger string
to hook a second grammar off of in that mode, and llama.cpp has no notion of
an argument-only (as opposed to whole-response) custom grammar for native
tool calls. So _grammar_extra_body() stops attaching this grammar once
native mode is confirmed, rather than fighting or duplicating llama.cpp's
own constraint.
"""
from __future__ import annotations

import json
from typing import Any

WS_RULE = r'ws ::= [ \t\n]*'
# JSON-quoted string, used ONLY inside a JSON object/array body (an array- or
# object-typed property). A literal `<` is allowed there: that region ends at
# the property's own closing tag, which the parser finds non-greedily, so a
# `<div>` inside a quoted edit anchor still parses back intact.
STRING_RULE = (
    r'string ::= "\"" ( [^"\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws'
)
# Raw tag-body text: everything up to (not including) the next `<`. This is how
# SYSTEM writes a value; a value needing a literal `<` is written JSON-quoted
# with a `<` escape (a quote is not a `<`, so xraw covers it) and
# json.loads decodes it.
XRAW_RULE = r'xraw ::= [^<]* ws'
INTEGER_RULE = r'integer ::= "-"? ("0" | [1-9] [0-9]*) ws'
BOOLEAN_RULE = r'boolean ::= ("true" | "false") ws'
# Fallback for property schemas outside the supported subset — permissive
# rather than raising, so an unusual tool schema degrades to unconstrained
# JSON instead of breaking grammar construction.
VALUE_RULE = (
    r'value ::= string | integer | boolean | "null" ws | object | array'
)
OBJECT_RULE = r'object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws'
ARRAY_RULE = r'array ::= "[" ws (value ("," ws value)*)? "]" ws'
PRIMITIVES = [WS_RULE, STRING_RULE, XRAW_RULE, INTEGER_RULE, BOOLEAN_RULE,
              VALUE_RULE, OBJECT_RULE, ARRAY_RULE]

# Rule name for the shared `<name>` region. Deliberately free of `-`, and a
# tool name is a `-`-free Python identifier, so this can never collide with the
# `args-<tool>` / `<args>-<property>` helper rules derived from schema names.
NAME_REGION_RULE = "name"


def _gbnf_literal(s: str) -> str:
    """A GBNF quoted-literal matching the exact JSON encoding of `s`."""
    return json.dumps(json.dumps(s))


def _enum_rule(values: list[str]) -> str:
    return " | ".join(f'{_gbnf_literal(v)} ws' for v in values)


def _assert_required_before_optional(schema: dict[str, Any], owner: str) -> None:
    """Raise on a required property declared after an optional one.

    P3-4 moves this from a test over m.TOOLS into the builder itself — cheap
    and total (every tool schema, probe tools, eval tools, future tool
    sources) — because a violation doesn't error, it silently produces a
    grammar that refuses legal calls. Recurses into array-item object schemas
    (multi_edit's edits, todo_write's todos), which get their own rules.
    """
    required = set(schema.get("required") or [])
    seen_optional = False
    for pname, pschema in (schema.get("properties") or {}).items():
        if pname not in required:
            seen_optional = True
        elif seen_optional:
            raise ValueError(
                f"{owner}: required property {pname!r} is declared after an optional "
                "one. The GBNF builder can only drop a trailing optional run, so "
                "reorder the properties (or make it optional).")
        if isinstance(pschema, dict) and pschema.get("type") == "array":
            item = pschema.get("items")
            if isinstance(item, dict) and item.get("type") == "object":
                _assert_required_before_optional(item, f"{owner}.{pname}")


def _json_value_ref(schema: Any, rule_name: str, rules: dict[str, str]) -> str:
    """A GBNF expression for `schema` INSIDE a JSON object/array rule (strict
    shapes; a literal `<` is allowed inside quoted strings there)."""
    if not isinstance(schema, dict):
        return "value"
    if "oneOf" in schema:
        return "value"
    if "enum" in schema:
        return f"({_enum_rule(schema['enum'])})"
    t = schema.get("type")
    if t == "string":
        return "string"
    if t == "integer":
        return "integer"
    if t == "boolean":
        return "boolean"
    if t == "object":
        rules[rule_name] = _object_rule_body(schema, rule_name, rules)
        return rule_name
    if t == "array":
        item_rule_name = f"{rule_name}-item"
        item_ref = _json_value_ref(schema.get("items", {}), item_rule_name, rules)
        return f'"[" ws ({item_ref} ("," ws {item_ref})*)? "]" ws'
    return "value"


def _xml_enum_rule(values: list[str]) -> str:
    """Raw-literal alternation for an enum inside an XML tag body (the JSON
    context's _enum_rule matches the QUOTED forms, which would teach the
    grammar `<status>"answered"</status>`)."""
    return " | ".join(f'{json.dumps(v)} ws' for v in values)


def _xml_value_ref(schema: Any, rule_name: str, rules: dict[str, str]) -> str:
    """A GBNF expression for `schema` as an XML tag BODY (`<k>VALUE</k>`).

    Scalars constrain their raw literals; string-typed properties are raw
    text (raw or JSON-quoted both parse back); array/object-typed properties
    — the only values that cannot be flat text — stay strict JSON; unknown
    shapes (oneOf, no type) degrade to the permissive raw-text rule, which
    the parser JSON-decodes when it can."""
    if not isinstance(schema, dict):
        return "xraw"
    if "oneOf" in schema:
        return "xraw"
    if "enum" in schema:
        return f"({_xml_enum_rule(schema['enum'])})"
    t = schema.get("type")
    if t == "string":
        return "xraw"
    if t == "integer":
        return "integer"
    if t == "boolean":
        return "boolean"
    if t == "object":
        rules[rule_name] = _object_rule_body(schema, rule_name, rules)
        return rule_name
    if t == "array":
        item_rule_name = f"{rule_name}-item"
        item_ref = _json_value_ref(schema.get("items", {}), item_rule_name, rules)
        return f'"[" ws ({item_ref} ("," ws {item_ref})*)? "]" ws'
    return "xraw"


def _object_rule_body(schema: dict[str, Any], rule_name: str, rules: dict[str, str]) -> str:
    """Build the GBNF body for a flat-object schema (JSON context).

    Requires (asserted by _assert_required_before_optional, for every caller
    schema) that required properties all precede optional ones in declaration
    order — each optional property and everything after it is wrapped as a
    droppable, comma-prefixed suffix.
    """
    props = list((schema.get("properties") or {}).items())
    required = set(schema.get("required") or [])
    if not props:
        return '"{" ws "}" ws'

    def cont(i: int) -> str:
        if i == len(props):
            return ""
        name, pschema = props[i]
        value_ref = _json_value_ref(pschema, f"{rule_name}-{name}", rules)
        kv = f'{_gbnf_literal(name)} ws ":" ws {value_ref}'
        tail = cont(i + 1)
        piece = f'"," ws {kv}' + (f" {tail}" if tail else "")
        if name in required:
            return piece
        return f"({piece})?"

    first_name, first_schema = props[0]
    first_ref = _json_value_ref(first_schema, f"{rule_name}-{first_name}", rules)
    first_kv = f'{_gbnf_literal(first_name)} ws ":" ws {first_ref}'
    rest = cont(1)
    body = first_kv + (f" {rest}" if rest else "")
    if first_name not in required:
        body = f"({body})?"
    return f'"{{" ws {body} "}}" ws'


def _name_region_rule(names: list[str], rules: dict[str, str]) -> str:
    """The `<name>` region shared by every tool: the literal open tag, one of
    the known tool names, the literal close tag. Hoisted out of the per-tool
    alternation so the grammar text carries one `<name>`/`</name>` pair instead
    of one per tool, and so constraining the name to `tools` (which kills
    unknown-tool calls) is one rule rather than a repetition."""
    alt = " | ".join(json.dumps(n) for n in names)
    rules[NAME_REGION_RULE] = (
        f'{json.dumps("<name>")} ({alt}) {json.dumps("</name>")} ws')
    return NAME_REGION_RULE


def _args_region_rule(params: dict[str, Any], rule_name: str,
                      rules: dict[str, str]) -> str:
    """The `arguments` region: one `<key>value</key>` tag per schema property,
    each optional (any subset, in schema order — the parser accepts any order;
    the grammar steers to schema order). Returns the rule NAME, which is always
    defined (empty schemas get an empty region) so the per-tool args-<name>
    rule exists for every tool."""
    parts: list[str] = []
    required = set(params.get("required") or [])
    for pname, pschema in (params.get("properties") or {}).items():
        ref = _xml_value_ref(pschema, f"{rule_name}-{pname}", rules)
        tag = (f'{json.dumps("<" + pname + ">")} {ref} '
               f'{json.dumps("</" + pname + ">")} ws')
        # Required properties are MANDATORY tags (required-before-optional is
        # asserted, so they all precede optionals): the grammar then steers
        # the model to include every required argument, instead of sampling a
        # call that dispatch would reject for a missing one.
        parts.append(tag if pname in required else f"({tag})?")
    rules[rule_name] = " ".join(parts) if parts else '"" ws'
    return rule_name


def build_tool_call_grammar(
    tools: list[dict[str, Any]],
    tag_open: str = "<tool_call>",
    tag_close: str = "</tool_call>",
) -> tuple[str | None, str]:
    """Build a GBNF grammar accepting exactly one text-protocol tool-call body
    in the dialect SYSTEM teaches —
    `<name>TOOL</name><arguments><k>v</k>...</arguments></tool_call>` — matching
    one of `tools`' schemas (the trigger tag itself is consumed by the
    lazy-grammar trigger, not the grammar).

    Returns (grammar_text, tag_open) — tag_open is the literal string a
    llama.cpp lazy-grammar trigger should fire on.
    """
    rules: dict[str, str] = {}
    named: list[tuple[str, str]] = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        _assert_required_before_optional(params, name)
        args_rule = _args_region_rule(params, f"args-{name}", rules)
        named.append((name, args_rule))
    if not named:
        return None, tag_open
    name_rule = _name_region_rule([n for n, _ in named], rules)
    alts = [
        f'{name_rule} ws '
        f'{json.dumps("<arguments>")} ws {args_rule} '
        f'{json.dumps("</arguments>")} ws '
        f'{json.dumps(tag_close)} ws'
        for _, args_rule in named
    ]
    root = " | ".join(f"({a})" for a in alts)
    lines = [f"root ::= {root}"] + PRIMITIVES
    for rname, rbody in rules.items():
        lines.append(f"{rname} ::= {rbody}")
    return "\n".join(lines), tag_open
