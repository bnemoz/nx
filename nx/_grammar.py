"""Optional GBNF grammar-constrained decoding for text-protocol tool calls.

_text_fallback.py exists because free-form generation of `<tool_call>{...}
</tool_call>` tags can come out with truncated/invalid JSON or a missing
"name" field — parse_text_calls_with_errors() copes with that after the fact.
llama.cpp's server can instead take a GBNF grammar and a lazy trigger string
so that once the model emits the trigger token sequence, subsequent tokens
are sampled only from strings the grammar accepts — malformed JSON becomes
structurally unsamplable rather than something to parse around.

This only covers the JSON-schema subset TOOLS actually uses (flat objects of
string/integer/boolean/enum properties, plus one array-of-objects case) —
see build_tool_call_grammar()'s docstring. Any schema shape outside that
subset falls back to a permissive `value` rule (any JSON scalar/object/array)
rather than raising, per the repo's "degrade, don't demand" convention.

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
import json

WS_RULE = r'ws ::= [ \t\n]*'
STRING_RULE = (
    r'string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws'
)
INTEGER_RULE = r'integer ::= "-"? ("0" | [1-9] [0-9]*) ws'
BOOLEAN_RULE = r'boolean ::= ("true" | "false") ws'
# Fallback for property schemas outside the supported subset (e.g. an
# unanticipated "number"/"null"/nested-object-as-property) — permissive
# rather than raising, so an unusual tool schema degrades to unconstrained
# JSON instead of breaking grammar construction.
VALUE_RULE = (
    r'value ::= string | integer | boolean | "null" ws | object | array'
)
OBJECT_RULE = r'object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws'
ARRAY_RULE = r'array ::= "[" ws (value ("," ws value)*)? "]" ws'
PRIMITIVES = [WS_RULE, STRING_RULE, INTEGER_RULE, BOOLEAN_RULE, VALUE_RULE, OBJECT_RULE, ARRAY_RULE]


def _gbnf_literal(s):
    """A GBNF quoted-literal matching the exact JSON encoding of `s`."""
    return json.dumps(json.dumps(s))


def _enum_rule(values):
    return " | ".join(f'{_gbnf_literal(v)} ws' for v in values)


def _kv_rule(name, value_ref):
    return f'{_gbnf_literal(name)} ws ":" ws {value_ref}'


def _value_ref(schema, rule_name, rules):
    """Return a GBNF symbol/expression for `schema`, adding any helper rules
    it needs (e.g. a nested object-array item rule) to `rules`."""
    if not isinstance(schema, dict):
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
        item_ref = _value_ref(schema.get("items", {}), item_rule_name, rules)
        return f'"[" ws ({item_ref} ("," ws {item_ref})*)? "]" ws'
    return "value"


def _object_rule_body(schema, rule_name, rules):
    """Build the GBNF body for a flat-object schema.

    Assumes (true of every schema in TOOLS today) that required properties
    all precede optional ones in declaration order — each optional property
    and everything after it is wrapped as a droppable, comma-prefixed
    suffix. A required property after an optional one would break this and
    isn't supported; TOOLS has none.
    """
    props = list((schema.get("properties") or {}).items())
    required = set(schema.get("required") or [])
    if not props:
        return '"{" ws "}" ws'

    def cont(i):
        if i == len(props):
            return ""
        name, pschema = props[i]
        value_ref = _value_ref(pschema, f"{rule_name}-{name}", rules)
        kv = _kv_rule(name, value_ref)
        tail = cont(i + 1)
        piece = f'"," ws {kv}' + (f" {tail}" if tail else "")
        if name in required:
            return piece
        return f"({piece})?"

    first_name, first_schema = props[0]
    first_ref = _value_ref(first_schema, f"{rule_name}-{first_name}", rules)
    first_kv = _kv_rule(first_name, first_ref)
    rest = cont(1)
    body = first_kv + (f" {rest}" if rest else "")
    if first_name not in required:
        body = f"({body})?"
    return f'"{{" ws {body} "}}" ws'


def build_tool_call_grammar(tools, tag_open="<tool_call>", tag_close="</tool_call>"):
    """Build a GBNF grammar accepting exactly one `{"name": ..., "arguments":
    {...}}</tool_call>` body (the trigger tag itself is consumed by the
    lazy-grammar trigger, not the grammar) matching one of `tools`' schemas.

    Returns (grammar_text, tag_open) — tag_open is the literal string a
    llama.cpp lazy-grammar trigger should fire on.
    """
    rules = {}
    alts = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        args_rule = f"args-{name}"
        rules[args_rule] = _object_rule_body(params, args_rule, rules)
        name_kv = _kv_rule("name", _gbnf_literal(name))
        args_kv = _kv_rule("arguments", args_rule)
        alts.append(f'"{{" ws {name_kv} "," ws {args_kv} ws "}}" ws {json.dumps(tag_close)} ws')
    if not alts:
        return None, tag_open
    root = " | ".join(f"({a})" for a in alts)
    lines = [f"root ::= {root}"] + PRIMITIVES
    for rname, body in rules.items():
        lines.append(f"{rname} ::= {body}")
    return "\n".join(lines), tag_open
