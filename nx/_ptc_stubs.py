"""PTC Phase 1 (docs/ptc-migration-spec.md): typed Python stubs generated from
the OpenAI tool envelopes — one generator, one source of truth, zero new schemas.

`render_stub_module(tools)` walks the `TOOLS` envelopes (`nx/__init__.py`) and
emits a Python module of stub functions whose signatures mirror the schemas
(string→str, integer→int, boolean→bool, number→float, `oneOf`/`anyOf`→`|`
union, array-of-object→list[dict], schema `default` preserved, otherwise
optional params default to `None`), with docstrings built from the schema's
function + property descriptions (the invariants in
tests/test_tool_schema_invariants.py guarantee those exist). Each stub body
routes through `_capture(name, args)`, which Phase 2 (`nx/_ptc_exec.py`) wires
to the harness pipe; in Phase 1 `_capture` raises — the module is generated and
unit-tested, never executed.

Gate awareness: the module must reflect what the model is actually offered this
turn. `render_stub_module_for_turn()` renders from `_active_tools()`
(`nx/_tool_gates.py`), which on sub-agent threads already returns the tiered
`_subagent_tools()` set (`nx/_subagent_tiers.py`). The render memo is keyed on
the same sorted-gate-state tuple as `_gate_vector()` plus the rendered tool
names, and the shared memo is SKIPPED entirely while `_subagent_ctx()` is
active — sub-agent state is thread-local, so serving one agent's memoized
stubs inside another agent's turn is not safe.

Phase 1 exclusions (spec §Phase 1): `spawn_agent`, `critique`, `read_image`
(returns content parts, not str), `call_project_tool` (dict args),
`final_answer` (end-of-turn semantics out of scope) and `run_tools` (must not
recurse into its own stub module).
"""
from __future__ import annotations

import keyword
from typing import Any

# Tools never rendered as stubs (spec §Phase 1 exclusions + the Phase 2
# no-recursion open item, resolved here: run_tools gets no stub).
EXCLUDED_TOOLS = frozenset({
    "spawn_agent", "critique", "read_image", "call_project_tool",
    "final_answer", "run_tools",
})

_JSON_TYPE_TO_PY = {
    "string": "str",
    "integer": "int",
    "boolean": "bool",
    "number": "float",
}

# memo: (_gate_vector(), sorted rendered names) -> rendered source
_STUB_CACHE: dict[tuple, str] = {}


def _py_name(name: str) -> str:
    """Schema property name -> legal Python parameter name (`async` is a
    keyword; names with dots/dashes fall back to underscores)."""
    ident = name.replace("-", "_").replace(".", "_")
    return ident + "_" if keyword.iskeyword(ident) else ident


def _render_type(prop: dict[str, Any]) -> str:
    """A schema property -> its annotation string.

    oneOf/anyOf -> `|` union of the branches (read_file/search_files `path`);
    array of objects -> list[dict] (edit_file/multi_edit `edits`, todo_write
    `todos` — the element schema is deliberately not typed out); array of
    scalars -> list[<py>]; enum (no declared type) -> str; anything else ->
    Any (permissive, matching the grammar's fall-through `value` rule)."""
    for key in ("oneOf", "anyOf"):
        branches = prop.get(key)
        if isinstance(branches, list) and branches:
            return " | ".join(_render_type(b) for b in branches)
        if key in prop:
            return "Any"
    t = prop.get("type")
    if t == "array":
        items = prop.get("items")
        if isinstance(items, dict):
            if items.get("type") == "object":
                return "list[dict]"
            inner = _render_type(items)
            return f"list[{inner}]" if inner != "Any" else "list"
        return "list"
    if isinstance(t, str) and t in _JSON_TYPE_TO_PY:
        return _JSON_TYPE_TO_PY[t]
    if "enum" in prop:
        return "str"
    return "Any"


def _render_default(value: Any) -> str:
    """A schema `default` literal -> Python literal source (JSON subset maps
    1:1; anything unrecognized renders as None rather than corrupting the
    module)."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_render_default(v) for v in value) + "]"
    return "None"


def _esc(text: str) -> str:
    """Escape for embedding inside a triple-quoted docstring."""
    return (str(text).replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", " ").replace("\n", " "))


def _docstring(fn: dict[str, Any], props: dict[str, Any]) -> list[str]:
    lines = [f'    """{_esc(fn.get("description", "")).rstrip()}']
    for pname, prop in props.items():
        desc = prop.get("description") if isinstance(prop, dict) else None
        if desc:
            lines.append(f"    {pname}: {_esc(desc)}")
    lines.append("    Returns tool result string.")
    lines.append('    """')
    return lines


def _render_stub(fn: dict[str, Any]) -> list[str]:
    """One OpenAI envelope -> one def block ("" when the tool is excluded or
    carries no parameters)."""
    name = fn["function"]["name"]
    if name in EXCLUDED_TOOLS:
        return []
    schema = fn["function"].get("parameters") or {}
    props = schema.get("properties") or {}
    required = [r for r in (schema.get("required") or []) if r in props]
    optional = [p for p in props if p not in required]
    args: list[str] = []
    for pname in required + optional:
        prop = props[pname] if isinstance(props[pname], dict) else {}
        anno = _render_type(prop)
        default = None
        if pname in optional:
            default = _render_default(prop["default"]) if "default" in prop else "None"
            if "default" not in prop:
                anno = f"{anno} | None"
        piece = f"{_py_name(pname)}: {anno}"
        if default is not None:
            piece += f" = {default}"
        args.append(piece)
    lines = [f"def {name}({', '.join(args)}) -> str:"]
    lines += _docstring(fn["function"], props)
    pairs = ", ".join(
        f"{pname!r}: {_py_name(pname)}" for pname in required + optional)
    lines.append(f"    return _capture({name!r}, {{{pairs}}})")
    lines.append("")
    lines.append("")
    return lines


def _render(tools: list[dict[str, Any]]) -> str:
    out = [
        '"""Generated PTC stub module (nx/_ptc_stubs.render_stub_module).',
        "",
        "Do not edit: regenerated from the live tool schemas every turn the",
        'gate vector changes."""',
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "",
        "def _capture(name: str, args: dict) -> str:",
        '    """Marshal one stub call back to the harness pipe.',
        "",
        "    Phase 2 (nx/_ptc_exec.py) replaces this with the pipe client that",
        "    routes through run_tool(); Phase 1 generates and unit-tests the",
        "    module only, so executing a stub is a hard error.",
        '    """',
        "    raise RuntimeError(",
        '        f"PTC stub call {name!r} has no execution path in this build")',
        "",
        "",
    ]
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn or "name" not in fn:
            continue
        out += _render_stub({"function": fn})
    while out and out[-1] == "":
        out.pop()
    out.append("")
    return "\n".join(out)


def _gate_vector() -> tuple:
    """Memo key source: `_tool_gates._gate_vector()` — the sorted
    (name, gate-open) tuple plus the breaker's broken set. Lazy import keeps
    this module cycle-free (and patchable in tests)."""
    from ._tool_gates import _gate_vector as _real_gate_vector
    return _real_gate_vector()


def _active_tools_for_turn() -> list[dict[str, Any]]:
    """This turn's offered schemas: `_tool_gates._active_tools()`, which
    already returns the tiered `_subagent_tools()` set on sub-agent threads.
    Wrapped so the seam is visible (and patchable) on this module."""
    from ._tool_gates import _active_tools
    return _active_tools()


def render_stub_module(tools: list[dict[str, Any]]) -> str:
    """Render the stub-module source for `tools` (the envelopes to mirror,
    normally `_active_tools()` for the current turn).

    Memoized on the same sorted-gate-state tuple as `_gate_vector()` (plus the
    rendered names, so a mid-session change to TOOLS membership can never hit
    a stale render). On sub-agent threads (`_subagent_ctx()` active) the
    shared memo is skipped entirely — the tiered set is per-thread state and
    must never be served across agents."""
    from ._subagent_tiers import _subagent_ctx
    names = tuple(sorted(t.get("function", {}).get("name", "") for t in tools))
    if _subagent_ctx() is not None:
        return _render(tools)
    key = (_gate_vector(), names)
    cached = _STUB_CACHE.get(key)
    if cached is not None:
        return cached
    src = _render(tools)
    _STUB_CACHE.clear()  # keep exactly the current gate-vector hot
    _STUB_CACHE[key] = src
    return src


def render_stub_module_for_turn() -> str:
    """Convenience wrapper: the stub module for THIS turn's offered tool set.
    `_active_tools()` is the single source for that set and already returns
    `_subagent_tools()` on sub-agent threads, so a sub-loop never sees the
    parent's memoized stubs."""
    return render_stub_module(_active_tools_for_turn())


def invalidate_stub_cache() -> None:
    """Drop the render memo — call alongside `_invalidate_tool_list_cache()`
    when TOOLS membership changes at runtime."""
    _STUB_CACHE.clear()
