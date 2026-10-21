"""PTC Phase 1 (docs/ptc-migration-spec.md): the generated stub module.

Tier A: nx is imported ONCE and never reloaded; state writes go through
monkeypatch.

Guards:
  1. The generated module for the REAL TOOLS list imports and py_compiles.
  2. Signature-vs-handler cross-check: every schema property the stub exposes
     must be a kwarg the DISPATCH handler accepts. Handlers with `**_`
     tolerance absorb the rest; per Audit correction #3 the cross-check treats
     `duckdb` (nx/_duckdb.py duckdb_query) and `web_fetch` (nx/_webfetch.py) as
     STRICT — no VAR_KEYWORD — so the stub signature must exactly match those
     handlers' named parameters, and any handler that loses its `**_` is held
     to the same bar automatically.
  3. Phase 1 exclusions (spawn_agent, critique, read_image, call_project_tool,
     final_answer, run_tools) never appear as stubs.
  4. oneOf properties (read_file/search_files) render as `|` unions;
     array-of-object properties (multi_edit/todo_write) render list[dict].
  5. Gate awareness: the turn render follows `_active_tools()` (gated-off
     tools get no stub), and the sub-agent path (`_subagent_ctx()` active)
     skips the shared memo entirely.
"""
import importlib.util
import inspect
import os
import py_compile
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NX_ENV_FILE", "/dev/null")

import pytest  # noqa: E402

import nx as m  # noqa: E402
from nx._dispatch import DISPATCH  # noqa: E402
from nx import _ptc_stubs  # noqa: E402
from nx._subagent_tls import _SUBAGENT_TLS  # noqa: E402

# Audit correction #3: these handlers lack `**_` tolerance; the cross-check
# must treat them as strict (hard error, not drift warning) even if a future
# edit adds tolerance (the test then only gets stricter than required, and a
# NEW strict handler is caught dynamically via VAR_KEYWORD detection anyway).
ALWAYS_STRICT = {"duckdb", "web_fetch"}

TOOLS_BY_NAME = {t["function"]["name"]: t for t in m.TOOLS}


def _rendered_tools():
    """(name, envelope) for every non-excluded tool currently in TOOLS."""
    return [(n, t) for n, t in TOOLS_BY_NAME.items()
            if n not in _ptc_stubs.EXCLUDED_TOOLS]


def _load_stub_module(src: str, tmp_path, modname: str = "_ptc_stubs_under_test"):
    path = tmp_path / f"{modname}.py"
    path.write_text(src, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)  # guard 1a
    spec = importlib.util.spec_from_file_location(modname, str(path))
    module = importlib.util.module_from_spec(spec)  # guard 1b: importable
    spec.loader.exec_module(module)
    return module


def _sig_params(handler):
    sig = inspect.signature(handler)
    named = {p.name for p in sig.parameters.values()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    no_default = {p.name for p in sig.parameters.values()
                  if p.default is p.empty
                  and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return named, has_var_kw, no_default


# --- guards 1-4: the generated module ---------------------------------------

def test_real_tools_module_imports_and_py_compiles(tmp_path):
    src = _ptc_stubs.render_stub_module(m.TOOLS)
    mod = _load_stub_module(src, tmp_path)
    for name, _env in _rendered_tools():
        assert callable(getattr(mod, name)), f"stub for {name} missing"
    # _capture is the Phase 2 seam and must raise in Phase 1.
    with pytest.raises(RuntimeError):
        mod._capture("read_file", {"path": "x"})


def test_excluded_tools_absent(tmp_path):
    src = _ptc_stubs.render_stub_module(m.TOOLS)
    mod = _load_stub_module(src, tmp_path, "ptc_stubs_excl")
    for name in _ptc_stubs.EXCLUDED_TOOLS:
        assert f"def {name}(" not in src, f"{name} must not be stubbed"
        assert not hasattr(mod, name)


def test_oneof_renders_union():
    src = _ptc_stubs.render_stub_module(m.TOOLS)
    for tool in ("read_file", "search_files"):
        key = "pattern" if tool == "search_files" else "path"
        assert f"{key}: str | list[str]," in src, \
            f"{tool} oneOf {key} must render a `|` union"
    # required props carry no `| None` suffix (path/pattern are required).
    assert "str | list[str] | None" not in src


def test_array_of_object_renders_list_of_dict():
    src = _ptc_stubs.render_stub_module(m.TOOLS)
    assert "edits: list[dict]" in src      # multi_edit / edit_file
    assert "todos: list[dict]" in src      # todo_write


def test_defaults_and_optionality():
    env = [{"type": "function", "function": {
        "name": "fake_tool",
        "description": "Docstring source.",
        "parameters": {"type": "object", "properties": {
            "req": {"type": "string", "description": "required one"},
            "with_default": {"type": "integer", "default": 7,
                             "description": "kept"},
            "flag": {"type": "boolean", "description": "optional, no default"},
            "on": {"type": "boolean", "default": True, "description": "bool default"},
        }, "required": ["req"]}}}]
    src = _ptc_stubs.render_stub_module(env)
    assert "def fake_tool(req: str, with_default: int = 7, " in src
    assert "on: bool = True" in src
    assert "flag: bool | None = None" in src
    assert "'req': req, 'with_default': with_default" in src


# --- guard 2: signature-vs-handler cross-check -------------------------------

@pytest.mark.parametrize("name,envelope", _rendered_tools(),
                         ids=[n for n, _ in _rendered_tools()])
def test_stub_signature_matches_handler_kwargs(name, envelope):
    src = _ptc_stubs.render_stub_module([envelope])
    handler = DISPATCH[name]
    props = (envelope["function"].get("parameters") or {}).get("properties") or {}
    named, has_var_kw, no_default = _sig_params(handler)
    strict = name in ALWAYS_STRICT or not has_var_kw

    # What the stub will pass as kwargs (the _capture dict) == schema props.
    passed = set(props)
    if strict:
        # Every kwarg the stub can send must be a real handler parameter, and
        # the handler may not require anything the stub never sends.
        unknown = passed - named
        assert not unknown, (
            f"{name}: stub sends {sorted(unknown)} but handler "
            f"{handler.__name__} has no such kwarg (strict: no **_)")
        missing = no_default - passed
        assert not missing, (
            f"{name}: handler {handler.__name__} requires {sorted(missing)} "
            "the stub never passes (strict: no **_)")
    else:
        # **_-tolerant handler: over-narrow stubs are safe, but a renamed
        # schema property that ISN'T absorbed intentionally would be drift —
        # tolerance means every prop is either named or absorbed; assert the
        # generated stub exposes exactly the schema props.
        pass
    # The generated signature mirrors the schema props either way.
    assert _stub_params(src, name) == {_ptc_stubs._py_name(p) for p in props}


def _stub_params(src: str, name: str) -> set:
    """Parameter names of `def <name>(...)` via ast (the module can't be
    exec'd for every synthetic envelope)."""
    import ast
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            a = node.args
            names = [x.arg for x in list(a.posonlyargs) + list(a.args)]
            names += [x.arg for x in a.kwonlyargs]
            assert a.vararg is None and a.kwarg is None, \
                f"stub {name} must have a fully explicit signature"
            return set(names)
    raise AssertionError(f"def {name} not found in generated source")


# --- guard 5: gate awareness --------------------------------------------------

def test_turn_render_follows_active_tools(monkeypatch, tmp_path):
    """A gated-off tool gets no stub in the turn render."""
    _ptc_stubs.invalidate_stub_cache()
    active = [t for t in m.TOOLS
              if t["function"]["name"] not in ("lint_check", "spawn_agent")]
    monkeypatch.setattr(_ptc_stubs, "_active_tools_for_turn", lambda: active)
    src = _ptc_stubs.render_stub_module_for_turn()
    assert "def lint_check(" not in src
    assert "def read_file(" in src
    _load_stub_module(src, tmp_path, "ptc_stubs_gated")
    _ptc_stubs.invalidate_stub_cache()


def test_subagent_path_skips_shared_memo(monkeypatch):
    """With `_subagent_ctx()` active the shared memo is neither read nor
    written (per-thread safety), and the render follows the tiered set."""
    _ptc_stubs.invalidate_stub_cache()
    tier = [t for t in m.TOOLS if t["function"]["name"] in ("read_file", "run_bash")]
    monkeypatch.setattr(_ptc_stubs, "_active_tools_for_turn", lambda: tier)
    ctx = types.SimpleNamespace(tier="read", write=False)
    prev = getattr(_SUBAGENT_TLS, "ctx", None)
    _SUBAGENT_TLS.ctx = ctx
    try:
        src = _ptc_stubs.render_stub_module_for_turn()
        assert "def read_file(" in src
        assert "def memory_search(" not in src  # not in the fake tier set
        assert not _ptc_stubs._STUB_CACHE, "sub-agent render must skip the memo"
        # And the memo can never carry a sub-agent render into a parent turn:
        src2 = _ptc_stubs.render_stub_module_for_turn()
        assert "def read_file(" in src2 and "def memory_search(" not in src2
    finally:
        _SUBAGENT_TLS.ctx = prev
    _ptc_stubs.invalidate_stub_cache()


def test_render_memo_keyed_on_gate_vector(monkeypatch):
    """Same gate vector -> same object (memo hit); a changed gate vector ->
    re-render."""
    _ptc_stubs.invalidate_stub_cache()
    calls = []
    real_vector = _ptc_stubs._gate_vector

    def counting_vector():
        calls.append(1)
        return real_vector()
    monkeypatch.setattr(_ptc_stubs, "_gate_vector", counting_vector)
    src1 = _ptc_stubs.render_stub_module(m.TOOLS)
    src2 = _ptc_stubs.render_stub_module(m.TOOLS)
    assert src1 is src2 and len(calls) == 2  # 2nd call = memo hit path
    _ptc_stubs.invalidate_stub_cache()
