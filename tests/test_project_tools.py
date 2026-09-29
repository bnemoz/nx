"""Regression coverage for manifest-backed project tool gateways."""
from __future__ import annotations

import json


def _manifest(tmp_path, entries):
    tools = tmp_path / ".nx" / "tools"
    tools.mkdir(parents=True)
    (tools / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return tools


def test_search_reads_manifest_without_importing_handler(monkeypatch, tmp_path):
    from nx._project_tools import search_project_tools

    tools = _manifest(tmp_path, {
        "hello": {
            "module": "hello", "handler": "run", "description": "Return a greeting.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "risk": "read",
        }
    })
    (tools / "hello.py").write_text("raise RuntimeError('must not import while searching')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NX_PROJECT_TOOLS", "1")

    result = search_project_tools()

    assert "hello" in result
    assert "Return a greeting." in result


def test_gate_exposes_stable_gateway_pair_only_when_enabled(monkeypatch):
    import nx

    monkeypatch.delenv("NX_PROJECT_TOOLS", raising=False)
    nx._invalidate_tool_list_cache()
    assert "search_project_tools" not in {t["function"]["name"] for t in nx._active_tools()}

    monkeypatch.setenv("NX_PROJECT_TOOLS", "1")
    nx._invalidate_tool_list_cache()
    names = {t["function"]["name"] for t in nx._active_tools()}
    assert {"search_project_tools", "call_project_tool"} <= names


def test_call_validates_then_invokes_read_handler(monkeypatch, tmp_path):
    from nx._project_tools import call_project_tool

    tools = _manifest(tmp_path, {
        "greet": {
            "module": "greet", "handler": "run", "description": "Return a named greeting.",
            "parameters": {
                "type": "object", "properties": {"who": {"type": "string"}},
                "required": ["who"], "additionalProperties": False,
            },
            "risk": "read",
        }
    })
    (tools / "greet.py").write_text("def run(who):\n    return 'hello ' + who\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NX_PROJECT_TOOLS", "1")

    assert "missing required who" in call_project_tool("greet", {})
    assert "unknown properties extra" in call_project_tool("greet", {"who": "Ada", "extra": 1})
    assert call_project_tool("greet", {"who": "Ada"}) == "hello Ada"


def test_write_tool_requires_approval_before_import(monkeypatch, tmp_path):
    import nx._project_tools as project_tools

    tools = _manifest(tmp_path, {
        "deploy": {
            "module": "deploy", "handler": "run", "description": "Deploy the preview.",
            "parameters": {"type": "object", "properties": {}}, "risk": "write",
        }
    })
    (tools / "deploy.py").write_text("def run():\n    return 'deployed'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NX_PROJECT_TOOLS", "1")
    monkeypatch.setattr(project_tools, "_confirm", lambda _action: False)

    assert project_tools.call_project_tool("deploy", {}) == "DENIED by user"
