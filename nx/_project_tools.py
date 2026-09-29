"""Opt-in, manifest-backed project tools.

Project tools live under ``.nx/tools/`` but never become individual provider
tool schemas: the stable ``search_project_tools`` and ``call_project_tool``
gateways keep the request's tool prefix cacheable.  Discovery reads only
``manifest.json``; a project's Python is imported only after a call is
approved and dispatched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._approval import _confirm

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class ProjectTool:
    name: str
    module: str
    handler: str
    description: str
    parameters: dict[str, Any]
    risk: str
    tools_dir: Path


def _enabled() -> bool:
    return os.environ.get("NX_PROJECT_TOOLS", "0") not in ("", "0", "false", "False")


def _manifest_path() -> Path:
    # Keep project tools alongside the repo root even when nx was launched
    # from a subdirectory. Tests/one-shot use still fall back to the cwd when
    # no root-level manifest exists.
    from . import _state as st
    root = Path(getattr(st, "PROJECT_ROOT", None) or Path.cwd())
    candidate = root / ".nx" / "tools" / "manifest.json"
    return candidate if candidate.exists() else Path.cwd() / ".nx" / "tools" / "manifest.json"


def _error(message: str) -> str:
    return f"ERROR: project tools: {message}"


def _registry() -> tuple[dict[str, ProjectTool], str | None]:
    """Read and validate manifest metadata without importing project code."""
    if not _enabled():
        return {}, "disabled (set NX_PROJECT_TOOLS=1 to enable .nx/tools/manifest.json)"
    path = _manifest_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, f"no manifest at {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"cannot read {path}: {exc}"
    if not isinstance(raw, dict):
        return {}, "manifest root must be an object mapping tool names to definitions"
    out: dict[str, ProjectTool] = {}
    for name, item in raw.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            return {}, f"invalid tool name {name!r}"
        if not isinstance(item, dict):
            return {}, f"{name}: definition must be an object"
        module = item.get("module")
        handler = item.get("handler")
        description = item.get("description")
        parameters = item.get("parameters")
        risk = item.get("risk", "write")
        if not all(isinstance(v, str) and _NAME.fullmatch(v)
                   for v in (module, handler)):
            return {}, f"{name}: module and handler must be Python identifiers"
        if not isinstance(description, str) or not description.strip():
            return {}, f"{name}: description must be a non-empty string"
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            return {}, f"{name}: parameters must be an object JSON Schema"
        if risk not in ("read", "write"):
            return {}, f"{name}: risk must be 'read' or 'write'"
        out[name] = ProjectTool(name, module, handler, description, parameters,
                                risk, path.parent)
    return out, None


def search_project_tools(pattern: str = "") -> str:
    """Return local tool metadata and schemas without importing handlers."""
    tools, problem = _registry()
    if problem:
        return _error(problem) if problem.startswith("cannot") else f"Project tools: {problem}."
    try:
        rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    except re.error as exc:
        return _error(f"invalid pattern {pattern!r}: {exc}")
    found = []
    for tool in tools.values():
        record = {"name": tool.name, "description": tool.description,
                  "parameters": tool.parameters, "risk": tool.risk}
        if rx and not rx.search(tool.name + " " + tool.description + " " + json.dumps(tool.parameters)):
            continue
        found.append(record)
    if not found:
        return "No project tools matched." if tools else "Project tools: no tools registered."
    return "Project tool schemas:\n" + json.dumps(found, ensure_ascii=False)


def _validate(value: Any, schema: dict[str, Any], label: str = "arguments") -> str | None:
    """Small, dependency-free JSON-Schema subset for project call arguments."""
    kind = schema.get("type")
    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict}
    if kind in types and (not isinstance(value, types[kind])
                           or kind in ("integer", "number") and isinstance(value, bool)):
        return f"{label} must be a {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{label} must be one of {schema['enum']!r}"
    if isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict) or not isinstance(required, list):
            return f"{label} has an invalid object schema"
        missing = [key for key in required if key not in value]
        if missing:
            return f"{label} is missing required {', '.join(map(str, missing))}"
        if schema.get("additionalProperties") is False:
            unknown = [key for key in value if key not in props]
            if unknown:
                return f"{label} has unknown properties {', '.join(map(str, unknown))}"
        for key, child in props.items():
            if key in value and isinstance(child, dict):
                err = _validate(value[key], child, f"{label}.{key}")
                if err:
                    return err
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, child_value in enumerate(value):
            err = _validate(child_value, schema["items"], f"{label}[{index}]")
            if err:
                return err
    return None


def _load_handler(tool: ProjectTool):
    path = (tool.tools_dir / f"{tool.module}.py").resolve()
    try:
        path.relative_to(tool.tools_dir.resolve())
    except ValueError as exc:
        raise ValueError("module resolves outside .nx/tools") from exc
    if not path.is_file():
        raise ValueError(f"module file not found: {path.name}")
    module_name = f"_nx_project_tool_{tool.module}_{abs(hash(str(path)))}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load {path.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    handler = getattr(module, tool.handler, None)
    if not callable(handler):
        raise ValueError(f"handler {tool.handler!r} is not callable")
    return handler


def call_project_tool(name: str, arguments: dict[str, Any]) -> str | Any:
    """Validate, approve when declared write-capable, then invoke one handler."""
    tools, problem = _registry()
    if problem:
        return _error(problem)
    tool = tools.get(name)
    if tool is None:
        return _error(f"unknown tool {name!r}; use search_project_tools first")
    if not isinstance(arguments, dict):
        return _error("arguments must be an object")
    validation = _validate(arguments, tool.parameters)
    if validation:
        return _error(f"{tool.name}: {validation}")
    if tool.risk == "write":
        from ._plan import plan_active, gate_message
        if plan_active():
            return gate_message()
        if not _confirm(f"project tool {tool.name}: {tool.description}"):
            return "DENIED by user"
    try:
        return _load_handler(tool)(**arguments)
    except Exception as exc:  # noqa: BLE001 - handler errors belong in the tool result
        return _error(f"{tool.name}: {type(exc).__name__}: {exc}")
