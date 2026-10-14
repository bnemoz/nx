"""Schema-level argument repair: the layer below JSON repair.

nx/_text_fallback.py's repair_json_candidates() fixes args that fail to PARSE;
this module fixes args that parse fine but don't MATCH the tool's declared
schema — the wrong type in a field, a near-miss key name, a JSON-encoded array.
A naive harness hands `{"limit": "5"}` to read_file, Python raises
`TypeError: str indices...`-shaped errors (or worse, silently mis-binds), and
the model burns a turn re-reading the schema. The failure is mechanically
fixable, so fix it before the body sees it.

What is repaired (all demanded to be UNAMBIGUOUS — ambiguity never guesses):

  1. string→integer ("5" → 5, "−3" → -3), string→boolean ("true" → True),
     float→integer when exactly integral (12.0 → 12), int→string (42 → "42"),
     bool→string ("true"/"false").
  2. enum case fold, only when exactly one member matches case-insensitively
     (todo_write's "In_Progress" → "in_progress").
  3. difflib key rename, only when ratio >= _RENAME_CUTOFF (0.8) against
     exactly ONE declared property the args don't already carry ("maxresults"
     → "max_results"). Close-but-ambiguous or clobbering renames are skipped.
  4. string→array when the string is a JSON array literal (todo_write's
     todos="[ {...} ]").
  5. pure per-tool arg ALIASES similarity can never reach ("cmd" → run_bash's
     "command", "old_str" → edit_file's "old") — consulted after the
     similarity pass, same no-clobber rule. Deliberate NON-repair: a comma-/
     newline-separated string is never split into a oneOf array (filenames
     contain commas); that failure is reported, not guessed at.
  6. tool-NAME resolution (resolve_tool_name): one alias-table hit or one
     difflib match >= _RENAME_CUTOFF among the live tool schemas, else None —
     called from run_tool() before the dispatch lookup.
  7. required-args pre-check (check_required): an ERROR naming the missing
     keys and the declared schema, instead of a TypeError from the body.
     Values are never fabricated (missing stays missing).

Every change is appended to a note that run_tool() prepends to the tool
result, so the model learns its args were normalized (same "say so"
discipline as edit_file's whitespace-tolerant note). Nothing here is silent.

Defaults ON (NX_ARG_REPAIR=1); =0 restores strict pass-through. Never raises:
any malformed schema or arg shape degrades to untouched pass-through, per the
repo's "degrade, don't demand" convention.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any

_INT_RE = re.compile(r"[+-]?\d+")

# difflib similarity a candidate key needs before it may stand in for a
# declared property. 0.8 keeps "maxresults"→"max_results" and "pth"→"path",
# rejects a key that sits between two real properties.
_RENAME_CUTOFF = 0.8

# _schemas() cache: (identity of the TOOLS list object, parsed map). Whole-list
# monkeypatching (the dynamic seam the tests rely on) gets a new object and
# invalidates; the import-time TOOLS.append calls happen before any read.
# Kept in module state so the conftest rewind drops it between tests.
_SCHEMAS_CACHE: tuple[Any, dict] | None = None

# Tool-NAME aliases: spellings from other tool ecosystems / a model's own
# shorthand that similarity can never reach ("bash" vs "run_bash" is 0.67).
# Only applied when the alias is not itself a real tool name (checked at use
# time against the live schema list, so a future real "bash" tool wins).
_TOOL_ALIASES: dict[str, str] = {
    "bash": "run_bash", "shell": "run_bash", "execute_bash": "run_bash",
    "run_command": "run_bash", "execute": "run_bash",
    "grep": "search_files", "search": "search_files",
    "find": "search_files", "ripgrep": "search_files",
    "ls": "list_dir", "dir": "list_dir",  # list_dir retired (0.129.8): still dispatches
    "gst": "git_status", "git_status_short": "git_status",  # retired (0.158.9)
    "cat": "read_file", "read": "read_file",
    "str_replace_editor": "edit_file", "str_replace": "edit_file",
    "edit": "edit_file",
    "write": "write_file", "create_file": "write_file",
    "remove": "delete_file", "rm": "delete_file", "delete": "delete_file",
    "test": "run_tests", "pytest": "run_tests", "run_test": "run_tests",
    "commit": "git_commit", "git_add": "git_commit", "gitcommit": "git_commit",
}

# Tools retired from the offered floor (0.129.8) that still dispatch: part of
# the resolution universe so the alias table and near-miss repair reach them
# (run_tool() then answers with the actionable retired-tool refusal).
_RETIRED_DISPATCHABLE = frozenset({"list_dir", "git_status", "git_diff",
                                   "git_log", "sed_edit"})

# Per-tool ARG aliases for near-miss keys the 0.8 similarity cutoff
# deliberately rejects ("cmd"→"command" is 0.6; lowering the cutoff starts
# guessing). Same no-clobber rule as the similarity rename, and the target
# must be a declared property of that tool.
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "run_bash": {"cmd": "command", "commandline": "command",
                 "script": "command", "bash": "command"},
    "read_file": {"file": "path", "file_path": "path", "filepath": "path",
                  "filename": "path"},
    "write_file": {"file": "path", "file_path": "path", "filepath": "path",
                   "filename": "path", "text": "content", "body": "content",
                   "contents": "content"},
    "edit_file": {"old_str": "old", "new_str": "new", "old_text": "old",
                  "new_text": "new", "old_string": "old", "new_string": "new",
                  "file": "path", "file_path": "path", "filepath": "path"},
    "multi_edit": {"file": "path", "file_path": "path", "filepath": "path",
                   "changes": "edits", "replacements": "edits"},
    "search_files": {"query": "pattern", "regex": "pattern",
                     "search": "pattern", "needle": "pattern"},
    "list_dir": {"dir": "path", "directory": "path", "folder": "path"},
    "delete_file": {"file": "path", "file_path": "path", "filename": "path"},
    "git_diff": {"staged_only": "staged"},
    "git_log": {"staged_only": "staged"},
    "todo_write": {"items": "todos", "task_list": "todos"},
    "run_tests": {"target": "targets", "filter": "targets", "path": "targets", "scope": "targets"},
    "git_commit": {"message": "message", "msg": "message", "commit_message": "message",
                   "paths": "files", "file": "files", "file_list": "files"},
}


def enabled() -> bool:
    return os.environ.get("NX_ARG_REPAIR", "1") != "0"


def _expected_type(prop: Any) -> str | None:
    """The declared type of a schema property, in the vocabulary repair
    understands. Returns None for anything unconstrained (no type, or a oneOf
    that isn't the string|array pair read_file/search_files use) so it is
    passed through untouched."""
    if not isinstance(prop, dict):
        return None
    if "enum" in prop:
        return "string"
    t = prop.get("type")
    if isinstance(t, str):
        return t
    arms = prop.get("oneOf")
    if isinstance(arms, list):
        types = {a.get("type") for a in arms
                 if isinstance(a, dict) and isinstance(a.get("type"), str)}
        if types == {"string", "array"}:
            return "string"
    return None


def _coerce(val: Any, exp: str) -> Any:
    """One mechanical coercion, or `val` unchanged when no clean one applies.
    Never raises; unknown shapes fall through."""
    if exp == "integer":
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, str) and _INT_RE.fullmatch(val.strip()):
            return int(val.strip())
        if isinstance(val, float) and val.is_integer():
            return int(val)
    elif exp == "boolean":
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off"):
                return False
    elif exp == "string":
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
    elif exp == "array" and isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except ValueError:
                return val
            if isinstance(parsed, list):
                return parsed
    return val


def _fix_enum(val: Any, prop: dict) -> Any:
    """Case-fold a string enum value when exactly one member matches
    case-insensitively ("In_Progress" → "in_progress"); ambiguous or
    already-valid values pass through."""
    members = prop.get("enum")
    if (not isinstance(members, list) or not isinstance(val, str)
            or val in members):
        return val
    hits = [m for m in members
            if isinstance(m, str) and m.lower() == val.lower()]
    return hits[0] if len(hits) == 1 else val


def _items_props(prop: Any) -> dict | None:
    """Property schema of an array-of-objects element (multi_edit's `edits`,
    todo_write's `todos`), or None."""
    try:
        items = prop.get("items")
        props = items.get("properties")
    except AttributeError:
        return None
    return props if isinstance(props, dict) and props else None


def _repair_mapping(props: dict, out: dict, label: str,
                    notes: list[str], tool_name: str = "") -> dict:
    """The one repair pass, recursive for array-of-objects properties.
    Mutates and returns `out`; appends human-readable notes."""
    aliases = _ARG_ALIASES.get(tool_name, {})
    for key in list(out.keys()):
        val = out[key]
        if key in props:
            prop = props[key]
            exp = _expected_type(prop)
            if exp:
                nv = _coerce(val, exp)
                if exp == "string":
                    nv = _fix_enum(nv, prop)
                if nv is not val and nv != val:
                    out[key] = nv
                    notes.append(f"{label}{key}: {_show(val)} -> {_show(nv)}")
            item_props = _items_props(prop)
            if item_props and isinstance(out.get(key), list):
                for i, item in enumerate(out[key]):
                    if isinstance(item, dict):
                        out[key][i] = _repair_mapping(
                            item_props, item, f"{key}[{i}].", notes)
            continue
        # Unknown key: maybe a near-miss spelling of a declared property.
        # Demands: high similarity to exactly ONE candidate, and that
        # candidate not already present in the args (never clobber).
        if not isinstance(key, str):
            continue
        candidates = [p for p in props
                      if isinstance(p, str) and p not in out
                      and difflib.SequenceMatcher(None, key.lower(),
                                                  p.lower()).ratio()
                      >= _RENAME_CUTOFF]
        if len(candidates) == 1:
            out[candidates[0]] = out.pop(key)
            notes.append(f"{label}{key} -> {candidates[0]} (renamed)")
            continue
        # Pure alias: similarity can't reach it ("cmd" vs "command"), the
        # mapping is curated so there is exactly one target, and the same
        # no-clobber rule applies (never overwrite a real key).
        target = aliases.get(key)
        if target and target in props and target not in out:
            out[target] = out.pop(key)
            notes.append(f"{label}{key} -> {target} (renamed)")
    return out


def _show(v: Any) -> str:
    s = json.dumps(v) if isinstance(v, str) else str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def _schemas() -> dict[str, tuple[dict, list]]:
    """name -> (declared property schemas, required list), from the live nx
    module (read at call time like the rest of the dynamic seams). TOOLS
    carries the 17 always-on schemas plus the appended conditional ones;
    final_answer lives in its own module constant. Malformed entries are
    skipped, not fatal.

    0.30.0: cached on the IDENTITY of the live TOOLS list object — monkey-
    patching a whole new list (the dynamic seam tests rely on) invalidates,
    while repeated calls in a turn (repair_args + check_required per tool
    call) stop re-walking every schema."""
    global _SCHEMAS_CACHE
    import nx as _m
    tools_obj = getattr(_m, "TOOLS", None)
    tools = list(tools_obj or ())
    fa = getattr(_m, "FINAL_ANSWER_TOOL", None)
    if fa is not None:
        tools = tools + [fa]
    if _SCHEMAS_CACHE is not None and _SCHEMAS_CACHE[0] is tools_obj:
        return _SCHEMAS_CACHE[1]
    out: dict[str, tuple[dict, list]] = {}
    for t in tools:
        try:
            fn = t["function"]
            props = fn["parameters"]["properties"]
            required = fn["parameters"].get("required") or []
        except (KeyError, TypeError, IndexError):
            continue
        if isinstance(props, dict) and props:
            out[fn["name"]] = (props, required if isinstance(required, list) else [])
    _SCHEMAS_CACHE = (tools_obj, out)
    return out


def _tool_names() -> set[str]:
    """Every dispatchable tool name, from the live schemas (final_answer
    included — it is a real tool to the model even though it never lands in
    the DISPATCH table)."""
    return set(_schemas())


def resolve_tool_name(name: Any) -> tuple[str | None, str | None]:
    """Map a near-miss tool NAME onto the real one, or give up.

    Order: exact hit → alias table → difflib >= _RENAME_CUTOFF against the
    live schema names, demanding exactly ONE candidate (ambiguity never
    guesses — the caller's unknown-tool error lists the close names instead).
    Returns (resolved_name_or_None, note_or_None). Never raises; non-string
    names degrade to (None, None).

    Resolution deliberately spans the FULL schema set (not the tools offered
    this turn): the caller still has to get the name through run_tool(), which
    re-checks the breaker and the session gates against the RESOLVED name so it
    can answer with the reason the tool is unavailable rather than 'unknown'."""
    if not isinstance(name, str) or not name.strip():
        return None, None
    cand = name.strip()
    tools = _tool_names()
    # Retired-from-the-floor tools (0.129.8) stay resolvable: they are no
    # longer in TOOLS, but they still dispatch (tests, direct callers), and a
    # model that emits the name should reach run_tool()'s actionable
    # "retired, use X instead" refusal rather than "unknown tool".
    tools = tools | _RETIRED_DISPATCHABLE
    if cand in tools:
        return cand, None
    if cand in _TOOL_ALIASES:
        target = _TOOL_ALIASES[cand]
        if target in tools:  # an alias never shadows a real tool name
            return target, f"tool {cand} -> {target} (aliased)"
        return None, None
    hits = [t for t in sorted(tools)
            if difflib.SequenceMatcher(None, cand.lower(), t.lower()).ratio()
            >= _RENAME_CUTOFF]
    if len(hits) == 1:
        return hits[0], f"tool {cand} -> {hits[0]} (renamed)"
    return None, None  # ambiguous or no match: caller reports, never guesses


def _type_label(prop: Any) -> str:
    """Terse declared-type label for the required-args message ("string",
    "integer", "enum", "string|array"); unconstrained shapes read "any"."""
    if not isinstance(prop, dict):
        return "any"
    if "enum" in prop:
        return "enum"
    t = prop.get("type")
    if isinstance(t, str):
        return t
    arms = prop.get("oneOf")
    if isinstance(arms, list):
        types = sorted({a.get("type") for a in arms
                        if isinstance(a, dict) and isinstance(a.get("type"), str)})
        if types:
            return "|".join(types)
    return "any"


def check_required(name: str, args: Any) -> str | None:
    """ERROR text when required args are missing, else None.

    Fires after repair_args (so a renamed key counts as present) and before
    dispatch, where the body would otherwise die with a TypeError the model
    can't act on. Lists the declared schema so the model can re-send without
    a round-trip. Values are never fabricated: a missing required arg is
    reported, not invented."""
    if not isinstance(args, dict):
        return None
    entry = _schemas().get(name)
    if not entry:
        return None
    props, required = entry
    if not required:
        return None
    missing = [r for r in required if r not in args or args[r] is None]
    if not missing:
        return None
    schema = ", ".join(f"{k}: {_type_label(v)}" for k, v in props.items())
    quoted = ", ".join(f'"{m_}"' for m_ in missing)
    return (f"ERROR: {name} missing required {quoted} — schema: {schema}. "
            "Re-send with all required arguments.")


def repair_args(name: str, args: Any) -> tuple[Any, list[str]]:
    """Repair `args` for tool `name` against its declared schema.

    Returns (args, notes): args unchanged unless something was mechanically
    fixable; notes are the terse "what was fixed" strings run_tool() prepends
    to the result. A non-dict args object, an unknown tool, or a disabled
    gate returns (args, []) — strict pass-through."""
    if not enabled() or not isinstance(args, dict) or not args:
        return args, []
    entry = _schemas().get(name)
    if not entry:
        return args, []
    notes: list[str] = []
    repaired = _repair_mapping(entry[0], dict(args), "", notes, tool_name=name)
    if notes:
        try:
            from . import _state as _st
            _st._TALLY_ARG_REPAIRS += len(notes)
        except Exception:  # tally is observability, never a failure path
            pass
    return (repaired, notes) if notes else (args, [])
