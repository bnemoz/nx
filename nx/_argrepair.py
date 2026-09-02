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
                    notes: list[str]) -> dict:
    """The one repair pass, recursive for array-of-objects properties.
    Mutates and returns `out`; appends human-readable notes."""
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
    return out


def _show(v: Any) -> str:
    s = json.dumps(v) if isinstance(v, str) else str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def _schemas() -> dict[str, dict]:
    """name -> declared property schemas, from the live nx module (read at
    call time like the rest of the dynamic seams). TOOLS carries the 17
    always-on schemas plus the appended conditional ones; final_answer lives
    in its own module constant. Malformed entries are skipped, not fatal."""
    import nx as _m
    tools = list(getattr(_m, "TOOLS", ()) or ())
    fa = getattr(_m, "FINAL_ANSWER_TOOL", None)
    if fa is not None:
        tools = tools + [fa]
    out: dict[str, dict] = {}
    for t in tools:
        try:
            fn = t["function"]
            props = fn["parameters"]["properties"]
        except (KeyError, TypeError, IndexError):
            continue
        if isinstance(props, dict) and props:
            out[fn["name"]] = props
    return out


def repair_args(name: str, args: Any) -> tuple[Any, list[str]]:
    """Repair `args` for tool `name` against its declared schema.

    Returns (args, notes): args unchanged unless something was mechanically
    fixable; notes are the terse "what was fixed" strings run_tool() prepends
    to the result. A non-dict args object, an unknown tool, or a disabled
    gate returns (args, []) — strict pass-through."""
    if not enabled() or not isinstance(args, dict) or not args:
        return args, []
    props = _schemas().get(name)
    if not props:
        return args, []
    notes: list[str] = []
    repaired = _repair_mapping(props, dict(args), "", notes)
    return (repaired, notes) if notes else (args, [])
