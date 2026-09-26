"""Plan mode: a write-gate state machine, not a prompt nudge.

The failure mode: a coding model spends its whole turn *reasoning about*
edits it never makes — the reasoning-stall nudges (REASONING_STALL_NUDGE &
friends) fight the symptom every turn. Plan mode makes the constraint
structural: while a plan is active, every mutating tool call is refused
*before it runs* (and before its approval prompt) with a gate message that
tells the model exactly how to proceed — edit the plan, or get it confirmed.
Writing code before thinking is simply not expressible until the gate is
released, so "think first" stops depending on the model's mood.

State lives in `st._PLAN_*` (see _state.py) and is persisted into the session
JSON ("plan" keys) so /resume restores it, exactly like st._TODOS.

Lifecycle:
  /plan <goal>   activate — writes .nx/<slug>.md, injects a plan message block
  /plan          show the active plan + path
  /plan done     release the gate (the plan file stays on disk)
  /plan off      alias for `done` (plus `discard`, which also deletes the file)
  /plan edit ... deprecated alias for `done`+re-activate — see _cmd_plan

Model-side: the `plan_edit` tool is the *only* mutating call allowed while the
gate is down (plan-file writes and re-reads are always permitted — locking the
model out of its own plan would force it back into reasoning-only stalls).
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from . import _state as st
from ._ansi import BOLD, DIM, RESET, C

# Tools the gate blocks. NOT listed: read-only tools (which never mutate),
# plan_edit (the gate's own pressure valve), and apply_patch/sed_edit/
# delete_file/spawn_agent etc. are listed explicitly so an unreviewed addition
# to the toolset stays *in* the safe default (deny-by-default for the rest).
PLAN_GATE_TOOLS = frozenset((
    "write_file", "edit_file", "multi_edit", "sed_edit", "apply_patch",
    "delete_file", "run_bash", "run_tests", "memory_write", "memory_delete",
    "git_commit", "duckdb",
))

# Read-only tools are never gated. Listed explicitly (rather than "everything
# else") so a future state-touching tool can't silently slip past the gate.
PLAN_READONLY_TOOLS = frozenset((
    "read_file", "list_dir", "search_files", "git_status", "git_diff",
    "git_log", "memory_search", "read_image", "plan_edit", "todo_write",
    "spawn_agent", "wait_background",
))


def slugify(goal: str, max_words: int = 8) -> str:
    """First `max_words` words of `goal`, lowercased, joined with '-'.

    Matches _memory_cmd._slugify's shape (ascii-word chars, one dash between)
    but drops non-leading digits so a goal like "3 bugs" can't slugify to a
    bare number and collide with session-file-ish names. The cap is tunable
    via NX_PLAN_MAX_WORDS (read lazily so tests can flip it).
    """
    try:
        import nx as _m
        max_words = int(getattr(_m, "NX_PLAN_MAX_WORDS", max_words))
    except (AttributeError, TypeError, ValueError):
        from ._env import _env_int
        max_words = _env_int("NX_PLAN_MAX_WORDS", max_words)
    words = re.findall(r"[A-Za-z0-9]+", goal.lower())
    words = [w for w in words if not w.isdigit() or not re.match(r"^[0-9]", w)]
    return "-".join(words[:max_words]) or "plan"


def plan_dir() -> str:
    """The directory plan files live in: `<project root>/.nx`, falling back to
    cwd when PROJECT_ROOT isn't set (one-shot / tests)."""
    root = st.PROJECT_ROOT or os.getcwd()
    return os.path.join(root, ".nx")


def plan_path() -> "str | None":
    """The active plan's file path, or None when no plan is active."""
    if not st._PLAN_ACTIVE:
        return None
    return os.path.join(plan_dir(), f"{st._PLAN_SLUG}.md")


def render_plan_display() -> str:
    """Colored terminal rendering of the active plan (for /plan with no arg)."""
    path = plan_path()
    if not st._PLAN_ACTIVE or not path or not os.path.isfile(path):
        return "(no plan active)"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            body = f.read()
    except OSError:
        return "(no plan active)"
    lines = body.splitlines()
    if len(lines) > 40:
        lines = lines[:40] + [f"... [{len(lines) - 40} more lines]"]
    return "\n".join(lines)


def plan_active() -> bool:
    """The one-line gate check run_tool() consults. Kept a function (not a
    bare `st._PLAN_ACTIVE` read at call sites) so tests and future callers
    have one name to patch, and so the session-restore path (which may set
    the flag from saved JSON) has a single source of truth."""
    return bool(st._PLAN_ACTIVE)


def _latest_plan_turn(body: list[dict[str, Any]]) -> int | None:
    """Index of the latest plan_edit call+result pair in `body` (a foldable
    head slice, no system message), or None.

    Same contract as _todo._latest_todo_turn: compress() expands pins to
    complete tool_call/tool-result pairs, so pinning the assistant turn that
    last touched the plan carries the plan's latest state through every fold —
    the gate contract never goes stale mid-compression. Suppressed while the
    gate is down (/plan done): a released plan is just a file, not live state.
    """
    if not plan_active():
        return None
    for i in range(len(body) - 1, -1, -1):
        for tc in body[i].get("tool_calls") or []:
            if isinstance(tc, dict) and (tc.get("function") or {}).get("name") == "plan_edit":
                return i
    return None


# The gate message returned as the tool result when a mutating call is blocked.
# Told to the model, so it must be self-sufficient: name the two ways out and
# the fact that plan_edit is exempt.
PLAN_GATE_MESSAGE = (
    "[Plan mode is active — this tool call was refused. A plan exists at "
    "{path} and must be settled before the code changes:\n"
    "- To adjust the approach: call plan_edit(old=..., new=...) to update the "
    "plan, then re-attempt.\n"
    "- If the plan is right as written: stop and tell the user to run "
    "`/plan done` (they may also say `/plan off`, or edit .nx/{slug}.md "
    "themselves). Do not attempt the mutation again this turn.\n"
    "Read-only tools and plan_edit remain available.]"
)


def gate_message() -> str:
    path = plan_path() or f"{plan_dir()}/{st._PLAN_SLUG}.md"
    slug = st._PLAN_SLUG or "plan"
    return PLAN_GATE_MESSAGE.format(path=path, slug=slug)


def check_gate(name: str) -> str | None:
    """Return the gate message when tool `name` must be refused under plan
    mode, else None. The single choke point run_tool() calls before dispatch."""
    if not plan_active() or name not in PLAN_GATE_TOOLS:
        return None
    return gate_message()


# --- plan_edit tool ----------------------------------------------------------
PLAN_EDIT_TOOL = {"type": "function", "function": {
    "name": "plan_edit",
    "description": "Edit the active plan file (.nx/<slug>.md) while plan mode is active -- the only mutation plan mode allows.",
    "parameters": {"type": "object", "properties": {
        "new": {"type": "string", "description": "The text to put in place (required unless using `write`)"},
        "old": {"type": "string", "description": "Literal passage to replace, matching exactly once"},
        "old_lines": {"type": "string", "description": "Line range to replace, 'N' or 'N-M', as numbered by read_file"},
        "write": {"type": "string", "description": "Full replacement plan body; omit `old`/`old_lines`"},
    }, "oneOf": [
        # A full rewrite is the documented alternative to a splice.  Keep
        # this at the schema level so native callers may send {"write": ...}
        # without inventing a meaningless `new` value.
        {"required": ["write"], "not": {"anyOf": [
            {"required": ["new"]},
            {"required": ["old"]},
            {"required": ["old_lines"]},
        ]}},
        {"required": ["new", "old"]},
        {"required": ["new", "old_lines"]},
    ]},
}}


def plan_edit(old: str | None = None, new: str = "", old_lines: str | None = None,
              write: str | None = None) -> str:
    """The plan_edit tool body: splice/replace the active plan file.

    Deliberately does NOT go through _confirm(): the plan is the model's own
    scratch state, the user sees every edit in the tool box (and can /plan
    done|off|discard it at will), and a Y/n prompt on every plan tweak would
    push the model back toward reasoning-only stalls — the exact failure mode
    plan mode exists to remove. Writes that don't touch the active plan's file
    are refused (a plan edit is not a code-write loophole)."""
    path = plan_path()
    if not path:
        return "ERROR: no plan is active — start one with /plan <goal>"
    arg_preview = (write if write is not None else new) or ""
    print(f"\n{C.ACCENT}  ┌─ plan_edit {path}{RESET}")
    print(f"{C.ACCENT}  │ {RESET}{DIM}{arg_preview[:120]}{RESET}")
    if not os.path.isfile(path):
        # A /plan done → /plan <goal> cycle reuses the slug; treat a missing
        # file as a fresh plan body rather than erroring the model into a
        # dead end.
        content = ""
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            print(f"{C.ACCENT}  └─{RESET}")
            return f"ERROR: can't read plan file: {type(e).__name__}: {e}"
    if write is not None:
        body = write
        action = "rewrote"
    elif old_lines:
        m = re.match(r"^(\d+)(?:-(\d+))?$", old_lines.strip())
        if not m:
            print(f"{C.ACCENT}  └─{RESET}")
            return f"ERROR: old_lines must be 'N' or 'N-M', got {old_lines!r}"
        start = int(m.group(1))
        end = int(m.group(2) or start)
        lines = content.split("\n")
        if start < 1 or end < start or end > len(lines):
            print(f"{C.ACCENT}  └─{RESET}")
            return (f"ERROR: old_lines {old_lines!r} out of range "
                    f"(file has {len(lines)} lines)")
        body = "\n".join(lines[:start - 1] + [new] + lines[end:])
        action = f"replaced lines {start}-{end}"
    elif old:
        if content.count(old) != 1:
            n = content.count(old)
            print(f"{C.ACCENT}  └─{RESET}")
            return (f"ERROR: old matches {n} time(s) in the plan "
                    f"(must be exactly once) — widen or split the edit")
        body = content.replace(old, new, 1)
        action = "edited"
    else:
        print(f"{C.ACCENT}  └─{RESET}")
        return "ERROR: give old (literal splice), old_lines, or write (full replacement)"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
    except OSError as e:
        print(f"{C.ACCENT}  └─{RESET}")
        return f"ERROR: can't write plan file: {type(e).__name__}: {e}"
    # Announce like a tool box close so the plan edit is visible in the flow.
    print(f"{C.ACCENT}  └─{RESET} {DIM}{action}{RESET}")
    return f"plan updated ({action}): {path}"


# --- /plan command -----------------------------------------------------------
def _cmd_plan(user: str, state: Any) -> None:
    """/plan — plan-mode state machine.

        /plan <goal>   activate: write .nx/<slug>.md + inject a plan block
        /plan          show the active plan
        /plan done     release the gate (plan file stays)
        /plan off      alias for done
        /plan discard  release AND delete the plan file
        /plan edit ... re-activate against the existing file (deprecated alias
                       of `done` + /plan <goal>, kept so old muscle memory
                       still lands somewhere sane)
    """
    arg = user[len("/plan"):].strip()
    if arg in ("done", "off"):
        if not st._PLAN_ACTIVE:
            print(f"{DIM}  no plan active{RESET}")
            return
        path = plan_path()
        st._PLAN_ACTIVE = False
        st._PLAN_GOAL = None
        st._PLAN_SLUG = None
        st._PLAN_STARTED = None
        if state is not None and getattr(state, "session_dirty", None) is not None:
            state.session_dirty = True
        print(f"{BOLD}{C.SUCCESS}  → plan released{RESET} {DIM}(file kept: {path}){RESET}")
        print(f"{DIM}  writes and bash are unlocked{RESET}")
        return
    if arg == "discard":
        if not st._PLAN_ACTIVE:
            print(f"{DIM}  no plan active{RESET}")
            return
        path = plan_path()
        st._PLAN_ACTIVE = False
        st._PLAN_GOAL = None
        st._PLAN_SLUG = None
        st._PLAN_STARTED = None
        if state is not None and getattr(state, "session_dirty", None) is not None:
            state.session_dirty = True
        try:
            if path and os.path.isfile(path):
                os.remove(path)
                print(f"{BOLD}{C.SUCCESS}  → plan discarded{RESET} {DIM}({path} deleted){RESET}")
            else:
                print(f"{BOLD}{C.SUCCESS}  → plan discarded{RESET} {DIM}(file already gone){RESET}")
        except OSError as e:
            print(f"{C.WARNING}  ⚠ released the gate but couldn't delete {path}: "
                  f"{type(e).__name__}: {e}{RESET}")
        return
    if not arg:
        # No arg: show the active plan (or the not-active hint).
        if not st._PLAN_ACTIVE:
            print(f"{DIM}  no plan active — start one with /plan <goal>{RESET}")
            return
        path = plan_path()
        print(f"{DIM}  ── plan ── {RESET}{C.LABEL}{path}{RESET}")
        for line in render_plan_display().splitlines():
            print(f"  {line}")
        print(f"{DIM}  ──────────{RESET}")
        print(f"{DIM}  writes/bash gated. /plan done to release.{RESET}")
        return
    # `/plan edit <goal>`: legacy alias — the same thing as /plan done followed
    # by /plan <goal> against the existing file, minus the user retyping it.
    reactivating = False
    if arg.startswith("edit "):
        arg = arg[len("edit "):].strip()
        reactivating = True
        if not arg:
            print(f"{DIM}  /plan edit <goal> — goal required{RESET}")
            return
    # Activate (or re-activate) with a goal.
    slug = slugify(arg)
    if st._PLAN_ACTIVE and st._PLAN_SLUG != slug:
        # A different goal while one is active: replace it (the old file is
        # kept on disk — the new one overwrites only if the slugs collide).
        print(f"{DIM}  replacing active plan {st._PLAN_SLUG!r}{RESET}")
    if st._PLAN_ACTIVE and st._PLAN_SLUG == slug and not reactivating:
        print(f"{DIM}  plan {slug!r} already active — /plan to view, "
              f"/plan done to release{RESET}")
        return
    path = os.path.join(plan_dir(), f"{slug}.md")
    goal = arg
    try:
        os.makedirs(plan_dir(), exist_ok=True)
        # A fresh goal writes a fresh file; a re-activation (or a same-slug
        # replace) keeps whatever body the model already edited into it.
        fresh = not (os.path.isfile(path) and (reactivating or
                                               (st._PLAN_ACTIVE and st._PLAN_SLUG == slug)))
        if fresh:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Plan: {goal}\n\n(working — edit freely with plan_edit)\n")
    except OSError as e:
        print(f"{C.ERROR}  ✗ can't write plan file: {type(e).__name__}: {e}{RESET}")
        return
    st._PLAN_ACTIVE = True
    st._PLAN_GOAL = goal
    st._PLAN_SLUG = slug
    st._PLAN_STARTED = time.time()
    if state is not None and getattr(state, "session_dirty", None) is not None:
        state.session_dirty = True
    if reactivating:
        print(f"{BOLD}{C.SUCCESS}  → plan re-activated{RESET} {DIM}({path}){RESET}")
    else:
        print(f"{BOLD}{C.SUCCESS}  → plan mode{RESET} {DIM}({path}){RESET}")
    print(f"{DIM}  writes/bash gated until /plan done. The model can update "
          f"the plan with plan_edit.{RESET}")
    # Inject the plan block into the live conversation (user-role, so it rides
    # the same "[Runtime note: ...]" channel as the background/multi-edit
    # nudges and lands in the transcript where the model will actually read it).
    messages = getattr(state, "messages", None)
    if messages is None:
        return
    block = (
        f"[Plan mode active — goal: {goal}\n"
        f"Plan file: {path}\n"
        "While the plan is active every mutating tool call (write_file, "
        "edit_file, multi_edit, sed_edit, apply_patch, delete_file, run_bash, "
        "memory_write/delete) is refused with a gate message. Read-only tools "
        "and plan_edit still work.\n"
        "Read the plan with read_file, refine it with plan_edit until it "
        "covers the whole task, then STOP and ask the user to run `/plan done` "
        "to release the gate. Do not attempt mutations again this turn.]"
    )
    messages.append({"role": "user", "content": block})
