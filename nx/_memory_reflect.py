"""`/memory reflect` — the cross-project connection pass.

`/memory optimize` is a *janitor*: it works inside the store, folding
duplicates and retiring stale notes. `/memory reflect` is the opposite
disposition: it reads the store from OUTSIDE — every note in both scopes,
global and all projects — and asks the model a question the janitor never
would: what do these notes know collectively that no single note says?

Concretely, a reflect run may:
  • PROMOTE a preference that keeps re-appearing inside project scopes to a
    single global `kind=preference` note (the "user's coding preferences"
    profile the session-start note always injects);
  • CONNECT notes across projects into one cross-cutting insight (the
    recurring pattern the user can't see because they've never held two
    projects' notes side by side);
  • PROPOSE a fix to the store's composition (a note that contradicts a
    newer one, a preference buried in an episodic note) — printed as advice,
    never executed, because re-typing other notes' footers is optimize's
    territory.

Like optimize this is a *slash command*, not a tool: zero schema tokens
(see the optimize docstring for the full argument), and it's the user, not
the mid-task model, who decides when a reflective pass is worth a
round-trip.

Flow, bounded at every step:

1. `_collect_reflection` indexes the store (both scopes, newest first, capped
   at NX_MEMORY_REFLECT_MAX_NOTES) and renders one compact index line per
   note (scope, name, age, kind, one-line digest). No model call yet.
2. One round-trip (`_reflect_chat`, the same retry/backoff shape as
   optimize's `_optimize_chat`) returns a line protocol —
   `NOTE <name>`, `WRITE <scope> <kind>` … — a deliberately smaller grammar
   than optimize's, because reflect's job is breadth, not surgery.
3. Each WRITE's body comes from its own round-trip with the source notes'
   real bodies; concurrent (NX_MEMORY_REFLECT_WORKERS), written through the
   one writer with gate=False (the user approved the run) and the kind tag
   from the plan line.
4. Proposals (lines 3's advice) are printed verbatim and dropped.

Nothing is archived: reflect only ever ADDS notes. Deduplication across
reflect runs is the model's job via the "supersede, don't duplicate"
contract memory_write already documents — and the run says so when it
writes nothing.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple

from ._ansi import DIM, RESET, C
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from . import _state as st
from ._memory_cmd import (
    MEMORY_KINDS, _first_title, _iter_memory_dirs, _iter_notes, _note_kind,
    _write_memory_file,
)

# Notes indexed per reflect run (newest first). 0 = no ceiling. The index
# line is ~60 chars, so even the default 120-note store is a ~7K-token prompt.
REFLECT_DEFAULT_MAX_NOTES = 120


def _reflect_env_int(name: str, default: int) -> int:
    """Read an NX_* integer through the top-level module (test seam — same
    shape as _memory_cmd._memory_search_limit)."""
    import nx as _m
    try:
        return max(0, _m._env_int(name, default))
    except Exception:
        return default


def _reflect_max_notes() -> int:
    return _reflect_env_int("NX_MEMORY_REFLECT_MAX_NOTES", REFLECT_DEFAULT_MAX_NOTES)


def _reflect_workers() -> int:
    return min(8, max(1, _reflect_env_int("NX_MEMORY_REFLECT_WORKERS", 2)))


def _reflect_timeout() -> int:
    return _reflect_env_int("NX_MEMORY_REFLECT_TIMEOUT", 600)


# --- index -------------------------------------------------------------------

_REFLECT_DIGEST_CHARS = 480


class _RefNote(NamedTuple):
    name: str
    scope: str            # "global" | "project"
    dir: str
    mtime: float
    title: str
    kind: str             # '' when untagged
    digest: str


def _digest(content: str, max_chars: int = _REFLECT_DIGEST_CHARS) -> str:
    """First `max_chars` of a note's body (H1 and footer dropped), one line."""
    out: list[str] = []
    total = 0
    for ln in content.splitlines():
        if ln.startswith("# "):
            continue
        if ln.startswith("_scope:"):
            break
        s = ln.strip()
        if not s:
            continue
        if total + len(s) > max_chars:
            out.append(s[:max_chars - total])
            break
        out.append(s)
        total += len(s) + 1
        if total >= max_chars:
            break
    return " / ".join(out)


def _collect_reflection(cap: int) -> list[_RefNote]:
    """Every live note in both scopes, newest first, capped. Global first
    (same iteration order every scanner uses); a name present in both scopes
    is kept once per scope — unlike optimize, reflect never acts by name, so
    no collision splitting is needed."""
    notes: list[_RefNote] = []
    for d, label in _iter_memory_dirs("project"):
        for name, path, mtime, title, content in _iter_notes(d):
            notes.append(_RefNote(name, label, d, mtime, title,
                                  _note_kind(content), _digest(content)))
    notes.sort(key=lambda n: -n.mtime)
    if cap and len(notes) > cap:
        notes = notes[:cap]
    return notes


def _index_lines(notes: list[_RefNote]) -> str:
    """The model's one-line-per-note view of the store."""
    today = time.strftime("%Y-%m-%d")
    lines = [f"(today: {today})"]
    for n in notes:
        age_days = max(0, int((time.time() - n.mtime) // 86400))
        kind = n.kind or "untagged"
        lines.append(f"- {n.scope} | {n.name} | {age_days}d old | {kind} | "
                     f"{n.title}: {n.digest}")
    return "\n".join(lines)


# --- model round-trip ----------------------------------------------------------

def _reflect_chat(prompt: str, purpose: str) -> str | None:
    """One non-streaming round-trip with retry-on-truncation, the same shape
    as _memory_optimize._optimize_chat but self-contained (reflect's calls
    are single-shot, so the shared helper's batch plumbing is unneeded)."""
    mt_kwargs: dict[str, Any] = {}
    base_kwargs: dict[str, Any] = {}
    if st.ACTIVE:
        base_kwargs = st.ACTIVE.extra_request_kwargs()
    _log_event("req", {"model": st.MODEL, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "_purpose": purpose})
    assert st.client is not None and st.MODEL is not None  # caller guards st.client/st.MODEL is None; narrows for the checker
    try:
        resp = st.client.chat.completions.create(
            model=st.MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False, timeout=_reflect_timeout(),
            **mt_kwargs, **base_kwargs)
    except APIConnectionError:
        print(f"{C.ERROR}  ✗ can't reach {st.client.base_url}{RESET}")
        return None
    except Exception as e:
        print(f"{C.ERROR}  ✗ reflect pass failed: {type(e).__name__}: {e}{RESET}")
        return None
    try:
        _log_event("resp", {"_purpose": purpose, "data": resp.model_dump()})
    except Exception:
        pass
    _record_usage(getattr(resp, "usage", None),
                  fallback_chars=len(resp.choices[0].message.content or ""),
                  fallback_prompt_chars=len(prompt))
    return (resp.choices[0].message.content or "").strip()


def _confirm_reflect(action: str) -> bool:
    """Same seam as _memory_cmd._confirm_memory (tests monkeypatch nx._confirm)."""
    import nx as _m
    return _m._confirm(action)


# --- plan protocol ---------------------------------------------------------------

# One WRITE per line: `WRITE <scope> <kind>` — scope global|project, kind one
# of MEMORY_KINDS. The body arrives in the follow-up pass. NOTE lines carry
# source names; anything else is surfaced as advice. The tail after the kind
# is not anchored: models habitually append prose/title to the line, and the
# salvage keeps the write instead of demoting it to advice (which would make
# the run report "no cross-cutting signal" despite visible WRITE lines).
_WRITE_RE = re.compile(r"^WRITE\s+(global|project)\s+(\S+)", re.IGNORECASE)


class _RefPlan(NamedTuple):
    writes: list[tuple[str, str]]      # (scope, kind)
    sources: list[str]                 # NOTE'd names, in order
    advice: list[str]                  # everything else, printed verbatim


def _parse_reflection(text: str, known: dict[str, _RefNote]) -> _RefPlan:
    writes: list[tuple[str, str]] = []
    sources: list[str] = []
    advice: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _WRITE_RE.match(line)
        if m:
            kind = m.group(2).lower()
            if kind in MEMORY_KINDS:
                writes.append((m.group(1), kind))
                continue
        if line.upper().startswith("NOTE "):
            name = line[5:].strip()
            if name in known:
                sources.append(name)
                continue
        if line.startswith(("PROPOSE", "ADVICE")):
            advice.append(line)
            continue
        # Not protocol — but a bare `NOTE:`-ish or unmarked observation still
        # has value to the human reading the run.
        advice.append(line)
    return _RefPlan(writes, sources, advice)


# --- prompts -------------------------------------------------------------------

def _reflection_prompt(notes: list[_RefNote], focus: str) -> str:
    focus = f"\nThe user specifically asks to reflect on: {focus}\n" if focus else ""
    return (
        "You are reviewing the saved memory notes of a developer's AI coding "
        "assistant. The notes span two scopes: `global` (cross-project) and "
        "`project` (one specific repo). Find what they know COLLECTIVELY that "
        "no single note says:\n"
        "- Recurring user preferences or working style that appear only inside "
        "project-scoped notes → WRITE one global preference note that states "
        "the durable preference itself (not the per-project incidents).\n"
        "- A pattern, tool, or gotcha shared across projects that the user may "
        "not have connected → WRITE one global note stating the cross-project "
        "insight.\n"
        "- Contradictions, stale beliefs, or a preference buried in an "
        "episodic note → report as a PROPOSE line (advice only; the reflect "
        "pass never edits other notes).\n"
        "Rules:\n"
        "- Write NOTHING unless the store genuinely supports it. An empty "
        "store, or one with no cross-cutting signal, returns no WRITE lines.\n"
        "- At most 3 WRITE lines. Supersede, don't duplicate: assume notes on "
        "the same topic may already exist.\n"
        "- NOTE lines: list the source note names your WRITE lines draw on.\n"
        "Reply in this exact line protocol and nothing else:\n"
        "NOTE <name>\n"
        "WRITE <scope> <kind>        (scope: global|project; kind: "
        "preference|fact|gotcha|episodic)\n"
        "PROPOSE <one-line observation>\n"
        f"{focus}\n"
        "---\n"
        f"{_index_lines(notes)}\n---"
    )


def _write_body_prompt(scope: str, kind: str, sources: list[_RefNote]) -> str:
    src = "\n\n".join(
        f"### {n.title} [{n.scope}, {n.kind or 'untagged'}]\n{n.digest}"
        for n in sources) or "(no specific sources — write from the pattern above)"
    return (
        f"Write ONE concise markdown memory note: a durable `{kind}` that "
        f"generalizes across the user's projects. Draw ONLY on the material "
        "below (plus the general pattern it evidences); use backticks for "
        "code, paths, commands. State the durable rule/insight and when it "
        "applies — not the per-project incidents. "
        "Start with a `# Title` line.\n\n"
        f"---\n{src}\n---"
    )


# --- command -----------------------------------------------------------------

def _cmd_memory_reflect(arg: str = "") -> None:
    """`/memory reflect [focus…]` — the cross-project connection pass."""
    focus = (arg or "").strip()
    if st.client is None or st.MODEL is None:
        print(f"{C.ERROR}  ✗ no active source — reflect needs a model{RESET}")
        return
    cap = _reflect_max_notes()
    notes = _collect_reflection(cap)
    if len(notes) < 2:
        print(f"{DIM}  nothing to reflect on ({len(notes)} note"
              f"{'s' if len(notes) != 1 else ''} saved — reflect needs at least 2){RESET}")
        return
    print(f"{DIM}  reflecting over {len(notes)} memories…{RESET}")
    text = _reflect_chat(_reflection_prompt(notes, focus), "memory_reflect_plan")
    if not text:
        return
    known = {n.name: n for n in notes}
    plan = _parse_reflection(text, known)
    for ln in plan.advice:
        print(f"{DIM}  · {ln[:160]}{RESET}")
    if not plan.writes:
        print(f"{DIM}  no cross-cutting signal found ({len(notes)} notes reviewed) "
              f"— nothing written{RESET}")
        return
    writes = plan.writes[:3]
    if len(plan.writes) > 3:
        print(f"{DIM}  {len(plan.writes) - 3} WRITE line(s) past the 3-write cap dropped{RESET}")

    # Source assignment: every write draws on the NOTE'd sources (they are
    # deliberately not per-write — the pass reads the store as one corpus).
    srcs = [known[n] for n in plan.sources if n in known]
    action = (f"reflect: write {len(writes)} note(s) — "
              + ", ".join(f"{s}/{k}" for s, k in writes))
    if not _confirm_reflect(action):
        print(f"{DIM}  cancelled{RESET}")
        return

    def one(item: tuple[str, str]) -> str:
        scope, kind = item
        body = _reflect_chat(_write_body_prompt(scope, kind, srcs),
                             f"memory_reflect_write_{kind}")
        if not body:
            return f"  ✗ {kind} ({scope}): model returned nothing"
        title = _first_title(body, "") or kind
        try:
            # `/memory reflect` generalizes over digests of EXISTING notes
            # (_write_body_prompt's sources are _RefNote digests), so its output
            # is memory-derived by construction — stamp the provenance.
            written = _write_memory_file(title, body, scope, gate=False,
                                         kind=kind, derived="memory")
        except OSError as e:
            return f"  ✗ {kind} ({scope}): {type(e).__name__}: {e}"
        if written is None:
            return f"  ✗ {kind} ({scope}): write failed"
        return f"  ✓ wrote {scope} {kind}: {title}\n    {written[0]}"

    results: list[str] = []
    workers = min(_reflect_workers(), len(writes))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, w) for w in writes]
        for fut in as_completed(futures):
            results.append(fut.result())
    for r in sorted(results):
        print(f"{DIM}{r}{RESET}")
    print(f"{DIM}  reflect pass done — the session's memory note refreshes on the "
          f"next turn (or /memory optimize to fold overlaps){RESET}")
