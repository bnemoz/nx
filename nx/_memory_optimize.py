"""`/memory optimize` — consolidate and retire saved memories.

A memory store only grows: `/memory save` and `memory_write` add notes, and
nothing ever notices that a note written in week 1 describes a file the
project deleted in week 8, or duplicates a note written in week 9. After a
hundred notes the session-start title list and every `memory_search` are
mostly noise, and the model spends turns recalling things that are simply
wrong now.

This is the janitor pass. It is a *slash command* rather than a tool on
purpose: the model already has `memory_write`/`memory_delete` for
one-at-a-time hygiene mid-task, and a schema line here would cost every turn
in the repo forever (see CLAUDE.md's token floor rule).

Flow, bounded at every step:

1. `_collect_notes` indexes the store (global + project, newest first) — no
   model call, capped at NX_MEMORY_OPTIMIZE_MAX_NOTES so a 300-note store
   can't turn one command into 300 notes of prompt.
2. One librarian round-trip per NX_MEMORY_OPTIMIZE_BATCH notes sends a
   compact index (name, age, size, one-line digest) and asks for plan lines
   only — `RETIRE <name>` / `MERGE a + b => Title`. A line protocol, not JSON:
   small models quote markdown bodies badly, and unparseable plans are
   discarded line-by-line instead of losing the whole pass. From the second
   batch on, the previous batch's unactioned titles ride along, so a
   near-duplicate pair straddling a batch boundary can still meet.
3. Each merge group gets its own round-trip with the real bodies (source
   `_scope:`/`_saved:` footers stripped first — the writer re-adds them),
   returning one markdown note — written through `_write_memory_file`, so merged notes
   are byte-for-byte conventional (`<timestamp>_<slug>.md`, 0600, `_scope:`
   footer) and never clobber anything.
4. Source notes are never unlinked: they move to
   `<memories_dir>/archive/<run-stamp>/` inside their own scope dir. A bad
   librarian answer is therefore reversible by `mv`, and the scanner can't
   see the archive (it lists `.md` *files*, and `archive` has no suffix).
5. The live transcript's `[Saved memories …]` note is refreshed, so the
   model's view of the store matches the store (`_refresh_memory_note`).

Merges may span scopes; the merged note lands in the scope holding the
majority of its sources (ties → project), and every source is archived.
`/memory optimize dry` runs 1-2 and prints the plan without writing.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, NamedTuple

from ._ansi import DIM, RESET, C
from ._logging import _log_event
from ._metrics import _record_usage
from ._openai_client import APIConnectionError
from . import _state as st
from ._memory_cmd import (
    _first_title, _iter_memory_dirs, _read_memory_file, _refresh_memory_note,
    _write_memory_file,
)

# Notes reviewed per librarian call. Small enough that the index stays a
# fraction of the context window even on a 4K model; large enough that a
# 100-note store is a handful of calls, not a hundred.
OPTIMIZE_DEFAULT_BATCH = 12

# Hard ceiling on how many notes one run reviews (newest first; the rest are
# reported as unexamined rather than silently skipped). 0 = no ceiling.
OPTIMIZE_DEFAULT_MAX_NOTES = 120

# Per-note digest size in the index (the librarian judges on titles + gist;
# it sees full bodies only for the groups it is asked to merge).
_NOTE_DIGEST_CHARS = 480

# Merges cost one model call each, so the whole run is capped independently
# of how many RETIRE lines come back. A safety limit, not a tunable. Viable
# groups past it are reported (in _Plan.overflow) rather than silently dropped.
_MAX_MERGES_PER_RUN = 8

# Timeout for the librarian/merge round-trips (heavier prompts than /memory
# save's single distillation call).
_OPTIMIZE_TIMEOUT = 60

# Completion budget for the librarian/merge round-trips. Reasoning models burn
# this on `reasoning_content` before writing a visible character, and many
# servers cap at 4096 when the request doesn't set it — the answer then arrives
# as content=None with finish_reason="length". Sized so a long merge finishes;
# a server that refuses the number is retried without the kwarg.
DEFAULT_OPTIMIZE_MAX_TOKENS = 36 * 1024

_DRY_WORDS = ("dry", "dry-run", "--dry", "--dry-run")


def _opt_env_int(name: str, default: int) -> int:
    """Read an NX_* integer through the top-level module (same seam
    `_memory_search_limit` uses, so tests can just set the env var)."""
    import nx as _m
    try:
        return max(0, _m._env_int(name, default))
    except Exception:
        return default


def _optimize_batch(total: int) -> int:
    """Notes per librarian pass (NX_MEMORY_OPTIMIZE_BATCH; 0 or oversized =
    one pass over everything). Takes the run's note count so the env read
    stays a pure function of the environment (no module state)."""
    batch = _opt_env_int("NX_MEMORY_OPTIMIZE_BATCH", OPTIMIZE_DEFAULT_BATCH)
    return batch if 0 < batch < total else max(1, total)


def _optimize_max_notes() -> int:
    """Notes reviewed per run (NX_MEMORY_OPTIMIZE_MAX_NOTES; 0 = all)."""
    return _opt_env_int("NX_MEMORY_OPTIMIZE_MAX_NOTES", OPTIMIZE_DEFAULT_MAX_NOTES)


class _OptNote(NamedTuple):
    name: str
    title: str
    label: str        # "global" | "project"
    dir: str          # the memories dir holding it
    mtime: float
    chars: int
    digest: str


class _Plan(NamedTuple):
    retire: list[str]              # note names
    merges: list[tuple[list[str], str]]   # (names, proposed title)
    overflow: int = 0              # viable merge groups past the run's cap


# --- index -------------------------------------------------------------------

def _digest(content: str, max_chars: int = _NOTE_DIGEST_CHARS) -> str:
    """First `max_chars` of a note's body (H1 and the `_scope:` footer
    dropped), collapsed to one line — the librarian's view of a note it isn't
    being asked to merge."""
    out: list[str] = []
    total = 0
    for ln in content.splitlines():
        s = ln.strip()
        if not s or s.startswith("# ") or s.startswith("_scope:") or s == "---":
            continue
        out.append(s)
        total += len(s)
        if total >= max_chars:
            break
    text = " / ".join(out)
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def _collect_notes(scope: str = "project") -> list[_OptNote]:
    """Every saved note, newest first, with the fields a librarian pass needs.
    The `archive/` subdirs are invisible here by construction (a directory
    never ends in `.md`), so an optimize run never re-ingests its own backups."""
    notes: list[_OptNote] = []
    for d, label in _iter_memory_dirs(scope):
        try:
            names = sorted(os.listdir(d), reverse=True)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(d, name)
            read = _read_memory_file(path, name)
            if read is None:
                continue
            title, content = read
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            notes.append(_OptNote(name, title, label, d, mtime, len(content), _digest(content)))
    notes.sort(key=lambda n: -n.mtime)
    return notes


def _age(mtime: float) -> str:
    """Human age for the index: '3d', '5h', '12m' — staleness is the signal
    the librarian is being asked to weigh, so it has to be in front of it."""
    secs = max(0.0, time.time() - mtime) if mtime else 0.0
    days = secs / 86400.0
    if days >= 1:
        return f"{int(days)}d"
    hours = secs / 3600.0
    if hours >= 1:
        return f"{int(hours)}h"
    return f"{int(secs / 60)}m"


def _index_block(notes: list[_OptNote]) -> str:
    out = []
    for n in notes:
        out.append(f"[{n.label}] {n.name}  ({_age(n.mtime)} old, {n.chars} chars)")
        out.append(f"  title: {n.title}")
        if n.digest:
            out.append(f"  gist: {n.digest}")
    return "\n".join(out)


# --- prompts -----------------------------------------------------------------

def _plan_prompt(notes: list[_OptNote], focus: str,
                 carry: list[_OptNote] | None = None) -> str:
    guidance = f"\nThe owner adds this guidance: {focus}\n" if focus else ""
    carry_block = ""
    if carry:
        lines = "\n".join(f"[{n.label}] {n.name}  title: {n.title}" for n in carry)
        carry_block = (
            "\n--- carryover from the previous batch (titles only) ---\n"
            "These notes were reviewed last pass and left unactioned. You may RETIRE "
            "or MERGE them together with this batch's notes when they overlap; "
            f"otherwise ignore them.\n{lines}\n"
        )
    return (
        "You are the librarian of a coding agent's saved memory notes. Each note is one "
        "markdown file distilled from a past session: learnings, file paths, commands, "
        "decisions and the reasons behind them. The project has matured, so some notes are "
        "now stale, duplicated or superseded.\n\n"
        "Reply with ONLY plan lines, one per line, no other text:\n"
        "  RETIRE <name>\n"
        "  MERGE <name1> + <name2> [ + <name3> ... ] => <short title for the merged note>\n"
        "  OK\n"
        "RETIRE a note when it is a strict subset of a newer note, when newer notes describe "
        "the same thing differently (the newer account wins), or when it records only session "
        "logistics with no durable fact. MERGE near-duplicate notes about one topic; never "
        "merge unrelated notes. Use only names from the index. Reply OK if nothing should "
        f"change.{guidance}{carry_block}\n--- index ---\n{_index_block(notes)}\n--- end ---"
    )


_FOOTER_RE = re.compile(r"\n---\n_scope:.*$", re.DOTALL)


def _strip_footer(body: str) -> str:
    """Drop the writer's trailing scope/saved bookkeeping block from a note
    body. The merge model must not re-emit it (and often would, verbatim);
    `_write_memory_file` appends a fresh footer for the merged note anyway.
    Anything that does not end in that shape is returned untouched."""
    return _FOOTER_RE.sub("", body).rstrip() + "\n"


def _merge_prompt(group: list[_OptNote]) -> str:
    parts = []
    for n in group:
        try:
            with open(os.path.join(n.dir, n.name), encoding="utf-8") as f:
                body = _strip_footer(f.read())
        except (OSError, UnicodeDecodeError):
            body = f"(unreadable — title only: {n.title})"
        parts.append(f"--- {n.name} ({_age(n.mtime)} old, scope: {n.label}) ---\n{body}")
    return (
        "Merge the memory notes below into ONE note. Keep every durable fact, file path, "
        "command, config value and reason; drop duplicated text; where two notes disagree, "
        "keep the newer note's version and say the older claim was superseded. "
        "Start with a single `# Title` line, then markdown. No preamble and no commentary.\n\n"
        + "\n\n".join(parts)
    )


# --- model round-trips -------------------------------------------------------

def _optimize_max_tokens() -> int:
    """Completion budget for the librarian/merge round-trips
    (NX_MEMORY_OPTIMIZE_MAX_TOKENS; 0 = don't send max_tokens at all, letting
    the server default apply)."""
    return _opt_env_int("NX_MEMORY_OPTIMIZE_MAX_TOKENS", DEFAULT_OPTIMIZE_MAX_TOKENS)


def _bad_request(e: Exception) -> bool:
    """Is this an APIError whose server response was a 4xx (as opposed to a
    transport failure)? Used to decide whether a retry without max_tokens can
    help — retrying on 5xx/timeout would just repeat the failure."""
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500


def _empty_reason(choice: Any) -> str:
    """Human-readable reason an HTTP-200 response carries no visible text.
    The interesting case is finish_reason="length" after real thinking: a
    reasoning model that burned the whole completion budget on
    `reasoning_content` and was cut off before the answer."""
    try:
        fr = getattr(choice, "finish_reason", None)
        msg = getattr(choice, "message", None)
        if getattr(msg, "reasoning_content", None) and fr == "length":
            return ("the model ran out of completion tokens while reasoning "
                    "(finish_reason=length) — raise NX_MEMORY_OPTIMIZE_MAX_TOKENS "
                    "or disable thinking on this source")
        if fr == "length":
            return "the answer was cut off by the token cap (finish_reason=length)"
        if getattr(msg, "reasoning_content", None):
            return "the model reasoned but produced no answer text"
    except Exception:
        pass
    return "the model returned no content"


def _optimize_chat(prompt: str, purpose: str) -> str | None:
    """One non-streaming librarian/merge call. Mirrors `/memory save`'s call
    shape (same logging purposes, same usage recording, same error surface)
    and returns None after printing why, so callers just stop.

    Sends max_tokens (NX_MEMORY_OPTIMIZE_MAX_TOKENS, default 36K) so a
    reasoning model can't silently spend a server's small default cap on
    thinking; a server that rejects the kwarg with a 4xx is retried once
    without it, and an HTTP-200 answer with no visible text prints why."""
    if st.client is None or st.MODEL is None:
        print(f"{C.ERROR}  ✗ no active source — memory optimize aborted{RESET}")
        return None
    max_tokens = _optimize_max_tokens()
    mt_kwargs = {} if max_tokens == 0 else {"max_tokens": max_tokens}
    base_kwargs = dict(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {})
    choice: Any = None
    try:
        _log_event("req", {"model": st.MODEL, "messages": [{"role": "user", "content": prompt}],
                           "stream": False, "_purpose": purpose,
                           "max_tokens": max_tokens})
        try:
            resp = st.client.chat.completions.create(
                model=st.MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False, timeout=_OPTIMIZE_TIMEOUT,
                **mt_kwargs, **base_kwargs)
        except Exception as first_e:
            # Some backends reject unknown/unusual completion params; the
            # kwarg is nx's idea, so one no-kwarg retry on a 4xx is cheap
            # and keeps those sources working.
            if not mt_kwargs or not _bad_request(first_e):
                raise
            _log_event("req", {"model": st.MODEL,
                               "messages": [{"role": "user", "content": prompt}],
                               "stream": False, "_purpose": purpose,
                               "max_tokens": None, "retry": "without max_tokens"})
            resp = st.client.chat.completions.create(
                model=st.MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False, timeout=_OPTIMIZE_TIMEOUT, **base_kwargs)
        try:
            _log_event("resp", {"_purpose": purpose, "data": resp.model_dump()})
        except Exception:
            pass
        try:
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
        except (AttributeError, IndexError):
            text = ""
        _record_usage(
            getattr(resp, "usage", None),
            fallback_chars=len(text),
            fallback_prompt_chars=len(prompt),
        )
        if not text:
            print(f"{C.WARNING}  ⚠ empty answer — {_empty_reason(choice)}{RESET}")
            return None
    except APIConnectionError:
        print(f"{C.ERROR}  ✗ can't reach {st.client.base_url} — memory optimize stopped{RESET}")
        return None
    except Exception as e:
        print(f"{C.ERROR}  ✗ memory optimize failed: {type(e).__name__}: {e}{RESET}")
        return None
    return text


# --- plan parsing ------------------------------------------------------------

_RETIRE_RE = re.compile(r"^RETIRE\s+(\S+)\s*$", re.IGNORECASE)
_MERGE_RE = re.compile(r"^MERGE\s+(.+?)\s*=>\s*(\S.*)$", re.IGNORECASE)


def _resolve_name(raw: str, known: dict[str, _OptNote]) -> str | None:
    """A plan name → the actual filename. Exact, or with `.md` added, or a
    unique stem prefix (models drop extensions and occasionally a slug tail);
    an ambiguous prefix is refused rather than guessed."""
    raw = raw.strip().strip("`*'\"")
    if not raw:
        return None
    if raw in known:
        return raw
    if f"{raw}.md" in known:
        return f"{raw}.md"
    cands = [k for k in known if k.lower().startswith(raw.lower())]
    return cands[0] if len(cands) == 1 else None


def _plan_shaped(ln: str) -> bool:
    """Does this line *attempt* to be a plan line? Only those get reported
    when they fail to parse; a model that adds a sentence of prose or a
    ``` fence shouldn't produce a wall of "ignored" noise."""
    up = ln.upper()
    return up.startswith(("RETIRE", "MERGE", "OK")) or "=>" in ln


def _parse_plan(text: str, known: dict[str, _OptNote]) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Plan text → (retire names, [(names, title)], rejected plan lines).

    Line-oriented and forgiving: unrelated prose, bullets and fences pass
    unnoticed, while a line that clearly meant to be an instruction but
    doesn't work (bad syntax, unknown or ambiguous name, a MERGE without two
    resolvable members) is reported per line instead of poisoning the pass."""
    retire: list[str] = []
    merges: list[tuple[list[str], str]] = []
    skipped: list[str] = []
    for raw_ln in (text or "").splitlines():
        ln = raw_ln.strip().lstrip("-*• `").strip()
        if not ln:
            continue
        if ln.upper() == "OK":
            continue
        if not _plan_shaped(ln):
            continue
        m = _RETIRE_RE.match(ln)
        if m:
            name = _resolve_name(m.group(1), known)
            if name is None:
                skipped.append(raw_ln.strip())
            else:
                retire.append(name)
            continue
        if ln.upper().startswith("MERGE"):
            m = _MERGE_RE.match(ln)
            if m is None:
                skipped.append(raw_ln.strip())
                continue
            names: list[str] = []
            bad = False
            for part in m.group(1).split("+"):
                name = _resolve_name(part, known)
                if name is None or name in names:
                    bad = True
                    break
                names.append(name)
            if bad or len(names) < 2:
                skipped.append(raw_ln.strip())
                continue
            merges.append((names, m.group(2).strip().strip('"')))
            continue
        skipped.append(raw_ln.strip())
    return retire, merges, skipped


def _merge_plan(total: _Plan, chunk: tuple[list[str], list[tuple[str, str]]],
                budget: int) -> _Plan:
    """Fold one pass's plan into the run's totals, keeping every name in
    exactly one action (a merge wins over a retire of the same note) and
    honouring the run-wide merge budget. Viable groups past the budget are
    counted in overflow rather than silently dropped."""
    retire = list(total.retire)
    merges = list(total.merges)
    overflow = total.overflow
    used = {n for names, _t in merges for n in names}
    for name in chunk[0]:
        if name not in used and name not in retire:
            retire.append(name)
    for names, title in chunk[1]:
        picked = [n for n in names if n not in used]
        if len(picked) < 2:
            continue
        if len(set(picked)) != len(picked):
            continue
        if len(merges) >= budget:
            overflow += 1
            continue
        merges.append((picked, title))
        used.update(picked)
        for n in picked:
            if n in retire:
                retire.remove(n)
    return _Plan(retire, merges, overflow)


# --- applying ----------------------------------------------------------------

def _archive(note: _OptNote, stamp: str) -> str | None:
    """Move one note into `<its dir>/archive/<stamp>/` (recoverable, and
    invisible to the scanners). Returns the new path, or None if the move
    failed — in which case the note is simply left in place."""
    dst_dir = os.path.join(note.dir, "archive", stamp)
    try:
        os.makedirs(dst_dir, exist_ok=True)
        try:
            os.chmod(dst_dir, 0o700)
        except OSError:
            pass
        dst = os.path.join(dst_dir, note.name)
        os.replace(os.path.join(note.dir, note.name), dst)
        return dst
    except OSError as e:
        print(f"{C.WARNING}  ⚠ couldn't archive {note.name}: {e}{RESET}")
        return None


def _group_scope(group: list[_OptNote]) -> str:
    """Where a merged note belongs: the scope holding most of its sources,
    project winning ties (with no project root `_write_memory_file` already
    falls back to global on its own)."""
    project = sum(1 for n in group if n.label == "project")
    return "project" if project * 2 >= len(group) else "global"


def _confirm_optimize(action: str) -> bool:
    """`_confirm` is monkeypatched by tests through the top-level module —
    same indirection as `_memory_cmd._confirm_memory`."""
    import nx as _m
    try:
        return bool(_m._confirm(action))
    except _m._EscToChat:
        print(f"{C.WARNING}  ↳ escaped memory optimize approval — back to chat input{RESET}")
        return False


def _drop_memory_caches() -> None:
    """A rewritten store invalidates any cached memory_search index (the same
    drop run_tool() does after memory_write/memory_delete)."""
    for k in [k for k in st._TOOL_RESULT_CACHE if k and k[0] == "memory_search"]:
        del st._TOOL_RESULT_CACHE[k]


def _print_plan(plan: _Plan, known: dict[str, _OptNote]) -> None:
    for names, title in plan.merges:
        src = " + ".join(known[n].title for n in names if n in known)
        print(f"  {C.SUCCESS}merge{RESET} {src} {DIM}→{RESET} {title}")
    for name in plan.retire:
        note = known.get(name)
        print(f"  {C.WARNING}retire{RESET} {note.title if note else name} "
              f"{DIM}({name}){RESET}")


def _apply_plan(plan: _Plan, known: dict[str, _OptNote],
                messages: list[dict[str, Any]] | None) -> tuple[int, int, str | None]:
    """Write merged notes, archive every retired/merged source, refresh the
    live memory note and the search cache. Returns
    (merged_notes, archived_notes, archive_dir_hint)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive_dir: str | None = None
    merged = 0
    archived = 0

    for names, prop_title in plan.merges:
        group = [known[n] for n in names if n in known]
        if len(group) < 2:
            continue
        text = _optimize_chat(_merge_prompt(group), "memory_optimize_merge")
        if text is None:
            continue
        if not text:
            print(f"{C.WARNING}  ⚠ empty merge answer — left {len(group)} notes alone{RESET}")
            continue
        title = _first_title(text, prop_title) or prop_title
        try:
            written = _write_memory_file(title, text, _group_scope(group))
        except OSError as e:
            print(f"{C.ERROR}  ✗ couldn't write merged note: {e}{RESET}")
            continue
        if written is None:
            continue
        merged += 1
        for note in group:
            dst = _archive(note, stamp)
            if dst is not None:
                archived += 1
                archive_dir = os.path.dirname(dst)

    for name in plan.retire:
        note = known.get(name)
        if note is None:
            continue
        dst = _archive(note, stamp)
        if dst is not None:
            archived += 1
            archive_dir = os.path.dirname(dst)

    if archived or merged:
        _drop_memory_caches()
        if messages is not None:
            _refresh_memory_note(messages)
    return merged, archived, archive_dir


# --- command -----------------------------------------------------------------

def _parse_opts(arg: str) -> tuple[bool, str]:
    """`/memory optimize [dry] [focus…]` → (dry_run, focus)."""
    words = (arg or "").split()
    if words and words[0].lower() in _DRY_WORDS:
        return True, " ".join(words[1:])
    return False, " ".join(words)


def _split_collisions(notes: list[_OptNote]) -> tuple[list[_OptNote], list[str]]:
    """(reviewable notes, colliding names). A name present in both scopes is
    unaddressable — a plan line naming it must not silently pick one scope's
    file to archive — so those notes sit this run out instead, with the run
    saying so. The writer's second-resolution timestamped filenames make a
    collision impossible unless a note was hand-copied across scopes."""
    by_name: dict[str, _OptNote] = {}
    colliding: set[str] = set()
    for n in notes:
        if n.name in by_name or n.name in colliding:
            colliding.add(n.name)
        else:
            by_name[n.name] = n
    if colliding:
        for name in colliding:
            del by_name[name]
    return list(by_name.values()), sorted(colliding)


def _cmd_memory_optimize(messages: list[dict[str, Any]] | None = None,
                         arg: str = "") -> None:
    """`/memory optimize [dry] [focus…]` — librarian pass over the store.

    Reviews the newest NX_MEMORY_OPTIMIZE_MAX_NOTES notes in batches of
    NX_MEMORY_OPTIMIZE_BATCH, prints the resulting plan (merge groups +
    retirements), asks once, then writes merged notes through the normal
    writer and moves every retired note into an `archive/` subdir of its own
    scope. `dry` prints the plan and changes nothing; `focus…` is guidance
    for the librarian (\"focus on the build system notes\").
    """
    dry, focus = _parse_opts(arg)
    notes = _collect_notes("project")
    total_before = len(notes)
    if total_before < 2:
        print(f"{DIM}  nothing to optimize ({len(notes)} memor"
              f"{'y' if len(notes) == 1 else 'ies'} saved){RESET}")
        return
    notes, colliding = _split_collisions(notes)
    if colliding:
        print(f"{C.WARNING}  ⚠ {len(colliding)} note name(s) exist in both scopes — "
              f"left untouched this run (rename or delete one to review it){RESET}")
    if len(notes) < 2:
        print(f"{DIM}  nothing else to optimize{RESET}")
        return

    cap = _optimize_max_notes()
    unexamined = 0
    if cap and len(notes) > cap:
        unexamined = len(notes) - cap
        notes = notes[:cap]
    batch = _optimize_batch(len(notes))
    known = {n.name: n for n in notes}

    print(f"{DIM}  reviewing {len(notes)} memories"
          f"{' (dry run)' if dry else ''}…{RESET}")
    plan = _Plan([], [])
    carry: list[_OptNote] = []
    for start in range(0, len(notes), batch):
        chunk = notes[start:start + batch]
        if len(notes) > batch:
            print(f"{DIM}  · librarian pass {start // batch + 1}"
                  f"/{-(-len(notes) // batch)}{RESET}")
        text = _optimize_chat(_plan_prompt(chunk, focus, carry), "memory_optimize_plan")
        if text is None:
            if not (plan.retire or plan.merges):
                return
            print(f"{C.WARNING}  ⚠ partial run — applying what earlier passes found{RESET}")
            break
        retire, merges, skipped = _parse_plan(text, known)
        plan = _merge_plan(plan, (retire, merges), _MAX_MERGES_PER_RUN)
        for ln in skipped:
            print(f"{DIM}  · ignored plan line: {ln[:80]}{RESET}")
        # The next pass sees this chunk's unactioned titles, so a pair split
        # across a batch boundary can still meet.
        actioned = set(plan.retire) | {n for names, _t in plan.merges for n in names}
        carry = [n for n in chunk if n.name not in actioned]

    if not plan.retire and not plan.merges:
        print(f"{DIM}  librarian found nothing to change ({len(notes)} notes reviewed){RESET}")
        return
    if unexamined:
        print(f"{DIM}  {unexamined} older notes not reviewed "
              f"(raise NX_MEMORY_OPTIMIZE_MAX_NOTES){RESET}")

    print(f"{DIM}  plan:{RESET}")
    _print_plan(plan, known)
    if plan.overflow:
        print(f"{DIM}  {plan.overflow} more merge group"
              f"{'s' if plan.overflow != 1 else ''} found past this run's merge cap "
              f"— run /memory optimize again{RESET}")
    if dry:
        print(f"{DIM}  dry run — nothing changed{RESET}")
        return

    before = total_before
    action = (f"optimize memories (write {len(plan.merges)} merged, "
              f"archive {len(plan.retire) + sum(len(n) for n, _t in plan.merges)})")
    if not _confirm_optimize(action):
        print(f"{DIM}  cancelled{RESET}")
        return

    merged, archived, archive_dir = _apply_plan(plan, known, messages)
    after = len(_collect_notes("project"))
    print(f"{DIM}  ✓ {before} → {after} memories "
          f"({merged} merged note{'s' if merged != 1 else ''}, {archived} archived){RESET}")
    if archive_dir:
        print(f"{DIM}    originals kept in {archive_dir}{RESET}")
    if merged or archived:
        print(f"{DIM}    the session's memory note was refreshed to match{RESET}")
