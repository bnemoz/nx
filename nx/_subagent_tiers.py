"""Sub-agent tool tiers: the read-only toolset, the write tier (NX_SUBAGENT_
WRITE=1), and the per-thread restricted dispatch tables run_tool() resolves
through.

Split out of `_subagent.py` (which keeps the spawn machinery). A sub-agent is
run through the same `_run_model_turn_loop` machinery as the parent but with a
RESTRICTED dispatch table: read-only by default; the write tier binds the four
file-mutation tools to confirm-through wrappers that tear down the sub-agent
context around the parent's real body, so every write crosses the same
risk-classified Y/n gate the parent shows."""
from __future__ import annotations

import fnmatch
import os
from typing import Any

from . import _state as st
from ._env import _env_int
from ._subagent_tls import _SUBAGENT_TLS, _subagent_ctx  # noqa: F401 (re-exports)
# Tools the sub-agent may call: the read-only dispatch entries (derived from
# the same tuple run_tool() result-caches) + run_bash, which is gated to the
# read-only allowlist by _subagent_bash(). Never mutating tools, never
# wait_background. list_dir is retired from the offered floor (0.129.8) — a
# sub-agent lists directories via its allowlisted bash `ls`.
_READ_ONLY_TOOL_NAMES = (
    "read_file", "search_files", "git_status", "git_diff", "git_log",
    "memory_search", "recall_lessons", "run_bash",
)

# Tools offered only when the write tier is on (NX_SUBAGENT_WRITE=1): the
# schemas come from the parent TOOLS list, but inside a tier-enabled sub-loop
# they resolve to the confirm-through wrappers in _write_tier_dispatch().
# delete_file is deliberately excluded (removal, not creation — a delete is
# one edit away from a recreate, but its undo snapshot belongs to the parent);
# so are apply_patch (one call already covers multi-file restructures),
# memory_write, and run_bash (the read-only allowlist tier covers inspection).
# run_tests IS included (0.49.0): a write-tier agent that can edit but not
# verify can only guess. It runs through the same user-gated _confirm() the
# parent's run_tests hits (the wrapper tears down the sub-agent context), so
# it is exactly as approved as a parent run — just tagged (sub-agent). It
# stays refused for READ-ONLY agents: their table never contains it.
# git_commit IS included (0.51.0): an agent that edits but cannot land its
# work strands its changes. Same gate: the parent's _confirm shows the staged
# diff (and the secret scan runs before it) exactly as a parent call would.
_WRITE_TIER_TOOL_NAMES = ("write_file", "edit_file", "multi_edit", "sed_edit",
                          "run_tests", "git_commit")

# Per-agent sub-loop context. A global flag/table can't express two agents
# running at once (the second's finally would clear the first's table), so
# spawn_agent() installs its restricted table on the shared thread-local slot
# from _subagent_tls and run_tool() resolves tool calls through it. Falls back
# to the plain globals for code (tests, REPL) that sets them directly.


class _SubagentContext:
    """One running sub-agent's restricted-dispatch state (thread-local)."""

    __slots__ = ("table", "active", "write", "quiet", "model_target", "scope",
                 "worktree", "job_id", "consults_left", "advisor", "tier",
                 "write_calls", "churn_nudged", "churn_escalated")

    def __init__(self, table: dict[str, Any], write: bool = False,
                 quiet: bool = False, scope: "list[str] | None" = None,
                 worktree: "str | None" = None,
                 job_id: "str | None" = None) -> None:
        self.table = table
        self.active = True
        self.write = write  # write tier (NX_SUBAGENT_WRITE=1): write tools resolve
        self.quiet = quiet  # no terminal writes (async worker threads)
        # Optional path scope (spawn_agent(scope=[...])): gitignore-style globs
        # relative to the project root. When set, every file-mutation tool call
        # whose target falls OUTSIDE the scope is refused with an explicit
        # error naming the scope — never a silent denial. None = unrestricted.
        self.scope = list(scope) if scope else None
        # Per-agent git worktree (NX_SUBAGENT_WORKTREE=1, 0.97.0): when set,
        # the write-tier wrapper rewrites relative paths into this tree and
        # run_tests/git_commit default to it — the agent edits and verifies in
        # its own checkout, never the parent's working tree. None = shared tree.
        self.worktree = worktree
        # (client, model, extra_request_kwargs) this agent's model calls use,
        # or None to inherit the parent's active source/model. Pinned by
        # _subagent_model.install() at agent start (NX_SUBAGENT_SOURCE/_MODEL).
        self.model_target = None
        # Job id for an async worker (0.106.0): lets the turn loop resolve this
        # thread's per-agent steering queue and stamp steer breadcrumbs on the
        # job record. None for foreground/parent loops.
        self.job_id = job_id
        # Advisor-tier consult budget (0.112.0): remaining consult_advisor
        # calls this agent may make (NX_SUBAGENT_CONSULT_BUDGET, default 3).
        self.consults_left = _consult_budget()
        # Write-churn tracking (0.117.0/0.118.0): successful file-mutation
        # calls and whether each mid-run "start editing" nudge has fired. A
        # write/implement agent that burns turns on pure deliberation gets two
        # escalating in-loop steering notes instead of being detected only at
        # completion.
        self.write_calls = 0
        self.churn_nudged = False
        self.churn_escalated = False
        self.advisor = False  # True only for advisor-tier agents
        # Spawn tier string (0.121.2): "read"/"write"/"implement"/"advisor".
        # Recovery-budget scoping in _subagent.py uses it to give write-tier
        # agents a larger rescue allowance than the read-tier clamp.
        self.tier = "read"


def _subagent_table() -> dict[str, Any] | None:
    """The restricted dispatch table for THIS thread, or None outside a
    sub-agent loop. TLS wins over the global flag (concurrent agents)."""
    ctx = _subagent_ctx()
    if ctx is not None:
        return ctx.table
    return st._SUBAGENT_DISPATCH


def _subagent_active() -> bool:
    if _subagent_ctx() is not None:
        return True
    # global fallback is for serial callers/tests that set it directly; under
    # concurrent agents it's racy (a sibling's True is visible here), so TLS
    # is the authoritative signal and the global is never read on agent
    # threads' entry paths.
    return bool(st._SUBAGENT_ACTIVE)


def _write_tier_enabled() -> bool:
    """NX_SUBAGENT_WRITE=1 opts into write-capable sub-agents. Read from the
    top-level module attribute first (resolved from the env at import, like
    NX_SUBAGENTS) so tests/REPL code can flip it live; the env is the
    fallback when the attribute isn't there (partially-initialized module)."""
    import nx as _m
    try:
        return bool(int(_m.NX_SUBAGENT_WRITE))
    except (AttributeError, TypeError, ValueError):
        return bool(_env_int("NX_SUBAGENT_WRITE", 0))


def _subagent_tools() -> list[dict[str, Any]]:
    """Tool schemas for the CURRENT thread's sub-loop (read via the TLS
    context, so concurrent agents each get their own tier's set): read-only
    entries, plus the write-tier tools when the agent was spawned with a
    write/implement tier. Matches the restricted dispatch table exactly —
    the agent is never shown a schema its dispatch would refuse."""
    import nx as _m
    wanted = set(_READ_ONLY_TOOL_NAMES)
    ctx = _subagent_ctx()
    if ctx is not None and getattr(ctx, "tier", "") == "critique":
        # Critique tier: the read-only set exactly — no write tools, no
        # consult_advisor, no run_tests (see _critique_tier_dispatch).
        return [t for t in _m.TOOLS if t["function"]["name"] in wanted]
    if ctx is not None and ctx.write:
        wanted.update(_WRITE_TIER_TOOL_NAMES)
        if ctx.advisor and ctx.consults_left is not None:
            wanted.add("consult_advisor")  # advisor tier (0.112.0)
    elif ctx is None and _write_tier_enabled():
        # legacy global-table fallback (serial tests/REPL)
        wanted.update(_WRITE_TIER_TOOL_NAMES)
    tools = [t for t in _m.TOOLS if t["function"]["name"] in wanted]
    if ctx is not None and ctx.advisor and ctx.consults_left is not None:
        tools.append(CONSULT_ADVISOR_TOOL)
    return tools


# Schema for the advisor-tier consult tool. Deliberately NOT in the parent
# TOOLS list: it exists only in the advisor dispatch table, so it costs zero
# tokens on the per-turn floor (the always-on budget in
# tests/test_initial_context.py must not move).
CONSULT_ADVISOR_TOOL = {"type": "function", "function": {
    "name": "consult_advisor",
    "description": "Ask the parent model (your advisor) to decide something you "
        "cannot resolve yourself — an ambiguous requirement, a fork with real "
        "tradeoffs, or a repeated failure. One-shot: the advisor replies with a "
        "decision and guidance; it has no tools and never sees your transcript, "
        "so put the needed context in context_digest. Budget-limited (default 3 "
        "per agent); when it is exhausted, unavailable, or its answer is "
        "unusable, take the conservative option, log it under '## Deviations' "
        "in implementation-notes.md, and continue without consulting.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "The specific decision you cannot make, stated so it can be answered in one or two sentences."},
        "options": {"type": "array", "items": {"type": "string"}, "description": "2-3 candidate paths, each with its main tradeoff."},
        "context_digest": {"type": "string", "description": "Compact context the advisor needs: your instruction, current plan, turns spent, and the relevant findings so far. The advisor cannot see anything else."},
    }, "required": ["question"]},
}}


# Schema for the critique tier (the scoped minimal reviewer, 0.132.0). It is
# appended to TOOLS in nx/__init__.py but gated on NX_SUBAGENTS exactly like
# spawn_agent: no sub-agents, no reviewer, 0 schema tokens on the floor.
CRITIQUE_TOOL = {"type": "function", "function": {
    "name": "critique",
    "description": "Hand the work in progress to a read-only REVIEWER sub-agent "
        "and get back ranked findings. The reviewer cannot fix, edit or run "
        "tests, so a finding is never quietly patched before you judge it. Use "
        "it after a substantive change or before a risky merge: critique(focus="
        "'the timeout changes in _turn.py vs the claim that every wait is "
        "bounded') returns AT MOST 5 findings, one per line — SEVERITY — "
        "file:line — what is wrong — fix sketch — or the literal 'No findings "
        "above the floor.' with what was checked. Not for: style nits a linter "
        "catches, a review with no focus (the scope IS the feature — it returns "
        "noise), or asking the reviewer to fix what it finds (it cannot; you do).",
    "parameters": {"type": "object", "properties": {
        "focus": {"type": "string", "description": "The claim, change or diff to judge, and against what — e.g. 'the retry loop in nx/_turn.py as changed by this diff: does it ever exit early?'. Required: without a focus the reviewer returns noise."},
        "paths": {"type": "string", "description": "Optional paths/globs to confine the reading to, space- or comma-separated (e.g. 'nx/_turn.py nx/_compact*.py'). The reviewer reads them; it never writes."},
        "severity": {"enum": ["critical", "major", "minor", "info"], "description": "Findings below this floor are dropped (default 'major'). 'critical' = would this lose data or lie to the user; 'minor' = show me everything the reviewer sees."},
        "max_turns": {"type": "integer", "description": "Tool-turn budget for the reviewer (default 100; a smaller request is raised to the floor, ceiling 500). A review wraps up in far fewer turns than an investigation — 30 is usually ample."},
    }, "required": ["focus"]},
}}


def _read_only_dispatch() -> dict[str, Any]:
    from ._subagent_bash import _subagent_bash
    from ._dispatch import DISPATCH
    table = {name: DISPATCH[name] for name in _READ_ONLY_TOOL_NAMES if name in DISPATCH}
    table["run_bash"] = _subagent_bash  # ty: ignore[invalid-assignment]  # allowlist gate in front of run_bash
    return table


def _write_tier_dispatch() -> dict[str, Any]:
    """Dispatch table for a write-tier sub-agent: the read-only entries (bash
    still allowlist-gated — git diff/grep for verifying an edit) plus the four
    file-mutation tools bound to _write_tier_wrapper, which runs the parent's
    real tool body OUTSIDE the sub-agent context. The body's own _confirm()
    then hits the normal gate — risk classification, the Y/n prompt with the
    same colored diff preview the parent's writes show — so a sub-agent edit
    is exactly as approved as a parent edit, just tagged (sub-agent) in the
    tool box."""
    table = _read_only_dispatch()
    for name in _WRITE_TIER_TOOL_NAMES:
        table[name] = _write_tier_wrapper(name)
    return table


def _advisor_tier_dispatch() -> dict[str, Any]:
    """Dispatch table for an advisor-tier sub-agent: the write-tier table
    (the executor must be able to edit and maintain implementation-notes.md)
    plus consult_advisor, the round-trip to the parent model."""
    table = _write_tier_dispatch()
    table["consult_advisor"] = _consult_advisor
    return table


def _critique_tier_dispatch() -> dict[str, Any]:
    """Dispatch table for a critique-tier sub-agent: the READ-ONLY table and
    nothing else. A reviewer that could edit would fix what it found instead of
    reporting it — and the parent, which owns the diff and the approval gate,
    would lose the chance to judge the fix. So the restriction is the feature:
    no write tier, no run_tests (it mutates caches), never wait_background."""
    return _read_only_dispatch()


# Consult budget/timeout, resolved per-call from env so tests can flip them.
def _consult_budget() -> int:
    return max(0, _env_int("NX_SUBAGENT_CONSULT_BUDGET", 3))


def _consult_timeout() -> int:
    return max(1, _env_int("NX_SUBAGENT_CONSULT_TIMEOUT", 60))


def _consult_advisor(**args: Any) -> str:
    """The advisor-tier executor's round-trip to the parent model. One-shot,
    no-tools side-call (ADVISOR_SYSTEM prompt), exactly like _risk's
    classifier call — never the parent turn loop, never the user. Returns a
    compact decision+guidance string; on timeout/malformed/exhausted budget
    returns an ERROR string telling the executor to take the conservative
    option and log a deviation (its own failure mode, not a hard stop)."""
    import json as _json
    import time as _time
    ctx = _subagent_ctx()
    if ctx is None:
        return "ERROR: consult_advisor used outside a sub-agent"
    budget = ctx.consults_left
    if budget <= 0:
        return ("ERROR: advisor consult budget exhausted — take the "
                "conservative option, log it under '## Deviations' in "
                "implementation-notes.md, and continue without consulting")
    ctx.consults_left = budget - 1
    from ._prompts import ADVISOR_SYSTEM
    if st.client is None or st.MODEL is None:
        return ("ERROR: no active model source — take the conservative "
                "option and log it under '## Deviations'")
    question = str(args.get("question") or "").strip()
    options = args.get("options")
    digest = str(args.get("context_digest") or "").strip()
    if not question:
        return "ERROR: consult_advisor requires question"
    opts = ""
    if isinstance(options, list) and options:
        opts = "\n".join(f"- {o}" for o in options if str(o).strip())
    user = (f"Question: {question}\n\nOptions:\n{opts or '(none given)'}\n\n"
            f"Context: {digest or '(none given)'}")
    payload = [{"role": "system", "content": ADVISOR_SYSTEM},
               {"role": "user", "content": user}]
    deadline = _time.monotonic() + _consult_timeout()
    from ._openai_client import APIConnectionError
    last_err = ""
    # One parse retry — same ladder philosophy as _risk._assess_risk, but
    # cheaper: an advisor fallback is a logged deviation, not a gate.
    for _attempt in range(2):
        resp = None
        while _time.monotonic() < deadline:
            try:
                resp = st.client.chat.completions.create(
                    model=st.MODEL, messages=payload, stream=False,
                    timeout=max(1, int(deadline - _time.monotonic())),
                    **(st.ACTIVE.extra_request_kwargs() if st.ACTIVE else {}))
            except APIConnectionError:
                _time.sleep(1)
            # A reply that only arrives after the deadline is a timeout too —
            # discard it and fall back rather than acting on stale guidance.
            if resp is not None and _time.monotonic() >= deadline:
                resp = None
            if resp is not None:
                break
        if resp is None:
            return ("ERROR: advisor did not answer in time — take the "
                    "conservative option and log it under '## Deviations'")
        text = (resp.choices[0].message.content or "").strip()
        try:
            from ._metrics import _record_api_call
            _record_api_call(resp,
                             fallback_chars=len(text),
                             fallback_prompt_chars=lambda: sum(
                                 len(str(m.get("content") or "")) for m in payload))
        except Exception:
            pass
        try:
            obj = _json.loads(text)
            decision = str(obj.get("decision") or "").strip()
            guidance = str(obj.get("guidance") or "").strip()
            stop = bool(obj.get("stop"))
            if decision or stop:
                out = f"[advisor] decision: {decision or 'STOP — halt and report back'}"
                if guidance:
                    out += f"\n[advisor] guidance: {guidance}"
                if stop:
                    out += ("\n[advisor] STOP requested: wrap up now — record state "
                            "in implementation-notes.md and report back to the parent")
                return out
            last_err = "advisor reply missing decision"
        except (_json.JSONDecodeError, AttributeError, TypeError):
            last_err = f"unparseable advisor reply: {text[:200]}"
    return (f"ERROR: advisor reply unusable ({last_err}) — take the "
            "conservative option and log it under '## Deviations'")


def _scope_check(path: str) -> str | None:
    """Refusal message when `path` falls outside the CURRENT thread's sub-agent
    scope, or None when allowed (no scope set, or any glob matches). Matched
    against the path relative to the agent's worktree when it has one (the
    agent's relative paths name ITS tree), else PROJECT_ROOT, plus the
    basename as a fallback so scope=["*.md"] works."""
    ctx = _subagent_ctx()
    if ctx is None or not ctx.scope:
        return None
    root = getattr(ctx, "worktree", None) or st.PROJECT_ROOT
    rel = os.path.relpath(os.path.abspath(path), root)
    for pat in ctx.scope:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat):
            return None
    return (f"ERROR: {os.path.basename(rel)!r} is outside this sub-agent's "
            f"scope {ctx.scope} — write only inside the scoped paths, or "
            "report back that the change needs a wider scope")


def _normalize_tier(tier: Any) -> str:
    """Map the `tier` argument to one of read/write/implement/advisor/critique.
    Empty string = invalid (caller refuses); None = "read" — the schema's
    documented default.

    0.110.1: None used to fall back to the NX_SUBAGENT_WRITE env flag, silently
    upgrading an omitted tier to "write". That broke the schema's "'read'
    (default)" contract, and since 0.107.0 a write tier + NX_SUBAGENT_WORKTREE=1
    + a non-git project root is a REFUSED spawn — so a plain read investigation
    from a non-git cwd died on "worktree isolation failed". NX_SUBAGENT_WRITE=1
    now only gates explicit write/implement requests (checked in spawn_agent)."""
    if tier is None:
        return "read"
    v = str(tier).strip().lower()
    return v if v in ("read", "write", "implement", "advisor", "critique") else ""


# The critique tier's severity floor, shared by the tier prompt injection and
# the caller-side validation: the reviewer reports only findings at or above
# this level, so a parent asking for "critical" gets no minor noise.
CRITIQUE_SEVERITIES = ("critical", "major", "minor", "info")
DEFAULT_CRITIQUE_SEVERITY = "major"


def _critique_severity(value: Any) -> str:
    """Normalize a severity floor to one of CRITIQUE_SEVERITIES, defaulting to
    major when omitted (an unstated floor is the common case; 'major' is the
    level a parent usually means by 'review this')."""
    v = str(value or "").strip().lower()
    return v if v in CRITIQUE_SEVERITIES else DEFAULT_CRITIQUE_SEVERITY


def _write_tier_wrapper(name: str):
    """Build the confirm-through binding for one write-tier tool.

    Closures over the parent's real body (resolved at spawn time) so a test
    or hook that swaps nx.write_file still governs what actually runs. The
    wrapper tears down this thread's sub-agent context around the body so
    _confirm() sees the parent's real approval state (and a nested
    spawn_agent from inside a hook would be refused, not looped)."""
    from ._dispatch import DISPATCH

    def _run(**args: Any) -> str:
        body = DISPATCH.get(name)
        if body is None:
            return f"ERROR: {name} is unavailable"
        ctx = _subagent_ctx()
        if ctx is None:  # defensive: a wrapper escaped its sub-agent context
            return f"ERROR: {name} write-tier wrapper used outside a sub-agent"
        worktree = getattr(ctx, "worktree", None)
        if worktree:
            args = _worktree_args(name, args, worktree)
        # Path scope gate BEFORE the body/approval: an out-of-scope write is
        # an explicit refusal, never a prompt and never a silent denial.
        target = _scope_target(name, args)
        if target is not None:
            refusal = _scope_check(target)
            if refusal is not None:
                return refusal
        prev_ctx = _SUBAGENT_TLS.ctx
        _SUBAGENT_TLS.ctx = None  # body + _confirm() run as the parent
        try:
            result = body(**args)
        finally:
            _SUBAGENT_TLS.ctx = prev_ctx
        # Count successful mutations for the write-churn nudge (a refused or
        # failed edit is not progress): any result that is not an ERROR/
        # refusal prefix counts. Cheap string peek, best-effort by design.
        if isinstance(result, str) and not result.startswith(("ERROR", "REFUSED")):
            ctx.write_calls += 1
        return result

    _run.__name__ = f"_subagent_write_{name}"
    return _run


def _worktree_args(name: str, args: dict[str, Any], worktree: str) -> dict[str, Any]:
    """Rewrite a write-tier call's path arguments into the agent's worktree
    (NX_SUBAGENT_WORKTREE=1). Relative paths — what the agent names — map to
    the tree; absolute paths inside it pass through; absolute paths outside
    are left for the scope gate / write fence to refuse. run_tests gets its
    targets and cwd anchored; git_commit lands in the tree."""
    from ._worktree import rewrite_path
    args = dict(args)
    for key in ("path", "file", "targets", "files"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            args[key] = rewrite_path(v, worktree)
        elif isinstance(v, list):
            args[key] = [rewrite_path(x, worktree) if isinstance(x, str) else x
                         for x in v]
    if name == "git_commit" and not str(args.get("path", "")).strip():
        args["path"] = worktree
    if name == "run_tests":
        args["cwd"] = worktree
    return args


def _scope_target(name: str, args: dict[str, Any]) -> str | None:
    """The primary filesystem target of a write-tier tool call, for the scope
    gate (None when the tool has no single target, e.g. run_tests)."""
    for key in ("path", "file"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v
    if name == "edit_file":
        files = args.get("path") or args.get("files")
        if isinstance(files, str):
            return files
    return None
