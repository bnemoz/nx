"""Approval gating: three risk levels (low < medium < high) plus an explicit
"prompt all" state and YOLO mode. APPROVE_LEVEL is the maximum risk level to
AUTO-APPROVE: actions classified at <= APPROVE_LEVEL run without prompting;
anything strictly above prompts. APPROVE_LEVEL=None means prompt for every
classified action. YOLO=True short-circuits entirely and skips the
classifier call.
  None     -> prompt low + medium + high
  "low"    -> auto-approve low (reads, grep, wc); prompt medium + high
  "medium" -> auto-approve low + medium (edits, cp, mv, tests); prompt high
  "high"   -> auto-approve low + medium + high

On top of the level threshold, a persistent per-pattern allowlist
(nx/_allowed.py) can auto-allow specific action shapes the user blessed with
`A` at a prompt (or /approval allow) — checked BEFORE the classifier call, so
an allowed action costs no model round-trip.

Also owns the interactive Y/n/esc prompt (_ask_approval), the top-level
_confirm() gate every write/edit/bash tool call goes through, and the
_EscToChat control-flow exception that lets the user bail out of an
approval prompt back to the chat input.
"""
import difflib
import sys
from contextlib import contextmanager
from typing import Optional

from ._ansi import DIM, RESET, C, notice, raw_mode
from ._env import _env_int
from ._keys import print_overlay, prompt_tail
from ._layout import DIFF_CONTEXT_LINES
from ._render import commit_line as _commit_line
from ._render import diff as _render_diff
from ._render import notice as _role_notice
from ._tty import read_byte, wait_ready
from ._risk import _local_risk_hint, LEVEL_ORDER
from . import _state as st

# Approval diff-preview context lines (nx/_layout.py, pure move).
_DIFF_CONTEXT = DIFF_CONTEXT_LINES


def _diff_preview(before: str, after: str, *, context: int = _DIFF_CONTEXT) -> str:
    """Render a colored unified diff of `before` -> `after` for the approval
    prompt, with old/new line numbers prefixed on each line. Returns "" if the
    two are identical. Purely a terminal-UX helper — never sent to the model,
    so it doesn't affect the token floor.

    0.135.0 (UIUX_SPEC B1): the rendering moved to `_render.diff()` — one
    implementation shared with the /review-style diff surfaces — and its added/
    removed lines now take the DIFF_ADD/DIFF_REMOVE roles instead of the raw
    green/red palette hues. This name stays (it is re-exported via
    `nx._diff_preview` and used by `_edit_tools`)."""
    return _render_diff(before, after, context)


def _diff_hunk(before: str, after: str, *, context: int = 3) -> str:
    """Plain (no-ANSI) unified diff of `before` -> `after`, for feeding back to
    the model as tool output so it doesn't need to read_file the whole file
    again just to see what changed. Returns "" if identical.

    Bounded (PF-19, 0.61.0): a one-line edit in a repetitive 30k-line file can
    produce a 520 KB diff, and this text *is* fed back to the model. The hunk
    is capped at NX_DIFF_HUNK_MAX_CHARS (default 8192) — explicit here rather
    than silently relying on a later overflow filter — and a trailing notice
    says how much was dropped so the model can read_file the rest instead of
    assuming it saw everything."""
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), n=context, lineterm="")
    body = "\n".join(line for line in lines
                     if not (line.startswith("+++") or line.startswith("---")))
    cap = _env_int("NX_DIFF_HUNK_MAX_CHARS", 8192)
    if cap > 0 and len(body) > cap:
        dropped = len(body) - cap
        note = "\n[diff truncated: " + str(dropped) + " chars elided; read_file for the rest]"
        body = body[:cap] + note
    return body

# Accept full words and common abbreviations (med, hi, lo, m, h, l …).
_LEVEL_ALIASES = {
    "l": "low", "lo": "low", "low": "low",
    "m": "medium", "med": "medium", "mid": "medium", "medium": "medium",
    "h": "high", "hi": "high", "high": "high",
}


def _normalize_level(arg: str) -> Optional[str]:
    """Resolve 'med' → 'medium', 'hi' → 'high', etc. Returns the canonical
    level name or None if the input isn't recognised."""
    return _LEVEL_ALIASES.get(arg.lower().strip())


def _normalize_approval(arg: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve approval setting aliases.

    Returns (kind, value):
      ("level", "low"|"medium"|"high") for auto-approval thresholds
      ("prompt_all", None) for strict approval of every classified action
      ("yolo", None) for never-prompt mode
      (None, None) when unrecognized
    """
    raw = arg.lower().strip()
    if raw in ("all", "prompt", "prompt-all", "strict", "none"):
        return "prompt_all", None
    if raw == "yolo":
        return "yolo", None
    if lvl := _normalize_level(raw):
        return "level", lvl
    return None, None


def _approval_display() -> str:
    if st.YOLO:
        return "off (yolo)"
    if st.APPROVE_LEVEL is None:
        return "all"
    return st.APPROVE_LEVEL


def _apply_approval(kind: Optional[str], level: Optional[str] = None, *, update_default: bool = True) -> bool:
    if kind == "yolo":
        st.YOLO = True
        st.APPROVE_LEVEL = None
        if update_default:
            st.DEFAULT_APPROVE_LEVEL = None
    elif kind == "prompt_all":
        st.YOLO = False
        st.APPROVE_LEVEL = None
        if update_default:
            st.DEFAULT_APPROVE_LEVEL = None
    elif kind == "level":
        st.YOLO = False
        st.APPROVE_LEVEL = level
        if update_default:
            st.DEFAULT_APPROVE_LEVEL = level
    else:
        return False
    return True


# A special exception that propagates up from _confirm (via the tool function
# body → run_tool → model_turn) when the user presses Esc at an approval prompt.
# It's not a real error — it's a control-flow signal meaning "stop this turn and
# drop the user back to the chat input so they can add more guidance."
class _EscToChat(Exception):
    pass


@contextmanager
def _paused_spinner():
    """Pause the active tool spinner (if any) around direct-to-stdout prompts
    so its periodic CLEAR_LINE redraws don't erase the prompt. Callers that
    bypass _confirm (the secret/clobber guards in run_bash) wrap their
    _ask_approval in this; _confirm inlines the same pause/resume."""
    sp = st._ACTIVE_SPINNER
    paused = False
    try:
        if sp is not None:
            sp.stop()
            paused = True
        yield
    finally:
        if paused:
            sp.start()


def _ask_approval(prompt: str, allow_hint: Optional[str] = None,
                  diff_src: Optional[tuple] = None,
                  context: int = _DIFF_CONTEXT) -> str:
    """Read a single keypress for a Y/n/esc approval prompt. Returns one of
    'y', 'n', 'esc', 'always'. Esc is accepted both as the bare Esc key (read
    in raw mode) and as the letter 'e'/'E' for terminals where raw mode isn't
    usable. Falls back to line-based `input()` when stdin isn't a TTY.

    Only an explicit Y (or Enter, the documented default) permits an action.
    Invalid input is discarded and the prompt is shown again; it must never
    inherit the default approval, especially for high-risk actions.
    `allow_hint` (the action string) enables the `a` key: "always allow
    actions matching this shape" — the prompt advertises [Y/n/a/esc] and the
    saved rule auto-allows future matches via nx/_allowed.py.

    C3 (0.136.0) keys (registry: nx/_keys.py "approval", same source as the
    prompt's hint tail and the `?` overlay):
      +/-   widen/narrow the diff preview — needs `diff_src=(before, after)`;
            re-renders `_render.diff` at the new `context` (default
            `_layout.DIFF_CONTEXT_LINES`) and re-prompts.
      ?     one-frame key overlay above the prompt, any key dismisses.
      Shift+Tab  toggle YOLO; after confirming the toggle the PENDING action
            proceeds (returns 'y') instead of being re-prompted — the prompt
            only ever appears with YOLO off, so the toggle at a prompt is
            always ON, and a user who switches to auto-approve mid-turn means
            it for the action in front of them, not just the next one.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        ans = input(prompt).strip().lower()
        if ans in ("", "y", "yes"):
            return "y"
        if ans in ("esc", "e", "\x1b"):
            return "esc"
        if ans in ("a", "always"):
            return "a"
        if ans == "n":
            return "n"
        notice("Unrecognized response; please enter Y, n, a (always), or esc.")
        return _ask_approval(prompt, allow_hint)
    fd = sys.stdin.fileno()
    with raw_mode(fd):
        sys.stdout.write(prompt)
        sys.stdout.flush()
        c = read_byte(fd).decode("utf-8", "replace")
        if c == "\x1b":
            # Could be bare Esc OR the lead byte of an escape sequence
            # (arrow keys, Home/End, Shift+Tab, bracketed paste, …). Wait
            # briefly; if no more bytes arrive, it's a bare Esc.
            r = wait_ready(fd, 0.05)
            if r:
                rest = ""
                # Drain the rest of the sequence (bounded, so a runaway
                # paste can't hang us).
                for _ in range(8):
                    if not wait_ready(fd, 0.02):
                        break
                    try:
                        rest += read_byte(fd).decode("utf-8", "replace")
                    except OSError:
                        break
                # C3 (0.136.0): Shift+Tab toggles YOLO AND proceeds with the
                # pending action (spec: "after confirming, proceed with the
                # pending action instead of re-prompting"). A prompt can only
                # appear with YOLO off, so this toggle is always ON — the
                # visible confirmation stays (never a silent flip), and the
                # action rides the just-enabled auto-approve.
                if rest in ("[Z", "[1;2Z"):
                    st.toggle_yolo()
                    on = "ON" if st.YOLO else "OFF"
                    notice(f"{DIM}\n  yolo {on}  approval={_approval_display()}"
                           f"{'  — proceeding' if st.YOLO else ''}{RESET}")
                    if st.YOLO:
                        return "y"
                    return _ask_approval(prompt, allow_hint,
                                         diff_src=diff_src, context=context)
                # C2: `?` opens the one-frame key overlay (registry-driven,
                # same data as the prompt hint tail); any key dismisses — the
                # next iteration re-prompts.
                if rest and rest.startswith("[") and "?" in rest:
                    print_overlay("approval")
                    return _ask_approval(prompt, allow_hint,
                                         diff_src=diff_src, context=context)
                # Any other non-bare-Esc sequence (arrow keys, Home/End,
                # bracketed paste, …) is meaningless at a Y/n/esc prompt.
                notice(f"{DIM}unrecognized key sequence; please press Y, n, or esc{RESET}")
                return _ask_approval(prompt, allow_hint,
                                     diff_src=diff_src, context=context)
            sys.stdout.write(f"{DIM}esc{RESET}\n")
            sys.stdout.flush()
            return "esc"
        # C3: `?` (plain byte) — same overlay as the escape-sequence guard.
        if c == "?":
            print_overlay("approval")
            return _ask_approval(prompt, allow_hint,
                                 diff_src=diff_src, context=context)
        # C3: +/- widen/narrow the diff context (bounded 0..30 — a full-file
        # view is what read_file is for). Needs the structured diff source;
        # without it (bash/commit previews) the keys are inert, like before.
        if c in ("+", "=") or c == "-":
            new_ctx = min(30, context + 1) if c in ("+", "=") else max(0, context - 1)
            if diff_src is not None and new_ctx != context:
                sys.stdout.write(f"\n{DIM}  ·· diff context {new_ctx} ··{RESET}\n")
                sys.stdout.write(_render_diff(diff_src[0], diff_src[1],
                                              context=new_ctx) + "\n")
                sys.stdout.flush()
                return _ask_approval(prompt, allow_hint, diff_src=diff_src,
                                     context=new_ctx)
            return _ask_approval(prompt, allow_hint, diff_src=diff_src,
                                 context=context)
        # Echo the choice for feedback
        if c in ("\r", "\n"):
            shown, result = "Y (default)", "y"
        elif c in ("y", "Y"):
            shown, result = "Y", "y"
        elif c in ("n", "N"):
            shown, result = "n", "n"
        elif c in ("a", "A") and allow_hint:
            shown, result = "A (always allow this shape)", "a"
        elif c in ("e", "E"):
            shown, result = "esc", "esc"
        else:
            notice(f"{DIM}unrecognized response; please press Y, n, or esc{RESET}")
            return _ask_approval(prompt, allow_hint, diff_src=diff_src,
                                 context=context)
        sys.stdout.write(f"{DIM}{shown}{RESET}\n")
        sys.stdout.flush()
        return result


def _confirm(action: str, preview: Optional[str] = None,
             diff_src: Optional[tuple] = None) -> bool:
    """Decide whether to run `action`. Returns True to proceed, False to deny.
    Raises `_EscToChat` if the user presses Esc — propagating up so the turn
    stops and the REPL drops back to the chat input.

    Flow:
      1. YOLO → True (no call, no prompt). `preview` is still printed so
         diffs stay colored instead of falling back to plain tool-output text
         — unless TOOL_OUTPUT_VISIBLE is off, which collapses previews too
         (0.53.1: a git_commit diff used to print in full even with
         /tooloutput collapsed, defeating the toggle).
      2. A saved allow rule (from a previous `A` / /approval allow) matching
         this action → True, before any classifier round-trip.
      3. Ask the model for a risk level (skipped in step 1).
      4. If APPROVE_LEVEL is set and level ≤ APPROVE_LEVEL → auto-allow,
         printing `preview` (if given) after the auto-allow line.
      5. Otherwise prompt (Y/n/a/esc), showing the level + reason so the user
         has context. If `preview` (e.g. a diff from `_diff_preview`) is
         given, it's printed just above the prompt. `a` saves a persistent
         allow rule for this action's shape (see nx/_allowed.py).

    Reads module globals YOLO and APPROVE_LEVEL at call time — /yolo and
    /approval reassign them mid-session, and we must always see the latest
    value, not a snapshot from when the tool function was defined.
    """
    # A quiet loop (async sub-agent, batch task) can never prompt and must
    # never write to the shared terminal — not even the YOLO diff preview,
    # which is a stdout render and would pollute a batch run's JSONL stream.
    # Check this BEFORE the YOLO fast path for exactly that reason, but keep
    # the approval decision honest: without --yolo a batch task's gated call
    # is auto-DENIED (documented: batch is headless, so nothing may approve a
    # write the user never opted into) — only an explicit yolo posture
    # approves. Async read-only sub-agents are unaffected: their tools never
    # reach _confirm.
    from ._subagent import _subagent_quiet
    if _subagent_quiet():
        if st.YOLO:
            return True
        # C4 (0.135.1): NX_APPROVAL_POLICY=headless-allow:<glob> lets an
        # operator pre-approve tool names on headless surfaces (Telegram,
        # async sub-agent bash, `nx run` batch) — all three funnel through
        # this one deny point. Anything unmatched, and any unset/other
        # policy value, keeps the documented auto-deny.
        from ._approval_policy import headless_allow
        return headless_allow(action)
    # The approval preview is tool output on screen too (git_commit's staged
    # diff, edit_file's before/after), so /tooloutput collapsed must collapse
    # it like any other result body — the decision is unchanged either way.
    show_preview = bool(preview) and st.TOOL_OUTPUT_VISIBLE
    if st.YOLO:
        if show_preview:
            _commit_line(preview)
        return True
    # Saved allow rules ("A" at a previous prompt, /approval allow) come
    # before the risk classifier: an explicitly-blessed action shape needs no
    # model round-trip and never prompts, regardless of the level threshold.
    from ._allowed import check_allow, _allowlist_tool
    hit = check_allow(_allowlist_tool(action), action)
    if hit is not None:
        pattern, scope = hit
        short = action if len(action) <= 80 else action[:77] + "..."
        _role_notice("muted", f"  ↳ auto-allow [rule: {pattern!r} ({scope})] {short}")
        if show_preview:
            _commit_line(preview)
        return True
    # If a tool spinner is running, keep it alive during the risk-classifier
    # request and pause it only around our own terminal I/O. Otherwise a slow
    # classifier call looks exactly like a frozen tool.
    sp = st._ACTIVE_SPINNER
    paused = False

    def pause_spinner():
        nonlocal paused
        if sp is not None and not paused:
            sp.stop()
            paused = True

    # Go through the top-level package for _assess_risk (not a bare/local
    # reference) so a test that monkeypatches `nx._assess_risk` (e.g. to
    # avoid a real model round-trip) takes effect — _assess_risk lives in
    # __init__.py, in a different module from this function.
    import nx as _m
    # C3: pass the new kwargs only when there is a structured diff to widen
    # AND the (possibly test-patched) _ask_approval actually accepts them —
    # patches with the historical `(prompt, allow_hint=None)` signature keep
    # working untouched.
    _ask_kw: dict = {}
    if diff_src is not None:
        import inspect
        try:
            if "diff_src" in inspect.signature(_m._ask_approval).parameters:
                _ask_kw["diff_src"] = diff_src
        except (TypeError, ValueError):
            pass
    try:
        level, reason = _local_risk_hint(action) or _m._assess_risk(action)
        # APPROVE_LEVEL is the max level to AUTO-APPROVE. level ≤ threshold → run.
        if st.APPROVE_LEVEL is not None and LEVEL_ORDER[level] <= LEVEL_ORDER[st.APPROVE_LEVEL]:
            # Auto-allow. Show the assessment so the user has a paper trail.
            short = reason if len(reason) <= 80 else reason[:77] + "..."
            pause_spinner()
            _role_notice("muted", f"  ↳ auto-allow [{level}] {action}  ({short})")
            if show_preview:
                _commit_line(preview)
            return True
        short = reason if len(reason) <= 80 else reason[:77] + "..."
        lvl_color = {"low": DIM, "medium": C.WARNING, "high": C.ERROR}[level]
        pause_spinner()
        if show_preview:
            _commit_line(preview)
        choice = _m._ask_approval(
            f"{C.WARNING}  allow {action}? {lvl_color}[risk: {level.upper()} — {short}]{RESET} "
            # C3: the key tail is generated from the same nx/_keys.py registry
            # the `?` overlay renders — prompt hint and overlay can't drift.
            f"{C.WARNING}[{prompt_tail('approval')}]{RESET} ",
            allow_hint=action, **_ask_kw)
        if choice == "esc":
            raise _EscToChat(action)
        if choice == "a":
            # Save a persistent allow rule derived from this action, then run.
            # The rule matches the exact shape shown above (tool + path/command
            # prefix), so a one-off permission never silently widens into yolo.
            from ._allowed import add_rule, _rule_from_action
            rule = add_rule(*_rule_from_action(action))
            _role_notice("muted", f"  ↳ always allowing {rule['tool']} "
                                 f"{rule['pattern']!r} ({rule['scope']}) — "
                                 "/approval allow list to review")
            if show_preview:
                _commit_line(preview)
            return True
        return choice != "n"
    finally:
        if sp is not None and paused:
            sp.start()
