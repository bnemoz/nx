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
import re
import sys
from typing import Optional

from ._ansi import DIM, RESET, C, raw_mode
from ._tty import read_byte, wait_ready
from ._risk import _local_risk_hint, LEVEL_ORDER
from . import _state as st

_DIFF_CONTEXT = 2


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _diff_preview(before: str, after: str, *, context: int = _DIFF_CONTEXT) -> str:
    """Render a colored unified diff of `before` -> `after` for the approval
    prompt, with old/new line numbers prefixed on each line. Returns "" if the
    two are identical. Purely a terminal-UX helper — never sent to the model,
    so it doesn't affect the token floor."""
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), n=context, lineterm="")
    out = []
    old_no = new_no = 0
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(2))
            out.append(f"{DIM}{line}{RESET}")
        elif line.startswith("+"):
            out.append(f"{C.GREEN}{'':>4} {new_no:>4} {line}{RESET}")
            new_no += 1
        elif line.startswith("-"):
            out.append(f"{C.RED}{old_no:>4} {'':>4} {line}{RESET}")
            old_no += 1
        else:
            out.append(f"{DIM}{old_no:>4} {new_no:>4} {line}{RESET}")
            old_no += 1
            new_no += 1
    return "\n".join(out)


def _diff_hunk(before: str, after: str, *, context: int = 3) -> str:
    """Plain (no-ANSI) unified diff of `before` -> `after`, for feeding back to
    the model as tool output so it doesn't need to read_file the whole file
    again just to see what changed. Returns "" if identical."""
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), n=context, lineterm="")
    return "\n".join(line for line in lines if not (line.startswith("+++") or line.startswith("---")))

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


def _ask_approval(prompt: str, allow_hint: Optional[str] = None) -> str:
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
    """
    keys_hint = "[Y/n/a/esc]" if allow_hint else "[Y/n/esc]"
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
        print("Unrecognized response; please enter Y, n, a (always), or esc.")
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
                # Shift+Tab toggles YOLO here too — mirroring the chatbox
                # editor so a user who started in manual mode can switch to
                # auto-approve mid-turn, right at the approval prompt,
                # instead of being stuck in manual mode until the turn ends
                # and the chatbox comes back. A visible confirmation is
                # printed (never a silent flip) and the SAME action is
                # re-prompted: the toggle changes future approvals, not this
                # one, so a stray muscle-memory Shift+Tab can't sneak a
                # high-risk action through as a side effect.
                if rest in ("[Z", "[1;2Z"):
                    st.toggle_yolo()
                    on = "ON" if st.YOLO else "OFF"
                    sys.stdout.write(
                        f"{DIM}\n  yolo {on}  approval={_approval_display()}{RESET}\n")
                    sys.stdout.flush()
                    return _ask_approval(prompt, allow_hint)
                # Any other non-bare-Esc sequence (arrow keys, Home/End,
                # bracketed paste, …) is meaningless at a Y/n/esc prompt.
                sys.stdout.write(f"{DIM}unrecognized key sequence; please press Y, n, or esc{RESET}\n")
                sys.stdout.flush()
                return _ask_approval(prompt, allow_hint)
            sys.stdout.write(f"{DIM}esc{RESET}\n")
            sys.stdout.flush()
            return "esc"
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
            sys.stdout.write(f"{DIM}unrecognized response; please press Y, n, or esc{RESET}\n")
            sys.stdout.flush()
            return _ask_approval(prompt, allow_hint)
        sys.stdout.write(f"{DIM}{shown}{RESET}\n")
        sys.stdout.flush()
        return result


def _confirm(action: str, preview: Optional[str] = None) -> bool:
    """Decide whether to run `action`. Returns True to proceed, False to deny.
    Raises `_EscToChat` if the user presses Esc — propagating up so the turn
    stops and the REPL drops back to the chat input.

    Flow:
      1. YOLO → True (no call, no prompt). `preview` is still printed so
         diffs stay colored instead of falling back to plain tool-output text.
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
    if st.YOLO:
        if preview:
            print(preview)
        return True
    # Saved allow rules ("A" at a previous prompt, /approval allow) come
    # before the risk classifier: an explicitly-blessed action shape needs no
    # model round-trip and never prompts, regardless of the level threshold.
    from ._allowed import check_allow, _allowlist_tool
    hit = check_allow(_allowlist_tool(action), action)
    if hit is not None:
        pattern, scope = hit
        short = action if len(action) <= 80 else action[:77] + "..."
        print(f"{DIM}  ↳ auto-allow [rule: {pattern!r} ({scope})] {short}{RESET}")
        if preview:
            print(preview)
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
    try:
        level, reason = _local_risk_hint(action) or _m._assess_risk(action)
        # APPROVE_LEVEL is the max level to AUTO-APPROVE. level ≤ threshold → run.
        if st.APPROVE_LEVEL is not None and LEVEL_ORDER[level] <= LEVEL_ORDER[st.APPROVE_LEVEL]:
            # Auto-allow. Show the assessment so the user has a paper trail.
            short = reason if len(reason) <= 80 else reason[:77] + "..."
            pause_spinner()
            print(f"{DIM}  ↳ auto-allow [{level}] {action}  ({short}){RESET}")
            if preview:
                print(preview)
            return True
        short = reason if len(reason) <= 80 else reason[:77] + "..."
        lvl_color = {"low": DIM, "medium": C.WARNING, "high": C.ERROR}[level]
        pause_spinner()
        if preview:
            print(preview)
        choice = _m._ask_approval(
            f"{C.WARNING}  allow {action}? {lvl_color}[risk: {level.upper()} — {short}]{RESET} "
            f"{C.WARNING}[Y/n/a/esc] {RESET}", allow_hint=action)
        if choice == "esc":
            raise _EscToChat(action)
        if choice == "a":
            # Save a persistent allow rule derived from this action, then run.
            # The rule matches the exact shape shown above (tool + path/command
            # prefix), so a one-off permission never silently widens into yolo.
            from ._allowed import add_rule, _rule_from_action
            rule = add_rule(*_rule_from_action(action))
            print(f"{DIM}  ↳ always allowing {rule['tool']} {rule['pattern']!r} "
                  f"({rule['scope']}) — /approval allow list to review{RESET}")
            if preview:
                print(preview)
            return True
        return choice != "n"
    finally:
        if sp is not None and paused:
            sp.start()
