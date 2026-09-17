"""Rewrite grep-family invocations in run_bash commands to ripgrep.

The model occasionally writes a shell `grep`/`egrep`/`fgrep` inside run_bash
even when ripgrep is installed — usually for a path outside the project where
search_files wasn't the natural pick. Since the harness knows whether `rg` is
on PATH, the command is upgraded instead of run slower:

  - grep / egrep / fgrep at COMMAND POSITION -> rg
    (start of the string or right after a shell operator; `git grep`,
    `/usr/bin/grep` and other non-command positions pass through untouched)
  - -r / -R      -> dropped (rg recurses by default)
  - -h           -> dropped (rg always prints filenames)
  - --include=PAT     -> -g PAT
  - --exclude-dir=DIR -> -g '!DIR'
  - combined short flags: grep-only letters (-E -r -R) removed from the bundle
  - BRE alternation `\\|` in the pattern (the FIRST non-flag operand) -> `|`
    — the one grep-ism models emit most, which rg would read as a literal
    backslash. Operands after the pattern are paths; their `\\|` is untouched.

Anything else passes through. The rewrite is deliberately conservative:
shell-split the command, map token by token, re-quote. A command that fails
to split (unbalanced quotes) or has no grep-family word at command position
is returned verbatim; if rg isn't installed, grep is what the user asked for.
"""
from __future__ import annotations

import re
import shlex
import shutil

# Word-boundary match that ignores anything glued to a path or flag
# (/usr/bin/grep stays put; "grep -rn" matches).
_GREP_WORD = re.compile(r"(?<![\w./-])(?:e?grep|fgrep)(?![\w-])")
_COMBINED_FLAGS = re.compile(r"^-(?P<body>[A-Za-z]{2,})$")
# Short flags shared by grep and rg, kept as-is; grep-only letters that must
# disappear are in _DROP_CHARS.
_KEEP_CHARS = set("niclqvwxFHoxasP")
_DROP_CHARS = set("ErR")
# Operators that end one command and start the next.
_OPERATORS = {"|", "|&", "&&", "||", ";", "&", ">", ">>", "<", "<<"}
# Redirection targets like 2>&1: never a pattern, never re-quoted as a word.
_REDIRECTION = re.compile(r"^\d*[<>]")


def _rewrite_combined_flag(tok: str) -> str:
    m = _COMBINED_FLAGS.match(tok)
    body = "".join(ch for ch in m.group("body") if ch not in _DROP_CHARS)
    return f"-{body}"


def _split_shell(command: str) -> list[str] | None:
    """shlex split that keeps operators (|, &&, >, ;) as separate tokens.
    Returns None when the command doesn't parse (unbalanced quotes, stray
    operators the lexer can't place)."""
    lex = shlex.shlex(command, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError:
        return None


def _requote(tokens: list[str]) -> str:
    """Rejoin tokens, quoting words but never shell operators."""
    return " ".join(t if t in _OPERATORS or _REDIRECTION.match(t)
                    else shlex.quote(t) for t in tokens)


def prefer_ripgrep(command: str) -> str:
    """Return `command` with grep-family calls rewritten to ripgrep.

    Returns the original string unchanged when rg is absent, the command
    doesn't shell-split, or no grep-family word appears at command position.
    """
    if not shutil.which("rg") or not _GREP_WORD.search(command):
        return command
    tokens = _split_shell(command)
    if tokens is None:
        return command
    out: list[str] = []
    pattern_done = False   # first non-flag operand of a rewritten grep seen
    in_rewritten = False   # tokens of the grep invocation being rewritten
    for i, tok in enumerate(tokens):
        if tok in _OPERATORS or _REDIRECTION.match(tok):
            out.append(tok)
            in_rewritten = False  # operator ends the grep invocation
            continue
        at_command_start = i == 0 or tokens[i - 1] in _OPERATORS
        if at_command_start and _GREP_WORD.fullmatch(tok):
            out.append("rg")
            in_rewritten = True
            pattern_done = False
            continue
        if in_rewritten:
            if tok.startswith("--include="):
                out.extend(("-g", tok.split("=", 1)[1]))
                continue
            if tok.startswith("--exclude-dir="):
                out.extend(("-g", "!" + tok.split("=", 1)[1]))
                continue
            if tok in ("-E", "-r", "-R", "-h"):
                continue  # dropped: rg defaults make them no-ops
            if _COMBINED_FLAGS.match(tok):
                tok = _rewrite_combined_flag(tok)
                if tok == "-":
                    continue
            elif not tok.startswith("-") and not pattern_done \
                    and not _REDIRECTION.match(tok):
                # Pattern is the first non-flag operand: BRE `\|` -> ERE `|`.
                tok = tok.replace(r"\|", "|")
                pattern_done = True
            out.append(tok)
            continue
        # A token of a command that is not being rewritten (git grep, tail,
        # echo after a pipe, ...) passes through byte-for-byte.
        out.append(tok)
    return _requote(out)
