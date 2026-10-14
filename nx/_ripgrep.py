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
shell-split the command, map token by token, and only re-emit a command that
was actually rewritten — when no grep-family word qualifies (git grep, a
subshell, a glob carrying argument, a heredoc) the ORIGINAL string is
returned verbatim, because a token-level re-quote would collapse `$()`,
globs, tildes and heredocs into literals. If rg isn't installed, grep is
what the user asked for.

Forms at command position that still count as a grep invocation: env-assignment
prefixes (`LC_ALL=C grep …`), wrapper binaries (`sudo/env/time/command/nohup/
nice/exec grep …`), and control keywords (`if/while/until/do grep …`).
"""
from __future__ import annotations

import re
import shlex
import shutil

# Word-boundary match that ignores anything glued to a path or flag
# (/usr/bin/grep stays put; "grep -rn" matches).
_GREP_WORD = re.compile(r"(?<![\w./-])(?:e?grep|fgrep)(?![\w-])")
_COMBINED_FLAGS = re.compile(r"^-(?P<body>[A-Za-z]{2,})$")
# env-assignment head: FOO=1 grep ... (value may be empty).
_ASSIGNMENT_HEAD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=[^=]*")
# Wrapper binaries whose next positional token is still the command, and
# control keywords after which a command position follows.
_WRAPPER_HEADS = frozenset(
    {"sudo", "env", "time", "command", "nohup", "nice", "exec"})
_CONTROL_HEADS = frozenset({"if", "while", "until", "do", "then"})
# Characters whose meaning a token-level re-quote cannot preserve (command
# substitution, globs, tilde expansion, brace expansion, heredoc bodies):
# a command containing any of them is passed through verbatim.
_UNSAFE_CHARS = frozenset("$`*?[~{")
# Short flags shared by grep and rg, kept as-is; grep-only letters that must
# disappear are in _DROP_CHARS.
_KEEP_CHARS = set("niclqvwxFHoxasP")
_DROP_CHARS = set("ErR")
# Operators that end one command and start the next. Note `;`: shlex's
# punctuation set had to include it explicitly (see _split_shell), and here
# it must sit AFTER multi-char operators so `&&`/`||`/`;&` pairs tokenize whole.
_OPERATORS = {"|", "|&", ">&", "&&", "||", ";", "&", ">", ">>", "<", "<<"}
# Redirection targets like 2>&1: never a pattern, never re-quoted as a word.
_REDIRECTION = re.compile(r"^\d*[<>]")


def _rewrite_combined_flag(tok: str) -> str:
    m = _COMBINED_FLAGS.match(tok)
    if m is None:
        return tok
    body = "".join(ch for ch in m.group("body") if ch not in _DROP_CHARS)
    return f"-{body}"


def _split_shell(command: str) -> list[str] | None:
    """shlex split that keeps operators (|, &&, >, ;) as separate tokens.
    Returns None when the command doesn't parse (unbalanced quotes, stray
    operators the lexer can't place)."""
    # `;` is NOT in shlex's default punctuation set, but it is the most common
    # command separator models emit (`cd x; grep foo f`) — pass it explicitly
    # so the command-position test at prefer_ripgrep sees a fresh start after it.
    lex = shlex.shlex(command, posix=True, punctuation_chars="();<>|&;")
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError:
        return None


def _requote(tokens: list[str]) -> str:
    """Rejoin tokens, quoting words but never shell operators."""
    return " ".join(t if t in _OPERATORS or _REDIRECTION.match(t)
                    else shlex.quote(t) for t in tokens)


def _merge_fd_redirects(tokens: list[str]) -> list[str]:
    """Stitch fd-redirections back together after punctuation_chars splitting.

    shlex(punctuation_chars=True) tokenizes ``2>&1`` as ``2``, ``>&``, ``1``
    (the digit is a word char, the operators are punctuation), which defeats
    _REDIRECTION's same-token guard and leaves the ``2`` to be emitted as an
    ordinary argument. Merge a lone digit word with the operator token that
    follows it (plus the target token for ``>&``/``>``) so the guard sees the
    combined form again. Only digits qualify: a plain word before ``>`` is a
    real argument, never a fd number.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isdigit() and i + 1 < len(tokens) and tokens[i + 1] in _OPERATORS:
            merged = tok + tokens[i + 1]
            i += 2
            if merged in ("2>", "1>", "2>&", "1>&") and i < len(tokens):
                merged += tokens[i]   # 2>&1 / 2>/dev/null target token
                i += 1
            out.append(merged)
        else:
            out.append(tok)
            i += 1
    return out


def _is_transparent_head(tok: str) -> bool:
    """A token that can precede a command without changing what it is:
    env assignments (FOO=1), wrapper binaries (sudo/env/time/command/nohup/
    nice/exec), and control keywords after which a command position follows
    (if/while/until/do/then)."""
    return bool(_ASSIGNMENT_HEAD.fullmatch(tok)) or tok in _WRAPPER_HEADS \
        or tok in _CONTROL_HEADS


def _command_position_grep(tokens: list[str], i: int) -> bool:
    """True when the token at index i starts a command position (string start
    or right after an operator) AND the grep-family word there follows only
    transparent heads."""
    if not (i == 0 or tokens[i - 1] in _OPERATORS):
        return False
    j = i
    while j < len(tokens):
        tok = tokens[j]
        if _is_transparent_head(tok):
            j += 1
            continue
        return bool(_GREP_WORD.fullmatch(tok))
    return False


def _command_head_len(tokens: list[str], i: int) -> int:
    """Number of transparent-head tokens between index i and the grep word."""
    j = i
    while j < len(tokens) and _is_transparent_head(tokens[j]):
        j += 1
    return j - i


def prefer_ripgrep(command: str) -> str:
    """Return `command` with grep-family calls rewritten to ripgrep.

    Returns the original string unchanged when rg is absent, the command
    doesn't shell-split, contains characters a token-level re-quote would
    corrupt ($(), globs, tilde, heredocs), or no grep-family word appears at
    command position (including after assignment/wrapper/control heads).
    """
    if not shutil.which("rg") or not _GREP_WORD.search(command):
        return command
    # A token-level split/re-quote cannot preserve these: pass through whole.
    # Heredocs are covered by `<<` in the same spirit (the body is data).
    if any(ch in command for ch in _UNSAFE_CHARS):
        return command
    tokens = _split_shell(command)
    if tokens is None:
        return command
    tokens = _merge_fd_redirects(tokens)
    out: list[str] = []
    pattern_done = False   # first non-flag operand of a rewritten grep seen
    in_rewritten = False   # tokens of the grep invocation being rewritten
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _OPERATORS or _REDIRECTION.match(tok):
            out.append(tok)
            in_rewritten = False  # operator ends the grep invocation
            i += 1
            continue
        if not in_rewritten and _command_position_grep(tokens, i):
            # Re-emit transparent heads (sudo/env/FOO=1/if/...) verbatim,
            # then `rg` in place of the grep word, and jump the loop index
            # past everything consumed (heads + the grep word itself).
            n = _command_head_len(tokens, i)
            out.extend(tokens[i:i + n])
            out.append("rg")
            i += n + 1
            in_rewritten = True
            pattern_done = False
            continue
        if in_rewritten:
            if tok.startswith("--include="):
                out.extend(("-g", tok.split("=", 1)[1]))
                i += 1
                continue
            if tok.startswith("--exclude-dir="):
                out.extend(("-g", "!" + tok.split("=", 1)[1]))
                i += 1
                continue
            if tok in ("-E", "-r", "-R", "-h"):
                i += 1  # dropped: rg defaults make them no-ops
                continue
            if _COMBINED_FLAGS.match(tok):
                tok = _rewrite_combined_flag(tok)
                if tok == "-":
                    i += 1
                    continue
            elif not tok.startswith("-") and not pattern_done \
                    and not _REDIRECTION.match(tok):
                # Pattern is the first non-flag operand: BRE `\|` -> ERE `|`.
                tok = tok.replace(r"\|", "|")
                pattern_done = True
            out.append(tok)
            i += 1
            continue
        # A token of a command that is not being rewritten (git grep, tail,
        # echo after a pipe, ...) passes through byte-for-byte.
        out.append(tok)
        i += 1
    return _requote(out)
