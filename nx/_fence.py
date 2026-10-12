"""The hard write fence (2.1): a deterministic, approval-independent check on
every path-carrying tool call, enforced in run_tool() *before* the tool body.

Why a fence when there is already a risk classifier and an approval prompt:
both are judgement. The classifier is a model call keyed off the approval
ACTION string (`_extract_action_path` only parses "edit/delete/write <path>"
prefixes — sed_edit/apply_patch had no path judgement at all), and YOLO
skips the prompt entirely. A committed `config -> ~/.ssh/authorized_keys`
symlink converts an approved in-project edit into an out-of-project write,
because `_atomic_write` deliberately follows symlinks (link-farm dotfile
management) and the judgement realpath'd at approval time while the write
resolves fresh (TOCTOU). A fence is a boundary, not an opinion: roots are
computed by realpath, candidate paths are realpath'd at the same moment the
body would resolve them, and the answer is in/out — no model, no prompt.

Downstream layers stay: the classifier shapes approval (low/medium/high),
the user confirms, allow-rules still short-circuit. The fence only decides
where a write may land, the one thing judgement must never be asked.

Env:
  NX_WRITE_ROOTS   os.pathsep-separated extra absolute roots, empty default
  NX_WRITE_FENCE   0 disables the check entirely (documented escape hatch)
"""
from __future__ import annotations

import os
import re
from typing import Any

from . import _state as st
from ._env import _env_int

# Tools whose primary write target is `path` (same list run_tool() already
# special-cases for undo snapshots and cache invalidation).
_TOOL_PATH_TOOLS = ("write_file", "edit_file", "multi_edit", "delete_file")

# Tools with extra write surfaces beyond `path`.
#   sed_edit: `path` is a glob — the body expands it against the real cwd,
#     so fence each matched file plus the literal (pre-expansion) arg.
#   apply_patch: `diff` headers name files relative to `path`; also fence
#     each extracted +++/--- target joined against the patch base.
#   run_bash: no `path` args, but shell is still a write surface — a model
#     routed a refused edit through `python3 - <<'EOF' ... open(p,'w') ... EOF`
#     in the wild. A full shell parser still isn't worth it, but the common
#     rewrite shapes are scannable: _bash_write_candidates() covers
#     redirections, tee, write-capable binaries, and interpreted-language
#     file-mutation calls. Deliberately heuristic — anything it can't see it
#     doesn't fence, so it stays a backstop behind the classifier/approval,
#     not a sandbox (P3-1). NX_BASH_FENCE=0 disables just this layer.
_BASH_FENCE_ENV = "NX_BASH_FENCE"

_SHELL_ONLY = "run_bash"

def _writable_roots() -> list[str]:
    """Realpath'd writable roots: the project root (git root or cwd), plus
    NX_WRITE_ROOTS extras, plus the session home (~/.nx or NX_HOME) so nx's
    own sinks (sessions, memories, images, overflow, agents) stay writable
    without every caller having to add them by hand."""
    roots = [os.path.realpath(getattr(st, "PROJECT_ROOT", None) or os.getcwd())]
    session_home = getattr(st, "SESSION_HOME", None) or os.path.expanduser(
        os.environ.get("NX_HOME", "~/.nx"))
    roots.append(os.path.realpath(session_home))
    raw = os.environ.get("NX_WRITE_ROOTS", "")
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        part = os.path.expanduser(part)
        if not os.path.isabs(part):
            part = os.path.join(roots[0], part)
        roots.append(os.path.realpath(part))
    return roots


def _under_any(path: str, roots: list[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except (OSError, ValueError):
            continue
    return False


# Committed-symlink targets that convert an approved in-project edit into an
# out-of-project write: the classic dotfile link farm, ~/.ssh authorized_keys
# chroots, shell rc files, and git hooks (self-executing on commit/checkout).
# The target being IN the denylist wins even when the link itself sits inside
# a root — _atomic_write writes THROUGH symlinks by design, so the real file
# is what gets modified and the real file is what gets judged.
_DENIED_TARGET_REALS = frozenset({
    os.path.realpath(os.path.expanduser("~/.ssh")),
    os.path.realpath(os.path.expanduser("~/.aws")),
    os.path.realpath(os.path.expanduser("~/.gnupg")),
    os.path.realpath(os.path.expanduser("~/.kube")),
    os.path.realpath(os.path.expanduser("~/.docker")),
    os.path.realpath(os.path.expanduser("~/.config/gcloud")),
    os.path.realpath(os.path.expanduser("~/.netrc")),
    os.path.realpath(os.path.expanduser("~/.gitconfig")),
})
_DENIED_TARGET_BASENAMES = {
    ".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile",
    ".zshenv", ".cshrc", ".tcshrc", ".gitconfig", ".gitignore_global",
    ".netrc", ".wgetrc", ".curlrc", ".ssh", ".aws", ".gnupg", ".kube",
    ".docker", ".npmrc", ".pypirc", ".netrc", ".hgrc", ".pgpass",
    "authorized_keys", "authorized_keys2", "config", "id_rsa", "id_ed25519",
    "id_ecdsa", "identity",
}
_DENIED_TARGET_SUFFIXES = (".pem", ".key",)


def _denied_target(path: str) -> bool:
    """True for real targets that stay fenced even from inside a root:
    credential dirs/files and the home shell rc files."""
    if path in _DENIED_TARGET_REALS:
        return True
    base = os.path.basename(path)
    low = base.lower()
    if low in _DENIED_TARGET_BASENAMES:
        return True
    if low.endswith(_DENIED_TARGET_SUFFIXES):
        return True
    return False


def _fence_candidates(name: str, args: dict[str, Any]) -> list[tuple[str, str]]:
    """(display, realpath) pairs every write surface a tool call could touch.
    Ordered list of (candidate path as written, realpath)."""
    cands: list[tuple[str, str]] = []

    def add(raw: Any) -> None:
        if not isinstance(raw, str) or not raw.strip():
            return
        cands.append((raw, os.path.realpath(os.path.expanduser(raw))))

    _diff = args.get("diff")
    if isinstance(_diff, str) and _diff.strip():
        # diff mode (merged edit_file, or the legacy apply_patch name): the
        # patch headers name the real targets, `path` is only the base dir.
        base = args.get("path") or "."
        add(base)
        import re as _re
        for m in _re.finditer(r"^\+\+\+ (?:b/)?(\S+)", _diff, _re.M):
            tgt = m.group(1)
            if tgt == "/dev/null":
                continue
            add(os.path.join(base, tgt))
    elif name in _TOOL_PATH_TOOLS:
        add(args.get("path"))
    elif name == "sed_edit":
        # The body globs `path` itself; fence the literal and every file the
        # glob would match (same _BINARY_EXTS skip is the body's business —
        # we fence before the body, so a broad glob is judged on what it
        # would touch, not on what the body later filtered out).
        import glob as _glob
        raw = args.get("path")
        if isinstance(raw, str) and raw.strip():
            add(raw)
            if _glob.has_magic(raw):
                try:
                    for m in sorted(_glob.glob(raw, recursive=True)):
                        add(m)
                    # A globlexists-but-no-file case is fenced via the literal
                    # above; no matches at all means nothing to write.
                except Exception:
                    pass
    return cands


def check_write_fence(name: str, args: dict[str, Any]) -> str:
    """The fence verdict for a tool call: "" = allow, else the ERROR refusal.

    Never raises, never prompts, never calls the model. Called from run_tool()
    for every tool with a structured write surface (the `path` tools, sed_edit,
    and any call carrying a `diff`). run_bash is deliberately NOT fenced here — no
    static analysis covers shell without either bypasses or false refusals;
    it keeps its classifier+approval story (see P3-1 for the real fix).
    """
    if not _env_int("NX_WRITE_FENCE", 1):
        return ""
    roots = _writable_roots()
    outside: list[str] = []
    denied: list[str] = []
    for raw, real in _fence_candidates(name, args):
        if _denied_target(real):
            denied.append(raw)
            continue
        if _under_any(real, roots):
            continue
        outside.append(raw)
    if outside:
        return (
            "ERROR: write fence: " + ", ".join(sorted(set(outside)))
            + " resolves outside every writable root (project root "
            + roots[0] + (", plus NX_WRITE_ROOTS" if len(roots) > 2 else "")
            + "; session home is always writable). No part of the call ran."
            " Write inside the project, add a root via NX_WRITE_ROOTS (colon-"
            "separated, ~ expands), or set NX_WRITE_FENCE=0 to disable the"
            " fence for this environment."
        )
    if denied:
        return (
            "ERROR: write fence: refusing to write through a path that lands in a"
            " protected location even though it may sit inside a writable root"
            " (committed symlink, mount point, or direct write): " + ", ".join(sorted(set(denied)))
            + ". nx writes THROUGH symlinks (link-farm dotfile management), so the"
            " link's target is what gets modified. Ask the user to perform this"
            " write themselves, or add the target directory via NX_WRITE_ROOTS"
            " if it is genuinely meant to be writable."
        )
    return ""


# --- the bash surface ---------------------------------------------------------

# Write-capable binaries scanned for a positional destination argument. Not
# exhaustive: unknown binaries and exotic flags are exactly the false
# negatives a backstop accepts.
_BASH_WRITE_BINS = frozenset((
    "tee", "cp", "mv", "install", "rsync", "ln", "truncate", "dd",
    "tar", "unzip", "wget", "curl", "git",
))

# Interpreted-language file-mutation calls, scanned as text over the whole
# command (here-doc bodies included — the incident shape). Only the
# destination of each is fenced: open()'s first argument, and for
# `Path(x).write_text(content)` the RECEIVER — the argument is the content,
# not the path (anchoring on the paren used to fence the content instead).
(_BASH_PY_OPEN_RE, _BASH_WRITE_CALL_RE) = (
    re.compile(r"""open\s*\(\s*([^,()]+?)\s*,\s*["'][wax+]"""),
    # The call site only — its receiver is scanned backwards from here (see
    # _bash_receiver). A regex that tried to capture the receiver either
    # backtracked catastrophically or, given quotes in its class, opened the
    # capture at the enclosing shell quote and mis-paired the literals inside
    # it (`"Path('out.txt')"` captured as `"Path(`).
    re.compile(r"""\.write_(?:text|bytes)\s*\("""),
)
# A quoted string literal inside a language capture is the only part of it we
# can resolve: Path("out.txt") -> out.txt.
_BASH_QUOTED_RE = re.compile(r"""["']([^"']+)["']""")
# What such a literal must look like to be believed as a destination: the
# receiver is scraped as text, so a quote can mis-pair and yield junk.
_BASH_PATHISH_RE = re.compile(r"[\w./~$@\[\]{}#+%@:-]+")
# Characters a receiver expression is made of (the backward identifier scan).
# Quotes are deliberately NOT among them.
_BASH_RECV_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "._/$@{}[]#%-+~*")
# A shell variable in a candidate path ($VAR, ${VAR}) means we cannot know
# where it lands: refuse rather than guess.
_BASH_VAR_RE = re.compile(r"\$\{?\w")

# Candidate source tag: an interpreted-language capture with no resolvable
# literal in it (open(p, 'w') yields the VARIABLE NAME "p", self.path yields
# an attribute chain) means the scanner saw a write it cannot locate, so it
# must refuse — the same rule as a shell variable. That reasoning does NOT
# apply to a bare word from a redirection or a binary's argv: `cp a b`,
# `tee -a notes`, `truncate -s 0 logfile` name ordinary relative paths, and
# refusing them was pure over-refusal. Candidates without this tag ("") are
# plain paths and get judged like paths.
_LANG_VAR = "lang-var"

# Here-doc intro: <<DELIM, <<-DELIM, <<'DELIM', <<"DELIM". `<<(?!<)` keeps a
# right-shift (`a << 2`) and a here-string (`<<<`) out; a genuine intro is
# additionally required to end its line in nothing-or-redirections (see
# _REDIR_TAIL_RE) before its body is blanked.
_HEREDOC_RE = re.compile(r"<<(?!<)(-?)[ \t]*([\"']?)(\w+)\2")
# What a here-doc intro line may legitimately end with: nothing, or redirects.
_REDIR_TAIL_RE = re.compile(r"(?:\d*(?:>>|<<|&>?|<|>)\s*\S*\s*)*")


def _strip_heredoc_bodies(command: str) -> str:
    """`command` with every here-doc body blanked out (intro lines kept — a
    redirection written after the intro IS a real redirection).

    Only the redirection / argv scans use this: a `>` inside a script body
    (`print('>' if a > b else '<')`, `sys.version > '3'`) is python, not a
    shell redirection, and reading it as one fenced operands as filenames.
    The interpreted-language scan runs over the FULL command — those bodies
    are the whole point. Unrecognized or unterminated intros are left alone.
    """
    out: list[str] = []
    pos = 0
    while pos < len(command):
        m = _HEREDOC_RE.search(command, pos)
        if not m:
            out.append(command[pos:])
            break
        nl = command.find("\n", m.end())
        rest = command[m.end():len(command) if nl == -1 else nl]
        if nl == -1 or not _REDIR_TAIL_RE.fullmatch(rest.strip()):
            # Not an intro (shift operator, quoted text, unterminated): the
            # line keeps its text and scanning continues after the match.
            out.append(command[pos:m.end()])
            pos = m.end()
            continue
        out.append(command[pos:nl])
        i = nl + 1
        term = m.group(3)
        dash = m.group(1) == "-"
        found = False
        while i < len(command):
            eol = command.find("\n", i)
            last = eol == -1
            eol = len(command) if last else eol
            line = command[i:eol]
            if (line.lstrip("\t") if dash else line) == term:
                found = True
                i = eol if last else eol + 1
                break
            i = eol + 1
        if not found:
            out.append(command[nl:i])
            pos = i
            continue
        # Preserve line structure so offsets after the body stay on their own
        # lines (a stray `#` can't comment out the rest of the command).
        out.append("\n" * command.count("\n", nl, i))
        pos = i
    return "".join(out)


def _bash_receiver(command: str, dot: int) -> str:
    """The expression a `.write_text(` / `.write_bytes(` call hangs off,
    scanned BACKWARDS from the dot: `Path("out.txt").write_text(c)` ->
    `Path("out.txt")`. Linear, and it stops at anything that isn't receiver
    syntax (unbalanced paren, stray quote) instead of guessing."""
    i = dot
    while i > 0 and command[i - 1] in " \t":
        i -= 1
    if i <= 0:
        return ""
    last = command[i - 1]
    if last in "\"'":
        start = command.rfind(last, 0, i - 1)
        return command[start:i] if start != -1 else ""
    if last == ")":
        depth, j = 0, i - 1
        while j >= 0:
            if command[j] == ")":
                depth += 1
            elif command[j] == "(":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        if j < 0:
            return ""
        k = j  # include the callee name / dotted chain in front of the paren
        while k > 0 and command[k - 1] in _BASH_RECV_CHARS:
            k -= 1
        return command[k:i]
    while i > 0 and command[i - 1] in _BASH_RECV_CHARS:
        i -= 1
    return command[i:dot]


def _lang_capture(text: str) -> tuple[str, str]:
    """Normalize an interpreted-language capture to a (path, source) pair: a
    quoted literal that looks like a path is a destination we can judge; a
    bare name / attribute chain / expression is a variable we cannot resolve
    (the caller refuses those rather than guessing they land inside)."""
    for m in _BASH_QUOTED_RE.finditer(text):
        lit = m.group(1).strip()
        if not lit or not _BASH_PATHISH_RE.fullmatch(lit):
            continue
        # `{...}` is an f-string hole, not a literal name: unresolvable.
        if "{" in lit or "}" in lit:
            return lit, _LANG_VAR
        return lit, ""
    return text.strip(), _LANG_VAR


def _bash_write_candidates(command: str) -> list[tuple[str, str]]:
    """(path, source) pairs a bash command may write to. Over-inclusive on the
    shapes it knows; unrecognized shapes (eval'd strings, subshell gymnastics,
    unknown binaries) yield nothing — backstop, not sandbox.

    `source` is "" for a plain path and _LANG_VAR for an interpreted-language
    capture whose destination is a variable (see _LANG_VAR).
    """
    cands: list[tuple[str, str]] = []

    def add(raw: str, source: str = "") -> None:
        raw = raw.strip()
        # a symmetric quote pair is shell quoting, not part of the name
        if len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1].strip()
        raw = raw.rstrip(";")
        if raw and (raw, source) not in cands:
            cands.append((raw, source))

    shell = _strip_heredoc_bodies(command)

    # 1. redirections: > file, >> file, 2> file, &> file, <> file.
    #    &1/&2 are fd-dup targets, not files. The lookbehind drops operator
    #    contexts (`a -> b`, `x => y`, the second `>` of a >=-style run) and
    #    here-doc bodies are already blanked, so a comparison inside a script
    #    body can't be read as a redirection. A quoted target is captured whole
    #    (`> "my file.txt"`) so spaces survive.
    for m in re.finditer(r"(?<![-=>])\d?>+\s*((?:[\"'][^\"']*[\"']|\S+))", shell):
        if not re.fullmatch(r"&[12]", m.group(1)):
            add(m.group(1))

    # 2. here-doc / inline interpreted-language write calls
    #    (python3 - <<'EOF' ... open(p,'w') ... EOF) — over the FULL command.
    for m in _BASH_PY_OPEN_RE.finditer(command):
        add(*_lang_capture(m.group(1)))
    for m in _BASH_WRITE_CALL_RE.finditer(command):
        recv = _bash_receiver(command, m.start())
        if recv:
            add(*_lang_capture(recv))

    # 3. write-capable binaries, over the here-doc-stripped command so a body
    #    line can't masquerade as argv. Segments are split on [;|&] so each is
    #    judged on its own argv rather than leaking into the next one.
    for seg in re.split(r"[;|&]", shell):
        toks = seg.split()
        if not toks:
            continue
        bin_ = toks[0].rsplit("/", 1)[-1]
        args_ = toks[1:]
        if bin_ == "dd":
            for tok in args_:
                if tok.startswith("of="):
                    add(tok[3:])
            continue
        if bin_ == "git":
            # `git config --file <p>` names its own target
            for i, tok in enumerate(args_):
                if tok in ("--file", "-f") and i + 1 < len(args_):
                    add(args_[i + 1])
            continue
        if bin_ not in _BASH_WRITE_BINS:
            continue
        # wget/curl -o / -O name their own target; the rest take the output
        # as the last non-flag positional in the common forms — the only
        # claim this scanner makes.
        if bin_ in ("wget", "curl"):
            for i, tok in enumerate(args_):
                if tok in ("-o", "-O", "--output", "--output-document") and i + 1 < len(args_):
                    add(args_[i + 1])
                # `-o-` (curl -o-) writes to stdout; `-oX` names X.
                elif tok.startswith("-o") and len(tok) > 2 and tok[2] != "-":
                    add(tok[2:])
            continue
        positional = [t for t in args_ if not t.startswith("-")]
        if positional:
            add(positional[-1])
    return cands


def check_bash_write_fence(command: str) -> str:
    """The fence verdict for a bash command: "" = allow, else the refusal.

    Backstop layer for run_bash (the structured tools are fenced exactly in
    check_write_fence; shell gets the heuristic scan). Never raises, never
    prompts, never calls the model. Called from run_tool() before run_bash's
    body, so the refusal means no part of the command ran.
    """
    if not _env_int("NX_WRITE_FENCE", 1) or not _env_int(_BASH_FENCE_ENV, 1):
        return ""
    roots = _writable_roots()
    outside: list[str] = []
    for raw, source in _bash_write_candidates(command):
        if raw in ("/dev/null", "/dev/stdout", "/dev/stderr"):
            continue
        # An interpreted-language destination we could not reduce to a
        # literal (open(p, 'w') yields "p"; self.path yields an attribute
        # chain) resolves to nothing predictable: refuse, the same rule as a
        # shell variable. Scoped to that source — a bare word named by a
        # redirection or a binary's argv (cp a b, tee -a notes) is an ordinary
        # relative path, judged below like one.
        if source == _LANG_VAR:
            outside.append(raw)
            continue
        expanded = os.path.expanduser(raw)
        if not os.path.isabs(expanded):
            # run_bash executes in the project root (or the tool's _cwd);
            # resolve relative destinations there, not against nx's cwd.
            expanded = os.path.join(roots[0], expanded)
        real = os.path.realpath(expanded)
        if _denied_target(real):
            return (
                "ERROR: write fence: bash command would write through "
                + raw + ", which lands in a protected location (credential"
                " dir / shell rc / key material). Ask the user to perform"
                " this write themselves."
            )
        if _BASH_VAR_RE.search(raw) or not _under_any(real, roots):
            outside.append(raw)
    if outside:
        return (
            "ERROR: write fence: bash command writes to "
            + ", ".join(sorted(set(outside)))
            + ", which resolves outside every writable root (project root "
            + roots[0] + (", plus NX_WRITE_ROOTS" if len(roots) > 2 else "")
            + "; session home is always writable). No part of the command ran."
            " Write inside the project, add a root via NX_WRITE_ROOTS (colon-"
            "separated, ~ expands), or set NX_WRITE_FENCE=0 (or"
            " NX_BASH_FENCE=0 for this layer only) to disable the fence."
        )
    return ""
