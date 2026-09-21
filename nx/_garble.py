"""Garble guard: catch the "textually corrupted `new` payload" failure mode in
edit_file/multi_edit before it lands in a file.

Quantized local models garble long replacement text at a noticeable rate. The
corruption is rarely random — it takes a few recurring shapes, all of which
can still parse as Python (so _verify.py's ast gate sees nothing):

  1. duplicated lines — the model emits the same line twice in a row
     (re-decoding a line it just finished);
  2. repeated phrases — a 5-word gram restarts mid-stream ("...tuple ==
     short-circuiting... short-circuiting and never...");
  3. line-boundary splices — the tail of one line is re-emitted as the head
     of the next (the stream jumped back; the shared words are legit content
     in the FIRST line and a duplicate at the head of the SECOND);
  4. abandoned rewrites — an ellipsis mid-sentence with prose continuing
     after it: the model interrupted itself mid-construction and left the
     half-written clause behind ("...short-circuiting... we rely on identity
     here"). Detected, never auto-repaired.

Real chat example (glm-5.3-flash, an otherwise clean two-edit batch): the
docstring `new` contained a half-rewritten sentence spliced into the middle of
itself — "``==`` via tuple ``==`` short-circuiting... we rely on identity here"
— which the model then spent a whole extra turn rewriting.

The guard runs on the `new` payload once `old` has already matched (so it
never interferes with ordinary failed-match retries) and only flags when the
repeated text is NOT part of the surrounding file — a line the file itself
legitimately repeats is left alone. Both detection and repair demand real
locality: the two copies must be adjacent or overlapping (a restart, not two
separate mentions — loop bodies and parallel guard blocks legitimately repeat
a phrase a few words apart). Two outcomes:

  - repairable (duplicate line / near-duplicate phrase / splice): the
    duplicate copy is dropped, the edit proceeds, and the result says so;
  - not repairable: hard ERROR quoting the suspect text, so the model
    retries with clean text rather than persisting corruption.

NX_GARBLE_GUARD=0 disables the whole module. Never raises: detection is
best-effort and a false positive costs one retry, not a dead end (the ERROR
tells the model exactly what to resubmit).
"""
from __future__ import annotations

import hashlib
import re

from ._env import _env_int

# A "word" for gram purposes: letters/digits runs. Punctuation is noise for
# this check — stutters and splices repeat whole words, not punctuation.
_WORD_RE = re.compile(r"\w+")

# A duplicated line must be at least this long (chars, stripped) to count as a
# stutter — short repeated lines (")", "else:", scaffolding) are more often
# legitimate than garble.
_MIN_DUP_LINE = 12

# A duplicated line only counts when nothing separates the copies but a line
# break (or one blank line): model stutters are adjacent; one code line between
# them is loop scaffolding, ordinary code. Repairs delete a copy, so only
# true adjacency is safe to touch.

# Gram length for mid-line restart detection. The two occurrences must
# OVERLAP — the restart happens within the gram itself (gap <= _GRAM_N words),
# which is what makes the repair a safe single-period collapse. A phrase
# repeating a few words apart with no overlap is loop/parallel-block shape,
# not corruption.
_GRAM_N = 5

# Immediate word-echo (period-2 stutter, a token repeated thrice): a
# 5-gram check is blind to period 2 (odd/even gram positions never align), so
# a repeated word PAIR is checked separately. Four echo words minimum.
_STUTTER_ECHO_WORDS = 4
# ...and the two copies must be whitespace-joined on one line. Identifiers
# repeating across an assignment (`saved = m._confirm` then `m._confirm =
# ...`) or a line break (`return fast_path` above `fast_path = True`) are
# ordinary code, not echoes.
_ECHO_JOIN_RE = re.compile(r"[ \t]*\Z")

# Line-boundary overlap: max shared tail/head words considered, and the
# minimum that counts as a splice (2+ shared words is rarely coincidence).
_SPLICE_MAX = 8
_MIN_SPLICE = 2

# old/new-swap hint needs a `new` long enough that verbatim presence in the
# file isn't coincidence (shorter than the dup-line bar — one swapped line is
# the common case).
_MIN_SWAP_LEN = 8

# Abandoned-rewrite marker: an ellipsis with enough prose before AND after it
# (whole-payload context — the interruption often spans a line break), where
# the continuation resumes with a fresh subject (NOT a coordinating
# conjunction — "load the config... then start the server" is deliberate
# prose), is a mid-sentence self-interruption: the model abandoned a
# construction and left half of it behind ("...short-circuiting... we rely on
# identity here"). Trailing ellipses ("todo...") are spared. Detected, never
# auto-repaired — but see the insist-once escape below.
_REWRITE_BEFORE = 3
_REWRITE_AFTER = 2
# clause starters that read as deliberate continuation, not an abandoned one
_DELIBERATE_CONTINUATIONS = frozenset((
    "then", "and", "so", "or", "but", "which", "that", "if", "with", "for",
    "to", "when", "while", "before", "after", "until", "e", "g", "i", "ie",
))

# A detect-only ERROR must never deadlock a model that legitimately insists on
# its text: the first attempt is blocked, but re-submitting the byte-identical
# edit is taken as insistence and written as-is with a warning note. Session-
# scoped memo of the digests that have been blocked once already.
_INSISTED: set[str] = set()
_INSISTED_MAX = 256

# Repair loop cap: each pass recomputes signatures on the repaired text, so a
# multi-line stutter needs several passes. Bounded so a pathological payload
# can't spin.
_MAX_REPAIR_PASSES = 6

# A pipe-delimited markdown table row ("| a | b |" or "| a | b |"). Column
# separators make short tokens alternate legitimately, which trips the
# same-line gram/echo checks, and real tables can have an identical header
# and data row — so table rows get a much wider false-positive margin
# (0.76.0: two blocked edits on a pipe-delimited header example).
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# Adjacent identical pipe rows needed for a dup-line signature inside a
# table (two is an ordinary header+data row; three is still suspicious).
_PIPE_DUP_LINES = 3


def enabled() -> bool:
    """NX_GARBLE_GUARD is read per-call so the env var (or a /set write)
    takes effect without a restart."""
    return _env_int("NX_GARBLE_GUARD", 1) != 0


def garble_signature(new: str) -> list[str]:
    """Human-readable descriptions of suspected corruption in *new*. Empty
    list = clean. Each signature names the suspect text so the tool result can
    quote it back at the model and the repairers can find it again."""
    sigs: list[str] = []
    if not new.strip():
        return sigs
    lines = new.split("\n")

    # 1. near-adjacent duplicated lines. Inside a pipe table, two identical
    #    adjacent rows are ordinary (header + data row); require three.
    seen_at: dict[str, int] = {}
    seen_at_first: dict[str, int] = {}
    for i, ln in enumerate(lines):
        key = ln.strip()
        if len(key) < _MIN_DUP_LINE:
            continue
        j = seen_at.get(key)
        if j is not None and (i - j == 1
                              or (i - j == 2 and not lines[j + 1].strip())):
            if _PIPE_ROW_RE.match(ln) and i - j == 1:
                run = i - seen_at_first.get(key, j) + 1
                if run < _PIPE_DUP_LINES:
                    seen_at[key] = i
                    continue
            sigs.append(f"duplicated line {j + 1}->{i + 1}: {key[:70]!r}")
        if key not in seen_at_first:
            seen_at_first[key] = i
        seen_at[key] = i

    # 2. repeated 5-word gram within a small window (mid-line restarts);
    #    blind to period-2 stutters, which 2b catches. The restart is a
    #    same-line event — the stream re-emits text it just finished — so the
    #    copies must be whitespace-joined on ONE line. The same 5 words a few
    #    words apart across a line break is a loop/guard body repeating, not
    #    corruption (0.23.0: this false positive broke repeated `if not
    #    quiet:`-style blocks and forced models into contortions).
    # Pipe-table rows are exempt from the same-line repetition checks (2 and
    # 2b): alternating short tokens around "|" separators are the table's own
    # shape, not a stream restart. Run the checks on the non-table lines only.
    prose = "\n".join(ln for ln in lines if not _PIPE_ROW_RE.match(ln))
    words = _WORD_RE.findall(prose)
    spans = _word_spans(prose)
    if len(words) >= _GRAM_N:
        grams: dict[tuple, int] = {}
        for i in range(len(words) - _GRAM_N + 1):
            g = tuple(words[i:i + _GRAM_N])
            j = grams.get(g)
            if j is not None and i - j <= _GRAM_N \
                    and _ECHO_JOIN_RE.fullmatch(
                        prose[spans[j + _GRAM_N - 1].end():spans[i].start()]):
                sigs.append(f"repeated phrase: {' '.join(g)!r}")
            grams[g] = i

    # 2b. immediate word-echo: w[i:i+2] == w[i+2:i+4] (period-2 run)
    seen_bigram: set[str] = set()
    for i in range(len(words) - _STUTTER_ECHO_WORDS + 1):
        if words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
            gap = prose[spans[i + 1].end():spans[i + 2].start()]
            if not _ECHO_JOIN_RE.fullmatch(gap):
                continue  # copies not whitespace-joined on one line -> code
            if ("\n" in prose[spans[i].start():spans[i + 1].end()]
                    or "\n" in prose[spans[i + 2].start():spans[i + 3].end()]):
                continue  # a copy straddles a line break -> code, not echo
            bg = f"{words[i]} {words[i + 1]}"
            if bg not in seen_bigram:
                seen_bigram.add(bg)
                sigs.append(f"stuttered words: '{bg}'")

    # 3. line-boundary splice: tail words of one line == head words of the
    # next. Also requires the residue left on line 2 after trimming the
    # shared head to still read as code (word or opening bracket); when it
    # starts with an operator instead, the shared words are the second
    # line's own operand and there was no splice.
    for i in range(len(lines) - 1):
        k = _boundary_overlap(lines[i], lines[i + 1])
        if k and _splice_residue(lines[i + 1], k):
            shared = " ".join(_WORD_RE.findall(lines[i])[-k:])
            sigs.append(f"spliced line boundary at line {i + 2}: {shared!r}")

    # 4. abandoned rewrite: mid-sentence "..." with prose resuming after it
    for m in re.finditer(r"\.\.\.|\u2026", new):
        before = _WORD_RE.findall(new[:m.start()])
        after = _WORD_RE.findall(new[m.end():])
        if len(before) < _REWRITE_BEFORE or len(after) < _REWRITE_AFTER:
            continue
        if after[0].lower() in _DELIBERATE_CONTINUATIONS:
            continue  # "config... then start" reads as deliberate, not abandoned
        if not after[0][:1].islower():
            continue  # "... See the docs" reads as deliberate, not abandoned
        frag = (" ".join(before[-_REWRITE_BEFORE:]) + "... "
                + " ".join(after[:_REWRITE_AFTER]))
        lineno = new.count("\n", 0, m.start()) + 1
        sigs.append(f"abandoned rewrite at line {lineno}: '{frag}'")
    return sigs


def _splice_residue(second: str, k: int) -> bool:
    """True when trimming the *k* shared words off line 2's head leaves a
    plausible continuation: a word, an opening bracket, or nothing at all
    (line 2 was nothing but the echo). A residue starting with an operator
    (an assignment equals, a comma, a closing bracket) means the shared
    words were line 2's own operand — one line saving ``m._confirm`` above
    another reassigning it — so there was no splice."""
    spans = _word_spans(second)
    if len(spans) < k:
        return True
    rest = second[spans[k - 1].end():].lstrip(" \t")
    return not rest or bool(re.match(r"[\w(\[{\"']", rest))


def _boundary_overlap(a: str, b: str) -> int:
    """Largest k (<= _SPLICE_MAX) such that a's last k words equal b's first
    k words, or 0 when either line is too short or the overlap is trivial."""
    aw, bw = _WORD_RE.findall(a), _WORD_RE.findall(b)
    if len(aw) < _MIN_SPLICE or len(bw) < _MIN_SPLICE:
        return 0
    for k in range(min(_SPLICE_MAX, len(aw), len(bw)), _MIN_SPLICE - 1, -1):
        if aw[-k:] == bw[:k]:
            return k
    return 0


def _novel_fragment(fragment: str, src: str, old: str) -> bool:
    """True when *fragment* appears in neither the target file nor the edit's
    own `old` — i.e. the repetition can't be legitimate content echoing."""
    return fragment not in src and (not old or fragment not in old)


def _word_spans(text: str) -> list[re.Match]:
    return list(_WORD_RE.finditer(text))


def _drop_dup_line(new: str, sig: str, context: str) -> str | None:
    """Repair `duplicated line N->M` by deleting the second copy."""
    m = re.search(r"duplicated line (\d+)->(\d+)", sig)
    if m is None:
        return None  # signature went stale after an earlier repair
    j, i = int(m.group(1)), int(m.group(2))
    lines = new.split("\n")
    if i > len(lines) or lines[j - 1].strip() != lines[i - 1].strip():
        return None  # signature went stale after an earlier repair
    if not _novel_fragment(lines[i - 1].strip(), context, ""):
        return None
    del lines[i - 1]
    return "\n".join(lines)


def _collapse_periodic(new: str, pw: list[str], context: str) -> str | None:
    """Shared stutter collapse: find the repeated word-run *pw*, and when the
    two occurrences are adjacent/overlapping (gap <= run length — a periodic
    echo, not two separate mentions), keep exactly one period. None when the
    run is novel-vs-context vetoed, occurs once, or the gap is too large."""
    n = len(pw)
    if n < 2 or not _novel_fragment(" ".join(pw), context, ""):
        return None
    words = _WORD_RE.findall(new)
    pos = [i for i in range(len(words) - n + 1) if words[i:i + n] == pw]
    if len(pos) < 2:
        return None
    g = pos[1] - pos[0]
    if g > n:
        # Two well-separated mentions of the same phrase — could be legit
        # prose repetition; never collapse (caller errors instead).
        return None
    spans = _word_spans(new)
    # extend while the run stays periodic, then keep exactly one period
    e = pos[1]
    while e + g <= len(words) and words[e:e + g] == words[pos[0]:pos[0] + g]:
        e += g
    keep_end = spans[pos[0] + g - 1].end()
    tail_at = spans[e - 1].end()
    return new[:keep_end] + new[tail_at:]


def _dedup_phrase(new: str, sig: str, context: str) -> str | None:
    """Repair `repeated phrase` by collapsing the periodic echo to one copy
    (the first is the one the model meant to keep)."""
    m = re.search(r"repeated phrase: '(.*)'$", sig)
    return None if m is None else _collapse_periodic(new, _WORD_RE.findall(m.group(1)), context)


def _destutter(new: str, sig: str, context: str) -> str | None:
    """Repair `stuttered words` (period-2 echo, e.g. hyphenated tokens the
    5-gram check can't see) the same way: keep one period."""
    m = re.search(r"stuttered words: '(.*)'$", sig)
    return None if m is None else _collapse_periodic(new, _WORD_RE.findall(m.group(1)), context)


def _desplice(new: str, sig: str, context: str) -> str | None:
    """Repair `spliced line boundary at line N`: the shared words are legit
    content in line N-1 (that's where the stream was) and a duplicate at the
    head of line N — trim them off line N's head."""
    m = re.search(r"spliced line boundary at line (\d+): '(.*)'$", sig)
    if m is None:
        return None  # signature went stale after an earlier repair
    i = int(m.group(1)) - 1  # index of the second line
    if not _novel_fragment(m.group(2), context, ""):
        return None
    lines = new.split("\n")
    a, b = lines[i - 1], lines[i]
    k = _boundary_overlap(a, b)
    if not k:
        return None
    bw = _WORD_RE.findall(b)
    if k == len(bw):
        # the second line is nothing but the duplicated words — drop the line
        return "\n".join(lines[:i] + lines[i + 1:])
    spans = _word_spans(b)
    cut = spans[k - 1].end()
    fixed_b = b[cut:].lstrip(" \t")
    return "\n".join(lines[:i] + [fixed_b] + lines[i + 1:])


_REPAIRERS = (
    ("duplicated line", _drop_dup_line),
    ("repeated phrase", _dedup_phrase),
    ("stuttered words", _destutter),
    ("spliced line boundary", _desplice),
)


def realign_new(new: str, sigs: list[str], context: str) -> str:
    """One best-effort repair pass over *new* given *sigs* from
    garble_signature(). Returns the repaired text (or *new* unchanged when no
    signature could be repaired — the caller treats no-progress as failure).
    Stale signatures (a line/phrase an earlier repair in the same pass already
    removed) are skipped rather than failing the batch — garble_check()
    recomputes signatures after each pass, which settles cascades.
    *context* is the text the edit targets (plus the edit's `old`) — the veto
    that keeps repairs from deleting content the file itself repeats."""
    work = new
    for sig in sigs:
        for prefix, repair in _REPAIRERS:
            if sig.startswith(prefix):
                fixed = repair(work, sig, context)
                if fixed is not None:
                    work = fixed
                break
    return work


def _sig_fragment_novel(sig: str, new: str, src: str, old: str) -> bool:
    """The repeated fragment named by *sig*, for garble_check()'s novelty
    veto: line dups test the duplicated line, grams the phrase, splices the
    shared words."""
    m = re.search(r"duplicated line \d+->(\d+)", sig)
    if m:
        return _novel_fragment(new.split("\n")[int(m.group(1)) - 1].strip(), src, old)
    m = re.search(r"repeated phrase: '(.*)'$", sig)
    if m:
        return _novel_fragment(m.group(1), src, old)
    m = re.search(r"stuttered words: '(.*)'$", sig)
    if m:
        return _novel_fragment(m.group(1), src, old)
    m = re.search(r"spliced line boundary at line \d+: '(.*)'$", sig)
    if m:
        return _novel_fragment(m.group(1), src, old)
    m = re.search(r"abandoned rewrite at line \d+: '(.*)'$", sig)
    if m:
        # the fragment embeds a literal "..."; test the words around it
        return _novel_fragment(m.group(1).replace("...", ""), src, old)
    return False


def swapped_args(new: str, src: str, old: str) -> str | None:
    """`old`/`new` filled the wrong way round: `new` is already verbatim file
    content while `old` matches nothing. Returns a resubmit-hint ERROR line,
    or None when the shape doesn't fit. Only meaningful where `old` failed to
    match (a matching `old` proves the arguments aren't swapped)."""
    if not enabled() or not old or not new:
        return None
    if len(new.strip()) < _MIN_SWAP_LEN:
        return None
    if new.strip() in src and old.strip() not in src:
        return ("ERROR: `new` already exists verbatim in the file but `old` "
                "matches nothing — old/new look swapped; resubmit with the "
                "arguments the right way round")
    return None


def garble_check(new: str, src: str, old: str = "",
                 path: str = "") -> tuple[str, str | None, str]:
    """The one-call API for the edit tools, run on `new` once `old` has
    matched. Returns (status, message, text):

      ("clean", None, new)             — proceed with `new` unchanged
      ("repaired", note, repaired)     — proceed with `repaired`, append note
      ("error", error_line, new)       — do NOT write; return the error

    An "error" is not a dead end: resubmitting the byte-identical edit counts
    as insistence and is written as-is with a warning note, so a false
    positive costs the model one retry, never a loop."""
    if not enabled() or not new.strip():
        return "clean", None, new
    key = hashlib.sha1(
        f"{path}\x00{old}\x00{new}".encode("utf-8", "replace")).hexdigest()
    work = new
    first_sigs: list[str] | None = None
    for _ in range(_MAX_REPAIR_PASSES):
        sigs = [s for s in garble_signature(work)
                if _sig_fragment_novel(s, work, src, old)]
        if not sigs:
            break
        if first_sigs is None:
            first_sigs = sigs
        fixed = realign_new(work, sigs, src + "\n" + (old or ""))
        if fixed == work:
            quoted = "; ".join(sigs[:2])
            if key in _INSISTED:
                return ("repaired",
                        "(garble guard: the same suspect `new` was resubmitted "
                        "— written as-is; " + quoted + ")", new)
            if len(_INSISTED) >= _INSISTED_MAX:
                _INSISTED.clear()
            _INSISTED.add(key)
            return ("error",
                    "ERROR: `new` looks textually corrupted (" + quoted + ") — "
                    "nothing was written. Resubmit the edit with clean "
                    "replacement text (re-dictate the block rather than "
                    "patching it).",
                    new)
        work = fixed
    if first_sigs is None:
        return "clean", None, new
    quoted = "; ".join(first_sigs[:2])
    if old and work == old:
        # auto-repair erased the edit: the only "corruption" was a stutter
        # adjacent to `old`, and collapsing it left `new` == `old` — writing it
        # would be a silent no-op (field bug fixed in 0.37.3: the model then resubmits
        # and gets a misleading "no change" error). Refuse with guidance; a
        # byte-identical resubmission is insistence and writes `new` as-is.
        if key in _INSISTED:
            return ("repaired",
                    "(garble guard: repairing would make `new` identical to "
                    "`old`; the same suspect `new` was resubmitted — written "
                    "as-is; " + quoted + ")", new)
        if len(_INSISTED) >= _INSISTED_MAX:
            _INSISTED.clear()
        _INSISTED.add(key)
        return ("error",
                "ERROR: the suspect text is only a stutter adjacent to `old` — "
                "repairing it would leave `new` identical to `old` (a no-op "
                "edit), so nothing was written. If `new` is really what you "
                "want, resubmit the byte-identical edit and it will be written "
                "as-is.", new)
    note = ("(garble guard: `new` contained " + quoted
            + " — repaired automatically; check the diff)")
    return "repaired", note, work
