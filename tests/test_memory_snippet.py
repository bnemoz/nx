"""Caps on `_memory_snippet`: per-line and total character bounds.

The snippet is display-only (ranking scores the raw body in `_rank_hit`), but
before 0.158.x a single long paragraph line — the wall of text a /memory save
or /memory optimize merge can produce — rode through unbounded into a search
result block, and up to 8 fused results ride together. These tests pin both
bounds, the visible truncation marker, backward compatibility for short notes,
and that ranking is unaffected by line length.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nx._memory_search import (  # noqa: E402
    MEMORY_SNIPPET_LINE_CHARS, MEMORY_SNIPPET_TOTAL_CHARS, _memory_snippet)


def test_short_note_snippet_is_byte_identical_to_pre_cap_behavior():
    content = "# Title\n\nFirst line\nSecond line\n\nThird line\nFourth line\nFifth\n"
    assert _memory_snippet(content, "q") == \
        "First line / Second line / Third line / Fourth line"


def test_long_single_line_is_capped_per_line_with_marker():
    long_line = "x" * (MEMORY_SNIPPET_LINE_CHARS + 500)
    out = _memory_snippet(f"# T\n\n{long_line}\n", "q")
    assert out == "x" * MEMORY_SNIPPET_LINE_CHARS + "…"
    assert len(out) == MEMORY_SNIPPET_LINE_CHARS + 1


def test_total_cap_bounds_the_joined_snippet():
    line = "y" * (MEMORY_SNIPPET_TOTAL_CHARS - 200)  # under per-line cap
    content = f"# T\n\n{line}\n{line}\n{line}\n{line}\n{line}\n"
    out = _memory_snippet(content, "q")
    assert len(out) <= MEMORY_SNIPPET_TOTAL_CHARS
    assert out.endswith("…")  # truncation is visible, never silent


def test_total_cap_applies_even_without_per_line_trigger():
    # Four lines each over half the total: joined, they exceed TOTAL.
    line = "z" * (MEMORY_SNIPPET_TOTAL_CHARS // 2 + 10)
    content = f"# T\n\n{line}\n{line}\n{line}\n{line}\n{line}\n"
    out = _memory_snippet(content, "q")
    assert len(out) <= MEMORY_SNIPPET_TOTAL_CHARS
    assert out.endswith("…")


def test_title_line_is_still_excluded():
    content = "# Secret Title Words\n\nbody line\n"
    out = _memory_snippet(content, "q")
    assert "Secret" not in out and out == "body line"


def test_max_lines_behavior_is_preserved():
    content = "# T\n\na\nb\nc\nd\ne\nf\n"
    assert _memory_snippet(content, "q") == "a / b / c / d"
    assert _memory_snippet(content, "q", max_lines=2) == "a / b"
    assert _memory_snippet(content, "q", max_lines=6) == "a / b / c / d / e / f"


def test_max_lines_0_returns_empty():
    assert _memory_snippet("# T\n\na\n", "q", max_lines=0) == ""


def test_ranking_unchanged_by_line_length():
    """Two notes identical except that one has a long line: the snippet caps
    must not reorder them. Rank via the raw content, as _rank_hit does."""
    from nx._memory_search import _rank_hit

    short = "# Alpha Widget\n\ngamma body text here\n"
    long_line = "gamma " + "g" * (MEMORY_SNIPPET_LINE_CHARS + 500)
    long = f"# Beta Widget\n\n{long_line}\n"
    assert _rank_hit(short, "gamma") == _rank_hit(long, "gamma")
    # And the snippets themselves differ in length without touching scores.
    assert len(_memory_snippet(long, "gamma")) <= MEMORY_SNIPPET_TOTAL_CHARS
