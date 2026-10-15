"""The commit guard's secret scan: added lines only, with a per-line test escape.

Contract under test (`nx/_gitcommit.py::_scan_staged_diff`):

- a secret-shaped string on an ADDED line is refused;
- a secret-shaped string that is only CONTEXT (already-committed content) does
  NOT refuse — otherwise a legitimately committed fixture would make every
  future edit to that file uncommittable, blocking an edit while being unable to
  undo the thing it objects to (0.158.20 hit this on the redaction tests);
- the `nx-allow-secret` exemption requires ALL of:
    * the marker on the SAME added line as the secret-shaped match (a per-line
      waiver, never file-wide);
    * a test-shaped path (so it cannot exempt application code);
    * the file already existing in the HEAD tree (`head_paths`) — a file
      created by this same commit cannot grant itself a waiver, which closes
      the self-granting bypass where a model authored tests/new_file.py with a
      marker plus a real credential in one commit;
- a test path WITHOUT the marker, or without HEAD history, is still refused —
  the escape is explicit and auditable, never a blanket tests/ carve-out;
- a removed line never trips it (deleting a secret is the good case).

Fixture literals below use the uppercase marker form and assembled prefixes so
this file's own diff does not trip the guard it is testing.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nx._gitcommit import _scan_staged_diff  # noqa: E402

# Assembled so THIS file's added lines contain no contiguous secret shape.
_OPENAI = "sk-" + "A" * 20
_AWS = "AKIA" + "B" * 16


def _diff(path: str, added: list[str], context: list[str] | None = None,
          removed: list[str] | None = None) -> str:
    """A minimal single-file unified diff body."""
    out = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}",
           "@@ -1,1 +1,1 @@"]
    out += [f" {ln}" for ln in (context or [])]
    out += [f"-{ln}" for ln in (removed or [])]
    out += [f"+{ln}" for ln in added]
    return "\n".join(out)


def test_added_secret_is_refused():
    d = _diff("nx/_config.py", [f'API_KEY = "{_OPENAI}"'])
    hit = _scan_staged_diff(d)
    assert hit, "a secret introduced by the diff must refuse"
    assert hit[0] == "nx/_config.py", "the refusal must name the offending path"


def test_added_aws_key_id_is_refused():
    assert _scan_staged_diff(_diff("nx/_config.py", [f"id={_AWS}"]))


def test_context_only_secret_does_not_refuse():
    """The 0.158.20 regression: an already-committed token in context must not
    block an unrelated edit to the same file."""
    d = _diff("tests/test_result_redaction.py", ["+x = 1  # unrelated edit"],
              context=[f'LEGACY = "{_OPENAI}"'])
    assert _scan_staged_diff(d) is None


def test_removed_secret_does_not_refuse():
    d = _diff("nx/_config.py", ["+API_KEY = os.environ['KEY']"],
              removed=[f'API_KEY = "{_OPENAI}"'])
    assert _scan_staged_diff(d) is None


def test_existing_test_path_with_same_line_marker_is_exempt():
    """The honest case: an already-committed test file, marker on the SAME
    added line as the token-shaped fixture."""
    d = _diff("tests/test_memory_tools.py",
              [f'CASE = "{_OPENAI}"  # nx-allow-secret'])
    assert _scan_staged_diff(d, {"tests/test_memory_tools.py"}) is None


def test_newly_added_test_file_with_marker_is_refused():
    """The self-granting bypass: a file CREATED by this commit cannot waive
    itself, no matter how many markers it carries."""
    d = _diff("tests/test_shiny_new.py",
              ["# nx-allow-secret: fixtures", f'CASE = "{_OPENAI}"'])
    assert _scan_staged_diff(d, set()), \
        "a new file must not be able to grant its own exemption"
    assert _scan_staged_diff(d, None), "absent HEAD info = conservative refuse"


def test_marker_on_a_different_line_is_refused():
    """Line-scoping bites: a marker elsewhere in the file waives nothing."""
    d = _diff("tests/test_memory_tools.py",
              ["# nx-allow-secret: fixtures below", f'CASE = "{_OPENAI}"'])
    assert _scan_staged_diff(d, {"tests/test_memory_tools.py"}), \
        "a file-wide marker must not exempt a marker-less secret line"


def test_test_path_without_marker_is_refused():
    """The escape is opt-in per line, never automatic for tests/."""
    d = _diff("tests/test_something.py", [f'CASE = "{_OPENAI}"'])
    assert _scan_staged_diff(d, {"tests/test_something.py"})


def test_marker_in_application_code_does_not_exempt():
    """A marker in nx/ (or anywhere non-test) buys nothing, even at HEAD."""
    d = _diff("nx/_memory_store.py",
              [f'X = "{_OPENAI}"  # nx-allow-secret'])
    assert _scan_staged_diff(d, {"nx/_memory_store.py"}), \
        "the marker must not exempt application code"


def test_marker_must_be_on_an_added_line():
    """A pre-existing marker in context does not exempt newly added secrets."""
    d = _diff("tests/test_x.py", [f'CASE = "{_OPENAI}"'],
              context=["# nx-allow-secret: from a previous commit"])
    assert _scan_staged_diff(d, {"tests/test_x.py"})


def test_multi_file_diff_flags_only_the_offending_file():
    good = _diff("tests/test_ok.py",
                 [f'A = "{_OPENAI}"  # nx-allow-secret'])
    bad = _diff("nx/_risk.py", [f'B = "{_OPENAI}"'])
    hit = _scan_staged_diff(good + "\n" + bad, {"tests/test_ok.py"})
    assert hit and hit[0] == "nx/_risk.py"
    assert _scan_staged_diff(good, {"tests/test_ok.py"}) is None


def test_mixed_marker_and_unmarked_lines_in_exempt_file():
    """Within a HEAD-known test file, only the marker-less lines are scanned."""
    d = _diff("tests/test_memory_tools.py",
              [f'GOOD = "{_OPENAI}"  # nx-allow-secret',
               f'BAD = "{_AWS}"'])
    assert _scan_staged_diff(d, {"tests/test_memory_tools.py"})


def test_diff_header_plus_plus_plus_is_not_an_added_line():
    """`+++ b/...` starts with '+' but is a header, not content."""
    d = ("--- a/nx/a.py\n+++ b/nx/a.py\n"
         "@@ -1 +1 @@\n"
         "+x = 1")
    assert _scan_staged_diff(d) is None


def test_empty_and_absent_diff():
    assert _scan_staged_diff("") is None
    assert _scan_staged_diff(None) is None
