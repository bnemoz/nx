"""The commit guard's secret scan: added lines only, with a test-fixture escape.

Contract under test (`nx/_gitcommit.py::_scan_staged_diff`):

- a secret-shaped string on an ADDED line is refused;
- a secret-shaped string that is only CONTEXT (already-committed content) does
  NOT refuse — otherwise a legitimately committed fixture would make every
  future edit to that file uncommittable, blocking an edit while being unable to
  undo the thing it objects to (0.158.20 hit this on the redaction tests);
- a test-shaped path carrying an added `nx-allow-secret` marker is exempt, so the
  tests that verify secret redaction can contain token-shaped fixtures;
- the marker does NOT exempt a non-test path, and a test path WITHOUT the marker
  is still refused — the escape is explicit and auditable, never a blanket tests/
  carve-out;
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
    assert _scan_staged_diff(d), "a secret introduced by the diff must refuse"


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


def test_test_path_with_marker_is_exempt():
    d = _diff("tests/test_memory_tools.py",
              ["# nx-allow-secret: synthetic redaction fixtures",
               f'CASE = "{_OPENAI}"'])
    assert _scan_staged_diff(d) is None


def test_test_path_without_marker_is_refused():
    """The escape is opt-in per file, never automatic for tests/."""
    d = _diff("tests/test_something.py", [f'CASE = "{_OPENAI}"'])
    assert _scan_staged_diff(d)


def test_marker_in_application_code_does_not_exempt():
    """A marker in nx/ (or anywhere non-test) buys nothing."""
    d = _diff("nx/_memory_store.py",
              ["# nx-allow-secret (not a test path)", f'X = "{_OPENAI}"'])
    assert _scan_staged_diff(d), "the marker must not exempt application code"


def test_marker_must_be_on_an_added_line():
    """A pre-existing marker in context does not exempt newly added secrets."""
    d = _diff("tests/test_x.py", [f'CASE = "{_OPENAI}"'],
              context=["# nx-allow-secret: from a previous commit"])
    assert _scan_staged_diff(d)


def test_multi_file_diff_flags_only_the_offending_file():
    good = _diff("tests/test_ok.py",
                 ["# nx-allow-secret: fixtures", f'A = "{_OPENAI}"'])
    bad = _diff("nx/_risk.py", [f'B = "{_OPENAI}"'])
    assert _scan_staged_diff(good + "\n" + bad)
    assert _scan_staged_diff(good) is None


def test_diff_header_plus_plus_plus_is_not_an_added_line():
    """`+++ b/...` starts with '+' but is a header, not content."""
    d = ("--- a/nx/a.py\n+++ b/nx/a.py\n"
         "@@ -1 +1 @@\n"
         "+x = 1")
    assert _scan_staged_diff(d) is None


def test_empty_and_absent_diff():
    assert _scan_staged_diff("") is None
    assert _scan_staged_diff(None) is None
