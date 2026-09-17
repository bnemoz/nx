"""run_bash grep->ripgrep upgrade (0.64.2).

When the model writes a grep-family command inside run_bash despite ripgrep
being installed, the harness rewrites it to rg before it runs. Covers token
mapping, flag translation, non-interference with non-grep commands, and the
rg-absent / unparseable bail-outs.
"""
import sys
import unittest
from unittest import mock

sys.path.insert(0, ".")

from nx._ripgrep import prefer_ripgrep


def _which(name):
    return "/usr/bin/rg" if name == "rg" else None


class TestGrepRewrite(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("shutil.which", _which)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_plain_grep(self):
        self.assertEqual(prefer_ripgrep("grep -i foo bar.md"), "rg -i foo bar.md")

    def test_recursive_and_extended_dropped(self):
        # -r (rg recurses by default) and -E (rg is always ERE) vanish;
        # the combined form -rn maps the same way.
        self.assertEqual(prefer_ripgrep("grep -rn foo src/"), "rg -n foo src/")
        self.assertEqual(prefer_ripgrep("grep -rn foo src/"),
                         prefer_ripgrep("grep -rn foo src/"))

    def test_bre_alternation_translated(self):
        self.assertEqual(prefer_ripgrep('grep -i "unlimited-ocr\\|ipv6\\|gai.conf" file'),
                         "rg -i 'unlimited-ocr|ipv6|gai.conf' file")

    def test_include_and_exclude_dir(self):
        self.assertEqual(
            prefer_ripgrep("grep -rn --include='*.py' --exclude-dir=.venv foo ."),
            "rg -n -g '*.py' -g '!.venv' foo .")

    def test_hidden_flag_dropped_filenames_kept(self):
        # -h (suppress filenames) is a no-op lie in rg; -H is already default.
        self.assertEqual(prefer_ripgrep("grep -h foo f"), "rg foo f")

    def test_combined_flag_partial_drop(self):
        self.assertEqual(prefer_ripgrep("grep -iEn foo f"), "rg -in foo f")

    def test_context_flags_kept(self):
        self.assertEqual(prefer_ripgrep("grep -C2 -n foo f"), "rg -C2 -n foo f")

    def test_egrep_fgrep(self):
        self.assertEqual(prefer_ripgrep("egrep 'a|b' f"), "rg 'a|b' f")
        self.assertEqual(prefer_ripgrep("fgrep -l x f"), "rg -l x f")

    def test_word_match_not_path_or_flag(self):
        # /usr/bin/grep stays untouched (lookbehind); a bare grep after a
        # word like `git` is NOT at command position, so `git grep` survives.
        self.assertEqual(prefer_ripgrep("/usr/bin/grep foo f"),
                         "/usr/bin/grep foo f")
        self.assertEqual(prefer_ripgrep("git grep foo"), "git grep foo")

    def test_grep_after_pipe_is_command_position(self):
        self.assertEqual(prefer_ripgrep("cat f | grep -i x"),
                         "cat f | rg -i x")

    def test_piped_grep_rewritten_pipeline_kept(self):
        result = prefer_ripgrep("grep -i x log.txt | head -5")
        self.assertTrue(result.startswith("rg -i x log.txt"))
        self.assertIn("head -5", result)

    def test_redirection_pattern_token_untouched(self):
        self.assertEqual(prefer_ripgrep("grep foo > out.txt"),
                         "rg foo > out.txt")

    def test_non_grep_command_untouched(self):
        self.assertEqual(prefer_ripgrep("ls -la && cat f.md"), "ls -la && cat f.md")

    def test_no_rg_installed_untouched(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(prefer_ripgrep("grep -rn foo src/"), "grep -rn foo src/")

    def test_unbalanced_quotes_untouched(self):
        self.assertEqual(prefer_ripgrep("grep -i 'unclosed f"), "grep -i 'unclosed f")

    def test_alternation_only_in_quoted_pattern_token(self):
        # shlex (posix) strips a backslash from unquoted tokens, so the BRE
        # `\\|` translation only applies to quoted patterns; it must also
        # never touch operands of neighbouring commands (echo's arg is a
        # path-like word, not a pattern).
        self.assertEqual(
            prefer_ripgrep('grep -i "a\\|b" f && echo "c\\|d"'),
            "rg -i 'a|b' f && echo 'c\\|d'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
