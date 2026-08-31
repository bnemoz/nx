#!/usr/bin/env python3
"""Tests for the garble guard (nx/_garble.py) and its wiring into
edit_file/multi_edit.

Covers the three corruption signatures (duplicated line, repeated phrase,
spliced line boundary) plus old/new-swapped detection, the novelty veto
(file already contains the repetition -> leave it alone), the adjacency
guard (far-apart legit repetition -> leave it alone), repair vs error
outcomes, and end-to-end behavior through edit_file/multi_edit with
monkeypatched _confirm.
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nx-gguard-")
os.environ["NX_ENV_FILE"] = "/dev/null"
os.environ["NX_SESSIONS_DIR"] = _tmp
os.environ["NX_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nx as m  # noqa: E402
from nx import _garble  # noqa: E402


def _write(name, text):
    path = os.path.join(_tmp, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def _allow():
    m._confirm = lambda action, preview=None: True


# --- signature detection -----------------------------------------------------

def test_detects_duplicated_line():
    sigs = _garble.garble_signature(
        "def render(self):\n    return self._snap\n    return self._snap\n")
    assert any(s.startswith("duplicated line 2->3") for s in sigs), sigs


def test_detects_repeated_phrase():
    sigs = _garble.garble_signature(
        "so tuple == hits the fast-path fast-path fast-path and never deep-compares")
    # words are \w+ runs, so hyphenated tokens split; period-2 echo is caught
    # by the bigram check, not the 5-gram one
    assert "stuttered words: 'fast path'" in sigs, sigs


def test_detects_spliced_line_boundary():
    # tail words of line 1 re-emitted at the head of line 2
    new = ("Kept cheap: scalars compare by value and the dicts are\n"
           "the dicts are cached objects their setters replace wholesale\n")
    sigs = _garble.garble_signature(new)
    assert any(s.startswith("spliced line boundary at line 2: 'the dicts are'")
               for s in sigs), sigs


def test_short_repeated_lines_are_ignored():
    # short scaffolding lines repeat legitimately all the time
    assert _garble.garble_signature("x = 1\n)\n)\nelse:\n    pass\n") == []


def test_far_apart_repetition_is_not_garble():
    # same decorator 40 lines apart — legitimate, not a stutter
    filler = "\n".join(f"# line {i} padding to separate the two blocks" for i in range(30))
    text = "@app.route('/api/v1/items', methods=['POST'])\n" + filler + \
        "\n@app.route('/api/v1/items', methods=['POST'])\n"
    assert not any(s.startswith("duplicated line")
                   for s in _garble.garble_signature(text)), "adjacency guard failed"


# --- novelty veto ------------------------------------------------------------

def test_novelty_veto_spares_repetition_the_file_already_has():
    src = "keep_a = compute(a)\nkeep_b = compute(a)\n"
    new = "one = f(a)\ntwo = f(a)\n"
    status, _msg, _text = _garble.garble_check(new, src, old="old-marker")
    assert status == "clean"


# --- repair ------------------------------------------------------------------

def test_repair_drops_duplicated_line():
    new = "alpha one\nbravo two with words\nbravo two with words\ncharlie three\n"
    status, msg, repaired = _garble.garble_check(new, "unrelated source", "")
    assert status == "repaired"
    assert repaired == "alpha one\nbravo two with words\ncharlie three\n"
    assert "garble guard" in msg


def test_repair_dedups_repeated_phrase():
    new = "the tuple compare hits the fast path fast path fast path and never recurses\n"
    status, _msg, repaired = _garble.garble_check(new, "unrelated source", "")
    assert status == "repaired"
    assert repaired == "the tuple compare hits the fast path and never recurses\n"


def test_repair_desplices_line_boundary():
    new = ("Kept cheap: scalars compare by value and the dicts are\n"
           "the dicts are cached objects their setters replace wholesale\n")
    status, _msg, repaired = _garble.garble_check(new, "unrelated source", "")
    assert status == "repaired"
    # shared words kept once, on the first line
    assert repaired == ("Kept cheap: scalars compare by value and the dicts are\n"
                        "cached objects their setters replace wholesale\n")


def test_clean_new_is_untouched():
    new = "def f():\n    return 42\n"
    status, msg, text = _garble.garble_check(new, "anything", "")
    assert (status, msg, text) == ("clean", None, new)


def test_detects_abandoned_rewrite_mid_sentence():
    # the exact failure from the real chat: half a rewritten clause left
    # behind mid-sentence, prose resuming lowercase after the "..."
    sigs = _garble.garble_signature(
        "they are wholesale-replaced by their setters, so ``is`` via tuple\n"
        "``==`` short-circuiting... we rely on identity here).")
    assert any(s.startswith("abandoned rewrite") for s in sigs), sigs


def test_deliberate_ellipsis_is_not_flagged():
    assert _garble.garble_signature("load the config... then start the server\n") == []
    assert _garble.garble_signature("see the docs... The details matter\n") == []


def test_insist_once_writes_after_identical_resubmit():
    _allow()
    path = _write("g8.py", "def f():\n    return 1\n")
    suspect = "def f():\n    return 2  # pause it... it resumes on the next tick\n"
    first = m.edit_file(path, old="def f():\n    return 1\n", new=suspect)
    assert first.startswith("ERROR") and "textually corrupted" in first, first
    second = m.edit_file(path, old="def f():\n    return 1\n", new=suspect)
    assert second.startswith("edited") and "written as-is" in second, second
    with open(path) as fh:
        assert fh.read() == suspect


def test_different_payload_not_affected_by_insist_memo():
    _allow()
    path = _write("g9.py", "def f():\n    return 1\n")
    a = "def f():\n    return 2  # pause it... it resumes on the next tick\n"
    b = "def f():\n    return 3  # stop it... the timer resets on write\n"
    assert m.edit_file(path, old="def f():\n    return 1\n", new=a).startswith("ERROR")
    assert m.edit_file(path, old="def f():\n    return 1\n", new=b).startswith("ERROR")


# --- old/new swap ------------------------------------------------------------

def test_swapped_args_detected():
    src = "TARGET = 1\nOTHER = 2\n"
    err = _garble.swapped_args(new="TARGET = 1\n", src=src, old="SOMETHING ELSE ENTIRELY")
    assert err is not None and err.startswith("ERROR") and "swapped" in err


def test_swapped_args_not_flagged_when_old_matches():
    assert _garble.swapped_args(new="x = 2\n", src="x = 1\n", old="x = 1\n") is None


# --- end-to-end through the edit tools ---------------------------------------

def test_edit_file_repairs_garbled_new():
    _allow()
    path = _write("g1.py", "def f():\n    return 1\n")
    garbled = "def f():\n    return 2\n    return 2\n"
    res = m.edit_file(path, old="def f():\n    return 1\n", new=garbled)
    assert res.startswith("edited")
    assert "garble guard" in res
    with open(path) as fh:
        assert fh.read() == "def f():\n    return 2\n"


def test_edit_file_errors_on_unrepairable_splice_with_conflicting_repeat():
    _allow()
    path = _write("g2.py", "def f():\n    return 1\n")
    # two independent signatures whose repairs can't both hold: a duplicated
    # line AND a splice sharing text with it -> realign fails -> hard ERROR
    garbled = ("alpha beta gamma delta epsilon\n"
               "alpha beta gamma delta epsilon\n"
               "zeta eta alpha beta gamma delta epsilon\n")
    res = m.edit_file(path, old="def f():\n    return 1\n", new=garbled)
    assert res.startswith("ERROR") and "textually corrupted" in res
    with open(path) as fh:
        assert fh.read() == "def f():\n    return 1\n", "nothing must be written"


def test_multi_edit_repairs_one_edit_and_reports_once():
    _allow()
    path = _write("g3.py", "A = 1\nB = 2\n")
    res = m.multi_edit(path, [
        {"old": "A = 1", "new": "A = 10  # set once set once set once set once set once"},
        {"old": "B = 2", "new": "B = 20"},
    ])
    assert res.startswith(f"edited {path} (2 edits applied)"), res
    assert res.count("garble guard") == 1
    with open(path) as fh:
        src = fh.read()
    assert src == "A = 10  # set once\nB = 20\n"


def test_multi_edit_garbled_edit_rolls_back_batch():
    _allow()
    path = _write("g4.py", "A = 1\nB = 2\n")
    # a 5-word phrase echoing twice with a gap no repairer may bridge —
    # nothing was written
    garbled = ("B = 20 alpha beta gamma delta epsilon and then more words\n"
               "alpha beta gamma delta epsilon follow here\n")
    res = m.multi_edit(path, [
        {"old": "A = 1", "new": "A = 10"},
        {"old": "B = 2", "new": garbled},
    ])
    assert res.startswith("ERROR") and "edit #2" in res, res
    with open(path) as fh:
        assert fh.read() == "A = 1\nB = 2\n", "atomic rollback must hold"


def test_clean_edits_unchanged_no_note():
    _allow()
    path = _write("g5.py", "A = 1\nB = 2\n")
    res = m.multi_edit(path, [{"old": "A = 1", "new": "A = 11"},
                              {"old": "B = 2", "new": "B = 22"}])
    assert "garble" not in res
    with open(path) as fh:
        assert fh.read() == "A = 11\nB = 22\n"


def test_edit_file_swap_hint_when_old_missing_and_new_is_file_text():
    _allow()
    path = _write("g6.py", "def real():\n    return 7\n")
    res = m.edit_file(path, old="def absent():\n    return 0\n",
                      new="def real():\n    return 7\n")
    assert res.startswith("ERROR") and "swapped" in res


def test_env_kill_switch():
    _allow()
    path = _write("g7.py", "A = 1\n")
    old_env = os.environ.get("NX_GARBLE_GUARD")
    os.environ["NX_GARBLE_GUARD"] = "0"
    try:
        res = m.edit_file(path, old="A = 1",
                          new="A = 2\nA = 2  # duplicated deliberately duplicated here")
        assert res.startswith("edited") and "garble" not in res
        # written verbatim, duplicate included
        with open(path) as fh:
            assert fh.read() == "A = 2\nA = 2  # duplicated deliberately duplicated here\n"
    finally:
        if old_env is None:
            os.environ.pop("NX_GARBLE_GUARD", None)
        else:
            os.environ["NX_GARBLE_GUARD"] = old_env
