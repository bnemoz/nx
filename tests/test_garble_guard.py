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


# --- pipe-delimited markdown table rows (0.76.0 regression) -------------------
# Alternating short tokens around "|" separators and an identical header+data
# row are a table's own shape, not stream corruption.

def test_pipe_table_rows_are_not_garble():
    rows = "| Name | Type | Description | Name | Type | Description |"
    assert _garble.garble_signature(rows) == []
    assert _garble.garble_signature(
        "| git commit | git commit | git commit | git commit |") == []
    assert _garble.garble_signature("| alpha | alpha | alpha | alpha |") == []


def test_pipe_table_header_plus_data_row_is_not_garble():
    header = "| Argument | Required | Description |"
    assert _garble.garble_signature(header + "\n" + header) == []


def test_three_identical_pipe_rows_still_flagged():
    row = "| one | two | three | four | five |"
    sigs = _garble.garble_signature("\n".join([row] * 3))
    assert any(s.startswith("duplicated line") for s in sigs), sigs


def test_garble_outside_table_rows_still_detected():
    sigs = _garble.garble_signature(
        "| Name | Type |\nso tuple == hits the fast-path fast-path fast-path and")
    assert "stuttered words: 'fast path'" in sigs, sigs


def test_far_apart_repetition_is_not_garble():
    # same decorator 40 lines apart — legitimate, not a stutter
    filler = "\n".join(f"# line {i} padding to separate the two blocks" for i in range(30))
    text = "@app.route('/api/v1/items', methods=['POST'])\n" + filler + \
        "\n@app.route('/api/v1/items', methods=['POST'])\n"
    assert not any(s.startswith("duplicated line")
                   for s in _garble.garble_signature(text)), "adjacency guard failed"


# --- loop/guard bodies legitimately repeat (0.23.0 regression) ---------------
# (fixtures build the repeated blocks at run time so this test module never
# itself carries a repeated-phrase shape)

def test_loop_body_phrase_repeat_is_not_garble():
    # the field-reported FP: parallel guard blocks repeat a phrase a few words
    # apart with NO overlap - loop shape, not a mid-stream restart. Must not
    # be flagged, repaired, or errored.
    guard = "    if not quiet:\n"
    new = (guard
           + '        print(f"kept {len(kept)}")\n'
           + guard
           + '        print(f"dropped {len(dropped)}")\n')
    assert _garble.garble_signature(new) == [], _garble.garble_signature(new)
    status, msg, out = _garble.garble_check(new, "def f():\n    pass\n", "")
    assert status == "clean" and msg is None and out == new, (status, msg)


def test_identical_loop_body_is_not_garble():
    # harsher variant: two adjacent loop bodies word-for-word the same (a
    # 5-gram echo across the line break) - still ordinary code
    body = "        for k in partners:\n            flush(k)\n"
    new = body + body
    assert _garble.garble_signature(new) == [], _garble.garble_signature(new)
    assert _garble.garble_check(new, "def f():\n    pass\n", "")[0] == "clean"


def test_code_line_between_dup_lines_is_not_garble():
    # one real line between two identical loop headers is scaffolding,
    # not a stutter
    body = "        for k in partners:\n            flush(k)\n"
    code = body + body.replace("flush", "reap")
    assert _garble.garble_signature(code) == [], _garble.garble_signature(code)


def test_blank_line_between_dup_lines_still_repaired():
    # a blank line between the copies is still a stutter (a paragraph
    # re-decoded), and collapsing it is safe
    line = "bravo two with words"
    new = "alpha one\n" + line + "\n\n" + line + "\ncharlie three\n"
    status, msg, repaired = _garble.garble_check(new, "unrelated source", "")
    assert status == "repaired", (status, msg)
    assert repaired == "alpha one\n" + line + "\n\ncharlie three\n"


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


def test_repeated_phrase_errors_instead_of_repairing():
    # 0.103.15: phrase/stutter/splice hits are heuristic guesses; a wrong
    # auto-repair silently deletes meant text (and poisoned trust in the whole
    # edit, forcing a re-diff against git). Only an exact duplicated line
    # auto-repairs; the rest take the error/insist path.
    new = "the tuple compare hits the fast path fast path fast path and never recurses\n"
    status, msg, text = _garble.garble_check(new, "unrelated source", "")
    assert status == "error" and "textually corrupted" in msg and text == new


def test_splice_errors_instead_of_repairing():
    new = ("Kept cheap: scalars compare by value and the dicts are\n"
           "the dicts are cached objects their setters replace wholesale\n")
    status, msg, text = _garble.garble_check(new, "unrelated source", "")
    assert status == "error" and "spliced line boundary" in msg and text == new


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

def test_edit_file_repairs_duplicated_line_new():
    _allow()
    path = _write("g1.py", "def f():\n    return 1\n")
    dup = "    return self._snap"
    garbled = "def f():\n" + dup + "\n" + dup + "\n"
    res = m.edit_file(path, old="def f():\n    return 1\n", new=garbled)
    assert res.startswith("edited")
    assert "garble guard" in res
    with open(path) as fh:
        assert fh.read() == "def f():\n" + dup + "\n"


def test_edit_file_errors_on_unrepairable_abandoned_rewrite():
    _allow()
    path = _write("g2.py", "def f():\n    return 1\n")
    # a detect-only signature (abandoned rewrite - no repairer exists) is the
    # hard-ERROR path: nothing may be written. Built indirectly so this test
    # module itself never carries the repeated-phrase shape.
    ell = "." * 3
    garbled = ("    return 2  # tuple short-circuiting" + ell
               + " we rely on identity here\n")
    res = m.edit_file(path, old="def f():\n    return 1\n", new=garbled)
    assert res.startswith("ERROR") and "textually corrupted" in res
    with open(path) as fh:
        assert fh.read() == "def f():\n    return 1\n", "nothing must be written"


def test_multi_edit_repairs_dup_line_and_reports_once():
    _allow()
    path = _write("g3.py", "A = 1\nB = 2\n")
    line = "A = 10  # cached marker note"
    res = m.multi_edit(path, [
        {"old": "A = 1", "new": line + "\n" + line},
        {"old": "B = 2", "new": "B = 20"},
    ])
    assert res.startswith(f"edited {path} (2 edits applied)"), res
    assert res.count("garble guard") == 1
    with open(path) as fh:
        src = fh.read()
    assert src == "A = 10  # cached marker note\nB = 20\n"


def test_multi_edit_garbled_edit_rolls_back_batch():
    _allow()
    path = _write("g4.py", "A = 1\nB = 2\n")
    # a detect-only signature on edit #2 (abandoned rewrite, no repairer)
    # -> hard ERROR, and the whole batch rolls back - nothing was written
    ell = "." * 3
    garbled = ("B = 20  # the timer pauses" + ell
               + " the timer resets on write\n")
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


# --- ordinary code shape must never be flagged (0.16.0 regression) ------------

def test_attribute_saved_then_reassigned_is_not_garble():
    # the field-reported misfire: two legit adjacent tokens separated by an
    # equals on the next line; the guard must not flag, repair, or error
    code = "        saved = m._confirm\n        m._confirm = lambda *a, **k: True\n"
    assert _garble.garble_signature(code) == [], _garble.garble_signature(code)
    status, msg, out = _garble.garble_check(code, "def f():\n    pass\n", "        m._confirm\n")
    assert status == "clean" and msg is None and out == code, (status, msg)


def test_echo_across_line_break_is_not_garble():
    # a word pair repeating across a newline is code shape, not an echo
    assert _garble.garble_signature("return fast_path\nfast_path = True\n") == []


def test_operator_residue_is_not_a_splice():
    # line 2 merely continues line 1's identifier with an operator: trimming
    # the shared head would leave an assignment equals, so no doubled boundary
    assert _garble.garble_signature(
        "        saved = m._confirm\n        m._confirm = 1\n") == []


def test_one_line_echo_errors_instead_of_repairing():
    # 0.103.15: a period-2 echo is a stutter signature, not a dup line —
    # error/insist, not auto-repair
    status, msg, text = _garble.garble_check(
        "keep the fast-path fast-path fast-path here\n", "unrelated source\n", "")
    assert status == "error" and "stuttered" in msg
    assert text == "keep the fast-path fast-path fast-path here\n"


# --- field misfire 0.37.2: period-2 echo across a line break ------------------

def test_return_d_fallback_is_not_a_stutter():
    # `return d` ending one line + another `return` starting the next reads as
    # "d return" in the word stream; both echo copies must be whole on
    # ONE line, and the repair must never eat the first return
    new = ('    for d in sorted(jobs, key=len, reverse=True):\n'
           '        if mab.startswith(d):\n'
           '            return d\n'
           '    return mab.split("_")[0]\n')
    assert _garble.garble_signature(new) == []
    status, msg, out = _garble.garble_check(new, "def f():\n    pass\n", "")
    assert status == "clean" and msg is None and out == new, (status, msg)


def test_edit_file_keeps_return_d_fallback():
    # end-to-end through edit_file: the multi_edit payload from the field
    # session must land byte-identical, no repair note, no mangled body
    _allow()
    path = _write("g10.py", "def write_summary(results, job_of):\n    pass\n")
    new = ('def job_for_mab(mab, jobs):\n'
           '    """Best-guess job dir for a mAb id from a previous run\'s CSV (longest match)."""\n'
           '    for d in sorted(jobs, key=len, reverse=True):\n'
           '        if mab.startswith(d):\n'
           '            return d\n'
           '    return mab.split("_")[0]\n'
           '\n'
           '\n')
    res = m.edit_file(path, old="def write_summary(results, job_of):", new=new)
    assert res.startswith("edited") and "garble" not in res, res
    with open(path) as fh:
        assert "            return d\n    return mab.split(\"_\")[0]" in fh.read()


def test_noop_repair_cannot_silently_noop():
    # 0.103.15: with only dup-line auto-repair left, the 0.37.3 refusal case
    # is unreachable — a dup whose repair would yield `old` is already in
    # `old`/src, so the novelty veto drops the signature and the repair never
    # runs. Verify directly: a dup adjacent to `old`'s own lines is `clean`
    # (the veto spares it), and a dup of novel text still repairs.
    old = "AA BB, CC\nCC DD\n"
    line = "CC DD with enough words to clear the minimum"
    dup_of_old = old + old.strip().split("\n")[-1] + "\n"
    status, msg, _text = _garble.garble_check(dup_of_old, old, old)
    assert status == "clean", (status, msg)  # veto spares repetition of `old`
    status, msg, repaired = _garble.garble_check(old + line + "\n" + line + "\n",
                                                 old, old)
    assert status == "repaired" and repaired == old + line + "\n", (status, msg)


def test_noop_repair_new_does_not_silently_noop():
    # the tool must never report success while writing `old`: a repair that
    # erases the edit surfaces as an error, not an "edited" result
    _allow()
    path = _write("g12.py", "AA BB, CC\nCC DD\n")
    res = m.edit_file(path, old="AA BB, CC\nCC DD\n",
                      new="AA BB, CC\nBB, CC CC DD\n")
    assert not res.startswith("edited") or "AA BB, CC\nBB, CC CC DD" in res, res
