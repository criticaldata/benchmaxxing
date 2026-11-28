"""Semantic check of the committed paraphrases (experiments/matched_baseline/paraphrase_semantic_check.py).

Offline. Covers the keyless rule check, the judge prompt and its reply parser, and the bands with
failed paraphrases left out, on small hand-built inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "matched_baseline"))
import matched_baseline_reason as mb  # noqa: E402
import paraphrase_semantic_check as sc  # noqa: E402

EXCEPT = "All of the following are true regarding Class-II inlay gingival margins preparation except?"


def test_a_faithful_rewording_of_an_exception_passes():
    """The rule compares presence of a marker, not the word: these rewordings keep the exception."""
    for reworded in (
        "Regarding Class-II inlay gingival margins preparation, all of these are true, with one exception?",
        "Which statement is not true concerning Class-II inlay gingival margins preparation?",
        "All of these hold for Class-II inlay gingival margins preparation, save for one. Which?",
        "Apart from one, every statement below about Class-II inlay gingival margins preparation is true?",
        "Which of these is false about Class-II inlay gingival margins preparation?",
    ):
        got = sc.rule_check(EXCEPT, reworded)
        assert got["rule_check"] and got["rule_reason"] == "kept", reworded
    assert sc.rule_check(EXCEPT, EXCEPT)["markers_original"] == ["except"]


def test_a_dropped_or_added_negation_fails():
    dropped = sc.rule_check(EXCEPT, "Which of the following is true of Class-II inlay gingival margins preparation?")
    assert (dropped["rule_check"], dropped["rule_reason"], dropped["markers_paraphrase"]) == (False, "dropped", [])
    least = sc.rule_check("Which of the following is the least likely diagnosis?", "Which diagnosis is most likely?")
    assert (least["rule_check"], least["rule_reason"], least["markers_original"]) == (False, "dropped", ["least"])
    added = sc.rule_check("Which drug is indicated?", "Which drug is not indicated?")
    assert (added["rule_check"], added["rule_reason"], added["markers_paraphrase"]) == (False, "added", ["not"])
    contraction = sc.rule_check("Which finding isn't expected?", "Which finding is expected?")
    assert (contraction["rule_reason"], contraction["markers_original"]) == ("dropped", ["n't"])
    plain = sc.rule_check("Drug of choice for X is", "The preferred drug for X is")
    assert (plain["rule_check"], plain["rule_reason"]) == (True, "none")


def test_only_the_question_part_is_read():
    """Negations among the facts of a vignette are not what the rule compares; the judge reads those."""
    original = ("A 45-year-old man has no fever. His pH is 7.4 and he does not smoke. "
                "Which of the following is the most likely diagnosis?")
    assert sc.question_part(original) == "Which of the following is the most likely diagnosis?"
    assert sc.markers(original) == []
    assert sc.rule_check(original, "An afebrile man aged 45 has a pH of 7.4. What diagnosis is most likely?")["rule_check"]
    # a decimal point is not a sentence end
    assert sc.question_part("The pH is 7.4. Which value of 7.4 is not normal?") == "Which value of 7.4 is not normal?"
    # no question mark: the last sentence
    assert sc.question_part("Regarding X. All are true except") == "All are true except"
    record = "Patient record:\n- diabetes: no\n- dementia: no\n\nWill this patient survive to hospital discharge?"
    assert sc.markers(record) == [] and sc.question_part(record) == "Will this patient survive to hospital discharge?"
    assert sc.markers("Is it true that X is other than Y, or neither?") == ["other than", "neither"]
    # a question of fewer than five words is read with the sentence before it
    assert sc.question_part("He has no fever. All of these hold, save for one. Which?") == \
        "All of these hold, save for one. Which?"
    assert sc.question_part("One is false. Which one is the false one?") == "Which one is the false one?"


def test_judge_prompt_shows_both_versions_and_the_options_once():
    prompt = sc.judge_prompt("ORIGINAL TEXT?", "REWORDED TEXT?", ["alpha", "beta"])
    assert "ORIGINAL:\nORIGINAL TEXT?\n\nREWORDED:\nREWORDED TEXT?\n\nOPTIONS:\nA. alpha\nB. beta" in prompt
    assert "VERDICT: yes or no" in prompt and "REASON: one short sentence" in prompt
    assert prompt.count("alpha") == 1


def test_judge_reply_parser():
    assert sc.parse_judge("VERDICT: yes\nREASON: Same facts and question.") == ("yes", "Same facts and question.")
    assert sc.parse_judge("**VERDICT:** No\n**REASON:** The exception is dropped.") == ("no", "The exception is dropped.")
    # no verdict line is not a yes
    assert sc.parse_judge("They look equivalent to me.") == (None, "")
    assert sc.parse_judge("") == (None, "") and sc.parse_judge(None) == (None, "")


def _records():
    """Baseline A, decoy B, two paraphrases per case: c1 moves on read 2, c2 never, c3 on both paraphrases."""
    recs = []
    for cid, answers in (("c1", ["A", "A", "B"]), ("c2", ["A", "A", "A"]), ("c3", ["A", "B", "B"])):
        rec = mb._repeat_record(cid, answers[0], "A", "B", answers[1:])
        rec["n_paraphrases"] = 2
        recs.append(rec)
    return recs


def _checks(failed_rule=(), failed_judge=()):
    out = {}
    for cid in ("c1", "c2", "c3"):
        for i in (1, 2):
            rule, judge = (cid, i) not in failed_rule, (cid, i) not in failed_judge
            out[(cid, i)] = {"rule": rule, "judge": judge, "either": rule and judge}
    return out


def test_bands_leave_out_the_reads_of_failed_paraphrases():
    bands = sc.lineage_bands(_records(), _checks(failed_rule=[("c1", 2)], failed_judge=[("c3", 1)]), 2)
    assert bands["cases"] == 3 and bands["paraphrase_reads"] == 6
    assert bands["reads_left_out"] == {"rule": 1, "judge": 1, "either": 2}

    def any_dis(name, reads="reads_0_to_k"):
        return bands[name][reads]["cases_with_any_disagreement"]["n"], bands[name][reads]["n"]

    assert any_dis("all_paraphrases") == (2, 3)
    # c1 loses the read that disagreed; c3 still has read 2 against read 0
    assert any_dis("failed_left_out") == (1, 3)
    assert any_dis("rule_failed_left_out") == (1, 3) and any_dis("judge_failed_left_out") == (2, 3)
    # without read 0, c1 and c3 keep a single paraphrase read each and cannot disagree with themselves
    assert any_dis("failed_left_out", "reads_1_to_k") == (0, 1)
    assert bands["failed_left_out"]["reads_1_to_k"]["cases_with_fewer_than_two_reads"] == ["c1", "c3"]
    assert bands["failed_left_out"]["reads_0_to_k"]["cases_per_read"] == [3, 2, 2]
    clean = bands["cases_with_every_paraphrase_passing"]
    assert clean["n"] == 1 and clean["reads_0_to_k"]["cases_with_any_disagreement"]["n"] == 0


def test_with_nothing_failed_the_bands_are_the_committed_band():
    recs = _records()
    bands = sc.lineage_bands(recs, _checks(), 2)
    want = mb.summarise_repeats(recs, 2)
    for name in ("all_paraphrases", "failed_left_out", "cases_with_every_paraphrase_passing"):
        got = bands[name]["reads_0_to_k"]
        for k in ("n", "cases_with_any_disagreement", "mean_pairwise_disagreement", "flip_vs_baseline_band", "decoy_band"):
            assert got[k] == want[k], (name, k)
    assert bands["reads_left_out"] == {"rule": 0, "judge": 0, "either": 0}


def test_reverse_question_turns_the_first_reversible_marker_around():
    """The control: a paraphrase turned around by rule asks the opposite question."""
    want = {
        "Which statement among the following is not true concerning X?":
            ("Which statement among the following is true concerning X?", "not"),
        "Regarding X, all of the following statements are true, with one exception?":
            ("Regarding X, all of the following statements are true, with no exception?", "exception"),
        "All of the following are true regarding X except?": ("All of the following are true regarding X?", "except"),
        "Which of the following is NOT a feature of X?": ("Which of the following is a feature of X?", "not"),
        "Which of these is false about X?": ("Which of these is true about X?", "false"),
        "Which is the Least likely diagnosis?": ("Which is the Most likely diagnosis?", "least"),
        # only the question part is edited: the negated fact before it stays
        "A man has no fever. Which finding isn't expected?": ("A man has no fever. Which finding is expected?", "n't"),
    }
    for text, (reversed_text, marker) in want.items():
        got = sc.reverse_question(text)
        assert (got["text"], got["marker"]) == (reversed_text, marker), text
    # a marker with no entry in the table leaves the paraphrase out of the control
    for text in ("All of these hold, save for one. Which?", "Apart from one, every statement below is true?",
                 "Each of these, except one, is true. Which is it?", "Drug of choice for X is"):
        assert sc.reverse_question(text) is None, text


def test_the_rule_refuses_a_reversal_that_removes_the_marker_and_not_one_that_keeps_one():
    """What the rule can and cannot see, on known reversals; the judge is there for the second kind."""
    removed = sc.reverse_question("Which statement is not true concerning Class-II inlay margins?")["text"]
    assert sc.rule_check(EXCEPT, removed)["rule_reason"] == "dropped"
    still_marked = sc.reverse_question("All of these are true of Class-II inlay margins, with one exception?")["text"]
    assert "with no exception" in still_marked
    assert sc.rule_check(EXCEPT, still_marked)["rule_reason"] == "kept"


def test_control_summary_counts_refusals():
    checks = [{"rule_reason": "kept"}] * 4 + [{"rule_reason": "none"}] * 6
    control = [{"marker": "not", "rule_reason": "dropped", "judge_verdict": "no"},
               {"marker": "exception", "rule_reason": "kept", "judge_verdict": "no"},
               {"marker": "not", "rule_reason": "dropped", "judge_verdict": "yes"}]
    got = sc.control_summary(checks, control)
    assert (got["paraphrases_with_a_marker_kept"], got["reversed"], got["not_reversible_by_the_table"]) == (4, 3, 1)
    assert (got["judge_said_no"], got["judge_said_yes"], got["judge_gave_no_verdict"]) == (2, 1, 0)
    assert got["rule_refused"] == 2 and got["refused_by_judge_or_rule"] == 3
    assert got["reversed_by_marker"] == {"not": 2, "exception": 1}

