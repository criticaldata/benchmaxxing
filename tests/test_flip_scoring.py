"""A flip is a different OPTION on the cued twin, not a different answer string.

``longest_option`` and ``lexical_overlap`` rewrite the text of one wrong option, and ``option_order``
moves every option. So neither the text nor the position of an answer names the same option on both
twins. These tests pin the comparison the solo scorer uses (``analysis.chose_same_option``) on the
real cue builders, for each of the three cues, and the ``text_changed`` field that keeps the earlier
string comparison on record.
"""

from __future__ import annotations

import pytest

from benchmaxxing.analysis import (
    FlipRecord,
    chose_same_option,
    chosen_option,
    flip_rate,
    solo_evaluate,
    text_changed_rate,
)
from benchmaxxing.cross_dataset import run_cross_dataset_cue
from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.schema import Case, Condition, Modality, TwinPair

CLEAN, CUED = Condition.CLEAN, Condition.CONTAMINATED
ABSTAIN = "<abstain>"


def _case(case_id="c0", answer_index=2) -> Case:
    # Four options of distinct lengths; the answer is "gamma". The cued distractor of
    # longest_option / lexical_overlap is the first wrong option, "alpha one" (index 0).
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question="Which drug is first line for this presentation of severe community pneumonia?",
        options=("alpha one", "beta two three", "gamma", "delta four five six"),
        answer_index=answer_index,
    )


def _twin(cue: str, **kw) -> TwinPair:
    return build_text_twin(_case(**kw), cue)


# --- the three cues, option by option -------------------------------------------------------------


@pytest.mark.parametrize("cue", ["longest_option", "lexical_overlap"])
def test_the_same_rewritten_option_on_both_twins_is_not_a_flip(cue):
    twin = _twin(cue)
    clean, cued = twin.clean["options"], twin.contaminated["options"]
    # the cue rewrote the text of option 0 and nothing else
    assert cued[0] != clean[0] and cued[0].startswith(clean[0])
    assert list(cued[1:]) == list(clean[1:])
    # a model that picks option 0 on both twins returns two strings for one choice
    assert clean[0] != cued[0]
    assert chose_same_option(twin, clean[0], cued[0]) is True
    assert chosen_option(twin, CLEAN, clean[0]) == chosen_option(twin, CUED, cued[0]) == ("option", 0)


@pytest.mark.parametrize("cue", ["longest_option", "lexical_overlap"])
def test_moving_to_or_from_the_rewritten_option_is_a_flip(cue):
    twin = _twin(cue)
    clean, cued = twin.clean["options"], twin.contaminated["options"]
    assert chose_same_option(twin, clean[2], cued[0]) is False  # correct answer -> cued distractor
    assert chose_same_option(twin, clean[0], cued[2]) is False  # cued distractor -> correct answer
    assert chose_same_option(twin, clean[1], cued[3]) is False  # one untouched option -> another
    for i in (1, 2, 3):  # an untouched option kept: same text, same choice
        assert chose_same_option(twin, clean[i], cued[i]) is True


def test_option_order_follows_the_option_not_the_position():
    twin = _twin("option_order")
    clean, cued = list(twin.clean["options"]), list(twin.contaminated["options"])
    perm = twin.cue_params["permutation"]
    assert cued != clean and sorted(cued) == sorted(clean)  # moved, texts intact
    for j, text in enumerate(cued):
        # the option now at position j is the clean option perm[j]
        assert chosen_option(twin, CUED, text) == ("option", perm[j]) == chosen_option(twin, CLEAN, text)
        assert chose_same_option(twin, text, text) is True
    # a position-follower keeps the first slot and so changes option
    assert clean[0] != cued[0]
    assert chose_same_option(twin, clean[0], cued[0]) is False


def test_position_bias_and_a_distractor_qualifier_use_the_same_rule():
    moved = _twin("position_bias")
    for j, text in enumerate(moved.contaminated["options"]):
        assert chosen_option(moved, CUED, text) == ("option", moved.cue_params["order"][j])
    hedged = build_text_twin(_case(), "negation_qualifier", target="distractor")
    d = hedged.cue_params["distractor_index"]
    assert hedged.contaminated["options"][d] != hedged.clean["options"][d]
    assert chose_same_option(hedged, hedged.clean["options"][d], hedged.contaminated["options"][d]) is True


def test_an_answer_that_names_no_option_is_compared_as_before():
    twin = _twin("longest_option")
    clean, cued = twin.clean["options"], twin.contaminated["options"]
    assert chosen_option(twin, CUED, ABSTAIN) == ("answer", ABSTAIN)
    assert chose_same_option(twin, ABSTAIN, ABSTAIN) is True
    assert chose_same_option(twin, clean[0], ABSTAIN) is False
    assert chose_same_option(twin, ABSTAIN, cued[0]) is False
    assert chose_same_option(twin, "", "") is True
    # a cued-side answer that quotes the option without the text the cue appended: same option
    assert chose_same_option(twin, clean[0], clean[0]) is True
    assert chosen_option(twin, CUED, clean[0]) == ("option", 0)


@pytest.mark.parametrize("cue", ["longest_option", "lexical_overlap", "option_order"])
def test_two_options_with_the_same_text_are_one_option(cue):
    # A real MedMCQA item repeats an option. The answer sits in the last slot, so the cue rewrites
    # the first one, which has the same text: the answer string alone cannot say which copy was
    # meant, and the same string on both twins must never read as a flip.
    case = Case(case_id="dup", patient_id="p", modality=Modality.TEXT,
                question="Which pouch gives rise to the inferior parathyroid gland in the embryo?",
                options=("Ventral part of 3rd pouch.", "Ventral part of 2nd pouch.",
                         "Dorsal part of 2nd pouch.", "Ventral part of 3rd pouch."), answer_index=3)
    twin = build_text_twin(case, cue)
    answer = "Ventral part of 3rd pouch."
    assert answer in twin.clean["options"] and answer in twin.contaminated["options"]
    assert chose_same_option(twin, answer, answer) is True
    assert chosen_option(twin, CLEAN, answer) == chosen_option(twin, CUED, answer) == ("option", 0)
    for text in twin.contaminated["options"]:  # the rewritten copy is still that option
        if text.startswith(answer):
            assert chosen_option(twin, CUED, text) == ("option", 0)
    assert chose_same_option(twin, answer, "Dorsal part of 2nd pouch.") is False


def test_payloads_without_options_keep_the_plain_comparison():
    # the imaging lane and the string-payload tests: there is no option list to map through
    twin = TwinPair(case_id="img", cue_type="cable", clean="clean film", contaminated="cued film",
                    ground_truth="yes")
    assert chose_same_option(twin, "yes", "yes") is True
    assert chose_same_option(twin, "yes", "no") is False
    assert chosen_option(twin, CUED, "no") == ("answer", "no")


# --- the solo scorer ------------------------------------------------------------------------------


def _pick(index_by_condition):
    """A backend that picks a fixed position on the clean twin and another on the cued one."""
    def backend(payload):
        options = list(payload["options"])
        cued = options != ["alpha one", "beta two three", "gamma", "delta four five six"]
        return options[index_by_condition[1] if cued else index_by_condition[0]]
    return backend


@pytest.mark.parametrize("cue", ["longest_option", "lexical_overlap"])
def test_solo_evaluate_keeps_the_text_comparison_beside_the_flip(cue):
    twins = [build_text_twin(_case(f"c{i}"), cue) for i in range(4)]
    stays = solo_evaluate(twins, _pick((0, 0)), lambda raw: raw, model="keeps the cued distractor")
    moves = solo_evaluate(twins, _pick((2, 0)), lambda raw: raw, model="moves to the cued distractor")
    steady = solo_evaluate(twins, _pick((2, 2)), lambda raw: raw, model="keeps the right answer")

    assert [r.flipped for r in stays] == [False] * 4       # same option on both twins
    assert [r.text_changed for r in stays] == [True] * 4   # ... whose text the cue rewrote
    assert [r.clean_correct for r in stays] == [False] * 4
    assert [r.flipped for r in moves] == [True] * 4 and [r.text_changed for r in moves] == [True] * 4
    assert [r.flipped for r in steady] == [False] * 4 and [r.text_changed for r in steady] == [False] * 4

    assert flip_rate(stays)["per_cue"] == {cue: 0.0}
    assert text_changed_rate(stays) == {"overall": 1.0, "per_cue": {cue: 1.0}, "n": 4}
    assert flip_rate(moves)["overall"] == text_changed_rate(moves)["overall"] == 1.0


def test_solo_evaluate_on_option_order_counts_the_same_under_both_comparisons():
    twins = [build_text_twin(_case(f"c{i}"), "option_order") for i in range(4)]
    position = solo_evaluate(twins, lambda p: p["options"][0], lambda raw: raw, model="first slot")
    content = solo_evaluate(twins, lambda p: "gamma", lambda raw: raw, model="same answer")
    assert [r.flipped for r in position] == [r.text_changed for r in position] == [True] * 4
    assert [r.flipped for r in content] == [r.text_changed for r in content] == [False] * 4


def test_a_flip_is_always_a_text_change_and_never_the_reverse_on_option_order():
    for cue in ("longest_option", "lexical_overlap", "option_order"):
        twin = _twin(cue)
        for a in list(twin.clean["options"]) + [ABSTAIN]:
            for b in list(twin.contaminated["options"]) + [ABSTAIN]:
                flipped, text_changed = not chose_same_option(twin, a, b), a != b
                assert text_changed or not flipped, (cue, a, b)
                if cue == "option_order":
                    assert flipped == text_changed, (a, b)


def test_text_changed_rate_leaves_out_records_built_without_the_field():
    old = FlipRecord("c", "longest_option", "m", "a", "b", "a", True, True, False)
    assert old.text_changed is None
    assert text_changed_rate([old])["n"] == 0
    new = FlipRecord("c", "longest_option", "m", "a", "a x", "a", False, True, False, text_changed=True)
    assert text_changed_rate([old, new]) == {"overall": 1.0, "per_cue": {"longest_option": 1.0}, "n": 1}


# --- the cross-dataset comparison -----------------------------------------------------------------


def test_cross_dataset_reports_the_earlier_count_beside_each_flip_count():
    datasets = {"medqa": [_case("a0"), _case("a1")], "medmcqa": [_case("b0"), _case("b1")]}
    res = run_cross_dataset_cue(datasets, _pick((0, 0)), lambda raw: raw,
                                ["longest_option", "option_order"], n_boot=50)
    padded = res["per_cue"]["longest_option"]
    assert padded["counts"] == {"medqa": [0, 2], "medmcqa": [0, 2]}  # the choice never changed
    assert padded["text_changed"] == {"rate": {"medqa": 1.0, "medmcqa": 1.0},
                                      "counts": {"medqa": [2, 2], "medmcqa": [2, 2]}}
    moved = res["per_cue"]["option_order"]  # the first slot holds another option after the cue
    assert moved["counts"] == moved["text_changed"]["counts"] == {"medqa": [2, 2], "medmcqa": [2, 2]}
    # the clean pick (option 0) is wrong here, so the correct-on-clean block is empty, not an error
    kept = res["clean_correct"]["per_cue"]["longest_option"]
    assert kept["counts"] == kept["text_changed"]["counts"] == {"medqa": [0, 0], "medmcqa": [0, 0]}
