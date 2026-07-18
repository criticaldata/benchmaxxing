"""Tests for benchmaxxing.cues.text.

These exercise the answer-preserving text cues fully offline: no API keys and no optional
dependencies (numpy, scipy, scikit-learn and pillow are enough; none are needed beyond numpy).
Each test asserts both the invariant that the correct answer is preserved and that the builder is
deterministic.
"""

from __future__ import annotations

import re

import pytest

from benchmaxxing.cues.text import (
    build_text_twin,
    lexical_overlap_bias,
    longest_option_bias,
    option_order_permutation,
    report_demographic_hint,
)
from benchmaxxing.schema import Case, Modality, TwinPair


def make_case() -> Case:
    return Case(
        case_id="c1",
        patient_id="p1",
        modality=Modality.TEXT,
        report="History of cough and fever.",
        question="Which finding is most consistent with pneumonia on chest radiograph?",
        options=("Lobar consolidation", "Normal lungs", "Rib fracture", "Pneumothorax"),
        answer_index=0,
    )


def _correct_text(payload: dict) -> str:
    return payload["options"][payload["answer_index"]]


def _tokset(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(stem: str, option: str) -> int:
    return len(_tokset(stem) & _tokset(option))


# --------------------------------------------------------------------------------------------
# option_order_permutation
# --------------------------------------------------------------------------------------------


def test_option_order_preserves_answer():
    case = make_case()
    tp = option_order_permutation(case, seed=3)

    assert isinstance(tp, TwinPair)
    assert tp.ground_truth == "Lobar consolidation"
    # Correct answer preserved in BOTH conditions.
    assert _correct_text(tp.clean) == tp.ground_truth
    assert _correct_text(tp.contaminated) == tp.ground_truth
    # The option set is a pure permutation (same multiset of strings).
    assert sorted(tp.clean["options"]) == sorted(tp.contaminated["options"])
    # Report and question untouched.
    assert tp.contaminated["question"] == case.question
    assert tp.contaminated["report"] == case.report


def test_option_order_is_deterministic():
    case = make_case()
    assert option_order_permutation(case, seed=7) == option_order_permutation(case, seed=7)


def test_option_order_actually_reorders_for_some_seed():
    case = make_case()
    reordered = [
        option_order_permutation(case, seed=s).contaminated["options"] != case.options
        for s in range(10)
    ]
    assert any(reordered)


# --------------------------------------------------------------------------------------------
# longest_option_bias
# --------------------------------------------------------------------------------------------


def test_longest_option_makes_distractor_longest_and_preserves_answer():
    case = make_case()
    tp = longest_option_bias(case)

    assert tp.ground_truth == "Lobar consolidation"
    # Answer preserved: same index, same correct text.
    assert tp.contaminated["answer_index"] == case.answer_index
    assert _correct_text(tp.contaminated) == tp.ground_truth
    # The strictly longest option is NOT the correct one (the cue points at a distractor).
    opts = tp.contaminated["options"]
    longest_index = max(range(len(opts)), key=lambda i: len(opts[i]))
    assert longest_index != tp.contaminated["answer_index"]
    lengths = sorted(len(o) for o in opts)
    assert lengths[-1] > lengths[-2]  # strictly longest, no tie
    # Clean condition is the untouched case.
    assert tp.clean["options"] == case.options


def test_longest_option_is_deterministic():
    case = make_case()
    assert longest_option_bias(case) == longest_option_bias(case)


# --------------------------------------------------------------------------------------------
# lexical_overlap_bias
# --------------------------------------------------------------------------------------------


def test_lexical_overlap_increases_overlap_and_preserves_answer():
    case = make_case()
    tp = lexical_overlap_bias(case)

    assert tp.ground_truth == "Lobar consolidation"
    assert tp.contaminated["answer_index"] == case.answer_index
    assert _correct_text(tp.contaminated) == tp.ground_truth

    target = tp.cue_params["option_index"]
    assert target != case.answer_index  # cue lands on a distractor
    stem = case.question
    clean_overlap = _overlap(stem, tp.clean["options"][target])
    dirty_overlap = _overlap(stem, tp.contaminated["options"][target])
    assert dirty_overlap > clean_overlap
    # The correct option text itself was not edited.
    assert tp.contaminated["options"][case.answer_index] == case.options[case.answer_index]


def test_lexical_overlap_rejects_targeting_correct_option():
    case = make_case()
    with pytest.raises(ValueError):
        lexical_overlap_bias(case, option_index=case.answer_index)


def test_lexical_overlap_is_deterministic():
    case = make_case()
    assert lexical_overlap_bias(case) == lexical_overlap_bias(case)


# --------------------------------------------------------------------------------------------
# report_demographic_hint
# --------------------------------------------------------------------------------------------


def test_demographic_hint_appends_and_preserves_answer():
    case = make_case()
    hint = "The patient is a 68-year-old man."
    tp = report_demographic_hint(case, hint)

    assert tp.ground_truth == "Lobar consolidation"
    assert _correct_text(tp.contaminated) == tp.ground_truth
    assert tp.contaminated["answer_index"] == case.answer_index
    assert tp.contaminated["options"] == case.options
    assert hint in tp.contaminated["report"]
    assert case.report in tp.contaminated["report"]
    assert tp.clean["report"] == case.report


def test_demographic_hint_with_no_existing_report():
    case = Case(
        case_id="c2",
        patient_id="p2",
        modality=Modality.TEXT,
        report=None,
        question="Best next step?",
        options=("Observe", "Operate"),
        answer_index=1,
    )
    hint = "The patient is a neonate."
    tp = report_demographic_hint(case, hint)
    assert tp.contaminated["report"] == hint
    assert tp.clean["report"] is None
    assert tp.ground_truth == "Operate"


def test_demographic_hint_is_deterministic():
    case = make_case()
    hint = "The patient is a 50-year-old woman."
    assert report_demographic_hint(case, hint) == report_demographic_hint(case, hint)


def test_demographic_hint_rejects_empty_hint():
    case = make_case()
    with pytest.raises(ValueError):
        report_demographic_hint(case, "   ")


# --------------------------------------------------------------------------------------------
# build_text_twin dispatch
# --------------------------------------------------------------------------------------------


def test_build_text_twin_dispatches_each_cue():
    case = make_case()
    assert build_text_twin(case, "option_order", seed=1) == option_order_permutation(case, seed=1)
    assert build_text_twin(case, "longest_option") == longest_option_bias(case)
    assert build_text_twin(case, "lexical_overlap") == lexical_overlap_bias(case)
    assert build_text_twin(case, "demographic_hint", hint="X.") == report_demographic_hint(
        case, "X."
    )


def test_build_text_twin_rejects_unknown_cue():
    case = make_case()
    with pytest.raises(ValueError):
        build_text_twin(case, "does_not_exist")


# --------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------


def test_cue_requires_options():
    case = Case(case_id="c3", patient_id="p3", modality=Modality.TEXT, question="Q?")
    with pytest.raises(ValueError):
        option_order_permutation(case)


def test_cue_rejects_out_of_range_answer_index():
    case = Case(
        case_id="c4",
        patient_id="p4",
        modality=Modality.TEXT,
        question="Q?",
        options=("a", "b"),
        answer_index=5,
    )
    with pytest.raises(ValueError):
        longest_option_bias(case)


def test_distractor_cue_requires_two_options():
    case = Case(
        case_id="c5",
        patient_id="p5",
        modality=Modality.TEXT,
        question="Q?",
        options=("only",),
        answer_index=0,
    )
    with pytest.raises(ValueError):
        longest_option_bias(case)
