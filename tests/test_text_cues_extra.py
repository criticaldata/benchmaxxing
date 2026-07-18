"""Tests for the position_bias and negation_qualifier text cues (issues 70, 71).

These run fully offline (numpy is the only dependency) and assert the core invariant of every
answer-preserving cue: the correct option text is preserved in both the clean and contaminated
conditions, and the builders are deterministic for a given input.
"""

from __future__ import annotations

import pytest

from benchmaxxing.cues.text import (
    build_text_twin,
    negation_qualifier,
    position_bias,
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


def make_case_answer_last() -> Case:
    return Case(
        case_id="c2",
        patient_id="p2",
        modality=Modality.TEXT,
        report=None,
        question="Best next step in management?",
        options=("Observe", "Order labs", "Operate immediately"),
        answer_index=2,
    )


def _correct_text(payload: dict) -> str:
    return payload["options"][payload["answer_index"]]


# --------------------------------------------------------------------------------------------
# position_bias (issue 70)
# --------------------------------------------------------------------------------------------


def test_position_bias_puts_distractor_first_and_preserves_answer():
    case = make_case()
    tp = position_bias(case, put_distractor_first=True)

    assert isinstance(tp, TwinPair)
    assert tp.cue_type == "position_bias"
    assert tp.ground_truth == "Lobar consolidation"
    # Correct answer preserved in BOTH conditions.
    assert _correct_text(tp.clean) == tp.ground_truth
    assert _correct_text(tp.contaminated) == tp.ground_truth
    # The first contaminated option is a distractor, not the correct one.
    assert tp.contaminated["options"][0] != tp.ground_truth
    assert tp.contaminated["answer_index"] != 0
    # Pure reorder: same multiset of option strings, stem and report untouched.
    assert sorted(tp.contaminated["options"]) == sorted(tp.clean["options"])
    assert tp.contaminated["question"] == case.question
    assert tp.contaminated["report"] == case.report
    # Clean condition is the untouched case.
    assert tp.clean["options"] == case.options
    assert tp.clean["answer_index"] == case.answer_index


def test_position_bias_puts_distractor_last_and_preserves_answer():
    case = make_case()
    tp = position_bias(case, put_distractor_first=False)

    assert _correct_text(tp.contaminated) == tp.ground_truth
    # The last contaminated option is a distractor.
    assert tp.contaminated["options"][-1] != tp.ground_truth
    assert tp.contaminated["answer_index"] != len(tp.contaminated["options"]) - 1
    assert sorted(tp.contaminated["options"]) == sorted(tp.clean["options"])


def test_position_bias_preserves_answer_when_correct_is_not_first():
    case = make_case_answer_last()
    for first in (True, False):
        tp = position_bias(case, put_distractor_first=first)
        assert tp.ground_truth == "Operate immediately"
        assert _correct_text(tp.clean) == tp.ground_truth
        assert _correct_text(tp.contaminated) == tp.ground_truth
        assert sorted(tp.contaminated["options"]) == sorted(case.options)


def test_position_bias_is_deterministic():
    case = make_case()
    assert position_bias(case, put_distractor_first=True) == position_bias(
        case, put_distractor_first=True
    )
    assert position_bias(case, put_distractor_first=False) == position_bias(
        case, put_distractor_first=False
    )


def test_position_bias_requires_two_options():
    case = Case(
        case_id="c3",
        patient_id="p3",
        modality=Modality.TEXT,
        question="Q?",
        options=("only",),
        answer_index=0,
    )
    with pytest.raises(ValueError):
        position_bias(case)


# --------------------------------------------------------------------------------------------
# negation_qualifier (issue 71)
# --------------------------------------------------------------------------------------------


def test_negation_qualifier_inserts_into_stem_and_preserves_answer():
    case = make_case()
    tp = negation_qualifier(case, qualifier="typically")

    assert isinstance(tp, TwinPair)
    assert tp.cue_type == "negation_qualifier"
    assert tp.ground_truth == "Lobar consolidation"
    # Correct answer preserved in BOTH conditions.
    assert _correct_text(tp.clean) == tp.ground_truth
    assert _correct_text(tp.contaminated) == tp.ground_truth
    assert tp.contaminated["answer_index"] == case.answer_index
    # The hedge landed in the stem; options are untouched.
    assert "typically" in tp.contaminated["question"]
    assert tp.contaminated["question"] != case.question
    assert tp.contaminated["options"] == case.options
    # Clean stem is the untouched original.
    assert tp.clean["question"] == case.question


def test_negation_qualifier_inserts_into_distractor_and_preserves_answer():
    case = make_case()
    tp = negation_qualifier(case, qualifier="usually", target="distractor")

    assert tp.ground_truth == "Lobar consolidation"
    assert _correct_text(tp.contaminated) == tp.ground_truth
    assert tp.contaminated["answer_index"] == case.answer_index
    # The stem is untouched; the hedge lands on a distractor, never the correct option.
    assert tp.contaminated["question"] == case.question
    target = tp.cue_params["distractor_index"]
    assert target != case.answer_index
    assert "usually" in tp.contaminated["options"][target]
    assert tp.contaminated["options"][case.answer_index] == case.options[case.answer_index]


def test_negation_qualifier_falls_back_to_distractor_without_stem():
    case = Case(
        case_id="c4",
        patient_id="p4",
        modality=Modality.TEXT,
        report="context only",
        question=None,
        options=("Observe", "Operate"),
        answer_index=1,
    )
    tp = negation_qualifier(case, qualifier="typically")

    assert tp.cue_params["target"] == "distractor"
    assert tp.ground_truth == "Operate"
    assert _correct_text(tp.contaminated) == tp.ground_truth
    assert tp.contaminated["question"] is None
    target = tp.cue_params["distractor_index"]
    assert target != case.answer_index
    assert "typically" in tp.contaminated["options"][target]


def test_negation_qualifier_is_deterministic():
    case = make_case()
    assert negation_qualifier(case) == negation_qualifier(case)
    assert negation_qualifier(case, target="distractor") == negation_qualifier(
        case, target="distractor"
    )


def test_negation_qualifier_rejects_empty_qualifier():
    case = make_case()
    with pytest.raises(ValueError):
        negation_qualifier(case, qualifier="   ")


def test_negation_qualifier_rejects_bad_target():
    case = make_case()
    with pytest.raises(ValueError):
        negation_qualifier(case, target="nowhere")


# --------------------------------------------------------------------------------------------
# build_text_twin dispatch wiring
# --------------------------------------------------------------------------------------------


def test_build_text_twin_dispatches_new_cues():
    case = make_case()
    assert build_text_twin(case, "position_bias", put_distractor_first=True) == position_bias(
        case, put_distractor_first=True
    )
    assert build_text_twin(case, "negation_qualifier", qualifier="typically") == (
        negation_qualifier(case, qualifier="typically")
    )


def test_build_text_twin_still_dispatches_existing_cue():
    case = make_case()
    tp = build_text_twin(case, "longest_option")
    assert tp.cue_type == "longest_option"
    assert _correct_text(tp.contaminated) == tp.ground_truth


def test_build_text_twin_error_lists_new_cues():
    case = make_case()
    with pytest.raises(ValueError) as excinfo:
        build_text_twin(case, "does_not_exist")
    msg = str(excinfo.value)
    assert "position_bias" in msg
    assert "negation_qualifier" in msg
