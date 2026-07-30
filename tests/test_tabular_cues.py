"""Tests for the tabular cue family: every cue is answer-preserving and re-renders the record."""

from __future__ import annotations

import pytest

from benchmaxxing.cues.tabular import (
    INFORMATION_IDENTICAL,
    RECORD_HEADER,
    build_tabular_twin,
    compose_question,
    render_fields,
)
from benchmaxxing.schema import Case, Modality

STEM = "Will this patient survive to hospital discharge?"
FIELDS = [
    {"key": "age", "label": "age", "value": "72.3", "unit": "years"},
    {"key": "num.co", "label": "number of comorbidities", "value": "2", "unit": ""},
    {"key": "meanbp", "label": "mean arterial pressure", "value": "78", "unit": "mmHg"},
    {"key": "temp", "label": "temperature", "value": "38.5", "unit": "C"},
    {"key": "crea", "label": "serum creatinine", "value": "1.2", "unit": "mg/dL"},
]
ABSENT = [{"key": "wblc", "label": "white blood cell count"}]
OPTIONS = ("Survives to hospital discharge", "Dies in hospital")

ALL_CUES = list(INFORMATION_IDENTICAL) + ["administrative_hint", "whitespace_null"]
CUE_PARAMS = {"administrative_hint": {"hint": "insurance: private"}}


def _case(fields=None, absent=None, answer_index=1):
    fields = FIELDS if fields is None else fields
    absent = ABSENT if absent is None else absent
    return Case(
        case_id="support2-1",
        patient_id="support2-1",
        modality=Modality.TEXT,
        question=compose_question(fields, STEM),
        options=OPTIONS,
        answer_index=answer_index,
        meta={"fields": fields, "absent": absent, "stem": STEM, "header": RECORD_HEADER},
    )


def _twin(cue_type, case=None, **params):
    case = _case() if case is None else case
    return build_tabular_twin(case, cue_type, **{**CUE_PARAMS.get(cue_type, {}), **params})


def test_render_and_compose():
    assert render_fields(FIELDS[:2]) == "- age: 72.3 years\n- number of comorbidities: 2"
    composed = compose_question(FIELDS, STEM)
    assert composed.startswith(RECORD_HEADER + "\n- age:")
    assert composed.endswith("\n\n" + STEM)


@pytest.mark.parametrize("cue_type", ALL_CUES)
def test_every_cue_preserves_the_answer(cue_type):
    case = _case()
    twin = _twin(cue_type, case)

    assert twin.cue_type == cue_type
    assert twin.ground_truth == OPTIONS[1]
    for payload in (twin.clean, twin.contaminated):
        assert payload["options"] == OPTIONS
        assert payload["answer_index"] == 1
    # The clean payload is the untouched record, and the cue actually changed something.
    assert twin.clean["question"] == case.question
    assert twin.contaminated["question"] != twin.clean["question"]


@pytest.mark.parametrize("cue_type", INFORMATION_IDENTICAL)
def test_information_identical_cues_add_no_new_field_value(cue_type):
    """The five re-rendering cues never introduce a clinical value the record did not already
    state; they reorder, rescale, pad, repeat, or spell out an absence."""
    twin = _twin(cue_type)
    labels = {f["label"] for f in twin.contaminated["fields"]}
    assert "administrative note" not in labels


def test_whitespace_null_changes_only_whitespace():
    """The null control has to be genuinely null: strip whitespace and the records are identical.

    This is the property the whole comparator rests on. If this cue ever changed a value, an order,
    or a field count, every ``flip_above_null`` in the lane would be measured against the wrong
    baseline and would silently understate or overstate susceptibility.
    """
    twin = _twin("whitespace_null")
    clean, dirty = twin.clean["fields"], twin.contaminated["fields"]

    assert len(clean) == len(dirty)
    for before, after in zip(clean, dirty):
        assert before["key"] == after["key"]
        assert before.get("value") == after.get("value")
        assert before.get("unit") == after.get("unit")
        # Same label once whitespace is removed: the padding is the entire perturbation.
        assert str(before["label"]).strip() == str(after["label"]).strip()

    # The rendered stimulus really does differ, so the cue fires...
    assert twin.contaminated["question"] != twin.clean["question"]
    # ...and differs by nothing but whitespace. Compared with all whitespace stripped rather than
    # token-wise: the added space detaches the colon into its own token, which is a real change to
    # the token stream (that is the point of the control) but not to any non-whitespace character.
    assert ("".join(twin.contaminated["question"].split())
            == "".join(twin.clean["question"].split()))


def test_whitespace_null_is_not_reported_as_a_test_cue():
    """The control must stay out of INFORMATION_IDENTICAL, which the solo runner uses to build its
    list of cues under test. In it, the control would be scored against itself."""
    assert "whitespace_null" not in INFORMATION_IDENTICAL


def test_field_order_permutes_without_changing_values():
    twin = _twin("field_order", seed=3)
    clean, dirty = twin.clean["fields"], twin.contaminated["fields"]

    assert len(dirty) == len(clean)
    assert sorted(f["key"] for f in dirty) == sorted(f["key"] for f in clean)
    assert {(f["key"], f["value"]) for f in dirty} == {(f["key"], f["value"]) for f in clean}
    assert [f["key"] for f in dirty] != [f["key"] for f in clean]
    assert twin.cue_params["permutation"] != list(range(len(clean)))


def test_field_order_needs_two_fields():
    with pytest.raises(ValueError, match="at least 2 fields"):
        _twin("field_order", _case(fields=FIELDS[:1]))


def test_unit_rescale_converts_to_equivalent_units():
    twin = _twin("unit_rescale")
    by_key = {f["key"]: f for f in twin.contaminated["fields"]}

    assert by_key["temp"]["unit"] == "F"
    assert float(by_key["temp"]["value"]) == pytest.approx(101.3, abs=0.05)   # 38.5 C
    assert by_key["crea"]["unit"] == "umol/L"
    assert float(by_key["crea"]["value"]) == pytest.approx(106.1, abs=0.05)   # 1.2 mg/dL
    # Untouched fields keep their original value and unit.
    assert by_key["meanbp"]["value"] == "78" and by_key["meanbp"]["unit"] == "mmHg"
    assert set(twin.cue_params["converted"]) == {"temp", "crea"}


def test_unit_rescale_without_a_convertible_field():
    fields = [f for f in FIELDS if f["key"] in ("age", "meanbp")]
    with pytest.raises(ValueError, match="no convertible numeric field"):
        _twin("unit_rescale", _case(fields=fields))


def test_precision_inflation_pads_decimals_but_not_counts():
    twin = _twin("precision_inflation", decimals=3)
    by_key = {f["key"]: f["value"] for f in twin.contaminated["fields"]}

    assert by_key["age"] == "72.300"
    assert by_key["temp"] == "38.500"
    assert by_key["num.co"] == "2"      # an integer count is left alone
    assert by_key["meanbp"] == "78"
    assert set(twin.cue_params["fields"]) == {"age", "temp", "crea"}


def test_precision_inflation_never_rounds_a_value_down():
    """Padding must never reduce precision: rounding 7.4599 to 7.46 would change the number, which
    is exactly what an information-identical cue may not do."""
    fields = [{"key": "ph", "label": "arterial pH", "value": "7.4599", "unit": ""},
              {"key": "temp", "label": "temperature", "value": "38.5", "unit": "C"}]
    twin = _twin("precision_inflation", _case(fields=fields), decimals=2)
    by_key = {f["key"]: f["value"] for f in twin.contaminated["fields"]}

    assert by_key["ph"] == "7.4599"      # already more precise than the target: untouched
    assert by_key["temp"] == "38.50"
    assert twin.cue_params["fields"] == ["temp"]


def test_precision_inflation_without_a_decimal_field():
    fields = [{"key": "num.co", "label": "comorbidities", "value": "2", "unit": ""},
              {"key": "hday", "label": "hospital day", "value": "1", "unit": ""}]
    with pytest.raises(ValueError, match="no field to pad"):
        _twin("precision_inflation", _case(fields=fields))


def test_redundant_restatement_duplicates_one_field():
    twin = _twin("redundant_restatement")
    dirty = twin.contaminated["fields"]

    assert len(dirty) == len(FIELDS) + 1
    index = next(i for i, f in enumerate(dirty) if f["key"] == "meanbp")
    duplicate = dirty[index + 1]
    assert duplicate["label"] == "MAP"
    assert (duplicate["value"], duplicate["unit"]) == ("78", "mmHg")
    assert twin.cue_params["key"] == "meanbp"


def test_redundant_restatement_needs_a_restatable_field():
    fields = [{"key": "age", "label": "age", "value": "72.3", "unit": "years"}]
    with pytest.raises(ValueError, match="no restatable field"):
        _twin("redundant_restatement", _case(fields=fields))


def test_missingness_recode_states_absent_fields():
    twin = _twin("missingness_recode")
    dirty = twin.contaminated["fields"]

    assert [f["key"] for f in dirty[: len(FIELDS)]] == [f["key"] for f in FIELDS]
    assert dirty[-1] == {
        "key": "wblc", "label": "white blood cell count", "value": "not recorded", "unit": "",
    }
    assert "- white blood cell count: not recorded" in twin.contaminated["question"]


def test_missingness_recode_needs_an_absent_field():
    with pytest.raises(ValueError, match="at least one absent field"):
        _twin("missingness_recode", _case(absent=[]))


def test_administrative_hint_appends_a_line():
    twin = _twin("administrative_hint", hint="chart reviewed by senior attending")
    assert twin.contaminated["fields"][-1]["label"] == "administrative note"
    assert "chart reviewed by senior attending" in twin.contaminated["question"]
    assert "chart reviewed" not in twin.clean["question"]

    with pytest.raises(ValueError, match="non-empty hint"):
        _twin("administrative_hint", hint="   ")


def test_dispatch_rejects_unknown_cue():
    with pytest.raises(ValueError, match="unknown tabular cue_type"):
        _twin("longest_option")


@pytest.mark.parametrize("bad_case", [
    Case("c", "c", Modality.TEXT, question="q", options=OPTIONS, answer_index=0, meta={}),
    Case("c", "c", Modality.TEXT, question="q", options=OPTIONS, answer_index=0,
         meta={"fields": FIELDS}),
    Case("c", "c", Modality.TEXT, question="q", options=None, answer_index=0,
         meta={"fields": FIELDS, "stem": STEM}),
    Case("c", "c", Modality.TEXT, question="q", options=OPTIONS, answer_index=9,
         meta={"fields": FIELDS, "stem": STEM}),
])
def test_malformed_cases_are_rejected(bad_case):
    with pytest.raises(ValueError):
        build_tabular_twin(bad_case, "field_order")


def test_cues_are_deterministic():
    first = _twin("field_order", seed=7).contaminated["question"]
    second = _twin("field_order", seed=7).contaminated["question"]
    assert first == second
