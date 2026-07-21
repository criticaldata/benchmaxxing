
from __future__ import annotations

import json

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import pubmedqa
from benchmaxxing.schema import Modality

# Three synthetic records in the real PubMedQA ori_pqal.json shape (dict keyed by PMID).
FIXTURE_RECORDS = {
    "10000001": {
        "QUESTION": "Does drug X reduce mortality in condition Y?",
        "CONTEXTS": ["Background: condition Y is common.", "Results: mortality fell 20%."],
        "MESHES": ["Drug X", "Condition Y"],
        "YEAR": "2019",
        "final_decision": "yes",
        "LONG_ANSWER": "Drug X was associated with a significant mortality reduction.",
    },
    "10000002": {
        "QUESTION": "Is biomarker Z predictive of relapse?",
        "CONTEXTS": ["Background: biomarker Z was proposed as a predictor."],
        "MESHES": ["Biomarker Z"],
        "YEAR": "2020",
        "final_decision": "no",
        "LONG_ANSWER": "No association between biomarker Z and relapse was found.",
    },
    "10000003": {
        "QUESTION": "Does early mobilization improve recovery after surgery W?",
        "CONTEXTS": ["Background: mobilization timing varies.", "Results were mixed."],
        "MESHES": ["Surgery W", "Mobilization"],
        "YEAR": "2021",
        "final_decision": "maybe",
        "LONG_ANSWER": "Evidence was inconclusive; effect size was small and inconsistent.",
    },
}


def _write_json(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_build_manifest_from_json_file(tmp_path):
    raw = _write_json(tmp_path / "ori_pqal.json", FIXTURE_RECORDS)
    out = pubmedqa.build_manifest(raw, tmp_path / "pubmedqa.csv")
    cases = load_cases(out)

    assert len(cases) == 3
    assert all(c.modality is Modality.TEXT for c in cases)

    # Options are always the fixed yes/no/maybe triple, in that order.
    assert all(c.options == ("yes", "no", "maybe") for c in cases)

    # options[answer_index] resolves to each record's final_decision.
    for case, (pmid, record) in zip(cases, FIXTURE_RECORDS.items()):
        assert case.options[case.answer_index] == record["final_decision"]
        assert case.case_id == pmid

    assert cases[0].answer_index == 0  # "yes"
    assert cases[1].answer_index == 1  # "no"
    assert cases[2].answer_index == 2  # "maybe"


def test_case_has_empty_patient_id_in_text_lane():
    case = pubmedqa._case_from_obj("10000001", FIXTURE_RECORDS["10000001"], 0)
    assert case.patient_id == ""
    assert case.modality is Modality.TEXT


def test_case_context_and_meta_mapping():
    # Checked directly off _case_from_obj: manifest round trip for report/meta is a
    # separate concern from the schema mapping itself (see #142's note on meta round trip).
    case = pubmedqa._case_from_obj("10000001", FIXTURE_RECORDS["10000001"], 0)
    assert case.report == (
        "Background: condition Y is common. Results: mortality fell 20%."
    )
    assert case.meta["long_answer"] == FIXTURE_RECORDS["10000001"]["LONG_ANSWER"]
    assert case.meta["meshes"] == ["Drug X", "Condition Y"]
    assert case.meta["year"] == "2019"


def test_build_manifest_from_directory(tmp_path):
    _write_json(tmp_path / "ori_pqal.json", FIXTURE_RECORDS)
    out = pubmedqa.build_manifest(tmp_path, tmp_path / "pubmedqa.csv")
    cases = load_cases(out)
    assert len(cases) == 3
    assert cases[0].options[cases[0].answer_index] == "yes"


def test_build_manifest_respects_limit(tmp_path):
    raw = _write_json(tmp_path / "ori_pqal.json", FIXTURE_RECORDS)
    out = pubmedqa.build_manifest(raw, tmp_path / "pubmedqa.csv", limit=2)
    cases = load_cases(out)
    assert len(cases) == 2
    assert [c.case_id for c in cases] == ["10000001", "10000002"]


def test_missing_directory_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="ori_pqal.json"):
        pubmedqa.build_manifest(tmp_path, tmp_path / "pubmedqa.csv")


def test_bad_final_decision_raises(tmp_path):
    bad = {"99999999": dict(FIXTURE_RECORDS["10000001"], final_decision="unsure")}
    raw = _write_json(tmp_path / "ori_pqal.json", bad)
    with pytest.raises(ValueError, match="final_decision"):
        pubmedqa.build_manifest(raw, tmp_path / "pubmedqa.csv")