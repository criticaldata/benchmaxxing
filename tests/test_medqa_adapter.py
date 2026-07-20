"""Tests for the MedQA-USMLE adapter (text / MCQ, Lane B)."""

from __future__ import annotations

import json

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import medqa
from benchmaxxing.schema import Modality

# Three synthetic records in the real MedQA-USMLE (Jin et al. 2020) JSONL shape.
FIXTURE_ROWS = [
    {
        "question": "A 55-year-old man presents with crushing chest pain. Best next step?",
        "answer": "Aspirin",
        "options": {"A": "Aspirin", "B": "Warfarin", "C": "Heparin", "D": "Clopidogrel"},
        "answer_idx": "A",
        "meta_info": "step1",
        "metamap_phrases": ["chest pain"],
    },
    {
        "question": "Which vessel carries deoxygenated blood to the lungs?",
        "answer": "Pulmonary artery",
        "options": {
            "A": "Aorta",
            "B": "Vena cava",
            "C": "Pulmonary artery",
            "D": "Pulmonary vein",
            "E": "Coronary artery",
        },
        "answer_idx": "C",
        "meta_info": "step1",
        "metamap_phrases": ["vessel", "lungs"],
    },
    {
        "question": "A febrile child has a barking cough. Most likely diagnosis?",
        "answer": "Croup",
        "options": {"A": "Asthma", "B": "Croup", "C": "Pneumonia", "D": "Bronchiolitis"},
        "answer_idx": "B",
        "meta_info": "step2",
        "metamap_phrases": ["cough"],
    },
]


def _write_jsonl(path, rows):
    lines = "\n".join(json.dumps(row) for row in rows)
    path.write_text(lines + "\n", encoding="utf-8")
    return path


def test_build_manifest_from_jsonl_file(tmp_path):
    raw = _write_jsonl(tmp_path / "test.jsonl", FIXTURE_ROWS)
    out = medqa.build_manifest(raw, tmp_path / "medqa.csv")
    cases = load_cases(out)

    assert len(cases) == 3
    assert all(c.modality is Modality.TEXT for c in cases)

    # Options are ordered by sorted key (A, B, C, D[, E]).
    assert cases[0].options == ("Aspirin", "Warfarin", "Heparin", "Clopidogrel")
    assert cases[1].options == (
        "Aorta",
        "Vena cava",
        "Pulmonary artery",
        "Pulmonary vein",
        "Coronary artery",
    )

    # options[answer_index] resolves to the correct answer text for every row.
    for case, row in zip(cases, FIXTURE_ROWS):
        assert case.options[case.answer_index] == row["answer"]

    # answer_idx letter maps to its position in sorted key order.
    assert cases[0].answer_index == 0  # "A"
    assert cases[1].answer_index == 2  # "C"
    assert cases[2].answer_index == 1  # "B"

    # meta_info flows into the label.
    assert [c.label for c in cases] == ["step1", "step1", "step2"]

    # case_id is stable and generated when the record has none.
    assert [c.case_id for c in cases] == ["medqa-0", "medqa-1", "medqa-2"]


def test_case_has_empty_patient_id_in_text_lane():
    # build_manifest emits text-lane cases with no patient (patient_id="").
    case = medqa._case_from_obj(FIXTURE_ROWS[0], 0)
    assert case.patient_id == ""
    assert case.modality is Modality.TEXT


def test_build_manifest_from_directory(tmp_path):
    _write_jsonl(tmp_path / "test.jsonl", FIXTURE_ROWS)
    out = medqa.build_manifest(tmp_path, tmp_path / "medqa.csv")
    cases = load_cases(out)
    assert len(cases) == 3
    assert cases[1].options[cases[1].answer_index] == "Pulmonary artery"


def test_build_manifest_respects_limit(tmp_path):
    raw = _write_jsonl(tmp_path / "test.jsonl", FIXTURE_ROWS)
    out = medqa.build_manifest(raw, tmp_path / "medqa.csv", limit=2)
    cases = load_cases(out)
    assert len(cases) == 2
    assert [c.case_id for c in cases] == ["medqa-0", "medqa-1"]


def test_build_manifest_uses_provided_id(tmp_path):
    rows = [dict(FIXTURE_ROWS[0], id="usmle-42")]
    raw = _write_jsonl(tmp_path / "test.jsonl", rows)
    out = medqa.build_manifest(raw, tmp_path / "medqa.csv")
    cases = load_cases(out)
    assert cases[0].case_id == "usmle-42"


def test_missing_directory_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="test.jsonl"):
        medqa.build_manifest(tmp_path, tmp_path / "medqa.csv")


def test_bad_answer_idx_raises(tmp_path):
    bad = dict(FIXTURE_ROWS[0], answer_idx="Z")
    raw = _write_jsonl(tmp_path / "test.jsonl", [bad])
    with pytest.raises(ValueError, match="answer_idx"):
        medqa.build_manifest(raw, tmp_path / "medqa.csv")
