"""Tests for the MedMCQA adapter (text / MCQ, Lane B)."""

from __future__ import annotations

import json

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import medmcqa
from benchmaxxing.schema import Modality

# Three synthetic records in the real MedMCQA (Pal et al. 2022) JSON shape.
FIXTURE_ROWS = [
    {
        "question": "Which drug is first-line for an acute MI with chest pain?",
        "opa": "Aspirin",
        "opb": "Warfarin",
        "opc": "Heparin",
        "opd": "Clopidogrel",
        "cop": 0,
        "subject_name": "Pharmacology",
        "topic_name": "Cardiovascular drugs",
    },
    {
        "question": "Which vessel carries deoxygenated blood to the lungs?",
        "opa": "Aorta",
        "opb": "Vena cava",
        "opc": "Pulmonary artery",
        "opd": "Pulmonary vein",
        "cop": 2,
        "subject_name": "Anatomy",
        "topic_name": "Thorax",
    },
    {
        "question": "A febrile child has a barking cough. Most likely diagnosis?",
        "opa": "Asthma",
        "opb": "Croup",
        "opc": "Pneumonia",
        "opd": "Bronchiolitis",
        "cop": 1,
        "subject_name": "Pediatrics",
        "topic_name": "Respiratory",
    },
]


def _write_jsonl(path, rows):
    lines = "\n".join(json.dumps(row) for row in rows)
    path.write_text(lines + "\n", encoding="utf-8")
    return path


def test_build_manifest_from_jsonl_file(tmp_path):
    raw = _write_jsonl(tmp_path / "dev.json", FIXTURE_ROWS)
    out = medmcqa.build_manifest(raw, tmp_path / "medmcqa.csv")
    cases = load_cases(out)

    assert len(cases) == 3
    assert all(c.modality is Modality.TEXT for c in cases)

    # Options are always ordered opa, opb, opc, opd.
    assert cases[0].options == ("Aspirin", "Warfarin", "Heparin", "Clopidogrel")
    assert cases[1].options == ("Aorta", "Vena cava", "Pulmonary artery", "Pulmonary vein")

    # options[answer_index] resolves to the correct answer text, cop maps directly.
    assert cases[0].options[cases[0].answer_index] == "Aspirin"
    assert cases[1].options[cases[1].answer_index] == "Pulmonary artery"
    assert cases[2].options[cases[2].answer_index] == "Croup"
    assert [c.answer_index for c in cases] == [0, 2, 1]

    # case_id is stable and generated when the record has none.
    assert [c.case_id for c in cases] == ["medmcqa-0", "medmcqa-1", "medmcqa-2"]


def test_case_has_empty_patient_id_in_text_lane():
    case = medmcqa._case_from_obj(FIXTURE_ROWS[0], 0)
    assert case.patient_id == ""
    assert case.modality is Modality.TEXT


def test_subject_and_topic_flow_into_meta():
    # meta is a Case-level field the adapter populates; the manifest round trip (write_manifest
    # / load_cases) does not yet preserve it on main, so this checks the adapter's own output.
    case = medmcqa._case_from_obj(FIXTURE_ROWS[0], 0)
    assert case.meta == {"subject_name": "Pharmacology", "topic_name": "Cardiovascular drugs"}


def test_build_manifest_from_directory(tmp_path):
    _write_jsonl(tmp_path / "dev.json", FIXTURE_ROWS)
    out = medmcqa.build_manifest(tmp_path, tmp_path / "medmcqa.csv")
    cases = load_cases(out)
    assert len(cases) == 3
    assert cases[1].options[cases[1].answer_index] == "Pulmonary artery"


def test_build_manifest_respects_limit(tmp_path):
    raw = _write_jsonl(tmp_path / "dev.json", FIXTURE_ROWS)
    out = medmcqa.build_manifest(raw, tmp_path / "medmcqa.csv", limit=2)
    cases = load_cases(out)
    assert len(cases) == 2
    assert [c.case_id for c in cases] == ["medmcqa-0", "medmcqa-1"]


def test_build_manifest_uses_provided_id(tmp_path):
    rows = [dict(FIXTURE_ROWS[0], id="mcqa-42")]
    raw = _write_jsonl(tmp_path / "dev.json", rows)
    out = medmcqa.build_manifest(raw, tmp_path / "medmcqa.csv")
    cases = load_cases(out)
    assert cases[0].case_id == "mcqa-42"


def test_missing_directory_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="dev.json"):
        medmcqa.build_manifest(tmp_path, tmp_path / "medmcqa.csv")


def test_bad_cop_raises(tmp_path):
    bad = dict(FIXTURE_ROWS[0], cop=7)
    raw = _write_jsonl(tmp_path / "dev.json", [bad])
    with pytest.raises(ValueError, match="cop"):
        medmcqa.build_manifest(raw, tmp_path / "medmcqa.csv")
