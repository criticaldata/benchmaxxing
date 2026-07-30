"""Tests for the ProbMed adapter, using the REAL test.json qa_type vocabulary.

Verified against the gated rippleripple/ProbMed test.json (57,132 records): record keys are
id/i/image/image_type/qa_type/question/answer/src_dataset, and qa_type is
{modality,body_part}_{gt,hallu}, abnormality, entity_{gt,hallu}_<id>, grounding_{gt,hallu}_<id>.
"""

from __future__ import annotations

import json

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import probmed, registry
from benchmaxxing.schema import Modality

_RECORDS = [
    {"id": "img1", "i": 0, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "modality_gt", "question": "Is this an X-ray?", "answer": "yes"},
    {"id": "img1", "i": 1, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "modality_hallu", "question": "Is this an MRI?", "answer": "no"},
    {"id": "img1", "i": 2, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "body_part_gt", "question": "Is this a chest scan?", "answer": "yes"},
    {"id": "img1", "i": 3, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "body_part_hallu", "question": "Is this an abdominal scan?", "answer": "no"},
    {"id": "img1", "i": 4, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "abnormality", "question": "Any abnormality?", "answer": "yes"},
    {"id": "img1", "i": 5, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "entity_gt_0", "question": "Is there pneumonia?", "answer": "yes"},
    {"id": "img1", "i": 6, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "entity_hallu_0", "question": "Is there a rib fracture?", "answer": "no"},
    {"id": "img1", "i": 7, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "grounding_gt_1", "question": "Pneumonia in the left lung?", "answer": "yes"},
    {"id": "img1", "i": 8, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "src_dataset": "MIMIC", "qa_type": "grounding_hallu_1", "question": "Pneumonia in the right lung?", "answer": "no"},
]


def _write(dir_path, records=_RECORDS):
    path = dir_path / "test.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _build(tmp_path, records=_RECORDS):
    _write(tmp_path, records)
    return load_cases(probmed.build_manifest(tmp_path, tmp_path / "manifest.csv"))


def test_registered():
    assert registry.get("probmed") is probmed
    assert probmed.SPEC.name == "probmed"
    assert probmed.SPEC.modality is Modality.TEXT


def test_every_record_is_a_yes_no_text_case(tmp_path):
    cases = _build(tmp_path)
    assert len(cases) == len(_RECORDS)
    for c in cases:
        assert c.modality is Modality.TEXT
        assert c.options == ("yes", "no")
        assert c.answer_index in (0, 1)
        assert c.meta["image_ref"] == "probmed/img1.jpg"
        assert c.meta["image_type"] == "X-ray - Chest"
        assert c.meta["src_dataset"] == "MIMIC"
        assert c.patient_id == "img1"


def test_answer_maps_yes_0_no_1(tmp_path):
    by = {c.meta["qa_type"]: c for c in _build(tmp_path)}
    assert by["modality_gt"].answer_index == 0
    assert by["modality_hallu"].answer_index == 1
    assert by["entity_gt_0"].answer_index == 0
    assert by["entity_hallu_0"].answer_index == 1


def test_families(tmp_path):
    by = {c.meta["qa_type"]: c for c in _build(tmp_path)}
    assert by["modality_gt"].label == "modality"
    assert by["body_part_hallu"].label == "body_part"
    assert by["abnormality"].label == "abnormality"
    assert by["entity_gt_0"].label == "entity"
    assert by["grounding_hallu_1"].label == "grounding"


def test_polarity_from_gt_hallu_tag(tmp_path):
    by = {c.meta["qa_type"]: c for c in _build(tmp_path)}
    assert by["modality_gt"].meta["polarity"] == "gt"
    assert by["modality_hallu"].meta["polarity"] == "adversarial"
    assert by["body_part_gt"].meta["polarity"] == "gt"
    assert by["body_part_hallu"].meta["polarity"] == "adversarial"
    assert by["abnormality"].meta["polarity"] == "single"


def test_gt_and_adversarial_share_group_key(tmp_path):
    by = {c.meta["qa_type"]: c for c in _build(tmp_path)}
    assert by["modality_gt"].meta["group_key"] == by["modality_hallu"].meta["group_key"] == "img1:modality"
    assert by["entity_gt_0"].meta["group_key"] == by["entity_hallu_0"].meta["group_key"] == "img1:entity:0"
    assert by["grounding_gt_1"].meta["group_key"] == by["grounding_hallu_1"].meta["group_key"] == "img1:grounding:1"
    assert by["entity_gt_0"].meta["group_key"] != by["grounding_gt_1"].meta["group_key"]


def test_case_ids_unique(tmp_path):
    ids = [c.case_id for c in _build(tmp_path)]
    assert len(ids) == len(set(ids))


def test_round_trip_preserves_meta(tmp_path):
    ent = next(c for c in _build(tmp_path) if c.meta["qa_type"] == "entity_hallu_0")
    assert ent.meta == {
        "qa_type": "entity_hallu_0", "family": "entity", "polarity": "adversarial",
        "group_key": "img1:entity:0", "image_ref": "probmed/img1.jpg", "gt_answer": "no",
        "image_type": "X-ray - Chest", "src_dataset": "MIMIC",
    }
    assert ent.options == ("yes", "no")
    assert ent.answer_index == 1


def test_raw_root_accepts_file(tmp_path):
    path = _write(tmp_path)
    assert len(load_cases(probmed.build_manifest(path, tmp_path / "m.csv"))) == len(_RECORDS)


def test_limit(tmp_path):
    _write(tmp_path)
    assert len(load_cases(probmed.build_manifest(tmp_path, tmp_path / "m.csv", limit=3))) == 3


def test_missing_test_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="test.json"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_missing_required_key_raises(tmp_path):
    _write(tmp_path, [{"id": "i", "i": 0, "qa_type": "abnormality", "question": "?"}])
    with pytest.raises(ValueError, match="missing required key"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_non_binary_answer_raises(tmp_path):
    _write(tmp_path, [{"id": "i", "i": 0, "image": "a.jpg", "qa_type": "abnormality",
                       "question": "?", "answer": "maybe"}])
    with pytest.raises(ValueError, match="must be 'yes' or 'no'"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_unrecognized_qa_type_raises(tmp_path):
    # Agastya's concern preserved: an unknown qa_type must raise, not silently bucket to grounding.
    _write(tmp_path, [{"id": "i", "i": 0, "image": "a.jpg", "qa_type": "nonsense_type",
                       "question": "?", "answer": "yes"}])
    with pytest.raises(ValueError, match="Unrecognized ProbMed qa_type"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_non_list_json_raises(tmp_path):
    (tmp_path / "test.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON list"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")
