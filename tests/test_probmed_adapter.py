"""Tests for the ProbMed adapter.

Fixtures mirror the real ProbMed test.json layout: a flat JSON list of QA records grouped by
consecutive ``id``, each with ``id``/``gpt_idx``/``image``/``image_type``/``qa_type``/
``question``/``answer``. One image yields ~9 questions spanning all five families.
"""

from __future__ import annotations

import json

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import probmed, registry
from benchmaxxing import runner
from benchmaxxing.config import Config
from benchmaxxing.schema import Modality, TwinPair

_RECORDS = [
    {"id": "img1", "gpt_idx": 0, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "modality", "question": "Is this an X-ray?", "answer": "yes"},
    {"id": "img1", "gpt_idx": 1, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "modality", "question": "Is this an MRI?", "answer": "no"},
    {"id": "img1", "gpt_idx": 2, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "body_part", "question": "Is this a chest scan?", "answer": "yes"},
    {"id": "img1", "gpt_idx": 3, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "body_part", "question": "Is this an abdominal scan?", "answer": "no"},
    {"id": "img1", "gpt_idx": 4, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "abnormality", "question": "Is there any abnormality?", "answer": "yes"},
    {"id": "img1", "gpt_idx": 5, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "entity_gt_1", "question": "Is there pneumonia?", "answer": "yes"},
    {"id": "img1", "gpt_idx": 6, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "entity_1", "question": "Is there a rib fracture?", "answer": "no"},
    {"id": "img1", "gpt_idx": 7, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "grounding_gt_2", "question": "Is the pneumonia in the left lung?", "answer": "yes"},
    {"id": "img1", "gpt_idx": 8, "image": "probmed/img1.jpg", "image_type": "X-ray - Chest",
     "qa_type": "grounding_2", "question": "Is the pneumonia in the right lung?", "answer": "no"},
]


def _write_test_json(dir_path, records=_RECORDS):
    path = dir_path / "test.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _build(tmp_path, records=_RECORDS):
    _write_test_json(tmp_path, records)
    out = probmed.build_manifest(tmp_path, tmp_path / "manifest.csv")
    return load_cases(out)


def test_registered():
    assert registry.get("probmed") is probmed
    assert probmed.SPEC.name == "probmed"
    assert probmed.SPEC.modality is Modality.TEXT


def test_every_record_becomes_a_yes_no_text_case(tmp_path):
    cases = _build(tmp_path)
    assert len(cases) == len(_RECORDS)
    for case in cases:
        assert case.modality is Modality.TEXT
        assert case.options == ("yes", "no")
        assert case.answer_index in (0, 1)
        assert case.image_ref == "probmed/img1.jpg"
        assert case.meta["image_ref"] == "probmed/img1.jpg"
        assert case.meta["image_type"] == "X-ray - Chest"
        assert case.patient_id == "img1"


def test_answer_maps_yes_to_0_no_to_1(tmp_path):
    cases = _build(tmp_path)
    uniq = {c.meta["qa_type"]: c for c in cases}
    assert uniq["entity_gt_1"].answer_index == 0
    assert uniq["entity_1"].answer_index == 1
    assert uniq["grounding_2"].answer_index == 1
    modality = [c for c in cases if c.label == "modality"]
    assert [c.answer_index for c in modality] == [0, 1]


def test_families_and_labels(tmp_path):
    by_qa = {c.meta["qa_type"]: c for c in _build(tmp_path)}
    assert by_qa["modality"].label == "modality"
    assert by_qa["body_part"].label == "body_part"
    assert by_qa["abnormality"].label == "abnormality"
    assert by_qa["entity_gt_1"].label == "entity"
    assert by_qa["grounding_gt_2"].label == "grounding"


def test_polarity_positional_for_modality_and_body_part(tmp_path):
    cases = _build(tmp_path)
    modality = [c for c in cases if c.label == "modality"]
    assert [c.meta["polarity"] for c in modality] == ["gt", "adversarial"]
    body_part = [c for c in cases if c.label == "body_part"]
    assert [c.meta["polarity"] for c in body_part] == ["gt", "adversarial"]


def test_gt_and_adversarial_share_group_key(tmp_path):
    by_qa = {c.meta["qa_type"]: c for c in _build(tmp_path)}
    assert by_qa["entity_gt_1"].meta["group_key"] == by_qa["entity_1"].meta["group_key"]
    assert by_qa["entity_gt_1"].meta["polarity"] == "gt"
    assert by_qa["entity_1"].meta["polarity"] == "adversarial"
    assert by_qa["grounding_gt_2"].meta["group_key"] == by_qa["grounding_2"].meta["group_key"]
    assert by_qa["entity_gt_1"].meta["group_key"] != by_qa["grounding_gt_2"].meta["group_key"]
    assert by_qa["abnormality"].meta["polarity"] == "single"


def test_entity_hallu_is_an_unpaired_entity_question(tmp_path):
    records = [
        {
            "id": "normal",
            "gpt_idx": 9,
            "image": "probmed/normal.jpg",
            "qa_type": "entity_hallu",
            "question": "Is there a pneumothorax?",
            "answer": "no",
        }
    ]
    case = _build(tmp_path, records)[0]
    assert case.label == "entity"
    assert case.answer_index == 1
    assert case.meta["polarity"] == "single"
    assert case.meta["group_key"] == "normal:entity:hallu"


def test_case_ids_are_unique(tmp_path):
    cases = _build(tmp_path)
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_round_trip_preserves_meta_and_answer(tmp_path):
    cases = _build(tmp_path)
    entity = next(c for c in cases if c.meta["qa_type"] == "entity_1")
    assert entity.meta == {
        "qa_type": "entity_1",
        "family": "entity",
        "polarity": "adversarial",
        "group_key": "img1:entity:1",
        "image_ref": "probmed/img1.jpg",
        "gt_answer": "no",
        "image_type": "X-ray - Chest",
    }
    assert entity.options == ("yes", "no")
    assert entity.answer_index == 1
    assert entity.image_ref == "probmed/img1.jpg"


def test_image_cue_runner_sends_probmed_question_options_and_pixels(tmp_path, monkeypatch):
    import numpy as np

    image_dir = tmp_path / "probmed"
    image_dir.mkdir()
    (image_dir / "img1.jpg").write_bytes(b"image fixture")
    # Keep this pipeline test independent of the optional Pillow image decoder/renderers. The
    # full image-cue tests cover those dependencies; here the important contract is that the
    # decoded pixels stay attached to ProbMed's own MCQ through run_stage.
    monkeypatch.setattr(
        runner, "load_image", lambda _path: np.full((32, 32), 120, dtype="uint8")
    )
    monkeypatch.setattr(runner, "_cue_types", lambda _cue_set: ("cable",))
    monkeypatch.setattr(
        runner,
        "build_image_twin",
        lambda image, cue_type, ground_truth, case_id: TwinPair(
            case_id=case_id,
            cue_type=cue_type,
            clean=image.copy(),
            contaminated=image + 1,
            ground_truth=ground_truth,
        ),
    )
    case = _build(tmp_path)[0]

    class RecordingBackend:
        def __init__(self):
            self.calls = []

        def complete(self, prompt, image=None, decoding=None):
            self.calls.append((prompt, image))
            return "A"

    backend = RecordingBackend()
    config = Config.from_dict({"models": ["vision-model"], "cue_set": "image-v1"})
    result = runner.run_stage(
        "pilot", [case], config, lambda _model: backend, image_root=tmp_path
    )

    assert result["n"] == 1
    # clean and contaminated calls for each image cue use ProbMed's real binary MCQ, not the
    # generic NIH finding question, and both receive image pixels.
    assert len(backend.calls) == 2
    assert all("Question: Is this an X-ray?" in prompt for prompt, _ in backend.calls)
    assert all("A. yes" in prompt and "B. no" in prompt for prompt, _ in backend.calls)
    assert all(image is not None and image.shape == (32, 32) for _, image in backend.calls)


def test_raw_root_accepts_the_json_file_directly(tmp_path):
    path = _write_test_json(tmp_path)
    out = probmed.build_manifest(path, tmp_path / "m.csv")
    assert len(load_cases(out)) == len(_RECORDS)


def test_limit(tmp_path):
    _write_test_json(tmp_path)
    out = probmed.build_manifest(tmp_path, tmp_path / "m.csv", limit=3)
    assert len(load_cases(out)) == 3


def test_missing_test_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="test.json"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_missing_required_key_raises(tmp_path):
    bad = [{"id": "i", "gpt_idx": 0, "qa_type": "abnormality", "question": "?"}]
    _write_test_json(tmp_path, bad)
    with pytest.raises(ValueError, match="missing required key"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_non_binary_answer_raises(tmp_path):
    bad = [{"id": "i", "gpt_idx": 0, "image": "a.jpg", "qa_type": "modality",
            "question": "What modality?", "answer": "X-ray"}]
    _write_test_json(tmp_path, bad)
    with pytest.raises(ValueError, match="must be 'yes' or 'no'"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


@pytest.mark.parametrize("qa_type", ["abnormality_gt", "Modality", "position", ""])
def test_unknown_qa_type_raises_instead_of_becoming_grounding(tmp_path, qa_type):
    bad = [{"id": "i", "gpt_idx": 0, "image": "a.jpg", "qa_type": qa_type,
            "question": "?", "answer": "yes"}]
    _write_test_json(tmp_path, bad)
    with pytest.raises(ValueError, match="Unrecognized ProbMed qa_type"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")


def test_non_list_json_raises(tmp_path):
    (tmp_path / "test.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON list"):
        probmed.build_manifest(tmp_path, tmp_path / "m.csv")
