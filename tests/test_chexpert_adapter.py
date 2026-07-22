"""Tests for the CheXpert adapter: parse a tiny synthetic train.csv into schema.Case rows."""

from __future__ import annotations

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import chexpert
from benchmaxxing.schema import Modality

HEADER = (
    "Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,"
    "Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,"
    "Pleural Effusion,Pleural Other,Fracture,Support Devices"
)

# Row 1: Pneumothorax positive. Row 2: Support Devices positive, no pneumothorax.
# Row 3: all blank / no finding.
ROWS = [
    "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg,Female,55,Frontal,AP,"
    "0.0,,,,,,,,,1.0,,,,0.0",
    "CheXpert-v1.0/train/patient00002/study3/view1_frontal.jpg,Male,61,Frontal,PA,"
    "0.0,,1.0,,,,,,,0.0,,,,1.0",
    "CheXpert-v1.0/valid/patient00099/study1/view2_lateral.jpg,Male,30,Lateral,,"
    "1.0,,,,,,,,,,,,,",
]


def _write_csv(path):
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def test_spec_name_is_chexpert():
    assert chexpert.SPEC.name == "chexpert"


def test_build_manifest_parses_rows(tmp_path):
    csv_path = _write_csv(tmp_path / "train.csv")
    out = chexpert.build_manifest(csv_path, tmp_path / "manifest.csv")
    cases = load_cases(out)

    assert len(cases) == 3
    assert all(c.modality is Modality.IMAGE for c in cases)

    first = cases[0]
    assert first.case_id == "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
    assert first.image_ref == first.case_id
    assert first.patient_id == "patient00001"
    assert first.label == "pneumothorax"

    second = cases[1]
    assert second.patient_id == "patient00002"
    assert second.label == "no finding"

    third = cases[2]
    assert third.patient_id == "patient00099"
    assert third.label == "no finding"


def test_support_devices_flag_in_meta(tmp_path):
    csv_path = _write_csv(tmp_path / "train.csv")
    # meta is not persisted to the CSV manifest, so inspect the Cases the parser builds directly.
    cases = chexpert.read_cases(csv_path)

    assert cases[0].meta["support_devices"] is False
    assert cases[1].meta["support_devices"] is True
    assert cases[2].meta["support_devices"] is False

    # The 14 finding columns are all captured verbatim.
    labels = cases[1].meta["labels"]
    assert set(labels) == set(chexpert.FINDING_COLUMNS)
    assert len(chexpert.FINDING_COLUMNS) == 14
    assert labels["Cardiomegaly"] == "1.0"
    assert labels["Support Devices"] == "1.0"
    assert labels["Pneumothorax"] == "0.0"
    assert labels["Pleural Other"] == ""


def test_limit_caps_rows(tmp_path):
    csv_path = _write_csv(tmp_path / "train.csv")
    out = chexpert.build_manifest(csv_path, tmp_path / "limited.csv", limit=2)
    cases = load_cases(out)
    assert len(cases) == 2
    assert cases[0].patient_id == "patient00001"
    assert cases[1].patient_id == "patient00002"


def test_raw_root_directory_finds_train_csv(tmp_path):
    _write_csv(tmp_path / "train.csv")
    out = chexpert.build_manifest(tmp_path, tmp_path / "manifest.csv")
    cases = load_cases(out)
    assert len(cases) == 3
    assert cases[0].patient_id == "patient00001"


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        chexpert.build_manifest(tmp_path, tmp_path / "manifest.csv")
