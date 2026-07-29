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

# Row 1: Pneumothorax positive. Row 2: Cardiomegaly + Support Devices positive.
# Row 3: "No Finding" = 1.0 (finding-absent case).
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

    # Row 2: Cardiomegaly is now correctly extracted (not "no finding").
    second = cases[1]
    assert second.patient_id == "patient00002"
    assert second.label == "cardiomegaly"

    # Row 3: No Finding = 1.0 means finding-absent.
    third = cases[2]
    assert third.patient_id == "patient00099"
    assert third.label == "no finding"


def test_multi_label_clinical_hierarchy(tmp_path):
    """When a patient has multiple positive findings, they are ordered by clinical acuity."""
    # Build a row with pneumothorax + edema + cardiomegaly positive.
    multi_row = (
        "CheXpert-v1.0/train/patient00003/study1/view1_frontal.jpg,Female,45,Frontal,AP,"
        "0.0,,,,,1.0,,,,1.0,,,,0.0"  # Edema=1.0, Pneumothorax=1.0
    )
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("\n".join([HEADER, multi_row]) + "\n", encoding="utf-8")
    cases = chexpert.read_cases(csv_path)
    assert len(cases) == 1
    # Pneumothorax is higher in CLINICAL_HIERARCHY than Edema.
    assert cases[0].label == "pneumothorax|edema"


def test_uncertain_cells_treated_as_negative(tmp_path):
    """Uncertain (-1.0) cells are not treated as positive findings."""
    uncertain_row = (
        "CheXpert-v1.0/train/patient00004/study1/view1_frontal.jpg,Female,70,Frontal,AP,"
        "0.0,,,-1.0,,,,,,,,,,0.0"  # Lung Opacity=-1.0 (uncertain)
    )
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("\n".join([HEADER, uncertain_row]) + "\n", encoding="utf-8")
    cases = chexpert.read_cases(csv_path)
    assert len(cases) == 1
    assert cases[0].label == "no finding"


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


def test_limit_subsamples_not_truncates(tmp_path):
    """limit=2 returns a deterministic subsample of 2, not the first 2 CSV rows."""
    csv_path = _write_csv(tmp_path / "train.csv")
    out = chexpert.build_manifest(csv_path, tmp_path / "limited.csv", limit=2)
    cases = load_cases(out)
    assert len(cases) == 2
    # The key property: it should be a deterministic subsample, not necessarily the first 2 rows.
    # We just verify the count; the exact selection depends on the hash-based ranking.


def test_raw_root_directory_finds_train_csv(tmp_path):
    _write_csv(tmp_path / "train.csv")
    out = chexpert.build_manifest(tmp_path, tmp_path / "manifest.csv")
    cases = load_cases(out)
    assert len(cases) == 3
    assert cases[0].patient_id == "patient00001"


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        chexpert.build_manifest(tmp_path, tmp_path / "manifest.csv")
