"""Tests for the MIMIC-CXR adapter: tiny synthetic raw tree to a per-image manifest."""

from __future__ import annotations

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import mimic_cxr
from benchmaxxing.schema import Modality

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

REPORT_TEXT = (
    "                                 FINAL REPORT\n"
    "FINDINGS:  Large right pneumothorax with a chest tube in place.\n"
)


def _chexpert_row(subject_id, study_id, **cells):
    values = [subject_id, study_id] + [cells.get(name, "") for name in CHEXPERT_LABELS]
    return ",".join(values)


def _make_raw(tmp_path):
    """Two studies, three images: study s50414267 has a report + pneumothorax, s55555555 not."""
    raw = tmp_path / "raw"
    (raw / "mimic-cxr-2.0.0-metadata.csv").parent.mkdir(parents=True, exist_ok=True)
    (raw / "mimic-cxr-2.0.0-metadata.csv").write_text(
        "dicom_id,subject_id,study_id,ViewPosition,Rows,Columns,StudyDate\n"
        "d1aaa,10000032,50414267,PA,3056,2544,21800506\n"
        "d1bbb,10000032,50414267,LATERAL,3056,2544,21800506\n"
        "d2ccc,10001217,55555555,AP,2544,3056,21810213\n",
        encoding="utf-8",
    )
    (raw / "mimic-cxr-2.0.0-chexpert.csv").write_text(
        "subject_id,study_id," + ",".join(CHEXPERT_LABELS) + "\n"
        + _chexpert_row("10000032", "50414267", **{"Pneumothorax": "1.0", "Support Devices": "1.0"})
        + "\n"
        + _chexpert_row(
            "10001217", "55555555", **{"No Finding": "1.0", "Pneumothorax": "-1.0"}
        )
        + "\n",
        encoding="utf-8",
    )
    study1 = raw / "files" / "p10" / "p10000032" / "s50414267"
    study2 = raw / "files" / "p10" / "p10001217" / "s55555555"
    for study, dicoms in ((study1, ["d1aaa", "d1bbb"]), (study2, ["d2ccc"])):
        study.mkdir(parents=True)
        for dicom_id in dicoms:
            (study / f"{dicom_id}.jpg").write_bytes(b"")
    (study1.parent / "s50414267.txt").write_text(REPORT_TEXT, encoding="utf-8")
    return raw


def test_join_reports_and_labels(tmp_path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "manifest.csv"
    cases = mimic_cxr.build_manifest(raw, out)

    assert [c.case_id for c in cases] == ["d1aaa", "d1bbb", "d2ccc"]
    assert all(c.modality is Modality.IMAGE for c in cases)

    with_report = {c.case_id: c for c in cases}
    # Study with a report file: both of its images carry the exact report text.
    assert with_report["d1aaa"].report == REPORT_TEXT
    assert with_report["d1bbb"].report == REPORT_TEXT
    # Study without a report file: report is None.
    assert with_report["d2ccc"].report is None

    # Pneumothorax label from the CheXpert join ("-1.0" is not positive).
    assert with_report["d1aaa"].label == "pneumothorax"
    assert with_report["d1bbb"].label == "pneumothorax"
    assert with_report["d2ccc"].label == "no finding"

    # Support devices flag and remaining meta.
    assert with_report["d1aaa"].meta["support_devices"] is True
    assert with_report["d2ccc"].meta["support_devices"] is False
    assert with_report["d1aaa"].meta == {
        "study_id": "50414267",
        "view": "PA",
        "support_devices": True,
    }
    assert with_report["d1bbb"].meta["view"] == "LATERAL"
    assert with_report["d2ccc"].meta["view"] == "AP"


def test_image_refs_and_patient_ids(tmp_path):
    raw = _make_raw(tmp_path)
    cases = mimic_cxr.build_manifest(raw, tmp_path / "m.csv")
    by_id = {c.case_id: c for c in cases}
    assert by_id["d1aaa"].image_ref == "files/p10/p10000032/s50414267/d1aaa.jpg"
    assert by_id["d2ccc"].image_ref == "files/p10/p10001217/s55555555/d2ccc.jpg"
    assert by_id["d1aaa"].patient_id == "10000032"
    assert by_id["d2ccc"].patient_id == "10001217"
    # The referenced jpgs exist in the synthetic tree.
    assert all((raw / c.image_ref).is_file() for c in cases)


def test_manifest_written_and_loadable(tmp_path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "manifest.csv"
    mimic_cxr.build_manifest(raw, out)
    reloaded = load_cases(out)
    assert [c.case_id for c in reloaded] == ["d1aaa", "d1bbb", "d2ccc"]
    assert all(c.modality is Modality.IMAGE for c in reloaded)
    assert reloaded[2].report is None


def test_limit_is_respected(tmp_path):
    raw = _make_raw(tmp_path)
    out = tmp_path / "limited.csv"
    cases = mimic_cxr.build_manifest(raw, out, limit=2)
    assert [c.case_id for c in cases] == ["d1aaa", "d1bbb"]
    assert len(load_cases(out)) == 2


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mimic_cxr.build_manifest(tmp_path / "empty", tmp_path / "out.csv")
