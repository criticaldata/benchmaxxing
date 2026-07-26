"""Tests for the MIMIC-CXR text adapter: tiny synthetic reports + CheXpert CSV to an MCQ manifest."""

from __future__ import annotations

import gzip

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import mimic_cxr_text
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

# subject_id/study_id follow the real MIMIC-CXR convention: numeric strings, no "p"/"s" prefix
# (the adapter adds those when building report paths and case ids).
#  study 50000001 (subject 10000032): exactly one positive (Pneumothorax), >=3 confirmed
#    negatives -> kept.
#  study 50000002 (subject 10000032): "No Finding" positive -> excluded (no positive to answer).
#  study 50000003 (subject 10000898): two positives (Pneumonia + Edema) -> excluded (ambiguous
#    "primary finding").
#  study 50000004 (subject 10000898): one positive (Fracture) but only 2 confirmed negatives
#    (rest uncertain/blank) -> excluded.
#  study 50000005 (subject 10001217): one positive (Cardiomegaly), enough negatives, but no
#    report file on disk -> excluded.
_ROWS = {
    ("10000032", "50000001"): {"Pneumothorax": "1.0", "Atelectasis": "0.0", "Cardiomegaly": "0.0",
                                "Edema": "0.0"},
    ("10000032", "50000002"): {"No Finding": "1.0"},
    ("10000898", "50000003"): {"Pneumonia": "1.0", "Edema": "1.0", "Atelectasis": "0.0",
                                "Cardiomegaly": "0.0", "Fracture": "0.0"},
    ("10000898", "50000004"): {"Fracture": "1.0", "Atelectasis": "0.0", "Cardiomegaly": "0.0"},
    ("10001217", "50000005"): {"Cardiomegaly": "1.0", "Atelectasis": "0.0", "Edema": "0.0",
                                "Fracture": "0.0"},
}

_NO_REPORT_STUDY = "50000005"

REPORT_TEXT = "FINDINGS: Large right pneumothorax with a chest tube in place.\n"


def _chexpert_row(subject_id, study_id, cells):
    values = [subject_id, study_id] + [cells.get(name, "") for name in CHEXPERT_LABELS]
    return ",".join(values)


def _write_report(reports_root, subject_id, study_id):
    study_dir = reports_root / f"p{subject_id[:2]}" / f"p{subject_id}"
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / f"s{study_id}.txt").write_text(REPORT_TEXT, encoding="utf-8")


def _make_raw(tmp_path, gz=False):
    reports = tmp_path / "reports"
    labels = tmp_path / "labels"
    labels.mkdir()

    header = "subject_id,study_id," + ",".join(CHEXPERT_LABELS) + "\n"
    body = "".join(_chexpert_row(subj, study, cells) + "\n" for (subj, study), cells in _ROWS.items())
    csv_text = header + body
    if gz:
        with gzip.open(labels / "mimic-cxr-2.0.0-chexpert.csv.gz", "wt", encoding="utf-8") as handle:
            handle.write(csv_text)
    else:
        (labels / "mimic-cxr-2.0.0-chexpert.csv").write_text(csv_text, encoding="utf-8")

    # Report files for every study except the one missing on purpose.
    for subject_id, study_id in _ROWS:
        if study_id == _NO_REPORT_STUDY:
            continue
        _write_report(reports, subject_id, study_id)

    return reports, labels


def test_filters_ambiguous_and_no_finding_and_underdetermined_studies(tmp_path):
    reports, labels = _make_raw(tmp_path)
    out = tmp_path / "manifest.csv"
    cases = mimic_cxr_text.build_manifest(reports, labels, out)

    # Only study 50000001 survives: 50000002 (No Finding), 50000003 (2 positives),
    # 50000004 (too few negatives), 50000005 (no report file).
    assert [c.case_id for c in cases] == ["mimic-cxr-text-10000032-50000001"]
    assert cases[0].modality is Modality.TEXT


def test_mcq_shape_and_answer_index(tmp_path):
    reports, labels = _make_raw(tmp_path)
    cases = mimic_cxr_text.build_manifest(reports, labels, tmp_path / "m.csv")
    case = cases[0]

    assert len(case.options) == 4
    assert case.options[case.answer_index] == "Pneumothorax"
    assert case.label == "pneumothorax"
    assert REPORT_TEXT.strip() in case.question
    assert "primary finding" in case.question
    assert case.patient_id == "10000032"
    assert case.meta["study_id"] == "50000001"
    assert len(case.meta["distractors"]) == 3
    assert "Pneumothorax" not in case.meta["distractors"]


def test_gzipped_labels_csv_supported(tmp_path):
    reports, labels = _make_raw(tmp_path, gz=True)
    cases = mimic_cxr_text.build_manifest(reports, labels, tmp_path / "m.csv")
    assert [c.case_id for c in cases] == ["mimic-cxr-text-10000032-50000001"]


def test_manifest_written_and_loadable(tmp_path):
    reports, labels = _make_raw(tmp_path)
    out = tmp_path / "manifest.csv"
    mimic_cxr_text.build_manifest(reports, labels, out)
    reloaded = load_cases(out)
    assert [c.case_id for c in reloaded] == ["mimic-cxr-text-10000032-50000001"]
    assert reloaded[0].options[reloaded[0].answer_index] == "Pneumothorax"


def test_limit_is_respected(tmp_path):
    # Add a second eligible study so two cases qualify, then confirm limit caps at 1.
    reports, labels = _make_raw(tmp_path)
    _write_report(reports, "10009999", "50000009")
    csv_path = labels / "mimic-cxr-2.0.0-chexpert.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8")
        + _chexpert_row("10009999", "50000009",
                         {"Pneumothorax": "1.0", "Atelectasis": "0.0",
                          "Cardiomegaly": "0.0", "Edema": "0.0"})
        + "\n",
        encoding="utf-8",
    )
    unlimited = mimic_cxr_text.build_manifest(reports, labels, tmp_path / "full.csv")
    assert [c.case_id for c in unlimited] == [
        "mimic-cxr-text-10000032-50000001",
        "mimic-cxr-text-10009999-50000009",
    ]

    limited = mimic_cxr_text.build_manifest(reports, labels, tmp_path / "limited.csv", limit=1)
    assert [c.case_id for c in limited] == ["mimic-cxr-text-10000032-50000001"]


def test_missing_labels_csv_raises(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    with pytest.raises(FileNotFoundError):
        mimic_cxr_text.build_manifest(reports, tmp_path / "empty_labels", tmp_path / "out.csv")
