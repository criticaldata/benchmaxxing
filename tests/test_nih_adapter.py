"""Tests for the NIH ChestX-ray14 adapter (benchmaxxing.datasets.nih_cxr14).

The manifest CSV round-trip in benchmaxxing.data is lossy for Case.meta (no meta column),
so meta assertions inspect the Cases the parser produces via ``_row_to_case`` directly, while
end-to-end manifest assertions cover the columns that persist.
"""

from __future__ import annotations

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import nih_cxr14
from benchmaxxing.schema import Modality

HEADER = (
    "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,"
    "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]"
)
ROWS = [
    "00000001_000.png,Cardiomegaly|Effusion,0,1,58,M,PA,2048,2500,0.143,0.143",
    "00000002_000.png,No Finding,0,2,81,F,AP,2500,2048,0.168,0.168",
    "00000003_001.png,Atelectasis,1,3,74,M,PA,2992,2991,0.143,0.143",
]


def _write_csv(directory):
    path = directory / "Data_Entry_2017.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def _row(image_index, finding_labels, patient_id, view):
    return {
        "Image Index": image_index,
        "Finding Labels": finding_labels,
        "Patient ID": patient_id,
        "View Position": view,
    }


def test_spec_name_stable():
    assert nih_cxr14.SPEC.name == "nih_cxr14"
    assert nih_cxr14.SPEC.modality is Modality.IMAGE


def test_row_multi_finding_splits():
    case = nih_cxr14._row_to_case(_row("00000001_000.png", "Cardiomegaly|Effusion", "1", "PA"))
    assert case.modality is Modality.IMAGE
    assert case.case_id == "00000001_000.png"
    assert case.image_ref == "00000001_000.png"
    assert case.patient_id == "1"
    assert case.report is None
    # label: pipe-separated Finding Labels, lower-cased
    assert case.label == "cardiomegaly|effusion"
    # findings: split on '|'; view carried through meta
    assert case.meta["findings"] == ["Cardiomegaly", "Effusion"]
    assert case.meta["view"] == "PA"


def test_row_no_finding_single_element():
    case = nih_cxr14._row_to_case(_row("00000002_000.png", "No Finding", "2", "AP"))
    assert case.label == "no finding"
    assert case.meta["findings"] == ["No Finding"]
    assert case.patient_id == "2"
    assert case.meta["view"] == "AP"


def test_build_manifest_from_dir(tmp_path):
    _write_csv(tmp_path)
    out = tmp_path / "manifest.csv"
    # raw_root as the directory containing Data_Entry_2017.csv
    result = nih_cxr14.build_manifest(tmp_path, out)
    assert result == out
    cases = load_cases(out)
    assert len(cases) == 3
    assert all(c.modality is Modality.IMAGE for c in cases)
    assert [c.case_id for c in cases] == [
        "00000001_000.png",
        "00000002_000.png",
        "00000003_001.png",
    ]
    # columns that persist through the manifest CSV
    assert [c.patient_id for c in cases] == ["1", "2", "3"]
    assert cases[0].image_ref == "00000001_000.png"
    assert cases[0].label == "cardiomegaly|effusion"
    assert cases[1].label == "no finding"
    assert cases[0].report is None


def test_build_manifest_from_csv_path(tmp_path):
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "manifest.csv"
    # raw_root pointing directly at the csv file
    nih_cxr14.build_manifest(csv_path, out)
    cases = load_cases(out)
    assert len(cases) == 3


def test_limit_respected(tmp_path):
    _write_csv(tmp_path)
    out = tmp_path / "manifest.csv"
    nih_cxr14.build_manifest(tmp_path, out, limit=2)
    cases = load_cases(out)
    assert len(cases) == 2
    assert cases[-1].case_id == "00000002_000.png"


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data_Entry_2017.csv"):
        nih_cxr14.build_manifest(tmp_path, tmp_path / "manifest.csv")
