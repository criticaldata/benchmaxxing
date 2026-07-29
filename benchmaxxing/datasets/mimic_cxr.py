"""MIMIC-CXR adapter (imaging, Lane A).

Parses the credentialed MIMIC-CXR-JPG v2.0.0 raw layout (metadata CSV + CheXpert label CSV +
per-study free-text report files) into one schema.Case per image, emitted through
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality
from benchmaxxing.utils.clinical_labels import CLINICAL_HIERARCHY, FINDING_COLUMNS, _POSITIVE

SPEC = DatasetSpec(
    name="mimic_cxr",
    raw_hint=(
        "PhysioNet MIMIC-CXR-JPG v2.0.0: a 'files/pXX/pYYYY/sZZZZ/*.jpg' tree of frontal/lateral "
        "chest X-rays plus 'mimic-cxr-2.0.0-chexpert.csv' (per-study CheXpert labels) and "
        "'mimic-cxr-2.0.0-metadata.csv'. Requires credentialed PhysioNet access."
    ),
    modality=Modality.IMAGE,
    notes=(
        "One Case per image: case_id=dicom id, patient_id=subject id, image_ref=relative jpg "
        "path, report=study free-text report when present, label='pneumothorax' when the "
        "CheXpert Pneumothorax cell is 1.0 else 'no finding'."
    ),
)

_METADATA_CSV = "mimic-cxr-2.0.0-metadata.csv"
_CHEXPERT_CSV = "mimic-cxr-2.0.0-chexpert.csv"


def build_manifest(raw_root, out, limit=None, label_format="legacy"):
    """Build a per-image manifest from the raw MIMIC-CXR-JPG release under ``raw_root``.

    Joins the metadata CSV to the CheXpert CSV on (subject_id, study_id) and emits one
    schema.Case per dicom_id. The study report text is attached when the ``.txt`` file exists,
    otherwise ``report`` is None. Writes the manifest to ``out`` via ``finalize`` and returns
    the list of Cases (at most ``limit`` when given).
    """
    root = Path(raw_root)
    metadata_rows = _read_csv(root / _METADATA_CSV)
    chexpert = {
        (_cell(row, "subject_id"), _cell(row, "study_id")): row
        for row in _read_csv(root / _CHEXPERT_CSV)
    }
    report_cache: dict[Path, str | None] = {}
    cases: list[Case] = []
    for row in metadata_rows:
        if limit is not None and len(cases) >= limit:
            break
        subject_id = _cell(row, "subject_id")
        study_id = _cell(row, "study_id")
        dicom_id = _cell(row, "dicom_id")
        labels = chexpert.get((subject_id, study_id), {})
        
        # Uncertainty handling (-1.0) and NaNs are implicitly treated as negative by strictly matching _POSITIVE.
        if label_format == "stratified":
            positives = [
                col for col in CLINICAL_HIERARCHY
                if _cell(labels, col) == _POSITIVE
            ]
            label_str = "|".join(positives).lower() if positives else "no finding"
        else:
            label_str = "pneumothorax" if _cell(labels, "Pneumothorax") == _POSITIVE else "no finding"

        study_rel = f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}"
        report_path = root / "files" / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"
        cases.append(
            Case(
                case_id=dicom_id,
                patient_id=subject_id,
                modality=Modality.IMAGE,
                label=label_str,
                image_ref=f"{study_rel}/{dicom_id}.jpg",
                report=_load_report(report_path, report_cache),
                meta={
                    "study_id": study_id,
                    "view": _cell(row, "ViewPosition") or None,
                    "support_devices": _cell(labels, "Support Devices") == _POSITIVE,
                    "labels": {col: _cell(labels, col) for col in FINDING_COLUMNS},
                },
            )
        )
    finalize(cases, out)
    return cases


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected MIMIC-CXR-JPG file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(row: dict, key: str) -> str:
    """Return the stripped string value of a CSV cell, mapping missing/None to ''."""
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _load_report(path: Path, cache: dict) -> str | None:
    """Read a study report once, returning None when the file does not exist."""
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8") if path.is_file() else None
    return cache[path]
