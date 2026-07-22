"""NIH ChestX-ray14 adapter (imaging, Lane A).

Parses ``Data_Entry_2017.csv`` into shared schema.Case rows (one per image) and emits a
manifest via ``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

CSV_NAME = "Data_Entry_2017.csv"

SPEC = DatasetSpec(
    name="nih_cxr14",
    raw_hint=(
        "NIH ChestX-ray14 (NIH Clinical Center, also on Kaggle): 'images_XXX/images/*.png' folders "
        "plus 'Data_Entry_2017.csv' whose 'Finding Labels' column is a pipe-separated list over 14 "
        "pathologies ('No Finding' otherwise), keyed by 'Image Index' and 'Patient ID'."
    ),
    modality=Modality.IMAGE,
    notes=(
        "One Case per image: case_id from 'Image Index', patient_id from 'Patient ID', "
        "image_ref=png path, label from 'Finding Labels' (pick a policy for the multi-label case)."
    ),
)


def _resolve_csv(raw_root) -> Path:
    """Return the path to ``Data_Entry_2017.csv`` given a csv path or its parent dir."""
    path = Path(raw_root)
    if path.is_dir():
        path = path / CSV_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{SPEC.name}: expected the ChestX-ray14 metadata csv at {path!r} "
            f"(point raw_root at {CSV_NAME} or a directory containing it)."
        )
    return path


def _row_to_case(row: dict) -> Case:
    """Turn one ``Data_Entry_2017.csv`` row into a schema.Case."""
    image_index = (row.get("Image Index") or "").strip()
    raw_labels = (row.get("Finding Labels") or "").strip()
    findings = [part.strip() for part in raw_labels.split("|") if part.strip()]
    return Case(
        case_id=image_index,
        patient_id=str((row.get("Patient ID") or "").strip()),
        modality=Modality.IMAGE,
        label=raw_labels.lower() or None,
        image_ref=image_index,
        report=None,
        meta={"findings": findings, "view": (row.get("View Position") or "").strip()},
    )


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw ChestX-ray14 release.

    ``raw_root`` is the path to ``Data_Entry_2017.csv`` or a directory containing it. One
    schema.Case is emitted per row (up to ``limit`` rows if given), then written to ``out``.
    """
    csv_path = _resolve_csv(raw_root)
    cases: list[Case] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if limit is not None and len(cases) >= limit:
                break
            cases.append(_row_to_case(row))
    return finalize(cases, out)
