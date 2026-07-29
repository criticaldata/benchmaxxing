"""CheXpert adapter (imaging, Lane A).

Turns the raw Stanford CheXpert release (``train.csv`` / ``valid.csv``) into a shared manifest
of schema.Case rows via ``benchmaxxing.datasets.base.finalize``.

Uncertainty policy (explicit): CheXpert encodes uncertain findings as ``-1.0``. This adapter
treats ``-1.0`` as *negative* (only ``1.0`` counts as a confirmed positive finding). An uncertain
cell therefore cannot anchor a definitely-false plant in the cascade experiments. This is the
conservative policy recommended by Agastya191 in Issue #331.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality
from benchmaxxing.utils.clinical_labels import FINDING_COLUMNS, CLINICAL_HIERARCHY, _POSITIVE

SPEC = DatasetSpec(
    name="chexpert",
    raw_hint=(
        "Stanford CheXpert v1.0 (or CheXpert-small): 'train/' and 'valid/' folders of "
        "frontal/lateral JPGs plus 'train.csv' and 'valid.csv' whose 14 observation columns use "
        "1.0/0.0/-1.0/blank (positive/negative/uncertain/unmentioned). Requires a signed licence."
    ),
    modality=Modality.IMAGE,
    notes=(
        "One Case per image: case_id from the CSV 'Path', patient_id from the patient folder, "
        "image_ref=Path, label from the chosen observation column and uncertainty policy."
    ),
)


_PATIENT_RE = re.compile(r"patient\d+")


def _resolve_csv(raw_root) -> Path:
    """Return the CSV path, accepting either the file itself or a dir holding train/valid.csv."""
    root = Path(raw_root)
    if root.is_dir():
        for name in ("train.csv", "valid.csv"):
            candidate = root / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No train.csv or valid.csv found under {root}")
    if not root.exists():
        raise FileNotFoundError(f"CheXpert CSV not found: {root}")
    return root


def _patient_id(path: str) -> str:
    """Extract the ``patientNNNNN`` segment from a CheXpert image Path."""
    match = _PATIENT_RE.search(path or "")
    if match is None:
        raise ValueError(f"Could not parse a patient id from Path {path!r}")
    return match.group(0)


def read_cases(raw_root, limit=None) -> list[Case]:
    """Parse the CheXpert CSV at ``raw_root`` into schema.Case rows.

    ``raw_root`` is the ``train.csv``/``valid.csv`` file or a directory containing one. Each CSV
    row becomes one imaging Case. When ``limit`` is given, a deterministic subsample of that size
    is returned from the *full* parsed pool — NOT the first ``limit`` rows of the CSV, which would
    produce a contiguous block biased toward early patients.

    Multi-label: findings are ordered by clinical acuity (``CLINICAL_HIERARCHY``), pipe-separated.
    Only ``1.0`` is treated as a confirmed positive; ``-1.0`` (uncertain) and ``0.0``/blank
    (negative/unmentioned) are treated as absent.
    """
    csv_path = _resolve_csv(raw_root)
    cases: list[Case] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = (row.get("Path") or "").strip()
            labels = {col: (row.get(col) or "").strip() for col in FINDING_COLUMNS}
            positives = [
                col
                for col in CLINICAL_HIERARCHY
                if labels.get(col) == _POSITIVE
            ]
            label_str = "|".join(positives).lower() if positives else "no finding"
            cases.append(
                Case(
                    case_id=path,
                    patient_id=_patient_id(path),
                    modality=Modality.IMAGE,
                    label=label_str,
                    image_ref=path,
                    report=None,
                    meta={
                        "support_devices": labels["Support Devices"] == _POSITIVE,
                        "labels": labels,
                    },
                )
            )

    # Apply limit by deterministic subsampling, not by truncating the CSV.
    if limit is not None and limit < len(cases):
        from benchmaxxing.budget import RunBudget, subsample_cases
        cases = subsample_cases(cases, RunBudget(max_cases=limit, seed=42))

    return cases


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw CheXpert release and write it to ``out``."""
    return finalize(read_cases(raw_root, limit), out)
