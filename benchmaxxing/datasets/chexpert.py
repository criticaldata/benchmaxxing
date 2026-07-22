"""CheXpert adapter (imaging, Lane A).

Turns the raw Stanford CheXpert release (``train.csv`` / ``valid.csv``) into a shared manifest
of schema.Case rows via ``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

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

# The 14 CheXpert observation columns, in release order.
FINDING_COLUMNS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

_POSITIVE = "1.0"
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
    """Parse the CheXpert CSV at ``raw_root`` into schema.Case rows (respecting ``limit``).

    ``raw_root`` is the ``train.csv``/``valid.csv`` file or a directory containing one. Each CSV row
    becomes one imaging Case, keeping the full 14-observation label map and the support-devices
    flag in ``meta``.
    """
    csv_path = _resolve_csv(raw_root)
    cases: list[Case] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if limit is not None and len(cases) >= limit:
                break
            path = (row.get("Path") or "").strip()
            labels = {col: (row.get(col) or "").strip() for col in FINDING_COLUMNS}
            is_pneumothorax = labels["Pneumothorax"] == _POSITIVE
            cases.append(
                Case(
                    case_id=path,
                    patient_id=_patient_id(path),
                    modality=Modality.IMAGE,
                    label="pneumothorax" if is_pneumothorax else "no finding",
                    image_ref=path,
                    report=None,
                    meta={
                        "support_devices": labels["Support Devices"] == _POSITIVE,
                        "labels": labels,
                    },
                )
            )
    return cases


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw CheXpert release and write it to ``out``."""
    return finalize(read_cases(raw_root, limit), out)
