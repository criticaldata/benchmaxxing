"""MIMIC-CXR text adapter (report text -> MCQ, Lane B).

Parses MIMIC-CXR-JPG v2.0.0 free-text reports plus the CheXpert label CSV into one
schema.Case per study, formatted as a 4-way MCQ over CheXpert findings. Companion to the
imaging adapter (``mimic_cxr.py``, Lane A) which emits one Case per image instead.

MCQ template (see issue #330): report = the study's free-text report (surfaced as "Clinical
context" by ``experiments/medqa/reproduce.py``'s prompt builder); question = the fixed stem
"What is the primary finding described in this report?"; options = the study's single
confirmed-positive finding plus 3 confirmed-negative findings (deterministic, in CheXpert column
order); answer_index = position of the positive finding. Keeping the report out of ``question``
matters: cue injection (``benchmaxxing/cues/text.py``) draws its perturbation tokens from
``question`` alone, so folding the report text into it would let a cue quote the report verbatim
into a distractor (see #336 review). Per-study filtering (also #330):

- Studies with more than one confirmed-positive finding are skipped: "the primary finding" is
  undefined when CheXpert's multi-label positives disagree, and a distractor could actually be
  present.
- Studies with zero confirmed-positive findings ("No Finding") are skipped: there is no positive
  to serve as the correct answer.
- Distractors are drawn only from confirmed-negative ("0.0") findings, never from uncertain
  ("-1.0") or unmentioned (blank) ones, so a distractor is never secretly true.

Reports and CheXpert labels are separate credentialed PhysioNet downloads that do not share a
root directory in practice, so ``build_manifest`` takes both roots explicitly rather than the
single ``raw_root`` most adapters use.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="mimic_cxr_text",
    raw_hint=(
        "PhysioNet MIMIC-CXR v2.0.0 free-text reports ('files/pXX/pYYYY/sZZZZ.txt' or an "
        "equivalent flat report tree) plus 'mimic-cxr-2.0.0-chexpert.csv[.gz]' (per-study "
        "CheXpert labels). Requires credentialed PhysioNet access."
    ),
    modality=Modality.TEXT,
    notes=(
        "One Case per study (not per image): report=study free-text, question=a fixed prompt "
        "stem, options=1 confirmed-positive finding + 3 confirmed-negative findings, "
        "answer_index=position of the positive. Studies with zero or multiple "
        "confirmed-positive findings are skipped (see module docstring / issue #330)."
    ),
)

_CHEXPERT_CSV_STEM = "mimic-cxr-2.0.0-chexpert.csv"

# CheXpert observation columns in release order, excluding "No Finding" (handled separately: a
# positive "No Finding" excludes the study rather than acting as an answerable finding).
FINDING_COLUMNS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
)

_NO_FINDING_COL = "No Finding"
_POSITIVE = "1.0"
_NEGATIVE = "0.0"
_N_DISTRACTORS = 3

_QUESTION_PROMPT = "What is the primary finding described in this report?"


def _resolve_labels_csv(labels_root) -> Path:
    """Return the CheXpert CSV path, accepting a directory or the (optionally gzipped) file."""
    root = Path(labels_root)
    if root.is_dir():
        for name in (_CHEXPERT_CSV_STEM, f"{_CHEXPERT_CSV_STEM}.gz"):
            candidate = root / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No {_CHEXPERT_CSV_STEM}[.gz] found under {root}")
    if not root.exists():
        raise FileNotFoundError(f"MIMIC-CXR CheXpert CSV not found: {root}")
    return root


def _read_csv_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(row: dict, key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _report_path(reports_root: Path, subject_id: str, study_id: str) -> Path:
    return reports_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"


def _pick_finding_case(row: dict, subject_id: str, study_id: str) -> tuple[str, tuple[str, ...]] | None:
    """Return (positive_finding, distractor_findings) or None if the study should be skipped."""
    if _cell(row, _NO_FINDING_COL) == _POSITIVE:
        return None
    positives = [col for col in FINDING_COLUMNS if _cell(row, col) == _POSITIVE]
    if len(positives) != 1:
        return None
    positive = positives[0]
    negatives = [col for col in FINDING_COLUMNS if col != positive and _cell(row, col) == _NEGATIVE]
    if len(negatives) < _N_DISTRACTORS:
        return None
    return positive, tuple(negatives[:_N_DISTRACTORS])


def build_manifest(reports_root, labels_root, out, limit=None):
    """Build a per-study MCQ manifest from MIMIC-CXR report text and CheXpert labels.

    ``reports_root`` is the free-text report tree (``pXX/pYYYY/sZZZZ.txt``); ``labels_root`` is
    the CheXpert CSV or a directory containing it (see module docstring for why these are
    separate). One schema.Case is emitted per study that has exactly one confirmed-positive
    CheXpert finding and at least 3 confirmed-negative findings to draw distractors from, and
    whose report file exists. Writes the manifest to ``out`` via ``finalize`` and returns the
    list of Cases (at most ``limit``).
    """
    reports_root = Path(reports_root)
    rows = _read_csv_rows(_resolve_labels_csv(labels_root))
    cases: list[Case] = []
    for row in rows:
        if limit is not None and len(cases) >= limit:
            break
        subject_id = _cell(row, "subject_id")
        study_id = _cell(row, "study_id")
        report_path = _report_path(reports_root, subject_id, study_id)
        if not report_path.is_file():
            continue
        picked = _pick_finding_case(row, subject_id, study_id)
        if picked is None:
            continue
        positive, distractors = picked
        chosen = sorted((positive, *distractors), key=FINDING_COLUMNS.index)
        cases.append(
            Case(
                case_id=f"mimic-cxr-text-{subject_id}-{study_id}",
                patient_id=subject_id,
                modality=Modality.TEXT,
                label=positive.lower(),
                report=report_path.read_text(encoding="utf-8"),
                question=_QUESTION_PROMPT,
                options=tuple(chosen),
                answer_index=chosen.index(positive),
                meta={"study_id": study_id, "distractors": list(distractors)},
            )
        )
    finalize(cases, out)
    return cases
