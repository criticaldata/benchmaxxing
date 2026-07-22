"""Manifest I/O: read a CSV or JSONL manifest into schema.Case objects and write it back.

A manifest is a flat table. Two shapes are supported and auto-detected per row:

Imaging (Lane A): case_id, patient_id, image_ref, report[, label]
Text / MCQ (Lane B): case_id, question, options (pipe-separated), answer_index[, patient_id, label]

Detection is by column content: a non-empty ``image_ref`` marks an imaging case, a non-empty
``question`` marks a text case. An explicit ``modality`` column, when present, wins over inference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmaxxing.schema import Case, Modality

# Fixed column order used when writing a manifest to CSV. Irrelevant fields are left blank so a
# mixed imaging/text manifest round-trips through a single header.
MANIFEST_COLUMNS = [
    "case_id",
    "patient_id",
    "modality",
    "label",
    "image_ref",
    "report",
    "question",
    "options",
    "answer_index",
    "meta",
]

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_IMAGE_ALIASES = {"image", "images", "img", "cxr", "imaging"}
_TEXT_ALIASES = {"text", "mcq", "txt", "question"}


def load_cases(manifest_path) -> list[Case]:
    """Read a CSV or JSONL manifest at ``manifest_path`` into a list of schema.Case.

    Raises FileNotFoundError if the path is missing and ValueError on any malformed or
    incomplete row (with the row index and case_id in the message).
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = _read_rows(path)
    cases = [_row_to_case(row, i) for i, row in enumerate(rows)]
    if not cases:
        raise ValueError(f"Manifest {path} contains no rows.")
    return cases


def write_manifest(cases, path) -> Path:
    """Write ``cases`` to a CSV manifest at ``path`` using the fixed MANIFEST_COLUMNS header."""
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for case in cases:
            writer.writerow(_case_to_row(case))
    return out


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("cases", [])
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest {path} must be a list of rows or have a 'cases' list.")
        return data
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _get(row: dict, key: str):
    """Return a cleaned value from a row, mapping empty strings to None."""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _detect_modality(row: dict) -> Modality | None:
    explicit = _get(row, "modality")
    if explicit is not None:
        token = str(explicit).lower()
        if token in _IMAGE_ALIASES:
            return Modality.IMAGE
        if token in _TEXT_ALIASES:
            return Modality.TEXT
    if _get(row, "image_ref") is not None:
        return Modality.IMAGE
    if _get(row, "question") is not None:
        return Modality.TEXT
    return None


def _row_to_case(row: dict, index: int) -> Case:
    case_id = _get(row, "case_id")
    if case_id is None:
        raise ValueError(f"Row {index}: missing required field 'case_id'.")
    case_id = str(case_id)
    modality = _detect_modality(row)
    if modality is None:
        raise ValueError(
            f"Row {index} (case_id={case_id!r}): cannot detect modality. Provide 'image_ref' "
            f"for an imaging case or 'question' plus 'options' for a text/MCQ case."
        )
    if modality is Modality.IMAGE:
        return _image_case(row, index, case_id)
    return _text_case(row, index, case_id)


def _image_case(row: dict, index: int, case_id: str) -> Case:
    patient_id = _get(row, "patient_id")
    if patient_id is None:
        raise ValueError(f"Row {index} (case_id={case_id!r}): imaging case missing 'patient_id'.")
    image_ref = _get(row, "image_ref")
    if image_ref is None:
        raise ValueError(f"Row {index} (case_id={case_id!r}): imaging case missing 'image_ref'.")
    label = _get(row, "label")
    return Case(
        case_id=case_id,
        patient_id=str(patient_id),
        modality=Modality.IMAGE,
        label=None if label is None else str(label),
        image_ref=str(image_ref),
        report=_none_or_str(_get(row, "report")),
        meta=_parse_meta(row.get("meta")),
    )


def _text_case(row: dict, index: int, case_id: str) -> Case:
    question = _get(row, "question")
    if question is None:
        raise ValueError(f"Row {index} (case_id={case_id!r}): text/MCQ case missing 'question'.")
    options = _parse_options(row.get("options"))
    if not options:
        raise ValueError(
            f"Row {index} (case_id={case_id!r}): text/MCQ case missing 'options' "
            f"(expected a pipe-separated list or a JSON array)."
        )
    answer_index = _parse_answer_index(row.get("answer_index"), index, case_id, len(options))
    patient_id = _get(row, "patient_id")
    label = _get(row, "label")
    return Case(
        case_id=case_id,
        patient_id=str(patient_id) if patient_id is not None else case_id,
        modality=Modality.TEXT,
        label=None if label is None else str(label),
        question=str(question),
        options=options,
        answer_index=answer_index,
        meta=_parse_meta(row.get("meta")),
    )


def _parse_options(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        # JSON array first (the lossless format _case_to_row writes); an option containing a
        # literal '|' would be silently split by the legacy pipe format, corrupting
        # answer_index. Fall back to pipe-split only for legacy hand-written manifests.
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, list):
                parts = [str(item).strip() for item in loaded]
                parts = [part for part in parts if part]
                return tuple(parts) if parts else None
        parts = [part.strip() for part in text.split("|")]
    parts = [part for part in parts if part]
    return tuple(parts) if parts else None


def _parse_answer_index(value, index: int, case_id: str, n_options: int) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Row {index} (case_id={case_id!r}): text/MCQ case missing 'answer_index'.")
    try:
        answer_index = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Row {index} (case_id={case_id!r}): 'answer_index' must be an integer, got {value!r}."
        ) from None
    if answer_index < 0 or answer_index >= n_options:
        raise ValueError(
            f"Row {index} (case_id={case_id!r}): 'answer_index' {answer_index} is out of range for "
            f"{n_options} option(s)."
        )
    return answer_index


def _none_or_str(value):
    return None if value is None else str(value)


def _parse_meta(value) -> dict:
    """Restore the per-case meta dict (adapters store support-device flags, views, label maps
    here; dropping it on the round trip would lose the natural-cue signal)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _case_to_row(case: Case) -> dict:
    # JSON array, not a pipe join: an option containing '|' must round-trip intact, or
    # answer_index would silently point at the wrong ground truth after a reload.
    options = json.dumps(list(case.options)) if case.options else ""
    answer_index = "" if case.answer_index is None else case.answer_index
    return {
        "case_id": case.case_id,
        "patient_id": case.patient_id,
        "modality": case.modality.value,
        "label": case.label or "",
        "image_ref": case.image_ref or "",
        "report": case.report or "",
        "question": case.question or "",
        "options": options,
        "answer_index": answer_index,
        "meta": json.dumps(case.meta) if case.meta else "",
    }
