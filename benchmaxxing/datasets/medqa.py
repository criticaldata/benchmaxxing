"""MedQA adapter (text / MCQ, Lane B).

Owners: implement ``build_manifest`` to turn the raw release into a shared manifest via
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="medqa",
    raw_hint=(
        "MedQA-USMLE (Jin et al. 2020): JSONL splits (train/dev/test) where each line has "
        "'question', an 'options' map (A..E) and 'answer_idx'/'answer'. The English 'US' subset "
        "is 4-5 way multiple choice."
    ),
    modality=Modality.TEXT,
    notes=(
        "One Case per question: case_id=stable question id, question=stem, options=ordered A..E "
        "values, answer_index=position of answer_idx in that order."
    ),
)


def _resolve_jsonl(raw_root) -> Path:
    """Return the JSONL file to parse: ``raw_root`` itself, or ``test.jsonl`` inside it."""
    root = Path(raw_root)
    if root.is_dir():
        candidate = root / "test.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"No test.jsonl found in MedQA directory: {root}")
        return candidate
    if not root.exists():
        raise FileNotFoundError(f"MedQA JSONL not found: {root}")
    return root


def _read_objects(path: Path):
    """Yield one parsed JSON object per non-empty line of ``path``."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            yield json.loads(stripped)


def _case_from_obj(obj: dict, index: int) -> Case:
    """Turn one MedQA-USMLE record into a schema.Case (text lane)."""
    options_map = obj["options"]
    keys = sorted(options_map)
    options = tuple(str(options_map[key]) for key in keys)
    answer_letter = obj["answer_idx"]
    if answer_letter not in keys:
        raise ValueError(
            f"Row {index}: answer_idx {answer_letter!r} is not among option keys {keys}."
        )
    answer_index = keys.index(answer_letter)
    case_id = str(obj.get("id") or obj.get("case_id") or f"medqa-{index}")
    label = obj.get("meta_info")
    return Case(
        case_id=case_id,
        patient_id="",
        modality=Modality.TEXT,
        label=None if label is None else str(label),
        question=str(obj["question"]),
        options=options,
        answer_index=answer_index,
    )


def build_manifest(raw_root, out, limit=None):
    """Parse the MedQA-USMLE JSONL at ``raw_root`` into a manifest at ``out``.

    ``raw_root`` is the ``.jsonl`` file itself or a directory holding ``test.jsonl``. Each line is
    a record with a ``question`` stem, an ``options`` map (A..E), and an ``answer_idx`` letter.
    Options are ordered by sorted key (A, B, C, D[, E]) and ``answer_index`` is the position of
    ``answer_idx`` in that order. ``limit`` keeps only the first N records.
    """
    path = _resolve_jsonl(raw_root)
    cases: list[Case] = []
    for index, obj in enumerate(_read_objects(path)):
        if limit is not None and index >= limit:
            break
        cases.append(_case_from_obj(obj, index))
    return finalize(cases, out)
