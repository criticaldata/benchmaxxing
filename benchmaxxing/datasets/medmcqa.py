"""MedMCQA adapter (text / MCQ, Lane B).

Owners: implement ``build_manifest`` to turn the raw release into a shared manifest via
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="medmcqa",
    raw_hint=(
        "MedMCQA (Pal et al. 2022): JSON/JSONL splits (train/dev/test) where each line has "
        "'question', four options 'opa'..'opd', a correct-option pointer 'cop' (0-3), and "
        "'subject_name'/'topic_name'. ~194k 4-way medical-entrance MCQs."
    ),
    modality=Modality.TEXT,
    notes=(
        "One Case per question: case_id=stable record id (or a generated one), question=stem, "
        "options=(opa, opb, opc, opd) in that fixed order, answer_index=cop, subject_name and "
        "topic_name carried in meta."
    ),
)


def _resolve_jsonl(raw_root) -> Path:
    """Return the JSON/JSONL file to parse: ``raw_root`` itself, or ``dev.json`` inside it."""
    root = Path(raw_root)
    if root.is_dir():
        candidate = root / "dev.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No dev.json found in MedMCQA directory: {root}")
        return candidate
    if not root.exists():
        raise FileNotFoundError(f"MedMCQA JSON/JSONL not found: {root}")
    return root


def _read_objects(path: Path):
    """Yield one parsed JSON object per non-empty line of ``path``."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            yield json.loads(stripped)


def _case_from_obj(obj: dict, index: int) -> Case:
    """Turn one MedMCQA record into a schema.Case (text lane)."""
    options = tuple(str(obj[key]) for key in ("opa", "opb", "opc", "opd"))
    cop = obj["cop"]
    if not isinstance(cop, int) or not (0 <= cop < len(options)):
        raise ValueError(f"Row {index}: cop {cop!r} is not a valid index into {options}.")
    case_id = str(obj.get("id") or obj.get("question_id") or obj.get("case_id") or f"medmcqa-{index}")
    meta = {}
    if obj.get("subject_name") is not None:
        meta["subject_name"] = obj["subject_name"]
    if obj.get("topic_name") is not None:
        meta["topic_name"] = obj["topic_name"]
    return Case(
        case_id=case_id,
        patient_id="",
        modality=Modality.TEXT,
        question=str(obj["question"]),
        options=options,
        answer_index=cop,
        meta=meta,
    )


def build_manifest(raw_root, out, limit=None):
    """Parse the MedMCQA JSON/JSONL at ``raw_root`` into a manifest at ``out``.

    ``raw_root`` is the ``.json``/``.jsonl`` file itself or a directory holding ``dev.json``.
    Each line is a record with a ``question`` stem, four options ``opa``..``opd``, and a
    correct-option pointer ``cop`` (0-3). ``subject_name``/``topic_name`` flow into
    ``Case.meta``. ``limit`` keeps only the first N records.
    """
    path = _resolve_jsonl(raw_root)
    cases: list[Case] = []
    for index, obj in enumerate(_read_objects(path)):
        if limit is not None and index >= limit:
            break
        cases.append(_case_from_obj(obj, index))
    return finalize(cases, out)
