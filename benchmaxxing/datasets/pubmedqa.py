from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="pubmedqa",
    raw_hint=(
        "PubMedQA pqa_labeled (Jin et al. 2019): the official release JSON "
        "(ori_pqal.json), a single JSON object keyed by PMID. Each value has "
        "'QUESTION', 'CONTEXTS' (list), 'MESHES', 'YEAR', 'final_decision' "
        "(yes/no/maybe), and 'LONG_ANSWER'."
    ),
    modality=Modality.TEXT,
    notes=(
        "One Case per PMID: case_id=PMID, question=QUESTION, options=('yes','no','maybe') "
        "fixed, answer_index=position of final_decision in that order, report=joined "
        "CONTEXTS, meta carries long_answer/meshes/year."
    ),
)

_OPTIONS = ("yes", "no", "maybe")


def _resolve_json(raw_root) -> Path:
    """Return the JSON file to parse: ``raw_root`` itself, or ``ori_pqal.json`` inside it."""
    root = Path(raw_root)
    if root.is_dir():
        candidate = root / "ori_pqal.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No ori_pqal.json found in PubMedQA directory: {root}")
        return candidate
    if not root.exists():
        raise FileNotFoundError(f"PubMedQA JSON not found: {root}")
    return root


def _read_records(path: Path):
    """Yield (pmid, record) pairs from the PMID-keyed JSON object at ``path``, in file order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    yield from data.items()


def _case_from_obj(pmid: str, obj: dict, index: int) -> Case:
    """Turn one PubMedQA record into a schema.Case (text lane, fixed yes/no/maybe options)."""
    decision = str(obj["final_decision"]).strip().lower()
    if decision not in _OPTIONS:
        raise ValueError(
            f"Row {index} (pmid={pmid}): final_decision {decision!r} is not one of {_OPTIONS}."
        )
    answer_index = _OPTIONS.index(decision)

    contexts = obj.get("CONTEXTS") or obj.get("contexts") or []
    report = " ".join(str(c) for c in contexts) if contexts else None

    meta: dict = {}
    if obj.get("LONG_ANSWER") is not None:
        meta["long_answer"] = obj["LONG_ANSWER"]
    if obj.get("MESHES") is not None:
        meta["meshes"] = obj["MESHES"]
    if obj.get("YEAR") is not None:
        meta["year"] = obj["YEAR"]

    return Case(
        case_id=str(pmid),
        patient_id="",
        modality=Modality.TEXT,
        label=None,
        report=report,
        question=str(obj["QUESTION"]),
        options=_OPTIONS,
        answer_index=answer_index,
        meta=meta,
    )


def build_manifest(raw_root, out, limit=None):
    """Parse the PubMedQA ori_pqal.json at ``raw_root`` into a manifest at ``out``.

    ``raw_root`` is the JSON file itself or a directory holding ``ori_pqal.json``: a single
    JSON object keyed by PMID. Options are the fixed triple ("yes", "no", "maybe") and
    ``answer_index`` is the position of each record's ``final_decision`` in that order.
    ``limit`` keeps only the first N records (in file/insertion order).
    """
    path = _resolve_json(raw_root)
    cases: list[Case] = []
    for index, (pmid, obj) in enumerate(_read_records(path)):
        if limit is not None and index >= limit:
            break
        cases.append(_case_from_obj(pmid, obj, index))
    return finalize(cases, out)