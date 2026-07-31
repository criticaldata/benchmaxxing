"""ProbMed adapter (medical VQA / probing, rendered into the text lane).

ProbMed (Yan et al., "Worse than Random?", ACL 2025 Findings; HF ``rippleripple/ProbMed``).
Real test.json record keys: id, i, image, image_type, qa_type, question, answer, src_dataset.
qa_type vocabulary (verified on the real gated data):
    modality_gt / modality_hallu           (one gt + one hallucination per image)
    body_part_gt / body_part_hallu
    abnormality                            (single ungated yes/no)
    entity_gt_<id> / entity_hallu_<id>     (gt/adversarial pair keyed by <id>)
    grounding_gt_<id> / grounding_hallu_<id>
gt questions answer "yes"; hallu (adversarial, false attribute) answer "no".

Rendered into the text/MCQ lane (options=("yes","no")) so existing flip-rate scoring runs; the
grounding image path is a first-class Case.image_ref (also mirrored in meta); an
image-v1 run sends the pixels. Text/MCQ scoring runs today via options=("yes","no").
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="probmed",
    raw_hint=(
        "ProbMed (HF 'rippleripple/ProbMed', gated). Ships test.json (flat JSON list) + images. "
        "Point raw_root at test.json or its directory. Record: "
        "{id, i, image, image_type, qa_type, question, answer, src_dataset}."
    ),
    modality=Modality.TEXT,
    notes=(
        "Imaging-grounded VQA rendered into the text/MCQ lane as yes/no probes; image path in "
        "meta['image_ref']. qa_type -> family (modality|body_part|abnormality|entity|grounding) "
        "and polarity (_gt=gt/'yes', _hallu=adversarial/'no'); gt/adversarial partners share "
        "meta['group_key']."
    ),
)

_OPTIONS = ("yes", "no")
_REQUIRED_KEYS = ("id", "qa_type", "question", "answer", "image")
_FAMILIES = ("modality", "body_part", "abnormality", "entity", "grounding")


def _resolve_json(raw_root) -> Path:
    root = Path(raw_root)
    if root.is_dir():
        candidate = root / "test.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No test.json found in ProbMed directory: {root}")
        return candidate
    if not root.exists():
        raise FileNotFoundError(f"ProbMed test.json not found: {root}")
    return root


def _load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(  # noqa: TRY004 - a wrong top-level shape is a data error, not a type bug
            f"ProbMed test.json must be a JSON list of records, got {type(data).__name__}."
        )
    return data


def _family_of(qa_type: str) -> str:
    """Map a qa_type to one of the five families; raise on anything unrecognized."""
    for fam in _FAMILIES:
        if qa_type == fam or qa_type.startswith(fam + "_"):
            return fam
    raise ValueError(
        f"Unrecognized ProbMed qa_type {qa_type!r}. Expected one of {_FAMILIES} "
        f"optionally suffixed _gt/_hallu(/_<id>)."
    )


def _polarity_and_group(qa_type: str, family: str, image_id: str) -> tuple[str, str]:
    """Derive (polarity, group_key) from the explicit _gt/_hallu tag in the real data."""
    if family == "abnormality":
        return "single", f"{image_id}:abnormality"
    rest = qa_type[len(family):].lstrip("_")  # "gt", "hallu", "gt_0", "hallu_3"
    if rest.startswith("gt"):
        polarity = "gt"
    elif rest.startswith("hallu"):
        polarity = "adversarial"
    else:
        raise ValueError(f"ProbMed qa_type {qa_type!r} has no _gt/_hallu polarity tag.")
    if family in ("entity", "grounding"):
        group_id = rest.split("_")[-1]  # the trailing <id>
        return polarity, f"{image_id}:{family}:{group_id}"
    return polarity, f"{image_id}:{family}"  # modality / body_part: gt+hallu pair per image


def _answer_index(answer, index: int, case_id: str) -> int:
    norm = str(answer).strip().lower().rstrip(".")
    if norm == "yes":
        return 0
    if norm == "no":
        return 1
    raise ValueError(
        f"Row {index} (case_id={case_id!r}): ProbMed answer must be 'yes' or 'no', got {answer!r}."
    )


def _case_id(record: dict, index: int) -> str:
    suffix = record.get("gpt_idx", record.get("i", index))
    return f"probmed-{record['id']}-{record['qa_type']}-{suffix}"


def _case_from_record(record: dict, index: int) -> Case:
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(
            f"Row {index}: ProbMed record missing required key(s) {missing}. "
            f"Present keys: {sorted(record)}."
        )
    image_id = str(record["id"])
    qa_type = str(record["qa_type"])
    family = _family_of(qa_type)
    polarity, group_key = _polarity_and_group(qa_type, family, image_id)
    case_id = _case_id(record, index)
    answer_index = _answer_index(record["answer"], index, case_id)
    meta = {
        "qa_type": qa_type,
        "family": family,
        "polarity": polarity,
        "group_key": group_key,
        "image_ref": str(record["image"]),
        "gt_answer": _OPTIONS[answer_index],
    }
    for extra in ("image_type", "src_dataset"):
        if record.get(extra) is not None:
            meta[extra] = str(record[extra])
    return Case(
        case_id=case_id,
        patient_id=image_id,
        modality=Modality.TEXT,
        image_ref=str(record["image"]),
        label=family,
        question=str(record["question"]),
        options=_OPTIONS,
        answer_index=answer_index,
        meta=meta,
    )


def build_manifest(raw_root, out, limit=None):
    """Parse the ProbMed test.json at ``raw_root`` into a manifest at ``out``."""
    path = _resolve_json(raw_root)
    records = _load_records(path)
    cases: list[Case] = []
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        cases.append(_case_from_record(record, index))
    return finalize(cases, out)
