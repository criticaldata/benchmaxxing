"""ProbMed adapter (medical VQA / probing, represented as a multimodal MCQ).

ProbMed (Yan et al., "Worse than Random?", ACL 2025 Findings; HF ``rippleripple/ProbMed``) is a
medical VQA probe set: for each image it asks a set of binary yes/no questions across five
families (modality, body_part, abnormality, entity, grounding). The signal is *adversarial*: a
ground-truth question about a real attribute (answer "yes") is paired with a question about a
hallucinated attribute (answer "no"). A model that hallucinates answers "yes" to both, so the
gt/adversarial pair is the benchmark-gaming probe.

Schema note: ProbMed is image *and* yes/no question.  It uses the text/MCQ modality so its
question/options/answer index retain the mature scoring semantics, while ``image_ref`` remains a
first-class field.  With ``cue_set="image-v1"`` the runner sends each original MCQ together with
the clean or cued image to a vision-capable backend.

Raw layout: the ProbMed dataset repo ships ``test.json`` (a flat JSON list of QA records) and a
``probmed/`` image folder. Each record has ``id`` (image id; records for one image are
consecutive), ``gpt_idx`` (global index), ``image`` (path), ``image_type`` (modality-organ, e.g.
"X-ray - Chest"), ``qa_type``, ``question``, and ``answer`` (yes/no). ``qa_type`` encodes the
family and polarity, mirroring the official ``eval/calculate_score.py``:
    modality / body_part -> two yes/no questions per image, positional (first=gt, second=adversarial)
    abnormality          -> a single ungated yes/no
    entity_gt_<ID> / entity_<ID>, grounding_gt_<ID> / grounding_<ID>
                         -> gt/adversarial pairs keyed by the trailing <ID>
    entity_hallu          -> a single negative entity question with no pair
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="probmed",
    raw_hint=(
        "ProbMed (Yan et al., ACL 2025 Findings; HF 'rippleripple/ProbMed', gated). Download the "
        "dataset repo, which ships 'test.json' (a flat JSON list of QA records) and a 'probmed/' "
        "image folder. Point raw_root at that directory or at test.json directly. Each record: "
        "{id, gpt_idx, image, image_type, qa_type, question, answer}."
    ),
    modality=Modality.TEXT,
    notes=(
        "Imaging-grounded VQA represented as text/MCQ binary yes/no probes with a first-class "
        "image_ref; use cue_set='image-v1' and --image-root to run a vision backend. label=family "
        "(modality|body_part|abnormality|entity|grounding); ground-truth questions answer 'yes', "
        "adversarial hallucinated-attribute questions answer 'no'. gt/adversarial partners share "
        "meta['group_key'] (entity/grounding by trailing id; modality/body_part positionally with "
        "first=gt; abnormality and entity_hallu are single ungated yes/no questions)."
    ),
)

_OPTIONS = ("yes", "no")
_REQUIRED_KEYS = ("id", "qa_type", "question", "answer", "image")
_PAIRED_QA_TYPE = re.compile(r"^(?P<family>entity|grounding)_(?:(?P<gt>gt)_)?(?P<id>[^_]+)$")


def _resolve_json(raw_root) -> Path:
    """Return the test.json to parse: ``raw_root`` itself, or ``test.json`` inside it."""
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
    """Load test.json as a list of records, raising if the top-level shape is wrong."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(  # noqa: TRY004 - a wrong top-level shape is a data error, not a type bug
            f"ProbMed test.json must be a JSON list of records, got {type(data).__name__}."
        )
    return data


def _family_of(qa_type: str) -> str:
    """Map a raw qa_type to one of the five ProbMed families (mirrors calculate_score.py)."""
    if qa_type == "modality":
        return "modality"
    if qa_type == "body_part":
        return "body_part"
    if qa_type == "abnormality":
        return "abnormality"
    if qa_type == "entity_hallu":
        return "entity"
    match = _PAIRED_QA_TYPE.fullmatch(qa_type)
    if match:
        return match["family"]
    raise ValueError(
        f"Unrecognized ProbMed qa_type {qa_type!r}. Expected modality, body_part, abnormality, "
        "entity_hallu, entity_gt_<id>, entity_<id>, grounding_gt_<id>, or grounding_<id>."
    )


def _polarity_and_group(
    qa_type: str, family: str, image_id: str, order: dict[tuple[str, str], int]
) -> tuple[str, str]:
    """Derive (polarity, group_key) for a record, mirroring calculate_score.py's qa_type logic.

    ``order`` tracks how many questions of a given (image_id, family) have been seen, so the
    positional gt/adversarial split for modality/body_part matches the scoring script's
    "position 0 is ground truth" convention. It is mutated in place.
    """
    if family in ("modality", "body_part"):
        # ProbMed asks two yes/no questions per image for these families and does not tag which
        # is which; the scoring treats the first as ground truth (real attribute, answer "yes")
        # and the second as the hallucination (false attribute, answer "no").
        seen = order.get((image_id, family), 0)
        order[(image_id, family)] = seen + 1
        polarity = "gt" if seen == 0 else "adversarial"
        return polarity, f"{image_id}:{family}"
    if family == "abnormality":
        return "single", f"{image_id}:abnormality"
    if qa_type == "entity_hallu":
        # The official scorer treats this as a singleton, not the adversarial half of a pair.
        return "single", f"{image_id}:entity:hallu"
    # entity / grounding: gt/adversarial pairs keyed by the trailing id token.
    match = _PAIRED_QA_TYPE.fullmatch(qa_type)
    if not match or match["family"] != family:  # defensive: _family_of has already validated it
        raise ValueError(f"Invalid paired ProbMed qa_type {qa_type!r} for family {family!r}.")
    polarity = "gt" if match["gt"] else "adversarial"
    return polarity, f"{image_id}:{family}:{match['id']}"


def _answer_index(answer, index: int, case_id: str) -> int:
    """Map a ProbMed yes/no answer to an index into ``_OPTIONS``; raise on anything else."""
    norm = str(answer).strip().lower().rstrip(".")
    if norm == "yes":
        return 0
    if norm == "no":
        return 1
    raise ValueError(
        f"Row {index} (case_id={case_id!r}): ProbMed answer must be 'yes' or 'no', got {answer!r}."
    )


def _case_id(record: dict, index: int) -> str:
    """Stable, unique case id. gpt_idx is a global index; fall back to the row index."""
    gpt_idx = record.get("gpt_idx")
    suffix = str(gpt_idx) if gpt_idx is not None else str(index)
    return f"probmed-{record['id']}-{record['qa_type']}-{suffix}"


def _case_from_record(record: dict, index: int, order: dict[tuple[str, str], int]) -> Case:
    """Turn one ProbMed test.json record into a schema.Case (text lane, yes/no MCQ)."""
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(
            f"Row {index}: ProbMed record missing required key(s) {missing}. "
            f"Present keys: {sorted(record)}."
        )
    image_id = str(record["id"])
    qa_type = str(record["qa_type"])
    family = _family_of(qa_type)
    polarity, group_key = _polarity_and_group(qa_type, family, image_id, order)
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
    image_type = record.get("image_type")
    if image_type is not None:
        meta["image_type"] = str(image_type)
    return Case(
        case_id=case_id,
        patient_id=image_id,
        modality=Modality.TEXT,
        label=family,
        image_ref=str(record["image"]),
        question=str(record["question"]),
        options=_OPTIONS,
        answer_index=answer_index,
        meta=meta,
    )


def build_manifest(raw_root, out, limit=None):
    """Parse the ProbMed test.json at ``raw_root`` into a manifest at ``out``.

    ``raw_root`` is the ``test.json`` file itself or the ProbMed directory that holds it. Each
    record becomes one yes/no Case: family goes to ``label``, image path to ``image_ref``, and
    parsed family/polarity/group_key to ``meta``. ``limit`` keeps only the first N records.
    """
    path = _resolve_json(raw_root)
    records = _load_records(path)
    order: dict[tuple[str, str], int] = {}
    cases: list[Case] = []
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        cases.append(_case_from_record(record, index, order))
    return finalize(cases, out)
