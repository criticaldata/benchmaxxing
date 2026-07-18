"""MedQA adapter (text / MCQ, Lane B).

Owners: implement ``build_manifest`` to turn the raw release into a shared manifest via
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality

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


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw MedQA release. Not yet implemented."""
    raise NotImplementedError(
        f"{SPEC.name}.build_manifest is a stub for dataset owners to fill in. Point raw_root at: "
        f"{SPEC.raw_hint} Then construct schema.Case rows and emit them to {out!r} via "
        f"benchmaxxing.datasets.base.finalize (respecting limit={limit!r})."
    )
