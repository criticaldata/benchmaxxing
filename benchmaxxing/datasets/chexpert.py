"""CheXpert adapter (imaging, Lane A).

Owners: implement ``build_manifest`` to turn the raw release into a shared manifest via
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality

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


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw CheXpert release. Not yet implemented."""
    raise NotImplementedError(
        f"{SPEC.name}.build_manifest is a stub for dataset owners to fill in. Point raw_root at: "
        f"{SPEC.raw_hint} Then construct schema.Case rows and emit them to {out!r} via "
        f"benchmaxxing.datasets.base.finalize (respecting limit={limit!r})."
    )
