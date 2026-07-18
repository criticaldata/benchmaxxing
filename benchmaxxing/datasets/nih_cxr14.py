"""NIH ChestX-ray14 adapter (imaging, Lane A).

Owners: implement ``build_manifest`` to turn the raw release into a shared manifest via
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality

SPEC = DatasetSpec(
    name="nih_cxr14",
    raw_hint=(
        "NIH ChestX-ray14 (NIH Clinical Center, also on Kaggle): 'images_XXX/images/*.png' folders "
        "plus 'Data_Entry_2017.csv' whose 'Finding Labels' column is a pipe-separated list over 14 "
        "pathologies ('No Finding' otherwise), keyed by 'Image Index' and 'Patient ID'."
    ),
    modality=Modality.IMAGE,
    notes=(
        "One Case per image: case_id from 'Image Index', patient_id from 'Patient ID', "
        "image_ref=png path, label from 'Finding Labels' (pick a policy for the multi-label case)."
    ),
)


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw ChestX-ray14 release. Not yet implemented."""
    raise NotImplementedError(
        f"{SPEC.name}.build_manifest is a stub for dataset owners to fill in. Point raw_root at: "
        f"{SPEC.raw_hint} Then construct schema.Case rows and emit them to {out!r} via "
        f"benchmaxxing.datasets.base.finalize (respecting limit={limit!r})."
    )
