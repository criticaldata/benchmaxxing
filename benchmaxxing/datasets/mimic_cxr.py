"""MIMIC-CXR adapter (imaging, Lane A).

Owners: implement ``build_manifest`` to turn the credentialed raw release into a shared manifest
via ``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality

SPEC = DatasetSpec(
    name="mimic_cxr",
    raw_hint=(
        "PhysioNet MIMIC-CXR-JPG v2.0.0: a 'files/pXX/pYYYY/sZZZZ/*.jpg' tree of frontal/lateral "
        "chest X-rays plus 'mimic-cxr-2.0.0-chexpert.csv' (per-study CheXpert labels) and "
        "'mimic-cxr-2.0.0-metadata.csv'. Requires credentialed PhysioNet access."
    ),
    modality=Modality.IMAGE,
    notes=(
        "Map each study to one Case: case_id=study id, patient_id=subject id, image_ref=jpg path, "
        "report=free-text report, label from the CheXpert label columns."
    ),
)


def build_manifest(raw_root, out, limit=None):
    """Build a manifest from the raw MIMIC-CXR release. Not yet implemented."""
    raise NotImplementedError(
        f"{SPEC.name}.build_manifest is a stub for dataset owners to fill in. Point raw_root at: "
        f"{SPEC.raw_hint} Then construct schema.Case rows and emit them to {out!r} via "
        f"benchmaxxing.datasets.base.finalize (respecting limit={limit!r})."
    )
