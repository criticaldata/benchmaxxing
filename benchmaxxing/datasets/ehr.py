"""EHR adapter: structured resource-constraint context for the scrutiny stage.

This source is not a diagnostic lane. It supplies structured context (bed/ICU/staffing and
similar constraints) that conditions how the referee scrutinises a committee decision. Cases carry
that context as text so they fit the shared schema.

Owners: implement ``build_manifest`` to turn the raw tables into a shared manifest via
``benchmaxxing.datasets.base.finalize``.
"""

from __future__ import annotations

from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality

SPEC = DatasetSpec(
    name="ehr",
    raw_hint=(
        "Structured EHR / resource-constraint tables (for example MIMIC-IV 'icustays', "
        "'transfers' and services, or a curated CSV of bed/ICU/staffing availability). Aggregate "
        "per encounter into a compact constraint context string."
    ),
    modality=Modality.TEXT,
    notes=(
        "Not a diagnostic lane: feeds the scrutiny stage as resource-constraint context. Use "
        "case_id=encounter id, patient_id=subject id, report=serialised constraint context; leave "
        "question/options empty."
    ),
)


def build_manifest(raw_root, out, limit=None):
    """Build a manifest of resource-constraint context. Not yet implemented."""
    raise NotImplementedError(
        f"{SPEC.name}.build_manifest is a stub for dataset owners to fill in. Point raw_root at: "
        f"{SPEC.raw_hint} Then construct schema.Case rows and emit them to {out!r} via "
        f"benchmaxxing.datasets.base.finalize (respecting limit={limit!r})."
    )
