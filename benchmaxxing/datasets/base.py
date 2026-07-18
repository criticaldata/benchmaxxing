"""Shared scaffolding for dataset adapters.

Each dataset lives in its own module that publishes a :class:`DatasetSpec` (metadata plus a
raw-layout hint for owners) and a ``build_manifest`` entry point. Adapters call :func:`finalize`
to validate and emit a manifest in the shared schema.Case format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Modality


@dataclass(frozen=True)
class DatasetSpec:
    """Owner-free description of a source dataset.

    name: short registry key, e.g. "mimic_cxr".
    raw_hint: where to obtain the raw data and the on-disk layout ``build_manifest`` expects.
    modality: which lane the produced cases feed (IMAGE or TEXT).
    notes: free-form guidance (licensing, label conventions, caveats).
    """

    name: str
    raw_hint: str
    modality: Modality
    notes: str = ""


def finalize(cases, out_path) -> Path:
    """Validate a list of schema.Case and write it to a manifest at ``out_path``.

    Enforces that every ``case_id`` is unique, then delegates to
    :func:`benchmaxxing.data.write_manifest`. Raises ValueError on duplicate ids.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            duplicates.add(case.case_id)
        seen.add(case.case_id)
    if duplicates:
        raise ValueError(f"Duplicate case_id values are not allowed: {sorted(duplicates)}")
    return write_manifest(cases, out_path)
