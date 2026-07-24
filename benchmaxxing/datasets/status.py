"""Dataset staged/coded/blocked status.

This module is the single source of truth for dataset readiness. The adapter registry says
which datasets are importable today; this module says where each dataset stands in the study
plan: data availability, adapter state, experiment coverage, and any blocker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetStatus:
    """Readiness status for one dataset in the benchmaxxing study plan."""

    lane: str
    data: str
    adapter: str
    experiments: str
    blocker: str = ""


DATASET_STATUS: dict[str, DatasetStatus] = {
    "medqa": DatasetStatus(
        lane="text",
        data="staged",
        adapter="coded",
        experiments="coverage grid done: solo, cascade, plausibility, referee",
    ),
    "nih_cxr14": DatasetStatus(
        lane="imaging",
        data="staged",
        adapter="coded",
        experiments="coverage grid done: solo, cascade, referee; plausibility not applicable",
    ),
    "pubmedqa": DatasetStatus(
        lane="text",
        data="not staged",
        adapter="coded",
        experiments="coverage grid pending: solo, cascade, plausibility, referee",
        blocker="stage the official PubMedQA ori_pqal.json release",
    ),
    "chexpert": DatasetStatus(
        lane="imaging",
        data="blocked",
        adapter="coded",
        experiments="coverage grid pending: solo, cascade, referee; plausibility not applicable",
        blocker="requires signed CheXpert license and local data staging",
    ),
    "mimic_cxr": DatasetStatus(
        lane="imaging+text",
        data="blocked",
        adapter="coded",
        experiments="coverage grid pending: solo, cascade, plausibility, referee",
        blocker="requires credentialed PhysioNet MIMIC-CXR-JPG access",
    ),
    "ehr": DatasetStatus(
        lane="tabular",
        data="not staged",
        adapter="coded loader",
        experiments="coverage grid pending: solo, cascade, referee; plausibility not applicable",
        blocker="requires a derived MIMIC-IV resource CSV",
    ),
    "medmcqa": DatasetStatus(
        lane="text",
        data="not staged",
        adapter="coded",
        experiments="coverage grid pending: solo, cascade, plausibility, referee",
        blocker="stage the official MedMCQA release",
    ),
}


def names():
    """Return dataset names with status entries."""

    return sorted(DATASET_STATUS)


def get(name: str):
    """Return the dataset status for ``name`` with a helpful error if missing."""

    try:
        return DATASET_STATUS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset status {name!r}. Available: {names()}") from None
