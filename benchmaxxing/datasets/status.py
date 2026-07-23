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
        experiments="solo, cascade, referee, and blind-metric runs done",
    ),
    "nih_cxr14": DatasetStatus(
        lane="imaging",
        data="staged",
        adapter="coded",
        experiments="solo, cascade, referee, and blind-metric pilot runs done",
    ),
    "pubmedqa": DatasetStatus(
        lane="text",
        data="not staged",
        adapter="coded",
        experiments="pending solo and cascade runs",
        blocker="stage the official PubMedQA ori_pqal.json release",
    ),
    "chexpert": DatasetStatus(
        lane="imaging",
        data="blocked",
        adapter="coded",
        experiments="pending natural-cue cascade",
        blocker="requires signed CheXpert license and local data staging",
    ),
    "mimic_cxr": DatasetStatus(
        lane="imaging+text",
        data="blocked",
        adapter="coded",
        experiments="pending image and report cascade runs",
        blocker="requires credentialed PhysioNet MIMIC-CXR-JPG access",
    ),
    "ehr": DatasetStatus(
        lane="tabular",
        data="deferred",
        adapter="coded loader",
        experiments="pending scrutiny-stage resource-constraint scenarios",
        blocker="requires a derived MIMIC-IV resource CSV",
    ),
    "medmcqa": DatasetStatus(
        lane="text",
        data="not staged",
        adapter="not registered",
        experiments="pending solo and cascade runs",
        blocker="adapter is not registered in this package yet",
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
