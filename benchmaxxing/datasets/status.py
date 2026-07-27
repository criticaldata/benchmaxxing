"""Dataset staged/coded/blocked status.

This module is the single source of truth for dataset readiness. The adapter registry says
which datasets are importable today; this module says where each dataset stands in the study
plan: data availability, adapter state, experiment coverage, and any blocker.
"""

from __future__ import annotations

from dataclasses import dataclass


ADAPTER_STATUSES = frozenset({"coded", "coded loader", "not registered"})
CODED_ADAPTER_STATUSES = frozenset({"coded", "coded loader"})


@dataclass(frozen=True)
class DatasetStatus:
    """Readiness status for one dataset in the benchmaxxing study plan."""

    lane: str
    staged: str
    adapter: str
    solo: str
    cascade: str
    plausibility: str
    referee: str
    blocker: str = ""


DATASET_STATUS: dict[str, DatasetStatus] = {
    "medqa": DatasetStatus(
        lane="text",
        staged="staged",
        adapter="coded",
        solo="done",
        cascade="done",
        plausibility="done",
        referee="done",
    ),
    "nih_cxr14": DatasetStatus(
        lane="imaging",
        staged="staged",
        adapter="coded",
        solo="done",
        cascade="done",
        plausibility="not applicable",
        referee="done",
    ),
    "pubmedqa": DatasetStatus(
        lane="text",
        staged="not staged",
        adapter="coded",
        solo="pending",
        cascade="pending",
        plausibility="pending",
        referee="pending",
        blocker="stage the official PubMedQA ori_pqal.json release",
    ),
    "chexpert": DatasetStatus(
        lane="imaging",
        staged="blocked",
        adapter="coded",
        solo="pending",
        cascade="pending",
        plausibility="not applicable",
        referee="pending",
        blocker="requires signed CheXpert license and local data staging",
    ),
    "mimic_cxr": DatasetStatus(
        lane="imaging+text",
        staged="blocked",
        adapter="coded",
        solo="pending",
        cascade="pending",
        plausibility="pending",
        referee="pending",
        blocker="requires credentialed PhysioNet MIMIC-CXR-JPG access",
    ),
    "ehr": DatasetStatus(
        lane="tabular",
        staged="not staged",
        adapter="coded loader",
        solo="pending",
        cascade="pending",
        plausibility="not applicable",
        referee="pending",
        blocker="requires a derived MIMIC-IV resource CSV",
    ),
    "medmcqa": DatasetStatus(
        lane="text",
        staged="not staged",
        adapter="coded",
        solo="pending",
        cascade="pending",
        plausibility="pending",
        referee="pending",
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
