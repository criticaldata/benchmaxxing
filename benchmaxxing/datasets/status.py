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
    "openi_cxr": DatasetStatus(
        lane="imaging",
        staged="not staged",
        adapter="coded loader",
        solo="pending",
        cascade="pending",
        plausibility="pending",
        referee="pending",
        blocker="real run tracked on #119",
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
    "probmed": DatasetStatus(
        lane="multimodal",
        staged="not staged",
        adapter="coded",
        solo="pending",
        cascade="pending",
        plausibility="pending",
        referee="pending",
        blocker="requires HF access to the gated rippleripple/ProbMed, staging test.json and images, then real VLM runs",
    ),
    "chexpert": DatasetStatus(
        lane="imaging",
        staged="staged",
        adapter="coded",
        solo="done",
        cascade="done",
        plausibility="not applicable",
        referee="done",
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
    "support2": DatasetStatus(
        lane="tabular",
        staged="open download, not staged locally",
        adapter="coded loader",
        solo="done",
        cascade="done",
        # A binary outcome has exactly one wrong option, so there is no plausible-vs-implausible
        # distractor gradient to vary. Structurally absent, not merely unrun.
        plausibility="not applicable",
        referee="done",
        # The runs landed, but solo is a null and the cascade is saturated. The lane README carries
        # the caveats; this field only tracks whether an arm was run.
        blocker="run on n=120 (#297); solo null, cascade at ceiling, see the lane README",
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
        staged="staged",
        adapter="coded",
        solo="done",
        cascade="done",
        plausibility="done",
        referee="done",
    ),
    "mimic_cxr_text": DatasetStatus(
        lane="text",
        staged="staged",
        adapter="coded",
        solo="done",
        cascade="done",
        plausibility="done",
        referee="done",
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
