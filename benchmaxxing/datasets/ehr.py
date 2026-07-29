"""EHR adapter: structured resource-constraint context for the scrutiny stage (stage 5).

This source is not a diagnostic lane. It supplies the resource-constraint context that puts the
scrutiny-panel stakeholders under load (ICU-stay load per admission, budget pressure), conditioning
how the referee scrutinises a committee decision.

There is no single canonical MIMIC-IV "resource" table, so the entry point here is
:func:`load_resource_contexts`, a flexible loader over a structured CSV a contributor derives from
MIMIC-IV (icustays / admissions). ``build_manifest`` is intentionally not the entry point and raises
NotImplementedError pointing at that loader.

See scripts/mimic_iv/derive_ehr_resource_contexts.sql for a documented, reviewable derivation query.
icu_stay_count is a real, MIMIC-native signal (the number of distinct ICU stays recorded under a
hospital admission -- scenario_id is the admission's hadm_id, not an individual ICU stay), not a
fabricated one -- it stands in as a resource-load proxy, not a measurement of real bed occupancy or
staffing ratios. budget_pressure remains a fabricated proxy (hospital length of stay); MIMIC-IV has no
real cost tables. Earlier versions of this loader shipped two column names ("beds" and "staffing") for
what was algebraically one number, and separately emitted one row per ICU stay while both numeric
fields were admission-level constants, producing exact duplicate rows -- both caught in review and
fixed; see the script's header comment for the full history. Validated against a real MIMIC-IV
BigQuery export (#49).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality

SPEC = DatasetSpec(
    name="ehr",
    raw_hint=(
        "Derive from MIMIC-IV 'icustays' (join admissions): per scenario compute the distinct "
        "ICU-stay count for the admission and a hospital-LOS budget proxy, and write them to a "
        "structured CSV with columns scenario_id, icu_stay_count, budget_pressure (extra columns "
        "are kept as meta). Load it with load_resource_contexts, not build_manifest."
    ),
    modality=Modality.TEXT,
    notes=(
        "Not a diagnostic lane: feeds the scrutiny stage as resource-constraint context. The entry "
        "point is load_resource_contexts(csv_path) -> list[ResourceContext]; build_manifest is not "
        "used here and raises NotImplementedError."
    ),
)

# Columns every derived resource CSV must provide; anything else becomes ResourceContext.meta.
REQUIRED_COLUMNS = ("scenario_id", "icu_stay_count", "budget_pressure")
_NUMERIC_COLUMNS = ("icu_stay_count", "budget_pressure")


@dataclass(frozen=True)
class ResourceContext:
    """One resource-constraint scenario that loads the scrutiny panel.

    icu_stay_count is the number of distinct ICU stays recorded under the same hospital admission --
    a real, MIMIC-native signal used as a resource-load proxy (higher = more ICU transfers, plausibly
    a sicker or more unstable admission), not a measurement of real bed occupancy or staffing.
    budget_pressure is a fabricated numeric proxy (higher means tighter resources); MIMIC-IV has no
    real cost tables. Any extra CSV columns are preserved verbatim in ``meta``.
    """

    scenario_id: str
    icu_stay_count: float
    budget_pressure: float
    meta: dict = field(default_factory=dict)


def load_resource_contexts(csv_path) -> list[ResourceContext]:
    """Read a structured resource CSV into a list of :class:`ResourceContext`.

    The CSV must have columns scenario_id, icu_stay_count and budget_pressure; the numeric fields
    are parsed as float and any extra columns are collected into ``meta``.

    Raises FileNotFoundError if the path is missing and ValueError on a missing required column or a
    non-numeric numeric field (with the row index and scenario_id in the message).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Resource CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ValueError(
                f"Resource CSV {path} is missing required column(s) {missing}. "
                f"Expected columns: {list(REQUIRED_COLUMNS)}."
            )
        return [_row_to_context(row, index, path) for index, row in enumerate(reader)]


def _row_to_context(row: dict, index: int, path: Path) -> ResourceContext:
    scenario_id = row.get("scenario_id")
    scenario_id = "" if scenario_id is None else str(scenario_id).strip()
    values = {column: _parse_float(row.get(column), column, index, scenario_id, path)
              for column in _NUMERIC_COLUMNS}
    meta = {
        key: value
        for key, value in row.items()
        if key not in REQUIRED_COLUMNS and key is not None
    }
    return ResourceContext(
        scenario_id=scenario_id,
        icu_stay_count=values["icu_stay_count"],
        budget_pressure=values["budget_pressure"],
        meta=meta,
    )


def _parse_float(value, column: str, index: int, scenario_id: str, path: Path) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"{path} row {index} (scenario_id={scenario_id!r}): required numeric field "
            f"{column!r} is empty."
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{path} row {index} (scenario_id={scenario_id!r}): field {column!r} must be numeric, "
            f"got {value!r}."
        ) from None


def build_manifest(raw_root, out, limit=None):
    """Not the entry point for this source. Use :func:`load_resource_contexts` instead."""
    raise NotImplementedError(
        f"{SPEC.name}.build_manifest is intentionally not implemented: resource-constraint context "
        f"is not a diagnostic manifest lane. Load a structured CSV "
        f"(scenario_id, icu_stay_count, budget_pressure) with "
        f"benchmaxxing.datasets.ehr.load_resource_contexts instead "
        f"(raw_root={raw_root!r}, out={out!r}, limit={limit!r})."
    )