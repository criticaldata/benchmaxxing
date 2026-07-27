"""EHR adapter: structured resource-constraint context for the scrutiny stage (stage 5).

This source is not a diagnostic lane. It supplies the resource-constraint context that puts the
scrutiny-panel stakeholders under load (bed occupancy, staffing, budget pressure), conditioning how
the referee scrutinises a committee decision.

There is no single canonical MIMIC-IV "resource" table, so the entry point here is
:func:`load_resource_contexts`, a flexible loader over a structured CSV a contributor derives from
MIMIC-IV (icustays / beds / staffing). ``build_manifest`` is intentionally not the entry point and
raises NotImplementedError pointing at that loader.

See scripts/mimic_iv/derive_ehr_resource_contexts.sql for a documented, reviewable derivation
query. Validated against a real 87,287-row MIMIC-IV BigQuery export (#49): every row parsed
cleanly, including a real-world empty-string value in a non-required column (see
test_extra_column_empty_string_preserved_in_meta). The derivation bounds cross-patient
concurrency comparisons to the same anchor_year_group, since MIMIC-IV's per-patient date
shifting makes raw cross-patient timestamps otherwise incomparable -- see the script's header
comment for the full rationale.
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
        "Derive from MIMIC-IV 'icustays' (join beds / staffing): per scenario compute bed "
        "occupancy, a nurse-to-patient ratio, and a budget proxy, and write them to a structured "
        "CSV with columns scenario_id, staffing, beds, budget_pressure (extra columns are kept as "
        "meta). Load it with load_resource_contexts, not build_manifest."
    ),
    modality=Modality.TEXT,
    notes=(
        "Not a diagnostic lane: feeds the scrutiny stage as resource-constraint context. The entry "
        "point is load_resource_contexts(csv_path) -> list[ResourceContext]; build_manifest is not "
        "used here and raises NotImplementedError."
    ),
)

# Columns every derived resource CSV must provide; anything else becomes ResourceContext.meta.
REQUIRED_COLUMNS = ("scenario_id", "staffing", "beds", "budget_pressure")
_NUMERIC_COLUMNS = ("staffing", "beds", "budget_pressure")


@dataclass(frozen=True)
class ResourceContext:
    """One resource-constraint scenario that loads the scrutiny panel.

    staffing, beds and budget_pressure are numeric constraint proxies (higher budget_pressure means
    tighter resources). Any extra CSV columns are preserved verbatim in ``meta``.
    """

    scenario_id: str
    staffing: float
    beds: float
    budget_pressure: float
    meta: dict = field(default_factory=dict)


def load_resource_contexts(csv_path) -> list[ResourceContext]:
    """Read a structured resource CSV into a list of :class:`ResourceContext`.

    The CSV must have columns scenario_id, staffing, beds and budget_pressure; the numeric fields
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
        staffing=values["staffing"],
        beds=values["beds"],
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
        f"(scenario_id, staffing, beds, budget_pressure) with "
        f"benchmaxxing.datasets.ehr.load_resource_contexts instead "
        f"(raw_root={raw_root!r}, out={out!r}, limit={limit!r})."
    )
