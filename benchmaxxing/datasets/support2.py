"""SUPPORT2 adapter (tabular, Lane C).

SUPPORT2 (Knaus et al. 1995) is a 9,105-patient cohort of seriously ill hospitalised adults with
the structured predictors and outcomes used for prognosis modelling. Each row becomes a text-lane
``Case``: the record is rendered as a patient vignette followed by a binary prognostic question,
and the ordered field list is kept in ``meta["fields"]`` so the cues in
:mod:`benchmaxxing.cues.tabular` can perturb fields and re-render the same record.

Only baseline clinical predictors are rendered (``FEATURE_FIELDS``). Columns that encode or leak
the outcome are excluded; see :data:`EXCLUDED`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from benchmaxxing.cues.tabular import RECORD_HEADER, compose_question
from benchmaxxing.datasets.base import DatasetSpec, finalize
from benchmaxxing.schema import Case, Modality

SPEC = DatasetSpec(
    name="support2",
    raw_hint=(
        "SUPPORT2 (UCI ML repository id 880, or the widely mirrored support2.csv): one CSV, one "
        "row per patient, with columns age / sex / dzgroup / meanbp / crea / ... plus the outcome "
        "flags 'hospdead' and 'death'. Pass the CSV itself, or a directory holding support2.csv, "
        "as raw_root."
    ),
    modality=Modality.TEXT,
    notes=(
        "Tabular lane rendered into the text schema: question = patient vignette + binary "
        "prognostic question, options = (survives, dies), answer_index from the target column. "
        "meta carries 'fields' (the ordered record), 'absent' (columns with no value for this "
        "patient), 'stem', 'header' and 'target', which is what the tabular cues perturb."
    ),
)

# Binary outcome column -> (question stem, options). Option 0 is always the good outcome, so
# answer_index is the raw 0/1 flag.
TARGETS = {
    "hospdead": (
        "Will this patient survive to hospital discharge?",
        ("Survives to hospital discharge", "Dies in hospital"),
    ),
    "death": (
        "Will this patient be alive at the end of the study follow-up period?",
        ("Alive at follow-up", "Dead at follow-up"),
    ),
}

# The rendered record: (column, label, unit). Order is the clean vignette's order.
FEATURE_FIELDS = (
    ("age", "age", "years"),
    ("sex", "sex", ""),
    ("race", "race", ""),
    ("dzgroup", "primary disease group", ""),
    ("dzclass", "disease class", ""),
    ("num.co", "number of comorbidities", ""),
    ("ca", "cancer status", ""),
    ("diabetes", "diabetes", ""),
    ("dementia", "dementia", ""),
    ("scoma", "SUPPORT coma score", ""),
    ("adls", "activities of daily living", ""),
    ("hday", "day of hospital stay at enrolment", ""),
    ("meanbp", "mean arterial pressure", "mmHg"),
    ("hrt", "heart rate", "bpm"),
    ("resp", "respiratory rate", "breaths/min"),
    ("temp", "temperature", "C"),
    ("wblc", "white blood cell count", "10^3/uL"),
    ("pafi", "PaO2/FiO2 ratio", ""),
    ("alb", "serum albumin", "g/dL"),
    ("bili", "bilirubin", "mg/dL"),
    ("crea", "serum creatinine", "mg/dL"),
    ("sod", "serum sodium", "mEq/L"),
    ("ph", "arterial pH", ""),
    ("glucose", "serum glucose", "mg/dL"),
    ("bun", "blood urea nitrogen", "mg/dL"),
    ("urine", "urine output", "mL/24h"),
)

# Deliberately never rendered: the SUPPORT and APACHE models' own survival estimates and the
# physiology scores behind them (they are predictions of the label), the outcome and follow-up
# columns themselves, DNR status (recorded alongside the dying process), the post-enrolment
# treatment-intensity score, the cost columns, and the ADL variants redundant with 'adls'.
EXCLUDED = (
    "surv2m", "surv6m", "prg2m", "prg6m", "sps", "aps", "avtisst",
    "hospdead", "death", "d.time", "slos", "sfdm2", "dnr", "dnrday",
    "charges", "totcst", "totmcst", "adlp", "adlsc",
)

_MISSING_TOKENS = {"", "na", "n/a", "nan", ".", "null", "none", "?"}
_BINARY_LABELS = {"diabetes": ("no", "yes"), "dementia": ("no", "yes")}
_DISPLAY_DECIMALS = 2    # chart precision; the source stores labs at raw float precision
_INDEX_COLUMN = ""       # name given to the distributed CSV's unnamed leading row-index column
_ID_COLUMNS = ("id", "ID", "Unnamed: 0", _INDEX_COLUMN)
_TRUE_TOKENS = {"1", "yes", "true", "dead", "died"}
_FALSE_TOKENS = {"0", "no", "false", "alive", "survived"}


def _clean(value) -> str | None:
    """Strip a raw cell, mapping the dataset's missing markers to None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING_TOKENS else text


def _as_number(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _display(column: str, value: str) -> str:
    """Render one cell: 0/1 flags as no/yes, numbers at chart precision, else verbatim.

    The source stores values at float precision (``1.7998047`` for an albumin of 1.8), which no
    chart would show and which leaves ``cues.tabular.precision_inflation`` nothing to pad. Numbers
    are rounded to 2 decimals and trailing zeros dropped, so the clean record reads like a chart.
    """
    labels = _BINARY_LABELS.get(column)
    number = _as_number(value)
    if labels is not None and number in (0.0, 1.0):
        return labels[int(number)]
    if number is None:
        return value
    return f"{round(number, _DISPLAY_DECIMALS):g}"


def _parse_outcome(value) -> int | None:
    """The binary outcome flag as 0/1, or None when it is missing or unparseable."""
    text = _clean(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in _TRUE_TOKENS:
        return 1
    if lowered in _FALSE_TOKENS:
        return 0
    number = _as_number(text)
    return int(number) if number in (0.0, 1.0) else None


def _split_fields(row: dict) -> tuple[list[dict], list[dict]]:
    """Split the row into the rendered present fields and a record of the absent ones.

    The absent list is what ``cues.tabular.missingness_recode`` restates as "not recorded", so it
    has to be carried even though it contributes nothing to the clean vignette.
    """
    fields: list[dict] = []
    absent: list[dict] = []
    for column, label, unit in FEATURE_FIELDS:
        value = _clean(row.get(column))
        if value is None:
            absent.append({"key": column, "label": label})
            continue
        fields.append(
            {"key": column, "label": label, "value": _display(column, value), "unit": unit}
        )
    return fields, absent


def _row_id(row: dict, index: int) -> str:
    for column in _ID_COLUMNS:
        value = _clean(row.get(column))
        if value is not None:
            return value
    return str(index)


def _case_from_row(row: dict, index: int, target: str, stem: str, options) -> Case | None:
    """One SUPPORT2 row as a Case, or None when it has no scoreable outcome or no features."""
    outcome = _parse_outcome(row.get(target))
    if outcome is None:
        return None
    fields, absent = _split_fields(row)
    if not fields:
        return None
    case_id = f"support2-{_row_id(row, index)}"
    return Case(
        case_id=case_id,
        patient_id=case_id,
        modality=Modality.TEXT,
        label=options[outcome],
        question=compose_question(fields, stem),
        options=options,
        answer_index=outcome,
        meta={
            "fields": fields,
            "absent": absent,
            "stem": stem,
            "header": RECORD_HEADER,
            "target": target,
            "source": SPEC.name,
        },
    )


def resolve_csv(raw_root) -> Path:
    """Return the CSV to parse: ``raw_root`` itself, or ``support2.csv`` inside it."""
    root = Path(raw_root)
    if root.is_dir():
        for name in ("support2.csv", "SUPPORT2.csv", "support2.data.csv"):
            candidate = root / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No support2.csv found in SUPPORT2 directory: {root}")
    if not root.exists():
        raise FileNotFoundError(f"SUPPORT2 CSV not found: {root}")
    return root


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Read the CSV into row dicts, repairing its unnamed leading row-index column.

    The distributed support2.csv writes a row index that its header line does not name, so every
    data row carries one field more than the header. Read naively that shifts every column by one
    and 'hospdead' silently becomes 'sex', which would make the whole lane's ground truth the
    patient's sex. When the first data row is exactly one field longer, the header is realigned with
    a leading unnamed id column.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        names = header
        rows: list[dict] = []
        for index, values in enumerate(reader):
            if index == 0 and len(values) == len(header) + 1:
                names = [_INDEX_COLUMN] + header
            rows.append(dict(zip(names, values)))
    return names, rows


def _require_columns(header, target: str, path: Path) -> None:
    if target not in header:
        raise ValueError(
            f"{path}: SUPPORT2 CSV is missing the target column {target!r}. "
            f"Available targets: {sorted(TARGETS)}."
        )
    if not any(column in header for column, _, _ in FEATURE_FIELDS):
        raise ValueError(
            f"{path}: SUPPORT2 CSV has none of the expected feature columns "
            f"(age, sex, dzgroup, meanbp, crea, ...); is this the right file?"
        )


def build_manifest(raw_root, out, limit=None, target: str = "hospdead"):
    """Parse the SUPPORT2 CSV at ``raw_root`` into a manifest at ``out``.

    ``raw_root`` is the CSV itself or a directory holding ``support2.csv``. ``target`` selects the
    binary outcome (``"hospdead"``, the default, or ``"death"``); rows whose target flag is missing
    are skipped rather than guessed at, since they have no ground truth to score. ``limit`` keeps
    only the first N usable rows.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of: {sorted(TARGETS)}")
    stem, options = TARGETS[target]
    path = resolve_csv(raw_root)

    names, rows = read_rows(path)
    _require_columns(names, target, path)

    cases: list[Case] = []
    for index, row in enumerate(rows):
        if limit is not None and len(cases) >= limit:
            break
        case = _case_from_row(row, index, target, stem, options)
        if case is not None:
            cases.append(case)

    if not cases:
        raise ValueError(
            f"{path}: no rows had both a usable {target!r} outcome and at least one feature value."
        )
    return finalize(cases, out)
