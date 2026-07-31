"""Answer-preserving tabular cues (structured-record perturbations, Lane C).

Every builder operates on a tabular-lane :class:`~benchmaxxing.schema.Case`: a patient record
serialized into ``question`` plus the ordered field list kept in ``meta["fields"]``. Each returns a
:class:`~benchmaxxing.schema.TwinPair` whose ``clean`` payload is the untouched record and whose
``contaminated`` payload carries a single injected cue.

Five of the six cues are *information-identical*: the contaminated record states exactly the same
clinical facts in a different surface form (fields reordered, values restated in equivalent units,
decimals padded, one field named twice, an already-absent value stated as "not recorded"). A model
that changes its prognosis under them is reading surface form, not evidence. ``administrative_hint``
is the exception and is reported separately: it appends a line that was not in the record at all.
Options and ``answer_index`` are never touched, so the ground truth is preserved throughout.

Payloads are plain dicts with the keys ``question``, ``options`` (a tuple), ``answer_index`` (an
int), ``report`` and ``fields`` (the perturbed record). No API keys, no optional dependencies,
deterministic for a given input.
"""

from __future__ import annotations

import random

from benchmaxxing.schema import Case, TwinPair

RECORD_HEADER = "Patient record:"

# Same measurement, different unit: the physical quantity is unchanged and only the number and its
# unit label move. Keyed by field key; a field not listed here is left alone.
_UNIT_CONVERSIONS = {
    "temp": (lambda v: v * 9.0 / 5.0 + 32.0, "F"),
    "crea": (lambda v: v * 88.4, "umol/L"),
    "bili": (lambda v: v * 17.1, "umol/L"),
    "bun": (lambda v: v * 0.357, "mmol/L"),
    "glucose": (lambda v: v * 0.0555, "mmol/L"),
    "alb": (lambda v: v * 10.0, "g/L"),
}

# A second name for the same measurement, used by redundant_restatement.
_SYNONYMS = {
    "meanbp": "MAP",
    "hrt": "pulse",
    "resp": "respirations",
    "temp": "core temperature",
    "crea": "creatinine",
    "wblc": "leukocyte count",
}

# The cues under which the contaminated record states exactly the same facts as the clean one.
# administrative_hint adds a fact, so it is excluded from this set and reported on its own.
INFORMATION_IDENTICAL = (
    "field_order",
    "unit_rescale",
    "precision_inflation",
    "redundant_restatement",
    "missingness_recode",
)


def render_fields(fields) -> str:
    """Render the record as one ``- label: value unit`` line per field, in list order."""

    lines = []
    for field in fields:
        value = "" if field.get("value") is None else str(field["value"]).strip()
        unit = str(field.get("unit") or "").strip()
        lines.append(f"- {field.get('label')}: {(value + ' ' + unit).strip()}")
    return "\n".join(lines)


def compose_question(fields, stem: str, header: str = RECORD_HEADER) -> str:
    """The full stimulus: the rendered record followed by the prognostic question."""

    return f"{header}\n{render_fields(fields)}\n\n{stem}"


def _require_tabular_case(case: Case) -> list[dict]:
    """Validate the case and return a mutable copy of its field list."""

    meta = case.meta or {}
    fields = meta.get("fields")
    if not fields:
        raise ValueError("tabular cue requires case.meta['fields'] to be a non-empty list")
    if not meta.get("stem"):
        raise ValueError("tabular cue requires case.meta['stem'] (the prognostic question)")
    if not case.options:
        raise ValueError("tabular cue requires case.options to be a non-empty tuple")
    if case.answer_index is None or not (0 <= case.answer_index < len(case.options)):
        raise ValueError(
            f"case.answer_index={case.answer_index} is out of range for "
            f"{len(case.options) if case.options else 0} options"
        )
    return [dict(f) for f in fields]


def _as_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _payload(case: Case, fields) -> dict:
    meta = case.meta or {}
    return {
        "question": compose_question(fields, meta["stem"], meta.get("header", RECORD_HEADER)),
        "options": tuple(case.options),
        "answer_index": int(case.answer_index),
        "report": case.report,
        "fields": [dict(f) for f in fields],
    }


def _twin(case: Case, cue_type: str, cue_params: dict, clean_fields, dirty_fields) -> TwinPair:
    return TwinPair(
        case_id=case.case_id,
        cue_type=cue_type,
        cue_params=cue_params,
        clean=_payload(case, clean_fields),
        contaminated=_payload(case, dirty_fields),
        ground_truth=case.options[case.answer_index],
    )


def field_order_permutation(case: Case, seed: int = 0) -> TwinPair:
    """Deterministically permute the order of the record's fields.

    Same fields, same values, different order. A model that weights a row by where a value sits
    rather than what it says shifts its prognosis.
    """

    fields = _require_tabular_case(case)
    if len(fields) < 2:
        raise ValueError("field_order needs at least 2 fields to permute")

    # stdlib random, not numpy: this module is imported by the SUPPORT2 adapter and therefore by
    # the dataset registry, which stays dependency-light so `benchmaxxing datasets list` is cheap.
    perm = list(range(len(fields)))
    random.Random(seed).shuffle(perm)
    # Never emit the identity: the contaminated twin would be byte-identical and the cue would
    # silently never fire.
    if perm == list(range(len(fields))):
        perm = perm[1:] + perm[:1]

    return _twin(
        case, "field_order", {"seed": int(seed), "permutation": perm},
        fields, [fields[i] for i in perm],
    )


def unit_rescale(case: Case, keys=None) -> TwinPair:
    """Restate numeric fields in an equivalent alternative unit (mg/dL to SI, C to F).

    Every convertible field present is converted unless ``keys`` narrows the set. The measurement
    is unchanged; only the number and its unit label move, so a model reading magnitudes rather
    than clinical values shifts its prognosis.
    """

    fields = _require_tabular_case(case)
    wanted = set(_UNIT_CONVERSIONS) if keys is None else set(keys)

    converted: list[str] = []
    out: list[dict] = []
    for field in fields:
        key = field.get("key")
        value = _as_float(field.get("value"))
        if key in _UNIT_CONVERSIONS and key in wanted and value is not None:
            convert, unit = _UNIT_CONVERSIONS[key]
            rescaled = dict(field)
            rescaled["value"] = _fmt(convert(value))
            rescaled["unit"] = unit
            out.append(rescaled)
            converted.append(str(key))
        else:
            out.append(dict(field))

    if not converted:
        raise ValueError(
            "unit_rescale found no convertible numeric field on this case "
            f"(convertible keys: {sorted(_UNIT_CONVERSIONS)})"
        )
    return _twin(case, "unit_rescale", {"converted": converted}, fields, out)


def precision_inflation(case: Case, decimals: int = 3) -> TwinPair:
    """Re-render decimal values with extra trailing digits (78.3 -> 78.300).

    Only values already written with a decimal point are padded, so integer counts stay integers,
    and a value already carrying more decimals than ``decimals`` is left alone: rounding it down
    would change the number, which is the one thing this cue must not do. Only the apparent
    precision of the workup goes up.
    """

    fields = _require_tabular_case(case)
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")

    touched: list[str] = []
    out: list[dict] = []
    for field in fields:
        raw = str(field.get("value") or "")
        value = _as_float(raw)
        padded = dict(field)
        if value is not None and "." in raw and len(raw.rsplit(".", 1)[1]) < decimals:
            padded["value"] = f"{value:.{decimals}f}"
            touched.append(str(field.get("key")))
        out.append(padded)

    if not touched:
        raise ValueError(
            f"precision_inflation found no field to pad to {decimals} decimals on this case "
            "(every value is an integer or already at least that precise)"
        )
    return _twin(
        case, "precision_inflation", {"decimals": int(decimals), "fields": touched}, fields, out
    )


def redundant_restatement(case: Case, key: str | None = None) -> TwinPair:
    """State one field a second time under a synonym, immediately after the original.

    The duplicate carries the same value and unit under another name, so it adds no information.
    A model that weights a fact by how often it appears shifts its prognosis.
    """

    fields = _require_tabular_case(case)
    candidates = [
        i for i, f in enumerate(fields)
        if f.get("key") in _SYNONYMS and (key is None or f.get("key") == key)
    ]
    if not candidates:
        raise ValueError(
            f"redundant_restatement found no restatable field on this case (key={key!r}; "
            f"restatable keys: {sorted(_SYNONYMS)})"
        )

    index = candidates[0]
    source = fields[index]
    duplicate = dict(source)
    duplicate["label"] = _SYNONYMS[source["key"]]
    duplicate["key"] = f"{source['key']}__restated"
    out = fields[: index + 1] + [duplicate] + fields[index + 1 :]

    return _twin(
        case, "redundant_restatement",
        {"key": source.get("key"), "as_label": duplicate["label"]}, fields, out,
    )


def missingness_recode(case: Case, keys=None, marker: str = "not recorded") -> TwinPair:
    """State already-absent fields explicitly instead of omitting them.

    Omitting a value and writing "not recorded" say the same thing (it is unknown), so the evidence
    is unchanged; only whether the gap is spoken aloud moves. The absent fields come from
    ``meta["absent"]``, which the adapter fills with the columns this patient had no value for.
    """

    fields = _require_tabular_case(case)
    absent = list((case.meta or {}).get("absent") or [])
    if keys is not None:
        wanted = set(keys)
        absent = [a for a in absent if a.get("key") in wanted]
    if not absent:
        raise ValueError(
            "missingness_recode needs case.meta['absent'] to list at least one absent field; "
            "this record has no missing values to state explicitly"
        )

    marker_text = str(marker).strip()
    if not marker_text:
        raise ValueError("missingness_recode requires a non-empty marker")

    out = fields + [
        {"key": a.get("key"), "label": a.get("label"), "value": marker_text, "unit": ""}
        for a in absent
    ]
    return _twin(
        case, "missingness_recode",
        {"marker": marker_text, "keys": [a.get("key") for a in absent]}, fields, out,
    )


def administrative_hint(case: Case, hint: str) -> TwinPair:
    """Append a non-clinical administrative annotation to the record.

    The one cue here that is not a pure re-rendering: it adds a line the record did not contain (an
    insurance class, a chart-review flag). The outcome label is still untouched so the ground truth
    is preserved, but the stimulus is a hint rather than the same facts restated. This is the
    tabular analogue of the text lane's ``demographic_hint`` and is reported separately for that
    reason.
    """

    fields = _require_tabular_case(case)
    text = str(hint).strip()
    if not text:
        raise ValueError("administrative_hint requires a non-empty hint")

    out = fields + [
        {"key": "administrative_note", "label": "administrative note", "value": text, "unit": ""}
    ]
    return _twin(case, "administrative_hint", {"hint": text}, fields, out)


def whitespace_null(case: Case) -> TwinPair:
    """The null control: re-render the record changing **only** whitespace.

    Every other cue here changes something a hypothesis could plausibly care about: order, units,
    precision, duplication, an absence made explicit, an added line. This one changes one space per
    line, turning ``- label: value`` into ``- label : value``. Same fields, same order, same values,
    same units, same count. Nothing a clinician or a correct model could act on differs.

    That makes it the comparator this lane actually needs. The temperature-1.0 noise floor answers
    "how much does this model wobble when the decoding changes?", which is a different question from
    "how much does it wobble when the bytes change at temperature 0?". A cue only earns a
    susceptibility claim if it moves the prognosis more than this does.

    Perturbs every line on purpose, so the byte-level scale is comparable to ``field_order`` and the
    contrast isolates semantic content rather than edit size.
    """

    fields = _require_tabular_case(case)
    out = []
    for field in fields:
        padded = dict(field)
        padded["label"] = "{} ".format(field.get("label") or "")
        out.append(padded)

    if render_fields(out) == render_fields(fields):
        raise ValueError("whitespace_null produced an identical record; the cue would never fire")

    return _twin(case, "whitespace_null", {}, fields, out)


_DISPATCH = {
    "field_order": field_order_permutation,
    "unit_rescale": unit_rescale,
    "precision_inflation": precision_inflation,
    "redundant_restatement": redundant_restatement,
    "missingness_recode": missingness_recode,
    "administrative_hint": administrative_hint,
    "whitespace_null": whitespace_null,
}


def build_tabular_twin(case: Case, cue_type: str, **params) -> TwinPair:
    """Dispatch to a named tabular cue and return its :class:`TwinPair`.

    ``cue_type`` is one of the keys of :data:`_DISPATCH`; extra keyword arguments are forwarded to
    the underlying builder (for example ``seed=`` for the field order or ``hint=`` for the
    administrative hint).
    """

    if cue_type not in _DISPATCH:
        valid = ", ".join(sorted(_DISPATCH))
        raise ValueError(f"unknown tabular cue_type {cue_type!r}; expected one of: {valid}")
    return _DISPATCH[cue_type](case, **params)
