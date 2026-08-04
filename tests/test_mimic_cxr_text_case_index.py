"""The de-identified MIMIC-CXR text rows must still be joinable back to a cohort.

De-identifying the lane (replacing `case_id` with `case_index`) is only free if everything that
reads those rows was migrated with them. Two things went wrong the first time and these pin both:

1. Four arms (`break_it_a`, `break_it_d`, `push_c`, `deliberation_framing`) pick their hard-case
   cohort out of the committed `solo_records.jsonl`, and still read `r["case_id"]`, so they raised
   `KeyError` on the file the repo ships. Caught in review by @Agastya191 on #411.
2. `referee_deployable` writes two rows per case, `planted` and `clean`, which used to be told
   apart by a `::clean` suffix on the id. Dropping the suffix made the bare index collide across
   arms, so re-deriving the published `with_clean_control` cell scored 40 rows instead of 80 and
   gave tp=0 where the paper has tp=14.

Offline: no key, no network, no API calls. These read the committed rows and a synthetic manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mimic_cxr_text.case_index import (
    INDEX_SPACE_SIZE,
    CaseIndexError,
    build_index_map,
    hard_case_indices,
    hard_cases,
)

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "mimic_cxr_text" / "results"

# Files whose per-case rows are keyed on the index.
KEYED_FILES = [
    "blind_metric.jsonl",
    "break_it_a_per_case.jsonl",
    "break_it_d_per_case.jsonl",
    "deliberation_framing.jsonl",
    "push_c_per_case.jsonl",
    "referee_deployable.jsonl",
    "referee_judge.jsonl",
    "solo_records.jsonl",
]


class _Case:
    """Enough of schema.Case for the index map, which only reads `case_id`."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line.strip()]


def _full_manifest() -> list[_Case]:
    return [_Case(f"mimic-cxr-text-{10000000 + i}-{50000000 + i}") for i in range(INDEX_SPACE_SIZE)]


@pytest.mark.parametrize("name", KEYED_FILES)
def test_committed_rows_carry_an_index_and_no_identifier(name):
    rows = _rows(name)
    assert rows, f"{name} is empty"
    for r in rows:
        assert "case_index" in r, f"{name} row lost its join key: {sorted(r)}"
        assert "case_id" not in r, f"{name} row still carries a MIMIC identifier"
        assert isinstance(r["case_index"], int)


def test_hard_case_indices_reads_the_committed_deidentified_file():
    """The regression itself: this raised KeyError('case_id') on the shipped file."""
    hard = hard_case_indices(RESULTS / "solo_records.jsonl")
    solo = _rows("solo_records.jsonl")
    assert hard == {r["case_index"] for r in solo if r["clean_correct"] is False}
    # 474 of 600 clean-correct, so 126 hard. Pins the cohort size the four arms are defined over.
    assert len(hard) == 126


def test_hard_cases_selects_the_same_cohort_the_committed_arms_ran_on():
    """Each of the four arms' committed rows must be inside the cohort the code now selects."""
    hard = hard_case_indices(RESULTS / "solo_records.jsonl")
    for name in ("break_it_a_per_case.jsonl", "break_it_d_per_case.jsonl",
                 "push_c_per_case.jsonl", "deliberation_framing.jsonl"):
        ran = {r["case_index"] for r in _rows(name)}
        assert ran <= hard, f"{name} holds cases the hard-case filter would not select"


def test_hard_cases_joins_a_manifest_on_the_index():
    cases = _full_manifest()
    picked = hard_cases(cases, RESULTS / "solo_records.jsonl")
    hard = hard_case_indices(RESULTS / "solo_records.jsonl")
    index_of = build_index_map(cases)
    assert {index_of[c.case_id] for c in picked} == hard
    assert len(hard_cases(cases, RESULTS / "solo_records.jsonl", limit=20)) == 20


def test_hard_cases_still_works_on_pre_migration_rows(tmp_path):
    """A local un-migrated solo file joins on case_id, so an existing run does not break."""
    records = tmp_path / "solo_old.jsonl"
    records.write_text(
        '{"case_id": "mimic-cxr-text-10000000-50000000", "clean_correct": false}\n'
        '{"case_id": "mimic-cxr-text-10000001-50000001", "clean_correct": true}\n'
    )
    picked = hard_cases(_full_manifest(), records)
    assert [c.case_id for c in picked] == ["mimic-cxr-text-10000000-50000000"]


def test_a_partial_manifest_is_an_error_not_a_different_cohort():
    """Ranking a subset silently renumbers nearly every case, so it must fail loudly."""
    with pytest.raises(CaseIndexError, match="633"):
        build_index_map(_full_manifest()[:600])


def test_index_is_the_rank_over_the_sorted_ids():
    cases = _full_manifest()
    index_of = build_index_map(cases)
    assert index_of == {c.case_id: i for i, c in enumerate(sorted(cases, key=lambda c: c.case_id))}


def test_referee_rows_need_the_arm_in_the_key_to_reproduce_the_published_cell():
    """A bare case_index collapses the planted and clean arms onto each other."""
    rows = _rows("referee_deployable.jsonl")
    assert len(rows) == 80
    assert len({r["case_index"] for r in rows}) == 40, "planted and clean share an index"
    keyed = {(r["case_index"], r["arm"]) for r in rows}
    assert len(keyed) == 80, "the arm is what makes the join key unique"

    def contingency(field):
        adopted = {(r["case_index"], r["arm"]): r["adopted"] for r in rows}
        pred = {(r["case_index"], r["arm"]): r[field] for r in rows}
        tp = sum(1 for k in adopted if pred[k] and adopted[k])
        fp = sum(1 for k in adopted if pred[k] and not adopted[k])
        return tp, fp

    # The published with_clean_control cell: naive gate tp=14 fp=66, deployable tp=14 fp=12.
    assert contingency("naive") == (14, 66)
    assert contingency("deployable") == (14, 12)

    # Keyed on the index alone, the same rows lose every true positive.
    bare_adopted = {r["case_index"]: r["adopted"] for r in rows}
    bare_pred = {r["case_index"]: r["naive"] for r in rows}
    assert sum(1 for k in bare_adopted if bare_pred[k] and bare_adopted[k]) == 0
