"""The published MIMIC-CXR numbers must be recomputable from the shipped de-identified rows.

Those CSVs are the only row-level artefact a reader without PhysioNet access can check the paper
against, because the raw JSONL keys reads to MIMIC dicom_ids and is gitignored. They were once
produced by hand and silently fell a run behind the paper: recomputing from them reproduced
superseded numbers. This test is the guard against that recurring. It runs offline against the
committed CSVs, so every contributor exercises it, not only whoever holds the credentialed data.

The regeneration path needs the gitignored JSONL and is therefore skipped when it is absent; the
verification path is not, and must never be.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "experiments", "mimic_cxr_image", "export_deid.py")
_DEID = os.path.join(_ROOT, "experiments", "mimic_cxr_image", "results", "deid")


def _load():
    spec = importlib.util.spec_from_file_location("export_deid", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ed():
    if not os.path.exists(_SCRIPT):
        pytest.skip("export_deid.py not present")
    return _load()


def _cases(mod):
    return list(mod.CHECKS) + list(mod.UNOWNED_CHECKS)


def test_every_published_cell_recomputes(ed):
    """Each published MIMIC-CXR cell, recomputed from the CSVs alone."""
    failures = []
    for label, fn, want in _cases(ed):
        got = fn()
        if label.startswith("4.1"):          # a bound, not an equality
            ok = got[0] <= want[0] + 1e-9 and got[1] == want[1]
        else:
            ok = all(g is not None and abs(g - w) <= 0.011 for g, w in zip(got, want))
        if not ok:
            failures.append(f"{label}: got {got}, want {want}")
    assert not failures, "de-identified rows no longer reproduce the paper:\n" + "\n".join(failures)


def test_the_guard_can_actually_fail(ed):
    """A guard that cannot fail is the defect this paper is about, so prove this one can.

    Flipping referee flags must break the referee cells. If this test ever passes without the
    assertion firing, the check above is measuring nothing.
    """
    rows = ed.read_csv("referee.csv")
    flipped = 0
    for r in rows:
        if r["ref_flag"] == "1" and flipped < 12:
            r["ref_flag"] = "0"
            flipped += 1
    assert flipped == 12, "fixture no longer has enough positive flags to corrupt"
    original, ed.read_csv = ed.read_csv, lambda name: rows if name == "referee.csv" else original(name)
    try:
        assert ed.prf(rows, "ref_flag") != (0.41, 0.59, 0.21)
        assert ed.referee_restricted() != (0.66, 0.70, 0.36)
    finally:
        ed.read_csv = original


def test_no_credentialed_identifier_is_emitted(ed):
    """No dicom_id, study or subject id, path or free text in any shipped column or value."""
    banned = ed.FORBIDDEN
    for name in os.listdir(_DEID):
        if not name.endswith(".csv"):
            continue
        rows = ed.read_csv(name)
        assert rows, f"{name} is empty"
        assert not banned & set(rows[0]), f"{name} emits {banned & set(rows[0])}"
        assert rows[0].get("case_index") is not None, f"{name} is not keyed by case_index"
        if name == "provenance.csv":       # checksums are hex by design, not outcomes
            continue
        for col, val in rows[0].items():
            assert val == "" or val.lstrip("-").isdigit(), \
                f"{name}: column {col!r} holds non-numeric {val!r}"


def test_case_index_is_dense_and_ordered(ed):
    """case_index must be 0..n-1 in order, or a cross-file join by index is meaningless."""
    for name in os.listdir(_DEID):
        if not name.endswith(".csv"):
            continue
        idx = [int(r["case_index"]) for r in ed.read_csv(name)]
        # an empty file satisfies "dense and ordered" vacuously, which is how a truncated CSV
        # would slip through this check; require rows before believing the ordering
        assert idx, f"{name} has no rows"
        assert idx == list(range(len(idx))), f"{name} case_index is not dense and ordered"


def test_row_counts_match_the_published_contract(ed):
    """A truncated CSV is invisible to every other check: 0..9 is as dense as 0..833."""
    for name, want in sorted(ed.EXPECTED_ROWS.items()):
        if not os.path.exists(os.path.join(_DEID, name)):
            continue
        assert len(ed.read_csv(name)) == want, f"{name} is not {want} rows"


def test_cohorts_that_must_align_do(ed):
    """The 417-row arms share a manifest; the 834-row arms share the superset."""
    for a, b in (("referee.csv", "judge.csv"), ("judge.csv", "judge_with_image.csv"),
                 ("system_flag.csv", "strength_cascade.csv")):
        assert len(ed.read_csv(a)) == len(ed.read_csv(b)), f"{a} and {b} differ in length"
    sf = [r["clean_correct"] for r in ed.read_csv("system_flag.csv")]
    st = [r["clean_correct"] for r in ed.read_csv("strength_cascade.csv")]
    assert sf == st, "the two 834-row arms disagree on clean_correct; the orderings have drifted"


def test_derivation_rule_holds_where_it_can_be_checked(ed):
    """`clean != wrong` must reproduce the stored clean_correct. Needs the gitignored JSONL."""
    try:
        checked = ed.assert_derivation_holds()
    except FileNotFoundError:
        pytest.skip("raw per-case JSONL not present (gitignored under PhysioNet terms)")
    assert checked >= 600
