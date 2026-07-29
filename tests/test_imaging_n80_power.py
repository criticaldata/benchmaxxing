"""Guards for the expanded NIH referee cohort (n=80).

The n=35 primary cohort left the referee-versus-naive-gate contrast at achieved power 0.59 against a
requirement of 56 discordant pairs, so it had to be reported as under-powered. This cohort is a
superset of those 35 cases and exists only to power that one contrast.
"""
import csv
import json
from pathlib import Path

import pytest

R = Path(__file__).parent.parent / "experiments" / "imaging" / "results_n80"
PRIMARY = Path(__file__).parent.parent / "experiments" / "imaging" / "results"


def _rows(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_expanded_cohort_is_a_superset_of_the_primary_35():
    """If it were not a superset the two cohorts would not be comparable at all."""
    new = {r["case_id"] for r in csv.DictReader((R / "nih_manifest_80.csv").read_text().splitlines())}
    old = {r["case_id"] for r in csv.DictReader((PRIMARY / "nih_manifest.csv").read_text().splitlines())}
    assert len(new) == 80
    assert old <= new, sorted(old - new)


def test_every_expanded_row_plants_against_ground_truth():
    """Same post-#338 invariant the primary cohort carries: never against the model's clean read."""
    for name in ("imaging_cascade.jsonl", "imaging_referee.jsonl", "imaging_judge_referee.jsonl"):
        rows = _rows(R / name)
        assert len(rows) == 80, name
        assert all(r["wrong"] == "no" for r in rows), name


def test_referee_beats_the_gate_with_adequate_power():
    """The whole point of this cohort. Power must clear 0.8 and the interval must exclude zero."""
    blk = json.loads((R / "effect_sizes_n80.json").read_text())["referee_vs_naive"]
    assert blk["n"] == 80
    assert blk["achieved_power"] >= 0.8, blk["achieved_power"]
    lo, hi = blk["bootstrap95"]
    assert lo > 0, blk["bootstrap95"]
    assert blk["required_pairs_for_power_0.8"] <= blk["n"]


def test_effect_replicates_the_primary_estimate():
    """A tighter interval is only reassuring if it still contains the primary point estimate."""
    blk = json.loads((R / "effect_sizes_n80.json").read_text())["referee_vs_naive"]
    lo, hi = blk["bootstrap95"]
    assert lo <= 0.2571 <= hi, "n=35 point estimate falls outside the n=80 interval"
    assert blk["risk_difference"] == pytest.approx(0.25, abs=0.02)


def test_judge_still_collapses_onto_the_gate_at_n80():
    """The paper's exact-collapse claim must survive the cohort change, not just hold at n=35."""
    j = json.loads((R / "imaging_judge_referee_summary.json").read_text())["same_lineage_judge"]
    g = json.loads((R / "imaging_referee_summary.json").read_text())["naive_gate"]
    for k in ("precision", "recall", "fpr", "tp", "fp", "fn", "tn"):
        assert j[k] == g[k], f"{k}: judge {j[k]} vs gate {g[k]}"


def test_referee_false_alarm_rate_is_stable_across_cohorts():
    small = json.loads((PRIMARY / "imaging_referee_summary.json").read_text())["referee"]["fpr"]
    big = json.loads((R / "imaging_referee_summary.json").read_text())["referee"]["fpr"]
    assert abs(small - big) < 0.03, (small, big)
