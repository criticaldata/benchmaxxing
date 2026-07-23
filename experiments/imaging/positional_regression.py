"""Descriptive regression of adoption on round position, imaging lane (#174, part c). Pure
re-analysis of the already-committed `imaging_multi_round.jsonl` (K=3 rounds, #169); zero API
calls. Text-lane half (K=5, #130) is `experiments/cascade/positional_regression.py`.

Fits `benchmaxxing.stats.mixed_effects_logit` (adoption ~ round_index, random intercept per case)
separately for the shared and isolated arms, pooling every (case, round) observation. This is
explicitly a DESCRIPTIVE regression, not a causal one: round order was fixed (not randomized or
counterbalanced across cases), so any association between adoption and round_index is confounded
with whatever else correlates with position in a fixed sequence (fatigue, drift in the holdout's
own state, etc.) and cannot be attributed to "repeated exposure" specifically. Reported as a
descriptive slope with that caveat, per the issue's own framing ("position-confounded").
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import numpy as np

from benchmaxxing.stats import mixed_effects_logit


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _pool(rows, field):
    y, round_index, groups = [], [], []
    for r in rows:
        for k, adopt in enumerate(r[field]):
            y.append(1 if adopt else 0)
            round_index.append(k)
            groups.append(r["case_id"])
    return y, round_index, groups


def _fit(rows, field):
    y, round_index, groups = _pool(rows, field)
    fe = pd.DataFrame({"intercept": [1.0] * len(y), "round_index": round_index})
    result = mixed_effects_logit(y, fe, groups)
    intercept, slope = [float(v) for v in result.fe_mean]
    return {
        "n_observations": len(y), "n_cases": len(set(groups)),
        "intercept": round(intercept, 4), "round_index_coef": round(slope, 4),
        "fitted_predicted_probability_by_round": [
            round(float(1 / (1 + np.exp(-(intercept + slope * k)))), 4) for k in range(3)
        ],
    }


def main():
    results_dir = Path(__file__).parent / "results"
    rows = _load_jsonl(results_dir / "imaging_multi_round.jsonl")

    shared = _fit(rows, "shared_adopt")
    iso = _fit(rows, "iso_adopt")

    empirical_shared_by_round = [
        sum(r["shared_adopt"][k] for r in rows) / len(rows) for k in range(3)
    ]
    empirical_iso_by_round = [
        sum(r["iso_adopt"][k] for r in rows) / len(rows) for k in range(3)
    ]

    out = {
        "k_rounds": 3, "shared_arm": shared, "isolated_arm": iso,
        "empirical_adoption_by_round": {
            "shared": [round(v, 4) for v in empirical_shared_by_round],
            "isolated": [round(v, 4) for v in empirical_iso_by_round],
        },
        "position_confounded_caveat": (
            "Round order was fixed, not randomized or counterbalanced across cases, so any "
            "round_index coefficient reflects position in a fixed sequence, not a validated "
            "'repeated exposure' effect - it cannot rule out fatigue, drift, or other "
            "position-correlated confounds. Descriptive only."
        ),
        "read": (
            f"Shared-arm round_index coefficient = {shared['round_index_coef']} "
            f"(intercept {shared['intercept']}), a substantial positive log-odds slope. But the "
            "linear-in-round-index model this fits is misleading here: it predicts a monotonic "
            "climb (round 0/1/2 predicted probabilities approximately 0.94/0.99/1.00), while the "
            "already-reported empirical per-round adoption (#169's README) is NOT monotonic - "
            "0.89, 1.00, 0.97 (up then slightly down). The large coefficient is an artifact of "
            "forcing a straight-line log-odds fit onto a saturating, non-monotonic empirical "
            "curve, not evidence of a genuine escalating exposure effect; a linear descriptive "
            "slope is the wrong summary for this shape and should not be read as 'adoption keeps "
            f"climbing with more rounds'. Isolated-arm coefficient = {iso['round_index_coef']} "
            f"(intercept {iso['intercept']}) is comparatively small and closer to the already-"
            "reported flat isolated-arm pattern. The position-confounded caveat above applies to "
            "both regardless of this fit-quality issue: round order was never randomized."
        ),
    }
    (results_dir / "positional_regression.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
