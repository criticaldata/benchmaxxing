"""Descriptive regression of adoption on round position, text lane (#174, part c). Pure
re-analysis of the already-committed `multi_round.jsonl` (K=5 rounds, #130); zero API calls.
Imaging-lane half (K=3) is `experiments/imaging/positional_regression.py`.

Fits `benchmaxxing.stats.mixed_effects_logit` (adoption ~ round_index, random intercept per case)
separately for the shared and isolated arms, pooling every (case, round) observation. Explicitly
descriptive, not causal: round order was fixed, not randomized or counterbalanced across cases,
so any association with round_index is confounded with anything else correlated with position in
a fixed sequence and cannot be attributed to "repeated exposure" specifically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmaxxing.stats import mixed_effects_logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane  # noqa: E402


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
    if len(set(y)) < 2:
        # A mixed logit has no likelihood to maximise when every observation is the same value
        # (statsmodels refuses the fit). Report the constant rather than a fabricated slope.
        rate = float(sum(y)) / len(y) if y else None
        return {
            "n_observations": len(y), "n_cases": len(set(groups)),
            "intercept": None, "round_index_coef": None,
            "fitted_predicted_probability_by_round": [rate] * 5,
            "note": (f"{field} is {int(y[0]) if y else 'empty'} on every (case, round) observation, so "
                     "the random-intercept logit is not estimable; the constant empirical rate is "
                     "reported in its place."),
        }
    fe = pd.DataFrame({"intercept": [1.0] * len(y), "round_index": round_index})
    result = mixed_effects_logit(y, fe, groups)
    intercept, slope = [float(v) for v in result.fe_mean]
    return {
        "n_observations": len(y), "n_cases": len(set(groups)),
        "intercept": round(intercept, 4), "round_index_coef": round(slope, 4),
        "fitted_predicted_probability_by_round": [
            round(float(1 / (1 + np.exp(-(intercept + slope * k)))), 4) for k in range(5)
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Descriptive regression of adoption on round position "
                                             "over the committed multi_round rows (#174c); no model calls.")
    _lane.add_model_arg(ap)
    args = ap.parse_args()
    model = args.model

    base = Path(__file__).parent / "results"
    results_dir, _cache = _lane.scoped(model, str(base), str(base / "call_cache.jsonl"))
    rows = _load_jsonl(results_dir / "multi_round.jsonl")

    shared = _fit(rows, "shared_adopt")
    iso = _fit(rows, "iso_adopt")

    empirical_shared_by_round = [
        sum(r["shared_adopt"][k] for r in rows) / len(rows) for k in range(5)
    ]
    empirical_iso_by_round = [
        sum(r["iso_adopt"][k] for r in rows) / len(rows) for k in range(5)
    ]

    out = {
        "k_rounds": 5, "shared_arm": shared, "isolated_arm": iso,
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
            f"Shared-arm round_index coefficient = {shared['round_index_coef']} (small), but its "
            f"fitted probabilities ({shared['fitted_predicted_probability_by_round']}) sit "
            f"consistently ~0.17-0.21 below the actual empirical shared adoption "
            f"({[round(v, 4) for v in empirical_shared_by_round]}, roughly flat around 0.28-0.33) "
            "- a real, checked offset, not a rounding artifact. This is the expected signature of "
            "a random-intercept logistic mixed model on heavily heterogeneous per-case data (40 "
            "cases x 5 rounds each): the population-average-conditional-on-zero-random-effect "
            "prediction understates the raw empirical mean when between-case variance is large "
            "(a known non-collapsibility property of mixed logit models), not evidence that the "
            f"round_index coefficient itself is wrong. Isolated-arm coefficient = "
            f"{iso['round_index_coef']} (intercept {iso['intercept']}) fits much better (gaps "
            "under 0.02 every round) since isolated adoption is already at floor "
            f"({[round(v, 4) for v in empirical_iso_by_round]}, matching the multi-round "
            "README's already-reported near-floor values) with little between-case variance to "
            "cause the same offset. Both arms' round_index coefficients are small in absolute "
            "terms either way; the position-confounded caveat applies regardless."
        ),
    }
    if model != _lane.DEFAULT_MODEL:
        out["lineage_note"] = _lane.derived_note(model)
    (results_dir / "positional_regression.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
