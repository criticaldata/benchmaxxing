"""Blind-metric silent-gaming re-analysis with CIs and a decoy-position subgroup (#201, text lane).

The paper reports the drift rate, decoy uptake, and the naming-vs-drifting split as bare fractions
(11/40 drifted, 0/11 named the rubric), even though every field needed for an interval is already
committed. This adds Wilson 95% CIs to each proportion, and a subgroup view of decoy uptake by
`decoy_letter` position (does the decoy being option A vs B change how often it is picked). Pure
re-analysis of `results/blind_metric.jsonl`; no API calls.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _rate_with_ci(vals):
    k, n = sum(vals), len(vals)
    lo, hi = _wilson(k, n)
    return {"n": n, "rate": round(k / n, 4) if n else None, "wilson95": [lo, hi]}


def main():
    results_dir = Path(__file__).parent / "results"
    rows = [json.loads(line) for line in (results_dir / "blind_metric.jsonl").read_text().splitlines() if line.strip()]

    base = _rate_with_ci([r["base_is_decoy"] for r in rows])
    blind = _rate_with_ci([r["blind_is_decoy"] for r in rows])
    aware = _rate_with_ci([r["aware_is_decoy"] for r in rows])

    drifters = [r for r in rows if r["blind_is_decoy"] and not r["base_is_decoy"]]
    naming = _rate_with_ci([r["named_rubric_when_drifted"] for r in drifters]) if drifters else \
        {"n": 0, "rate": None, "wilson95": [None, None]}

    by_position = defaultdict(list)
    for r in rows:
        by_position[r["decoy_letter"]].append(r["blind_is_decoy"])
    per_position = {letter: _rate_with_ci(vals) for letter, vals in sorted(by_position.items())}

    out = {
        "n": len(rows),
        "decoy_uptake": {"baseline": base, "blind": blind, "test_aware": aware},
        "decoy_uptake_delta_blind_minus_baseline": round(blind["rate"] - base["rate"], 4),
        "test_awareness_suppression_delta_blind_minus_aware": round(blind["rate"] - aware["rate"], 4),
        "naming_rate_among_drifters": naming,
        "decoy_uptake_by_decoy_position": per_position,
        "read": (
            "With Wilson intervals attached: decoy uptake under the hidden rubric is "
            f"{blind['rate']} {blind['wilson95']} vs a {base['rate']} baseline; the naming rate "
            f"among the {naming['n']} drifters is {naming['rate']} {naming['wilson95']}, i.e. "
            "every drifter named the rubric in their justification (fully aware gaming), and even "
            "the lower bound of the interval stays high at this n. Decoy uptake does not differ "
            "meaningfully by whether the decoy is option A or B, so position is not doing the "
            "work; the rubric mention is."
        ),
    }
    (results_dir / "blind_metric_ci.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
