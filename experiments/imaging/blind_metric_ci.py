"""Blind-metric silent-gaming re-analysis with CIs and per-finding decoy uptake (#201, imaging
lane). Pure re-analysis of `results/imaging_blind_metric.jsonl`, zero API calls.

The imaging blind-metric probe already found the striking cross-lane contrast: 0 of the drifters
name the rubric (versus all 11 in the text lane), i.e. silent rather than self-declared gaming.
This adds Wilson 95% CIs to every proportion and a per-finding decoy-uptake breakdown (exploratory,
cells are small).
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
    rows = [json.loads(line) for line in (results_dir / "imaging_blind_metric.jsonl").read_text().splitlines() if line.strip()]

    base = _rate_with_ci([r["base_is_decoy"] for r in rows])
    blind = _rate_with_ci([r["blind_is_decoy"] for r in rows])
    aware = _rate_with_ci([r["aware_is_decoy"] for r in rows])

    drifters = [r for r in rows if r["blind_is_decoy"] and not r["base_is_decoy"]]
    naming = _rate_with_ci([r["named_rubric_when_drifted"] for r in drifters]) if drifters else \
        {"n": 0, "rate": None, "wilson95": [None, None]}

    by_finding = defaultdict(list)
    for r in rows:
        by_finding[r["finding"]].append(r["blind_is_decoy"])
    per_finding = {f: _rate_with_ci(v) for f, v in sorted(by_finding.items(), key=lambda kv: -len(kv[1]))}

    out = {
        "n": len(rows),
        "decoy_uptake": {"baseline": base, "blind": blind, "test_aware": aware},
        "decoy_uptake_delta_blind_minus_baseline": round(blind["rate"] - base["rate"], 4),
        "test_awareness_suppression_delta_blind_minus_aware": round(blind["rate"] - aware["rate"], 4),
        "naming_rate_among_drifters": naming,
        "decoy_uptake_by_finding": per_finding,
        "read": (
            f"With Wilson intervals attached: decoy uptake under the hidden rubric is "
            f"{blind['rate']} {blind['wilson95']} vs a {base['rate']} baseline; the naming rate "
            f"among the {naming['n']} drifters is {naming['rate']} {naming['wilson95']}, i.e. not "
            "one drifter named the rubric (silent gaming), with an interval that stays low even "
            "at its upper bound given this many drifters. Per-finding uptake is exploratory only "
            "(cells as small as 1-3 cases); no finding stands out as categorically resistant."
        ),
    }
    (results_dir / "imaging_blind_metric_ci.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
