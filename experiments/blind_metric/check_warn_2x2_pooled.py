"""Pool the check x warn 2x2 over the MedQA, MedMCQA and SUPPORT2 cohorts (no API calls).

Reads the committed per-case rows each cohort's check_warn_2x2.py run wrote and reports, over the
pooled cases, each cell's decoy uptake and disclosure and an exact McNemar of each cell against the
plain note. Each case appears once per comparison, so the pooled tests stay exact. The committed
test_aware uptake (answer-only, "clinically best", both sentences) is shown beside the cells from the
same cases' blind_metric rows, since that is the single condition the 2x2 takes apart.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.stats import mcnemar

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {
    "MedQA": "experiments/blind_metric/results",
    "MedMCQA": "experiments/medmcqa/results",
    "SUPPORT2": "experiments/support2/results",
}
CELLS = ("none", "check", "warn", "both")


def _rows(d):
    return [json.loads(line) for line in (ROOT / d / "check_warn_2x2.jsonl").read_text().splitlines() if line.strip()]


def _aware(d):
    return {r["case_id"]: r["aware_is_decoy"] for r in map(json.loads, (ROOT / d / "blind_metric.jsonl").read_text().splitlines()) if r}


def _pair(rows, a, b, col="is_decoy"):
    gain = sum(1 for r in rows if r[f"{b}_{col}"] and not r[f"{a}_{col}"])
    lose = sum(1 for r in rows if r[f"{a}_{col}"] and not r[f"{b}_{col}"])
    return {"gain": gain, "lose": lose, "pvalue": round(mcnemar(gain, lose).pvalue, 6)}


def main():
    rows, aware, per = [], {}, {}
    for name, d in COHORTS.items():
        rs = _rows(d)
        a = _aware(d)
        assert {r["case_id"] for r in rs} <= set(a), name
        for r in rs:
            r["cohort"] = name
            r["aware_is_decoy"] = a[r["case_id"]]
        rows += rs
        per[name] = len(rs)
    n = len(rows)
    cells = {}
    for c in CELLS:
        ddrift = [r for r in rows if r[f"{c}_declared"] == r["decoy_letter"]]
        cells[c] = {"decoy_uptake": round(sum(r[f"{c}_is_decoy"] for r in rows) / n, 4),
                    "n_decoy": sum(r[f"{c}_is_decoy"] for r in rows),
                    "n_declared_drifted": len(ddrift),
                    "n_named_when_declared_drifted": sum(r[f"{c}_named"] for r in ddrift),
                    "n_named_any": sum(r[f"{c}_named"] for r in rows)}
    test_aware = {"decoy_uptake": round(sum(r["aware_is_decoy"] for r in rows) / n, 4),
                  "n_decoy": sum(r["aware_is_decoy"] for r in rows)}
    out = {
        "n": n, "per_cohort_n": per, "model": "gemini-2.5-flash-lite", "new_api_calls_this_run": 0,
        "cells": cells,
        "committed_test_aware": test_aware,
        "vs_none": {c: _pair(rows, "none", c) for c in CELLS[1:]},
        "check_given_warn": _pair(rows, "warn", "both"),
        "warn_given_check": _pair(rows, "check", "both"),
        "both_vs_test_aware": _pair(
            [{**r, "ta_is_decoy": r["aware_is_decoy"]} for r in rows], "both", "ta"),
    }
    out["read"] = (
        f"Pooled over {n} cases, decoy uptake is {cells['none']['decoy_uptake']} under the plain note, "
        f"{cells['check']['decoy_uptake']} with the check announced, {cells['warn']['decoy_uptake']} with "
        f"the warning and {cells['both']['decoy_uptake']} with both; the committed test_aware prompt, "
        f"which also drops the justification and adds 'clinically best', gives "
        f"{test_aware['decoy_uptake']} on the same cases. The rubric is named in "
        f"{sum(cells[c]['n_named_any'] for c in CELLS)} of {4 * n} justifications across the four cells.")
    p = ROOT / "experiments/blind_metric/results/check_warn_2x2_pooled.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
