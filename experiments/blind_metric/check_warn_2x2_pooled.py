"""Pool the check x warn 2x2 over the MedQA, MedMCQA and SUPPORT2 cohorts (no API calls).

Reads the committed per-case rows each cohort's check_warn_2x2.py run wrote and reports, over the
pooled cases, each cell's decoy uptake and disclosure and an exact McNemar of each cell against the
plain note. Each case appears once per comparison, so the pooled tests stay exact. The committed
test_aware uptake (answer-only, "clinically best", both sentences) is shown beside the cells from the
same cases' blind_metric rows, since that is the single condition the 2x2 takes apart.

``check_vs_warn`` is the direct test of the two sentences against each other (exact McNemar of the
warning cell against the check cell), with the discordant counts of each cohort beside it. The tests
against the plain note do not say which sentence lowers uptake more; this one does.

``empty_replies`` lists the replies that came back with no text, as each cohort's summary recorded
them. An empty reply is scored as not following the note, so every one of them in a cell that is
reported lower pushes the reported difference up. ``empty_replies_worst_case`` repeats each test with
the empty replies scored against the difference it reports.

``--model`` pools another lineage's rows from each cohort's ``<slug>/`` subdirectory and writes
``experiments/blind_metric/results/<slug>/check_warn_2x2_pooled.json``; the default is the committed
gemini-2.5-flash-lite run, whose output and wording are unchanged.
"""
from __future__ import annotations

import argparse
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


def _empty(d):
    """Case id -> the cells whose reply was empty, from the cohort summary the runner wrote."""
    summary = json.loads((ROOT / d / "check_warn_2x2_summary.json").read_text())
    if "empty_replies" not in summary:
        raise SystemExit(f"{d}/check_warn_2x2_summary.json has no empty_replies: replay check_warn_2x2.py "
                         "for this cohort (zero calls from its cache) so the empty replies are recorded.")
    out = {}
    for cell, ids in summary["empty_replies"].items():
        for case_id in ids:
            out.setdefault(case_id, set()).add(cell)
    return out


def _aware(d):
    return {r["case_id"]: r["aware_is_decoy"] for r in map(json.loads, (ROOT / d / "blind_metric.jsonl").read_text().splitlines()) if r}


def _p6(p):
    """Six decimals, as committed; a p below 5e-7 keeps three significant figures instead of rounding to 0."""
    r = round(p, 6)
    return r if r else float(f"{p:.3g}")


def _pair(rows, a, b, col="is_decoy"):
    gain = sum(1 for r in rows if r[f"{b}_{col}"] and not r[f"{a}_{col}"])
    lose = sum(1 for r in rows if r[f"{a}_{col}"] and not r[f"{b}_{col}"])
    return {"gain": gain, "lose": lose, "pvalue": _p6(mcnemar(gain, lose).pvalue)}


def check_vs_warn(rows):
    """Exact McNemar of the warning cell against the check cell on decoy uptake.

    ``gain`` counts the cases that follow the note under the warning and not under the check,
    ``lose`` the cases that follow it under the check and not under the warning, so a warning that
    lowers uptake more than the check shows as lose above gain. The per-cohort counts are
    descriptive and carry no test of their own.
    """
    out = _pair(rows, "check", "warn")
    out["per_cohort"] = {}
    for name in dict.fromkeys(r["cohort"] for r in rows):
        p = _pair([r for r in rows if r["cohort"] == name], "check", "warn")
        out["per_cohort"][name] = {"gain": p["gain"], "lose": p["lose"]}
    return out


def rescore_empty(rows, a, b):
    """The rows with every empty reply in cells a and b scored against the difference between them.

    The lower cell is read off the committed scoring. An empty reply in that cell is counted as
    following the note and an empty reply in the other cell as not following it, which is the
    assignment least favourable to "the lower cell lowers uptake".
    """
    base = _pair(rows, a, b)
    lower, upper = (b, a) if base["lose"] >= base["gain"] else (a, b)
    return [{**r, f"{lower}_is_decoy": r[f"{lower}_is_decoy"] or lower in r["empty"],
             f"{upper}_is_decoy": r[f"{upper}_is_decoy"] and upper not in r["empty"]} for r in rows]


def worst_case(rows, a, b):
    """Exact McNemar of cell b against cell a after ``rescore_empty``."""
    return _pair(rescore_empty(rows, a, b), a, b)


DEFAULT_MODEL = "gemini-2.5-flash-lite"


def cohort_dirs(model):
    slug = model.replace("/", "_")
    return COHORTS if model == DEFAULT_MODEL else {k: f"{v}/{slug}" for k, v in COHORTS.items()}


def load_rows(model):
    """One model's rows over the three cohorts, each tagged with its cohort, its committed
    test_aware outcome and the cells whose reply was empty; returns (rows, cases per cohort)."""
    rows, per = [], {}
    for name, d in cohort_dirs(model).items():
        rs = _rows(d)
        a = _aware(d)
        e = _empty(d)
        assert {r["case_id"] for r in rs} <= set(a), name
        assert set(e) <= {r["case_id"] for r in rs}, name
        for r in rs:
            r["cohort"] = name
            r["aware_is_decoy"] = a[r["case_id"]]
            r["empty"] = e.get(r["case_id"], set())
        rows += rs
        per[name] = len(rs)
    return rows, per


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    model = ap.parse_args().model
    slug = model.replace("/", "_")
    rows, per = load_rows(model)
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
    empty = {k: [f"{r['cohort']} {r['case_id']}" for r in rows if k in r["empty"]] for k in ("baseline",) + CELLS}
    out = {
        "n": n, "per_cohort_n": per, "model": model, "new_api_calls_this_run": 0,
        "cells": cells,
        "committed_test_aware": test_aware,
        "vs_none": {c: _pair(rows, "none", c) for c in CELLS[1:]},
        "check_given_warn": _pair(rows, "warn", "both"),
        "warn_given_check": _pair(rows, "check", "both"),
        "check_vs_warn": check_vs_warn(rows),
        "empty_replies": empty,
        "empty_replies_worst_case": {
            "vs_none": {c: worst_case(rows, "none", c) for c in CELLS[1:]},
            "check_vs_warn": worst_case(rows, "check", "warn"),
        },
        "warn_vs_test_aware": _pair(
            [{**r, "ta_is_decoy": r["aware_is_decoy"]} for r in rows], "warn", "ta"),
        "both_vs_test_aware": _pair(
            [{**r, "ta_is_decoy": r["aware_is_decoy"]} for r in rows], "both", "ta"),
    }
    if model == DEFAULT_MODEL:
        out["read"] = (
            f"Pooled over {n} cases, decoy uptake is {cells['none']['decoy_uptake']} under the plain note, "
            f"{cells['check']['decoy_uptake']} with the check announced, {cells['warn']['decoy_uptake']} with "
            f"the warning and {cells['both']['decoy_uptake']} with both; the committed test_aware prompt, "
            f"which also drops the justification and adds 'clinically best', gives "
            f"{test_aware['decoy_uptake']} on the same cases; against the warning cell alone it moves "
            f"{out['warn_vs_test_aware']['gain']} cases one way and {out['warn_vs_test_aware']['lose']} the "
            f"other (P = {out['warn_vs_test_aware']['pvalue']}), so the warning accounts for the whole "
            f"test-aware drop. The rubric is named in "
            f"{sum(cells[c]['n_named_any'] for c in CELLS)} of {4 * n} justifications across the four cells.")
    else:
        def _p(x):
            return "< 1e-6" if x == 0 else f"= {x}"
        out["read"] = (
            f"Pooled over {n} cases, decoy uptake is {cells['none']['decoy_uptake']} under the plain note, "
            f"{cells['check']['decoy_uptake']} with the check announced, {cells['warn']['decoy_uptake']} with "
            f"the warning and {cells['both']['decoy_uptake']} with both; the committed test_aware prompt gives "
            f"{test_aware['decoy_uptake']} on the same cases. Against the plain note: check "
            f"{out['vs_none']['check']['gain']} vs {out['vs_none']['check']['lose']} (P {_p(out['vs_none']['check']['pvalue'])}), "
            f"warning {out['vs_none']['warn']['gain']} vs {out['vs_none']['warn']['lose']} (P {_p(out['vs_none']['warn']['pvalue'])}), "
            f"both {out['vs_none']['both']['gain']} vs {out['vs_none']['both']['lose']} (P {_p(out['vs_none']['both']['pvalue'])}). "
            f"Warning against check, the direct test of the two sentences: {out['check_vs_warn']['gain']} cases follow "
            f"the note under the warning only and {out['check_vs_warn']['lose']} under the check only "
            f"(P {_p(out['check_vs_warn']['pvalue'])}, exact McNemar, uncorrected). ")
        n_empty = sum(len(empty[c]) for c in CELLS)
        if n_empty:
            w = out["empty_replies_worst_case"]
            out["read"] += (
                f"{n_empty} replies in the four cells are empty and scored as not following the note; with each of "
                f"them scored against the difference a test reports: check {w['vs_none']['check']['gain']} vs "
                f"{w['vs_none']['check']['lose']} (P {_p(w['vs_none']['check']['pvalue'])}), warning "
                f"{w['vs_none']['warn']['gain']} vs {w['vs_none']['warn']['lose']} (P {_p(w['vs_none']['warn']['pvalue'])}), "
                f"both {w['vs_none']['both']['gain']} vs {w['vs_none']['both']['lose']} (P {_p(w['vs_none']['both']['pvalue'])}), "
                f"warning against check {w['check_vs_warn']['gain']} vs {w['check_vs_warn']['lose']} "
                f"(P {_p(w['check_vs_warn']['pvalue'])}). ")
        else:
            out["read"] += "No reply in the four cells is empty. "
        out["read"] += (
            f"The rubric is named in {sum(cells[c]['n_named_any'] for c in CELLS)} of {4 * n} justifications "
            f"across the four cells.")
    p = ROOT / "experiments/blind_metric/results" / ("" if model == DEFAULT_MODEL else slug) / "check_warn_2x2_pooled.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
