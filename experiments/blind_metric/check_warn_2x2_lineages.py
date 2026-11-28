"""Warning against check across the locally served lineages (no API calls).

Each lineage's pooled file tests the warning cell against the check cell on that model's own cases
(``check_vs_warn`` in ``check_warn_2x2_pooled.json``). This script puts those tests side by side and
counts how many fall below 0.05, before and after a Holm correction over the models, so "the warning
lowers uptake more than the check" is read against a test and not against the cell rates.

It adds the one comparison that uses every model at once. The models answer the same cases, so their
answers to one case are not independent and the model-case pairs cannot be pooled into one McNemar
test. The unit is the case: its uptake summed over the models under the warning, against the same
sum under the check, and an exact sign test on the cases where the two sums differ.

``check_vs_warn_empty_replies_worst_case`` repeats all of it with every empty reply scored against
the difference (see ``check_warn_2x2_pooled.rescore_empty``).

Writes ``experiments/blind_metric/results/check_warn_2x2_lineages.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from benchmaxxing.stats import mcnemar, multiple_comparison

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_warn_2x2_pooled import ROOT, _p6, check_vs_warn, load_rows, rescore_empty  # noqa: E402

MODELS = ("openai/gpt-oss-120b", "nemotron-3-super-120b-a12b-fp8-local", "Qwen/Qwen2.5-VL-72B-Instruct")
ALPHA = 0.05


def over_models(rows_by_model):
    """The case as the unit: per case, decoy uptake summed over the models, warning minus check."""
    diff = defaultdict(int)
    for rows in rows_by_model.values():
        for r in rows:
            diff[(r["cohort"], r["case_id"])] += int(r["warn_is_decoy"]) - int(r["check_is_decoy"])
    lower = sum(1 for d in diff.values() if d < 0)
    higher = sum(1 for d in diff.values() if d > 0)
    return {"n_cases": len(diff), "cases_lower_under_warning": lower, "cases_higher_under_warning": higher,
            "cases_equal": len(diff) - lower - higher, "sign_test_p": _p6(mcnemar(higher, lower).pvalue)}


def compare(rows_by_model):
    """Per model the exact McNemar of warning against check, Holm over the models, and the case-level test."""
    per = {}
    for model, rows in rows_by_model.items():
        t = check_vs_warn(rows)
        per[model] = {"n": len(rows), "gain": t["gain"], "lose": t["lose"], "pvalue": t["pvalue"]}
    # Holm on the unrounded p of each model's discordant counts, rounded afterwards
    raw = [mcnemar(v["gain"], v["lose"]).pvalue for v in per.values()]
    for v, h in zip(per.values(), multiple_comparison(raw, method="holm").pvalues_adjusted):
        v["p_holm"] = _p6(float(h))
    return {
        "per_model": per,
        "n_models": len(per),
        "alpha": ALPHA,
        "n_models_below_alpha": sum(1 for v in per.values() if v["pvalue"] < ALPHA),
        "n_models_below_alpha_holm": sum(1 for v in per.values() if v["p_holm"] < ALPHA),
        "model_case_pairs": {"gain": sum(v["gain"] for v in per.values()), "lose": sum(v["lose"] for v in per.values())},
        "over_models": over_models(rows_by_model),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    models = ap.parse_args().models
    rows_by_model = {m: load_rows(m)[0] for m in models}
    worst_rows = {m: rescore_empty(rows, "check", "warn") for m, rows in rows_by_model.items()}
    out = {"models": models, "new_api_calls_this_run": 0, "check_vs_warn": compare(rows_by_model),
           "empty_replies_in_check_or_warn": {m: sum(1 for r in rows for c in ("check", "warn") if c in r["empty"])
                                              for m, rows in rows_by_model.items()},
           "check_vs_warn_empty_replies_worst_case": compare(worst_rows)}

    def _say(c):
        om = c["over_models"]
        per = "; ".join(f"{m} {v['gain']} vs {v['lose']} (P = {v['pvalue']})" for m, v in c["per_model"].items())
        return (f"{per}. Below {ALPHA}: {c['n_models_below_alpha']} of {c['n_models']} models uncorrected, "
                f"{c['n_models_below_alpha_holm']} of {c['n_models']} after Holm over the models. With the case as the "
                f"unit, uptake summed over the models is lower under the warning on {om['cases_lower_under_warning']} "
                f"cases and higher on {om['cases_higher_under_warning']} of {om['n_cases']} "
                f"(exact sign test P = {om['sign_test_p']}).")
    out["read"] = (
        "Warning against check, cases that follow the note under the warning only vs under the check only, "
        f"exact McNemar per model: {_say(out['check_vs_warn'])} With every empty reply in the two cells scored "
        f"against the difference: {_say(out['check_vs_warn_empty_replies_worst_case'])}")
    p = ROOT / "experiments/blind_metric/results/check_warn_2x2_lineages.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
