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

``one_session`` reads the same four prompts from a second, already committed source. In the cells
above the plain note is replayed from the hidden-note cache of an earlier serving session and only
the three other cells are new, so serving differences between sessions sit inside every comparison
with the plain note. ``matched_conditions.py`` asked the four prompts (its conditions ``note``,
``note_salient``, ``note_flagged`` and ``note_salient_flagged``) five times each, interleaved in one
session per cohort, with the decoy fixed from a baseline read made in that session. Those rows give
every test with all four cells from one session: on the first repeat, which is what one read per
cell means there, and on the majority of the five repeats. ``read_to_read`` counts how often a cell's
outcome on one case differs between the two sources and between the five repeats of one session.

Writes ``experiments/blind_metric/results/check_warn_2x2_lineages.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmaxxing.stats import mcnemar, multiple_comparison

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_warn_2x2_pooled import CELLS, ROOT, _p6, _pair, check_vs_warn, load_rows, rescore_empty  # noqa: E402

MODELS = ("openai/gpt-oss-120b", "nemotron-3-super-120b-a12b-fp8-local", "Qwen/Qwen2.5-VL-72B-Instruct")
ALPHA = 0.05
# where matched_conditions.py wrote each cohort's one-session rows, and its name for each cell
ONE_SESSION = {
    "MedQA": "experiments/blind_metric/results/matched",
    "MedMCQA": "experiments/medmcqa/results/blind_metric/matched",
    "SUPPORT2": "experiments/support2/results/blind_metric/matched",
}
CONDITION = {"none": "note", "check": "note_salient", "warn": "note_flagged", "both": "note_salient_flagged"}


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_one_session(model, read):
    """One model's one-session rows in the shape of the 2x2 rows, and each cohort's call window.

    ``read`` is ``first`` (repeat 0 of each cell) or ``majority`` (the decoy in more than half of the
    repeats). ``empty`` marks the cells whose first-repeat reply had no text, from the per-call rows.
    """
    slug = model.replace("/", "_")
    rows, sessions = [], {}
    for name, d in ONE_SESSION.items():
        calls = [c for c in _jsonl(ROOT / d / slug / "matched_conditions_calls.jsonl") if c["condition"] in CONDITION.values()]
        blank = defaultdict(set)
        for c in calls:
            if c["rep"] == 0 and not (c["text"] or "").strip():
                blank[c["case_id"]].add(next(k for k, v in CONDITION.items() if v == c["condition"]))
        cases = _jsonl(ROOT / d / slug / "matched_conditions.jsonl")
        for r in cases:
            row = {"case_id": r["case_id"], "cohort": name, "empty": blank.get(r["case_id"], set())}
            for cell, cond in CONDITION.items():
                reps = r[f"{cond}_is_decoy"]
                row[f"{cell}_is_decoy"] = bool(reps[0]) if read == "first" else sum(reps) * 2 > len(reps)
            rows.append(row)
        ts = [c["ts"] for c in calls]
        sessions[name] = {"n": len(cases), "repeats": len(cases[0]["note_is_decoy"]), "calls": len(calls),
                          "first_call": _utc(min(ts)), "last_call": _utc(max(ts))}
    return rows, sessions


def read_to_read(model):
    """How much one cell's outcome moves between reads of the same prompt, per cell.

    ``differ`` counts the cases, among those the single-read rows and the one-session rows share with
    the same decoy, whose single-read outcome is not the one-session first repeat (two reads in two
    sessions). ``first_two_repeats_differ`` counts the one-session cases whose first and second
    repeat are not the same outcome (two reads in one session, minutes apart), and
    ``unequal_repeats`` those whose five repeats are not all the same.
    """
    slug = model.replace("/", "_")
    single = {(r["cohort"], r["case_id"]): r for r in load_rows(model)[0]}
    out = {c: {"shared_cases": 0, "differ": 0, "one_session_cases": 0, "first_two_repeats_differ": 0,
               "unequal_repeats": 0} for c in CELLS}
    for name, d in ONE_SESSION.items():
        for r in _jsonl(ROOT / d / slug / "matched_conditions.jsonl"):
            s = single.get((name, r["case_id"]))
            for cell, cond in CONDITION.items():
                reps = r[f"{cond}_is_decoy"]
                out[cell]["one_session_cases"] += 1
                out[cell]["first_two_repeats_differ"] += reps[0] != reps[1]
                out[cell]["unequal_repeats"] += len(set(reps)) > 1
                if s is not None and s["decoy_letter"] == r["decoy"]:
                    out[cell]["shared_cases"] += 1
                    out[cell]["differ"] += bool(reps[0]) != s[f"{cell}_is_decoy"]
    return out


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


def compare(rows_by_model, detail=False):
    """Per model the exact McNemar of warning against check, Holm over the models, and the case-level test.

    ``detail`` adds each cell's uptake and its test against the plain note, for rows that have no
    pooled file of their own.
    """
    per = {}
    for model, rows in rows_by_model.items():
        t = check_vs_warn(rows)
        per[model] = {"n": len(rows), "gain": t["gain"], "lose": t["lose"], "pvalue": t["pvalue"]}
        if detail:
            per[model]["decoy_uptake"] = {c: round(sum(r[f"{c}_is_decoy"] for r in rows) / len(rows), 4) for c in CELLS}
            per[model]["vs_none"] = {c: _pair(rows, "none", c) for c in CELLS[1:]}
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

    first = {m: load_one_session(m, "first") for m in models}
    first_rows = {m: v[0] for m, v in first.items()}
    out["one_session"] = {
        "source": "matched_conditions.py rows under each cohort's matched/<model>/ directory: conditions note, "
                  "note_salient, note_flagged and note_salient_flagged, five repeats each at temperature 0, "
                  "interleaved in one session per cohort; the decoy is fixed from a baseline read made in that session",
        "sessions": {m: v[1] for m, v in first.items()},
        "first_repeat": compare(first_rows, detail=True),
        "first_repeat_empty_replies_in_check_or_warn": {
            m: sum(1 for r in rows for c in ("check", "warn") if c in r["empty"]) for m, rows in first_rows.items()},
        "first_repeat_empty_replies_worst_case": compare(
            {m: rescore_empty(rows, "check", "warn") for m, rows in first_rows.items()}),
        "majority_of_repeats": compare({m: load_one_session(m, "majority")[0] for m in models}, detail=True),
        "read_to_read": {m: read_to_read(m) for m in models},
    }

    def _say(c):
        om = c["over_models"]
        per = "; ".join(f"{m} {v['gain']} vs {v['lose']} (P = {v['pvalue']})" for m, v in c["per_model"].items())
        return (f"{per}. Below {ALPHA}: {c['n_models_below_alpha']} of {c['n_models']} models uncorrected, "
                f"{c['n_models_below_alpha_holm']} of {c['n_models']} after Holm over the models. With the case as the "
                f"unit, uptake summed over the models is lower under the warning on {om['cases_lower_under_warning']} "
                f"cases and higher on {om['cases_higher_under_warning']} of {om['n_cases']} "
                f"(exact sign test P = {om['sign_test_p']}).")
    one = out["one_session"]
    out["read"] = (
        "Warning against check, cases that follow the note under the warning only vs under the check only, "
        f"exact McNemar per model: {_say(out['check_vs_warn'])} With every empty reply in the two cells scored "
        f"against the difference: {_say(out['check_vs_warn_empty_replies_worst_case'])} With all four cells "
        f"from one session (matched_conditions.py), first repeat: {_say(one['first_repeat'])} Majority of the "
        f"five repeats: {_say(one['majority_of_repeats'])}")
    p = ROOT / "experiments/blind_metric/results/check_warn_2x2_lineages.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
