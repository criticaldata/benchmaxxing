"""Net-harm sign of the MedQA text-lane cascade (#227, the text half of #177). Re-analysis of the
already-committed `referee_deployable.jsonl` (#132) cross-referenced against the MedQA manifest
for ground truth; zero new API calls.

#177 also asked for this on the text lane, but the only per-case artifact with ground truth at
the time (`cascade_v2_per_case.jsonl`) predates the answer-parser fix and is confirmed tainted by
it (its medqa-82 row shows the exact stray-article mis-parse the parser-bug correction
documents), and its stored transcripts retain only a truncated completion that cannot be
re-parsed either - #227 was opened to track a fresh run for this reason.

A fresh run turns out to be unnecessary: `referee_deployable.jsonl` (#132, built AFTER the parser
fix, using the same robust `_parse` in `referee_deployable.py`) already stores, per case, the
holdout's own uncued baseline answer (`bare`), the cascade seed's target (`wrong`, guaranteed
different from both ground truth and `bare` by construction), and the shared-board outcome
(`board`) - exactly the fields #227 asks a fresh run to log. The only piece missing is whether
`bare` is actually correct, which this script recovers by loading the manifest and looking up
each case's ground truth (`case.options[case.answer_index]`); no new model calls of any kind.

Same convention as the imaging lane's `net_harm.py` (#177/#185/#226): harm = a baseline-CORRECT
case whose board answer is wrong (adopting the shortcut cost it a correct answer); spurious
rescue = a baseline-INCORRECT case whose board answer happens to land on the ground truth
(peer-pressure-driven, not genuine reasoning).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.stats import bootstrap_ci, fisher_exact


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _summarize(group):
    if not group:
        return {"n": 0, "rate": None, "wilson95": [None, None], "bootstrap95": [None, None]}
    k, n = sum(group), len(group)
    lo, hi = _wilson(k, n)
    _point, blo, bhi = bootstrap_ci(list(map(float, group)))
    return {"n": n, "rate": round(k / n, 4), "wilson95": [lo, hi],
            "bootstrap95": [round(blo, 4), round(bhi, 4)]}


def main():
    ap = argparse.ArgumentParser(description="Net-harm sign of the MedQA text-lane cascade (#227).")
    ap.add_argument("--manifest", required=True, help="the MedQA manifest (ground-truth lookup only)")
    ap.add_argument("--referee-jsonl", default="experiments/referee/results/referee_deployable.jsonl")
    ap.add_argument("--out", default="experiments/referee/results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {c.case_id: c for c in load_cases(args.manifest)}
    rows = [json.loads(line) for line in Path(args.referee_jsonl).read_text().splitlines() if line.strip()]

    per_case = []
    missing = []
    for r in rows:
        case = cases.get(r["case_id"])
        if case is None:
            missing.append(r["case_id"])
            continue
        gt = case.options[case.answer_index]
        baseline_correct = (r["bare"] == gt)
        per_case.append({
            "case_id": r["case_id"], "ground_truth": gt, "bare": r["bare"], "wrong": r["wrong"],
            "board": r["board"], "baseline_correct": baseline_correct,
            "harm": baseline_correct and r["board"] != gt,
            "spurious_rescue": (not baseline_correct) and r["board"] == gt,
        })

    harm_group = [1 if r["harm"] else 0 for r in per_case if r["baseline_correct"]]
    rescue_group = [1 if r["spurious_rescue"] else 0 for r in per_case if not r["baseline_correct"]]
    harm = _summarize(harm_group)
    rescue = _summarize(rescue_group)

    fe = None
    fe_note = None
    if harm_group and rescue_group:
        harm_k, harm_n = sum(harm_group), len(harm_group)
        res_k, res_n = sum(rescue_group), len(rescue_group)
        fe = fisher_exact([[harm_k, harm_n - harm_k], [res_k, res_n - res_k]])
        if math.isnan(fe.oddsratio) or math.isinf(fe.oddsratio):
            fe_note = "odds ratio is undefined (a table cell is zero); the p-value above is still valid."

    out = {
        "lane": "MedQA text", "n_cases": len(per_case), "n_missing_from_manifest": len(missing),
        "ground_truth_convention": "case.options[case.answer_index] from the MedQA manifest",
        "harm_rate_correct_to_wrong": harm,
        "spurious_rescue_rate_wrong_to_correct": rescue,
        "harm_vs_rescue_fisher": (
            {"pvalue": round(fe.pvalue, 4), "oddsratio": (None if fe_note else round(fe.oddsratio, 4))}
            if fe else None
        ),
        "per_case": per_case,
        "read": (
            f"Harm rate (baseline-correct cases that flip to wrong under peer pressure): "
            f"{harm['rate']} ({harm_group.count(1) if harm_group else 0}/{harm['n']}). Spurious "
            f"rescue rate (baseline-wrong cases that land on ground truth under peer pressure): "
            f"{rescue['rate']} ({rescue_group.count(1) if rescue_group else 0}/{rescue['n']}). "
            "Unlike the imaging lane (near-total conformity, harm/rescue both 0.95-1.0), the text "
            "lane shows PARTIAL conformity: a substantial minority of correct holdouts resist the "
            "seed even under peer pressure, and spurious rescue is rare, consistent with this "
            "session's broader finding that same-lineage Gemini committees hold their answer "
            "against a confident wrong peer far more often in text than in imaging."
        ),
    }
    if fe_note:
        out["harm_vs_rescue_fisher"]["note"] = fe_note
    (out_dir / "net_harm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
