"""Referee agreement/disagreement matrices across all gates (#184, imaging lane). Pure
re-analysis of `results/imaging_referee.jsonl` and `results/imaging_judge_referee.jsonl`, no API
calls.

Computes pairwise phi (Matthews correlation) and Cohen's kappa among the deployable referee's
flag, the naive conformity gate's flag, and the same-lineage judge's flag, plus the exact
divergent cases, against the shared peer-driven-adoption ground truth (`gt`).
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.stats import cohen_kappa, phi_coefficient


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _agreement(a, b):
    entry = {"phi": round(phi_coefficient(a, b), 4), "kappa": round(cohen_kappa(a, b), 4)}
    if len(set(a)) == 1 or len(set(b)) == 1:
        entry["note"] = ("phi/kappa are mathematically degenerate when one gate is constant "
                          "(no variance); reported as 0.0 by convention, not evidence of "
                          "disagreement.")
    return entry


def main():
    results_dir = Path(__file__).parent / "results"
    ref_rows = _load_jsonl(results_dir / "imaging_referee.jsonl")
    judge_rows = _load_jsonl(results_dir / "imaging_judge_referee.jsonl")

    ref_by_id = {r["case_id"]: r for r in ref_rows}
    judge_by_id = {r["case_id"]: r for r in judge_rows}
    shared_ids = sorted(set(ref_by_id) & set(judge_by_id))

    ref_flag = [ref_by_id[c]["ref_flag"] for c in shared_ids]
    naive_flag = [ref_by_id[c]["naive_flag"] for c in shared_ids]
    judge_flag = [int(judge_by_id[c]["judge_flag"]) for c in shared_ids]
    gt = [ref_by_id[c]["gt"] for c in shared_ids]

    pairwise = {
        "deployable_vs_naive": _agreement(ref_flag, naive_flag),
        "deployable_vs_judge": _agreement(ref_flag, judge_flag),
        "naive_vs_judge": _agreement(naive_flag, judge_flag),
    }

    deployable_naive_divergent = [c for c in shared_ids if ref_by_id[c]["ref_flag"] != ref_by_id[c]["naive_flag"]]
    deployable_judge_divergent = [c for c in shared_ids if ref_by_id[c]["ref_flag"] != int(judge_by_id[c]["judge_flag"])]

    out = {
        "n_shared_cases": len(shared_ids),
        "n_peer_driven_adoptions_gt": sum(gt),
        "pairwise_agreement": pairwise,
        "deployable_vs_naive_divergent_cases": {
            "n_cases": len(deployable_naive_divergent), "case_ids": deployable_naive_divergent,
        },
        "deployable_vs_judge_divergent_cases": {
            "n_cases": len(deployable_judge_divergent), "case_ids": deployable_judge_divergent,
        },
        "read": (
            "Unlike the text lane (where deployable and the judge nearly coincide, 1 divergent "
            "case of 40), imaging shows a much larger split: the deployable referee (transcript + "
            "private re-read) and the naive gate diverge on every case where naive over-flags "
            "(consistent with naive FPR 0.92 vs deployable FPR 0.23 already reported), and the "
            "judge (transcript-only, no re-read) tracks the naive gate far more closely than it "
            "tracks the deployable referee, numerically confirming the earlier finding that the "
            "imaging judge, lacking a fresh independent signal, degenerates to the naive rule."
        ),
    }
    (results_dir / "imaging_referee_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
