"""Stratify the live-peer rows by what the holdout answered alone (no API calls).

The arm's own summary reports how often the holdout gives the peers' consensus answer. That rate
mixes two things: cases where the holdout would have given that answer anyway, and cases where the
board moved it. This reads the committed rows and separates them, so the figures a write-up quotes
come from a committed file:

  wrong consensus  holdout alone was correct / already gave the peers' wrong answer / gave another
                   wrong answer, and how many in each group end on the peers' answer.
  right consensus  holdout alone was correct / wrong, and how many end on the peers' answer.

    python experiments/medqa/live_peer_strata.py --rows experiments/medqa/results/live_peer_n600/live_peer_organic.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import fisher_exact


def _wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _cell(rows):
    k = sum(r["follows_consensus"] for r in rows)
    return {"n": len(rows), "ends_on_consensus": k}


def strata(rows):
    wrong = [r for r in rows if r["consensus_wrong"]]
    right = [r for r in rows if r["consensus_right"]]
    fw, fr = sum(r["follows_consensus"] for r in wrong), sum(r["follows_consensus"] for r in right)
    return {
        "n": len(rows),
        "peers_disagree": len(rows) - len(wrong) - len(right),
        "wrong_consensus": {
            "n": len(wrong), "ends_on_consensus": fw, "wilson95": _wilson(fw, len(wrong)),
            "holdout_correct_alone": _cell([r for r in wrong if r["bare"] == r["ground_truth"]]),
            "holdout_alone_gave_the_same_wrong_answer": _cell([r for r in wrong if r["bare"] == r["consensus"]]),
            "holdout_alone_gave_another_wrong_answer": _cell(
                [r for r in wrong if r["bare"] != r["ground_truth"] and r["bare"] != r["consensus"]]),
        },
        "right_consensus": {
            "n": len(right), "ends_on_consensus": fr, "wilson95": _wilson(fr, len(right)),
            "holdout_correct_alone": _cell([r for r in right if r["bare"] == r["ground_truth"]]),
            "holdout_wrong_alone": _cell([r for r in right if r["bare"] != r["ground_truth"]]),
        },
        # unrounded: an exact test never returns 0 or 1 unless the table forces it
        "wrong_vs_right_follow_fisher_pvalue": float(
            fisher_exact([[fw, len(wrong) - fw], [fr, len(right) - fr]])[1]),
        "net_harm_cases": sum(r["net_harm"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description="Stratify live-peer rows by the holdout's solo answer.")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default=None, help="defaults to live_peer_strata.json beside the rows")
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    out = Path(args.out) if args.out else Path(args.rows).with_name("live_peer_strata.json")
    out.write_text(json.dumps(strata(rows), indent=2) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
