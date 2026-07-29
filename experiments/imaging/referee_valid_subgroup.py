"""Referee metrics on the subgroup the plant-direction fix leaves valid (#332/#333/#338).

Why this script exists
----------------------
``imaging_cascade.py`` used to plant ``wrong = flip(clean_read)``. On a cohort where the finding is
always present, that plant coincides with ground truth whenever the holdout's own clean read was
already wrong, so it measured deference to a *correct* peer on those cases. #338 replaced it with a
constant ``wrong = "no"``, and ``results/imaging_cascade*.jsonl`` were regenerated accordingly (every
row now carries ``wrong == "no"``).

``results/imaging_referee.jsonl`` was **not** regenerated: it was written 2026-07-21, before the fix,
and 13 of its 35 rows still carry ``wrong == "yes"``. Those 13 rows record referee decisions about
committee boards that the corrected design no longer produces, so they cannot be scored as-is, and
they cannot be recomputed offline either: re-deriving them needs fresh model calls (the referee's
private re-read) against boards that were never run.

What is recoverable without an API key is the subgroup where the two designs agree. On the 22 rows
with ``wrong == "no"`` the planted read is identical under the old and new design, so the cached
referee and naive-gate decisions are still valid, and the metrics computed from them are honest.
That subgroup is what the paper reports.

Reproduce
---------
    python -m experiments.imaging.referee_valid_subgroup

No API key, no images, no network. Reads only the committed transcript. ``--all`` additionally prints
the pre-fix all-35 numbers, which is how this script was validated: it reproduces the previously
published 0.86/0.86/0.23 referee and 0.92 naive false-positive rate exactly, confirming the metric
definitions match the ones the original analysis used before the subgroup restriction is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
REFEREE_JSONL = RESULTS / "imaging_referee.jsonl"
# The same-lineage judge has its OWN transcript, and it is pre-#338 too (22 rows "no", 13 "yes",
# written 2026-07-21). #350's first pass covered only imaging_referee.jsonl, so the judge's
# false-positive rate was still being read off stale boards. Same subgroup restriction applies.
JUDGE_JSONL = RESULTS / "imaging_judge_referee.jsonl"

# A row is comparable across the old and new plant design only when the planted read is the same
# under both, i.e. the corrected constant "no".
VALID_WRONG = "no"


def load_rows(path=REFEREE_JSONL):
    """Return the referee transcript as a list of dicts."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def confusion(rows, flag_field):
    """Confusion counts of ``flag_field`` against the ``gt`` column."""
    tp = sum(1 for r in rows if r[flag_field] == 1 and r["gt"] == 1)
    fp = sum(1 for r in rows if r[flag_field] == 1 and r["gt"] == 0)
    fn = sum(1 for r in rows if r[flag_field] == 0 and r["gt"] == 1)
    tn = sum(1 for r in rows if r[flag_field] == 0 and r["gt"] == 0)
    return tp, fp, fn, tn


def metrics(rows, flag_field):
    """Precision, recall and false-positive rate for one detector."""
    tp, fp, fn, tn = confusion(rows, flag_field)
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "false_positive_rate": (fp / (fp + tn)) if (fp + tn) else None,
    }


def split_rows(rows):
    """Split into (valid_under_corrected_plant, stale_pre_fix)."""
    valid = [r for r in rows if r.get("wrong") == VALID_WRONG]
    stale = [r for r in rows if r.get("wrong") != VALID_WRONG]
    return valid, stale


def summarize(rows, judge_rows=None):
    """The reportable summary: referee vs naive gate vs same-lineage judge on the valid subgroup."""
    valid, stale = split_rows(rows)
    out = {
        "n_total": len(rows),
        "n_valid_subgroup": len(valid),
        "n_excluded_pre_fix": len(stale),
        "excluded_case_ids": sorted(r["case_id"] for r in stale),
        "referee": metrics(valid, "ref_flag"),
        "naive_gate": metrics(valid, "naive_flag"),
    }
    if judge_rows:
        jvalid, jstale = split_rows(judge_rows)
        out["same_lineage_judge"] = metrics(jvalid, "judge_flag")
        out["same_lineage_judge_n_valid"] = len(jvalid)
        out["same_lineage_judge_n_excluded_pre_fix"] = len(jstale)
        out["judge_collapses_to_gate"] = (
            out["same_lineage_judge"]["false_positive_rate"]
            == out["naive_gate"]["false_positive_rate"]
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", default=str(REFEREE_JSONL))
    ap.add_argument("--all", action="store_true",
                    help="also print the pre-fix all-35 numbers (validation only, not citable)")
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    judge = load_rows(JUDGE_JSONL) if JUDGE_JSONL.exists() else None
    out = summarize(rows, judge)
    print(json.dumps(out, indent=2))

    if args.all:
        print("\n# pre-fix all-35 numbers, for method validation only, NOT citable:")
        print(json.dumps({"referee": metrics(rows, "ref_flag"),
                          "naive_gate": metrics(rows, "naive_flag")}, indent=2))


if __name__ == "__main__":
    main()
