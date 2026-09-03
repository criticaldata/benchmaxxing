"""Referee self-inconsistency floor (#417).

Measures whether identical cache-bypassed temperature-0 private re-queries
produce different answers in the absence of committee influence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.extract import parse_legacy_string, declared_mcq_choice
from experiments.referee.referee_threshold import (
    _Cache,
    _key,
    _mcq,
    HOLDOUT,
)



def build_row(case_id, answer_1, answer_2, declared_1, declared_2):
    return {
        "case_id": case_id,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "declared_1": declared_1,
        "declared_2": declared_2,
        "temp0_flip": answer_1 != answer_2,
    }


def run_one(case, cache):
    opts = list(case.options)
    prompt, _ = _mcq(case)

    raw_1 = cache.complete(
        HOLDOUT, prompt, temperature=0.0, draw=1
    )
    raw_2 = cache.complete(
        HOLDOUT, prompt, temperature=0.0, draw=2
    )

    answer_1 = parse_legacy_string(raw_1, opts)
    answer_2 = parse_legacy_string(raw_2, opts)

    _, declared_1 = declared_mcq_choice(raw_1, opts)
    _, declared_2 = declared_mcq_choice(raw_2, opts)

    return build_row(
        case.case_id,
        answer_1,
        answer_2,
        declared_1,
        declared_2,
    )


def summarize(rows):
    n = len(rows)

    declared_pairs = sum(
        1
        for r in rows
        if r["declared_1"] and r["declared_2"]
    )

    undeclared_pairs = sum(
        1
        for r in rows
        if not (r["declared_1"] and r["declared_2"])
    )

    undeclared_draws = sum(
        int(not r["declared_1"]) + int(not r["declared_2"])
        for r in rows
    )

    unstable = sum(
        1
        for r in rows
        if r["declared_1"]
        and r["declared_2"]
        and r["temp0_flip"]
    )

    stable = declared_pairs - unstable

    return {
        "n": n,
        "temperature": 0,
        "declared_pairs": declared_pairs,
        "undeclared_pairs": undeclared_pairs,
        "undeclared_draws": undeclared_draws,
        "stable_cases": stable,
        "unstable_cases": unstable,
        "temp0_self_inconsistency_rate": (
            unstable / declared_pairs
            if declared_pairs
            else None
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Referee self-inconsistency floor (#417)."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--cache",
        default="experiments/referee/results/referee_self_inconsistency_cache.jsonl",
    )
    ap.add_argument(
        "--out",
        default="experiments/referee/results",
    )
    ap.add_argument("--n", type=int, default=40)

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = _Cache(args.cache, _key())

    rows = [
        run_one(case, cache)
        for case in load_cases(args.manifest)[:args.n]
    ]

    summary = summarize(rows)
    summary["new_api_calls_this_run"] = cache.calls

    (out / "referee_self_inconsistency.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    (out / "referee_self_inconsistency_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
