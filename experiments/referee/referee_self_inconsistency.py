"""Referee self-inconsistency floor (#417).

Measures whether identical cache-bypassed temperature-0 private re-queries
produce different answers in the absence of committee influence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.extract import declared_mcq_choice, parse_legacy_string
from experiments.referee.referee_threshold import (
    HOLDOUT,
    _mcq,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lane

_lock = threading.Lock()


class _Cache(_lane.Cache):
    """Draw-aware cache on the shared text-lane dispatch.

    Same key as referee_threshold's cache, sha256(model NUL temperature NUL draw NUL prompt), so the
    committed Gemini cache replays with no calls. Loading, the key check and the paced live call come
    from the shared class; only the key derivation differs, as in temperature_sensitivity._DrawCache.
    """

    def complete(self, model, prompt, temperature=0.0, draw=0):
        k = hashlib.sha256(f"{model}\x00{temperature}\x00{draw}\x00{prompt}".encode()).hexdigest()
        with _lane._lock:
            if k in self.store:
                return self.store[k]
        if not self.key:
            raise SystemExit(f"Cache miss and no {_lane.key_name(model)} set for {model} "
                             "(a fully cached run needs no key).")
        resp = _lane.paced_complete(model, self.key, prompt, decoding={"temperature": temperature})
        if resp is None:
            raise SystemExit(f"{model} returned an empty completion (content=None).")
        with _lane._lock:
            self.store[k] = resp
            self.calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "model": model, "temperature": temperature, "resp": resp}) + "\n")



def build_row(case_id, answer_1, answer_2, declared_1, declared_2):
    return {
        "case_id": case_id,
        "answer_1": answer_1,
        "answer_2": answer_2,
        "declared_1": declared_1,
        "declared_2": declared_2,
        "temp0_flip": answer_1 != answer_2,
    }


def run_one(case, cache, model=HOLDOUT):
    opts = list(case.options)
    prompt, _ = _mcq(case)

    raw_1 = cache.complete(
        model, prompt, temperature=0.0, draw=1
    )
    raw_2 = cache.complete(
        model, prompt, temperature=0.0, draw=2
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
    _lane.add_model_arg(ap, default=HOLDOUT)
    ap.add_argument(
        "--cache",
        default=None,
        help="Defaults to the committed cache for the default model, and to a model-scoped sibling otherwise.",
    )
    ap.add_argument(
        "--out",
        default="experiments/referee/results",
    )
    ap.add_argument("--n", type=int, default=40)

    args = ap.parse_args()
    model = args.model
    out, cache_path = _lane.scoped(
        model, args.out, "experiments/referee/results/referee_self_inconsistency_cache.jsonl", args.cache
    )

    cache = _Cache(cache_path, _lane.key_for(model), model)

    rows = [
        run_one(case, cache, model)
        for case in load_cases(args.manifest)[:args.n]
    ]

    summary = summarize(rows)
    if model != HOLDOUT:
        # The default summary stays byte-identical to the committed one, which predates this flag.
        summary["model"] = model
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
