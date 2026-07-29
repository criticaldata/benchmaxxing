"""Rebuild a push_c.py-compatible ``solo_records.jsonl`` from an already-cached solo run.

``experiments/medqa/push_c.py`` needs a per-case ``clean_correct`` record (to find "hard" cases:
where the holdout gets the clean question wrong) via ``--solo-records``. ``reproduce.py``'s
``run_solo`` only persists the aggregate ``solo_results.json``, not per-case rows, so this
recomputes the clean-condition correctness for ``push_c.py``'s HOLDOUT model directly from the
already-committed call cache (no new API calls).

Usage
-----
    python -m experiments.mimic_cxr_text.build_solo_records \
        --manifest experiments/mimic_cxr_text/mimic_cxr_text_manifest_n600.csv \
        --cache experiments/mimic_cxr_text/results_n600/call_cache.jsonl \
        --out experiments/mimic_cxr_text/results/solo_records.jsonl \
        --model gemini-2.5-flash-lite --n 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.data import load_cases
from benchmaxxing.schema import Condition
from experiments.medqa.reproduce import TEXT_CUES, _mcq_prompt, _parse_choice


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--n", type=int, default=None,
                     help="sample size N (must match the original reproduce.py --solo-n run)")
    ap.add_argument("--seed", type=int, default=0,
                     help="must match the original reproduce.py --seed so the same sample is reproduced")
    args = ap.parse_args()

    cache = {}
    for line in Path(args.cache).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["k"]] = row["resp"]

    all_cases = load_cases(args.manifest)
    cases = random.Random(args.seed).sample(all_cases, min(args.n, len(all_cases))) if args.n else all_cases

    rows = []
    misses = 0
    for case in cases:
        tp = build_text_twin(case, TEXT_CUES[0])
        payload = tp.payload(Condition.CLEAN)
        prompt = _mcq_prompt(payload)
        key = _cache_key(args.model, prompt)
        if key not in cache:
            misses += 1
            continue
        answer = _parse_choice(cache[key], list(payload["options"]))
        gt = case.options[case.answer_index]
        rows.append({"case_id": case.case_id, "clean_correct": answer == gt})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_hard = sum(1 for r in rows if r["clean_correct"] is False)
    print(f"wrote {len(rows)} records ({n_hard} hard / clean_correct=False), {misses} cache misses, to {out_path}")


if __name__ == "__main__":
    main()
