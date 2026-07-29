"""Refusal-aware re-analysis of a committed MIMIC-CXR text solo run (#336 review).

``experiments/medqa/reproduce.py``'s inline ``_parse_choice`` has no refusal/unparseable
sentinel: any non-answer falls through to returning raw text, which can spuriously differ
between the clean and contaminated call and inflate the flip rate. MedQA vignettes rarely
trigger this; real MIMIC-CXR report text does (Gemini sometimes claims the report wasn't
provided, ~35% of gemini-2.5-flash calls in the committed n=35 run, despite the report being
present in the prompt — a genuine model behavior, not a manifest/prompt bug, confirmed by
inspecting the exact rendered prompts).

This script re-derives flip rate from the already-committed, already-cached call responses
using the proper abstention-aware extractor (``benchmaxxing.extract.parse_mcq_choice``),
without making any new API calls and without modifying ``reproduce.py`` (kept out of scope
per #316's "replicate, don't rewrite" rule; this is a read-only re-analysis of one dataset's
own cache).

Usage
-----
    python -m experiments.mimic_cxr_text.refusal_aware_reanalysis \
        --manifest experiments/mimic_cxr_text/mimic_cxr_text_manifest.csv \
        --cache experiments/mimic_cxr_text/results/call_cache.jsonl \
        --solo-n 35
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.data import load_cases
from benchmaxxing.extract import Abstention, parse_mcq_choice
from benchmaxxing.schema import Condition
from experiments.medqa.reproduce import TEXT_CUES, TIERS, _mcq_prompt


def _load_cache(path) -> dict[str, str | None]:
    store: dict[str, str | None] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            store[row["k"]] = row["resp"]
    return store


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Refusal-aware re-analysis of a committed MIMIC-CXR text solo run.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None, help="defaults to <cache dir>/solo_results_refusal_aware.json")
    ap.add_argument("--solo-n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = _load_cache(args.cache)
    all_cases = load_cases(args.manifest)
    cases = random.Random(args.seed).sample(all_cases, min(args.solo_n, len(all_cases)))

    stats = {m: {"flips": 0, "pairs": 0, "abstentions": 0, "total_calls": 0,
                 "per_cue": {c: {"flips": 0, "pairs": 0, "abstentions": 0} for c in TEXT_CUES}}
              for m in TIERS}

    def lookup(model, payload):
        # The model is prompted (per _mcq_prompt) to answer with a bare letter, not the option
        # text, so parse_mcq_choice must be given the letters themselves to take its
        # letter-declaration branch (its "non-letter options" branch instead searches the reply
        # for the option's own words, which the model was never asked to repeat).
        options = payload["options"]
        letters = [chr(ord("A") + i) for i in range(len(options))]
        prompt = _mcq_prompt(payload)
        key = _cache_key(model, prompt)
        if key not in cache:
            raise SystemExit(f"Cache miss for model={model!r}; this script must not make new API calls. "
                              f"Re-run experiments.medqa.reproduce first to populate the cache.")
        text = cache[key]
        choice = parse_mcq_choice(text or "", letters)
        # Return the chosen option's TEXT, not its positional index: option_order permutes
        # positions between clean/contaminated, so the same index means different content and
        # comparing raw indices would spuriously "flip" on every permutation.
        if isinstance(choice, Abstention):
            return choice
        return options[choice]

    for model in TIERS:
        for case in cases:
            clean_tp = build_text_twin(case, TEXT_CUES[0])
            clean_choice = lookup(model, clean_tp.payload(Condition.CLEAN))
            for cue in TEXT_CUES:
                try:
                    tp = build_text_twin(case, cue)
                except (ValueError, TypeError):
                    continue
                cont_choice = lookup(model, tp.payload(Condition.CONTAMINATED))
                s = stats[model]
                cs = s["per_cue"][cue]
                s["total_calls"] += 1
                is_abst = isinstance(clean_choice, Abstention) or isinstance(cont_choice, Abstention)
                if is_abst:
                    s["abstentions"] += 1
                    cs["abstentions"] += 1
                    continue
                s["pairs"] += 1
                cs["pairs"] += 1
                if clean_choice != cont_choice:
                    s["flips"] += 1
                    cs["flips"] += 1

    result = {}
    for model in TIERS:
        s = stats[model]
        result[model] = {
            "total_calls": s["total_calls"],
            "abstention_rate": round(s["abstentions"] / s["total_calls"], 4) if s["total_calls"] else None,
            "flip_rate_excl_abstentions": round(s["flips"] / s["pairs"], 4) if s["pairs"] else None,
            "n_valid_pairs": s["pairs"],
            "per_cue": {
                cue: {
                    "flip_rate_excl_abstentions": round(cs["flips"] / cs["pairs"], 4) if cs["pairs"] else None,
                    "abstention_rate": round(cs["abstentions"] / (cs["pairs"] + cs["abstentions"]), 4)
                    if (cs["pairs"] + cs["abstentions"]) else None,
                    "n_valid_pairs": cs["pairs"],
                }
                for cue, cs in s["per_cue"].items()
            },
        }

    out_path = Path(args.out) if args.out else Path(args.cache).parent / "solo_results_refusal_aware.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
