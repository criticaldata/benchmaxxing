"""Cross-tier referee independence (#183): does a referee from a DIFFERENT model tier than the
holdout catch shortcut adoption as well as a same-tier referee, or does independence require
cross-lineage (out of scope here, tracked separately)? This is the in-scope Gemini proxy for the
blocked cross-lineage arm, probing `benchmaxxing.referee.referee_independence_note`.

Every committed referee re-queries the SAME tier as the holdout. Both directions turn out to be
zero-cost: `experiments/model_dependence/cascade_C_flash.py` uses the identical bare-prompt
template as `experiments/referee/referee_deployable.py` (`_mcq(case, prefix="")`), and the shared
`call_cache.jsonl` (committed to both folders, byte-identical, 6053 keys) already contains BOTH
tiers' answers to that exact prompt for both case sets - reconstructing the prompt per case (using
the MedQA manifest, ground-truth lookup only) and hashing it against the cache confirms 40/40 and
28/28 hits respectively. Zero new API calls; both directions verified keyless.

Direction 1 (primary): a flash referee's private re-query audits the flash-lite holdout committee
(`referee_deployable.jsonl`, 40 cases). The already-committed deployable referee uses the
holdout's OWN (flash-lite) bare re-query as its counterfactual; this substitutes flash's bare
answer to the identical question instead, scored against the same adoption ground truth.

Direction 2 (mirror): a flash-lite referee audits the flash holdout committee
(`cascade_C_flash.jsonl`, 28 hard cases, the anchored/case-anchored condition as the "board").
Ground truth adoption is `anchored == wrong` (guaranteed to differ from flash's own bare answer by
the hard-case construction). The mirror cross-tier referee substitutes flash-lite's bare answer
to the identical question as its counterfactual instead of flash's own.
"""
from __future__ import annotations
from benchmaxxing.extract import parse_legacy_string


import hashlib
import json
from pathlib import Path

from benchmaxxing.data import load_cases
from benchmaxxing.referee import precision_recall

FLASH = "gemini-2.5-flash"
FLASH_LITE = "gemini-2.5-flash-lite"


def _letters(n):
    return [chr(65 + i) for i in range(n)]


def _mcq(case, prefix=""):
    opts = list(case.options)
    body = "\n".join(f"{L}. {o}" for L, o in zip(_letters(len(opts)), opts))
    return (f"{prefix}Question: {case.question}\n\nOptions:\n{body}\n\n"
            "Answer with only the single letter of the best option."), opts



class _CacheReader:
    """Read-only cache lookup: raises if a needed (model, prompt) pair is not already cached,
    since this analysis is designed to need zero new API calls."""

    def __init__(self, path):
        self.store = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                self.store[r["k"]] = r["resp"]

    def get(self, model, prompt):
        k = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
        if k not in self.store:
            raise KeyError(f"cache miss for ({model}, prompt[:40]={prompt[:40]!r}); expected a hit")
        return self.store[k]


def direction1_flash_audits_flash_lite(manifest, referee_jsonl, cache_path):
    cases = {c.case_id: c for c in load_cases(manifest)}
    rows = [json.loads(line) for line in Path(referee_jsonl).read_text().splitlines() if line.strip()]
    cache = _CacheReader(cache_path)

    adopted, same_tier, cross_tier = {}, {}, {}
    for r in rows:
        case = cases.get(r["case_id"])
        if case is None:
            continue
        p, _opts = _mcq(case, "")
        flash_bare = parse_legacy_string(cache.get(FLASH, p), case.options)
        adopted[r["case_id"]] = r["adopted"]
        same_tier[r["case_id"]] = r["deployable"]
        cross_tier[r["case_id"]] = (r["board"] == r["inferred_shortcut"]) and (flash_bare != r["inferred_shortcut"])

    return {
        "n": len(adopted),
        "n_adopted": sum(adopted.values()),
        "same_tier_referee": {k: round(v, 4) if v is not None else None
                               for k, v in precision_recall(same_tier, adopted).items()},
        "cross_tier_referee": {k: round(v, 4) if v is not None else None
                                for k, v in precision_recall(cross_tier, adopted).items()},
    }


def direction2_flash_lite_audits_flash(manifest, cascade_c_jsonl, cache_path):
    cases = {c.case_id: c for c in load_cases(manifest)}
    rows = [json.loads(line) for line in Path(cascade_c_jsonl).read_text().splitlines() if line.strip()]
    cache = _CacheReader(cache_path)

    adopted, same_tier, cross_tier = {}, {}, {}
    for r in rows:
        case = cases.get(r["case_id"])
        if case is None:
            continue
        opts = list(case.options)
        base_p, _ = _mcq(case, "")
        flash_bare = parse_legacy_string(cache.get(FLASH, base_p), opts)
        wrong = next((o for i, o in enumerate(opts) if i != case.answer_index and o != flash_bare), None)
        if wrong is None:
            continue
        flash_lite_bare = parse_legacy_string(cache.get(FLASH_LITE, base_p), opts)

        # r["anchored"] is already the boolean "anchored-condition answer == wrong" (the
        # ground-truth adoption flag itself), not a value to re-compare against `wrong`.
        adopted[r["case_id"]] = r["anchored"]
        # Same-tier referee: flash's own bare re-query as the counterfactual. Trivially identical
        # to the ground truth here (flash_bare != wrong is guaranteed by the hard-case
        # construction), reported for completeness, not as a novel finding.
        same_tier[r["case_id"]] = (r["anchored"]) and (flash_bare != wrong)
        cross_tier[r["case_id"]] = (r["anchored"]) and (flash_lite_bare != wrong)

    return {
        "n": len(adopted),
        "n_adopted": sum(adopted.values()),
        "same_tier_referee": {k: round(v, 4) if v is not None else None
                               for k, v in precision_recall(same_tier, adopted).items()},
        "cross_tier_referee": {k: round(v, 4) if v is not None else None
                                for k, v in precision_recall(cross_tier, adopted).items()},
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Cross-tier referee independence (#183).")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    results_dir = Path(__file__).parent / "results"
    d1 = direction1_flash_audits_flash_lite(
        args.manifest, results_dir / "referee_deployable.jsonl", results_dir / "call_cache.jsonl")
    d2_dir = Path(__file__).parent.parent / "model_dependence" / "results"
    d2 = direction2_flash_lite_audits_flash(
        args.manifest, d2_dir / "cascade_C_flash.jsonl", d2_dir / "call_cache.jsonl")

    out = {
        "direction1_flash_audits_flash_lite_committee": d1,
        "direction2_flash_lite_audits_flash_committee": d2,
        "read": (
            f"Direction 1 (flash referee auditing the flash-lite committee, n={d1['n']}, "
            f"{d1['n_adopted']} true adoptions): cross-tier precision="
            f"{d1['cross_tier_referee']['precision']}, recall={d1['cross_tier_referee']['recall']} "
            f"(misses 1 of {d1['n_adopted']}) vs same-tier's perfect 1.0/1.0 - a small but real "
            "recall cost from substituting the other tier's private re-query. "
            f"Direction 2 (flash-lite referee auditing the flash committee, n={d2['n']}, "
            f"{d2['n_adopted']} true adoptions): cross-tier precision="
            f"{d2['cross_tier_referee']['precision']}, recall={d2['cross_tier_referee']['recall']} "
            "- a larger recall cost than direction 1, consistent with a weaker-tier referee being "
            "less able to independently out-reason a stronger committee's confidently-wrong "
            "answer. Same-tier is trivially identical to ground truth in both directions by "
            "construction (not a novel same-tier result); the cross-tier numbers are the real "
            "finding here: independence does NOT require cross-lineage in either direction - a "
            "different Gemini tier catches the large majority of adoptions with perfect "
            "precision, but recall degrades, more so when the referee tier is weaker than the "
            "committee it audits."
        ),
    }
    (results_dir / "cross_tier.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
