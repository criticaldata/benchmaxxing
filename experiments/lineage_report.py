"""Lineage report: the body claims of a second-model PR that are not per-arm numbers.

Some claims in a lineage PR body cannot be read off any single arm summary, because they are
statements ABOUT the set of arms: the unseeded-accuracy invariant across arms, the repeat-prompt
caveat behind "temperature 0 is reproducible", the coverage count, the row-count differences that
are eligibility rules rather than different cohorts, and the multiplicity correction over the whole
family of reported p-values. This script is the definition of each, so a fresh clone recomputes
what the body says with no API calls. It spans both modalities of this branch.

Definitions, so the numbers are checkable rather than asserted:

* Unseeded accuracy: an arm records it when its per-case rows carry both ``bare`` (the model's
  answer with no peer and no cue) and ``ground_truth``. Correct means the two strings are equal.
  The invariant the body states is that this count is identical in every arm that runs the full
  cohort; arms with a smaller cohort have their own eligibility rule and are listed separately.
* Repeat prompts: caches key on sha256(model, prompt), so one key stored in more than one arm's
  cache is a repeat measurement of the same model on the same input. A repeat "disagrees" when the
  stored completions differ; it "changes the answer" when both completions are a bare option letter
  and the letters differ, which excludes prose rewordings of the same choice.
* p-value family: every ``pvalue`` reported anywhere in this model's arm summaries, per lane,
  corrected with Benjamini-Hochberg at 0.05. The family is whatever the runners report, not a
  hand-picked subset.

Two lanes are deliberately NOT checkable here and the report says so rather than skipping them:
the MIMIC-CXR report-text and film lanes read data under the PhysioNet DUA, so neither their call
caches nor their cohort manifests are committed. Their cohort-identity claims (ground-truth
agreement at the same case_index, the solo-600 reproduction, the per-film sha256 match) can only be
rechecked on a host that holds the data.

Usage:
    python experiments/lineage_report.py --model Qwen/Qwen2.5-VL-72B-Instruct
    python experiments/lineage_report.py --model Qwen/Qwen2.5-VL-72B-Instruct --check
    python experiments/lineage_report.py --model Qwen/Qwen2.5-VL-72B-Instruct --out report.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
_LETTER = re.compile(r"^\s*\(?([A-E])\)?\s*[.):]?\s*$")

# The text lanes whose arms this branch adds, and the imaging lanes. MIMIC is handled separately:
# its caches and manifests are not committed, so it cannot be recomputed from a clone.
TEXT_LANES = ("medqa", "medmcqa", "referee", "support2", "cascade", "contamination",
              "model_dependence", "cross_dataset", "blind_metric")
IMAGING_LANES = ("imaging", "chexpert", "imaging_chexpert")
DUA_LANES = ("mimic_cxr_text", "mimic_cxr_image")


def slug_of(model: str) -> str:
    return model.replace("/", "_")


def _results_dirs(lane: str, slug: str):
    """Every model-scoped results directory of a lane, including per-cohort subdirectories."""
    base = os.path.join(HERE, lane, "results")
    hits = [base] if os.path.isdir(os.path.join(base, slug)) else []
    hits += [os.path.dirname(p) for p in glob.glob(os.path.join(base, "*", slug))]
    return [os.path.join(d, slug) for d in dict.fromkeys(hits)]


def unseeded(lane: str, slug: str):
    """Per arm: the unseeded correct count, its cohort size, and the file it came from."""
    rows = []
    for d in _results_dirs(lane, slug):
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            name = os.path.basename(p)
            if "cache" in name or "solo_records" in name:
                continue
            recs = [json.loads(l) for l in open(p) if l.strip()]
            if not recs or not {"bare", "ground_truth"} <= set(recs[0]):
                continue
            correct = sum(1 for r in recs if r.get("bare") == r.get("ground_truth"))
            rows.append({"arm": name[:-6], "n": len(recs), "unseeded_correct": correct,
                         "path": os.path.relpath(p, HERE)})
    return rows


def _completion(row):
    return row.get("resp") if "resp" in row else row.get("content")


def repeats(cache_files):
    """Per cache key: the stored completions and which arm caches hold it."""
    by_key, arms_of = collections.defaultdict(set), collections.defaultdict(set)
    for f in cache_files:
        arm = os.path.basename(f)
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("temperature") not in (None, 0, 0.0):
                continue  # a sampled draw is not a repeat measurement at temperature 0
            c = _completion(r)
            key = r.get("k", r.get("key"))
            if isinstance(key, list):
                key = tuple(key)  # one lane stores the key as its parts rather than a digest
            if c is None or key is None:
                continue
            by_key[key].add(c)
            arms_of[key].add(arm)
    rep = {k: v for k, v in by_key.items() if len(arms_of[k]) > 1}
    disagree = {k: v for k, v in rep.items() if len(v) > 1}
    letter_only = {k: v for k, v in rep.items() if all(_LETTER.match(x) for x in v)}
    changed = {k: v for k, v in letter_only.items()
               if len({_LETTER.match(x).group(1) for x in v}) > 1}
    return {"repeated_prompts": len(rep), "disagree_any_text": len(disagree),
            "repeated_prompts_letter_only": len(letter_only), "answer_changed": len(changed)}


def _pvalues(node, path=""):
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "pvalue" and isinstance(v, (int, float)):
                out.append((path, float(v)))
            else:
                out += _pvalues(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _pvalues(v, f"{path}[{i}]")
    return out


def family(lane: str, slug: str):
    rows = []
    for d in _results_dirs(lane, slug):
        for p in sorted(glob.glob(os.path.join(d, "*_summary.json"))):
            for path, pv in _pvalues(json.load(open(p))):
                rows.append({"arm": os.path.basename(p)[:-13], "contrast": path, "p_raw": pv})
    if rows:
        rej, padj, _, _ = multipletests([r["p_raw"] for r in rows], alpha=0.05, method="fdr_bh")
        for r, q, ok in zip(rows, padj, rej):
            r["q_bh"], r["survives"] = float(q), bool(ok)
    return rows


def coverage(slug: str):
    """Summaries per lane for this model, and the arms whose row count differs from the baseline."""
    per_lane, diffs = {}, []
    for lane in TEXT_LANES + IMAGING_LANES + DUA_LANES:
        dirs = _results_dirs(lane, slug)
        per_lane[lane] = sum(len(glob.glob(os.path.join(d, "*_summary.json"))) for d in dirs)
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
                if "cache" in os.path.basename(p):
                    continue
                base = os.path.join(os.path.dirname(os.path.dirname(p)), os.path.basename(p))
                if not os.path.exists(base):
                    continue
                n_model = sum(1 for l in open(p) if l.strip())
                n_base = sum(1 for l in open(base) if l.strip())
                if n_model != n_base:
                    diffs.append({"lane": lane, "arm": os.path.basename(p)[:-6],
                                  "baseline_rows": n_base, "model_rows": n_model})
    return per_lane, diffs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    ap.add_argument("--check", action="store_true",
                    help="assert the invariants the PR body states, exit non-zero if one fails")
    args = ap.parse_args(argv)
    slug = slug_of(args.model)

    rep = {"model": args.model, "unseeded": {}, "repeats": {}, "bh_family": {}}
    for lane in TEXT_LANES + IMAGING_LANES:
        arms = unseeded(lane, slug)
        if arms:
            rep["unseeded"][lane] = arms
        caches = glob.glob(os.path.join(HERE, lane, "results", f"*{slug}*cache*.jsonl"))
        if caches:
            rep["repeats"][lane] = repeats(caches)
        fam = family(lane, slug)
        if fam:
            rep["bh_family"][lane] = {
                "n_contrasts": len(fam), "n_survive_bh_0.05": sum(1 for r in fam if r["survives"]),
                "max_p_surviving": max([r["p_raw"] for r in fam if r["survives"]], default=None),
                "contrasts": fam}
    rep["coverage_summaries_per_lane"], rep["row_count_differences"] = coverage(slug)
    rep["not_checkable_from_a_clone"] = {
        "lanes": list(DUA_LANES),
        "why": ("PhysioNet DUA: neither the call caches nor the cohort manifests are committed, so "
                "the cohort-identity claims for these two lanes can only be rechecked on a host "
                "that holds the data"),
        "claims": ["ground-truth agreement at the same case_index on the 633-case index",
                   "the solo-600 cohort reproduced from the recorded seed and prefix window",
                   "per-film sha256 match against the committed deid/provenance.csv"]}

    medqa = rep["unseeded"].get("medqa", [])
    cohort = collections.Counter(a["n"] for a in medqa)
    main_n = cohort.most_common(1)[0][0] if cohort else None
    at_full = [a for a in medqa if a["n"] == main_n]
    scores = sorted({a["unseeded_correct"] for a in at_full})
    rep["medqa_unseeded_invariant"] = {
        "cohort": main_n, "arms_at_full_cohort": len(at_full), "distinct_unseeded_scores": scores,
        "arms_below_full_cohort": sorted((a["arm"], a["n"]) for a in medqa if a["n"] != main_n)}

    inv = rep["medqa_unseeded_invariant"]
    print(f"model: {args.model}")
    print(f"medqa unseeded: {inv['arms_at_full_cohort']} arms at n={inv['cohort']} record it, "
          f"scores {inv['distinct_unseeded_scores']}; {len(inv['arms_below_full_cohort'])} arms on "
          f"a smaller cohort by their own eligibility rule")
    for lane, r in rep["repeats"].items():
        print(f"{lane} repeats: {r['repeated_prompts']} prompts stored by more than one arm, "
              f"{r['answer_changed']} changed the answer letter, {r['disagree_any_text']} differ "
              f"in any text")
    for lane, f in rep["bh_family"].items():
        print(f"{lane} p-value family: {f['n_survive_bh_0.05']}/{f['n_contrasts']} survive BH 0.05")
    print(f"row-count differences vs the baseline: {len(rep['row_count_differences'])} arms")
    print(f"not checkable from a clone: {', '.join(DUA_LANES)} (DUA)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"wrote {args.out}")

    if args.check:
        assert len(scores) == 1, f"unseeded accuracy not invariant across full-cohort arms: {scores}"
        assert inv["arms_at_full_cohort"] >= 2, "fewer than two arms record unseeded accuracy"
        for lane, r in rep["repeats"].items():
            assert r["answer_changed"] <= r["repeated_prompts"], lane
        print("check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
