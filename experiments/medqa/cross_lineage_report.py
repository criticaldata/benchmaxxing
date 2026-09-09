"""Cross-lineage report for a second text-lane model against the committed Gemini arms.

Two claims in a lineage PR body are not per-arm numbers and so cannot be read off any single summary:
the multiplicity correction across the whole family of contrasts, and the repeat-prompt caveat about
temperature-0 reproducibility. This script is the definition of both, so a fresh clone can recompute
what the body says with no API calls.

Family: every ``pvalue`` reported anywhere in the arm summaries of the experiments/medqa lane, per
lineage, corrected with Benjamini-Hochberg at 0.05 within the lineage. The family is whatever the
runners report, not a hand-picked subset.

Repeat prompts: the caches key on sha256(model, prompt), so a prompt sent by more than one arm is a
repeat measurement of the same model on the same input, already paid for. A repeat "disagrees" when
the stored completions differ; it "changes the answer" when both completions are a bare option letter
and the letters differ, which excludes prose rewordings of the same choice.

Usage:
    python experiments/medqa/cross_lineage_report.py --model nvidia/nemotron-3-super-120b-a12b
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

from statsmodels.stats.multitest import multipletests

GEMINI_DIR = "experiments/medqa/results"
OTHER_LINEAGE_PREFIXES = ("nvidia_", "openai_", "Qwen_", "meta_", "deepseek")
_LETTER = re.compile(r"^\s*\**\(?([A-E])\)?\**[.:]?\s*$")


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


def family(summary_dir: str, arms):
    rows = []
    for arm in arms:
        p = os.path.join(summary_dir, f"{arm}_summary.json")
        if not os.path.exists(p):
            continue
        for path, pv in _pvalues(json.load(open(p))):
            rows.append({"arm": arm, "contrast": path, "p_raw": pv})
    if rows:
        rej, padj, _, _ = multipletests([r["p_raw"] for r in rows], alpha=0.05, method="fdr_bh")
        for r, q, ok in zip(rows, padj, rej):
            r["q_bh"], r["survives"] = float(q), bool(ok)
    return rows


def _completion(row):
    return row.get("resp") if "resp" in row else row.get("content")


def repeats(cache_files):
    """Per cache key: the set of stored completions and the arms that sent it."""
    by_key = collections.defaultdict(set)
    arms_of = collections.defaultdict(set)
    for f in cache_files:
        arm = os.path.basename(f)
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("temperature") not in (None, 0, 0.0):
                continue  # sampled draws are not repeat measurements at temperature 0
            c = _completion(r)
            if c is None:
                continue
            by_key[r["k"]].add(c)
            arms_of[r["k"]].add(arm)
    rep = {k: v for k, v in by_key.items() if len(arms_of[k]) > 1}
    disagree = {k: v for k, v in rep.items() if len(v) > 1}
    letter_only = {k: v for k, v in rep.items() if all(_LETTER.match(x) for x in v)}
    letter_changed = {k: v for k, v in letter_only.items() if len({_LETTER.match(x).group(1) for x in v}) > 1}
    return {
        "repeated_prompts": len(rep),
        "disagree_any_text": len(disagree),
        "repeated_prompts_letter_only": len(letter_only),
        "answer_changed": len(letter_changed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()
    slug = args.model.replace("/", "_")
    other_dir = os.path.join(GEMINI_DIR, slug)
    arms = sorted(os.path.basename(s)[: -len("_summary.json")] for s in glob.glob(f"{other_dir}/*_summary.json"))

    report = {"model": args.model, "arms": arms, "family": {}, "repeats": {}}
    for name, d in (("gemini", GEMINI_DIR), (args.model, other_dir)):
        fam = family(d, arms)
        dropped = [r["q_bh"] for r in fam if not r["survives"]]
        report["family"][name] = {
            "n_contrasts": len(fam),
            "n_survive_bh_0.05": sum(r["survives"] for r in fam),
            "smallest_dropped_q": min(dropped) if dropped else None,
            "contrasts": fam,
        }
    gem_caches = [f for f in glob.glob(f"{GEMINI_DIR}/*cache*.jsonl")
                  if not os.path.basename(f).startswith(OTHER_LINEAGE_PREFIXES)]
    other_caches = glob.glob(f"{GEMINI_DIR}/{slug}_*cache*.jsonl")
    report["repeats"]["gemini"] = repeats(gem_caches)
    report["repeats"][args.model] = repeats(other_caches)

    for name in ("gemini", args.model):
        f = report["family"][name]
        print(f"{name}: {f['n_survive_bh_0.05']}/{f['n_contrasts']} contrasts survive BH 0.05"
              f"; smallest dropped q = {f['smallest_dropped_q']}")
    for name in ("gemini", args.model):
        r = report["repeats"][name]
        print(f"{name}: {r['repeated_prompts']} prompts sent by more than one arm, "
              f"{r['disagree_any_text']} with differing completions; of the {r['repeated_prompts_letter_only']} "
              f"where every completion is a bare letter, {r['answer_changed']} changed the letter")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
