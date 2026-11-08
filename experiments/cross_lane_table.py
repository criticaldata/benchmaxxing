"""Gemini vs a second lineage on any lane whose results are model-scoped summaries.

Walks every ``<arm>_summary.json`` in ``experiments/<lane>/results/<model_slug>/``, pairs it with the
committed Gemini file of the same name one directory up, and reports every numeric leaf the two share.
Writes the rows as ``cross_lineage_summary.json`` beside the model-scoped results, which is the
committed format for this repo, and prints the Markdown table a PR body or a paper quotes. No API
calls: it reads committed files only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SKIP = {"read", "model", "new_api_calls_this_run", "manifest", "cache", "seed"}


def leaves(node, prefix=""):
    """Yield (dotted_key, value) for every numeric leaf, skipping bookkeeping fields."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP:
                continue
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield prefix, node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, help="e.g. medmcqa or support2")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    slug = a.model.replace("/", "_")
    G = Path("experiments") / a.lane / "results"
    N = G / slug
    rows = []
    for np_ in sorted(N.glob("*_summary.json")):
        gp = G / np_.name
        if not gp.exists():
            continue
        arm = np_.name.replace("_summary.json", "")
        n = dict(leaves(json.load(open(np_))))
        g = dict(leaves(json.load(open(gp))))
        for k in sorted(set(n) & set(g)):
            if k.endswith(("gain", "lose", "n_paired", "n")) and not k.endswith("_rate"):
                continue
            rows.append({"arm": arm, "metric": k, "gemini": g[k], "model": n[k]})

    out = Path(a.out) if a.out else N / "cross_lineage_summary.json"
    with open(out, "w") as fh:
        json.dump({"baseline": "gemini-2.5-flash-lite", "model": a.model, "lane": a.lane, "rows": rows}, fh, indent=1)
        fh.write("\n")

    print(f"| Arm | Metric | gemini | {a.model} |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| `{r['arm']}` | {r['metric']} | {r['gemini']} | {r['model']} |")


if __name__ == "__main__":
    main()
