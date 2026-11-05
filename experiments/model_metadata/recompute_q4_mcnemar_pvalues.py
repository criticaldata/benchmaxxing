"""Replace six-decimal McNemar zeros in committed Q4 summaries with exact values from counts.

This is a cache-only presentation correction: it makes no model/API calls and does not alter rows.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.degeneracy import REPO_ROOT, rounded_pvalues
from benchmaxxing.stats import mcnemar

MODEL_SLUG = "meta_llama-3.2-11b-vision-instruct"


def read_pair(parent: dict) -> tuple[int, int] | None:
    for left, right in (
        ("gain", "lose"),
        ("discordant_gain", "discordant_lose"),
        ("gained", "lost"),
        ("adopted_only_here", "resisted_only_here"),
    ):
        if isinstance(parent.get(left), int) and isinstance(parent.get(right), int):
            return parent[left], parent[right]
    return None


def main() -> None:
    files: dict[Path, dict] = {}
    changed = []
    for finding in rounded_pvalues(REPO_ROOT):
        if MODEL_SLUG not in finding.path or "nvidia_nim" in finding.path:
            continue
        doc = files.setdefault(REPO_ROOT / finding.path, json.loads((REPO_ROOT / finding.path).read_text()))
        trail = finding.locus.split(".")
        parent = doc
        for segment in trail[:-1]:
            parent = parent[int(segment)] if isinstance(parent, list) else parent[segment]
        if parent[trail[-1]] != 0.0:
            continue
        pair = read_pair(parent)
        if pair is None:
            raise SystemExit(f"no paired discordant counts for {finding.key}")
        exact = mcnemar(*pair).pvalue
        if exact <= 0.0:
            raise SystemExit(f"non-positive exact result for {finding.key}: {pair}")
        parent[trail[-1]] = exact
        changed.append((finding.path, finding.locus, pair, exact))

    if not changed:
        print("No six-decimal p-values needed correction.")
        return
    for path, doc in files.items():
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    for path, locus, pair, exact in changed:
        print(f"{path} | {locus} | discordants={pair} | p={exact:.6g}")
    print(f"Updated {len(changed)} exact McNemar p-values; no model calls made.")


if __name__ == "__main__":
    main()
