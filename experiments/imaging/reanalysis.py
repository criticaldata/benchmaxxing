"""Pure re-analysis of the committed imaging cascade transcripts (#185, #214). No API calls.

#185: quantifies claim 4 (contagion rides on case plausibility, not the injected cue's own solo
potency) with two computations that were never done despite every number being committed:
  (a) the rank (Spearman) correlation of each cue's solo flip-above-noise against its cascade
      contagion, across the four cues (n=4, so purely descriptive);
  (b) the per-case cross-cue adoption overlap: for the 35 cases shared across all four
      per-cue cascade files, build a per-case 4-cue shared_adopt vector and compute pairwise phi
      and Jaccard plus a single Cochran's Q across all four cues. High agreement means the SAME
      cases adopt regardless of cue (case-driven contagion); low agreement means adoption tracks
      the cue (cue-driven).

#214: breaks the pooled imaging cascade contagion out by radiographic finding (exploratory, n=35
so cells are small), with Wilson 95% CIs per finding. No paired test is applied across findings:
each finding involves a different set of cases, so it is not a repeated-measures design and a
Cochran's Q or McNemar comparison across findings would be invalid (not the same subjects under
different conditions). The per-finding intervals speak for themselves.

Reads only committed files under results/; writes new committed summaries. Deterministic, no
key/network dependency.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from benchmaxxing.stats import cochran_q, phi_coefficient, jaccard

CUES = ["cable", "corner_tag", "watermark", "laterality"]


def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cue_file(results_dir, cue):
    suffix = "" if cue == "watermark" else f"_{cue}"
    return Path(results_dir) / f"imaging_cascade{suffix}.jsonl"


def cue_summary_file(results_dir, cue):
    suffix = "" if cue == "watermark" else f"_{cue}"
    return Path(results_dir) / f"imaging_cascade{suffix}_summary.json"


def run_claim4(results_dir):
    solo = json.loads((Path(results_dir) / "imaging_solo_summary.json").read_text())
    solo_flip_above_noise = [solo["cues"][c]["flip_above_noise"] for c in CUES]
    contagion = []
    per_cue_rows = {}
    for c in CUES:
        summary = json.loads(cue_summary_file(results_dir, c).read_text())
        contagion.append(summary["contagion"])
        per_cue_rows[c] = {r["case_id"]: r["shared_adopt"] for r in _load_jsonl(cue_file(results_dir, c))}

    rho, pval = spearmanr(solo_flip_above_noise, contagion)

    shared_case_ids = sorted(set.intersection(*(set(per_cue_rows[c]) for c in CUES)))
    matrix = np.array([[per_cue_rows[c][cid] for c in CUES] for cid in shared_case_ids], dtype=float)
    q = cochran_q(matrix)
    pairwise = {}
    for i, ci in enumerate(CUES):
        for j, cj in enumerate(CUES):
            if j <= i:
                continue
            a = matrix[:, i]
            b = matrix[:, j]
            entry = {"phi": round(phi_coefficient(a, b), 4), "jaccard": round(jaccard(a, b), 4)}
            if len(set(a)) == 1 or len(set(b)) == 1:
                entry["note"] = ("phi is mathematically undefined when one cue's adoption is "
                                  "constant across all cases (no variance); reported as 0.0 by "
                                  "convention, not evidence of no relationship. Jaccard remains "
                                  "meaningful (overlap of the adopted cases).")
            pairwise[f"{ci}_vs_{cj}"] = entry

    return {
        "n_cues": len(CUES),
        "cues": CUES,
        "solo_flip_above_noise": solo_flip_above_noise,
        "cascade_contagion": contagion,
        "spearman_solo_vs_contagion": {"rho": round(float(rho), 4), "pvalue": round(float(pval), 4)},
        "n_shared_cases": len(shared_case_ids),
        "cross_cue_cochran_q": {"statistic": round(q.statistic, 4), "pvalue": round(q.pvalue, 4)},
        "pairwise_agreement": pairwise,
        "read": (
            "The Spearman correlation between a cue's own solo potency and its cascade contagion "
            "is the descriptive test of whether contagion tracks artifact strength (n=4, so "
            "indicative only, not a hypothesis test with real power). The cross-cue agreement "
            "(phi/Jaccard/Cochran's Q) is the sharper test: high agreement across cues on the SAME "
            "35 cases means the same cases are what cascade, regardless of which cue is present, "
            "supporting a case-driven (not cue-driven) account of contagion."
        ),
    }


def run_finding_subgroup(results_dir):
    rows = _load_jsonl(cue_file(results_dir, "watermark"))
    by_finding = defaultdict(list)
    for r in rows:
        by_finding[r["finding"]].append(r["shared_adopt"])
    per_finding = {}
    for finding, vals in sorted(by_finding.items(), key=lambda kv: -len(kv[1])):
        k, n = sum(vals), len(vals)
        lo, hi = _wilson(k, n)
        per_finding[finding] = {"n": n, "shared_adopt_rate": round(k / n, 4), "wilson95": [lo, hi]}

    return {
        "n": len(rows),
        "cue": "watermark",
        "per_finding": per_finding,
        "read": (
            "Exploratory only (n=35, several findings with fewer than 5 cases). No paired test is "
            "applied across findings, since each finding involves a different set of cases, not "
            "the same subjects under different conditions. No finding stands out as categorically "
            "immune to or uniquely susceptible to the cascade at this sample size; per-finding "
            "Wilson intervals are wide and overlapping."
        ),
    }


def main():
    results_dir = Path(__file__).parent / "results"
    claim4 = run_claim4(results_dir)
    finding_sub = run_finding_subgroup(results_dir)
    (results_dir / "claim4_quantification.json").write_text(json.dumps(claim4, indent=2))
    (results_dir / "finding_subgroup.json").write_text(json.dumps(finding_sub, indent=2))
    print(json.dumps({"claim4_quantification": claim4, "finding_subgroup": finding_sub}, indent=2))


if __name__ == "__main__":
    main()
