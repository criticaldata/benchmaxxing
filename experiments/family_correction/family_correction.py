"""Benchmark-wide (family) multiple-comparison correction across all reported p-values (#191).
Pure computation over a curated, cited list of already-committed p-values; no API calls.

Every within-arm correction in this project corrects only the tests inside that one arm. This
applies a single family-wide correction (Benjamini-Hochberg and Holm) across every headline
p-value reported anywhere in the project, to see which arms survive when tested against the full
set rather than in isolation.

SCOPE: only p-values from a source this project has independently verified as current
(post-answer-parser-fix) this session are included. Several older per-case artifacts on unmerged
branches (e.g. break_it_summary.json, push_c_summary.json, cascade_v2_per_case.jsonl on
results/medqa-break-it and results/medqa-experiments) were NOT independently re-verified for
parser-fix currency in this pass and are deliberately excluded rather than risk silently mixing
stale and corrected p-values into one family, which would make the correction itself misleading.
See EXCLUDED below for the full list and reasons.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.stats import multiple_comparison

# Each entry: (label, p-value, source citation). All independently verified this session.
FAMILY = [
    ("scale_c: anchored vs generic (text, plausibility C)", 0.04138946533203125,
     "experiments/medqa/results/scale_c_summary.json (PR #141), n=85"),
    ("scale_c: strongly-anchored vs generic (text)", 0.09625244140625,
     "experiments/medqa/results/scale_c_summary.json (PR #141), n=85"),
    ("clean_a: flash Fisher exact (text, system-flag contamination)", 0.021554169030062395,
     "recomputed here from experiments/medqa/results/clean_a_summary.json (PR #141) counts "
     "(6/23 flag vs 0/23 control); matches the paper's stated 0.022"),
    ("clean_a: flash-lite Fisher exact (text, system-flag contamination)", 0.009160521837776518,
     "recomputed here from clean_a_summary.json (PR #141) counts (7/23 vs 0/23); matches the "
     "paper's stated 0.009"),
    ("cascade_C_flash: model-dependence McNemar (text, flash holdout)", 1.0,
     "experiments/model_dependence/results/cascade_C_flash_summary.json (PR #223)"),
    ("contamination_audit: flash Fisher (text, flip correct-vs-wrong)", 0.0032995170692775274,
     "experiments/contamination/results/contamination_audit_summary.json (PR #224)"),
    ("contamination_audit: flash-lite Fisher (text)", 4.359882314463016e-07,
     "experiments/contamination/results/contamination_audit_summary.json (PR #224)"),
    ("multi_round: round-1-vs-round-K McNemar (text)", 0.625,
     "experiments/cascade/results/multi_round_summary.json (PR #225)"),
    ("multi_round: round-1-vs-round-K McNemar (imaging)", 0.25,
     "experiments/imaging/results/imaging_multi_round_summary.json (PR #166)"),
    ("net_harm: harm-vs-rescue Fisher, cable (imaging)", 1.0,
     "experiments/imaging/results/net_harm.json (PR #226)"),
    ("net_harm: harm-vs-rescue Fisher, corner_tag (imaging)", 1.0,
     "experiments/imaging/results/net_harm.json (PR #226)"),
    ("net_harm: harm-vs-rescue Fisher, watermark (imaging)", 1.0,
     "experiments/imaging/results/net_harm.json (PR #226)"),
    ("net_harm: harm-vs-rescue Fisher, laterality (imaging)", 1.0,
     "experiments/imaging/results/net_harm.json (PR #226)"),
]

EXCLUDED = [
    {"source": "experiments/medqa/results/break_it_summary.json (results/medqa-break-it, PR #140)",
     "reason": "not independently re-verified for parser-fix currency in this pass"},
    {"source": "experiments/medqa/results/push_c_summary.json (results/medqa-break-it, PR #140)",
     "reason": "not independently re-verified for parser-fix currency in this pass"},
    {"source": "experiments/medqa/results/cascade_v2_per_case.jsonl (results/medqa-experiments, PR #135)",
     "reason": "confirmed tainted by the pre-fix stray-article parser bug (see #177's imaging PR "
               "note); its medqa-82 row still shows the exact mis-parse"},
    {"source": "experiments/blind_metric, experiments/referee (main, PRs #157/#154)",
     "reason": "report precision/recall/FPR, not a p-value directly comparable in a p-value family"},
]


def main():
    results_dir = Path(__file__).parent / "results"
    labels = [f[0] for f in FAMILY]
    pvalues = [f[1] for f in FAMILY]
    sources = [f[2] for f in FAMILY]

    bh = multiple_comparison(pvalues, method="bh")
    holm = multiple_comparison(pvalues, method="holm")

    rows = []
    for label, p, src, p_bh, rej_bh, p_holm, rej_holm in zip(
        labels, pvalues, sources, bh.pvalues_adjusted, bh.reject, holm.pvalues_adjusted, holm.reject
    ):
        rows.append({
            "label": label, "p_raw": p, "source": src,
            "p_bh": round(float(p_bh), 6), "survives_bh": bool(rej_bh),
            "p_holm": round(float(p_holm), 6), "survives_holm": bool(rej_holm),
        })

    out = {
        "n_tests_in_family": len(FAMILY),
        "alpha": 0.05,
        "rows": rows,
        "n_survive_bh": sum(r["survives_bh"] for r in rows),
        "n_survive_holm": sum(r["survives_holm"] for r in rows),
        "excluded": EXCLUDED,
        "read": (
            "Against the full family (not per-arm): both contamination-audit Fisher tests "
            "(flash and flash-lite, correct-vs-wrong flip rate) survive both BH and Holm. Of the "
            "clean-A system-flag contamination tests, only flash-lite survives BH (p_bh=0.040); "
            "flash's clean-A (p_bh=0.070) and both scale_c plausibility comparisons (anchored "
            "p_bh=0.108, strongly-anchored p_bh=0.209) do NOT survive a family-wide correction, "
            "consistent with scale_c's anchored increment already failing a local BH within its "
            "own arm per the paper. The null results (model-dependence, both multi-round arms, "
            "all four imaging net-harm comparisons) correctly do not survive either, since they "
            "were never significant to begin with. Only 3 of 13 tests in this family survive BH; "
            "2 of 13 survive the stricter Holm correction."
        ),
    }
    (results_dir / "family_correction.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
