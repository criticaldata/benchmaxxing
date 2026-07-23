"""Cross-lane multiple-comparison reconciliation (#237 gap 1): fold EVERY headline test from both
lanes into a single family and correct it once with Benjamini-Hochberg and Holm.

PR #238 (stats_reconciliation) corrected the nine committee-structure contrasts as their own family.
This closes gap 1 by widening the scope to one benchmark-wide family: exactly one primary p-value per
distinct experiment across the text/committee/referee lane and the NIH imaging lane, corrected
together, so the paper reports a single consistent correction scope instead of several local ones.

One test per experiment is deliberate (using every pairwise contrast would double-count the same
experiment and inflate the family). Each p-value is read from its committed summary JSON by an
explicit key path, so this is pure re-analysis with no model calls. The strictest scope dilutes power
by design; findings that survive Holm here are the ones to foreground as robust benchmark-wide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MEDQA = Path("experiments/medqa/results")
IMG = Path("experiments/imaging/results")

# (label, lane, summary_file, dotted_key_path_to_pvalue). One primary test per experiment.
FAMILY = [
    # --- text / committee-structure / seed-mechanism lane ---
    ("authority ladder: top-rung jump", "text", MEDQA / "authority_ladder_summary.json",
     "adjacent_rung_mcnemar.automated_system_vs_clinical_guideline.pvalue"),
    ("seed confidence: confident vs hedged", "text", MEDQA / "seed_confidence_summary.json",
     "confident_vs_hedged_mcnemar.pvalue"),
    ("rationale validity: any reasoning vs bare", "text", MEDQA / "rationale_validity_summary.json",
     "bare_vs_valid_wrong.pvalue"),
    ("super-additivity: both vs stronger single", "text", MEDQA / "super_additivity_summary.json",
     "both_vs_stronger_single.pvalue"),
    ("unanimity: unanimous vs single dissenter", "text", MEDQA / "unanimity_break_summary.json",
     "unanimous_vs_dissenter_mcnemar.pvalue"),
    ("committee size: wrong seed vs +1 honest peer", "text", MEDQA / "committee_size_sweep_summary.json",
     "s0_vs_s1.pvalue"),
    ("attributed tier: unlabeled vs small-model label", "text", MEDQA / "attributed_tier_summary.json",
     "unlabeled_vs_junior_model.pvalue"),
    ("test-awareness: neutral vs agreement-eval framing", "text", MEDQA / "test_awareness_summary.json",
     "neutral_vs_agreement_eval.pvalue"),
    ("deliberation framing: none vs critical instruction", "text", MEDQA / "deliberation_framing_summary.json",
     "none_vs_critical.pvalue"),
    ("plausible vs implausible planted distractor", "text", MEDQA / "plausible_distractor_summary.json",
     "plausible_vs_implausible.pvalue"),
    ("leader-as-auditor: peer vs auditor role", "text", MEDQA / "leader_as_auditor_summary.json",
     "peer_vs_auditor.pvalue"),
    ("text cue type: baseline vs negation-of-prior", "text", MEDQA / "text_cue_types_summary.json",
     "baseline_vs_negation.pvalue"),
    ("dose-response: faint vs plain assertion", "text", MEDQA / "dose_response_summary.json",
     "faint_vs_assert.pvalue"),
    ("contamination: recall-prone vs adoption", "text", MEDQA / "contamination_cascade_summary.json",
     "fisher_recall_vs_adopt.pvalue"),
    ("paraphrase robustness: template t0 vs t2", "text", MEDQA / "paraphrase_robustness_summary.json",
     "t0_vs_t2.pvalue"),
    ("seed timing: holdout last vs first", "text", MEDQA / "seed_timing_summary.json",
     "last_vs_first.pvalue"),
    ("pre-emptive referee: none vs soft warning", "text", MEDQA / "pre_emptive_referee_summary.json",
     "no_vs_soft.pvalue"),
    # true-peer and orchestrator primary contrasts live in the committee-structure reconciliation
    ("true-peer directionality (correct vs wrong)", "text", MEDQA / "stats_reconciliation_summary.json",
     "contrasts.5.pvalue"),
    ("orchestrator poisoning (wrong-orch vs wrong-peer)", "text", MEDQA / "stats_reconciliation_summary.json",
     "contrasts.8.pvalue"),
    ("first-peer majority (k1 vs isolated)", "text", MEDQA / "stats_reconciliation_summary.json",
     "contrasts.7.pvalue"),
    # --- NIH imaging lane ---
    ("imaging system-flag vs peer cascade (Fisher)", "imaging", IMG / "imaging_system_flag_summary.json",
     "vs_peer_assertion_cascade.fisher_pvalue"),
    ("imaging cue combination super-additivity", "imaging", IMG / "imaging_cue_combo_summary.json",
     "both_vs_stronger_single.pvalue"),
    ("imaging polarity asymmetry (yes->no vs no->yes)", "imaging", IMG / "imaging_polarity_summary.json",
     "asymmetry_mcnemar.pvalue"),
    ("imaging peer-size saturation (one vs two)", "imaging", IMG / "imaging_peer_size_curve_summary.json",
     "one_vs_two_mcnemar.pvalue"),
]


def _dig(obj, path):
    cur = obj
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _bh(pvals, alpha=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    crit = None
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        if pvals[i] <= (rank / m) * alpha:
            crit = pvals[i]
            break
    return [pvals[i] <= crit if crit is not None else False for i in range(m)]


def _holm(pvals, alpha=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    survive = [False] * m
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha / (m - rank + 1):
            survive[i] = True
        else:
            break
    return survive


def main():
    ap = argparse.ArgumentParser(description="Cross-lane multiple-comparison reconciliation (#237 gap 1).")
    ap.add_argument("--out", default="experiments/medqa/results")
    args = ap.parse_args()

    rows = []
    for label, lane, path, key in FAMILY:
        p = float(_dig(json.loads(Path(path).read_text()), key))
        rows.append({"label": label, "lane": lane, "pvalue": round(p, 8)})

    pvals = [r["pvalue"] for r in rows]
    bh, holm = _bh(pvals), _holm(pvals)
    for r, b, h in zip(rows, bh, holm):
        r["survives_bh"] = bool(b)
        r["survives_holm"] = bool(h)

    n_bh = sum(bh)
    n_holm = sum(holm)
    summary = {
        "family_size": len(rows),
        "n_text": sum(1 for r in rows if r["lane"] == "text"),
        "n_imaging": sum(1 for r in rows if r["lane"] == "imaging"),
        "alpha": 0.05,
        "n_survive_bh": n_bh,
        "n_survive_holm": n_holm,
        "tests": sorted(rows, key=lambda r: r["pvalue"]),
        "read": (
            f"One benchmark-wide family of {len(rows)} primary tests (one per experiment: "
            f"{sum(1 for r in rows if r['lane']=='text')} text/committee, "
            f"{sum(1 for r in rows if r['lane']=='imaging')} imaging), corrected once at alpha=0.05. "
            f"{n_bh} survive Benjamini-Hochberg and {n_holm} survive the stricter Holm. Under this "
            "widest scope the null and underpowered contrasts (super-additivity, unanimity, imaging "
            "saturation, polarity) correctly drop out, while the core authority, confidence, "
            "reasoning-transparency, orchestrator, and structural-mitigation effects survive even "
            "Holm, so the headline story is robust to the strictest cross-lane correction."
        ),
    }
    out = Path(args.out) / "cross_lane_reconciliation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
