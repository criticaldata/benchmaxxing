"""Gemini vs a second lineage on the MIMIC-CXR text lane, one table from the committed files.

Reads the committed Gemini summaries and the model-scoped copies for --model, and prints the
Markdown table the PR body and the paper quote. No API calls, no report text touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

R = Path("experiments/mimic_cxr_text")


def load(p):
    return json.load(open(p)) if Path(p).exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    slug = a.model.replace("/", "_")
    G, N = R / "results", R / "results" / slug
    G600 = R / "results_n600"
    rows = []

    def row(arm, metric, g, n, note=""):
        rows.append((arm, metric, g, n, note))

    gs, ns = load(G600 / "solo_results.json"), load(N / "solo_results.json")
    gf = gs["flip_rate_by_model"]["gemini-2.5-flash-lite"]
    nf = ns["flip_rate_by_model"][a.model]
    row("solo n=600", "cue flip rate, overall", f"{gf['overall']:.3f}", f"{nf['overall']:.3f}", "flash-lite tier; 3 cues x 600")
    for c in ("lexical_overlap", "longest_option", "option_order"):
        row("", f"  {c}", f"{gf['per_cue'][c]:.3f}", f"{nf['per_cue'][c]:.3f}")
    row("", "noise floor (repeat call, temp 0)", f"{gs['noise_floor_by_model']['gemini-2.5-flash-lite']:.3f}",
        f"{ns['noise_floor_by_model'][a.model]:.3f}")
    gr, nr = load(G600 / "solo_results_refusal_aware.json"), load(N / "solo_results_refusal_aware.json")
    gk = gr.get("gemini-2.5-flash-lite", {}); nk = nr.get(a.model, {})
    if gk and nk:
        row("", "abstention rate", f"{gk['abstention_rate']:.4f}", f"{nk['abstention_rate']:.4f}")
    gsr = [json.loads(l) for l in open(G / "solo_records.jsonl")]
    nsr = [json.loads(l) for l in open(N / "solo_records.jsonl")]
    row("", "clean-prompt errors (hard cases)", f"{sum(not r['clean_correct'] for r in gsr)}/600",
        f"{sum(not r['clean_correct'] for r in nsr)}/600", "same 600 case indices")

    gc, nc = load(G / "cascade_results.json"), load(N / "cascade_results.json")
    if nc:
        for k in ("mean_shared_adopt", "mean_isolated_adopt", "mean_contagion"):
            row("cascade n=20" if k == "mean_shared_adopt" else "", k.replace("mean_", ""), f"{gc[k]:.3f}", f"{nc[k]:.3f}")

    gb, nb = load(G / "blind_metric_summary.json"), load(N / "blind_metric_summary.json")
    for k in ("baseline", "blind", "test_aware"):
        row("blind metric n=40" if k == "baseline" else "", f"decoy uptake, {k}", f"{gb['decoy_uptake'][k]:.3f}", f"{nb['decoy_uptake'][k]:.3f}")
    row("", "drifted / named the rubric", f"{gb['naming_vs_drifting']['n_drifted']} / {gb['naming_vs_drifting']['n_named_rubric']}",
        f"{nb['naming_vs_drifting']['n_drifted']} / {nb['naming_vs_drifting']['n_named_rubric']}")

    gj, nj = load(G / "referee_judge_summary.json"), load(N / "referee_judge_summary.json")
    row("referee judge n=40", "holdout adopted the shortcut", str(gj["n_holdout_adopted_shortcut"]), str(nj["n_holdout_adopted_shortcut"]))
    for k in ("precision", "recall", "fpr"):
        row("", f"  judge vs adoption, {k}", f"{gj['same_lineage_judge_vs_adoption'][k]:.3f}", f"{nj['same_lineage_judge_vs_adoption'][k]:.3f}")

    gd, nd = load(G / "referee_deployable_summary.json"), load(N / "referee_deployable_summary.json")
    gdep = gd["referees_vs_adoption_with_clean_control"]; ndep = nd["referees_vs_adoption_with_clean_control"]
    dk = [k for k in gdep if k.startswith("deployable")][0]
    row("referee deployable n=40", "false positives on clean control", str(gd["n_false_positive_on_clean_control"]), str(nd["n_false_positive_on_clean_control"]))
    for k in ("precision", "recall", "fpr"):
        row("", f"  deployable referee, {k}", f"{gdep[dk][k]:.3f}", f"{ndep[dk][k]:.3f}", "with clean control")

    ga, na = load(G / "break_it_a_summary.json")["A_contaminated_context"], load(N / "break_it_a_summary.json")["A_contaminated_context"]
    row("break-it A n=20", "flag adoption minus control", f"{ga['effect']:.3f}", f"{na['effect']:.3f}", "Gemini row pools two tiers")
    gD, nD = load(G / "break_it_d_summary.json"), load(N / "break_it_d_summary.json")
    row("break-it D", "decoy drift (incentive minus control)", f"{gD['decoy_drift']:+.3f} (n={gD['n']})", f"{nD['decoy_drift']:+.3f} (n={nD['n']})",
        f"McNemar p {gD['decoy_mcnemar']['pvalue']:.2f} / {nD['decoy_mcnemar']['pvalue']:.2f}")

    gp, np_ = load(G / "push_c_summary.json"), load(N / "push_c_summary.json")
    for k in ("generic", "anchored", "anchored_strong", "anchored_solo"):
        row("push C n=60" if k == "generic" else "", f"conformity, {k}", f"{gp[k]['rate']:.3f}", f"{np_[k]['rate']:.3f}")
    row("", "anchored_strong vs generic, McNemar p", f"{gp['anchored_strong_vs_generic_paired']['mcnemar_p']:.3f}",
        f"{np_['anchored_strong_vs_generic_paired']['mcnemar_p']:.4f}")

    gfm, nfm = load(G / "deliberation_framing_summary.json"), load(N / "deliberation_framing_summary.json")
    for k in ("none", "collaborative", "independent", "critical"):
        row("deliberation framing n=60" if k == "none" else "", f"adoption, {k}", f"{gfm['adoption_by_framing'][k]:.3f}", f"{nfm['adoption_by_framing'][k]:.3f}")
    for k in ("none_vs_independent", "none_vs_critical"):
        g, n = gfm[k], nfm[k]
        fmt = lambda x: "< 1e-6" if x["pvalue"] == 0 else f"{x['pvalue']:.2g}"
        row("", f"  {k}, McNemar p", f"{fmt(g)} ({g['gain']}/{g['lose']})", f"{fmt(n)} ({n['gain']}/{n['lose']})")

    print(f"| Arm | Metric | gemini-2.5-flash-lite | {a.model} | Note |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
