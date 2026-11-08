"""Tier swap on the hidden-rubric arm (no API calls): Flash on text (MedQA, MedMCQA), Flash-Lite on imaging.

The committed hidden-rubric arms run Gemini 2.5 Flash-Lite on text (MedQA, 40 cases) and Gemini 2.5
Flash on imaging (NIH ChestX-ray14, 35 films), so the "imaging drifts more" comparison confounds
modality with tier. This reads the committed rows of those two arms beside the swapped arms
(blind_metric.py --model gemini-2.5-flash; imaging_blind_metric.py --model gemini-2.5-flash-lite)
and reports, on the same cases, the tier contrast within each modality and the modality contrast
within each tier, with exact McNemar tests for the paired (within-modality) comparisons.

    python experiments/blind_metric/tier_swap_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmaxxing.stats import mcnemar

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    ("text", "gemini-2.5-flash-lite"): "experiments/blind_metric/results/blind_metric.jsonl",
    ("text", "gemini-2.5-flash"): "experiments/blind_metric/results/gemini-2.5-flash/blind_metric.jsonl",
    ("medmcqa", "gemini-2.5-flash-lite"): "experiments/medmcqa/results/blind_metric.jsonl",
    ("medmcqa", "gemini-2.5-flash"): "experiments/medmcqa/results/gemini-2.5-flash/blind_metric.jsonl",
    ("imaging", "gemini-2.5-flash"): "experiments/imaging/results/imaging_blind_metric.jsonl",
    ("imaging", "gemini-2.5-flash-lite"): "experiments/imaging/results/gemini-2.5-flash-lite/imaging_blind_metric.jsonl",
}
TIERS = ("gemini-2.5-flash-lite", "gemini-2.5-flash")


def _rows(rel):
    return {r["case_id"]: r for r in map(json.loads, (ROOT / rel).read_text().splitlines()) if r}


def _pair(a, b, ids, col):
    gain = sum(1 for i in ids if b[i][col] and not a[i][col])
    lose = sum(1 for i in ids if a[i][col] and not b[i][col])
    return {"gain": gain, "lose": lose, "pvalue": round(mcnemar(gain, lose).pvalue, 6)}


def main():
    rows = {k: _rows(v) for k, v in ARMS.items()}
    out = {"new_api_calls_this_run": 0, "arms": {}, "tier_within_modality": {}, "modality_within_tier": {}}
    for (lane, tier), r in rows.items():
        n = len(r)
        cell = {"n": n,
                "blind_uptake": round(sum(x["blind_is_decoy"] for x in r.values()) / n, 4),
                "aware_uptake": round(sum(x["aware_is_decoy"] for x in r.values()) / n, 4),
                "suppression_blind_vs_aware": _pair({i: {"c": x["aware_is_decoy"]} for i, x in r.items()},
                                                    {i: {"c": x["blind_is_decoy"]} for i, x in r.items()}, list(r), "c"),
                "n_drifted": sum(x["blind_is_decoy"] for x in r.values()),
                "n_named_rubric": sum(bool(x["named_rubric_when_drifted"]) for x in r.values() if x["blind_is_decoy"])}
        if lane == "imaging":
            # every film has the finding, so the correct read is yes and a decoy of "no" means the
            # baseline read was correct; the decoy is the opposite of the model's own read
            cell["baseline_correct"] = sum(1 for x in r.values() if str(x["decoy"]).lower() == "no")
        out["arms"][f"{lane}/{tier}"] = cell
    for lane in ("text", "medmcqa", "imaging"):
        a, b = rows[(lane, TIERS[0])], rows[(lane, TIERS[1])]
        ids = sorted(set(a) & set(b))
        out["tier_within_modality"][lane] = {
            "n_paired": len(ids),
            "blind_flash_vs_lite": _pair(a, b, ids, "blind_is_decoy"),
            "aware_flash_vs_lite": _pair(a, b, ids, "aware_is_decoy")}
    for tier in TIERS:
        t, i = out["arms"][f"text/{tier}"], out["arms"][f"imaging/{tier}"]
        out["modality_within_tier"][tier] = {
            "blind_text": t["blind_uptake"], "blind_medmcqa": out["arms"][f"medmcqa/{tier}"]["blind_uptake"],
            "blind_imaging": i["blind_uptake"],
            "imaging_minus_text_blind": round(i["blind_uptake"] - t["blind_uptake"], 4),
            "aware_text": t["aware_uptake"], "aware_imaging": i["aware_uptake"]}
    a = out["arms"]
    out["read"] = (
        f"Swapping tiers keeps the modality ordering: under Flash-Lite the hidden rubric takes "
        f"{a['text/gemini-2.5-flash-lite']['blind_uptake']} of MedQA answers and "
        f"{a['imaging/gemini-2.5-flash-lite']['blind_uptake']} of NIH reads; under Flash "
        f"{a['text/gemini-2.5-flash']['blind_uptake']} and {a['imaging/gemini-2.5-flash']['blind_uptake']}. "
        f"MedMCQA under Flash-Lite {a['medmcqa/gemini-2.5-flash-lite']['blind_uptake']} and under Flash "
        f"{a['medmcqa/gemini-2.5-flash']['blind_uptake']}. "
        f"Tier matters on its own: Flash drifts more than Flash-Lite on the same cases in every lane "
        f"(MedQA {out['tier_within_modality']['text']['blind_flash_vs_lite']['gain']} vs "
        f"{out['tier_within_modality']['text']['blind_flash_vs_lite']['lose']} discordant, "
        f"P = {out['tier_within_modality']['text']['blind_flash_vs_lite']['pvalue']}; MedMCQA "
        f"{out['tier_within_modality']['medmcqa']['blind_flash_vs_lite']['gain']} vs "
        f"{out['tier_within_modality']['medmcqa']['blind_flash_vs_lite']['lose']}, "
        f"P = {out['tier_within_modality']['medmcqa']['blind_flash_vs_lite']['pvalue']}; imaging "
        f"{out['tier_within_modality']['imaging']['blind_flash_vs_lite']['gain']} vs "
        f"{out['tier_within_modality']['imaging']['blind_flash_vs_lite']['lose']}, "
        f"P = {out['tier_within_modality']['imaging']['blind_flash_vs_lite']['pvalue']}). "
        f"Test-awareness suppresses uptake in every cell.")
    p = ROOT / "experiments/blind_metric/results/tier_swap_summary.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
