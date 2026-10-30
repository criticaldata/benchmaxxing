"""Patient-clustered read of the NIH watermark cascade on the 500-film, 132-patient manifest.

No API calls: reads the per-case rows imaging_scale.py wrote into results_n500/ and the manifest's
patient ids. Reports the image-level contagion (shared adopt minus isolated adopt), an exact McNemar
on the paired reads, and a 95% interval that resamples patients rather than films
(benchmaxxing.stats.cluster_bootstrap_ci, 20,000 resamples, as the paper's methods). Per patient, the
contagion and how many patients carry a positive, zero or negative value; the 35-film cohort's
films inside the 500 as a subgroup; and, for the 150 films the July run also read, how often the
current reads agree with that run, whose watermark was drawn before the cue font was vendored.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmaxxing.stats import bootstrap_ci, cluster_bootstrap_ci, mcnemar

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "experiments/imaging/results"
OUT = ROOT / "experiments/imaging/results_n500"
N_BOOT = 20000


def _jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def main():
    rows = [r for r in _jsonl(OUT / "imaging_cascade_scale.jsonl") if "err" not in r]
    pid = {r["case_id"]: r["patient_id"] for r in csv.DictReader(open(R / "nih_manifest_scaled.csv"))}
    small = {r["case_id"] for r in csv.DictReader(open(R / "nih_manifest.csv"))}
    n = len(rows)
    d = np.array([r["shared_adopt"] - r["iso_adopt"] for r in rows], dtype=float)
    clusters = [pid[r["case_id"]] for r in rows]
    gain = sum(1 for r in rows if r["shared_adopt"] and not r["iso_adopt"])
    lose = sum(1 for r in rows if r["iso_adopt"] and not r["shared_adopt"])
    point, lo, hi = cluster_bootstrap_ci(d, clusters, n_boot=N_BOOT, seed=0)
    _, ilo, ihi = bootstrap_ci(d, n_boot=N_BOOT, seed=0)

    by_p = defaultdict(list)
    for r, c in zip(rows, clusters):
        by_p[c].append(r["shared_adopt"] - r["iso_adopt"])
    per_p = {c: float(np.mean(v)) for c, v in by_p.items()}
    sizes = sorted((len(v) for v in by_p.values()), reverse=True)

    sub = [r for r in rows if r["case_id"] in small]
    july = {r["case_id"]: r for r in _jsonl(R / "imaging_cascade_scale.jsonl")}
    both = [r for r in rows if r["case_id"] in july]
    agree = {k: sum(r[k] == july[r["case_id"]][k] for r in both) for k in ("clean", "iso", "shared")}

    out = {
        "n_films": n, "n_patients": len(by_p), "model": rows and json.loads(
            (OUT / "imaging_cascade_scale_summary.json").read_text())["model"],
        "new_api_calls_this_run": 0,
        "shared_adopt": round(sum(r["shared_adopt"] for r in rows) / n, 4),
        "isolated_adopt": round(sum(r["iso_adopt"] for r in rows) / n, 4),
        "contagion": round(point, 4),
        "ci95_films": [round(ilo, 4), round(ihi, 4)],
        "ci95_patients": [round(lo, 4), round(hi, 4)],
        "n_boot": N_BOOT,
        "mcnemar_shared_vs_isolated": {"gain": gain, "lose": lose, "pvalue": mcnemar(gain, lose).pvalue},
        "films_per_patient": {"max": sizes[0], "median": float(np.median(sizes)), "top5": sizes[:5],
                              "singletons": sum(s == 1 for s in sizes)},
        "per_patient_contagion": {"positive": sum(v > 0 for v in per_p.values()),
                                  "zero": sum(v == 0 for v in per_p.values()),
                                  "negative": sum(v < 0 for v in per_p.values()),
                                  "mean_of_patient_means": round(float(np.mean(list(per_p.values()))), 4)},
        "the_35_film_cohort_within": {
            "n": len(sub),
            "shared_adopt": round(sum(r["shared_adopt"] for r in sub) / len(sub), 4) if sub else None,
            "isolated_adopt": round(sum(r["iso_adopt"] for r in sub) / len(sub), 4) if sub else None},
        "july_150_overlap": {"n": len(both), "reads_agreeing": agree,
                             "july_contagion": round(sum(july[r["case_id"]]["shared_adopt"] - july[r["case_id"]]["iso_adopt"] for r in both) / len(both), 4),
                             "current_contagion_same_films": round(sum(r["shared_adopt"] - r["iso_adopt"] for r in both) / len(both), 4)},
    }
    # hidden-rubric arm and its matched no-note control on the same 500 films
    m = _jsonl(OUT / "matched_baseline_reason.jsonl")
    mc = [pid[r["case_id"]] for r in m]
    note = np.array([float(r["blind_is_decoy"]) for r in m])
    req = np.array([float(r["reason_is_decoy"]) for r in m])
    pn, ln_, hn = cluster_bootstrap_ci(note, mc, n_boot=N_BOOT, seed=0)
    pd_, ld, hd = cluster_bootstrap_ci(note - req, mc, n_boot=N_BOOT, seed=0)
    g = sum(1 for r in m if r["blind_is_decoy"] and not r["reason_is_decoy"])
    lo_ = sum(1 for r in m if r["reason_is_decoy"] and not r["blind_is_decoy"])
    by_pm = defaultdict(list)
    for r, c in zip(m, mc):
        by_pm[c].append(r["blind_is_decoy"] - r["reason_is_decoy"])
    out["hidden_note"] = {
        "n_films": len(m), "n_patients": len(by_pm),
        "note_uptake": round(pn, 4), "note_uptake_ci95_patients": [round(ln_, 4), round(hn, 4)],
        "request_only_uptake": round(float(req.mean()), 4),
        "matched_effect": round(pd_, 4), "matched_effect_ci95_patients": [round(ld, 4), round(hd, 4)],
        "mcnemar_note_vs_request": {"gain": g, "lose": lo_, "pvalue": mcnemar(g, lo_).pvalue},
        "patients_with_positive_matched_effect": int(sum(np.mean(v) > 0 for v in by_pm.values())),
        "patients_with_negative_matched_effect": int(sum(np.mean(v) < 0 for v in by_pm.values())),
        "named_rubric_when_drifted": sum(bool(r.get("named_rubric_when_drifted")) for r in _jsonl(OUT / "imaging_blind_metric.jsonl")),
    }
    out["read"] = (
        f"On {n} films from {len(by_p)} patients the watermark contagion is {out['contagion']:+.3f}; resampling "
        f"patients gives {out['ci95_patients']}, against {out['ci95_films']} resampling films.")
    (OUT / "patient_clustered.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
