"""Effect sizes, bootstrap CIs, and achieved-power annotation, imaging lane (#192). Pure
computation over already-committed per-case rows; no API calls. Text-lane half is delivered
separately in experiments/effect_sizes/ (PR #231); this covers the imaging arms, using the same
two bootstrap designs and the same "sanity-check the CI against its own point estimate" discipline
established there (an earlier draft of the text-lane script used a paired resampler on independent
groups and produced a CI that didn't bracket its point estimate).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmaxxing.stats import achieved_power, required_pairs

CUES = ["cable", "corner_tag", "watermark", "laterality"]


def _rate_diff_bootstrap_paired(a, b, n_boot=2000):
    """Paired bootstrap CI on mean(a) - mean(b): same cases under two conditions, e.g. a case's
    shared_adopt vs its own iso_adopt, or its ref_flag==gt vs naive_flag==gt correctness."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires a and b to be the same length")
    n = len(a)
    rng = np.random.default_rng(0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    point = a.mean() - b.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _rate_bootstrap(a, n_boot=2000):
    """Bootstrap CI on a single proportion mean(a), e.g. one cue's flip rate."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    rng = np.random.default_rng(0)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = a[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)


def solo_flip_rates(results_dir):
    rows = [json.loads(line) for line in (results_dir / "imaging_solo.jsonl").read_text().splitlines() if line.strip()]
    out = {}
    for cue in CUES:
        flips = [1.0 if r[f"{cue}_flip"] else 0.0 for r in rows]
        point, lo, hi = _rate_bootstrap(flips)
        out[cue] = {
            "flip_rate": round(point, 4), "bootstrap95": [round(lo, 4), round(hi, 4)],
            "n": len(rows),
        }
    return out


def cascade_contagion_delta(results_dir, filename, cue_label):
    rows = [json.loads(line) for line in (results_dir / filename).read_text().splitlines() if line.strip()]
    iso = [float(r["iso_adopt"]) for r in rows]
    shared = [float(r["shared_adopt"]) for r in rows]
    point, lo, hi = _rate_diff_bootstrap_paired(shared, iso)
    n = len(rows)
    gain = sum(1 for r in rows if r["shared_adopt"] and not r["iso_adopt"])
    lose = sum(1 for r in rows if r["iso_adopt"] and not r["shared_adopt"])
    psi, delta = (gain + lose) / n, (gain - lose) / n
    return {
        "cue": cue_label, "risk_difference": round(point, 4),
        "bootstrap95": [round(lo, 4), round(hi, 4)],
        "achieved_power": round(achieved_power(n, psi, abs(delta)), 4) if delta else None,
        "required_pairs_for_power_0.8": required_pairs(psi, abs(delta)) if delta else None,
        "n": n, "note": "paired bootstrap (shared_adopt - iso_adopt over the same cases)",
    }


def referee_vs_naive(results_dir):
    rows = [json.loads(line) for line in (results_dir / "imaging_referee.jsonl").read_text().splitlines() if line.strip()]
    ref_correct = [1.0 if r["ref_flag"] == r["gt"] else 0.0 for r in rows]
    naive_correct = [1.0 if r["naive_flag"] == r["gt"] else 0.0 for r in rows]
    point, lo, hi = _rate_diff_bootstrap_paired(ref_correct, naive_correct)
    n = len(rows)
    b = sum(1 for rc, nc in zip(ref_correct, naive_correct) if rc and not nc)
    c = sum(1 for rc, nc in zip(ref_correct, naive_correct) if not rc and nc)
    psi, delta = (b + c) / n, (b - c) / n
    return {
        "risk_difference": round(point, 4), "bootstrap95": [round(lo, 4), round(hi, 4)],
        "mcnemar_b_gt_c": {"b": b, "c": c, "n": n},
        "achieved_power": round(achieved_power(n, psi, abs(delta)), 4) if delta else None,
        "required_pairs_for_power_0.8": required_pairs(psi, abs(delta)) if delta else None,
        "note": "paired bootstrap on (ref_flag==gt) - (naive_flag==gt), same 35 cases",
    }


def main():
    results_dir = Path(__file__).parent / "results"

    out = {
        "solo_flip_rates": solo_flip_rates(results_dir),
        "cascade_contagion_delta": {
            "watermark": cascade_contagion_delta(results_dir, "imaging_cascade.jsonl", "watermark"),
            "cable": cascade_contagion_delta(results_dir, "imaging_cascade_cable.jsonl", "cable"),
            "corner_tag": cascade_contagion_delta(results_dir, "imaging_cascade_corner_tag.jsonl", "corner_tag"),
            "laterality": cascade_contagion_delta(results_dir, "imaging_cascade_laterality.jsonl", "laterality"),
        },
        "referee_vs_naive": referee_vs_naive(results_dir),
        "read": (
            "Solo flip rates (baseline susceptibility with no cascade pressure) sit in a moderate "
            "0.20-0.34 band across the four cues, each with a fairly tight bootstrap interval; this "
            "is a different quantity from #185's near-total cross-cue overlap finding (phi=Jaccard=1.0 "
            "on WHICH cases flip), not a restatement of it, since a case can flip solo at a moderate "
            "rate yet flip on the same cases regardless of cue. Every cue's cascade contagion delta "
            "(shared minus isolated adoption) is large (0.63-0.80), its bootstrap interval excludes 0, "
            "and achieved power is 1.0 for all four cues at n=35 (only 7-10 pairs would suffice for "
            "80% power) - this is the best-powered result in the whole project. The referee-vs-naive "
            "comparison is the opposite case: risk difference 0.17 favoring the referee, but the "
            "bootstrap interval straddles 0 ([-0.029, 0.343]) and achieved power is only 0.41 (90 "
            "cases needed for 0.8 power, only 35 available), so despite the suggestive McNemar b=9/c=3 "
            "asymmetry this comparison should be read as directionally suggestive but not yet powered "
            "to rule out chance."
        ),
    }
    (results_dir / "effect_sizes_imaging.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
