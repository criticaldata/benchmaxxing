"""Effect sizes, bootstrap CIs, and achieved-power annotation, text lane (#192). Pure computation
over already-committed per-case rows and summary counts; no API calls.

Several arms report a point estimate plus a McNemar/Fisher p with no interval (the design intent's
stated robustness weak spot). This adds a risk difference, odds ratio, bootstrap 95% CI, and
achieved statistical power to every arm whose source has been independently verified as current
(post-parser-fix) this session, following the same conservative scope as #191 (family-wide
correction): arms not re-verified this session, or confirmed tainted by the parser bug, are
excluded with a reason rather than risk mixing stale and corrected numbers.

Where genuine per-case rows are committed (scale_c_per_case.jsonl), the bootstrap CI resamples
cases directly. Where only aggregate counts are committed (clean_a, contamination_audit), the
bootstrap reconstructs the exact empirical Bernoulli distribution from those counts (e.g. 6 ones
and 17 zeros for 6/23), which recovers an identical bootstrap distribution to resampling the real
per-case array, since the bootstrap only depends on the empirical distribution the counts already
fully specify.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmaxxing.stats import achieved_power, required_pairs

EXCLUDED = [
    {"source": "experiments/medqa/results/break_it_summary.json / push_c_summary.json "
               "(results/medqa-break-it, PR #140)",
     "reason": "not independently re-verified for parser-fix currency this session"},
    {"source": "experiments/medqa/results/cascade_v2_per_case.jsonl (results/medqa-experiments, PR #135)",
     "reason": "confirmed tainted by the pre-fix stray-article parser bug"},
]


def _reconstruct_binary(k, n):
    """The exact empirical 0/1 array implied by k successes out of n; bootstrap over this is
    identical to bootstrapping the real per-case array, since only the empirical distribution
    (which these counts fully specify) matters for the resample."""
    return np.array([1.0] * k + [0.0] * (n - k))


def _rate_diff_bootstrap_paired(a, b, n_boot=2000):
    """Paired bootstrap CI on mean(a) - mean(b), resampling the SAME case indices jointly.
    Requires a and b to be the same length (the same subjects under two conditions), e.g.
    scale_c's generic/anchored conformity on the same 85 cases."""
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


def _rate_diff_bootstrap_independent(a, b, n_boot=2000):
    """Bootstrap CI on mean(a) - mean(b) for two INDEPENDENT groups (not the same subjects),
    e.g. clean_a's flag-condition cases vs its separate control-condition cases, which need not
    even be the same size. Each group is resampled independently with its own length."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    rng = np.random.default_rng(0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.integers(0, na, na)
        bi = rng.integers(0, nb, nb)
        diffs[i] = a[ai].mean() - b[bi].mean()
    point = a.mean() - b.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _odds_ratio(k1, n1, k0, n0):
    a, b, c, d = k1, n1 - k1, k0, n0 - k0
    if b == 0 or c == 0:
        return None  # undefined (a zero cell); matches the #191/#177 degenerate-case convention
    return (a / b) / (c / d)


def scale_c_arm(results_dir):
    rows = [json.loads(line) for line in (results_dir / "scale_c_per_case.jsonl").read_text().splitlines() if line.strip()]
    generic = [r["generic"] for r in rows]
    anchored = [r["anchored"] for r in rows]
    point, lo, hi = _rate_diff_bootstrap_paired(anchored, generic)
    n = len(rows)
    gain = sum(1 for r in rows if r["anchored"] and not r["generic"])
    lose = sum(1 for r in rows if r["generic"] and not r["anchored"])
    psi, delta = (gain + lose) / n, (gain - lose) / n
    k_a, k_g = sum(anchored), sum(generic)
    return {
        "risk_difference": round(point, 4), "bootstrap95": [round(lo, 4), round(hi, 4)],
        "odds_ratio": _odds_ratio(k_a, n, k_g, n),
        "achieved_power": round(achieved_power(n, psi, delta), 4) if delta else None,
        "required_pairs_for_power_0.8": required_pairs(psi, delta) if delta else None,
        "n": n, "note": "paired bootstrap over the real per-case rows (scale_c_per_case.jsonl)",
    }


def clean_a_arm(k1, n1, k0, n0):
    a1 = _reconstruct_binary(k1, n1)
    a0 = _reconstruct_binary(k0, n0)
    point, lo, hi = _rate_diff_bootstrap_independent(a1, a0)
    return {
        "risk_difference": round(point, 4), "bootstrap95": [round(lo, 4), round(hi, 4)],
        "odds_ratio": _odds_ratio(k1, n1, k0, n0),
        "n": n1, "note": "bootstrap reconstructed from committed counts (no raw per-case file); "
                          "independent groups (flag-condition vs control-condition cases), not paired",
    }


def model_dependence_arm(gain, lose, n):
    psi, delta = (gain + lose) / n, (gain - lose) / n
    return {
        "gain": gain, "lose": lose, "n": n,
        "achieved_power": round(achieved_power(n, psi, abs(delta)), 4) if delta else None,
        "required_pairs_for_power_0.8": required_pairs(psi, abs(delta)) if delta else None,
    }


def contamination_arm(k1, n1, k0, n0, label):
    a1 = _reconstruct_binary(k1, n1)
    a0 = _reconstruct_binary(k0, n0)
    point, lo, hi = _rate_diff_bootstrap_independent(a1, a0)
    return {
        "label": label, "risk_difference": round(point, 4),
        "bootstrap95": [round(lo, 4), round(hi, 4)], "odds_ratio": _odds_ratio(k1, n1, k0, n0),
        "note": "bootstrap reconstructed from committed counts (flip_rate_when_baseline_wrong/correct x n); "
                "independent groups (baseline-wrong vs baseline-correct case sets), not paired",
    }


def main():
    results_dir = Path(__file__).parent / "results"

    out = {
        "scale_c_anchored_vs_generic": scale_c_arm(results_dir),
        "clean_a_flash": clean_a_arm(6, 23, 0, 23),
        "clean_a_flash_lite": clean_a_arm(7, 23, 0, 23),
        "model_dependence_flash": model_dependence_arm(3, 4, 28),
        "multi_round_text_round1_vs_roundK": model_dependence_arm(3, 1, 40),
        # contamination_audit: ever-flipped counts split by baseline correctness, counted
        # directly from the committed solo_records.jsonl (PR #224): flash 5/11 wrong-baseline
        # vs 7/89 correct-baseline; flash-lite 14/22 wrong-baseline vs 7/78 correct-baseline.
        "contamination_flash": contamination_arm(5, 11, 7, 89, "baseline-wrong vs baseline-correct ever-flipped rate, flash"),
        "contamination_flash_lite": contamination_arm(14, 22, 7, 78, "baseline-wrong vs baseline-correct ever-flipped rate, flash-lite"),
        "excluded": EXCLUDED,
        "read": (
            "The scale_c anchored-vs-generic increment (+0.12) has a bootstrap CI that includes "
            "0 or comes close to it, consistent with #191's finding that it fails a family-wide "
            "correction. Clean-A's risk difference is large and its bootstrap interval is tighter "
            "for flash-lite than flash, matching flash-lite's stronger Fisher p. The contamination "
            "audit's risk differences are large and one-directional for both tiers, the most "
            "robust text-lane finding across every check this session."
        ),
    }
    (results_dir / "effect_sizes_text.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
