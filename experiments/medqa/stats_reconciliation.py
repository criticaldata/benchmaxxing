"""Stats reconciliation (#237): fold the committee-structure battery into one multiple-comparison
family and attach bootstrap 95% CIs to each headline adoption-rate delta.

Pure re-analysis of already-committed per-case JSONL outputs. No model calls, so it runs even while
live flash-lite runs are blocked. For each experiment a single headline paired contrast is defined
(two per-case adoption flags); we report the adoption-rate delta with a paired bootstrap 95% CI and
the paired McNemar p-value. The full family of headline p-values is then corrected with both
Benjamini-Hochberg (BH) and Holm, so the paper can report one consistent correction scope instead
of a mix of corrected old tests and uncorrected new ones.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from benchmaxxing.stats import mcnemar

RESULTS = Path("experiments/medqa/results")

# Each headline contrast: (label, filename, flag_a, flag_b) with the claim being "b differs from a".
CONTRASTS = [
    ("authority_ladder: guideline vs control", "authority_ladder.jsonl", "control_adopt", "clinical_guideline_adopt"),
    ("authority_ladder: senior vs control", "authority_ladder.jsonl", "control_adopt", "senior_attending_adopt"),
    ("seed_confidence: confident vs hedged", "seed_confidence.jsonl", "hedged_adopt", "confident_adopt"),
    ("rationale_validity: any-reasoning vs bare", "rationale_validity.jsonl", "bare_adopt", "valid_wrong_adopt"),
    ("super_additivity: both vs peer-alone", "super_additivity.jsonl", "peer_adopt", "both_adopt"),
    ("true_peer: correct vs wrong peer", "true_peer_control.jsonl", "wrong_peer_adopt", "correct_peer_adopt"),
    ("unanimity: unanimous vs with-dissenter", "unanimity_break.jsonl", "with_dissenter_adopt", "unanimous_wrong_adopt"),
    ("majority_pressure: k1 vs isolated", "majority_pressure.jsonl", "isolated_adopt", "k1_adopt"),
    ("orchestrator: wrong-orch vs wrong-peer", "orchestrator_failure.jsonl", "wrong_peer_output_wrong", "wrong_orch_output_wrong"),
]


def _load(fn):
    p = RESULTS / fn
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _paired(rows, a, b):
    pairs = [(int(r[a]), int(r[b])) for r in rows if a in r and b in r]
    n = len(pairs)
    ra = sum(x for x, _ in pairs) / n
    rb = sum(y for _, y in pairs) / n
    gain = sum(1 for x, y in pairs if y and not x)
    lose = sum(1 for x, y in pairs if x and not y)
    p = mcnemar(gain, lose).pvalue
    # paired bootstrap 95% CI on the delta rate(b) - rate(a)
    rng = random.Random(12345)
    deltas = []
    idx = range(n)
    for _ in range(5000):
        sample = [pairs[rng.choice(idx)] for _ in idx]
        da = sum(x for x, _ in sample) / n
        db = sum(y for _, y in sample) / n
        deltas.append(db - da)
    deltas.sort()
    lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]
    return {"n": n, "rate_a": round(ra, 4), "rate_b": round(rb, 4), "delta": round(rb - ra, 4),
            "delta_ci95": [round(lo, 4), round(hi, 4)], "mcnemar_gain": gain, "mcnemar_lose": lose,
            "pvalue": round(p, 8)}


def _bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    survive = [False] * m
    crit = None
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        if pvals[i] <= (k / m) * 0.05:
            crit = pvals[i]
            break
    if crit is not None:
        for i in range(m):
            survive[i] = pvals[i] <= crit
    return survive


def _holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    survive = [False] * m
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= 0.05 / (m - rank + 1):
            survive[i] = True
        else:
            break
    return survive


def main():
    ap = argparse.ArgumentParser(description="Stats reconciliation over the committee-structure battery (#237).")
    ap.add_argument("--out", default="experiments/medqa/results")
    args = ap.parse_args()

    rows_out = []
    for label, fn, a, b in CONTRASTS:
        if not (RESULTS / fn).exists():
            continue
        rows = _load(fn)
        stat = _paired(rows, a, b)
        stat["label"] = label
        rows_out.append(stat)

    pvals = [r["pvalue"] for r in rows_out]
    bh = _bh(pvals)
    holm = _holm(pvals)
    for r, s_bh, s_holm in zip(rows_out, bh, holm):
        r["survives_bh"] = s_bh
        r["survives_holm"] = s_holm

    summary = {
        "family_size": len(rows_out),
        "correction": "BH and Holm at alpha=0.05 over the committee-structure headline family",
        "contrasts": rows_out,
        "read": (
            "One consistent multiple-comparison scope over the committee-structure battery, each "
            "headline contrast reported as an adoption-rate delta with a paired bootstrap 95% CI and "
            "a paired McNemar p, then BH- and Holm-corrected together. Findings that survive Holm "
            "(the strictest) are the robust ones to foreground; those surviving only BH or neither "
            "are real but correction-sensitive and should be stated as such."
        ),
    }
    out = Path(args.out) / "stats_reconciliation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
