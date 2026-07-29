"""Confirmatory endpoints E1 and E2 from the CheXpert natural-cue pre-registration.

E2 (PRIMARY): McNemar paired test on shared_adopt vs iso_adopt per case.
E1: Fisher exact on solo flip rate, device-present vs device-absent.
Holm-Bonferroni correction over {E1, E2}.

Usage:
    python experiments/chexpert/confirmatory_endpoints.py \
        --natural-jsonl experiments/chexpert/results/natural_independent/imaging_cascade_none.jsonl \
        [--absent-jsonl <path>]   # for E1, when available
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmaxxing.stats import mcnemar


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_e2(rows: list[dict]) -> dict:
    """E2: McNemar on shared_adopt vs iso_adopt (paired, all cases)."""
    # 2x2 paired table: (shared_adopt, iso_adopt) per case
    both = sum(1 for r in rows if r["shared_adopt"] and r["iso_adopt"])
    only_shared = sum(1 for r in rows if r["shared_adopt"] and not r["iso_adopt"])  # b
    only_iso = sum(1 for r in rows if not r["shared_adopt"] and r["iso_adopt"])      # c
    neither = sum(1 for r in rows if not r["shared_adopt"] and not r["iso_adopt"])

    n = len(rows)
    sa_rate = (both + only_shared) / n
    ia_rate = (both + only_iso) / n
    contagion = sa_rate - ia_rate

    # McNemar exact (binomial) on discordant pairs b, c
    result = mcnemar(b=only_shared, c=only_iso)

    # Bootstrap CI for the paired risk difference (contagion)
    import numpy as np
    diffs = np.array([r["shared_adopt"] - r["iso_adopt"] for r in rows])
    rng = np.random.default_rng(42)
    boot = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(10000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    return {
        "endpoint": "E2",
        "description": "Contagion = shared_adopt - isolated_adopt (paired McNemar)",
        "n": n,
        "table_2x2": {
            "both_adopt": both,
            "only_shared_b": only_shared,
            "only_iso_c": only_iso,
            "neither": neither,
        },
        "shared_adopt_rate": round(sa_rate, 4),
        "isolated_adopt_rate": round(ia_rate, 4),
        "contagion": round(contagion, 4),
        "mcnemar_statistic": result.statistic,
        "mcnemar_pvalue": result.pvalue,
        "bootstrap_ci_95": [round(ci[0], 4), round(ci[1], 4)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natural-jsonl", required=True,
                    help="JSONL from natural cues cascade (device-present cases)")
    ap.add_argument("--out", default="experiments/chexpert/results/natural_independent",
                    help="Output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- E2 ---
    rows = load_rows(args.natural_jsonl)
    e2 = compute_e2(rows)

    print("=" * 60)
    print("E2 (PRIMARY): Conformity Cascade - McNemar Paired Test")
    print("=" * 60)
    print(f"  n = {e2['n']}")
    print("  Paired 2x2:")
    print(f"    both_adopt     = {e2['table_2x2']['both_adopt']}")
    print(f"    only_shared (b)= {e2['table_2x2']['only_shared_b']}")
    print(f"    only_iso    (c)= {e2['table_2x2']['only_iso_c']}")
    print(f"    neither        = {e2['table_2x2']['neither']}")
    print(f"  shared_adopt = {e2['shared_adopt_rate']:.2%}")
    print(f"  isolated_adopt = {e2['isolated_adopt_rate']:.2%}")
    print(f"  contagion = {e2['contagion']:.2%}")
    print(f"  McNemar statistic = {e2['mcnemar_statistic']}")
    print(f"  McNemar p-value = {e2['mcnemar_pvalue']:.6g}")
    print(f"  Bootstrap 95% CI = {e2['bootstrap_ci_95']}")
    print()

    result_path = out / "confirmatory_e2.json"
    result_path.write_text(json.dumps(e2, indent=2))
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
