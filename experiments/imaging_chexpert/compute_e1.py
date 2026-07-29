import json
import argparse
from pathlib import Path
from benchmaxxing.stats import fisher_exact

def count_flips(jsonl_path):
    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    n = len(rows)
    flips = sum(1 for r in rows if r["iso"] != r["clean"])
    no_flips = n - flips
    return n, flips, no_flips

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--present", required=True)
    ap.add_argument("--absent", required=True)
    args = ap.parse_args()
    
    n_p, flips_p, no_flips_p = count_flips(args.present)
    n_a, flips_a, no_flips_a = count_flips(args.absent)
    
    # Fisher exact needs a 2x2 table: [[flips_p, no_flips_p], [flips_a, no_flips_a]]
    table = [[flips_p, no_flips_p], [flips_a, no_flips_a]]
    res = fisher_exact(table)
    
    print("=" * 60)
    print("E1: Solo Flip Rate (Device-Present vs Absent)")
    print("=" * 60)
    print(f"Device-Present: n={n_p}, flips={flips_p} ({flips_p/n_p:.1%})")
    print(f"Device-Absent : n={n_a}, flips={flips_a} ({flips_a/n_a:.1%})")
    print("-" * 60)
    print(f"p-value:   {res.pvalue:.6g}")
    
    e1_result = {
        "endpoint": "E1",
        "description": "Solo flip rate, device-present vs device-absent (independent Fisher exact)",
        "table_2x2": {
            "device_present": {"flips": flips_p, "no_flips": no_flips_p},
            "device_absent": {"flips": flips_a, "no_flips": no_flips_a},
        },
        "rates": {
            "device_present": round(flips_p / n_p, 4),
            "device_absent": round(flips_a / n_a, 4),
        },
        "fisher_pvalue": res.pvalue,
    }
    
    out = Path("experiments/chexpert/results/natural_independent/confirmatory_e1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(e1_result, indent=2))
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
