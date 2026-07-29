"""Plant-direction check (#332): does the imaging cascade's `wrong = flip(clean_read)` design
contaminate contagion with accept-the-correction cases?

NIH cascades on finding-present cases (ground truth read = "yes"), so a planted read of "no" is
genuinely false, while "yes" is actually the truth (the committee correcting a wrong holdout).
This offline re-analysis of the committed transcripts splits contagion by the two subgroups. On NIH
the genuinely-false subgroup matches the overall (~+0.73), so the effect is real harmful sycophancy,
not an artifact; on a low-accuracy cohort (e.g. all-pneumothorax CheXpert) the two diverge sharply.

Run: PYTHONPATH=. python3 experiments/imaging/plant_direction_check.py
"""
import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FILES = {
    "watermark": "imaging_cascade.jsonl",
    "cable": "imaging_cascade_cable.jsonl",
    "corner_tag": "imaging_cascade_corner_tag.jsonl",
    "laterality": "imaging_cascade_laterality.jsonl",
}


def _contagion(rows):
    if not rows:
        return None
    shared = sum(r["shared_adopt"] for r in rows) / len(rows)
    iso = sum(r["iso_adopt"] for r in rows) / len(rows)
    return {"shared": shared, "iso": iso, "contagion": shared - iso, "n": len(rows)}


def analyze():
    out = {}
    for cue, fname in FILES.items():
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()
                and "shared_adopt" in line]
        out[cue] = {
            "all": _contagion(rows),
            "genuinely_false": _contagion([r for r in rows if r["wrong"] == "no"]),
            "planted_truth": _contagion([r for r in rows if r["wrong"] == "yes"]),
        }
    return out


if __name__ == "__main__":
    res = analyze()
    for cue, d in res.items():
        a, f, t = d["all"], d["genuinely_false"], d["planted_truth"]
        print(f"{cue:12} ALL {a['contagion']:+.3f} (n={a['n']}) | "
              f"genuinely-false {f['contagion']:+.3f} (n={f['n']}) | "
              f"planted-truth {t['contagion']:+.3f} (n={t['n']})")
