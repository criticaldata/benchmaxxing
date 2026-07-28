"""Recompute the cascade-derived imaging analyses from the post-#338 transcripts.

Why this script exists
----------------------
#338 fixed the plant direction in ``imaging_cascade.py`` and regenerated the four per-cue cascade
transcripts, but nothing downstream of them was rebuilt. Several committed analyses therefore still
published pre-fix contagion, five days after the fix landed:

- ``results/effect_sizes_imaging.json``: ``cascade_contagion_delta.*.risk_difference``
- ``results/claim4_quantification.json``: the ``cascade_contagion`` vector and the Spearman
  correlation derived from it
- ``README.md``: the headline cascade table

Each of those is a pure function of the committed transcripts, so all of them can be rebuilt with no
API key and no images. This script is that rebuild, so the numbers stop being hand-copied.

What this script deliberately does NOT touch
--------------------------------------------
``results/imaging_referee.jsonl`` and everything scored from it. 13 of its 35 rows describe committee
boards the corrected design no longer produces, and recomputing them needs fresh referee re-read
calls, not arithmetic. See ``referee_valid_subgroup.py`` for the honest offline treatment (restrict
to the 22 rows where both designs agree).

Reproduce
---------
    python -m experiments.imaging.recompute_derived --check   # verify committed files, exit 1 on drift
    python -m experiments.imaging.recompute_derived --write   # rewrite them in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# cue name -> transcript. The unsuffixed file is the watermark arm (its summary records cue=watermark).
CUE_TRANSCRIPTS = {
    "cable": "imaging_cascade_cable.jsonl",
    "corner_tag": "imaging_cascade_corner_tag.jsonl",
    "laterality": "imaging_cascade_laterality.jsonl",
    "watermark": "imaging_cascade.jsonl",
}

# claim4 orders its vector cable, corner_tag, watermark, laterality (see its own "read" field).
CLAIM4_ORDER = ("cable", "corner_tag", "watermark", "laterality")

CORRECTED_PLANT = "no"


def load(name):
    with open(RESULTS / name) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contagion(rows):
    """shared_adopt - iso_adopt, the definition used by imaging_cascade.py's own summary."""
    n = len(rows)
    shared = sum(r["shared_adopt"] for r in rows) / n
    iso = sum(r["iso_adopt"] for r in rows) / n
    return {"n": n, "shared_adopt": shared, "iso_adopt": iso, "risk_difference": shared - iso}


def per_cue():
    """Recompute contagion per cue, refusing any transcript that is still pre-fix."""
    out = {}
    for cue, fname in CUE_TRANSCRIPTS.items():
        rows = load(fname)
        planted = {r.get("wrong") for r in rows}
        if planted != {CORRECTED_PLANT}:
            raise SystemExit(
                f"{fname} is not post-#338: planted reads are {sorted(planted)}, expected "
                f"{{'{CORRECTED_PLANT}'}}. Refusing to derive numbers from a pre-fix transcript."
            )
        out[cue] = contagion(rows)
    return out


def pairwise_agreement():
    """Cross-cue agreement on which cases adopt, recomputed from the post-#338 transcripts.

    This is the statistic behind the case-driven (not cue-driven) reading: if the same cases cascade
    whichever cue is present, it is the case that cascades. ``phi`` is undefined when one cue's
    adoption is constant across all cases (cable adopts 35/35), so it is reported as None there and
    the Jaccard overlap carries the comparison.
    """
    import itertools
    import math

    adopt = {}
    for cue, fname in CUE_TRANSCRIPTS.items():
        adopt[cue] = {r["case_id"]: r["shared_adopt"] for r in load(fname)}
    ids = sorted(set.intersection(*[set(v) for v in adopt.values()]))

    out = {}
    for x, y in itertools.combinations(CUE_TRANSCRIPTS, 2):
        a, b = adopt[x], adopt[y]
        n11 = sum(1 for i in ids if a[i] == 1 and b[i] == 1)
        n10 = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
        n01 = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
        n00 = sum(1 for i in ids if a[i] == 0 and b[i] == 0)
        den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        out[f"{x}_vs_{y}"] = {
            "phi": None if den == 0 else (n11 * n00 - n10 * n01) / den,
            "jaccard": n11 / (n11 + n10 + n01) if (n11 + n10 + n01) else 1.0,
            "n": len(ids),
        }
    return out


def spearman(xs, ys):
    """Spearman rho on tiny vectors, no SciPy dependency. Average ranks on ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def rebuild(cues):
    """Return {filename: updated json object} for every derived file."""
    updates = {}

    p = RESULTS / "effect_sizes_imaging.json"
    if p.exists():
        obj = json.loads(p.read_text())
        block = obj.get("cascade_contagion_delta")
        if isinstance(block, dict):
            for cue, stats in cues.items():
                if cue in block and isinstance(block[cue], dict):
                    block[cue]["risk_difference"] = round(stats["risk_difference"], 4)
                    block[cue]["shared_adopt"] = round(stats["shared_adopt"], 4)
                    block[cue]["iso_adopt"] = round(stats["iso_adopt"], 4)
                    block[cue]["n"] = stats["n"]
            obj["provenance"] = (
                "cascade_contagion_delta recomputed from the post-#338 per-cue transcripts by "
                "experiments/imaging/recompute_derived.py"
            )
            updates[p.name] = obj

    p = RESULTS / "claim4_quantification.json"
    if p.exists():
        obj = json.loads(p.read_text())
        if "cascade_contagion" in obj:
            vec = [round(cues[c]["risk_difference"], 4) for c in CLAIM4_ORDER]
            obj["cascade_contagion"] = vec
            obj["cascade_contagion_cue_order"] = list(CLAIM4_ORDER)
            solo = obj.get("solo_flip_above_noise")
            if isinstance(solo, list) and len(solo) == len(vec):
                rho = spearman(solo, vec)
                obj["spearman_solo_vs_contagion"] = {
                    "rho": None if rho is None else round(rho, 4),
                    "note": (
                        "Post-#338 contagion is near-uniform across cues, so this rank correlation "
                        "is no longer informative; the case-overlap statistic (phi) is the evidence "
                        "for the case-driven reading."
                    ),
                }
            obj["provenance"] = (
                "cascade_contagion recomputed from the post-#338 per-cue transcripts by "
                "experiments/imaging/recompute_derived.py"
            )
            updates[p.name] = obj

    return updates


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report drift, exit 1 if any")
    g.add_argument("--write", action="store_true", help="rewrite the derived files in place")
    args = ap.parse_args()

    cues = per_cue()
    print("Contagion recomputed from post-#338 transcripts:")
    for cue in CLAIM4_ORDER:
        s = cues[cue]
        print(f"  {cue:11} shared {s['shared_adopt']:.4f}  iso {s['iso_adopt']:.4f}  "
              f"contagion {s['risk_difference']:+.4f}  (n={s['n']})")

    updates = rebuild(cues)
    drift = 0
    for name, obj in updates.items():
        path = RESULTS / name
        new = json.dumps(obj, indent=2) + "\n"
        old = path.read_text()
        if new != old:
            drift += 1
            if args.write:
                path.write_text(new)
                print(f"  rewrote {name}")
            else:
                print(f"  DRIFT {name} does not match the transcripts")
        else:
            print(f"  ok {name}")

    if args.check and drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
