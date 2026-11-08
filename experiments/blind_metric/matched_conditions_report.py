"""Cross-cohort report for the matched-conditions blind metric, plus the two derived files each
cohort needs: the referee calibration rows and, for a DUA cohort, the de-identified export.

Reads one or more cohort result directories written by matched_conditions.py and writes

    <out>/matched_conditions_report.json      per cohort and pooled: the note's effect net of the
                                              reason request, the noise band, the referee
                                              calibration, naming under both conditions, per model
    <cohort dir>/blind_metric_calibrated.jsonl per case: base_is_decoy and blind_is_decoy as the
                                              MAJORITY over repeats plus the noise band, in the row
                                              shape counterfactual_referee.py reads
    <cohort dir>/deid/matched_conditions.csv  with --deid: case_index (rank of case_id, the rule
                                              export_deid.py uses, so it pairs with the committed
                                              blind_metric.csv) plus the answer vectors as 0/1 and
                                              the per-case flags. No ids, no text.

The per-case rows carry no completion text, so the report runs identically on a public cohort and
on a de-identified MIMIC export; the text lives only in matched_conditions_calls.jsonl, which for
MIMIC stays where it was generated.

Usage
    PYTHONPATH=. python experiments/blind_metric/matched_conditions_report.py \
        --cohort medqa=experiments/blind_metric/results/matched \
        --cohort nih=experiments/imaging/results/matched \
        --cohort chexpert=experiments/imaging_chexpert/results/matched \
        --cohort mimic=experiments/mimic_cxr_image/results/matched \
        --out experiments/blind_metric/results/matched
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched_conditions import ALL_CONDITIONS, exact_mcnemar, majority, summarise, wilson  # noqa: E402


def load_rows(d: Path):
    p = d / "matched_conditions.jsonl"
    if p.exists():
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    q = d / "deid" / "matched_conditions.csv"
    if q.exists():
        return rows_from_deid(q)
    raise FileNotFoundError(f"{d}: neither matched_conditions.jsonl nor deid/matched_conditions.csv")


def rows_from_deid(path: Path):
    """Rebuild the per-case rows summarise() needs from the de-identified export."""
    out = []
    for r in csv.DictReader(open(path)):
        conds = [c for c in ALL_CONDITIONS if f"{c}_is_decoy_0" in r]
        k = max(int(c.split("_")[-1]) for c in r if c.startswith("reason_is_decoy_")) + 1
        row = {"case_id": r["case_index"], "patient_id": r.get("patient_index", r["case_index"]), "conditions": conds}
        for c in conds:
            row[f"{c}_is_decoy"] = [r[f"{c}_is_decoy_{i}"] == "1" for i in range(k)]
            # answers reduce to decoy / not-decoy in the export; disagreement is preserved exactly
            # for a binary read and is a lower bound for a letter read
            row[f"{c}_answers"] = [r[f"{c}_is_decoy_{i}"] for i in range(k)]
            row[f"{c}_named"] = [r[f"{c}_named_{i}"] == "1" for i in range(k)]
        out.append(row)
    return out


def write_calibrated(d: Path, rows):
    out = []
    for r in rows:
        out.append({"case_id": r["case_id"],
                    "base_is_decoy": majority(r["reason_is_decoy"]),
                    "blind_is_decoy": majority(r["note_is_decoy"]),
                    "noise_band": r.get("noise_band"), "note_driven": r.get("note_driven"),
                    "named_rubric_when_drifted": bool(majority(r["note_is_decoy"]) and any(r["note_named"]))})
    (d / "blind_metric_calibrated.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    return len(out)


def write_deid(d: Path, rows, repeats: int):
    rows = sorted(rows, key=lambda r: str(r["case_id"]))
    ids = [r["case_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate case_id, case_index would be ambiguous"
    conds = [c for c in ALL_CONDITIONS if all(f"{c}_is_decoy" in r for r in rows)]
    pats = sorted({str(r.get("patient_id")) for r in rows})
    prank = {p: i for i, p in enumerate(pats)}
    cols = ["case_index", "patient_index"]
    for c in conds:
        cols += [f"{c}_is_decoy_{i}" for i in range(repeats)]
    for c in conds:
        cols += [f"{c}_named_{i}" for i in range(repeats)]
    cols += ["noise_band", "note_effect", "note_driven", "strict_flip", "majority_flip"]
    (d / "deid").mkdir(exist_ok=True)
    with open(d / "deid" / "matched_conditions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(rows):
            vals = [i, prank[str(r.get("patient_id"))]]
            for c in conds:
                vals += [int(x) for x in r[f"{c}_is_decoy"]]
            for c in conds:
                vals += [int(x) for x in r[f"{c}_named"]]
            vals += [r.get("noise_band"), r.get("note_effect"), int(bool(r.get("note_driven"))),
                     int(bool(r.get("strict_flip"))), int(bool(r.get("majority_flip")))]
            w.writerow(vals)
    return len(rows)


def pooled(all_rows, repeats):
    s = summarise([dict(r) for r in all_rows], repeats)
    return {k: s[k] for k in ("n", "conditions", "effect_net_of_reason", "no_note", "effect_by_condition", "noise_band",
                              "two_by_two", "note_position", "image_ablated", "referee_calibration", "naming") if k in s}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deid", action="append", default=[], metavar="NAME",
                    help="cohorts whose per-case rows must be exported de-identified")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    report = {"cohorts": {}, "repeats": args.repeats}
    all_rows, by_model = [], defaultdict(list)
    for spec in args.cohort:
        name, d = spec.split("=", 1)
        d = Path(d)
        rows = load_rows(d)
        s = summarise(rows, args.repeats)  # also annotates rows with band and flags
        model = rows[0].get("model") if rows and rows[0].get("model") else None
        summ_path = d / "matched_conditions_summary.json"
        if summ_path.exists():
            model = model or json.load(open(summ_path)).get("model")
        s["model"] = model
        s["calibrated_rows"] = write_calibrated(d, rows)
        if name in args.deid:
            s["deid_rows"] = write_deid(d, rows, args.repeats)
        report["cohorts"][name] = s
        all_rows.extend(rows)
        by_model[model].extend(rows)

    report["pooled"] = pooled(all_rows, args.repeats)
    report["per_model"] = {m or "unknown": pooled(rs, args.repeats) for m, rs in by_model.items()}
    # The headline the committed arm reports is "N of M answers moved to the rewarded option";
    # the matched version is the same count under the note, beside the count under reason alone,
    # on the first repeat of each.
    n = len(all_rows)
    note1 = sum(r["note_is_decoy"][0] for r in all_rows)
    reason1 = sum(r["reason_is_decoy"][0] for r in all_rows)
    g = sum(1 for r in all_rows if r["note_is_decoy"][0] and not r["reason_is_decoy"][0])
    l = sum(1 for r in all_rows if r["reason_is_decoy"][0] and not r["note_is_decoy"][0])
    report["headline"] = {
        "n_cases": n,
        "moved_to_rewarded_option_under_note_first_repeat": note1,
        "moved_to_rewarded_option_under_reason_alone_first_repeat": reason1,
        "paired_gain_lose": [g, l], "paired_pvalue_exact": exact_mcnemar(g, l),
        "note_rate_wilson95": wilson(note1, n), "reason_rate_wilson95": wilson(reason1, n),
        "flips_above_noise_band": sum(1 for r in all_rows if r.get("note_driven")),
        "flips_strict": sum(1 for r in all_rows if r.get("strict_flip")),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_conditions_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
