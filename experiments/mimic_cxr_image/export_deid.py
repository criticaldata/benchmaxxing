#!/usr/bin/env python3
"""Regenerate results/deid/*.csv from the per-case JSONL of the current run.

The deid CSVs are the ONLY row-level artefact a reader without PhysioNet access can check the
MIMIC-CXR imaging rates against, because the raw JSONL keys reads to MIMIC dicom_ids and stays
gitignored. They were previously produced by hand, which is how they came to sit a run behind the
paper: recomputing from them reproduced superseded numbers rather than the published ones. This
script exists so that can never be a manual step again.

Contract, unchanged from results/deid/README.md: rows are sorted on case_id and numbered 0..n-1,
case_id is then dropped, and no dicom_id, patient or study id, image_ref, report text or raw read
is emitted. Only 0/1 outcome flags keyed by an anonymous case_index.

    python3 export_deid.py            # regenerate the owned files, then verify
    python3 export_deid.py --check    # verify only; non-zero exit if a published cell drifted

Verification is the point, not a formality: --check recomputes every MIMIC-CXR cell the paper
prints FROM THE CSVs ALONE and fails loudly on drift. It does not consult the summary JSONs, so a
summary and a CSV disagreeing cannot pass.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DEID = os.path.join(RES, "deid")
# A second lineage's arms live in the model-scoped subdirectory the shared runners write, so the
# source jsonl for every spec gains one path component. Set by --model; empty means the committed
# Gemini lane, whose paths carry no model component.
SLUG = ""


def _src(src: str) -> str:
    """Resolve a SPECS source path under RES, inserting the model slug the runners scope to."""
    if not SLUG:
        return os.path.join(RES, src)
    head, tail = os.path.split(src)
    return os.path.join(RES, head, SLUG, tail) if head else os.path.join(RES, SLUG, tail)

# (output name, source jsonl, columns after case_index). Column ORDER is part of the published
# contract in README.md; do not reorder without updating the dictionary there.
SPECS = [
    ("cascade.csv", "cascade_150/imaging_cascade.jsonl",
     ["clean_correct", "iso_adopt", "shared_adopt"]),
    ("referee_cascade.csv", "referee_300/imaging_cascade.jsonl",
     ["clean_correct", "iso_adopt", "shared_adopt"]),
    ("referee.csv", "referee_300/imaging_referee.jsonl",
     ["gt", "ref_flag", "naive_flag"]),
    ("judge.csv", "referee_300/imaging_judge_referee.jsonl",
     ["clean_correct", "gt", "judge_flag", "naive_flag"]),
    ("judge_with_image.csv", "judge_with_image/imaging_judge_referee.jsonl",
     ["clean_correct", "gt", "judge_flag", "naive_flag"]),
    ("system_flag.csv", "imaging_system_flag.jsonl",
     ["iso_adopt", "shared_adopt"]),
    ("strength_cascade.csv", "imaging_strength_cascade.jsonl",
     ["op0.15_iso_adopt", "op0.15_shared_adopt", "op0.15_solo_flip",
      "op0.3_iso_adopt", "op0.3_shared_adopt", "op0.3_solo_flip",
      "op0.45_iso_adopt", "op0.45_shared_adopt", "op0.45_solo_flip"]),
]

# The solo arm writes to the results root (out=""), not to a per-arm subdirectory, so its source has
# no directory component; _src() scopes it to the model slug all the same.
# A second lineage runs every arm in one pass, so it owns the three files the committed Gemini lane
# inherited from an earlier run. These specs are used ONLY with --model; the Gemini defaults below
# stay untouched, which is why NOT_OWNED still lists them for the default path.
SPECS_SECOND_LINEAGE = [
    ("solo.csv", "imaging_solo.jsonl",
     ["clean_correct", "cable_flip", "corner_tag_flip", "watermark_flip", "laterality_flip",
      "noise_flip"]),
    ("nih_match_solo.csv", "nih_match_35/imaging_solo.jsonl",
     ["solo_case_index", "clean_correct", "cable_flip", "corner_tag_flip", "watermark_flip",
      "laterality_flip", "noise_flip"]),  # published order puts solo_case_index first
    ("blind_metric.csv", "imaging_blind_metric.jsonl",
     ["base_is_decoy", "blind_is_decoy", "aware_is_decoy", "named_rubric_when_drifted"], False),
]

# Files in results/deid/ this script does NOT own, and why. They come from arms the #393 rerun did
# not touch (solo, nih_match_solo, blind_metric already reproduced correctly and were skipped), or
# they are checksums rather than outcomes. Listed so "regenerate everything" is never assumed.
NOT_OWNED = {
    "solo.csv": "solo arm, not rerun",
    "nih_match_solo.csv": "NIH-matched solo arm, not rerun",
    "blind_metric.csv": "blind-metric arm, not rerun",
    "provenance.csv": "per-image checksums, not an outcome file",
    "README.md": "documentation",
}

# Cohort sizes are part of the published contract in README.md. A truncated file is otherwise
# invisible: 0..9 is as dense and ordered as 0..833, so only the count catches it.
EXPECTED_ROWS = {
    "solo.csv": 834, "nih_match_solo.csv": 35, "cascade.csv": 215, "referee_cascade.csv": 417,
    "system_flag.csv": 834, "strength_cascade.csv": 834, "referee.csv": 417, "judge.csv": 417,
    "judge_with_image.csv": 417, "blind_metric.csv": 141, "provenance.csv": 834,
}

# Never emit any of these, asserted rather than assumed.
FORBIDDEN = {"case_id", "dicom_id", "study_id", "subject_id", "patient_id",
             "image_ref", "report", "path", "clean", "wrong", "planted_wrong",
             "iso", "shared", "reread", "judge_verdict", "finding"}


def load(rel):
    with open(_src(rel)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def flag(row, col, where):
    """Emit an outcome as 0/1.

    The type is asserted, not coerced. int(bool(x)) on a string is silently 1 for any non-empty
    value including "no", so a future arm emitting strings would encode every row as 1 and every
    rate would read 1.0. That must fail here rather than ship.
    """
    v = row.get(col)
    if v is None:
        return ""
    assert isinstance(v, (bool, int)) and not isinstance(v, float), \
        f"{where}: column {col!r} is {type(v).__name__}, expected bool or int"
    assert int(v) in (0, 1), f"{where}: column {col!r} has non-binary value {v!r}"
    return int(v)


def derive_clean_correct(rows, src):
    """Some arms carry no clean_correct column, only the clean read and the planted wrong answer.

    On an all-finding-present cohort the plant is a constant, so `clean != wrong` IS clean_correct.
    That is asserted below against every arm that stores the column, rather than trusted.
    """
    out = []
    for r in rows:
        wrong = r.get("planted_wrong", r.get("wrong"))
        out.append(int(r["clean"] != wrong) if "clean" in r and wrong is not None else "")
    return out


def assert_derivation_holds():
    """The derivation must reproduce the stored column wherever both exist. 632 rows do."""
    checked = 0
    for rel in ("cascade_150/imaging_cascade.jsonl", "referee_300/imaging_cascade.jsonl"):
        rows = load(rel)
        if "clean_correct" not in rows[0] or "clean" not in rows[0]:
            continue
        derived = derive_clean_correct(rows, rel)
        for i, r in enumerate(rows):
            assert derived[i] == int(bool(r["clean_correct"])), \
                f"{rel} row {i}: clean != wrong does not reproduce clean_correct"
        checked += len(rows)
    assert checked, "derivation rule was never exercised; do not trust it"
    return checked


def _join_solo(rows, src):
    """The shared imaging_solo runner writes no clean_correct and keeps noise_flip in a sibling
    imaging_noise_floor.jsonl. On this all-finding-present cohort clean_correct is `clean == "yes"`
    (the same identity derive_clean_correct asserts for the cascade arms, where the plant is the
    constant "no"), and noise_flip joins on case_id. Only applied to the solo specs."""
    if "clean_correct" not in rows[0] and "clean" in rows[0] and "wrong" not in rows[0]:
        for r in rows:
            r["clean_correct"] = int(r["clean"] == "yes")
    if "noise_flip" not in rows[0]:
        nf_path = _src(os.path.join(os.path.dirname(src), "imaging_noise_floor.jsonl"))
        if os.path.exists(nf_path):
            with open(nf_path) as fh:
                nf = {json.loads(l)["case_id"]: json.loads(l)["noise_flip"] for l in fh if l.strip()}
            for r in rows:
                if r["case_id"] in nf:
                    r["noise_flip"] = nf[r["case_id"]]
    return rows


def write_one(name, src, cols, emit_derived=True):
    rows = sorted(load(src), key=lambda r: r["case_id"])
    if name in ("solo.csv", "nih_match_solo.csv"):
        rows = _join_solo(rows, src)
    if name == "nih_match_solo.csv" and "solo_case_index" not in rows[0]:
        # The published column is the film's row in solo.csv, which is the rank of its case_id in
        # the 834-film solo arm sorted the same way this writer sorts. Read the sibling solo source.
        solo_ids = sorted(r["case_id"] for r in load("imaging_solo.jsonl"))
        rank = {cid: i for i, cid in enumerate(solo_ids)}
        for r in rows:
            r["solo_case_index"] = rank[r["case_id"]]
    want = EXPECTED_ROWS.get(name)
    assert want is None or len(rows) == want, \
        f"{name}: source has {len(rows)} rows, the published contract says {want}"
    ids = [r["case_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"{name}: duplicate case_id, case_index would be ambiguous"
    derived = derive_clean_correct(rows, src) if "clean_correct" not in rows[0] else None
    if not emit_derived:
        # The published file for this arm carries no clean_correct column: on the Gemini lane its
        # source cannot derive one, and a second lineage whose source can must not add a column the
        # contract does not have.
        derived = None
    # When the column is derived, it is emitted where the spec asks for it, and only prepended if
    # the spec does not name it. A second lineage's specs name it in the published column order, so
    # its files come out with the same header as the committed Gemini ones.
    derived_in_cols = derived is not None and "clean_correct" in cols
    header = ["case_index"] + ([] if derived_in_cols or derived is None else ["clean_correct"]) + cols
    leak = FORBIDDEN & set(header)
    assert not leak, f"{name} would emit {leak}"
    with open(os.path.join(DEID, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, r in enumerate(rows):
            rec = [i] + ([] if derived_in_cols or derived is None else [derived[i]])
            rec += [derived[i] if (derived_in_cols and c == "clean_correct")
                    else (r[c] if c == "solo_case_index" else flag(r, c, f"{name} row {i}"))
                    for c in cols]
            w.writerow(rec)
    return len(rows)


def read_csv(name):
    with open(os.path.join(DEID, name)) as fh:
        return list(csv.DictReader(fh))


def prf(rows, flag_col, gt_col="gt", restrict=None):
    if restrict:
        rows = [r for r in rows if r[restrict] == "1"]
    tp = sum(1 for r in rows if r[flag_col] == "1" and r[gt_col] == "1")
    fp = sum(1 for r in rows if r[flag_col] == "1" and r[gt_col] != "1")
    fn = sum(1 for r in rows if r[flag_col] != "1" and r[gt_col] == "1")
    tn = sum(1 for r in rows if r[flag_col] != "1" and r[gt_col] != "1")
    return (round(tp / (tp + fp), 2) if tp + fp else None,
            round(tp / (tp + fn), 2) if tp + fn else None,
            round(fp / (fp + tn), 2) if fp + tn else None)


def contagion(name):
    rows = [r for r in read_csv(name) if r["clean_correct"] == "1"]
    assert rows, f"{name} has no clean-correct rows; the file is empty or its columns have changed"
    shared = sum(r["shared_adopt"] == "1" for r in rows) / len(rows)
    iso = sum(r["iso_adopt"] == "1" for r in rows) / len(rows)
    return round(shared, 2), round(iso, 2), round(shared - iso, 2), len(rows)


def sweep():
    rows = [r for r in read_csv("strength_cascade.csv") if r["clean_correct"] == "1"]
    assert rows, "strength_cascade.csv has no clean-correct rows"
    out = []
    for op in ("0.15", "0.3", "0.45"):
        shared = sum(r[f"op{op}_shared_adopt"] == "1" for r in rows) / len(rows)
        iso = sum(r[f"op{op}_iso_adopt"] == "1" for r in rows) / len(rows)
        out.append(round(shared - iso, 2))
    return tuple(out) + (len(rows),)


def referee_restricted():
    """referee.csv carries no clean_correct, so join to judge.csv on case_index.

    Both come from the same 417-row manifest in the same sorted order. Asserted, not assumed.
    """
    ref, jud = read_csv("referee.csv"), read_csv("judge.csv")
    assert len(ref) == len(jud), "referee.csv and judge.csv are not the same cohort"
    cc = {r["case_index"]: r["clean_correct"] for r in jud}
    assert set(cc) == {r["case_index"] for r in ref}, "case_index sets differ; the join is unsafe"
    return prf([r for r in ref if cc[r["case_index"]] == "1"], "ref_flag")


CHECKS = [
    ("Table 2 referee, 88 clean-correct", referee_restricted, (0.66, 0.70, 0.36)),
    ("Table 2 judge, text-only, 88", lambda: prf(read_csv("judge.csv"), "judge_flag",
                                                 restrict="clean_correct"), (0.56, 1.0, 0.77)),
    ("Table 2 gate, 88", lambda: prf(read_csv("judge.csv"), "naive_flag",
                                     restrict="clean_correct"), (0.56, 1.0, 0.77)),
    ("4.4 judge given the film, 88", lambda: prf(read_csv("judge_with_image.csv"), "judge_flag",
                                                 restrict="clean_correct"), (0.75, 0.07, 0.02)),
    ("Table 3 watermark cascade", lambda: contagion("cascade.csv"), (0.93, 0.52, 0.40, 42)),
    ("Table 3 watermark, referee cohort", lambda: contagion("referee_cascade.csv"),
     (0.89, 0.40, 0.49, 88)),
    ("Table 3 opacity sweep", sweep, (0.59, 0.58, 0.48, 170)),
    ("Table 3 system flag, no peer", lambda: contagion("system_flag.csv"), (0.77, 0.44, 0.34, 170)),
    ("4.3 referee, unrestricted 417", lambda: prf(read_csv("referee.csv"), "ref_flag"),
     (0.41, 0.59, 0.21)),
    ("4.3 gate, unrestricted 417", lambda: prf(read_csv("referee.csv"), "naive_flag"),
     (0.20, 1.0, 0.96)),
]

def solo_null():  # noqa: D401
    """Every cue within 0.029 of the noise floor at n=834. Returns the largest gap and n.

    solo.csv is not regenerated here, but the paper leans on it, so it is still checked: a stale or
    swapped file shows up as a gap that no longer clears the published bound.
    """
    rows = read_csv("solo.csv")
    assert rows, "solo.csv is empty"
    cols = [c for c in rows[0] if c.endswith("_flip")]
    assert "noise_flip" in cols, "solo.csv has no noise_flip column to use as the floor"
    rate = {c: sum(r[c] == "1" for r in rows) / len(rows) for c in cols}
    floor = rate["noise_flip"]
    return round(max(abs(v - floor) for c, v in rate.items() if c != "noise_flip"), 3), len(rows)


def self_report():
    """Drifters naming the rubric, decoy uptake, and the structural zero of the control."""
    rows = read_csv("blind_metric.csv")
    assert rows, "blind_metric.csv is empty"
    return (sum(r["named_rubric_when_drifted"] == "1" for r in rows),
            sum(r["blind_is_decoy"] == "1" for r in rows),
            sum(r["base_is_decoy"] == "1" for r in rows),
            len(rows))


# Claims from arms this script does not regenerate. Checked anyway: not owning a file is no reason
# to leave the number it backs unverified.
UNOWNED_CHECKS = [
    ("4.1 solo null, max gap to floor", solo_null, (0.029, 834)),
    ("4.5 self-report: named, decoy, control", self_report, (1, 75, 0, 141)),
]


def verify():
    needed = [n for n, _, _ in SPECS] + ["solo.csv", "blind_metric.csv"]
    missing = [n for n in needed if not os.path.exists(os.path.join(DEID, n))]
    if missing:
        print("cannot verify, these files do not exist yet:", ", ".join(missing))
        print("run without --check to generate them")
        return False
    print("recomputing every published MIMIC-CXR cell from the CSVs alone:")
    ok = True
    for name, want in sorted(EXPECTED_ROWS.items()):
        path = os.path.join(DEID, name)
        if not os.path.exists(path):
            continue
        got = len(read_csv(name))
        if got != want:
            ok = False
            print(f"  [BAD] {name} has {got} rows, the published contract says {want}")
    for label, fn, want in CHECKS:
        got = fn()
        good = all(g is not None and abs(g - w) <= 0.011 for g, w in zip(got, want))
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {label:<38} got {got}  want {want}")
    print("claims from arms this script does not regenerate, checked anyway:")
    for label, fn, want in UNOWNED_CHECKS:
        got = fn()
        # the solo null is a bound, not an equality: the gap must not EXCEED the published 0.029
        good = (got[0] <= want[0] + 1e-9 and got[1] == want[1]) if label.startswith("4.1") \
            else got == want
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {label:<38} got {got}  want {want}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    ap.add_argument("--results-dir", default=None,
                    help="results root to read arms from; default is this lane's committed results/")
    ap.add_argument("--model", default=None,
                    help="second-lineage model id; its arms are read from the model-scoped "
                         "subdirectory and written to deid/<slug>/")
    args = ap.parse_args()
    if args.results_dir:
        RES = os.path.abspath(os.path.expanduser(args.results_dir))
        DEID = os.path.join(RES, "deid")
    if args.model:
        SLUG = args.model.replace("/", "_")
        DEID = os.path.join(DEID, SLUG)
    os.makedirs(DEID, exist_ok=True)
    if args.model:
        # verify() asserts the published Gemini claims. A second lineage is expected to differ from
        # them, so the claim check is not meaningful here and is skipped rather than reported failed.
        n = assert_derivation_holds()
        print(f"clean_correct derivation reproduces the stored column on {n} rows")
        for spec in SPECS + SPECS_SECOND_LINEAGE:
            name, src, cols = spec[0], spec[1], spec[2]
            emit = spec[3] if len(spec) > 3 else True
            print(f"wrote deid/{SLUG}/{name} ({write_one(name, src, cols, emit)} rows)")
        print("skipped verify(): its expected values are the published Gemini claims")
        sys.exit(0)
    if not args.check:
        n = assert_derivation_holds()
        print(f"clean_correct derivation reproduces the stored column on {n} rows")
        for name, src, cols in SPECS:
            print(f"wrote deid/{name} ({write_one(name, src, cols)} rows)")
        untouched = sorted(set(os.listdir(DEID)) - {n for n, _, _ in SPECS})
        print("left untouched:", ", ".join(f"{f} ({NOT_OWNED.get(f, 'unknown')})" for f in untouched))
    sys.exit(0 if verify() else 1)
