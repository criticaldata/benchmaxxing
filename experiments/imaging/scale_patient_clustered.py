"""Patient-clustered read of the NIH watermark cascade on the 500-film, 132-patient manifest.

No API calls: reads the per-case rows imaging_scale.py wrote into results_n500/ and the manifest's
patient ids. Reports the image-level contagion (shared adopt minus isolated adopt), an exact McNemar
on the paired reads, and a 95% interval that resamples patients rather than films
(benchmaxxing.stats.cluster_bootstrap_ci, 20,000 resamples, as the paper's methods). Per patient, the
contagion and how many patients carry a positive, zero or negative value; the 35-film cohort's
films inside the 500 as a subgroup; and, for the 150 films the July run also read, how often the
current reads agree with that run, whose watermark was drawn before the cue font was vendored.

``--model`` reads another lineage's rows from results_n500/<slug>/ and writes its summary there; its
35-film and 150-film comparisons are that lineage's own committed n=35 and 150-film runs under
results/<slug>/. The default is the committed gemini-2.5-flash run.

A row a runner marked with ``err`` carries no read. Such rows are set aside before any rate or
interval is computed, in every rows file read here, and counted: ``n_errored`` gives the count per
cohort and per rows file, so the number of films behind each interval is checkable against the file.

``watermark_cascade_by_cohort`` puts the lineage's watermark cascades side by side: the 500 NIH films,
the 80-film NIH referee cohort and the CheXpert 150, each with isolated adoption, shared adoption and
their difference under the same patient resampling. The difference is 1 minus isolated adoption, less
the share of films that do not adopt under the board (``shared_headroom``). Once shared adoption is at
the ceiling (CEILING or above) that share is all but gone, so the difference carries isolated adoption
and nothing else: ``shared_at_ceiling`` flags it, and cohorts or lineages are then to be compared on
isolated adoption, not on the difference. For a ``--model`` lineage,
``against_default_lineage_same_films`` makes that comparison film by film against the committed
gemini-2.5-flash rows, again resampling patients.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmaxxing.stats import bootstrap_ci, cluster_bootstrap_ci, mcnemar

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "experiments/imaging/results"
OUT = ROOT / "experiments/imaging/results_n500"
N80 = ROOT / "experiments/imaging/results_n80"
CHEXPERT = ROOT / "experiments/imaging_chexpert/results"
N_BOOT = 20000
CEILING = 0.95  # shared adoption at or above this leaves the cascade difference no room to vary


def _jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def _usable(p):
    """The rows of a per-film file that carry a read, and how many the runner marked ``err``."""
    rows = _jsonl(p)
    ok = [r for r in rows if "err" not in r]
    return ok, len(rows) - len(ok)


def _patients(manifest):
    return {r["case_id"]: r["patient_id"] for r in csv.DictReader(open(manifest))}


def _cohorts():
    """The watermark cascades with per-film rows: rows directory (a ``--model`` lineage's rows sit in
    its ``<slug>/``), rows file, and the manifest that names each film's patient."""
    return {
        "nih_500": (OUT, "imaging_cascade_scale.jsonl", R / "nih_manifest_scaled.csv"),
        "nih_80": (N80, "imaging_cascade.jsonl", N80 / "nih_manifest_80.csv"),
        "chexpert_150": (CHEXPERT / "full_runs", "imaging_cascade.jsonl", CHEXPERT / "cascade_150.csv"),
    }


def _cascade(rows, pid):
    """Isolated adoption, shared adoption and their difference on one cohort, patients resampled."""
    clusters = [pid[r["case_id"]] for r in rows]
    iso = np.array([r["iso_adopt"] for r in rows], dtype=float)
    shared = np.array([r["shared_adopt"] for r in rows], dtype=float)
    out = {"n_films": len(rows), "n_patients": len(set(clusters)),
           "isolated_adopt_films": int(iso.sum()), "shared_adopt_films": int(shared.sum())}
    for key, values in (("isolated_adopt", iso), ("shared_adopt", shared), ("contagion", shared - iso)):
        point, lo, hi = cluster_bootstrap_ci(values, clusters, n_boot=N_BOOT, seed=0)
        out[key] = round(point, 4)
        out[f"{key}_ci95_patients"] = [round(lo, 4), round(hi, 4)]
    # contagion = (1 - isolated adoption) - (1 - shared adoption): the second term is the headroom
    out["shared_headroom"] = round(1 - float(shared.mean()), 4)
    out["shared_at_ceiling"] = bool(shared.mean() >= CEILING)
    return out


def _against(rows, ref_rows, pid):
    """This lineage minus the default lineage, film by film on the films both read, patients resampled."""
    ref = {r["case_id"]: r for r in ref_rows}
    both = [r for r in rows if r["case_id"] in ref]
    clusters = [pid[r["case_id"]] for r in both]
    out = {"n_films": len(both), "n_patients": len(set(clusters))}
    reads = {"isolated_adopt": lambda r: r["iso_adopt"], "shared_adopt": lambda r: r["shared_adopt"],
             "contagion": lambda r: r["shared_adopt"] - r["iso_adopt"]}
    for key, read in reads.items():
        mine = np.array([read(r) for r in both], dtype=float)
        theirs = np.array([read(ref[r["case_id"]]) for r in both], dtype=float)
        point, lo, hi = cluster_bootstrap_ci(mine - theirs, clusters, n_boot=N_BOOT, seed=0)
        out[key] = {"this_lineage": round(float(mine.mean()), 4), "default_lineage": round(float(theirs.mean()), 4),
                    "difference": round(point, 4), "difference_ci95_patients": [round(lo, 4), round(hi, 4)]}
    out["isolated_adopt"]["films_adopting_alone_in_one_lineage_only"] = {
        "this_lineage": sum(1 for r in both if r["iso_adopt"] and not ref[r["case_id"]]["iso_adopt"]),
        "default_lineage": sum(1 for r in both if ref[r["case_id"]]["iso_adopt"] and not r["iso_adopt"])}
    return out


DEFAULT_MODEL = "gemini-2.5-flash"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    model = ap.parse_args(argv).model
    out_dir, own = OUT, R  # own: where this lineage's n=35 cascade and 150-film scale rows live
    slug = None
    if model != DEFAULT_MODEL:
        slug = model.replace("/", "_")
        out_dir, own = OUT / slug, R / slug
    errored = {}  # cohort -> rows file -> rows set aside because the runner marked them err

    def usable(cohort, path):
        ok, n_err = _usable(path)
        errored.setdefault(cohort, {})[Path(path).stem] = n_err
        return ok

    rows = usable("nih_500", out_dir / "imaging_cascade_scale.jsonl")
    m = usable("nih_500", out_dir / "matched_baseline_reason.jsonl")
    blind = usable("nih_500", out_dir / "imaging_blind_metric.jsonl")
    pid = _patients(R / "nih_manifest_scaled.csv")
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
    n35 = {r["case_id"]: r for r in usable("nih_35", own / "imaging_cascade.jsonl")}
    sub35 = [r for r in sub if r["case_id"] in n35]
    agree35 = {k: sum(r[k] == n35[r["case_id"]][k] for r in sub35) for k in ("clean", "iso", "shared")}
    iso_flips = {"to_adopt": sum(1 for r in sub35 if r["iso_adopt"] and not n35[r["case_id"]]["iso_adopt"]),
                 "to_hold": sum(1 for r in sub35 if n35[r["case_id"]]["iso_adopt"] and not r["iso_adopt"])}
    july = {r["case_id"]: r for r in usable("nih_150", own / "imaging_cascade_scale.jsonl")}
    both = [r for r in rows if r["case_id"] in july]
    agree = {k: sum(r[k] == july[r["case_id"]][k] for r in both) for k in ("clean", "iso", "shared")}

    out = {
        "n_films": n, "n_patients": len(by_p), "n_errored": errored, "model": rows and json.loads(
            (out_dir / "imaging_cascade_scale_summary.json").read_text())["model"],
        "new_api_calls_this_run": 0,
        "shared_adopt": round(sum(r["shared_adopt"] for r in rows) / n, 4),
        "isolated_adopt": round(sum(r["iso_adopt"] for r in rows) / n, 4),
        "contagion": round(point, 4),
        "shared_at_ceiling": bool(sum(r["shared_adopt"] for r in rows) / n >= CEILING),
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
            "isolated_adopt": round(sum(r["iso_adopt"] for r in sub) / len(sub), 4) if sub else None,
            "reads_agreeing_with_committed_n35": agree35,
            "isolated_flips_vs_committed_n35": iso_flips},
        "july_150_overlap": {"n": len(both), "reads_agreeing": agree,
                             "july_contagion": round(sum(july[r["case_id"]]["shared_adopt"] - july[r["case_id"]]["iso_adopt"] for r in both) / len(both), 4),
                             "current_contagion_same_films": round(sum(r["shared_adopt"] - r["iso_adopt"] for r in both) / len(both), 4)},
    }
    # the lineage's watermark cascades side by side, and against the default lineage on the same films
    by_cohort, against = {}, {"model": DEFAULT_MODEL}
    for name, (base, fname, manifest) in _cohorts().items():
        path = (base / slug if slug else base) / fname
        if not path.exists():
            continue
        c_rows, c_pid = (rows, pid) if name == "nih_500" else (usable(name, path), _patients(manifest))
        by_cohort[name] = _cascade(c_rows, c_pid)
        if slug and (base / fname).exists():
            ref_rows, ref_errored = _usable(base / fname)
            against[name] = {"default_lineage_rows_errored": ref_errored, **_against(c_rows, ref_rows, c_pid)}
    out["ceiling_threshold"] = CEILING
    out["watermark_cascade_by_cohort"] = by_cohort
    if slug:
        out["against_default_lineage_same_films"] = against
    # hidden-rubric arm and its matched no-note control on the same 500 films
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
        "named_rubric_when_drifted": sum(bool(r.get("named_rubric_when_drifted")) for r in blind),
    }
    if model == DEFAULT_MODEL:
        out["read"] = (
            f"On {n} films from {len(by_p)} patients the watermark contagion is {out['contagion']:+.3f}; resampling "
            f"patients gives {out['ci95_patients']}, against {out['ci95_films']} resampling films. The 35-film "
            f"cohort inside the 500 reproduces its committed rates, but film by film the isolated reads differ "
            f"from the committed n=35 run on {len(sub35) - agree35['iso']} films ({iso_flips['to_adopt']} now "
            f"adopting, {iso_flips['to_hold']} now holding) while the shared reads agree on {agree35['shared']} "
            f"of {len(sub35)}: the rates match by cancellation, not film-stable isolated adoption.")
    else:
        out["read"] = (
            f"On {n} films from {len(by_p)} patients the watermark contagion is {out['contagion']:+.3f}; resampling "
            f"patients gives {out['ci95_patients']}, against {out['ci95_films']} resampling films. Inside the 500, "
            f"the 35-film cohort's isolated reads differ from this lineage's committed n=35 run on "
            f"{len(sub35) - agree35['iso']} of {len(sub35)} films ({iso_flips['to_adopt']} now adopting, "
            f"{iso_flips['to_hold']} now holding) and its shared reads agree on {agree35['shared']}; on the "
            f"{len(both)} films of this lineage's earlier 150-film run, contagion was {out['july_150_overlap']['july_contagion']} "
            f"then and is {out['july_150_overlap']['current_contagion_same_films']} now.")
    top = by_cohort["nih_500"]
    if top["shared_at_ceiling"]:
        also = [k for k, v in by_cohort.items() if k != "nih_500" and v["shared_at_ceiling"]]
        out["read"] += (
            f" Shared adoption is at the ceiling ({top['shared_adopt']}, {top['shared_adopt_films']} of {n} films), "
            f"so this difference is 1 minus isolated adoption ({top['isolated_adopt']}, resampling patients "
            f"{top['isolated_adopt_ci95_patients']}) to within {top['shared_headroom']}"
            + (f"; shared adoption is also at the ceiling on {' and '.join(also)}" if also else "")
            + ". Compare cohorts and lineages on isolated adoption (watermark_cascade_by_cohort), not on the difference.")
    (out_dir / "patient_clustered.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
