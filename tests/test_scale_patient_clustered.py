"""experiments/imaging/scale_patient_clustered.py: rows a runner marked ``err`` are counted, not lost,
and each cascade difference is reported beside the isolated rate and the ceiling flag that qualify it.

The committed summaries are pinned to the script in tests/test_nih_scale_500.py. These tests cover
what that pin cannot: a tiny synthetic tree where the errored count has to be non-zero and where one
cohort is at the ceiling and another is not, and a recount of the committed rows files against the
committed summaries.
"""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/imaging/scale_patient_clustered.py"
OUT = ROOT / "experiments/imaging/results_n500"
R = ROOT / "experiments/imaging/results"
N80 = ROOT / "experiments/imaging/results_n80"
CHEXPERT = ROOT / "experiments/imaging_chexpert/results"
QWEN = "Qwen/Qwen2.5-VL-72B-Instruct"


def _load():
    spec = importlib.util.spec_from_file_location("scale_patient_clustered", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _cascade_row(case_id, iso_adopt, shared_adopt):
    return {"case_id": case_id, "finding": "f", "clean": "yes", "wrong": "no",
            "iso": "no" if iso_adopt else "yes", "shared": "no" if shared_adopt else "yes",
            "iso_adopt": iso_adopt, "shared_adopt": shared_adopt}


def _err(case_id):
    return {"case_id": case_id, "err": "could not build the cued film"}


def _tree(tmp_path, slug=None):
    """Eight films from four patients, with errored rows in three of the five rows files."""
    r, out = tmp_path / "results", tmp_path / "results_n500"
    own, out_dir = (r / slug, out / slug) if slug else (r, out)
    films = [f"{p:08d}_{k:03d}.png" for p in (1, 2, 3, 4) for k in (0, 1)]
    r.mkdir(parents=True, exist_ok=True)
    (r / "nih_manifest_scaled.csv").write_text(
        "case_id,patient_id\n" + "".join(f"{f},{int(f[:8])}\n" for f in films))
    (r / "nih_manifest.csv").write_text("case_id,patient_id\n" + "".join(f"{f},{int(f[:8])}\n" for f in films[:4]))
    # 500-film cascade: films 0-5 read (isolated adopts on 2, shared on 5), films 6-7 errored
    iso = [1, 1, 0, 0, 0, 0]
    shared = [1, 1, 1, 1, 1, 0]
    _write_jsonl(out_dir / "imaging_cascade_scale.jsonl",
                 [_cascade_row(f, i, s) for f, i, s in zip(films, iso, shared)] + [_err(films[6]), _err(films[7])])
    (out_dir / "imaging_cascade_scale_summary.json").write_text(json.dumps({"model": slug or "gemini-2.5-flash"}))
    # hidden note and its matched request: film 7 errored
    _write_jsonl(out_dir / "matched_baseline_reason.jsonl",
                 [{"case_id": f, "blind_is_decoy": k < 5, "reason_is_decoy": k < 1} for k, f in enumerate(films[:7])]
                 + [_err(films[7])])
    _write_jsonl(out_dir / "imaging_blind_metric.jsonl",
                 [{"case_id": f, "named_rubric_when_drifted": False} for f in films])
    # this lineage's own 35-film cascade (here the first four films, one errored) and earlier scaled run
    _write_jsonl(own / "imaging_cascade.jsonl",
                 [_cascade_row(f, i, s) for f, i, s in zip(films[:3], iso, shared)] + [_err(films[3])])
    _write_jsonl(own / "imaging_cascade_scale.jsonl", [_cascade_row(f, i, s) for f, i, s in zip(films[:6], iso, shared)])
    return r, out, out_dir


def _point_at(monkeypatch, mod, tmp_path):
    """Aim every path the script reads at the synthetic tree; the two other cohorts have no rows there
    unless a test writes them."""
    monkeypatch.setattr(mod, "R", tmp_path / "results")
    monkeypatch.setattr(mod, "OUT", tmp_path / "results_n500")
    monkeypatch.setattr(mod, "N80", tmp_path / "results_n80")
    monkeypatch.setattr(mod, "CHEXPERT", tmp_path / "chexpert")
    monkeypatch.setattr(mod, "N_BOOT", 200)


@pytest.mark.parametrize("model", [None, "some-lab/some-model"])
def test_errored_rows_are_set_aside_and_counted_per_cohort_and_rows_file(tmp_path, monkeypatch, capsys, model):
    mod = _load()
    slug = model.replace("/", "_") if model else None
    r, out, out_dir = _tree(tmp_path, slug)
    _point_at(monkeypatch, mod, tmp_path)
    mod.main(["--model", model] if model else [])
    capsys.readouterr()

    got = json.loads((out_dir / "patient_clustered.json").read_text())
    assert got["n_errored"] == {
        "nih_500": {"imaging_cascade_scale": 2, "matched_baseline_reason": 1, "imaging_blind_metric": 0},
        "nih_35": {"imaging_cascade": 1},
        "nih_150": {"imaging_cascade_scale": 0},
    }
    # the films behind each number are the rows that carry a read: 8 rows, 2 errored, 6 used
    assert got["n_films"] == 6 and got["n_patients"] == 3
    assert got["isolated_adopt"] == round(2 / 6, 4)
    assert got["shared_adopt"] == round(5 / 6, 4)
    assert got["mcnemar_shared_vs_isolated"]["gain"] == 3 and got["mcnemar_shared_vs_isolated"]["lose"] == 0
    assert got["hidden_note"]["n_films"] == 7
    assert got["hidden_note"]["note_uptake"] == round(5 / 7, 4)
    assert got["hidden_note"]["request_only_uptake"] == round(1 / 7, 4)
    # the film that errored in the lineage's own 35-film run is left out of the film-by-film agreement
    assert got["the_35_film_cohort_within"]["n"] == 4
    assert got["the_35_film_cohort_within"]["reads_agreeing_with_committed_n35"] == {"clean": 3, "iso": 3, "shared": 3}


def test_usable_splits_a_rows_file_into_reads_and_a_count(tmp_path):
    mod = _load()
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [_cascade_row("a", 0, 1), _err("b"), _cascade_row("c", 1, 1), _err("d"), _err("e")])
    ok, n_err = mod._usable(path)
    assert [r["case_id"] for r in ok] == ["a", "c"]
    assert n_err == 3


def _recount(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return len(rows), sum("err" in r for r in rows)


_COHORTS = {  # rows directory, rows file, patient manifest: written out again here, not imported
    "nih_500": (OUT, "imaging_cascade_scale.jsonl", R / "nih_manifest_scaled.csv"),
    "nih_80": (N80, "imaging_cascade.jsonl", N80 / "nih_manifest_80.csv"),
    "chexpert_150": (CHEXPERT / "full_runs", "imaging_cascade.jsonl", CHEXPERT / "cascade_150.csv"),
}
_SLUGS = [None, QWEN.replace("/", "_")]


def _cohort_rows(cohort, slug):
    base, fname, _ = _COHORTS[cohort]
    return (base / slug if slug else base) / fname


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _patients(cohort):
    return {r["case_id"]: r["patient_id"] for r in csv.DictReader(open(_COHORTS[cohort][2]))}


@pytest.mark.parametrize("slug", [None, QWEN.replace("/", "_")])
def test_committed_runs_dropped_no_row(slug):
    """Recount every rows file the summary read: its ``n_errored`` is the count, and it is zero."""
    out_dir, own = (OUT / slug, R / slug) if slug else (OUT, R)
    summary = json.loads((out_dir / "patient_clustered.json").read_text())
    files = {
        "nih_500": [out_dir / "imaging_cascade_scale.jsonl", out_dir / "matched_baseline_reason.jsonl",
                    out_dir / "imaging_blind_metric.jsonl"],
        "nih_35": [own / "imaging_cascade.jsonl"],
        "nih_150": [own / "imaging_cascade_scale.jsonl"],
        "nih_80": [_cohort_rows("nih_80", slug)],
        "chexpert_150": [_cohort_rows("chexpert_150", slug)],
    }
    assert set(summary["n_errored"]) == set(files)
    for cohort, paths in files.items():
        for path in paths:
            n_rows, n_err = _recount(path)
            assert summary["n_errored"][cohort][path.stem] == n_err == 0, (cohort, path.name)
    # so the films behind the intervals are every row of the two 500-film files they are computed from
    assert summary["n_films"] == _recount(out_dir / "imaging_cascade_scale.jsonl")[0] == 500
    assert summary["hidden_note"]["n_films"] == _recount(out_dir / "matched_baseline_reason.jsonl")[0] == 500


# --- the cascade difference beside the isolated rate, and the ceiling flag ------------------------


def _cohort_tree(tmp_path, slug):
    """Two more cohorts for a lineage and for the default lineage, on the same films.

    nih_80: six films from three patients. The lineage adopts on all six under the board (ceiling)
    and on three alone; the default lineage on four and one. One lineage row and one default row
    carry ``err``. chexpert_150: four studies from four patients, shared adoption 2 of 4 (no ceiling).
    """
    n80, cx = tmp_path / "results_n80", tmp_path / "chexpert"
    films = [f"{p:08d}_{k:03d}.png" for p in (1, 2, 3) for k in (0, 1)] + ["00000004_000.png", "00000004_001.png"]
    n80.mkdir(parents=True, exist_ok=True)
    (n80 / "nih_manifest_80.csv").write_text("case_id,patient_id\n" + "".join(f"{f},{int(f[:8])}\n" for f in films))
    mine = [_cascade_row(f, i, 1) for f, i in zip(films[:6], [1, 1, 1, 0, 0, 0])] + [_err(films[6]), _cascade_row(films[7], 0, 1)]
    theirs = [_cascade_row(f, i, s) for f, i, s in zip(films[:6], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0])]
    theirs += [_cascade_row(films[6], 0, 1), _err(films[7])]
    _write_jsonl(n80 / slug / "imaging_cascade.jsonl", mine)
    _write_jsonl(n80 / "imaging_cascade.jsonl", theirs)
    studies = [f"patient{p:05d}/study1/view1_frontal.jpg" for p in (11, 12, 13, 14)]
    cx.mkdir(parents=True, exist_ok=True)
    (cx / "cascade_150.csv").write_text("case_id,patient_id\n" + "".join(f"{f},{f.split('/')[0]}\n" for f in studies))
    _write_jsonl(cx / "full_runs" / slug / "imaging_cascade.jsonl",
                 [_cascade_row(f, i, s) for f, i, s in zip(studies, [1, 0, 0, 0], [1, 1, 0, 0])])
    _write_jsonl(cx / "full_runs" / "imaging_cascade.jsonl",
                 [_cascade_row(f, i, s) for f, i, s in zip(studies, [1, 0, 0, 0], [1, 1, 0, 0])])


def test_each_cohort_reports_the_isolated_rate_and_flags_a_shared_rate_at_the_ceiling(tmp_path, monkeypatch, capsys):
    mod = _load()
    model = "some-lab/some-model"
    slug = model.replace("/", "_")
    _, _, out_dir = _tree(tmp_path, slug)
    # the paired block needs the default lineage's 500-film rows too: the same reads, one film different
    default_rows = [_cascade_row(f"{p:08d}_{k:03d}.png", 0, 1) for p in (1, 2, 3) for k in (0, 1)]
    _write_jsonl(tmp_path / "results_n500" / "imaging_cascade_scale.jsonl", default_rows)
    _cohort_tree(tmp_path, slug)
    _point_at(monkeypatch, mod, tmp_path)
    mod.main(["--model", model])
    capsys.readouterr()
    got = json.loads((out_dir / "patient_clustered.json").read_text())

    by = got["watermark_cascade_by_cohort"]
    assert set(by) == {"nih_500", "nih_80", "chexpert_150"}
    assert got["ceiling_threshold"] == mod.CEILING == 0.95

    # nih_80: 8 rows, 1 errored, 7 used; all 7 adopt under the board, 3 alone
    n80 = by["nih_80"]
    assert got["n_errored"]["nih_80"] == {"imaging_cascade": 1}
    assert (n80["n_films"], n80["n_patients"]) == (7, 4)
    assert (n80["isolated_adopt_films"], n80["shared_adopt_films"]) == (3, 7)
    assert n80["isolated_adopt"] == round(3 / 7, 4) and n80["shared_adopt"] == 1.0
    assert n80["contagion"] == round(4 / 7, 4)
    assert n80["shared_headroom"] == 0.0 and n80["shared_at_ceiling"] is True
    # with no headroom the difference IS one minus the isolated rate
    assert n80["contagion"] == pytest.approx(1 - n80["isolated_adopt"], abs=1e-4)
    for key in ("isolated_adopt", "shared_adopt", "contagion"):
        lo, hi = n80[f"{key}_ci95_patients"]
        assert lo <= n80[key] <= hi

    # chexpert_150: shared adoption 2 of 4 leaves headroom, so no flag
    cx = by["chexpert_150"]
    assert (cx["n_films"], cx["n_patients"]) == (4, 4)
    assert (cx["isolated_adopt"], cx["shared_adopt"], cx["contagion"]) == (0.25, 0.5, 0.25)
    assert cx["shared_headroom"] == 0.5 and cx["shared_at_ceiling"] is False

    # the 500-film cohort of the synthetic tree: 5 of 6 under the board, below the ceiling
    top = by["nih_500"]
    assert (top["isolated_adopt"], top["shared_adopt"], top["contagion"]) == (
        got["isolated_adopt"], got["shared_adopt"], got["contagion"])
    assert top["contagion_ci95_patients"] == got["ci95_patients"]
    assert top["shared_at_ceiling"] is False and got["shared_at_ceiling"] is False
    assert "at the ceiling" not in got["read"]

    # against the default lineage on the films both read: the film each side errored on drops out
    paired = got["against_default_lineage_same_films"]
    assert paired["model"] == mod.DEFAULT_MODEL
    p80 = paired["nih_80"]
    assert p80["default_lineage_rows_errored"] == 1
    assert (p80["n_films"], p80["n_patients"]) == (6, 3)
    assert p80["isolated_adopt"]["this_lineage"] == 0.5
    assert p80["isolated_adopt"]["default_lineage"] == round(1 / 6, 4)
    assert p80["isolated_adopt"]["difference"] == round(2 / 6, 4)
    assert p80["isolated_adopt"]["films_adopting_alone_in_one_lineage_only"] == {"this_lineage": 2, "default_lineage": 0}
    assert p80["shared_adopt"]["difference"] == round(2 / 6, 4)
    assert p80["contagion"]["difference"] == 0.0
    assert paired["chexpert_150"]["isolated_adopt"]["difference"] == 0.0
    assert paired["nih_500"]["n_films"] == 6
    assert paired["nih_500"]["isolated_adopt"]["difference"] == round(2 / 6, 4)


def test_the_read_flags_the_ceiling_when_the_500_films_are_at_it(tmp_path, monkeypatch, capsys):
    mod = _load()
    _, out, out_dir = _tree(tmp_path)
    films = [f"{p:08d}_{k:03d}.png" for p in (1, 2, 3, 4) for k in (0, 1)]
    _write_jsonl(out / "imaging_cascade_scale.jsonl", [_cascade_row(f, i, 1) for f, i in zip(films, [1, 1, 0, 0, 0, 0, 0, 0])])
    _point_at(monkeypatch, mod, tmp_path)
    mod.main([])
    capsys.readouterr()
    got = json.loads((out_dir / "patient_clustered.json").read_text())
    assert got["shared_at_ceiling"] is True and got["watermark_cascade_by_cohort"]["nih_500"]["shared_at_ceiling"] is True
    assert "Shared adoption is at the ceiling (1.0, 8 of 8 films)" in got["read"]
    assert "1 minus isolated adoption (0.25," in got["read"]
    assert "against_default_lineage_same_films" not in got  # the default lineage is not compared with itself


@pytest.mark.parametrize("slug", _SLUGS)
@pytest.mark.parametrize("cohort", sorted(_COHORTS))
def test_committed_cohort_blocks_recount_from_the_rows(cohort, slug):
    """Counts, rates, headroom and flag of each committed cohort block, recounted from the rows file."""
    summary = json.loads(((OUT / slug if slug else OUT) / "patient_clustered.json").read_text())
    block = summary["watermark_cascade_by_cohort"][cohort]
    rows, pid = _read(_cohort_rows(cohort, slug)), _patients(cohort)
    assert {r["case_id"] for r in rows} == set(pid), "the rows cover the cohort's manifest, film for film"
    n = len(rows)
    iso, shared = sum(r["iso_adopt"] for r in rows), sum(r["shared_adopt"] for r in rows)
    assert (block["n_films"], block["n_patients"]) == (n, len({pid[r["case_id"]] for r in rows}))
    assert (block["isolated_adopt_films"], block["shared_adopt_films"]) == (iso, shared)
    assert block["isolated_adopt"] == round(iso / n, 4)
    assert block["shared_adopt"] == round(shared / n, 4)
    assert block["contagion"] == round((shared - iso) / n, 4)
    assert block["shared_headroom"] == round(1 - shared / n, 4)
    assert block["shared_at_ceiling"] is (shared / n >= summary["ceiling_threshold"])
    # the identity the flag rests on: the difference is one minus the isolated rate, less the headroom
    assert block["contagion"] == pytest.approx(1 - block["isolated_adopt"] - block["shared_headroom"], abs=1.5e-4)
    for key in ("isolated_adopt", "shared_adopt", "contagion"):
        lo, hi = block[f"{key}_ci95_patients"]
        assert lo <= block[key] <= hi
    if cohort == "nih_500":
        assert (block["shared_adopt"], block["isolated_adopt"], block["contagion"]) == (
            summary["shared_adopt"], summary["isolated_adopt"], summary["contagion"])
        assert block["contagion_ci95_patients"] == summary["ci95_patients"]
        assert block["shared_at_ceiling"] is summary["shared_at_ceiling"]
        assert ("Shared adoption is at the ceiling" in summary["read"]) is summary["shared_at_ceiling"]


def test_committed_qwen_cohorts_are_all_at_the_ceiling_and_gemini_only_on_the_80_films():
    flags = {}
    for slug in _SLUGS:
        summary = json.loads(((OUT / slug if slug else OUT) / "patient_clustered.json").read_text())
        flags[slug] = {k: v["shared_at_ceiling"] for k, v in summary["watermark_cascade_by_cohort"].items()}
    assert flags[_SLUGS[1]] == {"nih_500": True, "nih_80": True, "chexpert_150": True}
    assert flags[None] == {"nih_500": False, "nih_80": True, "chexpert_150": False}


@pytest.mark.parametrize("cohort", sorted(_COHORTS))
def test_committed_qwen_against_gemini_recounts_from_both_lineages_rows(cohort):
    summary = json.loads((OUT / _SLUGS[1] / "patient_clustered.json").read_text())
    paired = summary["against_default_lineage_same_films"]
    assert paired["model"] == "gemini-2.5-flash"
    block = paired[cohort]
    mine = {r["case_id"]: r for r in _read(_cohort_rows(cohort, _SLUGS[1]))}
    theirs = {r["case_id"]: r for r in _read(_cohort_rows(cohort, None))}
    assert set(mine) == set(theirs) and block["n_films"] == len(mine) and block["default_lineage_rows_errored"] == 0
    n = len(mine)
    reads = {"isolated_adopt": lambda r: r["iso_adopt"], "shared_adopt": lambda r: r["shared_adopt"],
             "contagion": lambda r: r["shared_adopt"] - r["iso_adopt"]}
    for key, read in reads.items():
        a, b = sum(read(r) for r in mine.values()), sum(read(r) for r in theirs.values())
        assert block[key]["this_lineage"] == round(a / n, 4)
        assert block[key]["default_lineage"] == round(b / n, 4)
        assert block[key]["difference"] == round((a - b) / n, 4)
        lo, hi = block[key]["difference_ci95_patients"]
        assert lo <= block[key]["difference"] <= hi
        own = summary["watermark_cascade_by_cohort"][cohort][key]
        assert block[key]["this_lineage"] == own
    only = block["isolated_adopt"]["films_adopting_alone_in_one_lineage_only"]
    assert only == {"this_lineage": sum(1 for k in mine if mine[k]["iso_adopt"] and not theirs[k]["iso_adopt"]),
                    "default_lineage": sum(1 for k in mine if theirs[k]["iso_adopt"] and not mine[k]["iso_adopt"])}
    assert block["isolated_adopt"]["difference"] == round((only["this_lineage"] - only["default_lineage"]) / n, 4)
