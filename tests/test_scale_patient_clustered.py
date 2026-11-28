"""experiments/imaging/scale_patient_clustered.py: rows a runner marked ``err`` are counted, not lost.

The committed summaries are pinned to the script in tests/test_nih_scale_500.py. These tests cover
what that pin cannot: a tiny synthetic tree with errored rows in it, where the count has to be
non-zero, and a recount of the committed rows files against the committed ``n_errored``.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/imaging/scale_patient_clustered.py"
OUT = ROOT / "experiments/imaging/results_n500"
R = ROOT / "experiments/imaging/results"
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


@pytest.mark.parametrize("model", [None, "some-lab/some-model"])
def test_errored_rows_are_set_aside_and_counted_per_cohort_and_rows_file(tmp_path, monkeypatch, capsys, model):
    mod = _load()
    slug = model.replace("/", "_") if model else None
    r, out, out_dir = _tree(tmp_path, slug)
    monkeypatch.setattr(mod, "R", r)
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "N_BOOT", 200)
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
    }
    for cohort, paths in files.items():
        for path in paths:
            n_rows, n_err = _recount(path)
            assert summary["n_errored"][cohort][path.stem] == n_err == 0, (cohort, path.name)
    # so the films behind the intervals are every row of the two 500-film files they are computed from
    assert summary["n_films"] == _recount(out_dir / "imaging_cascade_scale.jsonl")[0] == 500
    assert summary["hidden_note"]["n_films"] == _recount(out_dir / "matched_baseline_reason.jsonl")[0] == 500
