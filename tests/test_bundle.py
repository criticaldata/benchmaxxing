"""Tests for the per-run reproducibility bundle (benchmaxxing.bundle + `benchmaxxing report`).

Offline: bundles are produced with the mock backend, and the replay path reads only what is on
disk, which is the point of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmaxxing import bundle, cli
from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Modality

CASES = [
    Case(
        case_id=f"q{i}",
        patient_id=f"p{i}",
        modality=Modality.TEXT,
        question="Most likely diagnosis?",
        options=("Primary spontaneous pneumothorax on the left", "Rib fracture", "Effusion"),
        answer_index=0,
        report="A 34-year-old man with sudden pleuritic chest pain.",
    )
    for i in range(3)
]

CONFIG = {"models": ["mock-a", "mock-b", "mock-c"], "dataset": "medqa", "cue_set": "text-v1",
          "seed": 1}


@pytest.fixture
def cascade_bundle(tmp_path) -> Path:
    manifest = write_manifest(CASES, tmp_path / "cases.csv")
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    out = tmp_path / "bundle"
    rc = cli.main(
        ["run", "--stage", "cascade", "--manifest", str(manifest), "--config", str(config),
         "--out", str(out), "--backend", "mock"]
    )
    assert rc == 0
    return out


def test_bundle_holds_every_file_and_they_parse(cascade_bundle):
    for name in bundle.BUNDLE_FILES:
        assert (cascade_bundle / name).is_file(), name
    assert (cascade_bundle / "transcripts").is_dir()

    loaded = bundle.load_bundle(cascade_bundle)
    assert loaded["results"]["stage"] == "cascade"
    assert loaded["config"]["models"] == CONFIG["models"]
    assert loaded["summary"].startswith("# benchmaxxing run")


def test_versions_records_the_environment(cascade_bundle):
    versions = json.loads((cascade_bundle / "versions.json").read_text())
    assert versions["benchmaxxing"]
    assert versions["python"]
    assert versions["platform"]
    assert versions["libraries"]["numpy"]
    assert versions["git_sha"]  # 'unknown' outside a checkout, but always present


def test_incomplete_bundle_names_what_is_missing(tmp_path):
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="results.json"):
        bundle.load_bundle(tmp_path / "half")


def test_replay_reproduces_the_reported_cascade_numbers(cascade_bundle):
    # The whole point of committing transcripts: the headline numbers are recomputable from
    # disk, with no model calls, so a reviewer can check them instead of trusting them.
    replayed = bundle.replay_cascade(cascade_bundle)
    reported = json.loads((cascade_bundle / "results.json").read_text())["results"]

    assert replayed["n_cases"] == reported["n_cases"]
    for key in ("shared_adoption", "isolated_adoption", "adoption_delta"):
        assert replayed[key] == pytest.approx(reported[key])


def test_replay_refuses_a_non_cascade_bundle(tmp_path):
    manifest = write_manifest(CASES, tmp_path / "cases.csv")
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    out = tmp_path / "solo"
    assert cli.main(
        ["run", "--stage", "solo", "--manifest", str(manifest), "--config", str(config),
         "--out", str(out), "--backend", "mock"]
    ) == 0
    with pytest.raises(ValueError, match="cascade bundle"):
        bundle.replay_cascade(out)


def test_report_renders_from_the_bundle_alone(cascade_bundle):
    html = bundle.render_bundle_report(cascade_bundle)
    assert html == cascade_bundle / "report.html"
    text = html.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    for section in ("plan", "estimates", "environment", "run manifest"):
        assert section in text


def test_report_command_replays_and_reports(cascade_bundle, capsys):
    rc = cli.main(["report", str(cascade_bundle), "--replay"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "report:" in out
    assert "recomputed from transcripts, no model calls" in out
    assert "shared_adoption: reported" in out


def test_replay_reports_cases_whose_transcripts_are_missing(cascade_bundle):
    # Drop one case's transcript pair, as a partial copy or a pruned directory would: the replay
    # must name it as skipped rather than silently recomputing over the survivors.
    for path in (cascade_bundle / "transcripts").glob("q0*"):
        path.unlink()
    replayed = bundle.replay_cascade(cascade_bundle)
    reported = json.loads((cascade_bundle / "results.json").read_text())["results"]
    assert replayed["skipped"] == ["q0"]
    assert replayed["n_cases"] == reported["n_cases"] - 1


def test_report_flags_a_bundle_that_lost_transcripts(cascade_bundle, capsys):
    # The means over the surviving subset can still match (mock adoption is pinned), so the coverage
    # shortfall is what has to fail the check -- otherwise a partial bundle certifies numbers whose
    # evidence it no longer holds and exits 0.
    for path in (cascade_bundle / "transcripts").glob("q0*"):
        path.unlink()
    rc = cli.main(["report", str(cascade_bundle), "--replay"])
    out = capsys.readouterr().out
    assert rc != 0
    assert "coverage:" in out
    assert "MISMATCH" in out


def test_report_command_on_a_missing_bundle_exits_nonzero(tmp_path, capsys):
    rc = cli.main(["report", str(tmp_path / "nope")])
    assert rc != 0
    assert "error" in capsys.readouterr().err.lower()


def test_bundle_dir_name_convention():
    assert bundle.bundle_dir_name("cascade", "medqa", "20260723") == "cascade-medqa-20260723"
