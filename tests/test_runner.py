"""Tests for the `benchmaxxing run` entry point (runner wiring + CLI).

Everything here runs offline: the stages are driven with ``gateway.MockBackend`` through
``--backend mock``, so no key and no real data are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmaxxing import cli, runner
from benchmaxxing.config import Config
from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Modality

MANIFEST_ROWS = [
    Case(
        case_id="q1",
        patient_id="p1",
        modality=Modality.TEXT,
        question="Most likely diagnosis?",
        options=("Primary spontaneous pneumothorax on the left", "Rib fracture", "Effusion"),
        answer_index=0,
        report="A 34-year-old man with sudden pleuritic chest pain.",
    ),
    Case(
        case_id="q2",
        patient_id="p2",
        modality=Modality.TEXT,
        question="Best next step?",
        options=("Needle decompression then a chest tube", "Observation", "CT angiography"),
        answer_index=0,
        report="A 51-year-old woman after a fall.",
    ),
]

CONFIG = {"models": ["mock-a", "mock-b", "otherfamily-c"], "dataset": "medqa",
          "cue_set": "text-v1", "seed": 3}


@pytest.fixture
def manifest(tmp_path) -> Path:
    return write_manifest(MANIFEST_ROWS, tmp_path / "cases.csv")


@pytest.fixture
def config_file(tmp_path) -> Path:
    # JSON, not YAML, so the CLI path is covered without the optional config extra.
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(CONFIG), encoding="utf-8")
    return path


def _run(manifest_path, out_dir, stage, extra=()):
    return cli.main(
        ["run", "--stage", stage, "--manifest", str(manifest_path), "--out", str(out_dir),
         "--backend", "mock", *extra]
    )


def _mock_backend_for(_model_id):
    return runner.build_backend("mock", kind="mock")


# --- the wiring pieces ---------------------------------------------------------------------


def test_lineage_inferred_from_the_model_id():
    assert runner._lineage("gemini-2.5-flash") == "gemini"
    assert runner._lineage("qwen2.5-72b-instruct") == "qwen"
    assert runner._lineage("llama-3.1-70b") == "llama"


def test_solo_agent_parses_a_letter_reply_into_the_option():
    agent = runner.SoloAgent(runner.build_backend("mock", kind="mock"), name="mock")
    options = ("Short one", "A considerably longer option string", "Middle")
    answer = agent.run({"question": "Which?", "options": options, "report": None})
    # the mock follows length, and the reply comes back as an option, not as a bare letter
    assert answer == options[1]


def test_solo_agent_reports_an_unparseable_reply():
    class Silent:
        def complete(self, prompt, image=None, decoding=None):
            return "I have no idea."

    agent = runner.SoloAgent(Silent(), name="silent")
    answer = agent.run({"question": "Which?", "options": ("Aorta", "Vena cava"), "report": None})
    assert answer == "unparseable"


def test_solo_agent_rejects_an_image_payload():
    agent = runner.SoloAgent(runner.build_backend("mock", kind="mock"), name="mock")
    with pytest.raises(TypeError, match="text lane"):
        agent.run(object())


def test_image_cue_set_is_refused_with_a_pointer():
    config = Config.from_dict({**CONFIG, "cue_set": "image-v1"})
    with pytest.raises(ValueError, match="experiments/imaging"):
        runner.plan_run("pilot", MANIFEST_ROWS, config)


def test_plan_counts_twins_and_calls_without_calling_a_model():
    config = Config.from_dict(CONFIG)
    plan = runner.plan_run("solo", MANIFEST_ROWS, config)
    assert plan["n_cases"] == 2
    assert plan["n_twins"] > 0
    # clean plus contaminated, once per model
    assert plan["estimated_calls"] == 2 * plan["n_twins"] * len(CONFIG["models"])
    assert plan["lineages"]["otherfamily-c"] == "otherfamily"


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="unknown stage"):
        runner.run_stage("nope", MANIFEST_ROWS, Config.from_dict(CONFIG), _mock_backend_for)


def test_empty_case_list_raises():
    with pytest.raises(ValueError, match="no cases"):
        runner.run_stage("pilot", [], Config.from_dict(CONFIG), _mock_backend_for)


def test_cascade_spreads_the_seed_only_on_a_shared_board(tmp_path):
    # The mock follows the most recent peer, so a shared board carries the planted answer and an
    # isolated one cannot. This is the shape of the arm, driven end to end with no key.
    results = runner.run_stage(
        "cascade", MANIFEST_ROWS, Config.from_dict(CONFIG), _mock_backend_for,
        transcript_dir=tmp_path / "transcripts",
    )
    assert results["shared_adoption"] > results["isolated_adoption"]
    assert results["n_cases"] == 2
    # both conditions of both cases are saved for re-analysis
    assert len(list((tmp_path / "transcripts").glob("*.jsonl"))) == 4


# --- the CLI -------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["pilot", "solo", "overlap", "cascade"])
def test_each_stage_writes_a_run_directory(manifest, config_file, tmp_path, stage):
    out = tmp_path / stage
    assert _run(manifest, out, stage, extra=["--config", str(config_file)]) == 0

    results = json.loads((out / "results.json").read_text())
    assert results["stage"] == stage
    assert results["plan"]["n_cases"] == 2
    assert results["results"]

    summary = (out / "summary.md").read_text()
    assert f"stage {stage}" in summary
    assert "| metric | value |" in summary

    assert json.loads((out / "config.json").read_text())["models"]
    manifest_json = json.loads((out / "run_manifest.json").read_text())
    assert manifest_json["cue_set_version"]
    assert manifest_json["library_versions"]["numpy"]
    assert manifest_json["dataset_revision"] == str(manifest)


def test_dry_run_writes_nothing(manifest, tmp_path, capsys):
    out = tmp_path / "dry"
    assert _run(manifest, out, "solo", extra=["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "estimated model calls:" in printed
    assert not out.exists()


def test_missing_manifest_exits_nonzero(tmp_path, capsys):
    rc = _run(tmp_path / "nope.csv", tmp_path / "out", "pilot")
    assert rc != 0
    assert "error" in capsys.readouterr().err.lower()


def test_image_cue_set_exits_nonzero_through_the_cli(manifest, tmp_path, capsys):
    config_path = tmp_path / "image.json"
    config_path.write_text(json.dumps({**CONFIG, "cue_set": "image-v1"}), encoding="utf-8")
    rc = _run(manifest, tmp_path / "out", "pilot", extra=["--config", str(config_path)])
    assert rc != 0
    assert "image" in capsys.readouterr().err.lower()


def test_config_file_drives_the_roster(manifest, config_file, tmp_path):
    out = tmp_path / "cfg-run"
    assert _run(manifest, out, "solo", extra=["--config", str(config_file)]) == 0
    saved = json.loads((out / "config.json").read_text())
    assert saved["models"] == CONFIG["models"]
    assert saved["seed"] == 3


def test_yaml_config_still_loads(manifest, tmp_path):
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    out = tmp_path / "yaml-run"
    assert _run(manifest, out, "pilot", extra=["--config", str(config_path)]) == 0
    assert json.loads((out / "config.json").read_text())["seed"] == 3
