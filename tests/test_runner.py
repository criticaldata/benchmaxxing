"""Tests for the `benchmaxxing run` entry point (runner wiring + CLI).

Everything here runs offline: the stages are driven with ``gateway.MockBackend`` through
``--backend mock``, so no key and no real data are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmaxxing import cli, runner
from benchmaxxing.analysis import FlipRecord
from benchmaxxing.config import Config
from benchmaxxing.data import write_manifest
from benchmaxxing.schema import Case, Modality
from benchmaxxing.stats import multiple_comparison

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


# --- uncertainty ---------------------------------------------------------------------------


def _flip_records(n_flipped: int, n_total: int) -> list[FlipRecord]:
    """A fixed record set: the first ``n_flipped`` twins flipped from correct to wrong."""
    records = []
    for i in range(n_total):
        flipped = i < n_flipped
        records.append(
            FlipRecord(
                case_id=f"c{i}",
                cue_type="longest_option",
                model="mock",
                clean_answer="right",
                contaminated_answer="wrong" if flipped else "right",
                ground_truth="right",
                flipped=flipped,
                clean_correct=True,
                contaminated_correct=not flipped,
            )
        )
    return records


def test_bootstrap_ci_is_pinned_for_a_fixed_record_set():
    # 4 of 12 twins flipped. The interval is the seeded percentile bootstrap over those 12
    # observations, so it is reproducible: a change here is a change in the resampling, not noise.
    est = runner.estimates("solo", {"records": _flip_records(4, 12)}, seed=0)

    flip = est["flip_rate"]
    assert flip["point"] == pytest.approx(1 / 3)
    assert flip["n"] == 12
    assert flip["ci_level"] == 0.95
    assert (flip["ci_low"], flip["ci_high"]) == pytest.approx((0.08333333, 0.58333333))

    # the effect size resamples the paired difference, not the two accuracies separately
    effect = est["shortcut_reliance"]
    assert effect["point"] == pytest.approx(1 / 3)
    assert (effect["ci_low"], effect["ci_high"]) == pytest.approx((0.08333333, 0.58333333))


def test_bootstrap_ci_is_reproducible():
    # Same records plus same seed gives the same interval, which is what makes a reported CI
    # checkable by a reviewer rather than a number that moves on every rerun.
    records = {"records": _flip_records(4, 12)}
    assert runner.estimates("solo", records, seed=0) == runner.estimates("solo", records, seed=0)


def test_solo_estimates_are_per_model_not_pooled():
    results = {"records_by_model": {"m1": _flip_records(4, 12), "m2": _flip_records(2, 12)}}
    est = runner.estimates("solo", results, seed=0)

    assert est["m1::flip_rate"]["point"] == pytest.approx(1 / 3)
    assert est["m2::flip_rate"]["point"] == pytest.approx(1 / 6)
    # pooling the two models would report n=24 for a sample of 12 shared twins
    assert {row["n"] for row in est.values()} == {12}


def test_solo_ci_resamples_by_case_not_by_twin():
    # A case is seen through several cue twins that share its difficulty, so drawing twins iid
    # counts correlated observations as independent and reports a tighter interval than the
    # case-level evidence supports. The interval must be resampled over cases instead.
    records = []
    for case in range(4):
        flipped = case < 2  # two of four cases flip on every one of their cues
        for cue in ("longest_option", "position_bias", "lexical_overlap"):
            records.append(
                FlipRecord(
                    case_id=f"case{case}",
                    cue_type=cue,
                    model="mock",
                    clean_answer="right",
                    contaminated_answer="wrong" if flipped else "right",
                    ground_truth="right",
                    flipped=flipped,
                    clean_correct=True,
                    contaminated_correct=not flipped,
                )
            )
    flip = runner.estimates("solo", {"records": records}, seed=0)["flip_rate"]
    assert flip["point"] == pytest.approx(0.5)  # 6 of 12 twins flipped, unchanged by clustering
    assert flip["n"] == 12
    assert flip["n_cases"] == 4
    # the same 12 values drawn iid, ignoring the case structure, give a strictly narrower interval
    from benchmaxxing.stats import bootstrap_ci

    _, iid_low, iid_high = bootstrap_ci(
        np.array([1.0 if r.flipped else 0.0 for r in records]), seed=0
    )
    assert (flip["ci_high"] - flip["ci_low"]) > (iid_high - iid_low)


def test_cascade_estimates_pair_shared_against_isolated():
    results = {
        "per_case": [
            {"case_id": "a", "shared_adoption": 1.0, "isolated_adoption": 0.0, "onset": 1},
            {"case_id": "b", "shared_adoption": 0.5, "isolated_adoption": 0.5, "onset": None},
        ]
    }
    est = runner.estimates("cascade", results, seed=0)
    assert est["adoption_delta"]["point"] == pytest.approx(0.5)
    assert est["adoption_delta"]["n"] == 2
    assert est["shared_adoption"]["point"] == pytest.approx(0.75)


def test_estimates_are_empty_when_there_is_nothing_to_resample():
    assert runner.estimates("cascade", {"per_case": []}, seed=0) == {}


# --- multiple comparisons ------------------------------------------------------------------


def test_family_is_one_test_per_model_per_cue():
    records = _flip_records(4, 12)
    results = {"records_by_model": {"m1": records, "m2": records}}
    tests = runner.significance_tests("solo", results)

    assert len(tests) == 2  # one cue in the fixture, two models
    assert {row["test"] for row in tests} == {"mcnemar"}
    assert all(row["discordant"] == 4 for row in tests)


def test_cascade_family_is_a_paired_permutation_on_the_adoption_gap():
    # a tie (shared == isolated) carries no sign; the six clean positive gaps drive the test
    per_case = [
        {"shared_adoption": 1.0, "isolated_adoption": 0.0},
        {"shared_adoption": 0.67, "isolated_adoption": 0.0},
        {"shared_adoption": 0.5, "isolated_adoption": 0.5},
    ] * 3
    tests = runner.significance_tests("cascade", {"per_case": per_case}, seed=0)
    assert len(tests) == 1
    row = tests[0]
    assert row["test"] == "paired_permutation"
    # the statistic is the mean shared-minus-isolated gap: the same quantity as the delta CI
    gaps = [c["shared_adoption"] - c["isolated_adoption"] for c in per_case]
    assert row["statistic"] == pytest.approx(sum(gaps) / len(gaps))
    assert row["discordant"] == 6  # the three ties do not count
    # six unanimous positive pairs clear alpha where McNemar's sign-test floor could not
    assert row["p_value"] < 0.05


def test_overlap_permutation_test_joins_the_family():
    results = {
        "records_by_model": {"m1": _flip_records(4, 12)},
        "overlap": {"p_value": 0.03, "observed_diff": 0.2, "n_models": 3},
    }
    labels = [row["label"] for row in runner.significance_tests("overlap", results)]
    assert "within vs cross lineage failure overlap" in labels


def test_correction_matches_stats_multiple_comparison():
    raw = [0.001, 0.02, 0.04, 0.3, 0.9]
    tests = [{"label": f"t{i}", "p_value": p, "test": "mcnemar"} for i, p in enumerate(raw)]
    family = runner.family_correction(tests)

    expected = multiple_comparison(raw, method="bh", alpha=0.05)
    assert [row["p_adjusted"] for row in family["tests"]] == pytest.approx(
        list(expected.pvalues_adjusted)
    )
    assert [row["reject"] for row in family["tests"]] == list(expected.reject)
    assert family["family_size"] == 5
    assert family["method"] == "bh"


def test_holm_is_available_as_the_stricter_alternative():
    raw = [0.001, 0.02, 0.04]
    tests = [{"label": f"t{i}", "p_value": p} for i, p in enumerate(raw)]
    family = runner.family_correction(tests, method="holm")
    expected = multiple_comparison(raw, method="holm")
    assert [row["p_adjusted"] for row in family["tests"]] == pytest.approx(
        list(expected.pvalues_adjusted)
    )


def test_nan_pvalue_is_dropped_with_a_note_not_silently():
    # stats.multiple_comparison refuses a nan (one would wipe out every BH rejection), so the
    # family drops it and says so, and the family size counts only the tests actually corrected.
    tests = [
        {"label": "real", "p_value": 0.01},
        {"label": "undefined overlap", "p_value": float("nan")},
        {"label": "also real", "p_value": 0.2},
    ]
    family = runner.family_correction(tests)

    assert family["family_size"] == 2
    assert family["n_dropped"] == 1
    assert family["dropped"] == [
        {"label": "undefined overlap", "reason": "p-value is not finite"}
    ]
    assert [row["label"] for row in family["tests"]] == ["real", "also real"]


def test_empty_family_is_reported_not_crashed():
    family = runner.family_correction([])
    assert family["family_size"] == 0
    assert family["tests"] == []

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
    # every headline estimate ships with its interval
    assert "95% percentile bootstrap" in summary
    assert "| metric | estimate | 95% CI | n |" in summary
    assert results["estimates"]
    assert all(
        {"point", "ci_low", "ci_high", "n"} <= set(row) for row in results["estimates"].values()
    )

    # the family of tests is corrected once, and its size is stated
    family = results["family_correction"]
    assert "Family size:" in summary
    assert "| test | raw p | adjusted p | reject |" in summary
    assert family["family_size"] == len(family["tests"])
    assert all("p_adjusted" in row and "reject" in row for row in family["tests"])
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
