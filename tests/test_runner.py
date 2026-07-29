"""Tests for the `benchmaxxing run` entry point (runner wiring + CLI).

Everything here runs offline: the stages are driven with ``gateway.MockBackend`` through
``--backend mock``, so no key and no real data are involved.
"""

from __future__ import annotations

import json
import re
import warnings
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


def test_declared_lineage_wins_over_the_id_heuristic():
    # 'my-tuned-cxr' is exactly the case the prefix heuristic gets wrong: the id says nothing
    # about the family, so the arm split has to come from the config.
    config = Config.from_dict(
        {
            "models": [
                "gemini-2.5-flash",
                {"id": "my-tuned-cxr", "lineage": "llama", "tier": "8b", "open_weights": True},
            ]
        }
    )
    assert config.models == ["gemini-2.5-flash", "my-tuned-cxr"]

    specs = {spec.name: spec for spec in runner.model_specs(config)}
    assert specs["my-tuned-cxr"].lineage == "llama"
    assert specs["my-tuned-cxr"].tier == "8b"
    assert specs["my-tuned-cxr"].is_open_weights is True
    # the bare id keeps the inferred lineage
    assert specs["gemini-2.5-flash"].lineage == "gemini"
    assert runner.lineage_sources(config) == {
        "gemini-2.5-flash": "inferred",
        "my-tuned-cxr": "declared",
    }


def test_bare_id_roster_behaves_exactly_as_before():
    config = Config.from_dict({"models": ["gemini-2.5-flash", "qwen2.5-72b-instruct"]})
    assert config.roster == []
    assert [spec.lineage for spec in runner.model_specs(config)] == ["gemini", "qwen"]
    assert set(runner.lineage_sources(config).values()) == {"inferred"}


def test_org_prefixed_inferred_lineage_warns_at_run_time():
    # 'meta-llama/llama-3.1-70b' infers to 'meta' (the org), not 'llama' (the family), so it would
    # land in a different lineage than a plain 'llama-3.1-70b' and split the overlap arm wrong. The
    # heuristic stays the documented fallback, but an org-prefixed inferred id has to be loud.
    config = Config.from_dict({"models": ["meta-llama/llama-3.1-70b"]})
    with pytest.warns(UserWarning, match="org-prefixed"):
        specs = runner.model_specs(config)
    assert specs[0].lineage == "meta"  # the wart the warning is about


def test_declared_lineage_silences_the_org_prefix_warning():
    # Declaring the lineage is the fix the warning points at, so it must not also warn.
    config = Config.from_dict(
        {"models": [{"id": "meta-llama/llama-3.1-70b", "lineage": "llama"}]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        specs = runner.model_specs(config)
    assert specs[0].lineage == "llama"


def test_declared_roster_round_trips_through_the_config():
    declared = {"models": [{"id": "m", "lineage": "llama", "open_weights": True}], "seed": 2}
    config = Config.from_dict(declared)
    assert Config.from_dict(config.to_dict()) == config


def test_model_entry_without_an_id_is_rejected():
    with pytest.raises(ValueError, match="must carry an 'id'"):
        Config.from_dict({"models": [{"lineage": "llama"}]})

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


def test_solo_agent_rejects_a_payload_it_cannot_read():
    agent = runner.SoloAgent(runner.build_backend("mock", kind="mock"), name="mock")
    with pytest.raises(TypeError, match="twin payload"):
        agent.run(object())


def test_unknown_cue_set_is_refused():
    config = Config.from_dict({**CONFIG, "cue_set": "nope-v9"})
    with pytest.raises(ValueError, match="unknown cue_set"):
        runner.plan_run("pilot", MANIFEST_ROWS, config)


def test_image_cue_set_without_an_image_root_is_refused():
    config = Config.from_dict({**CONFIG, "cue_set": "image-v1"})
    with pytest.raises(ValueError, match="image-root"):
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


# --- imaging lane --------------------------------------------------------------------------

IMAGE_CONFIG = {"models": ["mock-a"], "dataset": "nih_cxr14", "cue_set": "image-v1", "seed": 5}


@pytest.fixture
def imaging(tmp_path):
    """A tiny imaging manifest plus its images: two readable, one missing, one negative label."""
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(0)
    cases = []
    for i in range(2):
        Image.fromarray(rng.integers(40, 200, size=(48, 48)).astype("uint8")).save(
            root / f"cxr{i}.png"
        )
        cases.append(
            Case(
                case_id=f"i{i}",
                patient_id=f"p{i}",
                modality=Modality.IMAGE,
                image_ref=f"cxr{i}.png",
                label="pneumothorax",
            )
        )
    cases.append(
        Case("i-missing", "p8", Modality.IMAGE, image_ref="gone.png", label="effusion")
    )
    cases.append(
        Case("i-negative", "p9", Modality.IMAGE, image_ref="cxr0.png", label="no finding")
    )
    manifest = write_manifest(cases, tmp_path / "imaging.csv")
    config = tmp_path / "img.json"
    config.write_text(json.dumps(IMAGE_CONFIG), encoding="utf-8")
    return manifest, root, config


def test_imaging_pilot_writes_a_run_directory(imaging, tmp_path):
    manifest, root, config = imaging
    out = tmp_path / "img"
    rc = _run(
        manifest, out, "pilot",
        extra=["--config", str(config), "--image-root", str(root)],
    )
    assert rc == 0

    results = json.loads((out / "results.json").read_text())
    assert results["plan"]["lane"] == "image"
    # 2 readable cases x 4 image cues
    assert results["plan"]["n_twins"] == 8
    assert (out / "summary.md").read_text().count("| ") > 0


def test_unresolvable_images_are_counted_not_fatal(imaging, tmp_path):
    manifest, root, config = imaging
    out = tmp_path / "img-skips"
    assert _run(manifest, out, "pilot",
                extra=["--config", str(config), "--image-root", str(root)]) == 0

    results = json.loads((out / "results.json").read_text())
    skipped = results["results"]["skipped_cases"]
    assert skipped["missing_image"] == 1
    assert skipped["no_finding_label"] == 1
    # a silent skip would read as coverage the run never had
    assert "Skipped cases:" in (out / "summary.md").read_text()


def test_multi_finding_case_counts_its_dropped_findings(tmp_path):
    # one question per case: a cardiomegaly|effusion|opacity case is asked only about the first
    # finding, and the other two must be counted, or the skip report claims coverage never had.
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    root = tmp_path / "images"
    root.mkdir()
    Image.fromarray(np.full((32, 32), 128, dtype="uint8")).save(root / "cxr.png")
    cases = [
        Case("m1", "p1", Modality.IMAGE, image_ref="cxr.png", label="cardiomegaly|effusion|opacity"),
        Case("s1", "p2", Modality.IMAGE, image_ref="cxr.png", label="pneumothorax"),
    ]
    twins, skipped = runner.image_twins(cases, ["watermark"], root)

    assert len(twins) == 2  # one twin per case (single cue), both asked
    assert skipped["dropped_findings"] == 2  # effusion + opacity, never asked
    assert skipped["no_finding_label"] == 0


def test_imaging_lane_needs_an_image_root(imaging, tmp_path, capsys):
    manifest, _root, config = imaging
    rc = _run(manifest, tmp_path / "out", "pilot", extra=["--config", str(config)])
    assert rc != 0
    assert "--image-root" in capsys.readouterr().err


def test_imaging_cascade_is_refused_with_a_pointer(imaging, tmp_path, capsys):
    manifest, root, config = imaging
    rc = _run(manifest, tmp_path / "out", "cascade",
              extra=["--config", str(config), "--image-root", str(root)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "text-lane only" in err and "#104" in err


def test_image_answers_parse_from_words_not_stray_letters():
    # 'yes ... (A)' must resolve to yes: the imaging options are words, so the letter heuristic
    # that serves the MCQ lane has to stay out of the way here.
    assert runner._parse_option(
        "Yes, there is a small apical pneumothorax (A).",
        runner.IMAGE_OPTIONS,
        letters_first=False,
    ) == "yes"

# --- noise floor ---------------------------------------------------------------------------


class _FlakyBackend:
    """Answers the same prompt differently every other call: a scripted non-zero noise floor."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, image=None, decoding=None):
        self.calls += 1
        letters = re.findall(r"^([A-Z])\.", prompt, flags=re.MULTILINE)
        return f"The answer is {letters[self.calls % 2]}."


def test_noise_floor_is_zero_for_a_deterministic_backend(manifest, config_file, tmp_path):
    out = tmp_path / "floor"
    assert _run(manifest, out, "solo", extra=["--config", str(config_file), "--noise-floor"]) == 0

    results = json.loads((out / "results.json").read_text())
    floors = results["results"]["noise_floor_by_model"]
    assert all(floor["rate"] == 0.0 for floor in floors.values())

    est = results["estimates"]
    model = CONFIG["models"][0]
    # with a zero floor the adjusted rate is the raw rate
    assert est[f"{model}::flip_rate_above_floor"]["point"] == pytest.approx(
        est[f"{model}::flip_rate"]["point"]
    )
    assert "noise floor" in (out / "summary.md").read_text()
    assert json.loads((out / "run_manifest.json").read_text())["config"]["noise_floor"] is True


def test_noise_floor_estimates_resample_by_case_like_the_rest_of_the_solo_lane():
    # The floor and the floor-adjusted rate are per-twin quantities on the same twins as
    # flip_rate, so they have to be clustered by case too. Otherwise the adjusted rate reports a
    # tighter interval than the raw rate it corrects, which reads as the correction buying
    # precision it did not.
    records = []
    for case in range(4):
        flipped = case < 2
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
    floor = {"rate": 0.0, "n": len(records), "per_twin": [0.0] * len(records)}
    est = runner._solo_estimates(records, seed=1, suffix="", floor=floor)
    for key in ("noise_floor", "flip_rate_above_floor"):
        # n_cases is only set when the estimate was clustered
        assert est[key]["n_cases"] == 4, key
        assert est[key]["n"] == 12, key
    raw = est["flip_rate"]
    adjusted = est["flip_rate_above_floor"]
    # a zero floor leaves the point estimate alone, and the interval is no tighter than the raw
    # rate's now that both draw the same 4 cases
    assert adjusted["point"] == pytest.approx(raw["point"])
    assert adjusted["ci_high"] - adjusted["ci_low"] >= raw["ci_high"] - raw["ci_low"] - 1e-12


def test_noise_floor_is_non_zero_for_a_flaky_backend():
    config = Config.from_dict({**CONFIG, "models": ["flaky"]})
    backend = _FlakyBackend()
    results = runner.run_stage(
        "solo", MANIFEST_ROWS, config, lambda _id: backend, noise_floor=True
    )

    floor = results["noise_floor_by_model"]["flaky"]
    assert floor["rate"] > 0.0
    assert floor["n"] == results["n_twins"]

    est = runner.estimates("solo", results, seed=0)
    raw = est["flaky::flip_rate"]["point"]
    adjusted = est["flaky::flip_rate_above_floor"]["point"]
    # the floor eats part of the raw rate, which is the whole point of measuring it
    assert adjusted == pytest.approx(raw - floor["rate"])


def test_noise_floor_is_off_by_default(manifest, config_file, tmp_path):
    out = tmp_path / "nofloor"
    assert _run(manifest, out, "solo", extra=["--config", str(config_file)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert "noise_floor_by_model" not in results["results"]
    assert json.loads((out / "run_manifest.json").read_text())["config"]["noise_floor"] is False


def test_noise_floor_pass_refuses_a_record_with_no_twin():
    with pytest.raises(ValueError, match="no twin for record"):
        runner.noise_floor_pass([], _flip_records(1, 1), object())


def test_noise_floor_matches_twins_to_records_by_key_not_position():
    # The twin list and the record list come from separate builders. If they ever fall out of the
    # same order, positional pairing re-asks the wrong case's clean payload and subtracts the wrong
    # noise. Keying by (case_id, cue_type) keeps each record matched to its own twin either way.
    class _Twin:
        def __init__(self, case_id, clean):
            self.case_id, self.cue_type, self._clean = case_id, "longest_option", clean

        def payload(self, _condition):
            return self._clean

    class _Echo:
        def run(self, payload):
            return payload  # the clean re-ask returns the case's own clean payload

    def _rec(case_id, clean):
        return FlipRecord(
            case_id=case_id, cue_type="longest_option", model="m", clean_answer=clean,
            contaminated_answer="x", ground_truth=None, flipped=True,
            clean_correct=None, contaminated_correct=None,
        )

    records = [_rec("a", "ans-a"), _rec("b", "ans-b")]
    twins = [_Twin("b", "ans-b"), _Twin("a", "ans-a")]  # reversed against the records
    floor = runner.noise_floor_pass(twins, records, _Echo())
    # each record was re-asked its OWN clean payload (== its clean_answer), so nothing moved;
    # the old positional zip would have paired a with b and reported a spurious floor of 1.0
    assert floor["per_twin"] == [0.0, 0.0]
    assert floor["rate"] == 0.0

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
    # the roster as the run used it, with where each lineage came from
    roster = manifest_json["config"]["resolved_roster"]
    assert [entry["id"] for entry in roster] == CONFIG["models"]
    assert all(entry["lineage_source"] == "inferred" for entry in roster)


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
