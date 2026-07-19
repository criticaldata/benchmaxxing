"""Regression tests for the adversarial-review findings (one test per confirmed defect)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing.blackboard import AgentResponse, run_committee
from benchmaxxing.data import load_cases, write_manifest
from benchmaxxing.roster import build_committee, default_roster
from benchmaxxing.schema import Case, Condition, Modality, Transcript, Turn


def test_seed_survives_pre_hook_injection():
    # Finding: a pre_hook-injected turn advanced the turn counter past seed_index, so the
    # planted seed silently never fired. The seed index now counts member slots only.
    committee = build_committee(default_roster())
    case = Case(case_id="c", patient_id="p", modality=Modality.IMAGE)

    def backend_for(spec):
        class B:
            def respond(self, view):
                return AgentResponse(content="x", answer="A", confidence=0.5)
        return B()

    def injecting_referee(state):
        # inject after every member turn, which used to consume the seed slot
        return Turn(turn_index=0, agent_id="referee", content="flag", answer=None)

    t = run_committee(committee, case, Condition.CONTAMINATED, backend_for,
                      shared=True, pre_hook=injecting_referee,
                      seed_turn=(2, "SEED", "planter"), rounds=2)
    seeded = [x for x in t.turns if x.seeded]
    assert len(seeded) == 1, "the planted seed must fire exactly once despite referee injections"
    assert seeded[0].answer == "SEED"


def test_options_with_pipes_round_trip(tmp_path):
    # Finding: '|'.join corrupted options containing a literal pipe and shifted answer_index.
    case = Case(case_id="q1", patient_id="q1", modality=Modality.TEXT,
                question="Best next step?",
                options=("Observation | serial exams", "Surgery", "Antibiotics"),
                answer_index=0)
    out = write_manifest([case], tmp_path / "m.csv")
    back = load_cases(out)[0]
    assert back.options == ("Observation | serial exams", "Surgery", "Antibiotics")
    assert back.options[back.answer_index] == "Observation | serial exams"


def test_legacy_pipe_manifests_still_parse(tmp_path):
    (tmp_path / "legacy.csv").write_text(
        "case_id,question,options,answer_index\nq1,Which?,A|B|C,1\n", encoding="utf-8")
    back = load_cases(tmp_path / "legacy.csv")[0]
    assert back.options == ("A", "B", "C") and back.answer_index == 1


def test_meta_round_trips_through_manifest(tmp_path):
    # Finding: adapters store support_devices/views/label maps in Case.meta, which the
    # manifest dropped, losing the natural-cue signal.
    case = Case(case_id="i1", patient_id="p1", modality=Modality.IMAGE, image_ref="a.jpg",
                meta={"support_devices": True, "view": "PA"})
    back = load_cases(write_manifest([case], tmp_path / "m.csv"))[0]
    assert back.meta == {"support_devices": True, "view": "PA"}


def test_cli_datasets_lists_registered_adapters(capsys):
    from benchmaxxing.cli import main
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    for name in ("chexpert", "ehr", "medqa", "mimic_cxr", "nih_cxr14"):
        assert name in out


def test_solo_evaluate_accepts_gateway_backend():
    # Finding: the docstring advertised gateway backends but _invoke had no complete() branch.
    from benchmaxxing.analysis import solo_evaluate
    from benchmaxxing.cues.text import build_text_twin
    from benchmaxxing.gateway import MockBackend

    case = Case(case_id="t", patient_id="t", modality=Modality.TEXT, question="q?",
                options=("short", "the longest correct option of them all", "mid"),
                answer_index=1)
    twins = [build_text_twin(case, "longest_option")]
    records = solo_evaluate(twins, MockBackend(), answer_fn=lambda x: x, model="mock")
    assert len(records) == 1  # no TypeError: the gateway backend drives the solo lane


def test_lineage_overlap_nan_yields_nan_pvalue():
    # Finding: an undefined observed statistic returned the SMALLEST possible p-value.
    from benchmaxxing.analysis import lineage_overlap_test
    vecs = {"m1": np.zeros(6, dtype=int), "m2": np.zeros(6, dtype=int),
            "m3": np.array([1, 0, 1, 0, 1, 0])}
    lineages = {"m1": "a", "m2": "a", "m3": "b"}
    res = lineage_overlap_test(vecs, lineages, metric="phi", n_permutations=50, seed=0)
    assert np.isnan(res["observed_diff"])
    assert np.isnan(res["p_value"]), "undefined statistic must not report a significant p"


def test_multiple_comparison_rejects_nan_input():
    # Finding: a nan p-value silently destroyed every BH rejection in the family.
    from benchmaxxing.stats import multiple_comparison
    with pytest.raises(ValueError, match="finite"):
        multiple_comparison([0.001, float("nan"), 0.02], method="bh")


def test_latch_rate_does_not_match_across_turn_boundaries():
    # Finding: joining all turns with spaces let a multi-word decoy term match across the
    # boundary between two adjacent turns, counting a reference nobody made.
    from benchmaxxing.blind_metric import latch_rate
    t = Transcript(run_id="r", case_id="c", condition=Condition.CONTAMINATED, turns=[
        Turn(turn_index=0, agent_id="a", content="the visible cable"),
        Turn(turn_index=1, agent_id="b", content="artifact is irrelevant"),
    ])
    assert latch_rate([t], ["cable artifact"]) == 0.0
    t2 = Transcript(run_id="r2", case_id="c", condition=Condition.CONTAMINATED, turns=[
        Turn(turn_index=0, agent_id="a", content="I see a cable artifact here"),
    ])
    assert latch_rate([t2], ["cable artifact"]) == 1.0


def test_transcript_serializes_numpy_answers(tmp_path):
    # Finding: a numpy-scalar answer crashed dump_transcript.
    from benchmaxxing.transcript import dump_transcript, load_transcript
    t = Transcript(run_id="r", case_id="c", condition=Condition.CLEAN, turns=[
        Turn(turn_index=0, agent_id="a", content="x", answer=np.int64(3)),
        Turn(turn_index=1, agent_id="b", content="y", answer=np.bool_(True)),
    ], committed={"a": np.int64(3), "b": np.bool_(True)})
    path = tmp_path / "t.jsonl"
    dump_transcript(t, path)
    back = load_transcript(path)
    assert back.turns[0].answer == 3
    assert back.turns[1].answer is True
    assert back.committed == {"a": 3, "b": True}


def test_fisher_docstring_states_sample_odds_ratio():
    from benchmaxxing import stats
    assert "unconditional" in (stats.fisher_exact.__doc__ or "")
