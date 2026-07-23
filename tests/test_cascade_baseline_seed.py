"""Tests for the baseline-relative cascade seed target and the stimulus-strength ablation (#104).

The first real cascade run was a null because the first-distractor seed coincided with the
committee's own independent answer in 16/20 cases: when the seed equals the baseline there is no
shared-vs-isolated gap and the contagion metric is undefined. These tests pin the fix: choosing a
seed the committee would NOT independently give, so contagion is measurable, while leaving the
default (first-distractor, bare) behavior byte-for-byte unchanged.

Everything runs offline with a deterministic conform-to-last backend.
"""

from __future__ import annotations

import pytest

from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.experiments import (
    _seed_adoption,
    run_cascade,
    run_cascade_ablation,
    terse_seed_content,
)
from benchmaxxing.schema import Case, Committee, Condition, Modality, ModelSpec, Transcript, Turn


class ConformBackend:
    """Echo the most recent committed answer, else answer independently with ``own``.

    In shared mode a later agent adopts a planted shortcut; in isolated mode it never sees it.
    """

    def __init__(self, name: str, own):
        self.name = name
        self.own = own

    def respond(self, view) -> AgentResponse:
        prior = [t for t in view.visible_turns if t.answer is not None]
        if prior:
            return AgentResponse(content=f"{self.name} conforms", answer=prior[-1].answer)
        return AgentResponse(content=f"{self.name} independent", answer=self.own)


def _case(case_id: str = "c1") -> Case:
    # Correct option is "A"; the committee's independent answer will be the distractor "B".
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question="Which is correct?",
        options=("A", "B", "C"),
        answer_index=0,
    )


def _committee(*names: str) -> Committee:
    return Committee(members=tuple(ModelSpec(name=nm, lineage="gemini") for nm in names))


def _backend_for(own="B"):
    def factory(spec: ModelSpec) -> ConformBackend:
        return ConformBackend(spec.name, own=own)

    return factory


# --------------------------------------------------------------------------- default unchanged


def test_default_is_first_distractor_and_carries_no_baseline_or_content():
    committee = _committee("m0", "m1", "m2")
    res = run_cascade(committee, _case(), _backend_for(own="A"), seed_index=1, rounds=3)
    # Default target and bare seed are unchanged; the new keys are inert.
    assert res["seed_target"] == "first_distractor"
    assert res["seeded_answer"] == "B"  # first non-correct option
    assert res["baseline_answer"] is None
    assert res["seed_content"] is None


def test_unknown_seed_target_raises():
    committee = _committee("m0", "m1")
    with pytest.raises(ValueError):
        run_cascade(committee, _case(), _backend_for(), seed_target="nonsense")


# ------------------------------------------------------------------- baseline-relative selection


def test_baseline_relative_seed_differs_from_baseline_and_truth():
    committee = _committee("m0", "m1", "m2")
    case = _case()
    res = run_cascade(committee, case, _backend_for(own="B"),
                      seed_index=1, rounds=3, seed_target="baseline_relative")
    # The committee would independently say "B"; the seed must be something else, and never truth.
    assert res["baseline_answer"] == "B"
    assert res["seeded_answer"] == "C"
    assert res["seeded_answer"] != res["baseline_answer"]
    assert res["seeded_answer"] != case.options[case.answer_index]


def test_baseline_relative_cascades_in_shared_but_not_isolated():
    committee = _committee("m0", "m1", "m2")
    res = run_cascade(committee, _case(), _backend_for(own="B"),
                      seed_index=1, rounds=3, seed_target="baseline_relative")
    # Shared adopts the seeded "C"; isolated members keep their own "B".
    assert res["shared_answers"].count("C") > res["isolated_answers"].count("C")
    assert res["isolated_answers"].count("C") == 1  # only the seed turn itself


def test_passing_baseline_answer_skips_recompute_and_is_used():
    committee = _committee("m0", "m1")
    # A caller-supplied baseline is honored (the ablation shares one baseline across arms).
    res = run_cascade(committee, _case(), _backend_for(own="B"),
                      seed_target="baseline_relative", baseline_answer="C")
    assert res["baseline_answer"] == "C"
    # With baseline "C" excluded (and truth "A"), the only remaining option is "B".
    assert res["seeded_answer"] == "B"


def test_seed_never_alters_ground_truth():
    committee = _committee("m0", "m1", "m2")
    case = _case()
    before_options, before_index = case.options, case.answer_index
    run_cascade(committee, case, _backend_for(own="B"),
                seed_target="baseline_relative", seed_index=1, rounds=3)
    assert case.options == before_options
    assert case.answer_index == before_index


def test_baseline_relative_is_deterministic():
    committee = _committee("m0", "m1", "m2")
    a = run_cascade(committee, _case(), _backend_for(own="B"),
                    seed_target="baseline_relative", seed_index=1, rounds=3)
    b = run_cascade(committee, _case(), _backend_for(own="B"),
                    seed_target="baseline_relative", seed_index=1, rounds=3)
    assert a["seeded_answer"] == b["seeded_answer"]
    assert a["baseline_answer"] == b["baseline_answer"]
    assert a["series"] == b["series"]


# ------------------------------------------------------------------------------ content builder


def test_content_builder_wires_a_confident_claim_into_the_seed_turn():
    committee = _committee("m0", "m1", "m2")
    res = run_cascade(committee, _case(), _backend_for(own="B"),
                      seed_target="baseline_relative", seed_index=1, rounds=3,
                      content_builder=terse_seed_content)
    assert res["seed_content"] == "The answer is C."
    seed_turns = [t for t in res["shared_transcript"].turns if t.agent_id == "seed"]
    assert len(seed_turns) == 1
    assert seed_turns[0].content == "The answer is C."
    assert seed_turns[0].answer == "C"


def test_terse_seed_content_is_a_deterministic_one_liner():
    case = _case()
    assert terse_seed_content(case, "C") == "The answer is C."


# --------------------------------------------------------------------------------- ablation


def test_ablation_censors_first_distractor_but_not_baseline_relative():
    committee = _committee("m0", "m1", "m2")
    cases = [_case("c0"), _case("c1")]
    # own="B": the committee's baseline is "B", which is exactly the first distractor. The
    # first-distractor arm therefore seeds the baseline (no counterfactual gap -> censored),
    # while the baseline-relative arm seeds "C" and stays well defined.
    out = run_cascade_ablation(committee, cases, _backend_for(own="B"),
                               seed_index=1, rounds=3)

    fd = out["arms"]["first_distractor"]
    br = out["arms"]["baseline_relative"]
    assert fd["n_censored"] == len(cases)      # every first-distractor case collapses onto baseline
    assert fd["n_well_defined"] == 0
    assert fd["contagion"] is None             # nothing well defined to average
    assert br["n_censored"] == 0
    assert br["n_well_defined"] == len(cases)
    assert br["contagion"] == pytest.approx(1.0)  # shared fully adopts "C", isolated never does

    # Per-case evidence that first-distractor coincides with the baseline.
    assert all(r["seed_is_baseline"] for r in out["per_case"]["first_distractor"])
    assert all(not r["seed_is_baseline"] for r in out["per_case"]["baseline_relative"])


def test_ablation_reports_structure_for_each_arm():
    committee = _committee("m0", "m1")
    out = run_cascade_ablation(committee, [_case("c0")], _backend_for(own="B"),
                               seed_targets=("baseline_relative",))
    assert out["n_cases"] == 1
    assert set(out["arms"]) == {"baseline_relative"}
    rec = out["per_case"]["baseline_relative"][0]
    assert set(rec) >= {"case_id", "seeded_answer", "baseline_answer", "contagion", "well_defined"}


# ----------------------------------------------------------------- codex-review regressions


class ConditionBackend:
    """Answers differently by condition when independent; conforms to the last committed answer."""

    def __init__(self, name: str, clean, contaminated):
        self.name = name
        self.clean = clean
        self.contaminated = contaminated

    def respond(self, view) -> AgentResponse:
        prior = [t for t in view.visible_turns if t.answer is not None]
        if prior:
            return AgentResponse(content=f"{self.name} conforms", answer=prior[-1].answer)
        own = self.clean if view.condition == Condition.CLEAN else self.contaminated
        return AgentResponse(content=f"{self.name} independent", answer=own)


def test_baseline_is_measured_in_the_same_condition_as_the_seeded_arms():
    # The seeded arms run under CONTAMINATED; the baseline must too, or a condition-sensitive
    # backend is baselined against an answer it never gives in the measured condition.
    committee = _committee("m0", "m1", "m2")

    def backend_for(spec: ModelSpec) -> ConditionBackend:
        return ConditionBackend(spec.name, clean="B", contaminated="C")

    res = run_cascade(committee, _case(), backend_for,
                      seed_index=1, rounds=3, seed_target="baseline_relative")
    # Contaminated (measured) baseline is "C", not the CLEAN answer "B".
    assert res["baseline_answer"] == "C"
    # So the seed is chosen distinct from "C" (and from truth "A"): "B".
    assert res["seeded_answer"] == "B"


def test_genuine_none_baseline_is_honored_not_recomputed():
    # None is a real committee verdict (abstention), not the "not supplied" sentinel: passing it
    # explicitly must be used as-is, never trigger a second baseline run.
    committee = _committee("m0", "m1", "m2")
    res = run_cascade(committee, _case(), _backend_for(own="B"),
                      seed_target="baseline_relative", baseline_answer=None)
    assert res["baseline_answer"] is None
    # With baseline None and truth "A", the first option that is neither is "B" (not "C", which is
    # what a spurious recompute against baseline "B" would have produced).
    assert res["seeded_answer"] == "B"


def test_seed_adoption_counts_a_real_member_named_seed():
    # The planted turn is excluded by its ``seeded`` flag, not by a reserved "seed" agent id, so a
    # genuine member named "seed" is still counted in adoption.
    turns = [
        Turn(turn_index=0, agent_id="m0", content="", answer="B", seeded=False),
        Turn(turn_index=1, agent_id="seed", content="[seeded]", answer="C", seeded=True),
        Turn(turn_index=2, agent_id="seed", content="", answer="B", seeded=False),
        Turn(turn_index=3, agent_id="m2", content="", answer="C", seeded=False),
    ]
    transcript = Transcript(run_id="r", case_id="c", condition=Condition.CONTAMINATED, turns=turns)
    # Post-seed non-injected turns are (seed->B, m2->C): adoption of "C" is 1/2, not 1/1.
    assert _seed_adoption(transcript, "C", 1) == 0.5
