"""Tests for the contaminated shared-context experiment (benchmaxxing.contaminated_context).

Everything runs offline with deterministic mock backends. The demonstrative backend anchors to
the workspace artifact and, in shared mode, conforms to peers who already anchored, so the runner
reproduces the two effects the issue asks for: present > absent (anchoring) and shared > isolated
(the shared channel amplifying the anchor beyond the solo pull).
"""

from __future__ import annotations

import math

import pytest

from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.contaminated_context import (
    AnchoringEffect,
    anchoring_effect,
    default_implied_answer,
    make_context_artifact,
    run_contaminated_context,
)
from benchmaxxing.schema import Case, Committee, Condition, Modality, ModelSpec


# --------------------------------------------------------------------------- fixtures


def _case(case_id: str, meta: dict | None = None) -> Case:
    """A text MCQ whose correct option is 'correct' and whose first distractor is 'implied'."""
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question="Which is correct?",
        options=("correct", "implied", "other"),
        answer_index=0,
        meta=meta or {},
    )


def _committee(*names: str) -> Committee:
    return Committee(members=tuple(ModelSpec(name=nm, lineage="gemini") for nm in names))


class AnchorThenConform:
    """Anchors to the workspace artifact; in shared mode conforms to peers who anchored.

    * No artifact visible -> the independent (correct) read.
    * Artifact visible and (first speaker OR a peer already on the implied answer) -> adopt implied.
    * Artifact visible but later and no peer on it (isolated followers) -> the correct read.

    The correct answer is read from the case, so the same backend serves every case.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        correct = view.case.options[view.case.answer_index]
        implied = next((t.answer for t in view.visible_turns if t.context), None)
        if implied is None:
            return AgentResponse(content=f"{self.name} independent", answer=correct, confidence=0.6)
        peer_on_implied = any(
            (not t.context) and t.agent_id != view.agent_id and t.answer == implied
            for t in view.visible_turns
        )
        if view.position == 0 or peer_on_implied:
            return AgentResponse(content=f"{self.name} anchors", answer=implied, confidence=0.6)
        return AgentResponse(content=f"{self.name} resists", answer=correct, confidence=0.6)


class ProxyChasing:
    """Adopts the artifact's implied answer only when the case's leaked proxy is high.

    Deterministic: adopts implied iff the artifact is visible AND ``case.meta['proxy'] > 0.5``,
    else the correct read. So the per-case anchoring fraction tracks the proxy, not ground truth,
    which is exactly the silent-hacking signal the blind-metric probe is meant to catch.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        correct = view.case.options[view.case.answer_index]
        implied = next((t.answer for t in view.visible_turns if t.context), None)
        if implied is not None and view.case.meta.get("proxy", 0.0) > 0.5:
            return AgentResponse(content=f"{self.name} chases proxy", answer=implied, confidence=0.6)
        return AgentResponse(content=f"{self.name} reads", answer=correct, confidence=0.6)


class ConditionKeying:
    """Adopts the implied distractor whenever the AgentView condition is CONTAMINATED.

    It never reads the board, so if presence were confounded with the condition label it would fake
    an anchoring effect. With the condition held constant across arms, it anchors equally in the
    present and absent arms, so the measured anchoring must be zero.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        correct = view.case.options[view.case.answer_index]
        if view.condition == Condition.CONTAMINATED:
            return AgentResponse(content="keys on condition", answer=default_implied_answer(view.case))
        return AgentResponse(content="reads", answer=correct)


def _backend_factory(cls):
    def backend_for(spec: ModelSpec):
        return cls(spec.name)

    return backend_for


# --------------------------------------------------------------------------- helpers


def test_default_implied_answer_picks_first_distractor():
    case = _case("c0")  # answer_index 0 -> correct='correct', first distractor='implied'
    assert default_implied_answer(case) == "implied"


def test_default_implied_answer_sentinel_without_options():
    case = Case(case_id="c", patient_id="p", modality=Modality.IMAGE, label="x")
    assert default_implied_answer(case) == "SHORTCUT"


def test_make_context_artifact_carries_flags_and_answer():
    art = make_context_artifact("IMPLIED")
    assert art.context is True and art.seeded is True
    assert art.answer == "IMPLIED"
    assert art.agent_id == "workspace"
    assert "IMPLIED" in art.content


def test_make_context_artifact_custom_text_and_id():
    art = make_context_artifact("X", text="prior cases were all X", agent_id="board")
    assert art.content == "prior cases were all X"
    assert art.agent_id == "board"


# --------------------------------------------------------------------------- anchoring_effect


def test_anchoring_effect_math():
    a = anchoring_effect(
        present_shared=[True, True, True],
        absent_shared=[False, False, False],
        present_isolated=[True, False, False],
        absent_isolated=[False, False, False],
    )
    assert isinstance(a, AnchoringEffect)
    assert a.present_shared == 1.0
    assert a.absent_shared == 0.0
    assert a.shared_anchoring == 1.0
    assert a.isolated_anchoring == pytest.approx(1 / 3)
    assert a.shared_amplification == pytest.approx(2 / 3)
    assert a.n == 3


def test_anchoring_effect_empty_arm_is_nan():
    a = anchoring_effect([], [], [], [])
    assert math.isnan(a.present_shared)
    assert math.isnan(a.shared_anchoring)
    assert a.n == 0


# --------------------------------------------------------------------------- runner: anchoring


def test_run_produces_anchoring_effect_shared_beats_isolated_beats_absent():
    committee = _committee("m1", "m2", "m3")
    cases = [_case("c0"), _case("c1")]
    res = run_contaminated_context(
        committee, cases, _backend_factory(AnchorThenConform), rounds=1, seed=0
    )
    a = res["anchoring"]

    # Present > absent (the artifact anchors) and shared > isolated (the channel amplifies it).
    assert a.present_shared == 1.0
    assert a.absent_shared == 0.0
    assert a.present_isolated == pytest.approx(1 / 3)
    assert a.absent_isolated == 0.0
    assert a.shared_anchoring == 1.0
    assert a.isolated_anchoring == pytest.approx(1 / 3)
    assert a.shared_amplification == pytest.approx(2 / 3)
    assert a.n == 6  # 2 cases x 3 members
    assert res["implied_answers"] == {"c0": "implied", "c1": "implied"}


def test_run_is_deterministic():
    committee = _committee("m1", "m2", "m3")
    cases = [_case("c0"), _case("c1")]
    res1 = run_contaminated_context(committee, cases, _backend_factory(AnchorThenConform), seed=0)
    res2 = run_contaminated_context(committee, cases, _backend_factory(AnchorThenConform), seed=0)
    assert res1["anchoring"] == res2["anchoring"]
    assert res1["referee"] == res2["referee"]


def test_run_leaves_ground_truth_intact_but_moves_decisions():
    committee = _committee("m1", "m2", "m3")
    cases = [_case("c0"), _case("c1")]
    res = run_contaminated_context(
        committee,
        cases,
        _backend_factory(AnchorThenConform),
        correct_answer_for=lambda c: c.options[c.answer_index],
    )
    rates = res["correct_rates"]
    # The artifact drives the shared committee entirely off the (unchanged) correct answer,
    # while the absent arms recover it: ground truth was never touched, only the decisions moved.
    assert rates["present_shared"] == 0.0
    assert rates["absent_shared"] == 1.0
    assert rates["present_isolated"] == pytest.approx(2 / 3)
    assert rates["absent_isolated"] == 1.0
    # The cases themselves are unchanged (answer_index still points at the correct option).
    assert all(c.answer_index == 0 for c in cases)


# --------------------------------------------------------------------------- runner: referee


def test_referee_flags_reliance_on_the_artifact():
    committee = _committee("m1", "m2", "m3")
    cases = [_case("c0"), _case("c1")]
    res = run_contaminated_context(committee, cases, _backend_factory(AnchorThenConform))

    ref = res["referee"]
    # Every member adopts the artifact in shared mode, so the referee flags full reliance.
    assert ref["present_shared"]["flagged"] is True
    assert ref["present_shared"]["reliance_rate"] == 1.0
    # In isolated mode only the first speaker anchors, so reliance is the lone-member rate.
    assert ref["present_isolated"]["reliance_rate"] == pytest.approx(1 / 3)
    # The workspace artifact is never scored as a member.
    assert set(ref["present_shared"]["flags_by_member"]) == {"m1", "m2", "m3"}


# --------------------------------------------------------------------------- runner: blind metric


def test_composes_with_blind_metric_probe_when_artifact_leaks_a_proxy():
    committee = _committee("m1", "m2", "m3")
    # Decisions (the anchoring fraction) will track the proxy; ground truth is anti-correlated.
    cases = [
        _case("c0", {"proxy": 0.9, "truth": 0.1}),
        _case("c1", {"proxy": 0.1, "truth": 0.9}),
        _case("c2", {"proxy": 0.8, "truth": 0.2}),
        _case("c3", {"proxy": 0.2, "truth": 0.8}),
    ]
    res = run_contaminated_context(
        committee,
        cases,
        _backend_factory(ProxyChasing),
        proxy_metric_for=lambda c: c.meta["proxy"],
        ground_truth_value_for=lambda c: c.meta["truth"],
    )

    bm = res["blind_metric"]
    assert bm is not None
    assert bm.n == 4
    # The committee's decisions drift toward the leaked proxy, not toward ground truth.
    assert bm.corr_hidden > bm.corr_ground
    assert bm.uptake_delta > 0.0


def test_blind_metric_is_none_without_proxy_encoders():
    committee = _committee("m1", "m2")
    res = run_contaminated_context(committee, [_case("c0")], _backend_factory(AnchorThenConform))
    assert res["blind_metric"] is None


# --------------------------------------------------------------------------- guards


def test_run_rejects_empty_cases():
    committee = _committee("m1")
    with pytest.raises(ValueError, match="at least one case"):
        run_contaminated_context(committee, [], _backend_factory(AnchorThenConform))


def test_run_rejects_empty_committee():
    committee = Committee(members=())
    with pytest.raises(ValueError, match="no members"):
        run_contaminated_context(committee, [_case("c0")], _backend_factory(AnchorThenConform))


def test_run_rejects_context_id_colliding_with_member():
    committee = _committee("workspace", "m2")
    with pytest.raises(ValueError, match="collides with a committee member"):
        run_contaminated_context(committee, [_case("c0")], _backend_factory(AnchorThenConform))


def test_run_rejects_duplicate_case_ids():
    committee = _committee("m1")
    with pytest.raises(ValueError, match="unique case_id"):
        run_contaminated_context(
            committee, [_case("c0"), _case("c0")], _backend_factory(AnchorThenConform)
        )


# --------------------------------------------------------------------------- adversarial fixes


def test_condition_held_constant_so_it_cannot_confound_anchoring():
    committee = _committee("m1", "m2", "m3")
    cases = [_case("c0"), _case("c1")]
    res = run_contaminated_context(committee, cases, _backend_factory(ConditionKeying))

    # The backend keys purely on the condition label and never reads the board. Because the
    # condition is identical across arms, it anchors in both present and absent, so the artifact's
    # measured pull is zero -- the label cannot masquerade as a shared-context effect.
    a = res["anchoring"]
    assert a.present_shared == 1.0
    assert a.absent_shared == 1.0
    assert a.shared_anchoring == 0.0
    assert a.isolated_anchoring == 0.0
    # Every arm was actually run under the same condition.
    conditions = {tr.condition for by_case in res["arms"].values() for tr in by_case.values()}
    assert conditions == {Condition.CONTAMINATED}


def test_run_rejects_none_implied_answer():
    committee = _committee("m1", "m2")
    with pytest.raises(ValueError, match="is None"):
        run_contaminated_context(
            committee, [_case("c0")], _backend_factory(AnchorThenConform),
            implied_answer_for=lambda _c: None,
        )


def test_run_rejects_implied_equal_to_correct_answer():
    committee = _committee("m1", "m2")
    with pytest.raises(ValueError, match="equals the correct answer"):
        run_contaminated_context(
            committee, [_case("c0")], _backend_factory(AnchorThenConform),
            implied_answer_for=lambda c: c.options[c.answer_index],
            correct_answer_for=lambda c: c.options[c.answer_index],
        )


def test_anchoring_effect_rejects_unequal_arms():
    with pytest.raises(ValueError, match="equal length"):
        anchoring_effect([True, False], [True], [False], [False, False, False])
