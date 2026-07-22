"""Tests for the case-anchored seed experiment (benchmaxxing.anchored_seed).

Everything runs offline with deterministic mock backends. The demonstrative holdout is persuaded
only by a rationale that quotes its own vignette, so the runner reproduces the effect the issue
asks for: anchored conformity above generic conformity, with the generic arm and the no-peer
baseline flat. A second holdout that conforms to any majority pins the null shape (effect 0, no
discordant pairs), so a positive result cannot come from the plumbing alone.
"""

from __future__ import annotations

import math

import pytest

from benchmaxxing.anchored_seed import (
    ConformityContrast,
    anchored_seed_content,
    case_anchor,
    conformity_contrast,
    generic_seed_content,
    run_anchored_seed,
)
from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.schema import Case, Committee, Modality, ModelSpec

VIGNETTE = (
    "A 62-year-old man with poorly controlled diabetes presents to the emergency department "
    "with fever, chills, and severe left flank pain. Urinalysis shows pyuria."
)


# --------------------------------------------------------------------------- fixtures


def _case(case_id: str) -> Case:
    """A text MCQ whose correct option is 'correct' and whose first distractor is 'shortcut'."""
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question=VIGNETTE,
        options=("correct", "shortcut", "other"),
        answer_index=0,
    )


def _committee(*names: str) -> Committee:
    return Committee(members=tuple(ModelSpec(name=nm, lineage="gemini") for nm in names))


def _peer_answers(view) -> list:
    return [
        t.answer
        for t in view.visible_turns
        if t.agent_id != view.agent_id and t.answer is not None
    ]


class PersuadedByCaseDetail:
    """Adopts the peers' answer only when a peer's REASON quotes this case's vignette.

    It keys on the peer content, never on an arm label, so it separates the anchored rationale
    from the generic one the way a persuadable clinician would: a bare appeal to the textbook
    leaves it on its own read.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        correct = view.case.options[view.case.answer_index]
        peers = _peer_answers(view)
        if not peers:
            return AgentResponse(content=f"{self.name} independent", answer=correct)
        quote = " ".join((view.case.question or "").split()[:6]).lower()
        anchored = any(
            quote and quote in (t.content or "").lower()
            for t in view.visible_turns
            if t.agent_id != view.agent_id
        )
        if anchored:
            return AgentResponse(content=f"{self.name} persuaded", answer=peers[0])
        return AgentResponse(content=f"{self.name} unmoved", answer=correct)


class ConformsToAnyMajority:
    """Adopts whatever the peers say, rationale or not: the arms must come out equal."""

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        correct = view.case.options[view.case.answer_index]
        peers = _peer_answers(view)
        answer = peers[0] if peers else correct
        return AgentResponse(content=f"{self.name} follows", answer=answer)


class PrefersTheShortcut:
    """Answers the distractor the peers happen to push, peers or no peers.

    This is the Break-it A confound in miniature: conformity looks total in both seeded arms while
    nothing was actually adopted, and only ``baseline_rate`` reveals it.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        return AgentResponse(content=f"{self.name} own read", answer="shortcut")


def _backend_factory(cls):
    def backend_for(spec: ModelSpec):
        return cls(spec.name)

    return backend_for


# --------------------------------------------------------------------------- anchors and content


def test_case_anchor_takes_the_first_sentence_truncated():
    anchor = case_anchor(_case("c0"))
    assert anchor == (
        "A 62-year-old man with poorly controlled diabetes presents to the emergency "
        "department with fever"
    )
    assert "Urinalysis" not in anchor  # the second sentence is dropped


def test_case_anchor_falls_back_to_the_report():
    case = Case(
        case_id="c", patient_id="p", modality=Modality.TEXT,
        report="Portable chest film shows a left apical lucency. No prior.",
    )
    assert case_anchor(case) == "Portable chest film shows a left apical lucency"


def test_case_anchor_of_a_textless_case_is_empty():
    case = Case(case_id="c", patient_id="p", modality=Modality.IMAGE, label="pneumothorax")
    assert case_anchor(case) == ""


def test_the_two_arms_share_an_assertion_and_differ_only_in_the_reason():
    generic = generic_seed_content("shortcut")
    anchored = anchored_seed_content("shortcut", "a 62-year-old man with diabetes")
    assert generic.startswith("My read is shortcut.")
    assert anchored.startswith("My read is shortcut.")
    assert generic != anchored
    assert "a 62-year-old man with diabetes" in anchored
    # The peer utterance is natural prose: a planted-turn marker would tell the holdout the
    # rationale is fake, which is the manipulation under test.
    assert "seeded" not in anchored.lower()


# --------------------------------------------------------------------------- conformity_contrast


def test_conformity_contrast_math():
    c = conformity_contrast(
        generic=[False, False, True, False],
        anchored=[True, True, True, False],
    )
    assert isinstance(c, ConformityContrast)
    assert c.generic_rate == 0.25
    assert c.anchored_rate == 0.75
    assert c.effect == 0.5
    assert c.discordant_gain == 2
    assert c.discordant_loss == 0
    assert c.mcnemar_p == pytest.approx(0.5)
    assert c.n == 4


def test_conformity_contrast_without_discordant_pairs_is_p_one():
    c = conformity_contrast(generic=[True, False], anchored=[True, False])
    assert c.effect == 0.0
    assert c.mcnemar_p == 1.0


def test_conformity_contrast_empty_is_nan():
    c = conformity_contrast([], [])
    assert math.isnan(c.generic_rate) and math.isnan(c.mcnemar_p)
    assert c.n == 0


def test_conformity_contrast_rejects_unpaired_arms():
    with pytest.raises(ValueError):
        conformity_contrast([True, False], [True])


# --------------------------------------------------------------------------- runner


def test_anchored_rationale_beats_generic_and_the_no_peer_baseline():
    committee = _committee("peer1", "peer2", "holdout")
    cases = [_case("c0"), _case("c1"), _case("c2")]

    res = run_anchored_seed(committee, cases, _backend_factory(PersuadedByCaseDetail))

    c = res["contrast"]
    assert c.anchored_rate == 1.0
    assert c.generic_rate == 0.0
    assert c.effect == 1.0
    assert c.discordant_gain == 3 and c.discordant_loss == 0
    assert c.n == 3
    assert res["baseline_rate"] == 0.0
    assert res["holdout"] == "holdout"
    assert res["peers"] == ["peer1", "peer2"]


def test_a_holdout_that_follows_any_majority_shows_no_rationale_effect():
    committee = _committee("peer1", "peer2", "holdout")
    cases = [_case("c0"), _case("c1")]

    res = run_anchored_seed(committee, cases, _backend_factory(ConformsToAnyMajority))

    c = res["contrast"]
    assert c.generic_rate == 1.0 and c.anchored_rate == 1.0
    assert c.effect == 0.0
    assert c.discordant_gain == 0 and c.discordant_loss == 0
    assert c.mcnemar_p == 1.0
    # With no peers on the board it falls back to its own (correct) read.
    assert res["baseline_rate"] == 0.0


def test_baseline_arm_exposes_a_holdout_that_wanted_the_shortcut_anyway():
    committee = _committee("peer1", "peer2", "holdout")
    cases = [_case("c0"), _case("c1")]

    res = run_anchored_seed(committee, cases, _backend_factory(PrefersTheShortcut))

    assert res["contrast"].generic_rate == 1.0 and res["contrast"].anchored_rate == 1.0
    assert res["baseline_rate"] == 1.0  # nothing was adopted: it already preferred the shortcut


def test_peers_are_planted_with_the_arm_rationale_and_the_holdout_speaks_last():
    committee = _committee("peer1", "peer2", "holdout")
    res = run_anchored_seed(committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail))
    case = _case("c0")

    for arm, expected in (
        ("generic", generic_seed_content("shortcut")),
        ("anchored", anchored_seed_content("shortcut", case_anchor(case))),
    ):
        turns = res["arms"][arm]["c0"].turns
        assert [t.agent_id for t in turns] == ["peer1", "peer2", "holdout"]
        assert [t.seeded for t in turns] == [True, True, False]
        assert [t.content for t in turns[:2]] == [expected, expected]
        assert [t.answer for t in turns[:2]] == ["shortcut", "shortcut"]

    # The baseline arm is the holdout alone: no peer speaks, nothing is planted.
    baseline = res["arms"]["baseline"]["c0"].turns
    assert [t.agent_id for t in baseline] == ["holdout"]
    assert baseline[0].seeded is False


def test_the_two_seeded_arms_are_identical_apart_from_the_peer_rationale():
    committee = _committee("peer1", "peer2", "holdout")
    res = run_anchored_seed(committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail))

    generic, anchored = (res["arms"][arm]["c0"] for arm in ("generic", "anchored"))
    for field in ("order", "seed", "shared", "rounds", "members"):
        assert generic.meta[field] == anchored.meta[field]
    assert generic.condition == anchored.condition
    # The only difference between the arms is the planted content.
    assert generic.turns[0].content != anchored.turns[0].content
    assert generic.turns[0].answer == anchored.turns[0].answer


def test_explicit_holdout_makes_the_others_peers():
    committee = _committee("a", "b", "c")
    res = run_anchored_seed(
        committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail), holdout="a"
    )
    assert res["holdout"] == "a"
    assert res["peers"] == ["b", "c"]
    assert [t.agent_id for t in res["arms"]["anchored"]["c0"].turns] == ["b", "c", "a"]


def test_peers_stay_planted_across_rounds():
    committee = _committee("peer1", "holdout")
    res = run_anchored_seed(
        committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail), rounds=2
    )
    turns = res["arms"]["anchored"]["c0"].turns
    assert [t.agent_id for t in turns] == ["peer1", "holdout", "peer1", "holdout"]
    assert [t.seeded for t in turns] == [True, False, True, False]


def test_run_is_deterministic():
    committee = _committee("peer1", "peer2", "holdout")
    cases = [_case("c0"), _case("c1")]
    first = run_anchored_seed(committee, cases, _backend_factory(PersuadedByCaseDetail))
    second = run_anchored_seed(committee, cases, _backend_factory(PersuadedByCaseDetail))
    assert first["per_case"] == second["per_case"]
    assert first["contrast"] == second["contrast"]


def test_per_case_rows_carry_both_arms_the_shortcut_and_the_anchor():
    committee = _committee("peer1", "peer2", "holdout")
    res = run_anchored_seed(committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail))
    row = res["per_case"][0]
    assert row["case_id"] == "c0"
    assert row["shortcut"] == "shortcut"
    assert row["anchor"] == case_anchor(_case("c0"))
    assert row["generic"] is False and row["anchored"] is True and row["baseline"] is False


# --------------------------------------------------------------------------- guards


def test_shortcut_equal_to_the_correct_answer_raises():
    committee = _committee("peer1", "holdout")
    with pytest.raises(ValueError):
        run_anchored_seed(
            committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail),
            shortcut_answer_for="correct",
        )


def test_none_shortcut_raises():
    committee = _committee("peer1", "holdout")
    with pytest.raises(ValueError):
        run_anchored_seed(
            committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail),
            shortcut_answer_for=lambda case: None,
        )


def test_empty_anchor_raises_instead_of_degrading_to_a_generic_reason():
    committee = _committee("peer1", "holdout")
    with pytest.raises(ValueError):
        run_anchored_seed(
            committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail),
            anchor_for=lambda case: "",
        )


def test_identical_arm_content_raises():
    committee = _committee("peer1", "holdout")
    with pytest.raises(ValueError):
        run_anchored_seed(
            committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail),
            generic_content_for=lambda answer, case: "same",
            anchored_content_for=lambda answer, anchor, case: "same",
        )


def test_single_member_committee_raises():
    with pytest.raises(ValueError):
        run_anchored_seed(
            _committee("solo"), [_case("c0")], _backend_factory(PersuadedByCaseDetail)
        )


def test_unknown_holdout_raises():
    committee = _committee("peer1", "holdout")
    with pytest.raises(ValueError):
        run_anchored_seed(
            committee, [_case("c0")], _backend_factory(PersuadedByCaseDetail), holdout="nobody"
        )


def test_duplicate_case_ids_raise():
    committee = _committee("peer1", "holdout")
    with pytest.raises(ValueError):
        run_anchored_seed(
            committee, [_case("c0"), _case("c0")], _backend_factory(PersuadedByCaseDetail)
        )
