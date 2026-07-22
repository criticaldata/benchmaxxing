"""Tests for the system pre-screen flag experiment (benchmaxxing.prescreen_flag).

Everything runs offline with deterministic mock backends. The demonstrative model defers to a
machine pre-classification but not to the same claim from a colleague, so the runner reproduces the
contrast the issue asks for: adoption above control under the system flag, flat under the peer note.
A model that takes any note on the board pins the null shape (authority effect 0), and a model that
already preferred the flagged option pins the control arm's job, so a positive result cannot come
from the plumbing alone.
"""

from __future__ import annotations

import math

import pytest

from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.prescreen_flag import (
    PEER_AGENT_ID,
    SYSTEM_AGENT_ID,
    FlagAdoption,
    adoption_contrast,
    flag_claim,
    peer_note_content,
    run_prescreen_flag,
    system_flag_content,
)
from benchmaxxing.schema import Case, Committee, Modality, ModelSpec

VIGNETTE = (
    "A 62-year-old man with poorly controlled diabetes presents to the emergency department "
    "with fever, chills, and severe left flank pain."
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


def _roster(*names: str) -> list[ModelSpec]:
    return [ModelSpec(name=nm, lineage="gemini") for nm in names]


def _board_notes(view) -> list:
    return [t for t in view.visible_turns if t.context]


def _own_read(view):
    """The mock's unprompted answer: the case's correct option, or a sentinel when it has no key."""
    case = view.case
    if case.options and case.answer_index is not None:
        return case.options[case.answer_index]
    return "own-read"


class TrustsTheSystem:
    """Adopts a flagged answer only when the note is sourced to the triage system.

    It keys on the note's attribution, never on an arm label, so it separates a machine
    pre-classification from a colleague's claim the way the hypothesis says a model might.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        for note in _board_notes(view):
            if note.agent_id == SYSTEM_AGENT_ID:
                return AgentResponse(content=f"{self.name} defers to triage", answer=note.answer)
        return AgentResponse(content=f"{self.name} own read", answer=_own_read(view))


class TrustsAnyNote:
    """Adopts whatever any board note asserts: the system and peer arms must come out equal."""

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        notes = _board_notes(view)
        answer = notes[0].answer if notes else _own_read(view)
        return AgentResponse(content=f"{self.name} follows the board", answer=answer)


class PrefersTheFlagged:
    """Answers the distractor the artifact happens to assert, note or no note.

    This is the Break-it A confound in miniature: adoption looks total in both artifact arms while
    nothing was actually adopted, and only the control arm reveals it.
    """

    def __init__(self, name: str):
        self.name = name

    def respond(self, view) -> AgentResponse:
        return AgentResponse(content=f"{self.name} own read", answer="shortcut")


def _backend_factory(cls):
    def backend_for(spec: ModelSpec):
        return cls(spec.name)

    return backend_for


def _backend_by_name(**classes):
    """A factory that gives each roster model its own mock class, keyed by model name."""

    def backend_for(spec: ModelSpec):
        return classes[spec.name](spec.name)

    return backend_for


# --------------------------------------------------------------------------- artifact content


def test_the_two_artifacts_share_a_claim_and_differ_only_in_the_source():
    system = system_flag_content("shortcut")
    peer = peer_note_content("shortcut")
    assert flag_claim("shortcut") == "Likely shortcut."
    assert flag_claim("shortcut") in system
    assert flag_claim("shortcut") in peer
    assert system != peer
    assert "triage system" in system and "colleague" in peer
    # Neither artifact announces itself as planted: a visible marker would tell the model the
    # signal is fake, which is the manipulation under test.
    assert "seeded" not in system.lower() and "seeded" not in peer.lower()


# --------------------------------------------------------------------------- adoption_contrast


def test_adoption_contrast_math():
    a = adoption_contrast(
        control=[False, False, True, False],
        flagged=[True, True, True, False],
    )
    assert isinstance(a, FlagAdoption)
    assert a.control_rate == 0.25
    assert a.flag_rate == 0.75
    assert a.effect == 0.5
    assert a.discordant_gain == 2
    assert a.discordant_loss == 0
    assert a.mcnemar_p == pytest.approx(0.5)
    assert a.n == 4


def test_adoption_contrast_without_discordant_pairs_is_p_one():
    a = adoption_contrast(control=[True, False], flagged=[True, False])
    assert a.effect == 0.0
    assert a.mcnemar_p == 1.0


def test_adoption_contrast_empty_is_nan():
    a = adoption_contrast([], [])
    assert math.isnan(a.control_rate) and math.isnan(a.mcnemar_p)
    assert a.n == 0


def test_adoption_contrast_rejects_unpaired_arms():
    with pytest.raises(ValueError):
        adoption_contrast([True, False], [True])


# --------------------------------------------------------------------------- runner


def test_the_system_flag_is_adopted_where_the_same_claim_from_a_peer_is_not():
    cases = [_case("c0"), _case("c1"), _case("c2")]

    res = run_prescreen_flag(_roster("m"), cases, _backend_factory(TrustsTheSystem))

    model = res["by_model"]["m"]
    assert model["rates"] == {"control": 0.0, "system": 1.0, "peer": 0.0}
    system = model["system"]
    assert system.control_rate == 0.0 and system.flag_rate == 1.0
    assert system.effect == 1.0
    assert system.discordant_gain == 3 and system.discordant_loss == 0
    assert system.n == 3
    assert model["peer"].effect == 0.0
    # The authority contrast: the same claim moves the model only when a machine makes it.
    assert model["authority"].effect == 1.0


def test_a_model_that_takes_any_note_shows_no_authority_effect():
    cases = [_case("c0"), _case("c1")]

    res = run_prescreen_flag(_roster("m"), cases, _backend_factory(TrustsAnyNote))

    model = res["by_model"]["m"]
    assert model["rates"] == {"control": 0.0, "system": 1.0, "peer": 1.0}
    assert model["system"].effect == 1.0 and model["peer"].effect == 1.0
    assert model["authority"].effect == 0.0
    assert model["authority"].discordant_gain == 0 and model["authority"].discordant_loss == 0
    assert model["authority"].mcnemar_p == 1.0


def test_control_arm_exposes_a_model_that_wanted_the_flagged_option_anyway():
    cases = [_case("c0"), _case("c1")]

    res = run_prescreen_flag(_roster("m"), cases, _backend_factory(PrefersTheFlagged))

    model = res["by_model"]["m"]
    assert model["rates"] == {"control": 1.0, "system": 1.0, "peer": 1.0}
    assert model["system"].effect == 0.0  # nothing was adopted: it already preferred the option
    assert model["correct_rates"] == {"control": 0.0, "system": 0.0, "peer": 0.0}


def test_adoption_is_reported_per_model():
    cases = [_case("c0"), _case("c1")]

    res = run_prescreen_flag(
        _roster("trusting", "stubborn"),
        cases,
        _backend_by_name(trusting=TrustsTheSystem, stubborn=PrefersTheFlagged),
    )

    assert res["models"] == ["trusting", "stubborn"]
    assert res["by_model"]["trusting"]["system"].effect == 1.0
    assert res["by_model"]["stubborn"]["system"].effect == 0.0
    # Pooled sits between the two models it aggregates.
    pooled = res["pooled"]["system"]
    assert pooled.n == 4
    assert pooled.control_rate == 0.5 and pooled.flag_rate == 1.0


def test_ground_truth_is_untouched_and_the_flag_costs_accuracy():
    cases = [_case("c0"), _case("c1")]

    res = run_prescreen_flag(_roster("m"), cases, _backend_factory(TrustsTheSystem))

    rates = res["by_model"]["m"]["correct_rates"]
    assert rates == {"control": 1.0, "system": 0.0, "peer": 1.0}
    assert res["flagged_answers"] == {"c0": "shortcut", "c1": "shortcut"}


def test_the_artifact_sits_on_the_board_and_the_control_arm_is_bare():
    res = run_prescreen_flag(_roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem))

    for arm, agent_id, expected in (
        ("system", SYSTEM_AGENT_ID, system_flag_content("shortcut")),
        ("peer", PEER_AGENT_ID, peer_note_content("shortcut")),
    ):
        turns = res["arms"][("m", arm)]["c0"].turns
        assert [t.agent_id for t in turns] == [agent_id, "m"]
        assert [t.context for t in turns] == [True, False]
        assert turns[0].seeded is True and turns[0].answer == "shortcut"
        assert turns[0].content == expected
        # The artifact is ambient context, so it never lands in the committed map.
        assert list(res["arms"][("m", arm)]["c0"].committed) == ["m"]

    control = res["arms"][("m", "control")]["c0"].turns
    assert [t.agent_id for t in control] == ["m"]


def test_the_two_artifact_arms_are_identical_apart_from_the_source():
    res = run_prescreen_flag(_roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem))

    system, peer = (res["arms"][("m", arm)]["c0"] for arm in ("system", "peer"))
    for field in ("order", "seed", "shared", "rounds", "members"):
        assert system.meta[field] == peer.meta[field]
    assert system.condition == peer.condition
    assert system.turns[0].content != peer.turns[0].content
    assert system.turns[0].answer == peer.turns[0].answer


def test_the_artifact_stays_on_the_board_across_rounds():
    res = run_prescreen_flag(
        _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem), rounds=2
    )
    turns = res["arms"][("m", "system")]["c0"].turns
    assert [t.agent_id for t in turns] == [SYSTEM_AGENT_ID, "m", "m"]
    assert res["by_model"]["m"]["rates"]["system"] == 1.0


def test_keyless_cases_report_no_correct_rates():
    case = Case(case_id="c0", patient_id="p", modality=Modality.TEXT, question=VIGNETTE)

    res = run_prescreen_flag(
        _roster("m"), [case], _backend_factory(TrustsTheSystem), flagged_answer_for="shortcut"
    )

    assert res["by_model"]["m"]["correct_rates"] is None
    assert "control_correct" not in res["per_case"][0]


def test_a_committee_is_accepted_as_a_roster_and_still_runs_each_model_alone():
    committee = Committee(members=tuple(_roster("a", "b")))

    res = run_prescreen_flag(committee, [_case("c0")], _backend_factory(TrustsTheSystem))

    assert res["models"] == ["a", "b"]
    for name in ("a", "b"):
        assert [t.agent_id for t in res["arms"][(name, "system")]["c0"].turns] == [
            SYSTEM_AGENT_ID,
            name,
        ]


def test_run_is_deterministic():
    cases = [_case("c0"), _case("c1")]
    first = run_prescreen_flag(_roster("m"), cases, _backend_factory(TrustsTheSystem))
    second = run_prescreen_flag(_roster("m"), cases, _backend_factory(TrustsTheSystem))
    assert first["per_case"] == second["per_case"]
    assert first["by_model"]["m"]["system"] == second["by_model"]["m"]["system"]


def test_per_case_rows_carry_every_arm_and_the_flagged_answer():
    res = run_prescreen_flag(_roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem))
    row = res["per_case"][0]
    assert row["model"] == "m" and row["case_id"] == "c0" and row["flagged"] == "shortcut"
    assert row["control"] is False and row["system"] is True and row["peer"] is False
    assert row["control_correct"] is True and row["system_correct"] is False


# --------------------------------------------------------------------------- guards


def test_flagged_answer_equal_to_the_correct_answer_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem),
            flagged_answer_for="correct",
        )


def test_none_flagged_answer_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem),
            flagged_answer_for=lambda case: None,
        )


def test_identical_artifact_content_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem),
            system_content_for=lambda answer, case: "same",
            peer_content_for=lambda answer, case: "same",
        )


def test_an_artifact_id_that_collides_with_a_model_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem), system_agent_id="m"
        )


def test_one_id_for_both_artifacts_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem),
            system_agent_id="src", peer_agent_id="src",
        )


def test_empty_roster_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag([], [_case("c0")], _backend_factory(TrustsTheSystem))


def test_duplicate_model_names_raise():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m", "m"), [_case("c0")], _backend_factory(TrustsTheSystem)
        )


def test_empty_case_set_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(_roster("m"), [], _backend_factory(TrustsTheSystem))


def test_duplicate_case_ids_raise():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0"), _case("c0")], _backend_factory(TrustsTheSystem)
        )


def test_zero_rounds_raises():
    with pytest.raises(ValueError):
        run_prescreen_flag(
            _roster("m"), [_case("c0")], _backend_factory(TrustsTheSystem), rounds=0
        )
