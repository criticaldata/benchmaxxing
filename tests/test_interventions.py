"""Tests for benchmaxxing.interventions (issues 16 and 12).

Everything runs offline: a deterministic blackboard-protocol backend drives the live
committee, and synthetic flip records drive the natural-cue validation. No network, no keys.
"""

from __future__ import annotations

import math

from benchmaxxing.blackboard import AgentResponse
from benchmaxxing.interventions import (
    intervention_effect,
    natural_cue_validation,
    run_referee_intervention,
)
from benchmaxxing.schema import Case, Committee, Modality, ModelSpec, Turn


# --------------------------------------------------------------------------- fixtures


class CascadeBackend:
    """A member that copies a visible shortcut unless a referee has flagged it.

    * If a referee turn is on the board, it resists and answers ``correct``.
    * Else if the planted shortcut is visible, it copies the shortcut (a cascade).
    * Otherwise it gives its independent (correct) read.
    """

    def __init__(self, name, correct="normal", shortcut="pneumo"):
        self.name = name
        self.correct = correct
        self.shortcut = shortcut

    def respond(self, view):
        saw_referee = any(t.agent_id == "referee" for t in view.visible_turns)
        saw_shortcut = any(
            t.seeded and t.answer == self.shortcut for t in view.visible_turns
        )
        if saw_referee:
            return AgentResponse(content=f"{self.name} heeds referee", answer=self.correct,
                                 confidence=0.9)
        if saw_shortcut:
            return AgentResponse(content=f"{self.name} copies shortcut", answer=self.shortcut,
                                 confidence=0.6)
        return AgentResponse(content=f"{self.name} independent read", answer=self.correct,
                             confidence=0.7)


class ResistantBackend:
    """A member that always answers the correct read, ignoring any shortcut on the board."""

    def __init__(self, name, correct="normal"):
        self.name = name
        self.correct = correct

    def respond(self, view):
        return AgentResponse(content=f"{self.name} correct", answer=self.correct, confidence=0.8)


def make_committee(*names):
    members = tuple(ModelSpec(name=nm, lineage="gemini") for nm in names)
    return Committee(members=members)


def make_case():
    return Case(
        case_id="case-1",
        patient_id="p-1",
        modality=Modality.IMAGE,
        image_ref="cxr-1.jpg",
        label="pneumo",
    )


def backend_factory(backends):
    def backend_for(spec):
        return backends[spec.name]

    return backend_for


def catch_shortcut_hook(state):
    """Fire once after the plant lands: warn the committee about the shortcut cue."""
    if any(t.agent_id == "referee" for t in state.turns):
        return None
    if any(t.seeded for t in state.turns):
        return Turn(turn_index=-1, agent_id="referee",
                    content="shortcut cue detected: disregard the planted answer",
                    answer="FLAG")
    return None


def always_fire_hook(state):
    """Fire once after the first committed turn regardless of any cascade (false-alarm case)."""
    if any(t.agent_id == "referee" for t in state.turns):
        return None
    if state.last_turn is not None:
        return Turn(turn_index=-1, agent_id="referee", content="precautionary flag", answer="FLAG")
    return None


# --------------------------------------------------------------------------- issue 16


def test_run_referee_intervention_returns_two_transcripts():
    committee = make_committee("A", "B", "C")
    backends = {nm: CascadeBackend(nm) for nm in ("A", "B", "C")}

    out = run_referee_intervention(
        committee, make_case(), backend_factory(backends), catch_shortcut_hook,
        seed_turn=(0, "pneumo", "planter"), rounds=1,
    )

    assert set(out) == {"intervention_transcript", "baseline_transcript"}
    interv = out["intervention_transcript"]
    base = out["baseline_transcript"]
    # The intervention transcript carries an injected referee turn; the baseline does not.
    assert any(t.agent_id == "referee" for t in interv.turns)
    assert not any(t.agent_id == "referee" for t in base.turns)


def test_intervention_reduces_shortcut_cascade():
    committee = make_committee("A", "B", "C")
    backends = {nm: CascadeBackend(nm) for nm in ("A", "B", "C")}

    out = run_referee_intervention(
        committee, make_case(), backend_factory(backends), catch_shortcut_hook,
        seed_turn=(0, "pneumo", "planter"), rounds=1,
    )
    effect = intervention_effect(out["baseline_transcript"], out["intervention_transcript"])

    assert effect["shortcut_answer"] == "pneumo"
    # Referee-absent baseline cascades onto the shortcut; the referee pulls it off entirely.
    assert math.isclose(effect["baseline_shortcut_fraction"], 1.0)
    assert math.isclose(effect["intervention_shortcut_fraction"], 0.0)
    assert math.isclose(effect["cascade_reduction"], 1.0)
    assert effect["referee_fired"] is True
    # A real cascade existed in the baseline, so this is not a false alarm.
    assert effect["false_alarm"] is False


def test_intervention_effect_flags_false_alarm():
    committee = make_committee("A", "B", "C")
    backends = {nm: ResistantBackend(nm) for nm in ("A", "B", "C")}

    out = run_referee_intervention(
        committee, make_case(), backend_factory(backends), always_fire_hook,
        seed_turn=(0, "pneumo", "planter"), rounds=1,
    )
    effect = intervention_effect(out["baseline_transcript"], out["intervention_transcript"])

    # Nobody adopts the shortcut even without a referee, yet the referee fired: a false alarm.
    assert math.isclose(effect["baseline_shortcut_fraction"], 0.0)
    assert effect["referee_fired"] is True
    assert effect["false_alarm"] is True
    assert "false alarm" in effect["note"]


# --------------------------------------------------------------------------- issue 12


class _Rec:
    """A tiny synthetic flip record (duck-types a FlipRecord for reduction)."""

    def __init__(self, model, flipped):
        self.model = model
        self.flipped = flipped


def test_natural_cue_validation_agreeing_rankings_validate():
    # Same reliance ordering under injected cues and under natural devices: m3 > m2 > m1.
    injected = {"m1": 0.1, "m2": 0.5, "m3": 0.9}
    natural = {"m1": 0.2, "m2": 0.4, "m3": 0.8}

    result = natural_cue_validation(injected, natural, threshold=0.7)

    assert result["models"] == ["m1", "m2", "m3"]
    assert result["injected_ranking"] == ["m3", "m2", "m1"]
    assert result["natural_ranking"] == ["m3", "m2", "m1"]
    assert math.isclose(result["correlation"], 1.0)
    assert result["injection_validated"] is True


def test_natural_cue_validation_disagreeing_rankings_do_not_validate():
    # Opposite orderings: injection does not reproduce the natural pattern.
    injected = {"m1": 0.1, "m2": 0.5, "m3": 0.9}
    natural = {"m1": 0.9, "m2": 0.5, "m3": 0.1}

    result = natural_cue_validation(injected, natural, threshold=0.7)

    assert math.isclose(result["correlation"], -1.0)
    assert result["injection_validated"] is False


def test_natural_cue_validation_reduces_record_lists():
    # Records-per-model form: m2 flips more than m1 under both regimes -> agreement.
    injected = {
        "m1": [_Rec("m1", False), _Rec("m1", False), _Rec("m1", True)],   # 1/3
        "m2": [_Rec("m2", True), _Rec("m2", True), _Rec("m2", False)],    # 2/3
    }
    natural = [
        _Rec("m1", False), _Rec("m1", False),   # 0/2
        _Rec("m2", True), _Rec("m2", False),    # 1/2
    ]

    result = natural_cue_validation(injected, natural, threshold=0.7)

    assert result["models"] == ["m1", "m2"]
    assert result["injected_ranking"] == ["m2", "m1"]
    assert result["natural_ranking"] == ["m2", "m1"]
    assert result["injection_validated"] is True


def test_natural_cue_validation_needs_two_common_models():
    try:
        natural_cue_validation({"m1": 0.5}, {"m1": 0.4})
        raise AssertionError("expected ValueError for fewer than 2 common models")
    except ValueError:
        pass
