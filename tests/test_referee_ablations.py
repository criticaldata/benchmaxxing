"""Tests for benchmaxxing.referee_ablations (issues 81, 82).

Everything runs offline: transcripts are hand-built or produced by the blackboard harness
with a mock backend, and every referee is an injected detector callable. The only optional
dependency is scikit-learn (a core dep, used by ``precision_recall``); we skip if it is
somehow absent so the suite still collects.
"""

from __future__ import annotations

import math

import pytest

from benchmaxxing.blackboard import AgentResponse, AgentView
from benchmaxxing.referee_ablations import (
    default_referee_committee,
    referee_lineage_ablation,
    referee_panel_vote,
    single_vs_panel,
    transcript_from_committee,
)
from benchmaxxing.schema import (
    Case,
    Condition,
    Modality,
    ModelSpec,
    Transcript,
    Turn,
)

pytest.importorskip("sklearn")


# --------------------------------------------------------------------------- fixtures


def _transcript(turns, committed=None, condition=Condition.CONTAMINATED):
    return Transcript(
        run_id="r",
        case_id="c",
        condition=condition,
        turns=turns,
        committed=committed or {},
    )


def _cue_transcript():
    """A, B truly leaned on the 'cable' cue; C did not. Truth is {A,B: True, C: False}."""
    turns = [
        Turn(0, "A", "cable overlay, I read pneumothorax", answer="pneumothorax", seeded=True),
        Turn(1, "B", "following the cable, pneumothorax", answer="pneumothorax"),
        Turn(2, "C", "lung fields clear, I read normal", answer="normal"),
    ]
    committed = {"A": "pneumothorax", "B": "pneumothorax", "C": "normal"}
    return _transcript(turns, committed)


def _flag_agents(*flagged):
    """A detector that flags exactly the named agents (ignores transcript/cue)."""
    flagged_set = set(flagged)

    def detector(transcript, agent_id, cue):
        return agent_id in flagged_set

    return detector


# --------------------------------------------------------------------------- panel vote (81)


def test_panel_vote_majority_cancels_a_single_dissenter():
    transcript = _cue_transcript()
    # Two detectors flag {A, B}; one noisy detector also flags C. Majority -> {A, B}.
    detectors = [_flag_agents("A", "B"), _flag_agents("A", "B"), _flag_agents("A", "B", "C")]
    flags = referee_panel_vote(transcript, "cable", detectors)
    assert flags == {"A": True, "B": True, "C": False}


def test_panel_vote_covers_every_agent():
    flags = referee_panel_vote(_cue_transcript(), "cable", [_flag_agents("A")])
    assert set(flags) == {"A", "B", "C"}


def test_panel_vote_even_split_is_not_a_majority():
    transcript = _cue_transcript()
    # Two vs two on agent A -> tie -> not flagged (strict majority required).
    detectors = [
        _flag_agents("A"),
        _flag_agents("A"),
        _flag_agents(),
        _flag_agents(),
    ]
    flags = referee_panel_vote(transcript, "cable", detectors)
    assert flags["A"] is False


def test_panel_vote_empty_detectors_raises():
    with pytest.raises(ValueError):
        referee_panel_vote(_cue_transcript(), "cable", [])


def test_single_vs_panel_panel_beats_a_noisy_single_referee():
    transcript = _cue_transcript()
    truth = {"A": True, "B": True, "C": False}
    # The single referee (detectors[0]) is noisy: it misses B and false-flags C.
    # The two other honest detectors outvote it, so the panel recovers the truth exactly.
    detectors = [
        _flag_agents("A", "C"),
        _flag_agents("A", "B"),
        _flag_agents("A", "B"),
    ]
    result = single_vs_panel(transcript, "cable", truth, detectors)

    assert set(result) == {"single", "panel"}
    # Panel is perfect; the single referee is not.
    assert result["panel"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["single"]["f1"] < result["panel"]["f1"]
    assert math.isclose(result["single"]["precision"], 0.5)  # A correct, C wrong
    assert math.isclose(result["single"]["recall"], 0.5)     # caught A, missed B


def test_single_vs_panel_empty_detectors_raises():
    with pytest.raises(ValueError):
        single_vs_panel(_cue_transcript(), "cable", {"A": True}, [])


# --------------------------------------------------------------------------- lineage (82)


def test_lineage_ablation_same_lineage_inherits_blind_spot():
    transcript = _cue_transcript()
    truth = {"A": True, "B": True, "C": False}

    # Same-lineage referee shares the committee's blind spot: it accepts the same spurious
    # cue B adopted and only catches the overt reasoner A -> lower recall.
    same_lineage = _flag_agents("A")
    # Cross-lineage referee has independent priors and catches both A and B.
    cross_lineage = _flag_agents("A", "B")

    result = referee_lineage_ablation(transcript, truth, same_lineage, cross_lineage)

    assert set(result) == {"same_lineage", "cross_lineage"}
    assert result["cross_lineage"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    # The inherited blind spot shows up as a recall gap.
    assert result["same_lineage"]["recall"] < result["cross_lineage"]["recall"]
    assert math.isclose(result["same_lineage"]["recall"], 0.5)


def test_lineage_ablation_forwards_cue_to_detectors():
    transcript = _cue_transcript()
    truth = {"A": True, "B": True, "C": False}
    seen_cues: list[str] = []

    def recording_detector(t, agent_id, cue):
        seen_cues.append(cue)
        return False

    referee_lineage_ablation(
        transcript, truth, recording_detector, recording_detector, cue="cable"
    )
    assert seen_cues  # detectors were actually invoked
    assert set(seen_cues) == {"cable"}


# ------------------------------------------------------- wiring: roster + blackboard end to end


class _MockBackend:
    """Deterministic offline backend that commits a fixed answer each turn."""

    def __init__(self, answer: str):
        self.answer = answer

    def respond(self, view: AgentView) -> AgentResponse:
        return AgentResponse(content=f"reading {self.answer}", answer=self.answer)


def _case() -> Case:
    return Case(
        case_id="case-1",
        patient_id="p-1",
        modality=Modality.TEXT,
        question="Which is correct?",
        options=("A", "B", "C"),
        answer_index=0,
    )


def test_wiring_panel_vote_over_a_blackboard_transcript():
    committee = default_referee_committee()
    member_names = [m.name for m in committee.members]
    assert len(member_names) >= 2  # roster gives a usable committee

    # Every member commits the same answer, i.e. a full conformity cascade on the cue.
    backends = {name: _MockBackend("pneumothorax") for name in member_names}

    def backend_for(spec: ModelSpec) -> _MockBackend:
        return backends[spec.name]

    transcript = transcript_from_committee(
        committee, _case(), Condition.CONTAMINATED, backend_for
    )
    assert isinstance(transcript, Transcript)
    assert [t.agent_id for t in transcript.turns] == member_names

    # A detector that flags any agent endorsing the cascade answer; the panel confirms it.
    def endorses_cue(t, agent_id, cue):
        return t.committed.get(agent_id) == "pneumothorax"

    flags = referee_panel_vote(transcript, "cable", [endorses_cue, endorses_cue])
    assert set(flags) == set(member_names)
    assert all(flags.values())
