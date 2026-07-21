"""Offline tests for transcript-level deference and confidence dynamics (issue 118)."""

from __future__ import annotations

import math

import pytest

from benchmaxxing.schema import Condition, Transcript, Turn
from benchmaxxing.transcript_dynamics import (
    confidence_trajectory,
    deference_rate,
    deference_summary,
    summarize_transcripts,
)


def _transcript(turns, *, shared=True, run_id="run"):
    return Transcript(
        run_id=run_id,
        case_id="case",
        condition=Condition.CONTAMINATED,
        turns=turns,
        meta={"shared": shared},
    )


def _deliberation(*, shared=True):
    return _transcript(
        [
            Turn(0, "a", "", answer="A", confidence=0.40),
            Turn(1, "b", "", answer="B", confidence=0.50),
            # a changes from A to the B supplied by b since a's previous turn: deference.
            Turn(2, "a", "", answer="B", confidence=0.70),
            # b keeps B: eligible follow-up, but not an answer change.
            Turn(3, "b", "", answer="B", confidence=0.60),
            # a changes again, but no intervening peer supplied C.
            Turn(4, "a", "", answer="C", confidence=0.80),
        ],
        shared=shared,
    )


def test_deference_rate_finds_intervening_peer_adoption():
    result = deference_summary(_deliberation())
    assert result.eligible_turns == 3
    assert result.deferred_turns == 1
    assert result.rate == pytest.approx(1 / 3)
    assert deference_rate(_deliberation()) == pytest.approx(1 / 3)
    event = result.events[0]
    assert (event.agent_id, event.previous_answer, event.answer) == ("a", "A", "B")
    assert (event.peer_agent_id, event.peer_turn_index) == ("b", 1)


def test_isolated_transcript_has_no_peer_deference():
    result = deference_summary(_deliberation(shared=False))
    assert result.eligible_turns == 3
    assert result.deferred_turns == 0
    assert result.rate == 0.0


def test_seed_can_influence_peer_but_is_not_agent_behavior():
    transcript = _transcript(
        [
            Turn(0, "a", "", answer="A"),
            Turn(1, "planter", "", answer="B", seeded=True),
            Turn(2, "a", "", answer="B"),
            # The planter's first real answer is not a follow-up to its externally seeded turn.
            Turn(3, "planter", "", answer="A"),
        ]
    )
    result = deference_summary(transcript)
    assert result.eligible_turns == 1
    assert result.deferred_turns == 1
    assert result.events[0].peer_agent_id == "planter"


def test_first_answer_cannot_count_as_deference():
    transcript = _transcript(
        [
            Turn(0, "a", "", answer="B"),
            Turn(1, "b", "", answer="B"),
        ]
    )
    result = deference_summary(transcript)
    assert result.eligible_turns == 0
    assert result.deferred_turns == 0
    assert result.rate == 0.0


def test_confidence_trajectory_infers_seed_and_detects_rise():
    transcript = _transcript(
        [
            Turn(0, "a", "", answer="A", confidence=0.80),
            Turn(1, "planter", "", answer="B", seeded=True),
            Turn(2, "b", "", answer="B", confidence=0.30),
            Turn(3, "c", "", answer="A", confidence=None),
            Turn(4, "a", "", answer="B", confidence=0.60),
            Turn(5, "b", "", answer="B", confidence=0.90),
        ]
    )
    result = confidence_trajectory(transcript)
    assert result.wrong_answer == "B"
    assert result.turn_indices == (0, 1, 2, 3, 4, 5)
    assert math.isnan(result.confidences[1])
    assert result.wrong_agreement == (False, True, True, False, True, True)
    assert result.n_confident_turns == 4
    assert result.confidence_coverage == pytest.approx(4 / 5)
    assert result.wrong_confidence_slope > 0
    assert result.rising_while_wrong is True


def test_missing_confidence_is_reported_as_unavailable():
    transcript = _transcript(
        [
            Turn(0, "a", "", answer="A"),
            Turn(1, "planter", "", answer="B", seeded=True),
            Turn(2, "b", "", answer="B"),
        ]
    )
    result = confidence_trajectory(transcript)
    assert result.n_confident_turns == 0
    assert result.confidence_coverage == 0.0
    assert math.isnan(result.wrong_confidence_slope)
    assert result.rising_while_wrong is None


def test_summary_compares_shared_and_isolated_runs():
    summary = summarize_transcripts(
        [_deliberation(shared=True), _deliberation(shared=False,)]
    )
    assert summary["shared"]["deference_rate"] == pytest.approx(1 / 3)
    assert summary["isolated"]["deference_rate"] == 0.0
    assert summary["shared"]["confidence_coverage"] == 1.0
    assert summary["isolated"]["confidence_coverage"] == 1.0
