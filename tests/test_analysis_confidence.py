"""Tests for benchmaxxing.analysis.confidence_climb (issue 79).

These run fully offline on synthetic ``Transcript`` objects. The measure asks whether stated
per-turn confidence rises as a committee converges on the cue-anchored wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing.analysis import confidence_climb
from benchmaxxing.schema import Condition, Transcript, Turn


def _transcript(turns, run_id="run", case_id="case"):
    return Transcript(
        run_id=run_id,
        case_id=case_id,
        condition=Condition.CONTAMINATED,
        turns=turns,
    )


def _rising_transcript():
    """Confidence on wrong answer 'B' rises turn over turn.

    Includes a distractor turn on the correct answer 'A' (high confidence, must be ignored)
    and a 'B' turn with no confidence (must be skipped).
    """
    return _transcript(
        [
            Turn(turn_index=0, agent_id="a1", content="", answer="B", confidence=0.30),
            Turn(turn_index=1, agent_id="a2", content="", answer="B", confidence=0.50),
            Turn(turn_index=2, agent_id="a3", content="", answer="A", confidence=0.99),
            Turn(turn_index=3, agent_id="a4", content="", answer="B", confidence=0.70),
            Turn(turn_index=4, agent_id="a5", content="", answer="B", confidence=None),
            Turn(turn_index=5, agent_id="a6", content="", answer="B", confidence=0.90),
        ]
    )


def _flat_transcript():
    """Confidence on wrong answer 'B' is constant across turns."""
    return _transcript(
        [
            Turn(turn_index=0, agent_id="a1", content="", answer="B", confidence=0.50),
            Turn(turn_index=1, agent_id="a2", content="", answer="B", confidence=0.50),
            Turn(turn_index=2, agent_id="a3", content="", answer="B", confidence=0.50),
            Turn(turn_index=3, agent_id="a4", content="", answer="B", confidence=0.50),
        ]
    )


def test_confidence_climb_rising_is_climbing():
    res = confidence_climb(_rising_transcript(), "B")
    # Four 'B' turns carry confidence (the 'A' turn and the None-confidence turn are dropped).
    assert res["n"] == 4
    assert res["confidences"] == [0.30, 0.50, 0.70, 0.90]
    assert res["slope"] > 0
    assert res["final_minus_first"] == pytest.approx(0.60)
    assert res["climbing"] is True


def test_confidence_climb_flat_is_not_climbing():
    res = confidence_climb(_flat_transcript(), "B")
    assert res["n"] == 4
    assert res["slope"] == pytest.approx(0.0, abs=1e-9)
    assert res["final_minus_first"] == pytest.approx(0.0)
    assert res["climbing"] is False


def test_confidence_climb_falling_slope_is_negative():
    t = _transcript(
        [
            Turn(turn_index=0, agent_id="a1", content="", answer="B", confidence=0.90),
            Turn(turn_index=1, agent_id="a2", content="", answer="B", confidence=0.60),
            Turn(turn_index=2, agent_id="a3", content="", answer="B", confidence=0.30),
        ]
    )
    res = confidence_climb(t, "B")
    assert res["slope"] < 0
    assert res["final_minus_first"] == pytest.approx(-0.60)
    assert res["climbing"] is False


def test_confidence_climb_ignores_other_answers():
    # No turn commits the tracked wrong answer 'Z'.
    res = confidence_climb(_rising_transcript(), "Z")
    assert res["n"] == 0
    assert np.isnan(res["slope"])
    assert np.isnan(res["final_minus_first"])
    assert res["climbing"] is False
    assert res["confidences"] == []


def test_confidence_climb_single_turn_slope_is_nan():
    t = _transcript(
        [Turn(turn_index=0, agent_id="a1", content="", answer="B", confidence=0.5)]
    )
    res = confidence_climb(t, "B")
    assert res["n"] == 1
    assert np.isnan(res["slope"])
    # A single usable turn: first and last confidence coincide.
    assert res["final_minus_first"] == pytest.approx(0.0)
    assert res["climbing"] is False


def test_confidence_climb_sorts_by_turn_index():
    # Turns supplied out of order; the fit and delta must respect turn_index order.
    t = _transcript(
        [
            Turn(turn_index=2, agent_id="a3", content="", answer="B", confidence=0.80),
            Turn(turn_index=0, agent_id="a1", content="", answer="B", confidence=0.20),
            Turn(turn_index=1, agent_id="a2", content="", answer="B", confidence=0.50),
        ]
    )
    res = confidence_climb(t, "B")
    assert res["confidences"] == [0.20, 0.50, 0.80]
    assert res["slope"] > 0
    assert res["final_minus_first"] == pytest.approx(0.60)
    assert res["climbing"] is True
