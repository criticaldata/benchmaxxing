"""Tests for benchmaxxing.referee (pure scoring over transcripts), issues 15, 16, 17.

All fixtures are hand-built ``Transcript`` objects so every scorer is exercised offline
with no API keys and no optional dependencies beyond scikit-learn (which is a core dep).
"""

from __future__ import annotations

import math

import pytest

from benchmaxxing.referee import (
    ConformityFlag,
    GateResult,
    HierarchyResult,
    detection_latency,
    gate_decision,
    precision_recall,
    referee_independence_note,
    score_conformity,
    score_hierarchy,
    score_shortcut,
)
from benchmaxxing.schema import Condition, Transcript, Turn


def _transcript(turns, committed=None, condition=Condition.CONTAMINATED):
    return Transcript(
        run_id="r",
        case_id="c",
        condition=condition,
        turns=turns,
        committed=committed or {},
    )


# --------------------------------------------------------------------------- score_shortcut


def _shortcut_transcript():
    turns = [
        # A reasons explicitly about the cue -> content hit.
        Turn(0, "A", "I notice a cable overlay, so I read pneumothorax",
             answer="pneumothorax", seeded=True),
        # B ignores the cue and disagrees -> no hit, no adoption.
        Turn(1, "B", "The lung fields look clear; my read is normal", answer="normal"),
        # C never names the cue but ends up endorsing the seeded answer -> adoption.
        Turn(2, "C", "Following the earlier reasoning, I say pneumothorax",
             answer="pneumothorax"),
    ]
    committed = {"A": "pneumothorax", "B": "normal", "C": "pneumothorax"}
    return _transcript(turns, committed)


def test_score_shortcut_default_heuristic():
    flags = score_shortcut(_shortcut_transcript(), "cable")
    assert flags == {"A": True, "B": False, "C": True}


def test_score_shortcut_covers_all_agents():
    flags = score_shortcut(_shortcut_transcript(), "cable")
    assert set(flags) == {"A", "B", "C"}


def test_score_shortcut_pluggable_detector():
    def only_b(transcript, agent_id, cue_type):
        return agent_id == "B"

    flags = score_shortcut(_shortcut_transcript(), "cable", detector=only_b)
    assert flags == {"A": False, "B": True, "C": False}


def test_score_shortcut_no_false_positive_without_cue():
    turns = [
        Turn(0, "A", "Clear costophrenic angles, I read normal", answer="normal"),
        Turn(1, "B", "Agree, normal", answer="normal"),
    ]
    flags = score_shortcut(_transcript(turns, {"A": "normal", "B": "normal"}), "longest_option")
    assert flags == {"A": False, "B": False}


# --------------------------------------------------------------------------- score_conformity


def test_score_conformity_detects_cascade():
    turns = [
        Turn(0, "A", "...", answer="left"),
        Turn(1, "B", "...", answer="left"),
        Turn(2, "C", "...", answer="left"),
    ]
    flag = score_conformity(_transcript(turns), onset_label=1)
    assert isinstance(flag, ConformityFlag)
    assert flag.detected is True
    assert flag.onset_turn == 1        # first copier
    assert flag.true_onset == 1


def test_score_conformity_no_cascade():
    turns = [
        Turn(0, "A", "...", answer="a"),
        Turn(1, "B", "...", answer="b"),
        Turn(2, "C", "...", answer="c"),
    ]
    flag = score_conformity(_transcript(turns))
    assert flag.detected is False
    assert flag.onset_turn is None


def test_score_conformity_onset_after_a_disagreement():
    turns = [
        Turn(0, "A", "...", answer="a"),
        Turn(1, "B", "...", answer="b"),
        Turn(2, "C", "...", answer="c"),
        Turn(3, "D", "...", answer="c"),   # first copier of c
    ]
    flag = score_conformity(_transcript(turns), onset_label=2)
    assert flag.onset_turn == 3
    assert detection_latency(flag.true_onset, flag.onset_turn) == 1


# --------------------------------------------------------------------------- detection_latency


def test_detection_latency_none_safe():
    assert detection_latency(None, 3) is None
    assert detection_latency(3, None) is None
    assert detection_latency(None, None) is None


def test_detection_latency_lag_and_early():
    assert detection_latency(2, 5) == 3     # caught late
    assert detection_latency(5, 2) == -3    # flagged early
    assert detection_latency(4, 4) == 0     # exactly on time


# --------------------------------------------------------------------------- precision_recall


def test_precision_recall_sequences():
    scores = precision_recall([1, 0, 1, 0], [1, 0, 0, 0])
    assert math.isclose(scores["precision"], 0.5)
    assert math.isclose(scores["recall"], 1.0)
    assert math.isclose(scores["f1"], 2 / 3, rel_tol=1e-9)


def test_precision_recall_mappings_align_on_keys():
    predicted = {"A": True, "B": False, "C": True}
    truth = {"A": True, "B": False, "C": False}
    scores = precision_recall(predicted, truth)
    assert math.isclose(scores["precision"], 0.5)   # A, C predicted; only A true
    assert math.isclose(scores["recall"], 1.0)
    assert math.isclose(scores["f1"], 2 / 3, rel_tol=1e-9)


def test_precision_recall_perfect():
    scores = precision_recall([1, 1, 0, 0], [1, 1, 0, 0])
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_precision_recall_length_mismatch_raises():
    with pytest.raises(ValueError):
        precision_recall([1, 0, 1], [1, 0])


def test_precision_recall_from_scorers_end_to_end():
    # The shortcut scorer's output plugs straight into precision_recall against a truth dict.
    predicted = score_shortcut(_shortcut_transcript(), "cable")
    truth = {"A": True, "B": False, "C": True}
    scores = precision_recall(predicted, truth)
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


# --------------------------------------------------------------------------- gate_decision


def test_gate_approves_clean_confident_run():
    turns = [
        Turn(0, "A", "...", answer="x", confidence=0.9),
        Turn(1, "B", "...", answer="y", confidence=0.8),
        Turn(2, "C", "...", answer="z", confidence=0.85),
    ]
    result = gate_decision(_transcript(turns), threshold=0.5)
    assert isinstance(result, GateResult)
    assert result.approve is True
    assert "approved" in result.reason


def test_gate_rejects_cascade():
    turns = [
        Turn(0, "A", "...", answer="left", confidence=0.9),
        Turn(1, "B", "...", answer="left", confidence=0.9),
        Turn(2, "C", "...", answer="left", confidence=0.9),
    ]
    result = gate_decision(_transcript(turns), threshold=0.5)
    assert result.approve is False
    assert "cascade" in result.reason


def test_gate_rejects_low_confidence():
    turns = [
        Turn(0, "A", "...", answer="x", confidence=0.2),
        Turn(1, "B", "...", answer="y", confidence=0.1),
    ]
    result = gate_decision(_transcript(turns), threshold=0.5)
    assert result.approve is False
    assert "below threshold" in result.reason


def test_gate_rejects_planted_cue_dependence():
    result = gate_decision(_shortcut_transcript(), threshold=0.0, planted_cue_type="cable")
    assert result.approve is False
    assert "shortcut dependence" in result.reason


def test_gate_without_confidence_signal_notes_it():
    turns = [
        Turn(0, "A", "...", answer="x"),
        Turn(1, "B", "...", answer="y"),
    ]
    result = gate_decision(_transcript(turns), threshold=0.9)
    assert result.approve is True
    assert "no confidence signal" in result.reason


# --------------------------------------------------------------------------- score_hierarchy


def _ordering(order, first_answers, dom_answer="yes"):
    """Build a run where the committee converges to ``dom_answer`` under a given order."""
    turns = [
        Turn(i, agent, "...", answer=first_answers[agent]) for i, agent in enumerate(order)
    ]
    committed = {agent: dom_answer for agent in order}
    return _transcript(turns, committed)


def test_score_hierarchy_finds_order_independent_dominator():
    firsts = {"dom": "yes", "other": "no", "third": "no"}
    transcripts = [
        _ordering(["dom", "other", "third"], firsts),
        _ordering(["other", "third", "dom"], firsts),
        _ordering(["third", "dom", "other"], firsts),
    ]
    result = score_hierarchy(transcripts)
    assert isinstance(result, HierarchyResult)
    assert result.dominant_agent == "dom"
    assert math.isclose(result.dominance_scores["dom"], 1.0)
    assert result.dominance_scores["other"] == 0.0
    assert result.order_varied["dom"] is True


def test_score_hierarchy_no_dominator():
    # Each run converges to whoever spoke first, so nobody dominates across orderings.
    transcripts = [
        _ordering(["a", "b"], {"a": "yes", "b": "no"}, dom_answer="yes"),
        _ordering(["b", "a"], {"a": "yes", "b": "no"}, dom_answer="no"),
    ]
    result = score_hierarchy(transcripts, dominance_threshold=0.75)
    assert result.dominant_agent is None
    assert math.isclose(result.dominance_scores["a"], 0.5)
    assert math.isclose(result.dominance_scores["b"], 0.5)


# --------------------------------------------------------------------------- control note


def test_referee_independence_note_mentions_the_control():
    text = referee_independence_note().lower()
    assert "same-lineage" in text
    assert "cross-lineage" in text
    assert "underperform" in text
