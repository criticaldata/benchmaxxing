"""Unit tests for the multi-round cascade-dynamics analysis (issue #130).

Hand-built transcript where the holdout's adoption of the planted shortcut GROWS over rounds; the
pure analysis must recover the per-round answers and detect the monotone increase. No key, no
network, no real data (module-level import of the experiment pulls only stdlib).
"""
from __future__ import annotations

from benchmaxxing.schema import Turn
from experiments.cascade.multi_round import (
    adoption_curve,
    is_monotone_nondecreasing,
    per_round_holdout_answers,
)

SHORTCUT = "B"  # the planted wrong answer


def _hand_built_growing_transcript():
    """3 rounds, order peer1, peer2, holdout. The holdout resists in round 1 ("A"), then adopts
    the shortcut "B" from round 2 on: a cascade that builds over rounds."""
    turns = []
    ti = 0
    holdout_by_round = ["A", "B", "B"]  # resist, adopt, adopt
    for r, hold in enumerate(holdout_by_round):
        for who, ans in (("peer1", SHORTCUT), ("peer2", SHORTCUT), ("holdout", hold)):
            turns.append(Turn(turn_index=ti, agent_id=who, content=f"r{r}:{who}", answer=ans))
            ti += 1
    return turns


def test_per_round_holdout_answers_extracts_in_order():
    turns = _hand_built_growing_transcript()
    assert per_round_holdout_answers(turns) == ["A", "B", "B"]
    # the peers are excluded; only the holdout's per-round trajectory is returned
    assert per_round_holdout_answers(turns, holdout_id="peer1") == [SHORTCUT, SHORTCUT, SHORTCUT]


def test_adoption_curve_detects_growth_on_hand_built_case():
    turns = _hand_built_growing_transcript()
    adopt = [a == SHORTCUT for a in per_round_holdout_answers(turns)]  # [False, True, True]
    curve = adoption_curve([adopt], k=3)
    assert curve == [0.0, 1.0, 1.0]
    assert is_monotone_nondecreasing(curve) is True


def test_monotone_flags_a_flat_or_falling_curve():
    # a real single-shot / non-building cascade: flat, or a dip, is NOT monotone-increasing growth
    assert is_monotone_nondecreasing([0.15, 0.15, 0.175, 0.15, 0.15]) is False
    assert is_monotone_nondecreasing([0.2, 0.1, 0.1]) is False
    # a genuinely building curve is
    assert is_monotone_nondecreasing([0.1, 0.2, 0.35, 0.4]) is True


def test_adoption_curve_averages_across_cases_and_skips_short_lists():
    # case A adopts from round 2, case B never adopts; round 0 = 0/2, round 1 = 1/2
    flags = [[False, True, True], [False, False, False]]
    assert adoption_curve(flags, k=3) == [0.0, 0.5, 0.5]
    # a case with a shorter list must not crash the later-round averages
    assert adoption_curve([[True], [True, True]], k=2) == [1.0, 1.0]
