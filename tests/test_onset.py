"""Tests for benchmaxxing.onset (cascade dynamics math), issues 14 and 64.

These exercise the pure-numpy fallback so they pass in an env without ruptures. The ruptures
path is covered by a test that skips cleanly when ruptures is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing.onset import (
    cascade_onset,
    confidence_trajectory,
    contagion_index,
    deference_rate,
)
from benchmaxxing.schema import Turn


# --------------------------------------------------------------------------- cascade_onset


def test_cascade_onset_finds_clean_step():
    # Five turns following evidence (0), then five following the first committed answer (1).
    series = [0.0] * 5 + [1.0] * 5
    assert cascade_onset(series) == 5


def test_cascade_onset_step_at_other_index():
    series = [0.2] * 3 + [0.9] * 9
    assert cascade_onset(series) == 3


def test_cascade_onset_with_noise_is_close():
    rng = np.random.default_rng(0)
    boundary = 8
    before = rng.normal(0.0, 0.05, size=boundary)
    after = rng.normal(1.0, 0.05, size=12)
    series = np.concatenate([before, after])
    onset = cascade_onset(series)
    assert onset is not None
    assert abs(onset - boundary) <= 1


def test_cascade_onset_flat_series_is_censored():
    assert cascade_onset([0.3, 0.3, 0.3, 0.3, 0.3]) is None


def test_cascade_onset_too_short_is_censored():
    # Length 3 cannot form two segments of size 2.
    assert cascade_onset([0.0, 1.0, 1.0], min_size=2) is None


def test_cascade_onset_respects_min_size():
    # The true step is at index 1, but min_size=3 forbids a segment that short,
    # so the onset must land in the admissible interior range.
    series = [0.0] + [1.0] * 9
    onset = cascade_onset(series, min_size=3)
    assert onset is not None
    assert 3 <= onset <= 7


def test_cascade_onset_min_size_validation():
    with pytest.raises(ValueError):
        cascade_onset([0.0, 1.0, 2.0], min_size=0)


def test_cascade_onset_returns_plain_int():
    onset = cascade_onset([0.0, 0.0, 1.0, 1.0])
    assert isinstance(onset, int)
    assert onset == 2


def test_cascade_onset_ruptures_backend_if_available():
    ruptures = pytest.importorskip("ruptures")
    assert ruptures is not None  # backend present; exercise the real detector
    series = [0.0] * 6 + [1.0] * 6
    onset = cascade_onset(series)
    assert onset is not None
    assert abs(onset - 6) <= 1


# --------------------------------------------------------------------------- contagion_index


def test_contagion_index_basic():
    # agent 0: exposed, committee flip, no solo flip -> seed-attributable
    # agent 1: exposed, committee flip, also solo flip -> not attributable
    # agent 2: exposed, no committee flip -> not in denominator
    # agent 3: not exposed, committee flip -> excluded
    solo_flip = [False, True, False, False]
    committee_flip = [True, True, False, True]
    seed_present = [True, True, True, False]
    # denominator = exposed & committee_flip = agents 0 and 1 -> 2
    # numerator = exposed & committee_flip & not solo = agent 0 -> 1
    assert contagion_index(solo_flip, committee_flip, seed_present) == 0.5


def test_contagion_index_all_attributable():
    solo_flip = [False, False, False]
    committee_flip = [True, True, True]
    seed_present = [True, True, True]
    assert contagion_index(solo_flip, committee_flip, seed_present) == 1.0


def test_contagion_index_empty_denominator_returns_zero():
    solo_flip = [True, False]
    committee_flip = [False, False]
    seed_present = [True, True]
    assert contagion_index(solo_flip, committee_flip, seed_present) == 0.0


def test_contagion_index_shape_mismatch_raises():
    with pytest.raises(ValueError):
        contagion_index([True, False], [True], [True, False])


# --------------------------------------------------------------------------- deference_rate


def test_deference_rate_scalar_seed():
    # seeded answer is "B" (a wrong shortcut)
    solo_correct = [True, True, True, False]
    committee_answer = ["B", "A", "B", "B"]
    # solo-correct agents: 0, 1, 2 -> denominator 3
    # of those, matched seed "B": agents 0 and 2 -> numerator 2
    # agent 3 matched seed but was not solo-correct, so excluded
    assert deference_rate(solo_correct, committee_answer, "B") == pytest.approx(2 / 3)


def test_deference_rate_per_agent_seed_array():
    solo_correct = [True, True]
    committee_answer = [1, 2]
    seeded_answer = [1, 3]
    # agent 0 solo-correct and matched its seed (1) -> deferred
    # agent 1 solo-correct but 2 != seed 3 -> not deferred
    assert deference_rate(solo_correct, committee_answer, seeded_answer) == 0.5


def test_deference_rate_no_solo_correct_returns_zero():
    solo_correct = [False, False]
    committee_answer = ["B", "B"]
    assert deference_rate(solo_correct, committee_answer, "B") == 0.0


def test_deference_rate_none_defer():
    solo_correct = [True, True, True]
    committee_answer = ["A", "C", "D"]
    assert deference_rate(solo_correct, committee_answer, "B") == 0.0


# --------------------------------------------------------------------------- confidence_trajectory


def _turn(idx, conf):
    return Turn(turn_index=idx, agent_id=f"a{idx}", content="", confidence=conf)


def test_confidence_trajectory_orders_by_turn_index():
    turns = [_turn(2, 0.9), _turn(0, 0.5), _turn(1, 0.7)]
    traj = confidence_trajectory(turns)
    assert np.allclose(traj, [0.5, 0.7, 0.9])


def test_confidence_trajectory_none_becomes_nan():
    turns = [_turn(0, 0.5), _turn(1, None), _turn(2, 0.8)]
    traj = confidence_trajectory(turns)
    assert traj[0] == 0.5
    assert np.isnan(traj[1])
    assert traj[2] == 0.8


def test_confidence_trajectory_empty():
    traj = confidence_trajectory([])
    assert isinstance(traj, np.ndarray)
    assert traj.size == 0


def test_confidence_trajectory_is_float_array():
    turns = [_turn(0, 1), _turn(1, 0)]
    traj = confidence_trajectory(turns)
    assert traj.dtype == np.float64
