"""Cascade dynamics math: onset detection and seed-contagion metrics.

This module quantifies how a planted shortcut (a "seed" turn) propagates through a committee.
The core signal is a per-turn series that captures the tug of war between two attractors:
agreement with the first committed answer versus agreement with the evidence. When a committee
tips from following the evidence to following the first committed (seeded) answer, that tipping
turn is the "cascade onset".

Functions:
    cascade_onset: single change-point (onset turn) detection on a 1-D series.
    contagion_index: fraction of committee flips attributable to the seed.
    deference_rate: fraction of solo-correct agents that abandoned truth for the seed.
    confidence_trajectory: per-turn confidence series pulled from a list of Turn records.

Reuse-the-originals: cascade_onset uses ruptures when it is importable, and otherwise falls
back to an exact pure-numpy single change-point detector (see cascade_onset for details).
"""

from __future__ import annotations

import numpy as np

from benchmaxxing.schema import Turn


def cascade_onset(series, min_size: int = 1) -> int | None:
    """Detect the single change-point (onset turn) in a 1-D series.

    The input is the turn-level "agreement-with-first-committed minus agreement-with-evidence"
    series. The onset is the index at which the second (post-onset) regime begins, matching the
    ruptures breakpoint convention: an onset of ``k`` splits the series into ``series[:k]`` and
    ``series[k:]``.

    Detection strategy:
        - If ruptures is importable, use ``ruptures.Binseg`` with an L2 (mean-shift) cost and a
          single breakpoint.
        - Otherwise, fall back to an exact pure-numpy detector that maximizes the between-segment
          sum of squares over every admissible split. For a mean-shift (L2) cost this argmax is
          the exact single change-point solution and is equivalent to a CUSUM-style mean-shift
          statistic: it picks the split maximizing ``k*(n-k)/n * (mean_left - mean_right)**2``.

    Args:
        series: 1-D array-like of per-turn values.
        min_size: minimum length of each of the two segments (both sides). Must be >= 1.

    Returns:
        The onset index (an int in ``[min_size, n - min_size]``), or ``None`` when the case is
        censored: the series is shorter than two segments of ``min_size``, or it has (near) zero
        variance so there is no change to locate.
    """
    if min_size < 1:
        raise ValueError("min_size must be >= 1")

    x = np.asarray(series, dtype=float).ravel()
    n = x.size

    # Not enough turns to form two segments of the required size: censored.
    if n < 2 * min_size:
        return None

    # A flat (zero-variance) series carries no change to detect: censored.
    total_ss = float(np.sum((x - x.mean()) ** 2))
    if total_ss <= 1e-12:
        return None

    try:
        import ruptures
    except ImportError:
        ruptures = None

    if ruptures is not None:
        onset = _ruptures_onset(ruptures, x, min_size)
        if onset is not None:
            return onset

    return _numpy_onset(x, min_size)


def _ruptures_onset(ruptures, x: np.ndarray, min_size: int) -> int | None:
    """Locate a single breakpoint with ruptures. Returns None on a degenerate result."""
    signal = x.reshape(-1, 1)
    algo = ruptures.Binseg(model="l2", min_size=min_size, jump=1).fit(signal)
    bkps = algo.predict(n_bkps=1)
    # predict returns end-exclusive breakpoints ending with n; the onset is the first one.
    k = int(bkps[0])
    if k <= 0 or k >= x.size:
        return None
    return k


def _numpy_onset(x: np.ndarray, min_size: int) -> int:
    """Exact single change-point via the maximum between-segment sum of squares (mean shift)."""
    n = x.size
    # Prefix sums let every candidate split be scored in O(1).
    csum = np.concatenate(([0.0], np.cumsum(x)))
    ks = np.arange(min_size, n - min_size + 1)
    left_mean = csum[ks] / ks
    right_mean = (csum[n] - csum[ks]) / (n - ks)
    # Between-group sum of squares for a split at k, which the argmax maximizes.
    bss = ks * (n - ks) / n * (left_mean - right_mean) ** 2
    return int(ks[int(np.argmax(bss))])


def contagion_index(solo_flip, committee_flip, seed_present) -> float:
    """Fraction of committee flips that are attributable to the seed, among exposed agents.

    A committee flip is attributed to the seed when the agent flipped in committee but not on its
    own (solo), so the flip cannot be explained by the agent's independent reasoning. The
    population is restricted to agents that were exposed to the seed.

    Args:
        solo_flip: per-agent booleans, True if the agent changed its answer in the solo condition.
        committee_flip: per-agent booleans, True if the agent changed its answer in committee.
        seed_present: per-agent booleans, True if the agent was exposed to the seed.

    Returns:
        count(seed & committee_flip & not solo_flip) / count(seed & committee_flip), or 0.0 when
        no exposed agent flipped in committee (empty denominator).
    """
    solo = np.asarray(solo_flip, dtype=bool)
    comm = np.asarray(committee_flip, dtype=bool)
    seed = np.asarray(seed_present, dtype=bool)
    if not (solo.shape == comm.shape == seed.shape):
        raise ValueError("solo_flip, committee_flip, seed_present must share the same shape")

    exposed_committee_flips = seed & comm
    denom = int(np.count_nonzero(exposed_committee_flips))
    if denom == 0:
        return 0.0

    seed_attributable = exposed_committee_flips & ~solo
    return float(np.count_nonzero(seed_attributable)) / denom


def deference_rate(solo_correct, committee_answer, seeded_answer) -> float:
    """Fraction of solo-correct agents that abandoned their answer to match the seeded answer.

    An agent defers when it was correct on its own (solo) yet, in committee, gave the seeded
    (planted shortcut) answer. Only agents that were solo-correct can abandon a solo-correct
    answer, so the denominator is the number of solo-correct agents (the population at risk).

    Args:
        solo_correct: per-agent booleans, True if the agent was correct in the solo condition.
        committee_answer: per-agent answers given in committee.
        seeded_answer: the planted answer, either a scalar (broadcast to all agents) or a
            per-agent array.

    Returns:
        count(solo_correct & committee_answer == seeded_answer) / count(solo_correct), or 0.0 when
        no agent was solo-correct (empty denominator).
    """
    solo = np.asarray(solo_correct, dtype=bool)
    committee = np.asarray(committee_answer, dtype=object)
    seeded = np.asarray(seeded_answer, dtype=object)
    if seeded.ndim == 0:
        seeded = np.broadcast_to(seeded, committee.shape)
    if not (solo.shape == committee.shape == seeded.shape):
        raise ValueError("solo_correct, committee_answer, seeded_answer must share the same shape")

    matched_seed = np.asarray(committee == seeded, dtype=bool)
    denom = int(np.count_nonzero(solo))
    if denom == 0:
        return 0.0

    deferred = solo & matched_seed
    return float(np.count_nonzero(deferred)) / denom


def confidence_trajectory(turns: list[Turn]) -> np.ndarray:
    """Extract the per-turn confidence series from a list of schema.Turn.

    Turns are ordered by ``turn_index`` so the trajectory is well defined even if the input list
    is unordered. A turn with no confidence (``None``) becomes ``numpy.nan`` so per-turn alignment
    is preserved.

    Args:
        turns: iterable of ``benchmaxxing.schema.Turn``.

    Returns:
        A 1-D float ``numpy.ndarray`` of confidences ordered by turn index (empty when no turns).
    """
    ordered = sorted(turns, key=lambda t: t.turn_index)
    conf = [np.nan if t.confidence is None else float(t.confidence) for t in ordered]
    return np.asarray(conf, dtype=float)


__all__ = [
    "cascade_onset",
    "contagion_index",
    "deference_rate",
    "confidence_trajectory",
]
