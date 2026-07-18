"""Tests for the paired flip-rate power / sample-size helpers in benchmaxxing.stats.

Covers :func:`benchmaxxing.stats.required_pairs` and its inverse companion
:func:`benchmaxxing.stats.achieved_power`. The closed-form sample size is checked against
an independent recomputation from scipy quantiles and against a hand-computed fixture, plus
the expected monotonicity in effect size, power, and alpha.
"""

from __future__ import annotations

import math

import pytest

from benchmaxxing import stats


def _closed_form_pairs(psi: float, delta: float, power: float, alpha: float) -> int:
    """Independent recomputation of the McNemar sample-size formula (ceil)."""
    from scipy.stats import norm

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    numer = z_alpha * math.sqrt(psi) + z_beta * math.sqrt(psi - delta * delta)
    return math.ceil((numer * numer) / (delta * delta))


def test_required_pairs_matches_hand_computed_fixture():
    # psi=0.4, delta=0.2, power=0.8, alpha=0.05.
    #   z_{0.975} = 1.9599639845, z_{0.80} = 0.8416212336
    #   bracket = 1.9599639845*sqrt(0.4) + 0.8416212336*sqrt(0.4 - 0.04)
    #           = 1.9599639845*0.632455532 + 0.8416212336*0.6
    #           = 1.23959009 + 0.50497274 = 1.74456283
    #   n = 1.74456283**2 / 0.2**2 = 3.04349946 / 0.04 = 76.0875 -> ceil = 77
    assert stats.required_pairs(0.4, 0.2, power=0.8, alpha=0.05) == 77


def test_required_pairs_matches_independent_closed_form():
    for psi, delta, power, alpha in [
        (0.4, 0.2, 0.8, 0.05),
        (0.3, 0.1, 0.9, 0.05),
        (0.6, 0.25, 0.8, 0.01),
        (0.5, 0.15, 0.85, 0.10),
    ]:
        assert stats.required_pairs(psi, delta, power, alpha) == _closed_form_pairs(
            psi, delta, power, alpha
        )


def test_required_pairs_increases_as_effect_shrinks():
    n_big = stats.required_pairs(0.4, 0.20)
    n_mid = stats.required_pairs(0.4, 0.15)
    n_small = stats.required_pairs(0.4, 0.10)
    assert n_big < n_mid < n_small


def test_required_pairs_increases_as_power_rises():
    n_low = stats.required_pairs(0.4, 0.2, power=0.7)
    n_mid = stats.required_pairs(0.4, 0.2, power=0.8)
    n_high = stats.required_pairs(0.4, 0.2, power=0.9)
    assert n_low < n_mid < n_high


def test_required_pairs_increases_as_alpha_shrinks():
    n_lax = stats.required_pairs(0.4, 0.2, alpha=0.10)
    n_strict = stats.required_pairs(0.4, 0.2, alpha=0.01)
    assert n_lax < n_strict


def test_required_pairs_effect_sign_is_irrelevant():
    assert stats.required_pairs(0.4, 0.2) == stats.required_pairs(0.4, -0.2)


def test_required_pairs_validation():
    with pytest.raises(ValueError):
        stats.required_pairs(0.0, 0.1)          # p_discordant must be > 0
    with pytest.raises(ValueError):
        stats.required_pairs(1.5, 0.1)          # p_discordant must be <= 1
    with pytest.raises(ValueError):
        stats.required_pairs(0.4, 0.0)          # effect must be non-zero
    with pytest.raises(ValueError):
        stats.required_pairs(0.4, 0.5)          # |effect| > p_discordant
    with pytest.raises(ValueError):
        stats.required_pairs(0.4, 0.2, power=1.0)
    with pytest.raises(ValueError):
        stats.required_pairs(0.4, 0.2, alpha=0.0)


def test_achieved_power_round_trips_with_required_pairs():
    for psi, delta, power, alpha in [
        (0.4, 0.2, 0.8, 0.05),
        (0.3, 0.1, 0.9, 0.05),
        (0.6, 0.25, 0.85, 0.01),
    ]:
        n = stats.required_pairs(psi, delta, power, alpha)
        got = stats.achieved_power(n, psi, delta, alpha)
        # Ceiling rounding only ever adds pairs, so achieved power meets or beats the target.
        assert got >= power
        # ... but not wildly so: one fewer pair should fall just short of the target.
        assert stats.achieved_power(n - 1, psi, delta, alpha) < power


def test_achieved_power_matches_hand_computed_fixture():
    # n=77, psi=0.4, delta=0.2, alpha=0.05.
    #   z = (sqrt(77)*0.2 - 1.9599639845*sqrt(0.4)) / sqrt(0.36)
    #     = (8.77496439*0.2 - 1.23959009) / 0.6 = (1.75499288 - 1.23959009)/0.6 = 0.85900465
    #   Phi(0.85900465) ~= 0.80487
    assert stats.achieved_power(77, 0.4, 0.2, 0.05) == pytest.approx(0.80487, abs=1e-4)


def test_achieved_power_increases_with_n():
    powers = [stats.achieved_power(n, 0.4, 0.2) for n in (40, 60, 80, 120)]
    assert all(a < b for a, b in zip(powers, powers[1:]))
    assert all(0.0 < p < 1.0 for p in powers)


def test_achieved_power_validation():
    with pytest.raises(ValueError):
        stats.achieved_power(0, 0.4, 0.2)
    with pytest.raises(ValueError):
        stats.achieved_power(50, 1.2, 0.2)
    with pytest.raises(ValueError):
        stats.achieved_power(50, 0.4, 0.0)
    with pytest.raises(ValueError):
        stats.achieved_power(50, 0.4, 0.5)
