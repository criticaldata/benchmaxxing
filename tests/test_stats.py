"""Tests for benchmaxxing.stats.

Every non-guarded wrapper is checked against a known-answer fixture computed independently
(by hand or via a second library path). The two statsmodels-backed functions are exercised
only when statsmodels is importable (pytest.importorskip), and their ImportError guard is
checked when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing import stats


def test_mcnemar_matches_binomial_reference():
    # b=10, c=3 -> exact two-sided binomial p over 13 discordant pairs, k=3.
    from scipy.stats import binomtest

    res = stats.mcnemar(10, 3)
    expected = binomtest(3, 13, 0.5, "two-sided").pvalue
    assert res.statistic == 3.0
    assert res.pvalue == pytest.approx(expected)
    assert res.pvalue == pytest.approx(0.09228515625, rel=1e-9)


def test_mcnemar_symmetric_in_arguments():
    assert stats.mcnemar(10, 3).pvalue == pytest.approx(stats.mcnemar(3, 10).pvalue)


def test_mcnemar_no_discordant_pairs():
    res = stats.mcnemar(0, 0)
    assert res.statistic == 0.0
    assert res.pvalue == 1.0


def test_cochran_q_reduces_to_mcnemar_chisquare_for_two_conditions():
    # For k=2, Q == (b - c)^2 / (b + c) with b = (1,0) count, c = (0,1) count.
    # Build columns: 6 rows (1,0), 2 rows (0,1), plus some concordant rows (dropped by Q).
    rows = (
        [[1, 0]] * 6
        + [[0, 1]] * 2
        + [[1, 1]] * 3
        + [[0, 0]] * 4
    )
    m = np.array(rows, dtype=int)
    res = stats.cochran_q(m)
    b, c = 6, 2
    expected_q = (b - c) ** 2 / (b + c)
    assert res.df == 1
    assert res.statistic == pytest.approx(expected_q)


def test_cochran_q_known_three_condition_value():
    # Hand-computed example, 4 subjects x 3 conditions.
    m = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )
    # C_j = [3, 3, 1], total = 7; R_i = [2, 1, 3, 1], sum R^2 = 4+1+9+1 = 15.
    # numer = (k-1)[k*sum(C^2) - total^2] = 2*[3*(9+9+1) - 49] = 2*[57-49] = 16
    # denom = k*total - sum(R^2) = 3*7 - 15 = 6 -> Q = 16/6.
    res = stats.cochran_q(m)
    assert res.statistic == pytest.approx(16.0 / 6.0)
    assert res.df == 2

    from scipy.stats import chi2

    assert res.pvalue == pytest.approx(chi2.sf(16.0 / 6.0, 2))


def test_cochran_q_degenerate_all_constant_rows():
    m = np.array([[1, 1], [0, 0], [1, 1]], dtype=int)
    res = stats.cochran_q(m)
    assert res.statistic == 0.0
    assert res.pvalue == 1.0


def test_cochran_q_rejects_non_binary():
    with pytest.raises(ValueError):
        stats.cochran_q(np.array([[0, 2], [1, 0]]))


def test_fisher_exact_known_answer():
    res = stats.fisher_exact([[8, 2], [1, 5]])
    assert res.oddsratio == pytest.approx(20.0)
    assert res.pvalue == pytest.approx(0.034965034965034975, rel=1e-9)


def test_bootstrap_ci_point_and_bracketing():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=5.0, scale=1.0, size=500)
    point, low, high = stats.bootstrap_ci(data, np.mean, n_boot=1000, ci=0.95, seed=0)
    assert point == pytest.approx(float(np.mean(data)))
    assert low < point < high
    # True mean (5.0) should sit inside a 95% CI for a sample this size.
    assert low < 5.0 < high


def test_bootstrap_ci_is_deterministic_with_seed():
    data = np.arange(50, dtype=float)
    a = stats.bootstrap_ci(data, np.median, n_boot=500, seed=7)
    b = stats.bootstrap_ci(data, np.median, n_boot=500, seed=7)
    assert a == b


def test_bootstrap_ci_validates_inputs():
    with pytest.raises(ValueError):
        stats.bootstrap_ci(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        stats.bootstrap_ci(np.array([]))
    with pytest.raises(ValueError):
        stats.bootstrap_ci(np.array([1.0, 2.0]), ci=1.5)


def test_phi_coefficient_perfect_and_inverse():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert stats.phi_coefficient(y, y) == pytest.approx(1.0)
    assert stats.phi_coefficient(y, 1 - y) == pytest.approx(-1.0)


def test_phi_coefficient_matches_manual_formula():
    # 2x2 confusion: a=TN, b=FP, c=FN, d=TP. phi = (ad - bc)/sqrt((a+b)(c+d)(a+c)(b+d)).
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    d = int(np.sum((y_true == 1) & (y_pred == 1)))
    a = int(np.sum((y_true == 0) & (y_pred == 0)))
    b = int(np.sum((y_true == 0) & (y_pred == 1)))
    c = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    expected = (a * d - b * c) / denom
    assert stats.phi_coefficient(y_true, y_pred) == pytest.approx(expected)


def test_jaccard_known_answer():
    ya = np.array([1, 1, 0, 0, 1])
    yb = np.array([1, 0, 0, 1, 1])
    # intersection of 1s = {0, 4} -> 2; union of 1s = {0, 1, 3, 4} -> 4; 2/4 = 0.5.
    assert stats.jaccard(ya, yb) == pytest.approx(0.5)


def test_cohen_kappa_perfect_and_chance():
    r = np.array([0, 1, 1, 0, 1])
    assert stats.cohen_kappa(r, r) == pytest.approx(1.0)
    # Complete disagreement on a balanced-ish pair gives kappa <= 0.
    assert stats.cohen_kappa(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) <= 0.0


def test_multiple_comparison_bh_matches_scipy():
    from scipy.stats import false_discovery_control

    ps = np.array([0.01, 0.04, 0.03, 0.005, 0.2])
    res = stats.multiple_comparison(ps, method="bh", alpha=0.05)
    expected = false_discovery_control(ps, method="bh")
    assert np.allclose(res.pvalues_adjusted, expected)
    assert res.reject.tolist() == (expected <= 0.05).tolist()


def test_multiple_comparison_holm_known_answer():
    ps = np.array([0.01, 0.04, 0.03, 0.005, 0.2])
    res = stats.multiple_comparison(ps, method="holm", alpha=0.05)
    # Step-down Holm: sorted 0.005*5, 0.01*4, 0.03*3, 0.04*2(->max 0.09), 0.2*1.
    expected = np.array([0.04, 0.09, 0.09, 0.025, 0.2])
    assert np.allclose(res.pvalues_adjusted, expected)
    # adjusted <= 0.05 at index 0 (0.04) and index 3 (0.025).
    assert res.reject.tolist() == [True, False, False, True, False]


def test_multiple_comparison_clips_to_one():
    ps = np.array([0.6, 0.7, 0.8])
    res = stats.multiple_comparison(ps, method="holm")
    assert np.all(res.pvalues_adjusted <= 1.0)
    assert not res.reject.any()


def test_multiple_comparison_empty_and_bad_method():
    empty = stats.multiple_comparison([], method="bh")
    assert empty.pvalues_adjusted.size == 0
    assert empty.reject.size == 0
    with pytest.raises(ValueError):
        stats.multiple_comparison([0.1, 0.2], method="bogus")


def test_cochran_mantel_haenszel_with_statsmodels():
    pytest.importorskip("statsmodels")
    tables = [
        np.array([[10, 5], [3, 12]]),
        np.array([[8, 6], [4, 10]]),
        np.array([[12, 4], [5, 9]]),
    ]
    res = stats.cochran_mantel_haenszel(tables)
    assert np.isfinite(res.statistic)
    assert 0.0 <= res.pvalue <= 1.0


def test_mixed_effects_logit_with_statsmodels():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    groups = np.repeat(np.arange(6), 10)
    logits = 0.5 + 1.2 * x
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    fe = pd.DataFrame({"intercept": np.ones(n), "x": x})
    result = stats.mixed_effects_logit(y, fe, groups)
    assert result is not None
    assert hasattr(result, "fe_mean")


def test_guarded_functions_raise_clean_importerror_when_statsmodels_absent():
    import importlib.util

    if importlib.util.find_spec("statsmodels") is not None:
        pytest.skip("statsmodels is installed; ImportError guard not exercised here")
    with pytest.raises(ImportError, match="statsmodels"):
        stats.cochran_mantel_haenszel([np.array([[1, 2], [3, 4]])])
    with pytest.raises(ImportError, match="statsmodels"):
        stats.mixed_effects_logit(np.array([0, 1]), np.ones((2, 1)), np.array([0, 1]))
