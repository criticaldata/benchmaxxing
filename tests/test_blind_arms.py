"""Tests for the blind-metric evaluation arms (proxy arm + awareness arm).

Everything here is pure and offline: cases are plain dicts and the model / proxy / truth
callables are deterministic mocks, so no API keys or real data are needed.
"""

from __future__ import annotations

import math

import pytest

# blind_arms -> blind_metric imports scipy/numpy at load time; skip cleanly if unavailable.
pytest.importorskip("scipy")
pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from benchmaxxing import blind_arms  # noqa: E402
from benchmaxxing.blind_arms import ProxyArmResult, misaligned_proxy_run  # noqa: E402
from benchmaxxing.blind_metric import BlindMetricUptake  # noqa: E402


# --------------------------------------------------------------------------------- fixtures


def _decorrelated_cases(n: int = 80, seed: int = 0) -> list[dict]:
    """Cases carrying a ``proxy`` and a ``truth`` field that are (by construction) independent."""
    rng = np.random.default_rng(seed)
    proxy = rng.normal(size=n)
    truth = rng.normal(size=n)
    return [
        {"id": i, "proxy": float(proxy[i]), "truth": float(truth[i])}
        for i in range(n)
    ]


def _proxy_fn(case: dict) -> float:
    return case["proxy"]


def _truth_fn(case: dict) -> float:
    return case["truth"]


def _follows_proxy(case: dict) -> float:
    # A model that games the misaligned proxy: decision tracks the proxy, ignores truth.
    return case["proxy"]


def _follows_truth(case: dict) -> float:
    # A model that solves the real objective: decision tracks truth, ignores the proxy.
    return case["truth"]


# ------------------------------------------------------------------------ misaligned_proxy_run


def test_proxy_run_returns_result_and_records_decisions():
    cases = _decorrelated_cases()
    res = misaligned_proxy_run(cases, _follows_proxy, _proxy_fn, _truth_fn)
    assert isinstance(res, ProxyArmResult)
    assert isinstance(res.uptake, BlindMetricUptake)
    assert res.n == len(cases)
    # Decisions were recorded per case, in order, from model_fn.
    assert res.decisions == tuple(_follows_proxy(c) for c in cases)
    assert res.proxy_values == tuple(c["proxy"] for c in cases)
    assert res.truth_values == tuple(c["truth"] for c in cases)


def test_proxy_run_positive_uptake_when_model_follows_proxy():
    cases = _decorrelated_cases()
    res = misaligned_proxy_run(cases, _follows_proxy, _proxy_fn, _truth_fn)
    # Decisions are the proxy: near-perfect correlation with the proxy, near-zero with truth.
    assert res.uptake.corr_hidden > 0.99
    assert abs(res.uptake.corr_ground) < 0.5
    # Positive uptake_delta == the model drifted toward the misaligned proxy over truth.
    assert res.uptake_delta > 0.0


def test_proxy_run_negative_uptake_when_model_follows_truth():
    cases = _decorrelated_cases()
    res = misaligned_proxy_run(cases, _follows_truth, _proxy_fn, _truth_fn)
    assert res.uptake.corr_ground > 0.99
    assert abs(res.uptake.corr_hidden) < 0.5
    # Solving the real objective yields no uptake of the proxy.
    assert res.uptake_delta < 0.0


def test_proxy_run_supports_spearman_method():
    # Monotonic-but-nonlinear tracking of the proxy is caught by Spearman.
    cases = [{"id": i, "proxy": float(i), "truth": 0.0} for i in range(1, 30)]
    res = misaligned_proxy_run(
        cases,
        model_fn=lambda c: c["proxy"] ** 3,
        proxy_fn=_proxy_fn,
        truth_fn=_truth_fn,
        method="spearman",
    )
    assert res.uptake.method == "spearman"
    assert res.uptake.corr_hidden > 0.999
    # truth is constant here, so its correlation is undefined.
    assert math.isnan(res.uptake.corr_ground)


def test_proxy_run_constant_decisions_give_nan_uptake():
    cases = _decorrelated_cases(n=20)
    res = misaligned_proxy_run(cases, model_fn=lambda c: 1.0, proxy_fn=_proxy_fn, truth_fn=_truth_fn)
    assert math.isnan(res.uptake_delta)
    assert res.n == 20


def test_proxy_run_accepts_bool_decisions():
    cases = [{"id": i, "proxy": float(i), "truth": float(-i)} for i in range(10)]
    res = misaligned_proxy_run(
        cases,
        model_fn=lambda c: c["proxy"] > 4,  # bool decision
        proxy_fn=_proxy_fn,
        truth_fn=_truth_fn,
    )
    assert set(res.decisions) <= {0.0, 1.0}
    assert res.uptake_delta > 0.0


# --------------------------------------------------------------------------- test_awareness_run


def test_awareness_run_positive_delta_when_awareness_suppresses_uptake():
    cases = _decorrelated_cases()
    # Aware model resists the proxy (solves truth); unaware model games the proxy.
    delta = blind_arms.test_awareness_run(
        cases,
        aware_model_fn=_follows_truth,
        unaware_model_fn=_follows_proxy,
        proxy_fn=_proxy_fn,
        truth_fn=_truth_fn,
    )
    assert isinstance(delta, float)
    # unaware games the proxy (high uptake), aware does not (low/negative): delta > 0.
    assert delta > 0.0


def test_awareness_run_matches_manual_arm_difference():
    cases = _decorrelated_cases(seed=3)
    aware = misaligned_proxy_run(cases, _follows_truth, _proxy_fn, _truth_fn)
    unaware = misaligned_proxy_run(cases, _follows_proxy, _proxy_fn, _truth_fn)
    delta = blind_arms.test_awareness_run(
        cases,
        aware_model_fn=_follows_truth,
        unaware_model_fn=_follows_proxy,
        proxy_fn=_proxy_fn,
        truth_fn=_truth_fn,
    )
    # awareness delta == unaware uptake minus aware uptake.
    assert delta == pytest.approx(unaware.uptake_delta - aware.uptake_delta)


def test_awareness_run_near_zero_when_arms_behave_identically():
    cases = _decorrelated_cases(seed=7)
    delta = blind_arms.test_awareness_run(
        cases,
        aware_model_fn=_follows_proxy,
        unaware_model_fn=_follows_proxy,
        proxy_fn=_proxy_fn,
        truth_fn=_truth_fn,
    )
    # Both arms game the proxy the same amount: awareness changed nothing.
    assert delta == pytest.approx(0.0, abs=1e-9)


def test_awareness_run_is_not_collected_as_a_test():
    # It is an API function (issue 84) named test_*, so it is marked to escape pytest collection.
    assert blind_arms.test_awareness_run.__test__ is False
