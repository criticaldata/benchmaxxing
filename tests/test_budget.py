"""Tests for benchmaxxing.budget: deterministic subsampling and the call-budget guard.

Everything here runs offline against gateway.MockBackend; no API key or network is needed.
"""

from __future__ import annotations

import pytest

from benchmaxxing import budget as bd
from benchmaxxing import gateway


# --------------------------------------------------------------------------- subsample_cases

def test_subsample_is_deterministic_for_a_fixed_seed():
    cases = list(range(50))
    b = bd.RunBudget(max_cases=10, seed=7)
    first = bd.subsample_cases(cases, b)
    second = bd.subsample_cases(cases, b)
    assert first == second
    assert len(first) == 10


def test_smaller_max_cases_is_a_prefix_of_a_larger_run():
    cases = list(range(50))
    small = bd.subsample_cases(cases, bd.RunBudget(max_cases=5, seed=3))
    large = bd.subsample_cases(cases, bd.RunBudget(max_cases=20, seed=3))
    assert small == large[: len(small)]


def test_different_seeds_can_pick_different_subsets():
    cases = list(range(50))
    a = bd.subsample_cases(cases, bd.RunBudget(max_cases=10, seed=1))
    b = bd.subsample_cases(cases, bd.RunBudget(max_cases=10, seed=2))
    assert a != b


def test_max_cases_none_or_over_pool_size_returns_everything():
    cases = list(range(5))
    assert bd.subsample_cases(cases, bd.RunBudget(max_cases=None)) == cases
    assert bd.subsample_cases(cases, bd.RunBudget(max_cases=100)) == cases


# --------------------------------------------------------------------------- CallAccountant

def test_call_accountant_tallies_per_model_and_total():
    acc = bd.CallAccountant()
    acc.record("gemini-flash")
    acc.record("gemini-flash")
    acc.record("gemini-lite", tokens=42)
    summary = acc.summary()
    assert summary["total_calls"] == 3
    assert summary["calls_per_model"] == {"gemini-flash": 2, "gemini-lite": 1}
    assert summary["tokens_per_model"] == {"gemini-lite": 42}


# --------------------------------------------------------------------------- BudgetedBackend

def test_budgeted_backend_records_each_call():
    acc = bd.CallAccountant()
    inner = gateway.MockBackend()
    backend = bd.BudgetedBackend(inner, acc, model="mock", max_calls=None)
    backend.complete("a")
    backend.complete("b")
    assert acc.total_calls == 2
    assert acc.calls_per_model == {"mock": 2}
    assert inner.n_calls == 2


def test_budgeted_backend_raises_once_max_calls_is_reached():
    acc = bd.CallAccountant()
    backend = bd.BudgetedBackend(gateway.MockBackend(), acc, model="mock", max_calls=2)
    backend.complete("a")
    backend.complete("b")
    with pytest.raises(bd.BudgetExceeded):
        backend.complete("c")
    # the rejected call must not have reached the wrapped backend or been tallied
    assert acc.total_calls == 2


def test_budgeted_backend_shared_across_models_enforces_one_combined_cap():
    acc = bd.CallAccountant()
    a = bd.BudgetedBackend(gateway.MockBackend(), acc, model="a", max_calls=3)
    b = bd.BudgetedBackend(gateway.MockBackend(), acc, model="b", max_calls=3)
    a.complete("1")
    b.complete("2")
    a.complete("3")
    with pytest.raises(bd.BudgetExceeded):
        b.complete("4")
    assert acc.total_calls == 3


# --------------------------------------------------------------------------- partial-result flush

def test_a_run_stops_cleanly_and_flushes_partial_results_on_budget_exceeded():
    # A minimal stand-in for a stage runner: iterate cases, call the budgeted backend, stop and
    # keep whatever was collected so far the moment the budget is hit.
    acc = bd.CallAccountant()
    backend = bd.BudgetedBackend(gateway.MockBackend(), acc, model="mock", max_calls=3)
    cases = list(range(10))
    results = []
    for case in cases:
        try:
            backend.complete(f"case-{case}")
        except bd.BudgetExceeded:
            break
        results.append(case)

    assert results == [0, 1, 2]
    note = bd.truncation_note(planned=len(cases), completed=len(results))
    assert note == "budget truncation: completed 3/10 planned cases"


def test_truncation_note_is_none_when_the_run_completed_everything():
    assert bd.truncation_note(planned=5, completed=5) is None
