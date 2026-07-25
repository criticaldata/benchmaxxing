"""Tests for benchmaxxing.budget: deterministic subsampling and the call-budget guard.

Everything here runs offline against gateway.MockBackend; no API key or network is needed.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

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
    # Unbounded (or over-sized) returns every case, in rank order rather than pool order.
    cases = list(range(5))
    assert set(bd.subsample_cases(cases, bd.RunBudget(max_cases=None))) == set(cases)
    assert set(bd.subsample_cases(cases, bd.RunBudget(max_cases=100))) == set(cases)


def test_bounded_pilot_is_a_prefix_of_the_unbounded_full_run():
    # The unbounded full run and every bounded pilot share one rank order, so a pilot is a strict
    # positional prefix of the full run, not merely a subset (Agastya's #263 review point).
    cases = list(range(50))
    full = bd.subsample_cases(cases, bd.RunBudget(max_cases=None, seed=7))
    pilot = bd.subsample_cases(cases, bd.RunBudget(max_cases=10, seed=7))
    assert pilot == full[: len(pilot)]


def test_subsample_is_invariant_to_input_order():
    # The selection depends on case content, not on the order the pool arrives in: a pilot and
    # the full run stay a nested pair even if an upstream step reorders the pool between them.
    cases = list(range(50))
    shuffled = cases[:]
    random.Random(99).shuffle(shuffled)
    b = bd.RunBudget(max_cases=10, seed=7)
    assert set(bd.subsample_cases(cases, b)) == set(bd.subsample_cases(shuffled, b))


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


def test_accountant_from_budget_takes_the_run_wide_cap():
    acc = bd.CallAccountant.from_budget(bd.RunBudget(max_calls=5, seed=1))
    assert acc.max_calls == 5


# --------------------------------------------------------------------------- BudgetedBackend

def test_budgeted_backend_records_each_call():
    acc = bd.CallAccountant()
    inner = gateway.MockBackend()
    backend = bd.BudgetedBackend(inner, acc, model="mock")
    backend.complete("a")
    backend.complete("b")
    assert acc.total_calls == 2
    assert acc.calls_per_model == {"mock": 2}
    assert inner.n_calls == 2


def test_budgeted_backend_raises_once_max_calls_is_reached():
    acc = bd.CallAccountant(max_calls=2)
    inner = gateway.MockBackend()
    backend = bd.BudgetedBackend(inner, acc, model="mock")
    backend.complete("a")
    backend.complete("b")
    with pytest.raises(bd.BudgetExceeded):
        backend.complete("c")
    # the rejected call must not have reached the wrapped backend or been tallied
    assert acc.total_calls == 2
    assert inner.n_calls == 2


def test_budgeted_backend_shared_across_models_enforces_one_combined_cap():
    # The cap is run-wide: one accountant shared by two models enforces a single combined budget.
    acc = bd.CallAccountant(max_calls=3)
    a = bd.BudgetedBackend(gateway.MockBackend(), acc, model="a")
    b = bd.BudgetedBackend(gateway.MockBackend(), acc, model="b")
    a.complete("1")
    b.complete("2")
    a.complete("3")
    with pytest.raises(bd.BudgetExceeded):
        b.complete("4")
    assert acc.total_calls == 3
    assert acc.calls_per_model == {"a": 2, "b": 1}


def test_independent_budgets_do_not_starve_each_other():
    # Models whose budgets must be independent get their own accountant, so one model spending
    # its whole budget never refuses another model's first call.
    acc_a = bd.CallAccountant(max_calls=10)
    acc_b = bd.CallAccountant(max_calls=10)
    a = bd.BudgetedBackend(gateway.MockBackend(), acc_a, model="a")
    b = bd.BudgetedBackend(gateway.MockBackend(), acc_b, model="b")
    # b exhausts its own budget entirely...
    for _ in range(10):
        b.complete("x")
    # ...and a still has every one of its own calls available.
    a.complete("y")
    assert acc_a.total_calls == 1
    assert acc_b.total_calls == 10


def test_a_failed_backend_call_still_counts_against_the_budget():
    # Claim-before-spend: a call that reaches the backend and errors still consumes budget, so a
    # flapping backend cannot retry-storm past the cap. Conservative by design (never overspend).
    class Boom(gateway.Backend):
        def complete(self, prompt, image=None, decoding=None):
            raise RuntimeError("backend down")

    acc = bd.CallAccountant(max_calls=5)
    backend = bd.BudgetedBackend(Boom(), acc, model="mock")
    with pytest.raises(RuntimeError):
        backend.complete("x")
    assert acc.total_calls == 1


def test_cap_is_atomic_under_concurrent_calls():
    # Every real experiment script fans calls out through a ThreadPoolExecutor, so a shared
    # accountant must never let concurrent callers overspend the cap.
    cap = 100
    acc = bd.CallAccountant(max_calls=cap)
    backend = bd.BudgetedBackend(gateway.MockBackend(), acc, model="mock")

    def one_call():
        try:
            backend.complete("p")
            return True
        except bd.BudgetExceeded:
            return False

    n_tasks = 500
    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(lambda _: one_call(), range(n_tasks)))

    # Exactly `cap` calls succeed, the rest are refused, and nothing is double-counted or lost.
    assert acc.total_calls == cap
    assert sum(outcomes) == cap
    assert len(outcomes) == n_tasks


# --------------------------------------------------------------------------- partial-result flush

def test_a_run_stops_cleanly_and_flushes_partial_results_on_budget_exceeded():
    # A minimal stand-in for a stage runner: iterate cases, call the budgeted backend, stop and
    # keep whatever was collected so far the moment the budget is hit.
    acc = bd.CallAccountant(max_calls=3)
    backend = bd.BudgetedBackend(gateway.MockBackend(), acc, model="mock")
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
