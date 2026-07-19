"""Tests for the moderator-effect runner (issue 85). Offline, injected critics/moderators."""

from __future__ import annotations

import math

from benchmaxxing.moderator_effect import (
    compare_moderators,
    moderator_effect,
    owner_deference_moderator,
    permissive_moderator,
    strict_majority_moderator,
)
from benchmaxxing.scrutiny import (
    CIRCULAR_PROVENANCE,
    COVERAGE_GAP,
    UNMOORED_PROXY,
    Benchmark,
    StakeholderRole,
)


def _benchmarks():
    return [
        Benchmark(
            owner_role=StakeholderRole.HOSPITAL,
            metric_name="throughput_auc",
            defects={CIRCULAR_PROVENANCE, COVERAGE_GAP},
        ),
        Benchmark(
            owner_role=StakeholderRole.PROCUREMENT,
            metric_name="vendor_score",
            defects={UNMOORED_PROXY},
        ),
    ]


def _critics():
    """A pluralistic panel where exactly one role can see each defect.

    patient sees the coverage gap, provider sees the unmoored proxy, procurement sees the
    circular provenance, hospital sees nothing (it owns the worst benchmark).
    """

    def patient(role, benchmark):
        return {COVERAGE_GAP} & set(benchmark.defects)

    def provider(role, benchmark):
        return {UNMOORED_PROXY} & set(benchmark.defects)

    def procurement(role, benchmark):
        return {CIRCULAR_PROVENANCE} & set(benchmark.defects)

    def hospital(role, benchmark):
        return set()

    return [
        (StakeholderRole.PATIENT, patient),
        (StakeholderRole.PROVIDER, provider),
        (StakeholderRole.HOSPITAL, hospital),
        (StakeholderRole.PROCUREMENT, procurement),
    ]


def test_permissive_moderator_costs_nothing():
    effect = moderator_effect(_benchmarks(), _critics(), permissive_moderator)
    assert effect.rate_without == effect.rate_with
    assert effect.delta == 0.0
    assert effect.n_vetoed == 0
    assert effect.vetoed == {}
    assert effect.concentrates_misses is False
    assert effect.improved is False


def test_strict_majority_moderator_vetoes_single_seat_insights():
    # every defect here is visible to exactly one role, so a >=2-role consensus rule
    # vetoes all of them: detection collapses to zero and every miss traces to the moderator
    effect = moderator_effect(_benchmarks(), _critics(), strict_majority_moderator(min_roles=2))
    assert effect.rate_without == 1.0
    assert effect.rate_with == 0.0
    assert effect.delta == -1.0
    assert effect.n_vetoed == 3
    assert effect.vetoed["throughput_auc"] == {CIRCULAR_PROVENANCE, COVERAGE_GAP}
    assert effect.vetoed["vendor_score"] == {UNMOORED_PROXY}
    assert effect.concentrates_misses is True


def test_owner_deference_moderator_inherits_the_owner_blind_spot():
    # the hospital owner flags nothing, so deferring to it vetoes everything on its benchmark
    effect = moderator_effect(_benchmarks(), _critics(), owner_deference_moderator)
    assert effect.rate_with < effect.rate_without
    assert "throughput_auc" in effect.vetoed
    assert effect.vetoed["throughput_auc"] == {CIRCULAR_PROVENANCE, COVERAGE_GAP}


def test_veto_only_moderator_never_improves():
    for moderator in (permissive_moderator, strict_majority_moderator(2),
                      owner_deference_moderator):
        effect = moderator_effect(_benchmarks(), _critics(), moderator)
        assert effect.delta <= 0.0
        assert effect.improved is False


def test_compare_moderators_ranks_cheapest_first():
    out = compare_moderators(
        _benchmarks(),
        _critics(),
        {
            "permissive": permissive_moderator,
            "majority2": strict_majority_moderator(2),
            "owner": owner_deference_moderator,
        },
    )
    assert out["ranking"][0] == "permissive"          # least detection lost
    # on this fixture the other two tie at maximal loss (both veto every planted defect)
    assert set(out["ranking"][1:]) == {"majority2", "owner"}
    assert out["permissive"].delta == 0.0
    assert out["majority2"].delta == -1.0
    assert out["owner"].delta == -1.0


def test_empty_benchmarks_reports_nan_rates():
    effect = moderator_effect([], _critics(), permissive_moderator)
    assert math.isnan(effect.rate_without) and math.isnan(effect.rate_with)
    assert effect.n_vetoed == 0
