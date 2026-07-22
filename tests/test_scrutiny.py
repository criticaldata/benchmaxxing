"""Tests for benchmaxxing.scrutiny (stage-5 benchmark scrutiny panel), issue 19.

The panel is pure logic: critics are injected callables, so every test runs offline with
hand-built benchmarks and synthetic critics and no API keys.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from benchmaxxing.scrutiny import (
    ALL_ROLES,
    CIRCULAR_PROVENANCE,
    CONTAMINATED_DATING,
    COVERAGE_GAP,
    DEFECTS,
    UNMOORED_PROXY,
    Benchmark,
    StakeholderRole,
    compare_panels,
    coverage_gap_estimate,
    flaw_detection_rate,
    plant_defects,
    run_scrutiny_panel,
    scrutiny_diversity_hypothesis,
)

# Each role is sensitive to two defects; the four roles' sensitivities union to all four
# defects, but any single role covers only half of them.
SENSITIVITY = {
    "patient": {COVERAGE_GAP, UNMOORED_PROXY},
    "provider": {UNMOORED_PROXY, CONTAMINATED_DATING},
    "hospital": {CONTAMINATED_DATING, CIRCULAR_PROVENANCE},
    "procurement": {CIRCULAR_PROVENANCE, COVERAGE_GAP},
}


def _critic(role, benchmark):
    """A synthetic critic: detects a planted defect only if its role is sensitive to it."""
    role_name = role.value if isinstance(role, StakeholderRole) else str(role)
    return set(benchmark.defects) & SENSITIVITY[role_name]


def _fully_defective_benchmark(metric="readmission_auc"):
    base = Benchmark(owner_role=StakeholderRole.HOSPITAL, metric_name=metric)
    return plant_defects(base, DEFECTS)


# --------------------------------------------------------------------------- vocabulary


def test_stakeholder_roles_present():
    values = {r.value for r in StakeholderRole}
    assert values == {"patient", "provider", "hospital", "procurement"}
    assert ALL_ROLES == (
        StakeholderRole.PATIENT,
        StakeholderRole.PROVIDER,
        StakeholderRole.HOSPITAL,
        StakeholderRole.PROCUREMENT,
    )


def test_defect_vocabulary():
    assert DEFECTS == {
        "circular_provenance",
        "contaminated_dating",
        "unmoored_proxy",
        "coverage_gap",
    }


# --------------------------------------------------------------------------- plant_defects


def test_plant_defects_returns_new_benchmark_with_defects():
    base = Benchmark(owner_role=StakeholderRole.PATIENT, metric_name="m")
    planted = plant_defects(base, {COVERAGE_GAP, UNMOORED_PROXY})
    assert planted is not base
    assert base.defects == set()                       # original untouched
    assert planted.defects == {COVERAGE_GAP, UNMOORED_PROXY}
    assert planted.metric_name == "m"
    assert planted.owner_role is StakeholderRole.PATIENT


def test_plant_defects_single_name_and_union():
    base = plant_defects(Benchmark(StakeholderRole.PATIENT, "m"), COVERAGE_GAP)
    combined = plant_defects(base, CIRCULAR_PROVENANCE)
    assert combined.defects == {COVERAGE_GAP, CIRCULAR_PROVENANCE}


def test_plant_defects_rejects_unknown():
    with pytest.raises(ValueError):
        plant_defects(Benchmark(StakeholderRole.PATIENT, "m"), "not_a_defect")


# --------------------------------------------------------------------------- run panel


def test_run_scrutiny_panel_catches_and_attributes_defects():
    benchmark = _fully_defective_benchmark()
    panel = [(role, _critic) for role in ALL_ROLES]
    results = run_scrutiny_panel([benchmark], panel)
    assert len(results) == 1
    result = results[0]
    # A full pluralistic panel catches every planted defect.
    assert result.detected == DEFECTS
    assert result.missed == set()
    # coverage_gap is named by exactly the two roles sensitive to it.
    assert result.caught_by[COVERAGE_GAP] == {"patient", "procurement"}
    assert result.caught_by[CONTAMINATED_DATING] == {"provider", "hospital"}
    assert result.roles == ("patient", "provider", "hospital", "procurement")


def test_homogeneous_panel_misses_half():
    benchmark = _fully_defective_benchmark()
    panel = [(StakeholderRole.PROVIDER, _critic) for _ in range(4)]
    result = run_scrutiny_panel([benchmark], panel)[0]
    assert result.detected == {UNMOORED_PROXY, CONTAMINATED_DATING}
    assert result.missed == {COVERAGE_GAP, CIRCULAR_PROVENANCE}


# --------------------------------------------------------------------------- detection rate


def test_flaw_detection_rate_pluralistic_beats_homogeneous():
    benchmark = _fully_defective_benchmark()
    pluralistic = run_scrutiny_panel([benchmark], [(r, _critic) for r in ALL_ROLES])
    homogeneous = run_scrutiny_panel(
        [benchmark], [(StakeholderRole.PROVIDER, _critic) for _ in range(4)]
    )
    plural_rate = flaw_detection_rate(pluralistic)
    hom_rate = flaw_detection_rate(homogeneous)
    assert math.isclose(plural_rate, 1.0)
    assert math.isclose(hom_rate, 0.5)
    assert plural_rate > hom_rate


def test_flaw_detection_rate_nan_when_nothing_planted():
    clean = Benchmark(StakeholderRole.HOSPITAL, "clean")
    results = run_scrutiny_panel([clean], [(r, _critic) for r in ALL_ROLES])
    assert math.isnan(flaw_detection_rate(results))


# --------------------------------------------------------------------------- compare_panels


def test_compare_panels_flags_pluralistic_as_better():
    benchmarks = [_fully_defective_benchmark("m1"), _fully_defective_benchmark("m2")]
    critic_by_role = {role: _critic for role in ALL_ROLES}
    report = compare_panels(
        benchmarks, critic_by_role, homogeneous_role=StakeholderRole.HOSPITAL
    )
    assert math.isclose(report["pluralistic_rate"], 1.0)
    assert math.isclose(report["homogeneous_rate"], 0.5)
    assert math.isclose(report["difference"], 0.5)
    assert report["pluralistic_better"] is True
    assert report["homogeneous_role"] == "hospital"


def test_compare_panels_accepts_string_role_keys():
    benchmarks = [_fully_defective_benchmark()]
    critic_by_role = {role.value: _critic for role in ALL_ROLES}  # string keys
    report = compare_panels(benchmarks, critic_by_role, homogeneous_role="patient")
    assert report["pluralistic_better"] is True


# --------------------------------------------------------------------------- moderator


def test_moderator_requiring_corroboration_favors_diverse_panel():
    def corroboration(benchmark, role_detections):
        counts = Counter()
        for detected in role_detections.values():
            counts.update(detected)
        return {defect for defect, n in counts.items() if n >= 2}

    benchmark = _fully_defective_benchmark()
    # In the pluralistic panel every defect is flagged by exactly two distinct roles.
    pluralistic = run_scrutiny_panel(
        [benchmark], [(r, _critic) for r in ALL_ROLES], moderator=corroboration
    )
    # The homogeneous panel collapses to one distinct role, so nothing is corroborated.
    homogeneous = run_scrutiny_panel(
        [benchmark],
        [(StakeholderRole.PROVIDER, _critic) for _ in range(4)],
        moderator=corroboration,
    )
    assert math.isclose(flaw_detection_rate(pluralistic), 1.0)
    assert math.isclose(flaw_detection_rate(homogeneous), 0.0)


def test_moderator_cannot_invent_detections():
    def liar(benchmark, role_detections):
        return DEFECTS  # claim everything, even undetected defects

    benchmark = plant_defects(Benchmark(StakeholderRole.PROVIDER, "m"), DEFECTS)
    # Only the provider seat runs, so only its two defects are actually flagged.
    result = run_scrutiny_panel(
        [benchmark], [(StakeholderRole.PROVIDER, _critic)], moderator=liar
    )[0]
    assert result.flagged == {UNMOORED_PROXY, CONTAMINATED_DATING}
    assert result.detected == {UNMOORED_PROXY, CONTAMINATED_DATING}


# --------------------------------------------------------------------------- coverage gap


def test_coverage_gap_estimate_basic():
    full = ["mortality", "readmission", "cost", "equity"]
    covered = ["mortality", "cost"]
    assert math.isclose(coverage_gap_estimate(covered, full), 0.5)


def test_coverage_gap_estimate_full_and_empty():
    full = ["a", "b", "c"]
    assert math.isclose(coverage_gap_estimate(full, full), 0.0)       # nothing missed
    assert math.isclose(coverage_gap_estimate([], full), 1.0)         # all missed
    assert math.isclose(coverage_gap_estimate([], []), 0.0)           # empty list, no gap


def test_coverage_gap_ignores_extraneous_covered():
    full = ["a", "b"]
    covered = ["a", "b", "x", "y"]   # extra names outside the held-out list
    assert math.isclose(coverage_gap_estimate(covered, full), 0.0)


# --------------------------------------------------------------------------- hypothesis


def test_hypothesis_documents_diversity_claim():
    text = scrutiny_diversity_hypothesis().lower()
    assert "pluralistic" in text
    assert "underperform" in text
    assert "blind spot" in text
