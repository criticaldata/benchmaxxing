"""End-to-end integration smoke: the whole pipeline composes and shows the phenomenon.

Unlike the per-module unit tests, this exercises the seams between modules offline (no API
keys, no real data). It is the fastest way to catch an interface drift between two modules.
The numeric assertions pin the exact reported values so a silent drift is caught here.
"""

from __future__ import annotations

import pytest

from benchmaxxing.demo import format_report, run_smoke


def test_smoke_runs_end_to_end():
    report = run_smoke()
    for key in (
        "solo_flip_rate", "committee_correct_rate_shared", "committee_correct_rate_isolated",
        "cascade_onset_turn", "referee_precision_recall", "gate_approve",
        "robust_agent_gamed", "robust_agent_flagged", "blind_metric_uptake_delta",
    ):
        assert key in report


def test_exact_committee_rates():
    # Pinned so the demo's reported numbers cannot silently drift. Two same-lineage agents
    # cascade onto the shortcut and one cross-lineage agent resists, over 3 rounds of 3 agents:
    # shared = 3/9 correct, isolated = 8/9 (only the single seeded turn is wrong).
    report = run_smoke()
    assert report["committee_correct_rate_shared"] == pytest.approx(3 / 9)
    assert report["committee_correct_rate_isolated"] == pytest.approx(8 / 9)
    assert report["committee_correct_rate_shared"] < report["committee_correct_rate_isolated"]


def test_referee_scored_against_independent_truth():
    report = run_smoke()
    # The gate rejects the cascaded run.
    assert report["gate_approve"] is False
    # The robust cross-lineage agent is a genuine true negative: it was not gamed, and the
    # referee must not flag it. This is what makes the precision/recall meaningful rather than
    # true by construction, the referee has to get a negative case right.
    assert report["robust_agent_gamed"] is False
    assert report["robust_agent_flagged"] is False
    pr = report["referee_precision_recall"]
    assert pr["precision"] == pytest.approx(1.0)
    assert pr["recall"] == pytest.approx(1.0)


def test_cascade_onset_is_detected():
    report = run_smoke()
    assert isinstance(report["cascade_onset_turn"], int)


def test_blind_metric_detects_drift():
    report = run_smoke()
    # decisions produced by the per-case run track the decoy more than the truth
    assert report["blind_metric_uptake_delta"] > 0.0


def test_report_formats():
    text = format_report(run_smoke())
    assert "composes end to end" in text
    assert "true negative" in text
