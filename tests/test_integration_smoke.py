"""End-to-end integration smoke: the whole pipeline composes and shows the phenomenon.

Unlike the per-module unit tests, this exercises the seams between modules offline (no API
keys, no real data). It is the fastest way to catch an interface drift between two modules.
"""

from __future__ import annotations

from benchmaxxing.demo import format_report, run_smoke


def test_smoke_runs_end_to_end():
    report = run_smoke()
    # every stage produced a result
    for key in (
        "solo_flip_rate", "committee_correct_rate_shared", "committee_correct_rate_isolated",
        "cascade_onset_turn", "referee_precision_recall", "gate_approve",
        "blind_metric_uptake_delta",
    ):
        assert key in report


def test_shared_context_produces_the_cascade():
    report = run_smoke()
    # the whole point: sharing context spreads the planted shortcut, so the shared committee
    # is less correct than the isolated one.
    assert report["committee_correct_rate_shared"] < report["committee_correct_rate_isolated"]


def test_referee_catches_the_cascade():
    report = run_smoke()
    # the gate must reject a run that cascaded onto the planted shortcut
    assert report["gate_approve"] is False
    pr = report["referee_precision_recall"]
    assert pr["recall"] > 0.0  # it flagged at least some of the deferring agents


def test_blind_metric_detects_drift():
    report = run_smoke()
    # decisions were built to track the decoy more than the truth, so uptake is positive
    assert report["blind_metric_uptake_delta"] > 0.0


def test_report_formats():
    assert "composes end to end" in format_report(run_smoke())
