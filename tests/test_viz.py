"""Tests for benchmaxxing.viz (result plotting), issues 86 and 87.

These run headless: matplotlib is skipped cleanly when absent, and the forced Agg backend means
no display is needed. Each test asserts the helper returns a matplotlib Axes on synthetic input
and does not raise, then closes the figure to keep the figure count bounded.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from benchmaxxing.viz import (  # noqa: E402
    plot_cascade_onset,
    plot_confidence_trajectory,
    plot_risk_coverage,
    plot_susceptibility_heatmap,
)
from benchmaxxing.schema import Condition, Transcript, Turn  # noqa: E402


# --------------------------------------------------------------------------- plot_cascade_onset


def test_plot_cascade_onset_returns_axes():
    series = [0.0] * 5 + [1.0] * 5
    ax = plot_cascade_onset(series, onset=5)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_plot_cascade_onset_without_onset():
    ax = plot_cascade_onset([0.2, 0.3, 0.4, 0.5])
    assert isinstance(ax, Axes)
    # No onset supplied means only the series line is drawn (no vertical marker).
    assert len(ax.lines) == 1
    plt.close(ax.figure)


def test_plot_cascade_onset_draws_marker_at_onset():
    ax = plot_cascade_onset([0.0, 0.0, 1.0, 1.0], onset=2)
    xs = [ln.get_xdata()[0] for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert 2 in xs
    plt.close(ax.figure)


def test_plot_cascade_onset_reuses_given_axes():
    fig, ax = plt.subplots()
    out = plot_cascade_onset([0.1, 0.9], onset=1, ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_cascade_onset_empty_series_does_not_raise():
    ax = plot_cascade_onset([])
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


# --------------------------------------------------------------- plot_confidence_trajectory


def test_plot_confidence_trajectory_highlights_wrong_agreement():
    transcript = Transcript(
        run_id="run",
        case_id="case",
        condition=Condition.CONTAMINATED,
        turns=[
            Turn(0, "a", "", answer="A", confidence=0.8),
            Turn(1, "planter", "", answer="B", confidence=None, seeded=True),
            Turn(2, "b", "", answer="B", confidence=0.4),
            Turn(3, "a", "", answer="B", confidence=0.7),
        ],
        meta={"shared": True},
    )
    ax = plot_confidence_trajectory(transcript)
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 1
    assert ax.get_title() == "Confidence trajectory: case"
    plt.close(ax.figure)


def test_plot_confidence_trajectory_handles_unrecorded_confidence():
    transcript = Transcript(
        run_id="run",
        case_id="case",
        condition=Condition.CONTAMINATED,
        turns=[Turn(0, "a", "", answer="A"), Turn(1, "b", "", answer="B")],
    )
    ax = plot_confidence_trajectory(transcript, wrong_answer="B")
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


# ------------------------------------------------------------------ plot_susceptibility_heatmap


def test_plot_susceptibility_heatmap_returns_axes():
    matrix = np.array([[0.1, 0.4, 0.9], [0.0, 0.5, 0.2]])
    models = ["gemini-flash", "qwen-7b"]
    cue_types = ["cable", "option_order", "longest_option"]
    ax = plot_susceptibility_heatmap(matrix, models, cue_types)
    assert isinstance(ax, Axes)
    assert [t.get_text() for t in ax.get_yticklabels()] == models
    assert [t.get_text() for t in ax.get_xticklabels()] == cue_types
    plt.close(ax.figure)


def test_plot_susceptibility_heatmap_handles_nan():
    matrix = np.array([[0.3, np.nan], [np.nan, 0.7]])
    ax = plot_susceptibility_heatmap(matrix, ["m0", "m1"], ["c0", "c1"])
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_plot_susceptibility_heatmap_reuses_given_axes():
    fig, ax = plt.subplots()
    out = plot_susceptibility_heatmap([[0.2, 0.8]], ["m"], ["a", "b"], ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_susceptibility_heatmap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        plot_susceptibility_heatmap(np.zeros((2, 3)), ["m0", "m1"], ["only_one_cue"])


def test_plot_susceptibility_heatmap_non_2d_raises():
    with pytest.raises(ValueError):
        plot_susceptibility_heatmap(np.array([0.1, 0.2]), ["m0"], ["c0"])


# --------------------------------------------------------------------------- plot_risk_coverage


def test_plot_risk_coverage_returns_axes():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.0, 1.0, size=50)
    # Higher confidence more likely correct, so the curve is well behaved.
    correct = rng.uniform(0.0, 1.0, size=50) < confidence
    ax = plot_risk_coverage(confidence, correct)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_plot_risk_coverage_perfect_ranking_low_aurc():
    # Confidence perfectly separates correct (high) from wrong (low): AURC should be small.
    confidence = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    correct = np.array([True, True, True, False, False])
    ax = plot_risk_coverage(confidence, correct)
    label = ax.get_legend().get_texts()[0].get_text()
    aurc = float(label.split("=")[1])
    assert 0.0 <= aurc < 0.2
    plt.close(ax.figure)


def test_plot_risk_coverage_reuses_given_axes():
    fig, ax = plt.subplots()
    out = plot_risk_coverage([0.9, 0.1], [True, False], ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_risk_coverage_length_mismatch_raises():
    with pytest.raises(ValueError):
        plot_risk_coverage([0.9, 0.1], [True])


def test_plot_risk_coverage_empty_raises():
    with pytest.raises(ValueError):
        plot_risk_coverage([], [])
