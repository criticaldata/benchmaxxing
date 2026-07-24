"""Plotting helpers for benchmaxxing results (issues 86 and 87).

Small, dependency-light figures for the three headline views:

    plot_cascade_onset: a turn-level agreement/confidence series with the detected onset marked.
    plot_confidence_trajectory: stated confidence over a transcript, highlighting wrong agreement.
    plot_susceptibility_heatmap: a models-by-cue flip-rate heatmap.
    plot_risk_coverage: a selective risk-coverage (AURC) curve.

matplotlib is an optional dependency: it is imported lazily inside a helper and a clear ImportError
is raised when it is missing. A non-interactive Agg backend is forced so the module is safe to use
headless, and no figure is ever shown (the caller owns display and saving).
"""

from __future__ import annotations

import numpy as np


def _pyplot():
    """Return ``matplotlib.pyplot`` bound to the non-interactive Agg backend.

    Raises a clear ImportError when matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "benchmaxxing.viz requires matplotlib. Install it with 'pip install matplotlib'."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_cascade_onset(series, onset=None, ax=None):
    """Plot a turn-level series and mark the detected cascade onset (issue 86).

    Args:
        series: 1-D array-like of per-turn values (agreement or confidence).
        onset: onset turn index to mark with a vertical line, or ``None`` to draw no marker
            (for example a censored case where no onset was detected).
        ax: an existing matplotlib Axes to draw on, or ``None`` to create a new figure.

    Returns:
        The matplotlib Axes the series was drawn on.
    """
    plt = _pyplot()
    y = np.asarray(series, dtype=float).ravel()
    x = np.arange(y.size)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y, marker="o", color="#1f77b4", label="series")
    if onset is not None:
        marker = int(onset)
        ax.axvline(marker, color="#d62728", linestyle="--", label=f"onset (turn {marker})")
    ax.set_xlabel("turn index")
    ax.set_ylabel("agreement / confidence")
    ax.set_title("Cascade onset")
    ax.legend(loc="best")
    return ax


def plot_confidence_trajectory(transcript, wrong_answer=None, ax=None):
    """Plot stated confidence by turn and highlight agreement with a planted/wrong answer.

    ``wrong_answer`` defaults to the first answer-bearing seeded turn. Missing confidence values
    remain gaps in the trajectory instead of being imputed. This is important for old saved runs,
    where confidence was not recorded at all.
    """
    from benchmaxxing.transcript_dynamics import confidence_trajectory

    plt = _pyplot()
    trajectory = confidence_trajectory(transcript, wrong_answer=wrong_answer)
    x = np.asarray(trajectory.turn_indices, dtype=int)
    y = np.asarray(trajectory.confidences, dtype=float)
    wrong = np.asarray(trajectory.wrong_agreement, dtype=bool)
    finite = np.isfinite(y)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y, marker="o", color="#1f77b4", label="stated confidence")
    highlighted = finite & wrong
    if np.any(highlighted):
        ax.scatter(
            x[highlighted],
            y[highlighted],
            color="#d62728",
            zorder=3,
            label="agrees with planted/wrong answer",
        )
    for turn in transcript.turns:
        if turn.seeded:
            ax.axvline(turn.turn_index, color="#7f7f7f", linestyle="--", alpha=0.6)

    ax.set_xlabel("turn index")
    ax.set_ylabel("stated confidence")
    ax.set_title(f"Confidence trajectory: {transcript.case_id}")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    return ax


def plot_susceptibility_heatmap(matrix, models, cue_types, ax=None):
    """Draw a models-by-cue flip-rate heatmap (issue 87).

    Args:
        matrix: 2-D array-like of flip rates with shape ``(len(models), len(cue_types))``. NaN
            entries (no record for that model/cue) are masked and drawn as blank cells.
        models: row labels, one per model.
        cue_types: column labels, one per cue type.
        ax: an existing matplotlib Axes to draw on, or ``None`` to create a new figure.

    Returns:
        The matplotlib Axes the heatmap was drawn on.
    """
    plt = _pyplot()
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D (models x cue_types)")
    models = list(models)
    cue_types = list(cue_types)
    if m.shape != (len(models), len(cue_types)):
        raise ValueError(
            f"matrix shape {m.shape} does not match "
            f"(len(models)={len(models)}, len(cue_types)={len(cue_types)})"
        )
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(np.ma.masked_invalid(m), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cue_types)))
    ax.set_xticklabels(cue_types, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("cue type")
    ax.set_ylabel("model")
    ax.set_title("Susceptibility (flip rate)")
    ax.figure.colorbar(im, ax=ax, label="flip rate")
    return ax


def plot_risk_coverage(confidence, correct, ax=None):
    """Plot a selective risk-coverage curve and report its AURC (issue 87).

    Samples are ranked by confidence (most confident first). At each coverage level ``k / n`` the
    selective risk is the error rate over the ``k`` most confident predictions. AURC is the mean
    selective risk over the uniform coverage grid: lower is a better confidence ranking.

    Args:
        confidence: 1-D array-like of per-prediction confidence scores.
        correct: 1-D array-like of booleans, True where the prediction was correct. Same length
            as ``confidence``.
        ax: an existing matplotlib Axes to draw on, or ``None`` to create a new figure.

    Returns:
        The matplotlib Axes the curve was drawn on.
    """
    plt = _pyplot()
    conf = np.asarray(confidence, dtype=float).ravel()
    corr = np.asarray(correct).astype(bool).ravel()
    if conf.shape != corr.shape:
        raise ValueError("confidence and correct must have the same length")
    if conf.size == 0:
        raise ValueError("confidence and correct must be non-empty")

    order = np.argsort(-conf, kind="stable")
    errors = (~corr[order]).astype(float)
    counts = np.arange(1, errors.size + 1)
    risk = np.cumsum(errors) / counts       # selective risk at each coverage level
    coverage = counts / errors.size
    aurc = float(np.mean(risk))             # area under the risk-coverage curve

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(coverage, risk, color="#1f77b4", label=f"AURC = {aurc:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_title("Risk-coverage curve")
    ax.legend(loc="best")
    return ax


__all__ = [
    "plot_cascade_onset",
    "plot_confidence_trajectory",
    "plot_susceptibility_heatmap",
    "plot_risk_coverage",
]
