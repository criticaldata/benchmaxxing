"""Referee ablations: single vs panel voting, and same- vs cross-lineage referees.

This module sits on top of the ``referee`` scoring arm and asks two ablation questions
about the referee itself, rather than about the committee it judges:

* issue 81 (single vs panel): does a *panel* of independent referee detectors, combined
  by majority vote, catch shortcut dependence more reliably than any single detector?
  :func:`referee_panel_vote` majority-votes a set of injected detectors into one flag set,
  and :func:`single_vs_panel` scores a single detector against the panel on precision and
  recall using a supplied independent truth.
* issue 82 (lineage ablation): a *same-lineage* referee shares the committee's blind spot,
  so it accepts the same spurious cue and flags fewer of the cases where the committee
  actually leaned on the shortcut. :func:`referee_lineage_ablation` scores a same-lineage
  referee against a cross-lineage referee on the same transcript and independent truth, so
  the inherited blind spot shows up as a recall gap.

Every referee here is an injected detector callable ``(transcript, agent_id, cue) -> bool``,
so the whole module is a pure function of its inputs and runs offline with hand-built
transcripts and mock detectors, no API keys. The convenience runners
(:func:`default_referee_committee`, :func:`transcript_from_committee`) wire the ``roster``
and ``blackboard`` cores so a transcript can be produced end to end from a mock backend, but
the scoring functions never require them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Iterable

from benchmaxxing.blackboard import run_committee
from benchmaxxing.referee import precision_recall, score_shortcut
from benchmaxxing.roster import build_committee, default_roster
from benchmaxxing.schema import Case, Committee, Condition, Transcript

__all__ = [
    "Detector",
    "default_referee_committee",
    "referee_lineage_ablation",
    "referee_panel_vote",
    "single_vs_panel",
    "transcript_from_committee",
]

# A referee detector: given the transcript, an agent id, and the planted cue type, return
# True if that agent's decision looks like it depended on the planted cue. This is the same
# contract as ``referee.CueDetector``, restated here so callers can inject their own.
Detector = Callable[[Transcript, str, str], bool]


# --------------------------------------------------------------------------- issue 81: panel


def referee_panel_vote(
    transcript: Transcript,
    cue: str,
    detectors: Iterable[Detector],
) -> dict[str, bool]:
    """Majority-vote several referee detectors into one per-agent flag set.

    Each detector in ``detectors`` is run over the transcript via
    :func:`benchmaxxing.referee.score_shortcut`, giving one ``{agent_id: bool}`` map per
    detector. For each agent the panel flag is ``True`` when a strict majority of detectors
    (more than half) flag that agent; an even split counts as not flagged. Returns one dict
    mapping every agent seen by any detector to its majority flag.

    Raises ``ValueError`` if ``detectors`` is empty (a panel needs at least one voter).
    """
    detectors = list(detectors)
    if not detectors:
        raise ValueError("referee_panel_vote needs at least one detector")

    per_detector = [score_shortcut(transcript, cue, detector=d) for d in detectors]

    agents: list[str] = []
    for flags in per_detector:
        for agent_id in flags:
            if agent_id not in agents:
                agents.append(agent_id)

    n = len(detectors)
    return {
        agent_id: sum(1 for flags in per_detector if flags.get(agent_id)) * 2 > n
        for agent_id in agents
    }


def single_vs_panel(
    transcript: Transcript,
    cue: str,
    truth: Mapping[str, Any] | Iterable[Any],
    detectors: Iterable[Detector],
) -> dict[str, dict[str, float]]:
    """Compare a single referee against the majority-vote panel on precision and recall.

    The single referee is ``detectors[0]`` scored on its own; the panel is the majority vote
    of all ``detectors`` (see :func:`referee_panel_vote`). Both flag sets are scored against
    the supplied independent ``truth`` (a per-agent mapping of who actually depended on the
    cue, or an aligned binary sequence) using
    :func:`benchmaxxing.referee.precision_recall`.

    Returns ``{"single": {precision, recall, f1}, "panel": {precision, recall, f1}}``. Raises
    ``ValueError`` if ``detectors`` is empty.
    """
    detectors = list(detectors)
    if not detectors:
        raise ValueError("single_vs_panel needs at least one detector")

    single_flags = score_shortcut(transcript, cue, detector=detectors[0])
    panel_flags = referee_panel_vote(transcript, cue, detectors)
    return {
        "single": precision_recall(single_flags, truth),
        "panel": precision_recall(panel_flags, truth),
    }


# --------------------------------------------------------------------------- issue 82: lineage


def referee_lineage_ablation(
    transcript: Transcript,
    truth: Mapping[str, Any] | Iterable[Any],
    same_lineage_detector: Detector,
    cross_lineage_detector: Detector,
    *,
    cue: str = "",
) -> dict[str, dict[str, float]]:
    """Score a same-lineage referee against a cross-lineage referee on the same transcript.

    Both referees are injected detectors run over ``transcript`` via
    :func:`benchmaxxing.referee.score_shortcut`, then scored against the supplied independent
    ``truth`` with :func:`benchmaxxing.referee.precision_recall`. A same-lineage referee
    shares the committee's blind spot, so it typically flags fewer of the agents that truly
    leaned on the cue and its recall lags the cross-lineage referee's. The optional ``cue`` is
    forwarded to both detectors (detectors that do not use it may ignore it).

    Returns ``{"same_lineage": {precision, recall, f1}, "cross_lineage": {...}}``.
    """
    same_flags = score_shortcut(transcript, cue, detector=same_lineage_detector)
    cross_flags = score_shortcut(transcript, cue, detector=cross_lineage_detector)
    return {
        "same_lineage": precision_recall(same_flags, truth),
        "cross_lineage": precision_recall(cross_flags, truth),
    }


# --------------------------------------------------------------------------- wiring helpers


def default_referee_committee() -> Committee:
    """The default illustrative committee, built from ``roster.default_roster`` (wiring)."""
    return build_committee(default_roster())


def transcript_from_committee(
    committee: Committee,
    case: Case,
    condition: Condition,
    backend_for: Callable[[Any], Any],
    **run_kwargs: Any,
) -> Transcript:
    """Run ``committee`` on ``case``+``condition`` and return its transcript (wiring).

    A thin pass-through to :func:`benchmaxxing.blackboard.run_committee` so an ablation can be
    driven end to end from an injected ``backend_for`` (a mock backend in tests). Extra
    keyword arguments are forwarded to ``run_committee`` unchanged.
    """
    return run_committee(committee, case, condition, backend_for, **run_kwargs)
