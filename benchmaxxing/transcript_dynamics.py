"""Turn-level deference and confidence dynamics for committee transcripts.

The older metrics in :mod:`benchmaxxing.onset` compare final committee answers with solo
baselines. This module instead reads a replayable :class:`~benchmaxxing.schema.Transcript` and
asks what happened during deliberation:

* did an agent change its answer to one supplied by a peer since its previous turn?
* did stated confidence rise while agents agreed with the planted/wrong answer?

Both analyses are pure and run offline on saved transcript JSONL files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from benchmaxxing.schema import Transcript, Turn


@dataclass(frozen=True)
class DeferenceEvent:
    """One answer change that adopts an intervening peer answer."""

    turn_index: int
    agent_id: str
    previous_answer: object
    answer: object
    peer_agent_id: str
    peer_turn_index: int


@dataclass(frozen=True)
class DeferenceSummary:
    """Transcript-level deference count and rate.

    ``eligible_turns`` counts unseeded answer-bearing turns for which the same agent has an
    earlier unseeded answer. First answers cannot demonstrate a change and are excluded.
    """

    rate: float
    deferred_turns: int
    eligible_turns: int
    events: tuple[DeferenceEvent, ...]


@dataclass(frozen=True)
class ConfidenceTrajectory:
    """Per-turn confidence and confidence-on-wrong-answer diagnostics."""

    turn_indices: tuple[int, ...]
    confidences: tuple[float, ...]
    wrong_answer: object
    wrong_agreement: tuple[bool, ...]
    n_confident_turns: int
    confidence_coverage: float
    wrong_confidence_slope: float
    rising_while_wrong: bool | None


def _answers_equal(left: object, right: object) -> bool:
    """Equality that also tolerates numpy-valued answers."""
    try:
        equal = left == right
        if isinstance(equal, np.ndarray):
            return bool(np.all(equal))
        return bool(equal)
    except (TypeError, ValueError):
        return False


def deference_summary(transcript: Transcript) -> DeferenceSummary:
    """Measure agents changing to an answer supplied by a peer since their previous turn.

    The chronological board is sufficient to reconstruct visibility for shared runs. An
    explicitly isolated transcript (``transcript.meta["shared"] is False``) has no visible peer
    turns and therefore no deference events. When old transcripts omit this metadata, shared
    visibility is assumed.

    Seeded turns may influence another agent, but are not treated as behavior by the agent to
    which the seed was attributed. Content-only turns and first answers are also excluded from
    the denominator.
    """
    ordered = sorted(transcript.turns, key=lambda turn: turn.turn_index)
    shared = transcript.meta.get("shared", True) is not False
    previous_by_agent: dict[str, tuple[int, object]] = {}
    events: list[DeferenceEvent] = []
    eligible = 0

    for position, turn in enumerate(ordered):
        if turn.seeded or turn.answer is None:
            continue

        previous = previous_by_agent.get(turn.agent_id)
        if previous is not None:
            eligible += 1
            previous_position, previous_answer = previous
            peer_matches: list[Turn] = []
            if shared and not _answers_equal(turn.answer, previous_answer):
                peer_matches = [
                    peer
                    for peer in ordered[previous_position + 1 : position]
                    if peer.agent_id != turn.agent_id
                    and peer.answer is not None
                    and _answers_equal(peer.answer, turn.answer)
                ]
            if peer_matches:
                source = peer_matches[-1]
                events.append(
                    DeferenceEvent(
                        turn_index=turn.turn_index,
                        agent_id=turn.agent_id,
                        previous_answer=previous_answer,
                        answer=turn.answer,
                        peer_agent_id=source.agent_id,
                        peer_turn_index=source.turn_index,
                    )
                )

        previous_by_agent[turn.agent_id] = (position, turn.answer)

    deferred = len(events)
    rate = deferred / eligible if eligible else 0.0
    return DeferenceSummary(rate, deferred, eligible, tuple(events))


def deference_rate(transcript: Transcript) -> float:
    """Return the fraction of eligible follow-up turns that defer to an intervening peer."""
    return deference_summary(transcript).rate


def _seeded_answer(transcript: Transcript) -> object:
    for turn in sorted(transcript.turns, key=lambda item: item.turn_index):
        if turn.seeded and turn.answer is not None:
            return turn.answer
    return None


def confidence_trajectory(
    transcript: Transcript,
    wrong_answer: object = None,
) -> ConfidenceTrajectory:
    """Return per-turn confidence plus a rising-confidence-on-wrong-answer flag.

    When ``wrong_answer`` is omitted, the first answer-bearing seeded turn supplies it. The
    slope uses unseeded agent turns that both endorse that answer and state confidence. Fewer
    than two such turns makes ``rising_while_wrong`` unavailable (``None``), rather than
    incorrectly reporting evidence of no rise.
    """
    ordered = sorted(transcript.turns, key=lambda turn: turn.turn_index)
    if wrong_answer is None:
        wrong_answer = _seeded_answer(transcript)

    indices = tuple(turn.turn_index for turn in ordered)
    confidences = tuple(
        float("nan") if turn.confidence is None else float(turn.confidence) for turn in ordered
    )
    wrong_agreement = tuple(
        wrong_answer is not None and _answers_equal(turn.answer, wrong_answer) for turn in ordered
    )

    agent_answer_turns = [
        turn for turn in ordered if not turn.seeded and turn.answer is not None
    ]
    n_confident = sum(turn.confidence is not None for turn in agent_answer_turns)
    coverage = n_confident / len(agent_answer_turns) if agent_answer_turns else 0.0

    tracked = [
        turn
        for turn in agent_answer_turns
        if wrong_answer is not None
        and _answers_equal(turn.answer, wrong_answer)
        and turn.confidence is not None
    ]
    if len(tracked) < 2:
        slope = float("nan")
        rising = None
    else:
        x = np.asarray([turn.turn_index for turn in tracked], dtype=float)
        y = np.asarray([turn.confidence for turn in tracked], dtype=float)
        if np.unique(x).size < 2:
            slope = float("nan")
            rising = None
        else:
            slope = float(np.polyfit(x, y, 1)[0])
            rising = bool(slope > 1e-9)

    return ConfidenceTrajectory(
        turn_indices=indices,
        confidences=confidences,
        wrong_answer=wrong_answer,
        wrong_agreement=wrong_agreement,
        n_confident_turns=n_confident,
        confidence_coverage=coverage,
        wrong_confidence_slope=slope,
        rising_while_wrong=rising,
    )


def summarize_transcripts(transcripts: Iterable[Transcript]) -> dict[str, dict[str, float | int]]:
    """Aggregate deference and confidence availability by shared/isolated context."""
    groups: dict[str, list[Transcript]] = {"shared": [], "isolated": []}
    for transcript in transcripts:
        key = "isolated" if transcript.meta.get("shared") is False else "shared"
        groups[key].append(transcript)

    result: dict[str, dict[str, float | int]] = {}
    for key, runs in groups.items():
        deference = [deference_summary(run) for run in runs]
        confidence = [confidence_trajectory(run) for run in runs]
        eligible = sum(item.eligible_turns for item in deference)
        deferred = sum(item.deferred_turns for item in deference)
        agent_turns = sum(
            sum(not turn.seeded and turn.answer is not None for turn in run.turns) for run in runs
        )
        confident = sum(item.n_confident_turns for item in confidence)
        evaluable = [item for item in confidence if item.rising_while_wrong is not None]
        result[key] = {
            "n_transcripts": len(runs),
            "eligible_turns": eligible,
            "deferred_turns": deferred,
            "deference_rate": deferred / eligible if eligible else 0.0,
            "confidence_coverage": confident / agent_turns if agent_turns else 0.0,
            "n_confidence_evaluable": len(evaluable),
            "rising_while_wrong_rate": (
                sum(bool(item.rising_while_wrong) for item in evaluable) / len(evaluable)
                if evaluable
                else float("nan")
            ),
        }
    return result


__all__ = [
    "ConfidenceTrajectory",
    "DeferenceEvent",
    "DeferenceSummary",
    "confidence_trajectory",
    "deference_rate",
    "deference_summary",
    "summarize_transcripts",
]
