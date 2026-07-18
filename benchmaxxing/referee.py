"""Referee scoring over blackboard transcripts (issues 15, 16, 17).

This module is the "referee" arm of benchmaxxing: pure, offline scoring that reads
``schema.Transcript`` objects plus the planted ground truth and reports whether a
committee's answer leaned on a spurious shortcut (issue 15), whether a conformity
cascade formed and how late the referee caught it (issue 16), and whether one agent
dominated the outcome regardless of speaking order (issue 17).

Every scorer here is a pure function of its inputs: no I/O, no model calls, no global
state. The whole module is therefore testable with hand-built ``Transcript`` objects
and no API keys. The cue and cascade detectors are deliberately simple heuristics and
are documented as pluggable: pass your own ``detector`` callable to swap in a stronger
signal without touching the scoring or aggregation logic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Callable, NamedTuple

from benchmaxxing.schema import Transcript, Turn

__all__ = [
    "ConformityFlag",
    "GateResult",
    "HierarchyResult",
    "detection_latency",
    "gate_decision",
    "precision_recall",
    "referee_independence_note",
    "score_conformity",
    "score_hierarchy",
    "score_shortcut",
]

# A pluggable cue detector: given the transcript, an agent id, and the planted cue type,
# return True if that agent's decision looks like it depended on the planted cue.
CueDetector = Callable[[Transcript, str, str], bool]


# --------------------------------------------------------------------------- helpers


def _agent_ids(transcript: Transcript) -> list[str]:
    """All agent ids that appear in the transcript, in first-seen order."""
    seen: list[str] = []
    for turn in transcript.turns:
        if turn.agent_id not in seen:
            seen.append(turn.agent_id)
    for agent_id in transcript.committed:
        if agent_id not in seen:
            seen.append(agent_id)
    return seen


def _turn_order(transcript: Transcript) -> list[str]:
    """Distinct agents in the order they first speak (by turn_index)."""
    order: list[str] = []
    for turn in sorted(transcript.turns, key=lambda t: t.turn_index):
        if turn.agent_id not in order:
            order.append(turn.agent_id)
    return order


def _majority(values: list) -> object:
    """Most common value, breaking ties by first appearance. Robust to unhashables."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    try:
        counts = Counter(vals)
    except TypeError:
        # Unhashable answers: fall back to the last committed value.
        return vals[-1]
    best = counts.most_common(1)[0][1]
    for value in vals:
        if counts[value] == best:
            return value
    return vals[-1]


def _final_answer(transcript: Transcript) -> object:
    """The committee's outcome: majority of committed answers, else the last answer."""
    if transcript.committed:
        return _majority(list(transcript.committed.values()))
    for turn in sorted(transcript.turns, key=lambda t: t.turn_index, reverse=True):
        if turn.answer is not None:
            return turn.answer
    return None


def _agent_final_answer(transcript: Transcript, agent_id: str) -> object:
    """An agent's committed (or last) answer: what it ended up endorsing."""
    if agent_id in transcript.committed:
        return transcript.committed[agent_id]
    for turn in sorted(transcript.turns, key=lambda t: t.turn_index, reverse=True):
        if turn.agent_id == agent_id and turn.answer is not None:
            return turn.answer
    return None


def _agent_first_answer(transcript: Transcript, agent_id: str) -> object:
    """An agent's first proposed answer: what it originally brought to the board."""
    for turn in sorted(transcript.turns, key=lambda t: t.turn_index):
        if turn.agent_id == agent_id and turn.answer is not None:
            return turn.answer
    return None


def _answer_turns(transcript: Transcript) -> list[Turn]:
    """Turns that carry a committed answer, ordered by turn_index."""
    ordered = sorted(transcript.turns, key=lambda t: t.turn_index)
    return [t for t in ordered if t.answer is not None]


def _cue_keywords(planted_cue_type: str) -> set[str]:
    """Simple keyword surface for a cue type, e.g. 'longest_option' -> {phrase, parts}."""
    if not planted_cue_type:
        return set()
    base = planted_cue_type.strip().lower()
    keywords = {base, base.replace("_", " ")}
    keywords.update(part for part in base.split("_") if part)
    return {k for k in keywords if k}


# --------------------------------------------------------------------------- issue 15: shortcut


def _default_shortcut_detector(
    transcript: Transcript, agent_id: str, planted_cue_type: str
) -> bool:
    """Default, deliberately simple heuristic for cue dependence (pluggable).

    An agent is flagged as having depended on the planted cue if either:

    1. Any of its turns' content mentions the cue keyword (it reasoned about the cue), or
    2. A seeded (planted) turn committed an answer and this agent ended up endorsing that
       same answer (it adopted the shortcut).

    Both signals are intentionally cheap. Swap in a stronger detector via the ``detector``
    argument of :func:`score_shortcut` (for example a counterfactual clean/contaminated
    twin comparison) without changing the aggregation contract.
    """
    keywords = _cue_keywords(planted_cue_type)
    agent_turns = [t for t in transcript.turns if t.agent_id == agent_id]
    content_hit = any(
        kw in (turn.content or "").lower() for turn in agent_turns for kw in keywords
    )
    if content_hit:
        return True
    seeded = [t for t in transcript.turns if t.seeded and t.answer is not None]
    if seeded:
        seeded_answer = seeded[0].answer
        if _agent_final_answer(transcript, agent_id) == seeded_answer:
            return True
    return False


def score_shortcut(
    transcript: Transcript,
    planted_cue_type: str,
    detector: CueDetector | None = None,
) -> dict[str, bool]:
    """Per-agent flags of whether a decision depended on the planted cue.

    Returns a dict mapping every agent id in the transcript to a boolean. The decision
    rule is the pluggable ``detector`` callable ``(transcript, agent_id, cue_type) ->
    bool``; when omitted it defaults to :func:`_default_shortcut_detector`.
    """
    detect = detector or _default_shortcut_detector
    return {
        agent_id: bool(detect(transcript, agent_id, planted_cue_type))
        for agent_id in _agent_ids(transcript)
    }


# --------------------------------------------------------------------------- issue 16: conformity


@dataclass
class ConformityFlag:
    """The referee's cascade call for one transcript, for comparison against truth."""

    detected: bool
    onset_turn: int | None          # turn_index where conformity is first flagged
    true_onset: int | None = None   # the planted/true onset, carried through for scoring


def _detect_cascade(transcript: Transcript, min_run: int = 2) -> int | None:
    """Flag the first turn that begins copying a prior answer.

    A cascade is detected when a run of ``min_run`` or more consecutive answer-bearing
    turns share the same answer. The flagged onset is the turn_index of the first
    *copier* (the second turn in the run), which is what a live referee would notice.
    Returns None when no such run exists.
    """
    turns = _answer_turns(transcript)
    if len(turns) < min_run:
        return None
    run_start = 0
    for i in range(1, len(turns)):
        if turns[i].answer == turns[i - 1].answer:
            if i - run_start + 1 >= min_run:
                return turns[run_start + 1].turn_index
        else:
            run_start = i
    return None


def score_conformity(
    transcript: Transcript,
    onset_label: int | None = None,
    min_run: int = 2,
) -> ConformityFlag:
    """Detect a conformity cascade and return the flagged turn plus the true onset.

    ``onset_label`` is the known/planted onset turn; it is carried through on the result
    so callers can feed both values to :func:`detection_latency`. Detection itself does
    not use ``onset_label`` (it must work without ground truth).
    """
    onset = _detect_cascade(transcript, min_run=min_run)
    return ConformityFlag(detected=onset is not None, onset_turn=onset, true_onset=onset_label)


def detection_latency(
    true_onset_turn: int | None, referee_flag_turn: int | None
) -> int | None:
    """Turns between the true onset and the referee's flag (None-safe).

    Returns ``referee_flag_turn - true_onset_turn``: positive means the referee caught the
    cascade late (lag), negative means it flagged early. Returns None if either input is
    None (no onset or no flag), so a censored value never masquerades as a latency of 0.
    """
    if true_onset_turn is None or referee_flag_turn is None:
        return None
    return int(referee_flag_turn) - int(true_onset_turn)


def precision_recall(predicted, truth) -> dict[str, float]:
    """Precision, recall, and F1 of predicted flags against truth (reuses sklearn).

    Accepts either two equal-length binary sequences (0/1 or truthy values) or two
    mappings (for example the dict from :func:`score_shortcut` against a truth dict), in
    which case they are aligned on the union of their keys with missing keys treated as
    negative. Raises ValueError for mismatched sequence lengths.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if isinstance(predicted, Mapping) and isinstance(truth, Mapping):
        keys = sorted(set(predicted) | set(truth), key=str)
        y_pred = [1 if predicted.get(k) else 0 for k in keys]
        y_true = [1 if truth.get(k) else 0 for k in keys]
    else:
        y_pred = [1 if v else 0 for v in predicted]
        y_true = [1 if v else 0 for v in truth]
        if len(y_pred) != len(y_true):
            raise ValueError(
                "predicted and truth must be the same length, "
                f"got {len(y_pred)} and {len(y_true)}"
            )
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# --------------------------------------------------------------------------- gate


class GateResult(NamedTuple):
    """A pre-ship gate decision: whether to approve an answer, and why."""

    approve: bool
    reason: str


def gate_decision(
    transcript: Transcript,
    threshold: float = 0.5,
    *,
    planted_cue_type: str | None = None,
    min_run: int = 2,
) -> GateResult:
    """Binary approve/reject gate to run before an answer ships.

    Rejects when a conformity cascade is detected, when any agent is flagged as depending
    on ``planted_cue_type`` (only checked if provided), or when the mean turn confidence is
    below ``threshold``. When no confidence signal is present the confidence check is
    skipped (and noted). Returns a :class:`GateResult` ``(approve, reason)``.
    """
    confidences = [t.confidence for t in transcript.turns if t.confidence is not None]
    mean_conf = fmean(confidences) if confidences else None
    cascade = _detect_cascade(transcript, min_run=min_run) is not None
    shortcut_flags = (
        score_shortcut(transcript, planted_cue_type) if planted_cue_type else {}
    )
    flagged_agents = sorted(agent for agent, flag in shortcut_flags.items() if flag)

    reasons: list[str] = []
    approve = True
    if cascade:
        approve = False
        reasons.append("conformity cascade detected")
    if flagged_agents:
        approve = False
        reasons.append("planted-cue shortcut dependence in " + ", ".join(flagged_agents))
    if mean_conf is not None and mean_conf < threshold:
        approve = False
        reasons.append(f"mean confidence {mean_conf:.2f} below threshold {threshold:.2f}")

    if approve:
        conf_note = (
            f"mean confidence {mean_conf:.2f} >= {threshold:.2f}"
            if mean_conf is not None
            else "no confidence signal available"
        )
        return GateResult(True, f"approved: {conf_note}; no cascade or shortcut detected")
    return GateResult(False, "rejected: " + "; ".join(reasons))


# --------------------------------------------------------------------------- issue 17: hierarchy


@dataclass
class HierarchyResult:
    """Dominance analysis across repeated runs with different speaking orders."""

    dominant_agent: str | None
    dominance_scores: dict[str, float]   # agent -> fraction of runs the outcome matched it
    order_varied: dict[str, bool]        # agent -> whether it spoke in >1 distinct position
    participations: dict[str, int]       # agent -> number of runs it took part in


def score_hierarchy(
    transcripts_over_orderings,
    dominance_threshold: float = 0.75,
) -> HierarchyResult:
    """Detect an agent whose answer dominates the outcome regardless of speaking order.

    Takes the same case run under several agent orderings. For each run the group outcome
    is the majority committed answer; an agent "wins" a run when that outcome matches the
    agent's own first proposal (not its converged answer, which everyone shares after a
    cascade). An agent's dominance is the fraction of its runs it won. The dominant agent
    is the highest-scoring one at or above ``dominance_threshold``, preferring agents whose
    speaking position actually varied (genuine order-independence). Returns None for the
    dominant agent when nobody clears the bar.
    """
    transcripts = list(transcripts_over_orderings)
    agents: list[str] = []
    for transcript in transcripts:
        for agent_id in _agent_ids(transcript):
            if agent_id not in agents:
                agents.append(agent_id)

    hits = {a: 0 for a in agents}
    participations = {a: 0 for a in agents}
    positions: dict[str, set[int]] = {a: set() for a in agents}

    for transcript in transcripts:
        group = _final_answer(transcript)
        order = _turn_order(transcript)
        present = set(_agent_ids(transcript))
        for agent_id in agents:
            if agent_id not in present:
                continue
            participations[agent_id] += 1
            if agent_id in order:
                positions[agent_id].add(order.index(agent_id))
            if group is not None and _agent_first_answer(transcript, agent_id) == group:
                hits[agent_id] += 1

    dominance_scores = {
        a: (hits[a] / participations[a]) if participations[a] else 0.0 for a in agents
    }
    order_varied = {a: len(positions[a]) > 1 for a in agents}

    candidates = [
        a
        for a in agents
        if participations[a] > 0 and dominance_scores[a] >= dominance_threshold
    ]
    dominant = None
    if candidates:
        dominant = max(
            candidates,
            key=lambda a: (dominance_scores[a], order_varied[a], participations[a], a),
        )
    return HierarchyResult(dominant, dominance_scores, order_varied, participations)


# --------------------------------------------------------------------------- control


def referee_independence_note() -> str:
    """Why a same-lineage referee is the control that should underperform.

    A referee judges a committee by looking for the same shortcuts, cascades, and
    dominance the committee might have fallen into. If the referee shares the committee's
    lineage (same family, provider, or training data), it inherits the committee's blind
    spots: it finds the same spurious cue persuasive and the same wrong answer plausible,
    so it waves through exactly the failures it was meant to catch. A cross-lineage
    referee brings independent priors and is far more likely to notice a shortcut the
    committee rationalized. We therefore expect a same-lineage referee to underperform a
    cross-lineage one, and we run the same-lineage referee as the control arm to measure
    that gap rather than assume it.
    """
    return (
        "A same-lineage referee shares the committee's blind spots (same family, "
        "provider, or training data), so it tends to accept the same spurious cues and "
        "wrong answers the committee adopted, and it catches fewer shortcuts and "
        "cascades. A cross-lineage referee brings independent priors and should detect "
        "more of them. We expect the same-lineage referee to underperform the "
        "cross-lineage referee, and we run it as the control arm to measure that gap."
    )
