"""Cascade ablation sweeps (issues 75, 76, 77, 78).

This module runs controlled ablations over the shared-context committee harness to expose the
knobs that drive shortcut cascades. Each sweep is a thin, pure orchestration layer over the
existing core: it builds committees/twins, drives :func:`benchmaxxing.blackboard.run_committee`
(or :func:`benchmaxxing.analysis.solo_evaluate` for the image dose-response), and reports the
cascade signals already implemented in :mod:`benchmaxxing.onset`,
:mod:`benchmaxxing.referee`, and :mod:`benchmaxxing.analysis`.

The four ablations:

* ``committee_size_sweep`` (issue 75): hold the case and the planted shortcut fixed and vary the
  committee size ``N``. For each ``N`` it reports the cascade onset turn and the fraction of
  agents that leaned on the shortcut, so you can see how conformity scales with committee size.
* ``cue_strength_sweep`` (issue 76): a dose-response curve. Build an image twin at each cue
  ``strength`` and report the per-strength flip rate, showing how much cue has to be present
  before the answer moves.
* ``order_permutation_run`` (issue 77): run one committee under several speaking orders and
  report each committed verdict, so you can see whether the verdict tracks who spoke first
  (position) rather than the evidence (correctness).
* ``orchestrator_vs_peer`` (issue 78): run a peer-only cascade against one with an orchestrator
  synthesizer (the ``orchestrator`` toggle), reporting onset and shortcut fraction for each so
  you can see whether a designated leader amplifies or dampens the cascade.

Every runner is driven by an injected backend (``backend_for`` factory for the committee sweeps,
a solo ``backend`` for the image dose-response), so the whole module runs offline against mock
backends with no API keys and no real data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from benchmaxxing.analysis import flip_rate, solo_evaluate
from benchmaxxing.blackboard import run_committee
from benchmaxxing.cues.image import build_image_twin
from benchmaxxing.onset import cascade_onset
from benchmaxxing.referee import score_shortcut
from benchmaxxing.roster import build_committee, default_roster
from benchmaxxing.schema import Case, Committee, Condition, Transcript

__all__ = [
    "SizeSweepPoint",
    "StrengthPoint",
    "OrderPoint",
    "SeedingPoint",
    "committee_size_sweep",
    "cue_strength_sweep",
    "order_permutation_run",
    "orchestrator_vs_peer",
]


# --------------------------------------------------------------------------- result records


@dataclass(frozen=True)
class SizeSweepPoint:
    """One committee size and the cascade signals measured at that size."""

    size: int
    onset: int | None                    # real transcript turn_index of the onset, or None
    shortcut_fraction: float             # fraction of agents flagged as cue-dependent
    n_turns: int
    transcript: Transcript


@dataclass(frozen=True)
class StrengthPoint:
    """One cue strength and the flip rate it produced (a dose-response point)."""

    strength: float
    flip_rate: float
    flipped: bool
    records: list = field(default_factory=list)   # the FlipRecord(s) behind this point


@dataclass(frozen=True)
class OrderPoint:
    """One speaking order and the verdict the committee committed under it."""

    order: tuple[int, ...]
    first_speaker: str
    verdict: object
    committed: dict
    transcript: Transcript


@dataclass(frozen=True)
class SeedingPoint:
    """One seeding regime (peer or orchestrator) and its cascade signals."""

    label: str                           # "peer" or "orchestrator"
    onset: int | None                    # real transcript turn_index of the onset, or None
    shortcut_fraction: float
    n_turns: int
    transcript: Transcript


# --------------------------------------------------------------------------- helpers


def _cascade_series(
    transcript: Transcript, seed_answer: Any, ground_truth: Any
) -> tuple[list[float], list[int]]:
    """Per-turn tug-of-war series plus the real turn index behind each series entry.

    For every answer-bearing turn (ordered by ``turn_index``) the value is ``+1`` when the turn
    matches the planted seed, ``-1`` when it matches the evidence (ground truth), and ``0`` when
    it matches neither (or both cancel). This is exactly the change-point signal
    :func:`benchmaxxing.onset.cascade_onset` consumes: the onset is the turn where the committee
    tips from following the evidence to following the seed.

    Because turns without an answer are skipped, an index into the series is NOT a transcript
    turn index. The second return value maps each series position back to the answer-bearing
    turn's real ``turn_index``, so callers can report a true turn.
    """
    ordered = sorted(transcript.turns, key=lambda t: t.turn_index)
    series: list[float] = []
    turn_indices: list[int] = []
    for turn in ordered:
        if turn.answer is None:
            continue
        agree_seed = 1.0 if turn.answer == seed_answer else 0.0
        agree_evidence = (
            1.0 if (ground_truth is not None and turn.answer == ground_truth) else 0.0
        )
        series.append(agree_seed - agree_evidence)
        turn_indices.append(turn.turn_index)
    return series, turn_indices


def _onset_turn(series: list[float], turn_indices: list[int], min_size: int = 1) -> int | None:
    """Detect the cascade onset on ``series`` and map it back to a real transcript turn index.

    Returns the ``turn_index`` of the answer-bearing turn where the onset occurs, or ``None``
    when the detector reports no change point (censored).
    """
    idx = cascade_onset(series, min_size=min_size)
    if idx is None:
        return None
    return turn_indices[idx] if 0 <= idx < len(turn_indices) else None


def _shortcut_fraction(transcript: Transcript, cue_type: str) -> float:
    """Fraction of agents flagged as depending on the planted cue (via referee.score_shortcut)."""
    flags = score_shortcut(transcript, cue_type)
    if not flags:
        return float("nan")
    return float(sum(1 for v in flags.values() if v)) / len(flags)


def _committee_verdict(transcript: Transcript) -> object:
    """The committee's committed verdict: majority committed answer, else last answer.

    Ties are broken by first appearance. This mirrors the referee's outcome rule but stays
    self-contained so the ablation layer does not depend on referee internals.
    """
    vals = [v for v in transcript.committed.values() if v is not None]
    if not vals:
        for turn in sorted(transcript.turns, key=lambda t: t.turn_index, reverse=True):
            if turn.answer is not None:
                return turn.answer
        return None
    try:
        counts = Counter(vals)
    except TypeError:
        return vals[-1]
    top = counts.most_common(1)[0][1]
    for value in vals:
        if counts[value] == top:
            return value
    return vals[-1]


def _committee_for_size(
    size: int,
    roster: Sequence | None,
    committees: Sequence[Committee] | None,
    index: int,
) -> Committee:
    """Resolve the committee to run at a given size.

    If explicit ``committees`` were supplied, use the one parallel to this size; otherwise take
    the first ``size`` members of ``roster`` (defaulting to ``roster.default_roster()``).
    """
    if committees is not None:
        return committees[index]
    pool = list(default_roster() if roster is None else roster)
    if size < 1:
        raise ValueError(f"committee size must be >= 1, got {size}")
    if size > len(pool):
        raise ValueError(
            f"cannot build a committee of size {size}: only {len(pool)} roster members "
            "available. Pass a larger roster or supply committees=."
        )
    return build_committee(pool[:size])


# --------------------------------------------------------------------------- issue 75


def committee_size_sweep(
    sizes: Iterable[int],
    case: Case,
    backend_for: Callable[[Any], Any],
    *,
    seed_turn: tuple[int, Any, str],
    rounds: int = 3,
    roster: Sequence | None = None,
    committees: Sequence[Committee] | None = None,
    condition: Condition = Condition.CONTAMINATED,
    cue_type: str = "",
    ground_truth: Any = None,
    shared: bool = True,
    min_size: int = 1,
) -> list[SizeSweepPoint]:
    """Sweep committee size and report cascade onset + shortcut fraction at each size (issue 75).

    For each ``N`` in ``sizes`` a committee is built (the first ``N`` of ``roster``, or the
    parallel entry of ``committees`` when given) and run in shared mode with the planted
    ``seed_turn`` shortcut. The onset is detected on the tug-of-war series with
    :func:`benchmaxxing.onset.cascade_onset`; the shortcut fraction is the fraction of agents
    flagged by :func:`benchmaxxing.referee.score_shortcut`.

    Parameters
    ----------
    sizes:
        Committee sizes to sweep.
    case, condition:
        The clinical case and the twin condition label for every run.
    backend_for:
        Factory ``backend_for(model_spec) -> backend`` injected into the harness.
    seed_turn:
        ``(index, answer, agent_id)`` planted shortcut passed to ``run_committee``. Its answer is
        the seed used to build the onset series.
    rounds:
        Rounds of the speaking order per run (default 3, so a cascade has room to form).
    roster, committees:
        Source of committees (see :func:`_committee_for_size`).
    cue_type:
        Cue label forwarded to ``score_shortcut``.
    ground_truth:
        The evidence answer for the onset series. When ``None`` only seed-agreement drives it.
    shared:
        Whether the board is shared (default True; the cascade only propagates when shared).
    min_size:
        Minimum segment length for onset detection.

    Returns
    -------
    list[SizeSweepPoint]
        One point per size, in the order given.
    """
    sizes = list(sizes)
    if committees is not None:
        committees = list(committees)
        if len(committees) != len(sizes):
            raise ValueError("committees must be parallel to sizes (same length)")
    seed_answer = seed_turn[1]

    points: list[SizeSweepPoint] = []
    for i, size in enumerate(sizes):
        committee = _committee_for_size(size, roster, committees, i)
        transcript = run_committee(
            committee,
            case,
            condition,
            backend_for,
            shared=shared,
            seed_turn=seed_turn,
            rounds=rounds,
        )
        series, turn_indices = _cascade_series(transcript, seed_answer, ground_truth)
        onset = _onset_turn(series, turn_indices, min_size=min_size)
        points.append(
            SizeSweepPoint(
                size=len(committee.members),
                onset=onset,
                shortcut_fraction=_shortcut_fraction(transcript, cue_type),
                n_turns=len(transcript.turns),
                transcript=transcript,
            )
        )
    return points


# --------------------------------------------------------------------------- issue 76


def cue_strength_sweep(
    img,
    cue_type: str,
    strengths: Iterable[float],
    backend,
    answer_fn,
    *,
    strength_param: str = "opacity",
    ground_truth: Any = None,
    case_id: str = "image_twin",
    model=None,
    **params,
) -> list[StrengthPoint]:
    """Dose-response over cue strength: per-strength flip rate on an image twin (issue 76).

    For each ``strength`` an image twin is built with
    :func:`benchmaxxing.cues.image.build_image_twin`, injecting the cue at that strength (mapped
    onto ``strength_param``, ``"opacity"`` by default). The clean/contaminated pair is scored with
    :func:`benchmaxxing.analysis.solo_evaluate` and the flip rate is read off with
    :func:`benchmaxxing.analysis.flip_rate`, giving one dose-response point per strength.

    Parameters
    ----------
    img:
        A uint8 image array (the clean chest X-ray).
    cue_type:
        One of the image cues (``"cable"``, ``"corner_tag"``, ``"watermark"``, ``"laterality"``).
    strengths:
        The cue strengths to sweep (each becomes ``strength_param`` on the injector).
    backend, answer_fn:
        A solo backend ``backend(payload) -> raw`` and normalizer ``answer_fn(raw) -> answer``,
        injected so the sweep is deterministic and offline.
    strength_param:
        The injector parameter that carries the strength (default ``"opacity"``).
    ground_truth:
        Optional ground truth stamped on each twin.
    case_id, model:
        Labels forwarded to the twin/record.
    params:
        Extra fixed injector parameters (for example ``thickness=`` for the cable cue).

    Returns
    -------
    list[StrengthPoint]
        One dose-response point per strength, in the order given.
    """
    points: list[StrengthPoint] = []
    for strength in strengths:
        injector_params = dict(params)
        injector_params[strength_param] = strength
        twin = build_image_twin(
            img,
            cue_type,
            ground_truth=ground_truth,
            case_id=case_id,
            **injector_params,
        )
        records = solo_evaluate([twin], backend, answer_fn, model=model)
        rate = flip_rate(records)["overall"]
        points.append(
            StrengthPoint(
                strength=float(strength),
                flip_rate=float(rate),
                flipped=bool(records[0].flipped) if records else False,
                records=records,
            )
        )
    return points


# --------------------------------------------------------------------------- issue 77


def order_permutation_run(
    committee: Committee,
    case: Case,
    backend_for: Callable[[Any], Any],
    orders: Iterable[Sequence[int]],
    *,
    seed_turn: tuple[int, Any, str] | None = None,
    rounds: int = 1,
    condition: Condition = Condition.CONTAMINATED,
    shared: bool = True,
) -> list[OrderPoint]:
    """Run one committee under several speaking orders and return each committed verdict (issue 77).

    The same committee and case are run once per speaking order. Each result records the order,
    the agent that spoke first, and the committed verdict (majority committed answer). Comparing
    verdicts across orders shows whether the outcome tracks who spoke first (position) rather than
    the evidence (correctness).

    Parameters
    ----------
    committee, case, condition:
        The fixed committee, case, and twin condition label.
    backend_for:
        Factory ``backend_for(model_spec) -> backend`` injected into the harness.
    orders:
        Iterable of speaking orders (each a permutation of member indices).
    seed_turn:
        Optional planted shortcut ``(index, answer, agent_id)`` applied to every order.
    rounds:
        Rounds of the speaking order per run.
    shared:
        Whether the board is shared (default True).

    Returns
    -------
    list[OrderPoint]
        One point per order, in the order given.
    """
    members = tuple(committee.members)
    points: list[OrderPoint] = []
    for order in orders:
        order = list(order)
        transcript = run_committee(
            committee,
            case,
            condition,
            backend_for,
            shared=shared,
            order=order,
            seed_turn=seed_turn,
            rounds=rounds,
        )
        first_speaker = members[order[0]].name if order else ""
        points.append(
            OrderPoint(
                order=tuple(order),
                first_speaker=first_speaker,
                verdict=_committee_verdict(transcript),
                committed=dict(transcript.committed),
                transcript=transcript,
            )
        )
    return points


# --------------------------------------------------------------------------- issue 78


def orchestrator_vs_peer(
    committee: Committee,
    case: Case,
    backend_for: Callable[[Any], Any],
    *,
    seed_index: int,
    seed_answer: Any,
    seed_agent: str | None = None,
    rounds: int = 1,
    condition: Condition = Condition.CONTAMINATED,
    cue_type: str = "",
    ground_truth: Any = None,
    shared: bool = True,
) -> dict[str, SeedingPoint]:
    """Compare a peer-only cascade against an orchestrator-led one (issue 78).

    The same committee, case, and planted seed are run twice, differing only in the
    ``orchestrator`` toggle:

    * ``"peer"``: a flat committee (no orchestrator). The seed is planted at ``seed_index``.
    * ``"orchestrator"``: identical seed, but a designated leader (the last speaker) takes a final
      synthesizer turn over the full board.

    Each run reports the cascade onset (via :func:`benchmaxxing.onset.cascade_onset`) and the
    shortcut fraction (via :func:`benchmaxxing.referee.score_shortcut`), so you can see whether the
    orchestrator amplifies or dampens the seeded cascade.

    Parameters
    ----------
    committee, case, condition:
        The fixed committee, case, and twin condition label.
    backend_for:
        Factory ``backend_for(model_spec) -> backend`` injected into the harness.
    seed_index, seed_answer, seed_agent:
        The planted shortcut ``(seed_index, seed_answer, seed_agent)`` used in BOTH runs.
    rounds:
        Rounds of the speaking order per run.
    cue_type:
        Cue label forwarded to ``score_shortcut``.
    ground_truth:
        The evidence answer for the onset series. When ``None`` only seed-agreement drives it.
    shared:
        Whether the board is shared (default True).

    Returns
    -------
    dict[str, SeedingPoint]
        ``{"peer": SeedingPoint, "orchestrator": SeedingPoint}``.
    """
    seed_turn = (seed_index, seed_answer, seed_agent)

    def _run(orchestrator: bool, label: str) -> SeedingPoint:
        transcript = run_committee(
            committee,
            case,
            condition,
            backend_for,
            shared=shared,
            orchestrator=orchestrator,
            seed_turn=seed_turn,
            rounds=rounds,
        )
        series, turn_indices = _cascade_series(transcript, seed_answer, ground_truth)
        return SeedingPoint(
            label=label,
            onset=_onset_turn(series, turn_indices),
            shortcut_fraction=_shortcut_fraction(transcript, cue_type),
            n_turns=len(transcript.turns),
            transcript=transcript,
        )

    return {
        "peer": _run(False, "peer"),
        "orchestrator": _run(True, "orchestrator"),
    }
