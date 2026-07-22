"""Real-time referee interventions (issue 16) and natural-cue validation (issue 12).

Two offline, dependency-light drivers built on the shared harness:

* :func:`run_referee_intervention` runs a LIVE committee through
  :func:`benchmaxxing.blackboard.run_committee` with a real-time referee wired in as the
  ``pre_hook`` (the pre-emptive tap the harness already exposes: it fires after a turn
  commits and before the next agent reads the board, and any :class:`~benchmaxxing.schema.Turn`
  it returns is injected so the next agent sees it). It also runs a matched referee-ABSENT
  baseline on the same case and seed, so the two transcripts differ only by the referee.
  :func:`intervention_effect` then scores how much the referee cut the shortcut cascade and
  whether the flag was a false alarm.

* :func:`natural_cue_validation` checks whether the models that flip on INJECTED (synthetic)
  cues are the same models that flip on NATURALLY present devices (CheXpert Support Devices).
  If the two rankings agree (Spearman rank correlation above a threshold), the injection
  methodology is validated against reality.

Everything here is a pure function of its inputs and runs offline with a mock backend or
synthetic records; no network and no API keys.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from statistics import fmean

from benchmaxxing.schema import Condition

__all__ = [
    "run_referee_intervention",
    "intervention_effect",
    "natural_cue_validation",
]


# --------------------------------------------------------------------------- issue 16: referee


def run_referee_intervention(
    committee,
    case,
    backend_for,
    referee_hook,
    *,
    seed_turn=None,
    rounds=3,
    condition: Condition = Condition.CONTAMINATED,
    seed: int = 0,
):
    """Run a live committee with a real-time referee, plus a matched referee-absent baseline.

    The referee is ``referee_hook``, passed straight through as the ``pre_hook`` of
    :func:`benchmaxxing.blackboard.run_committee`. That pre-emptive hook is a capability the
    blackboard harness already provides: it is invoked after each member turn commits and
    before the next agent reads the board, and a :class:`~benchmaxxing.schema.Turn` it returns
    is injected onto the board so the next agent conditions on it. The baseline reruns the same
    committee on the same ``case``, ``condition``, ``seed`` and ``seed_turn`` with no hook, so
    the only difference between the two transcripts is the referee.

    Parameters
    ----------
    committee, case:
        The roster and the clinical case to run.
    backend_for:
        Factory ``backend_for(model_spec) -> Backend`` (injected, so a mock backend runs it
        offline).
    referee_hook:
        The synchronous pre-emptive referee ``pre_hook(state)`` (see the harness contract).
    seed_turn:
        Optional shortcut injection ``(index, answer, agent_id)`` planted in BOTH runs.
    rounds:
        Number of committee rounds.
    condition:
        Which twin condition to run (defaults to CONTAMINATED, the injected-cue side).
    seed:
        Reproducibility seed threaded into both runs so they stay matched.

    Returns
    -------
    dict
        ``{"intervention_transcript": Transcript, "baseline_transcript": Transcript}``.
    """
    from benchmaxxing.blackboard import run_committee

    intervention = run_committee(
        committee,
        case,
        condition,
        backend_for,
        pre_hook=referee_hook,
        seed_turn=seed_turn,
        seed=seed,
        rounds=rounds,
    )
    baseline = run_committee(
        committee,
        case,
        condition,
        backend_for,
        pre_hook=None,
        seed_turn=seed_turn,
        seed=seed,
        rounds=rounds,
    )
    return {"intervention_transcript": intervention, "baseline_transcript": baseline}


def _members(transcript) -> set:
    """The committee member ids recorded on the transcript (empty set if unknown)."""
    return set(transcript.meta.get("members") or [])


def _member_answer_turns(transcript) -> list:
    """Answer-bearing turns from real committee members (no plant, no injected referee turn)."""
    members = _members(transcript)
    return [
        t
        for t in transcript.turns
        if t.answer is not None and not t.seeded and (not members or t.agent_id in members)
    ]


def _seeded_answer(transcript):
    """The planted shortcut answer, taken from the first seeded turn (or None)."""
    for t in transcript.turns:
        if t.seeded and t.answer is not None:
            return t.answer
    return None


def _majority_member_answer(transcript):
    """Most common member answer, used as the shortcut when nothing was seeded."""
    answers = [t.answer for t in _member_answer_turns(transcript)]
    if not answers:
        return None
    try:
        return Counter(answers).most_common(1)[0][0]
    except TypeError:
        return answers[-1]


def _shortcut_fraction(transcript, shortcut_answer) -> float:
    """Fraction of member answer turns that landed on ``shortcut_answer``.

    Returns 0.0 when there are no member answer turns or the shortcut is unknown, so the
    downstream difference stays finite.
    """
    considered = _member_answer_turns(transcript)
    if not considered or shortcut_answer is None:
        return 0.0
    on_shortcut = sum(1 for t in considered if t.answer == shortcut_answer)
    return on_shortcut / len(considered)


def _referee_fired(transcript) -> bool:
    """Whether any non-member, non-seeded turn was injected (a referee intervention)."""
    members = _members(transcript)
    if not members:
        return False
    return any(t.agent_id not in members and not t.seeded for t in transcript.turns)


def intervention_effect(baseline, intervention, *, shortcut_answer=None) -> dict:
    """Score how much the referee cut the shortcut cascade, and flag false alarms.

    The cascade measure is the fraction of committee member turns that committed the shortcut
    answer. ``cascade_reduction`` is ``baseline_fraction - intervention_fraction``: positive
    means the referee pulled the committee off the shortcut. The shortcut answer defaults to the
    planted seed answer (or, absent a seed, the baseline's majority member answer); pass
    ``shortcut_answer`` to override.

    A false alarm is recorded when the referee intervened yet the referee-absent baseline shows
    no shortcut adoption at all, so there was no cascade to catch.
    """
    shortcut = shortcut_answer
    if shortcut is None:
        shortcut = _seeded_answer(baseline)
    if shortcut is None:
        shortcut = _seeded_answer(intervention)
    if shortcut is None:
        shortcut = _majority_member_answer(baseline)

    base_frac = _shortcut_fraction(baseline, shortcut)
    interv_frac = _shortcut_fraction(intervention, shortcut)
    fired = _referee_fired(intervention)
    false_alarm = bool(fired and base_frac == 0.0)

    if false_alarm:
        note = (
            f"referee intervened but the referee-absent baseline showed no shortcut adoption "
            f"(fraction {base_frac:.2f}), so the flag was a false alarm"
        )
    elif fired:
        note = (
            f"referee intervened; baseline shortcut fraction {base_frac:.2f} was reduced to "
            f"{interv_frac:.2f} under the referee"
        )
    else:
        note = "referee did not intervene"

    return {
        "shortcut_answer": shortcut,
        "baseline_shortcut_fraction": base_frac,
        "intervention_shortcut_fraction": interv_frac,
        "cascade_reduction": base_frac - interv_frac,
        "referee_fired": fired,
        "false_alarm": false_alarm,
        "note": note,
    }


# --------------------------------------------------------------------------- issue 12: natural


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _reduce_records(records) -> float:
    """Mean flip rate over an iterable of records exposing a ``flipped`` attribute."""
    vals = [1.0 if getattr(r, "flipped") else 0.0 for r in records]
    return fmean(vals) if vals else float("nan")


def _to_scores(data) -> dict:
    """Coerce input into ``{model: reliance_score}``.

    Accepts a mapping ``{model: number}`` (used directly), a mapping
    ``{model: iterable-of-records}`` (reduced to a flip rate), or a flat iterable of records
    with ``.model`` and ``.flipped`` attributes (grouped by model, then reduced).
    """
    if isinstance(data, Mapping):
        scores: dict = {}
        for key, value in data.items():
            if _is_number(value):
                scores[str(key)] = float(value)
            else:
                scores[str(key)] = _reduce_records(value)
        return scores
    grouped: dict = {}
    for record in data:
        grouped.setdefault(str(record.model), []).append(record)
    return {model: _reduce_records(recs) for model, recs in grouped.items()}


def _ranking(scores: dict, models) -> list:
    """Models ordered by descending reliance score, ties broken by name."""
    return sorted(models, key=lambda m: (-scores[m], m))


def natural_cue_validation(injected_records, natural_records, *, threshold: float = 0.7) -> dict:
    """Validate injected cues against naturally present CheXpert Support Devices.

    ``injected_records`` and ``natural_records`` each give per-model flip / shortcut-reliance:
    under INJECTED (synthetic) cues and under NATURALLY occurring devices, respectively (see
    :func:`_to_scores` for accepted shapes). Models are ranked by each, and the Spearman rank
    correlation between the two score vectors is computed over the models common to both.

    If the two rankings agree (correlation at or above ``threshold``), the injection methodology
    is validated against reality: the models that flip on synthetic cues are the same ones that
    flip on real Support Devices, so the injection reproduces a real-world shortcut rather than
    an artefact.

    Returns a dict with the common models, the two rankings, the Spearman ``correlation`` and
    ``p_value``, and the boolean ``injection_validated``.
    """
    from scipy.stats import spearmanr

    injected = _to_scores(injected_records)
    natural = _to_scores(natural_records)
    models = sorted(set(injected) & set(natural))
    if len(models) < 2:
        raise ValueError(
            "need at least 2 models present in both injected and natural records, "
            f"got {len(models)}"
        )

    injected_vals = [injected[m] for m in models]
    natural_vals = [natural[m] for m in models]
    result = spearmanr(injected_vals, natural_vals)
    corr = float(result.statistic)
    p_value = float(result.pvalue)
    validated = bool(corr == corr and corr >= threshold)  # corr == corr rejects nan

    return {
        "models": models,
        "injected_ranking": _ranking(injected, models),
        "natural_ranking": _ranking(natural, models),
        "correlation": corr,
        "p_value": p_value,
        "threshold": threshold,
        "injection_validated": validated,
    }
