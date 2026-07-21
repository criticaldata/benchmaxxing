"""Stage runners that wire the existing pieces into end-to-end experiments.

Each runner is a thin orchestrator: it composes the cue builders, the solo analysis, the
blackboard committee harness, and the onset math into one call. Nothing here talks to a model
provider directly. Backends are always injected (a bare callable or a ``gateway`` backend for
the solo lane, a ``backend_for(model_spec)`` factory for the committee lane), so every runner
executes fully offline against a ``MockBackend`` in tests and unchanged against a real Gemini
roster in production.

Runners:
    run_pilot: stage-0 pilot: do the cues bite a single model at all? (issue 9)
    run_solo_baselines: per-model solo susceptibility matrix and failure vectors. (issue 10)
    run_holes_test: same-lineage vs cross-lineage failure-overlap test. (issue 11)
    run_cascade: shared vs isolated committee with a seeded shortcut and its onset. (issue 13)
"""

from __future__ import annotations

from benchmaxxing.analysis import (
    failure_vector,
    flip_rate,
    lineage_overlap_test,
    solo_evaluate,
    susceptibility_matrix,
)
from benchmaxxing.blackboard import run_committee
from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.onset import cascade_onset
from benchmaxxing.schema import Condition

__all__ = [
    "run_pilot",
    "run_solo_baselines",
    "run_holes_test",
    "run_cascade",
]


def _model_name(model) -> str:
    """Best-effort display name for a roster entry (``ModelSpec`` or a bare id)."""
    name = getattr(model, "name", None)
    return name if isinstance(name, str) and name else str(model)


def _build_twins(cases, cue_types, limit: int | None = None) -> list:
    """Build every ``build_text_twin(case, cue_type)`` over the first ``limit`` cases.

    A (case, cue) pair whose case lacks the fields that cue needs (or a cue that needs extra
    params) is skipped rather than aborting the whole run, so a heterogeneous case set still
    yields a pilot.
    """
    selected = list(cases)
    if limit is not None:
        selected = selected[:limit]
    twins = []
    for case in selected:
        for cue_type in cue_types:
            try:
                twins.append(build_text_twin(case, cue_type))
            except (ValueError, TypeError):
                continue
    return twins


def run_pilot(cases, backend, answer_fn, cue_types, limit: int = 50) -> dict:
    """Stage-0 pilot: build twins for the first ``limit`` cases and see if the cues bite.

    One model (``backend`` + ``answer_fn``) is run solo over every clean/contaminated twin.
    ``cues_bite`` is ``True`` when at least one twin flipped the model's answer, the go/no-go
    signal for spending compute on the full arms.

    Returns ``{"flip_rate", "n", "cue_types", "cues_bite"}`` where ``flip_rate`` is the full
    ``analysis.flip_rate`` breakdown and ``n`` is the number of twins evaluated.
    """
    twins = _build_twins(cases, cue_types, limit=limit)
    records = solo_evaluate(twins, backend, answer_fn)
    return {
        "flip_rate": flip_rate(records),
        "n": len(records),
        "cue_types": list(cue_types),
        "cues_bite": any(r.flipped for r in records),
    }


def run_solo_baselines(cases, models, backend_for, answer_fn, cue_types) -> dict:
    """Run each model solo over the shared twin set (issue 10).

    ``backend_for(model)`` yields the backend for one roster entry, so the same twins are scored
    by every model. Returns the models x cue-types susceptibility matrix plus the per-case binary
    failure vector for each model (aligned across models because they share the twin set).

    Returns ``{"susceptibility_matrix", "failure_vectors", "records_by_model", "n_twins"}``.
    """
    twins = _build_twins(cases, cue_types)
    records_by_model: dict[str, list] = {}
    for model in models:
        name = _model_name(model)
        records_by_model[name] = solo_evaluate(twins, backend_for(model), answer_fn, model=name)
    failure_vectors = {name: failure_vector(recs) for name, recs in records_by_model.items()}
    return {
        "susceptibility_matrix": susceptibility_matrix(records_by_model),
        "failure_vectors": failure_vectors,
        "records_by_model": records_by_model,
        "n_twins": len(twins),
    }


def run_holes_test(
    failure_vectors_by_model,
    lineage_by_model,
    *,
    metric: str = "phi",
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """Swiss-cheese overlap test: do same-lineage models fail on the same cases? (issue 11).

    A pass-through to ``analysis.lineage_overlap_test`` over the per-model failure vectors
    produced by :func:`run_solo_baselines`, returning its full result dict.
    """
    return lineage_overlap_test(
        failure_vectors_by_model,
        lineage_by_model,
        metric=metric,
        n_permutations=n_permutations,
        seed=seed,
    )


def _shortcut_answer(case):
    """A plausible planted-shortcut answer: a distractor option, or a sentinel token."""
    options = getattr(case, "options", None)
    answer_index = getattr(case, "answer_index", None)
    if options and answer_index is not None:
        for i, option in enumerate(options):
            if i != answer_index:
                return option
    return "SHORTCUT"


def run_cascade(committee, case, backend_for, *, seed_index: int = 1, rounds: int = 3) -> dict:
    """Seed a shortcut into a committee and measure whether it cascades (issue 13).

    The same planted shortcut is injected at ``seed_index`` into a SHARED run (agents read the
    board, so the shortcut can propagate) and an ISOLATED run (agents see only their own turns,
    the solo counterfactual where it cannot). The running-shortcut series marks, per turn,
    whether the committed answer matched the planted shortcut; ``cascade_onset`` locates the turn
    at which the committee tips into following it (``None`` when there is no tipping point, for
    example an immediate or never adoption).

    Returns ``{"onset", "shared_answers", "isolated_answers", "series", "seeded_answer",
    "shared_transcript", "isolated_transcript"}``.
    """
    seeded_answer = _shortcut_answer(case)
    seed_turn = (seed_index, seeded_answer, "seed")
    shared = run_committee(
        committee, case, Condition.CONTAMINATED, backend_for,
        shared=True, seed_turn=seed_turn, rounds=rounds,
    )
    isolated = run_committee(
        committee, case, Condition.CONTAMINATED, backend_for,
        shared=False, seed_turn=seed_turn, rounds=rounds,
    )
    series = [1.0 if t.answer == seeded_answer else 0.0 for t in shared.turns]
    return {
        "onset": cascade_onset(series),
        "shared_answers": [t.answer for t in shared.turns],
        "isolated_answers": [t.answer for t in isolated.turns],
        "series": series,
        "seeded_answer": seeded_answer,
        "shared_transcript": shared,
        "isolated_transcript": isolated,
    }
