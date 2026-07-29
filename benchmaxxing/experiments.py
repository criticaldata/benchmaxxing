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
    run_cascade_ablation: seed-target stimulus-strength sweep with per-arm contagion. (issue 104)
"""

from __future__ import annotations

from benchmaxxing.analysis import (
    failure_vector,
    flip_rate,
    lineage_overlap_test,
    solo_evaluate,
    susceptibility_matrix,
)

# The case-anchored peer rationale already exists for the Break-it C runner (issue #138); reuse it
# so the cascade's reasoned seed and that runner speak in identical words and stay comparable.
from benchmaxxing.anchored_seed import anchored_seed_content, case_anchor
from benchmaxxing.blackboard import run_committee
from benchmaxxing.cues.text import build_text_twin
from benchmaxxing.extract import Abstention, parse_mcq_choice
from benchmaxxing.onset import cascade_onset
from benchmaxxing.progress import ProgressReporter, progress_reporter

# The committee's outcome rule (majority of committed answers) lives in referee; reuse it for the
# unseeded baseline verdict rather than reimplementing it, so the two definitions never drift.
from benchmaxxing.referee import _final_answer as committee_verdict
from benchmaxxing.schema import Condition

__all__ = [
    "build_twins",
    "run_pilot",
    "run_solo_baselines",
    "run_holes_test",
    "run_cascade",
    "run_cascade_ablation",
    "terse_seed_content",
    "reasoned_seed_content",
    "SEED_STYLES",
]

# Sentinel for "baseline not supplied", distinct from a genuine ``None`` verdict (a committee that
# abstains commits ``None``), so an actual ``None`` baseline is reused rather than recomputed.
_UNSET = object()


def _model_name(model) -> str:
    """Best-effort display name for a roster entry (``ModelSpec`` or a bare id)."""
    name = getattr(model, "name", None)
    return name if isinstance(name, str) and name else str(model)


def build_twins(cases, cue_types, limit: int | None = None) -> list:
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


# Internal alias kept for the private callers below; ``build_twins`` is the public name.
_build_twins = build_twins


def run_pilot(
    cases,
    backend,
    answer_fn,
    cue_types,
    limit: int = 50,
    *,
    progress: ProgressReporter | None = None,
    progress_every_n: int | None = None,
    dataset_name: str = "text",
) -> dict:
    """Stage-0 pilot: build twins for the first ``limit`` cases and see if the cues bite.

    One model (``backend`` + ``answer_fn``) is run solo over every clean/contaminated twin.
    ``cues_bite`` is ``True`` when at least one twin flipped the model's answer, the go/no-go
    signal for spending compute on the full arms.

    Returns ``{"flip_rate", "n", "cue_types", "cues_bite"}`` where ``flip_rate`` is the full
    ``analysis.flip_rate`` breakdown and ``n`` is the number of twins evaluated.
    """
    cues = list(cue_types)
    selected_cases = list(cases)
    if limit is not None:
        selected_cases = selected_cases[:limit]
    twins = _build_twins(selected_cases, cues)
    if progress is None and progress_every_n is not None:
        progress = progress_reporter(
            len(twins), label="run_pilot", every_n=progress_every_n
        )
    if progress is not None:
        progress.start(
            f"dataset={dataset_name} cases={len(selected_cases)} "
            f"twins={len(twins)} cues={len(cues)}"
        )
    records = solo_evaluate(twins, backend, answer_fn, progress=progress)
    return {
        "flip_rate": flip_rate(records),
        "n": len(records),
        "cue_types": cues,
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


def _parse(text: str, case) -> str | Abstention:
    """Parse a free-text response into the exact option string.
    
    Delegates to extract.parse_mcq_choice. If an explicit choice is found,
    maps its index to the corresponding option text for the harness.
    """
    options = getattr(case, "options", None)
    if not options:
        return Abstention.UNPARSEABLE
    
    idx = parse_mcq_choice(text, options)
    if isinstance(idx, Abstention):
        return idx
    return options[idx]


def _ground_truth(case):
    """The correct option text, or ``None`` when the case has no options/answer_index."""
    options = getattr(case, "options", None)
    answer_index = getattr(case, "answer_index", None)
    if options and answer_index is not None and 0 <= answer_index < len(options):
        return options[answer_index]
    return None


def _shortcut_answer(case):
    """A plausible planted-shortcut answer: a distractor option, or a sentinel token."""
    options = getattr(case, "options", None)
    answer_index = getattr(case, "answer_index", None)
    if options and answer_index is not None:
        for i, option in enumerate(options):
            if i != answer_index:
                return option
    return "SHORTCUT"


def _baseline_relative_shortcut(case, baseline_answer):
    """A planted-shortcut answer chosen relative to the committee's own baseline (issue #104).

    Returns the first option that is neither the ground truth nor the committee's independent
    ``baseline_answer``. Seeding an answer the committee would have given anyway leaves no
    counterfactual gap between the shared and isolated arms, so shared-minus-isolated adoption
    (the contagion metric) is undefined; picking an answer the committee would *not* independently
    give is what makes contagion measurable.

    Falls back to the legacy first-distractor pick when the case has no option distinct from both
    the truth and the baseline (for example a two-option case whose only distractor is already the
    baseline); callers detect that degenerate case via ``seeded_answer == baseline_answer``.
    """
    options = getattr(case, "options", None)
    if options:
        truth = _ground_truth(case)
        for option in options:
            if option != truth and option != baseline_answer:
                return option
    return _shortcut_answer(case)


def terse_seed_content(case, seeded_answer) -> str:
    """A minimal, deterministic confident-claim content for the seeded turn.

    A one-liner asserting the shortcut answer, a step up in stimulus strength from a bare answer
    with no content. The cue-anchored *reasoned* rationale is a separate, richer content builder
    (issue #115); this and any such builder plug into ``run_cascade``'s ``content_builder`` seam.
    """
    return f"The answer is {seeded_answer}."


def reasoned_seed_content(case, seeded_answer) -> str:
    """A confident wrong answer plus a rationale anchored on a cue in this case (issue #115).

    The seeded peer states the shortcut answer and justifies it by citing a detail lifted from the
    case's own vignette, so the reason is voiceable and checkable rather than a naked assertion.
    Delegates to the Break-it C builders (issue #138) rather than composing a third rationale
    wording, so a seed planted here is word-for-word the one that runner plants.

    Deterministic given the case and the answer: :func:`~benchmaxxing.anchored_seed.case_anchor`
    quotes the vignette rather than inventing a detail. A case with no text raises instead of
    degrading to a generic rationale, which would silently file a bare seed under the reasoned arm
    and bias the contrast this style exists to measure.
    """
    anchor = case_anchor(case)
    if not anchor:
        raise ValueError(
            f"case {getattr(case, 'case_id', None)!r} has no question/report text to anchor a "
            "reasoned seed on; supply a content_builder or use seed_style='terse'"
        )
    return anchored_seed_content(seeded_answer, anchor)


# Named seed styles for ``run_cascade``. ``"bare"`` maps to no builder (the historical answer-only
# seed) so the default path is byte-identical to the pre-#115 behaviour.
SEED_STYLES = {
    "bare": None,
    "terse": terse_seed_content,
    "reasoned": reasoned_seed_content,
}


def _committee_baseline(committee, case, backend_for, *, rounds: int) -> object:
    """The committee's independent verdict with no seed planted: an ISOLATED, unseeded run.

    Isolated so each member answers on its own (no premature agreement), unseeded so the answer is
    what the committee believes before any shortcut is introduced. Run under the SAME condition as
    the seeded arms (``CONTAMINATED``) so the seed is chosen against the exact verdict it is meant
    to displace; a condition-sensitive backend would otherwise be baselined against a different
    answer than the one adoption is measured over. This is the reference the baseline-relative seed
    target is chosen against.
    """
    transcript = run_committee(
        committee, case, Condition.CONTAMINATED, backend_for, shared=False, rounds=rounds,
    )
    return committee_verdict(transcript)


def run_cascade(
    committee,
    case,
    backend_for,
    *,
    seed_index: int = 1,
    rounds: int = 3,
    seed_target: str = "first_distractor",
    baseline_answer: object = _UNSET,
    seed_style: str = "bare",
    content_builder=None,
) -> dict:
    """Seed a shortcut into a committee and measure whether it cascades (issue 13).

    The same planted shortcut is injected at ``seed_index`` into a SHARED run (agents read the
    board, so the shortcut can propagate) and an ISOLATED run (agents see only their own turns,
    the solo counterfactual where it cannot). The running-shortcut series marks, per turn,
    whether the committed answer matched the planted shortcut; ``cascade_onset`` locates the turn
    at which the committee tips into following it (``None`` when there is no tipping point, for
    example an immediate or never adoption).

    ``seed_target`` selects how the shortcut answer is chosen (issue #104):

    * ``"first_distractor"`` (default, unchanged): the first non-correct option. Simple, but on a
      real run it frequently coincides with the committee's own independent answer, collapsing the
      counterfactual gap that contagion is measured over.
    * ``"baseline_relative"``: a non-correct option the committee would *not* independently give.
      The committee's unseeded verdict is computed here (or passed in via ``baseline_answer`` to
      avoid recomputing it across arms) and the seed is chosen distinct from it, so shared-minus-
      isolated adoption is well defined.

    ``seed_style`` selects what the seeded peer SAYS, holding the seeded answer fixed (issue #115):

    * ``"bare"`` (default, unchanged): the answer with no content, a naked assertion.
    * ``"terse"``: a one-line confident claim.
    * ``"reasoned"``: the answer plus a rationale anchored on a detail of this case, the stimulus
      a real cascade runs on.

    ``content_builder`` is the escape hatch for a style not in :data:`SEED_STYLES`: an optional
    ``(case, seeded_answer) -> str`` hook. Passing it together with a non-bare ``seed_style``
    raises rather than silently letting one win. The default keeps the bare-answer seed, so the
    seed construction and every existing arm are unchanged; the result dict only gains additive
    metadata keys. Ground truth is never altered by any of this: only the planted peer turn
    changes, and the seeded answer stays a non-correct option under either ``seed_target``.

    Returns ``{"onset", "shared_answers", "isolated_answers", "series", "seeded_answer",
    "seed_target", "baseline_answer", "seed_style", "seed_content", "shared_transcript",
    "isolated_transcript"}``.
    """
    if seed_target == "first_distractor":
        seeded_answer = _shortcut_answer(case)
    elif seed_target == "baseline_relative":
        if baseline_answer is _UNSET:
            baseline_answer = _committee_baseline(committee, case, backend_for, rounds=rounds)
        seeded_answer = _baseline_relative_shortcut(case, baseline_answer)
    else:
        raise ValueError(
            "seed_target must be 'first_distractor' or 'baseline_relative', "
            f"got {seed_target!r}"
        )
    reported_baseline = None if baseline_answer is _UNSET else baseline_answer

    if seed_style not in SEED_STYLES:
        raise ValueError(
            f"seed_style must be one of {sorted(SEED_STYLES)}, got {seed_style!r}"
        )
    if content_builder is not None and seed_style != "bare":
        raise ValueError(
            "pass either seed_style or content_builder, not both "
            f"(got seed_style={seed_style!r} with a content_builder)"
        )
    builder = content_builder if content_builder is not None else SEED_STYLES[seed_style]
    content = None if builder is None else str(builder(case, seeded_answer))
    seed_turn = (
        (seed_index, seeded_answer, "seed")
        if content is None
        else (seed_index, seeded_answer, "seed", content)
    )
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
        "seed_target": seed_target,
        "baseline_answer": reported_baseline,
        "seed_style": seed_style,
        "seed_content": content,
        "shared_transcript": shared,
        "isolated_transcript": isolated,
    }


def _seed_adoption(transcript, seeded_answer, seed_index: int) -> float | None:
    """Fraction of post-seed member turns that committed the seeded answer.

    Restricted to member turns after the seed slot (the population that could have adopted it) and
    excluding injected turns via the ``seeded`` flag rather than a reserved ``"seed"`` agent id, so
    a real member that happens to be named ``"seed"`` is still counted. ``None`` when no such turn
    exists, so a censored arm is not silently read as zero adoption.
    """
    adopters = [
        t for t in transcript.turns
        if not t.seeded and t.answer is not None and t.turn_index > seed_index
    ]
    if not adopters:
        return None
    return sum(1 for t in adopters if t.answer == seeded_answer) / len(adopters)


def run_cascade_ablation(
    committee,
    cases,
    backend_for,
    *,
    seed_index: int = 1,
    rounds: int = 3,
    seed_targets=("first_distractor", "baseline_relative"),
    seed_style: str = "bare",
    content_builder=None,
) -> dict:
    """Stimulus-strength ablation over seed targets, per arm (issue #104).

    ``seed_style`` fixes what the seeded peer says across every arm of one sweep (issue #115), so
    a style contrast is two sweeps over the same cases rather than a third dimension inside the
    arm keys, which keeps ``arms`` keyed by seed target alone.

    For each case the committee's unseeded baseline is computed once and shared across arms, so
    every arm is judged against the same reference: a case whose seed coincides with that baseline
    has no counterfactual gap and is counted as censored (its contagion is undefined) rather than
    folded into the mean as a spurious zero. This is the degenerate condition the first real run
    hit in 16/20 cases with the first-distractor seed; the ablation makes it visible and shows the
    baseline-relative target restore a measurable gap.

    Per-arm ``contagion`` is mean (shared adoption - isolated adoption) of the seeded answer over
    the arm's well-defined cases; ``onset_rate`` is the fraction of well-defined cases with a
    detected onset.

    Returns ``{"arms": {target: {contagion, onset_rate, n_cases, n_well_defined, n_censored}},
    "per_case": {target: [record, ...]}, "n_cases"}``.
    """
    cases = list(cases)
    arms: dict[str, dict] = {}
    per_case: dict[str, list] = {}
    for target in seed_targets:
        arms[target] = {}
        per_case[target] = []

    for case in cases:
        baseline = _committee_baseline(committee, case, backend_for, rounds=rounds)
        for target in seed_targets:
            res = run_cascade(
                committee, case, backend_for,
                seed_index=seed_index, rounds=rounds,
                seed_target=target, baseline_answer=baseline,
                seed_style=seed_style, content_builder=content_builder,
            )
            seeded = res["seeded_answer"]
            shared_adopt = _seed_adoption(res["shared_transcript"], seeded, seed_index)
            iso_adopt = _seed_adoption(res["isolated_transcript"], seeded, seed_index)
            well_defined = (
                seeded != baseline and shared_adopt is not None and iso_adopt is not None
            )
            contagion = (
                shared_adopt - iso_adopt
                if shared_adopt is not None and iso_adopt is not None
                else None
            )
            per_case[target].append({
                "case_id": getattr(case, "case_id", None),
                "seeded_answer": seeded,
                "baseline_answer": baseline,
                "seed_is_baseline": seeded == baseline,
                "onset": res["onset"],
                "shared_adoption": shared_adopt,
                "isolated_adoption": iso_adopt,
                "contagion": contagion,
                "well_defined": well_defined,
            })

    for target in seed_targets:
        records = per_case[target]
        defined = [r for r in records if r["well_defined"]]
        contagions = [r["contagion"] for r in defined if r["contagion"] is not None]
        onsets = [r for r in defined if r["onset"] is not None]
        arms[target] = {
            "contagion": (sum(contagions) / len(contagions)) if contagions else None,
            "onset_rate": (len(onsets) / len(defined)) if defined else None,
            "n_cases": len(records),
            "n_well_defined": len(defined),
            "n_censored": len(records) - len(defined),
        }

    return {"arms": arms, "per_case": per_case, "n_cases": len(cases), "seed_style": seed_style}
