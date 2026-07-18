"""Stage 1 solo baselines and the lineage-overlap test.

This module measures single-model shortcut reliance on clean/contaminated twin pairs and
tests whether models of the same lineage fail on the same cases (a shared-shortcut signal).

Design notes:
- ``solo_evaluate`` injects both the model backend and the answer normalizer, so the whole
  pipeline is a pure function of its inputs and runs offline with no API keys.
- Overlap metrics reuse ``benchmaxxing.stats`` (phi / jaccard) when that module is present.
  Until it is, a numpy/scikit-learn fallback with the same contract is used, so this module
  imports and tests standalone. Neither ``benchmaxxing.stats`` nor ``benchmaxxing.gateway``
  is imported at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmaxxing.schema import Condition

__all__ = [
    "FlipRecord",
    "solo_evaluate",
    "flip_rate",
    "shortcut_reliance_index",
    "susceptibility_matrix",
    "failure_vector",
    "lineage_overlap_test",
]


@dataclass(frozen=True)
class FlipRecord:
    """One clean/contaminated observation for a single model on one twin pair.

    ``flipped`` is the headline signal: the normalized answer changed when the spurious cue
    was injected. ``clean_correct`` / ``contaminated_correct`` are ``None`` when the twin pair
    carries no ground truth, so accuracy-based measures can skip those records cleanly.
    """

    case_id: str
    cue_type: str
    model: str
    clean_answer: object
    contaminated_answer: object
    ground_truth: object
    flipped: bool
    clean_correct: bool | None
    contaminated_correct: bool | None


def _invoke(backend, payload):
    """Run a backend on one payload. Accepts a callable or a ``run``/``generate`` object."""
    run = getattr(backend, "run", None)
    if callable(run):
        return run(payload)
    generate = getattr(backend, "generate", None)
    if callable(generate):
        return generate(payload)
    if callable(backend):
        return backend(payload)
    raise TypeError("backend must be callable or expose a run()/generate() method.")


def _backend_name(backend) -> str:
    for attr in ("name", "model", "model_id"):
        val = getattr(backend, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(backend).__name__


def solo_evaluate(twin_pairs, backend, answer_fn, model=None):
    """Run ``backend`` on the clean and contaminated payload of each twin pair.

    Parameters
    ----------
    twin_pairs:
        Iterable of ``benchmaxxing.schema.TwinPair``.
    backend:
        A model backend. Either a callable ``backend(payload) -> raw`` or an object exposing
        ``run(payload)`` / ``generate(payload)`` (for example ``gateway.MockBackend``).
    answer_fn:
        Normalizer ``answer_fn(raw_output) -> answer``. Injected so tests stay deterministic.
    model:
        Optional model label. Defaults to a name derived from the backend.

    Returns
    -------
    list[FlipRecord]
    """
    model_name = model if model is not None else _backend_name(backend)
    records: list[FlipRecord] = []
    for tp in twin_pairs:
        clean_raw = _invoke(backend, tp.payload(Condition.CLEAN))
        contaminated_raw = _invoke(backend, tp.payload(Condition.CONTAMINATED))
        clean_answer = answer_fn(clean_raw)
        contaminated_answer = answer_fn(contaminated_raw)
        gt = tp.ground_truth
        clean_correct = None if gt is None else bool(clean_answer == gt)
        contaminated_correct = None if gt is None else bool(contaminated_answer == gt)
        records.append(
            FlipRecord(
                case_id=str(tp.case_id),
                cue_type=str(tp.cue_type),
                model=str(model_name),
                clean_answer=clean_answer,
                contaminated_answer=contaminated_answer,
                ground_truth=gt,
                flipped=bool(clean_answer != contaminated_answer),
                clean_correct=clean_correct,
                contaminated_correct=contaminated_correct,
            )
        )
    return records


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def flip_rate(records) -> dict:
    """Overall and per-cue flip rate.

    Returns ``{"overall": float, "per_cue": {cue: float}, "n": int}``. Rates are ``nan`` when
    there are no matching records.
    """
    records = list(records)
    overall = _mean([r.flipped for r in records])
    per_cue = {}
    for cue in sorted({r.cue_type for r in records}):
        per_cue[cue] = _mean([r.flipped for r in records if r.cue_type == cue])
    return {"overall": overall, "per_cue": per_cue, "n": len(records)}


def shortcut_reliance_index(records) -> dict:
    """Performance delta between the clean and contaminated conditions.

    The index is ``clean_accuracy - contaminated_accuracy``: a positive value means the
    injected cue degraded accuracy, that is the model leaned on the shortcut. Records without
    ground truth are ignored. Returns overall and per-cue deltas plus the two accuracies.
    """
    records = [
        r for r in records if r.clean_correct is not None and r.contaminated_correct is not None
    ]
    clean_acc = _mean([r.clean_correct for r in records])
    contaminated_acc = _mean([r.contaminated_correct for r in records])
    per_cue = {}
    for cue in sorted({r.cue_type for r in records}):
        sel = [r for r in records if r.cue_type == cue]
        per_cue[cue] = _mean([r.clean_correct for r in sel]) - _mean(
            [r.contaminated_correct for r in sel]
        )
    return {
        "overall": clean_acc - contaminated_acc,
        "clean_accuracy": clean_acc,
        "contaminated_accuracy": contaminated_acc,
        "per_cue": per_cue,
        "n": len(records),
    }


def susceptibility_matrix(records_by_model) -> dict:
    """Models x cue-types flip-rate matrix.

    Parameters
    ----------
    records_by_model:
        ``{model_name: list[FlipRecord]}``.

    Returns
    -------
    dict
        ``{"models": [...], "cues": [...], "matrix": np.ndarray}`` where ``matrix[i, j]`` is
        the flip rate of ``models[i]`` on ``cues[j]`` (``nan`` if that model has no record for
        that cue). The matrix has shape ``(len(models), len(cues))``.
    """
    models = sorted(records_by_model.keys())
    cues = sorted({r.cue_type for recs in records_by_model.values() for r in recs})
    matrix = np.full((len(models), len(cues)), np.nan, dtype=float)
    cue_index = {cue: j for j, cue in enumerate(cues)}
    for i, model in enumerate(models):
        by_cue: dict[str, list] = {}
        for r in records_by_model[model]:
            by_cue.setdefault(r.cue_type, []).append(bool(r.flipped))
        for cue, flips in by_cue.items():
            if flips:
                matrix[i, cue_index[cue]] = float(np.mean(flips))
    return {"models": models, "cues": cues, "matrix": matrix}


def failure_vector(records) -> np.ndarray:
    """Per-case binary flip vector for one model.

    Records are ordered by ``(case_id, cue_type)`` so vectors from different models line up
    when they were produced from the same twin-pair set. Returns an ``int`` array of 0/1.
    """
    ordered = sorted(records, key=lambda r: (str(r.case_id), str(r.cue_type)))
    return np.array([int(bool(r.flipped)) for r in ordered], dtype=int)


def _phi_fallback(x, y) -> float:
    """Phi coefficient (binary Pearson / Matthews correlation) via scikit-learn."""
    from sklearn.metrics import matthews_corrcoef

    x = np.asarray(x).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        # Phi is undefined when either vector is constant.
        return float("nan")
    return float(matthews_corrcoef(x, y))


def _jaccard_fallback(x, y) -> float:
    """Jaccard overlap of the positive (flipped) sets via scikit-learn."""
    from sklearn.metrics import jaccard_score

    x = np.asarray(x).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.sum() == 0 and y.sum() == 0:
        # Empty union: Jaccard is undefined.
        return float("nan")
    return float(jaccard_score(x, y))


def _resolve_metric(metric: str):
    """Return the overlap function for ``metric``, preferring ``benchmaxxing.stats``."""
    try:
        from benchmaxxing import stats as _stats

        fn = getattr(_stats, metric, None)
        if callable(fn):
            return fn
    except Exception:
        # benchmaxxing.stats not available yet: use the local reuse-based fallback.
        pass
    if metric == "phi":
        return _phi_fallback
    if metric == "jaccard":
        return _jaccard_fallback
    raise ValueError(f"Unknown overlap metric {metric!r}. Use 'phi' or 'jaccard'.")


def _normalize_vectors(failure_vectors, lineages):
    if isinstance(failure_vectors, dict):
        names = list(failure_vectors.keys())
        vecs = [np.asarray(failure_vectors[k]).astype(int).ravel() for k in names]
        if isinstance(lineages, dict):
            lin = [lineages[k] for k in names]
        else:
            lin = list(lineages)
    else:
        vecs = [np.asarray(v).astype(int).ravel() for v in failure_vectors]
        names = [f"model_{i}" for i in range(len(vecs))]
        lin = list(lineages)
    if len(lin) != len(vecs):
        raise ValueError("lineages length must match the number of failure vectors.")
    lengths = {int(v.shape[0]) for v in vecs}
    if len(lengths) > 1:
        raise ValueError(f"Failure vectors must be equal length, got lengths {sorted(lengths)}.")
    return names, vecs, lin


def lineage_overlap_test(
    failure_vectors,
    lineages,
    metric: str = "phi",
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """Within-lineage vs cross-lineage failure overlap, with a permutation p-value.

    For every pair of models the overlap of their binary failure vectors is scored with
    ``metric`` (``"phi"`` or ``"jaccard"``, reused from ``benchmaxxing.stats`` when present).
    Pairs are split by whether the two models share a lineage. The observed statistic is
    ``mean(within) - mean(cross)``. A one-sided permutation test then shuffles the lineage
    labels across models ``n_permutations`` times to estimate how often the gap is at least as
    large by chance.

    Parameters
    ----------
    failure_vectors:
        ``{model_name: vector}`` or a sequence of equal-length binary vectors.
    lineages:
        ``{model_name: lineage}`` or a sequence of lineage labels parallel to the vectors.
    metric:
        ``"phi"`` or ``"jaccard"``.
    n_permutations:
        Number of label shuffles for the null.
    seed:
        Seed for the numpy generator, so the p-value is reproducible.

    Returns
    -------
    dict
        ``within_mean``, ``cross_mean``, ``observed_diff``, ``p_value``, plus ``metric``,
        ``n_models`` and the number of valid permutations.
    """
    names, vecs, lin = _normalize_vectors(failure_vectors, lineages)
    n = len(names)
    if n < 2:
        raise ValueError("Need at least 2 models for an overlap test.")
    metric_fn = _resolve_metric(metric)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    overlap = np.array([metric_fn(vecs[i], vecs[j]) for (i, j) in pairs], dtype=float)

    def diff_for(labels) -> float:
        within, cross = [], []
        for k, (i, j) in enumerate(pairs):
            o = overlap[k]
            if np.isnan(o):
                continue
            (within if labels[i] == labels[j] else cross).append(o)
        wm = float(np.mean(within)) if within else float("nan")
        cm = float(np.mean(cross)) if cross else float("nan")
        return wm, cm

    within_mean, cross_mean = diff_for(lin)
    observed = within_mean - cross_mean

    rng = np.random.default_rng(seed)
    lin_arr = np.asarray(lin, dtype=object)
    count = 0
    valid = 0
    for _ in range(int(n_permutations)):
        perm_labels = lin_arr[rng.permutation(n)]
        wm, cm = diff_for(perm_labels)
        d = wm - cm
        if np.isnan(d):
            continue
        valid += 1
        if d >= observed - 1e-12:
            count += 1
    p_value = (count + 1) / (valid + 1) if valid else float("nan")

    return {
        "metric": metric,
        "within_mean": within_mean,
        "cross_mean": cross_mean,
        "observed_diff": observed,
        "p_value": p_value,
        "n_permutations": valid,
        "n_models": n,
    }
