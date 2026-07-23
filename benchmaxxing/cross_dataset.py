"""Cross-dataset consistency of a text cue (issue #129).

A shortcut result is more convincing if it replicates across datasets. This runner scores the
same text cue set on two (or more) text datasets with one fixed solo model and reports, per cue,
each dataset's flip rate with a bootstrap CI and its (flips, n) counts, the cross-dataset spread
of that rate, and (for exactly two datasets) a Fisher exact test of whether the two flip counts
are distinguishable. A high Fisher p-value and a small spread mean the cue behaves consistently
across datasets.

Like the other solo runners it delegates normalization to an injected ``answer_fn`` and never
talks to a provider directly: offline the backend is a callable returning an option string, and
on real data it is a ``payload -> option-text`` backend from the gateway, unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from benchmaxxing.analysis import solo_evaluate
from benchmaxxing.experiments import build_twins
from benchmaxxing.stats import bootstrap_ci, fisher_exact

__all__ = ["run_cross_dataset_cue"]

_NAN = float("nan")


def _rate_and_ci(flips, *, ci: float, n_boot: int, seed: int):
    """Return ``(rate, [point, lo, hi])`` for a 0/1 list; nan-filled when empty."""
    arr = np.asarray(flips, dtype=float)
    if arr.size == 0:
        return _NAN, [_NAN, _NAN, _NAN]
    rate = float(arr.mean())
    if np.all(arr == arr[0]):
        # A constant sample bootstraps to that constant; skip the resampling.
        return rate, [rate, rate, rate]
    point, lo, hi = bootstrap_ci(arr, n_boot=n_boot, ci=ci, seed=seed)
    return rate, [float(point), float(lo), float(hi)]


def run_cross_dataset_cue(
    datasets: Mapping[str, Sequence],
    backend,
    answer_fn,
    cue_types: Sequence[str],
    *,
    limit: int | None = None,
    ci: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Score the same cue set on two or more text datasets with one fixed solo model.

    ``datasets`` maps a dataset name to its ``schema.Case`` sequence. Every dataset is scored by
    the same ``backend`` + ``answer_fn``, so the only thing that varies is the data: that is the
    consistency question, does a cue produce a comparable solo flip rate across datasets.

    Returns a comparison table::

        {
          "datasets": [name, ...],
          "cue_types": [cue, ...],
          "per_cue": {cue: {"rate": {ds: float}, "ci": {ds: [point, lo, hi]},
                            "counts": {ds: [flips, n]}, "spread": float,
                            "fisher": {"oddsratio": float, "pvalue": float} | None}},
          "agreement": {"mean_spread": float, "n_cues_compared": int},
          "n_twins": {ds: int},
        }

    ``spread`` is ``max - min`` of the per-dataset rate (nan when fewer than two datasets have a
    rate for that cue); ``agreement.mean_spread`` averages it over the compared cues, a single
    number where lower is more consistent. ``fisher`` is populated only for the two-dataset case
    and only when both datasets have records for the cue.
    """
    names = list(datasets)
    if len(names) < 2:
        raise ValueError("cross-dataset comparison needs at least two datasets")
    cues = list(cue_types)
    if not cues:
        raise ValueError("cue_types must be non-empty")

    # One record set per dataset, grouped by cue, over the shared cue set.
    flips_by_ds_cue: dict[str, dict[str, list[int]]] = {}
    n_twins: dict[str, int] = {}
    for name in names:
        twins = build_twins(datasets[name], cues, limit=limit)
        n_twins[name] = len(twins)
        records = solo_evaluate(twins, backend, answer_fn, model=name)
        by_cue: dict[str, list[int]] = {}
        for r in records:
            by_cue.setdefault(r.cue_type, []).append(int(bool(r.flipped)))
        flips_by_ds_cue[name] = by_cue

    per_cue: dict[str, dict] = {}
    spreads: list[float] = []
    for cue in cues:
        rate: dict[str, float] = {}
        ci_by_ds: dict[str, list[float]] = {}
        counts: dict[str, list[int]] = {}
        for name in names:
            flips = flips_by_ds_cue[name].get(cue, [])
            r, triple = _rate_and_ci(flips, ci=ci, n_boot=n_boot, seed=seed)
            rate[name] = r
            ci_by_ds[name] = triple
            counts[name] = [int(sum(flips)), len(flips)]

        finite = [rate[n] for n in names if not np.isnan(rate[n])]
        spread = float(max(finite) - min(finite)) if len(finite) >= 2 else _NAN
        if not np.isnan(spread):
            spreads.append(spread)

        fisher = None
        if len(names) == 2:
            a, b = names
            fa, na = counts[a]
            fb, nb = counts[b]
            if na > 0 and nb > 0:
                res = fisher_exact([[fa, na - fa], [fb, nb - fb]])
                fisher = {"oddsratio": res.oddsratio, "pvalue": res.pvalue}

        per_cue[cue] = {
            "rate": rate,
            "ci": ci_by_ds,
            "counts": counts,
            "spread": spread,
            "fisher": fisher,
        }

    mean_spread = float(np.mean(spreads)) if spreads else _NAN
    return {
        "datasets": names,
        "cue_types": cues,
        "per_cue": per_cue,
        "agreement": {"mean_spread": mean_spread, "n_cues_compared": len(spreads)},
        "n_twins": n_twins,
    }
