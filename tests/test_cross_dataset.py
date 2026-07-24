"""Tests for the cross-dataset cue-consistency runner (issue #129).

Everything runs offline. The solo lane uses a deterministic ``pick-first`` backend over the real
text-cue payloads: under ``option_order`` (seed 0) a 3-option case is reordered so the first
option changes and the backend flips, while a 2-option case gets the identity permutation and
never flips. That lets a single fixed backend produce a chosen flip rate per dataset, so a
cross-dataset comparison can be built and asserted exactly. On real data the same call runs
against a gateway ``payload -> option-text`` backend unchanged; only the injected backend changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing.cross_dataset import run_cross_dataset_cue
from benchmaxxing.schema import Case, Modality


def _case(case_id: str, options, answer_index: int = 0) -> Case:
    return Case(
        case_id=case_id,
        patient_id=f"p-{case_id}",
        modality=Modality.TEXT,
        question=f"Which option is correct for {case_id}?",
        options=tuple(options),
        answer_index=answer_index,
    )


def _three(case_id: str) -> Case:
    # option_order (seed 0) permutes 3 options to [2, 0, 1], so options[0] changes -> flip.
    return _case(case_id, [f"{case_id}-o0", f"{case_id}-o1", f"{case_id}-o2"])


def _two(case_id: str) -> Case:
    # option_order (seed 0) is the identity on 2 options, so options[0] is unchanged -> no flip.
    return _case(case_id, [f"{case_id}-o0", f"{case_id}-o1"])


def _pick_first(payload) -> str:
    """Position shortcut: always take the first option (by text)."""
    return payload["options"][0]


def _identity(raw):
    return raw


def test_two_dataset_table_shape_and_consistency():
    # A [3-option, 2-option] mix flips the pick-first backend on exactly half the cases, so both
    # datasets sit at 0.5 -> a consistent cue: zero spread and an indistinguishable Fisher test.
    datasets = {
        "medqa": [_three("a0"), _two("a1")],
        "medmcqa": [_three("b0"), _two("b1")],
    }
    res = run_cross_dataset_cue(datasets, _pick_first, _identity, ["option_order"], n_boot=200)

    assert res["datasets"] == ["medqa", "medmcqa"]
    assert res["cue_types"] == ["option_order"]
    assert res["n_twins"] == {"medqa": 2, "medmcqa": 2}

    cue = res["per_cue"]["option_order"]
    assert cue["rate"]["medqa"] == pytest.approx(0.5)
    assert cue["rate"]["medmcqa"] == pytest.approx(0.5)
    assert cue["counts"] == {"medqa": [1, 2], "medmcqa": [1, 2]}
    assert cue["spread"] == pytest.approx(0.0)
    # 2x2 = [[1, 1], [1, 1]]: identical odds, so the datasets are statistically indistinguishable.
    assert cue["fisher"]["pvalue"] == pytest.approx(1.0)
    assert cue["fisher"]["oddsratio"] == pytest.approx(1.0)
    assert res["agreement"] == {"mean_spread": pytest.approx(0.0), "n_cues_compared": 1}


def test_spread_when_datasets_disagree():
    # All-3-option flips every case (1.0); all-2-option flips none (0.0): a maximal spread.
    datasets = {
        "medqa": [_three("a0"), _three("a1")],
        "medmcqa": [_two("b0"), _two("b1")],
    }
    res = run_cross_dataset_cue(datasets, _pick_first, _identity, ["option_order"], n_boot=200)

    cue = res["per_cue"]["option_order"]
    assert cue["rate"]["medqa"] == pytest.approx(1.0)
    assert cue["rate"]["medmcqa"] == pytest.approx(0.0)
    assert cue["counts"] == {"medqa": [2, 2], "medmcqa": [0, 2]}
    assert cue["spread"] == pytest.approx(1.0)
    # 2x2 = [[2, 0], [0, 2]]: exact Fisher two-sided p for this table is 1/3.
    assert cue["fisher"]["pvalue"] == pytest.approx(1.0 / 3.0)
    assert res["agreement"]["mean_spread"] == pytest.approx(1.0)
    assert res["agreement"]["n_cues_compared"] == 1


def test_ci_brackets_the_point_estimate():
    datasets = {
        "medqa": [_three("a0"), _two("a1")],
        "medmcqa": [_three("b0"), _two("b1")],
    }
    res = run_cross_dataset_cue(datasets, _pick_first, _identity, ["option_order"], n_boot=500)

    for name in ("medqa", "medmcqa"):
        point, lo, hi = res["per_cue"]["option_order"]["ci"][name]
        assert lo <= point <= hi
        assert 0.0 <= lo and hi <= 1.0
        assert point == pytest.approx(res["per_cue"]["option_order"]["rate"][name])


def test_unbuildable_cue_is_nan_and_excluded_from_agreement():
    # demographic_hint needs a `hint` param, so build_twins skips it for every case: no records,
    # a nan rate, no Fisher test, and it is left out of the agreement average.
    datasets = {
        "medqa": [_three("a0"), _two("a1")],
        "medmcqa": [_three("b0"), _two("b1")],
    }
    res = run_cross_dataset_cue(
        datasets, _pick_first, _identity, ["option_order", "demographic_hint"], n_boot=200
    )

    hint = res["per_cue"]["demographic_hint"]
    assert hint["counts"] == {"medqa": [0, 0], "medmcqa": [0, 0]}
    assert np.isnan(hint["rate"]["medqa"]) and np.isnan(hint["rate"]["medmcqa"])
    assert np.isnan(hint["spread"])
    assert hint["fisher"] is None
    # Only the buildable cue is counted in the agreement statistic.
    assert res["agreement"]["n_cues_compared"] == 1


def test_more_than_two_datasets_has_spread_but_no_pairwise_fisher():
    datasets = {
        "medqa": [_three("a0")],
        "medmcqa": [_two("b0")],
        "pubmedqa": [_three("c0")],
    }
    res = run_cross_dataset_cue(datasets, _pick_first, _identity, ["option_order"], n_boot=200)

    cue = res["per_cue"]["option_order"]
    assert set(cue["rate"]) == {"medqa", "medmcqa", "pubmedqa"}
    assert cue["spread"] == pytest.approx(1.0)  # max(1,0,1) - min(1,0,1)
    assert cue["fisher"] is None  # Fisher is defined only for the two-dataset case


def test_degenerate_fisher_oddsratio_is_none():
    # Both datasets flip every case -> 2x2 [[2, 0], [2, 0]]: valid p-value, undefined odds ratio.
    all_flip = {"medqa": [_three("a0"), _three("a1")], "medmcqa": [_three("b0"), _three("b1")]}
    res = run_cross_dataset_cue(all_flip, _pick_first, _identity, ["option_order"], n_boot=200)
    fisher = res["per_cue"]["option_order"]["fisher"]
    assert fisher["oddsratio"] is None
    assert fisher["pvalue"] == pytest.approx(1.0)
    assert res["per_cue"]["option_order"]["spread"] == pytest.approx(0.0)

    # Both datasets flip nothing -> 2x2 [[0, 2], [0, 2]]: same story.
    none_flip = {"medqa": [_two("a0"), _two("a1")], "medmcqa": [_two("b0"), _two("b1")]}
    res2 = run_cross_dataset_cue(none_flip, _pick_first, _identity, ["option_order"], n_boot=200)
    assert res2["per_cue"]["option_order"]["fisher"]["oddsratio"] is None


def test_invalid_ci_and_n_boot_raise_even_on_constant_data():
    # All 2-option cases never flip, so the constant-sample path would otherwise skip bootstrap_ci;
    # validation is hoisted so a bad ci / n_boot still fails loudly.
    datasets = {"medqa": [_two("a0")], "medmcqa": [_two("b0")]}
    with pytest.raises(ValueError, match="ci must be"):
        run_cross_dataset_cue(datasets, _pick_first, _identity, ["option_order"], ci=1.5)
    with pytest.raises(ValueError, match="n_boot must be"):
        run_cross_dataset_cue(datasets, _pick_first, _identity, ["option_order"], n_boot=0)


def test_requires_at_least_two_datasets():
    with pytest.raises(ValueError, match="at least two datasets"):
        run_cross_dataset_cue({"medqa": [_three("a0")]}, _pick_first, _identity, ["option_order"])


def test_requires_non_empty_cue_types():
    datasets = {"medqa": [_three("a0")], "medmcqa": [_two("b0")]}
    with pytest.raises(ValueError, match="cue_types must be non-empty"):
        run_cross_dataset_cue(datasets, _pick_first, _identity, [])
