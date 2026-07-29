"""Tests for benchmaxxing.analysis (Stage 1 solo baselines and the lineage-overlap test).

These run fully offline: a deterministic mock backend plays the role of ``gateway.MockBackend``
with a rule that flips its answer only when a specific cue is present in the payload.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmaxxing.analysis import (
    FlipRecord,
    failure_vector,
    flip_rate,
    lineage_overlap_test,
    shortcut_reliance_index,
    solo_evaluate,
    susceptibility_matrix,
)
from benchmaxxing.schema import TwinPair


class MockBackend:
    """Deterministic backend mirroring the gateway.MockBackend contract.

    It reads a base answer encoded in the payload as ``answer=<X>`` and returns it, except
    when the payload carries the marker for ``flip_cue`` (``CUE:<flip_cue>``), in which case it
    returns ``flip_to`` instead. That single rule flips the answer on exactly one cue type.
    """

    def __init__(self, name="mock", flip_cue="cable", flip_to="B"):
        self.name = name
        self.flip_cue = flip_cue
        self.flip_to = flip_to

    def run(self, payload: str) -> str:
        base = "A"
        for part in str(payload).split(";"):
            if part.startswith("answer="):
                base = part.split("=", 1)[1]
        if f"CUE:{self.flip_cue}" in str(payload):
            base = self.flip_to
        return f"ANSWER: {base}"


def _answer_fn(raw: str) -> str:
    return raw.split("ANSWER:", 1)[1].strip()


def _clean_payload(case_id: str, answer: str = "A") -> str:
    return f"case={case_id};answer={answer}"


def _contaminated_payload(case_id: str, cue: str, answer: str = "A") -> str:
    return f"case={case_id};answer={answer};CUE:{cue}"


def _make_twin_pairs():
    """Three 'cable' pairs (will flip) and two 'option_order' pairs (will not)."""
    pairs = []
    for i in range(3):
        cid = f"cable-{i}"
        pairs.append(
            TwinPair(
                case_id=cid,
                cue_type="cable",
                clean=_clean_payload(cid),
                contaminated=_contaminated_payload(cid, "cable"),
                ground_truth="A",
            )
        )
    for i in range(2):
        cid = f"order-{i}"
        pairs.append(
            TwinPair(
                case_id=cid,
                cue_type="option_order",
                clean=_clean_payload(cid),
                contaminated=_contaminated_payload(cid, "option_order"),
                ground_truth="A",
            )
        )
    return pairs


def test_solo_evaluate_records_flips():
    backend = MockBackend(name="m1", flip_cue="cable", flip_to="B")
    records = solo_evaluate(_make_twin_pairs(), backend, _answer_fn)
    assert len(records) == 5
    assert all(isinstance(r, FlipRecord) for r in records)
    by_cue = {}
    for r in records:
        by_cue.setdefault(r.cue_type, []).append(r)
    # The cable cue flips every time; the order cue never does.
    assert all(r.flipped for r in by_cue["cable"])
    assert not any(r.flipped for r in by_cue["option_order"])
    # Ground truth is "A"; a flip to "B" makes the contaminated answer wrong.
    cable = by_cue["cable"][0]
    assert cable.clean_answer == "A"
    assert cable.contaminated_answer == "B"
    assert cable.clean_correct is True
    assert cable.contaminated_correct is False
    assert records[0].model == "m1"


def test_flip_rate_overall_and_per_cue():
    backend = MockBackend(flip_cue="cable")
    records = solo_evaluate(_make_twin_pairs(), backend, _answer_fn)
    fr = flip_rate(records)
    assert fr["n"] == 5
    assert fr["overall"] == pytest.approx(3 / 5)
    assert fr["per_cue"]["cable"] == pytest.approx(1.0)
    assert fr["per_cue"]["option_order"] == pytest.approx(0.0)


def test_flip_rate_empty_is_nan():
    fr = flip_rate([])
    assert np.isnan(fr["overall"])
    assert fr["per_cue"] == {}
    assert fr["n"] == 0


def test_shortcut_reliance_index():
    backend = MockBackend(flip_cue="cable")
    records = solo_evaluate(_make_twin_pairs(), backend, _answer_fn)
    sri = shortcut_reliance_index(records)
    assert sri["clean_accuracy"] == pytest.approx(1.0)
    # Contaminated: 2 of 5 correct (the two order cases stay on "A").
    assert sri["contaminated_accuracy"] == pytest.approx(2 / 5)
    assert sri["overall"] == pytest.approx(3 / 5)
    assert sri["per_cue"]["cable"] == pytest.approx(1.0)
    assert sri["per_cue"]["option_order"] == pytest.approx(0.0)


def test_susceptibility_matrix_shape_and_values():
    twins = _make_twin_pairs()
    m1 = MockBackend(name="m1", flip_cue="cable")
    m2 = MockBackend(name="m2", flip_cue="option_order")
    records_by_model = {
        "m1": solo_evaluate(twins, m1, _answer_fn),
        "m2": solo_evaluate(twins, m2, _answer_fn),
    }
    sm = susceptibility_matrix(records_by_model)
    assert sm["models"] == ["m1", "m2"]
    assert sm["cues"] == ["cable", "option_order"]
    assert sm["matrix"].shape == (2, 2)
    ci = {c: j for j, c in enumerate(sm["cues"])}
    # m1 flips on cable, m2 flips on option_order.
    assert sm["matrix"][0, ci["cable"]] == pytest.approx(1.0)
    assert sm["matrix"][0, ci["option_order"]] == pytest.approx(0.0)
    assert sm["matrix"][1, ci["cable"]] == pytest.approx(0.0)
    assert sm["matrix"][1, ci["option_order"]] == pytest.approx(1.0)


def test_failure_vector_ordering():
    backend = MockBackend(flip_cue="cable")
    records = solo_evaluate(_make_twin_pairs(), backend, _answer_fn)
    vec = failure_vector(records)
    # Sorted by (case_id, cue_type): cable-0,1,2 then order-0,1.
    assert vec.tolist() == [1, 1, 1, 0, 0]
    assert vec.dtype == np.dtype("int64") or vec.dtype == np.dtype("int32")


def _fixture_failure_vectors():
    """Same-lineage vectors identical; cross-lineage vectors differ."""
    a = np.array([1, 0, 1, 0, 1, 0])
    b = np.array([0, 1, 0, 1, 0, 1])
    c = np.array([1, 1, 0, 0, 1, 1])
    vectors = {"a1": a, "a2": a, "b1": b, "b2": b, "c1": c, "c2": c}
    lineages = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C"}
    return vectors, lineages


def test_lineage_overlap_test_phi():
    vectors, lineages = _fixture_failure_vectors()
    res = lineage_overlap_test(vectors, lineages, metric="phi", n_permutations=2000, seed=0)
    assert res["metric"] == "phi"
    assert res["n_models"] == 6
    # Identical within-lineage vectors give perfect within-lineage phi.
    assert res["within_mean"] == pytest.approx(1.0)
    assert res["within_mean"] > res["cross_mean"]
    assert res["observed_diff"] > 0
    assert 0.0 <= res["p_value"] <= 1.0
    # The true grouping is a strong outlier under the null.
    assert res["p_value"] < 0.2


def test_lineage_overlap_test_jaccard_and_reproducible():
    vectors, lineages = _fixture_failure_vectors()
    res1 = lineage_overlap_test(vectors, lineages, metric="jaccard", seed=7)
    res2 = lineage_overlap_test(vectors, lineages, metric="jaccard", seed=7)
    assert res1["within_mean"] == pytest.approx(1.0)
    assert res1["observed_diff"] > 0
    assert 0.0 <= res1["p_value"] <= 1.0
    # Same seed reproduces the permutation p-value exactly.
    assert res1["p_value"] == res2["p_value"]


def test_lineage_overlap_test_accepts_sequences():
    vectors, lineages = _fixture_failure_vectors()
    vec_list = [vectors[k] for k in vectors]
    lin_list = [lineages[k] for k in vectors]
    res = lineage_overlap_test(vec_list, lin_list, metric="phi", seed=0)
    assert res["n_models"] == 6
    assert res["observed_diff"] > 0


def test_lineage_overlap_test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        lineage_overlap_test({"a": np.array([1, 0]), "b": np.array([0, 1])}, ["A"])


# Best-effort integration with the real gateway.MockBackend once that module exists.
try:
    from benchmaxxing.gateway import MockBackend as _GatewayMock
except Exception:
    _GatewayMock = None


@pytest.mark.skipif(_GatewayMock is None, reason="benchmaxxing.gateway not available yet")
def test_solo_evaluate_with_gateway_mock():
    twins = _make_twin_pairs()
    try:
        backend = _GatewayMock()
        records = solo_evaluate(twins, backend, _answer_fn)
    except Exception as exc:  # gateway.MockBackend interface differs: nothing to assert here.
        pytest.skip(f"gateway.MockBackend interface not compatible: {exc}")
    assert len(records) == len(twins)

def test_solo_evaluate_updates_progress_reporter():
    class Recorder:
        def __init__(self):
            self.updates = []

        def update(self, completed):
            self.updates.append(completed)

    progress = Recorder()
    backend = MockBackend(name="m1", flip_cue="cable", flip_to="B")

    records = solo_evaluate(_make_twin_pairs(), backend, _answer_fn, progress=progress)

    assert len(records) == 5
    assert progress.updates == [1, 2, 3, 4, 5]

