"""Tests for the datasets module: manifest I/O, the registry, and adapter stubs."""

from __future__ import annotations

import pytest

from benchmaxxing.data import load_cases, write_manifest
from benchmaxxing.datasets import base, registry, status
from benchmaxxing.schema import Case, Modality

EXPECTED_DATASETS = {"mimic_cxr", "chexpert", "nih_cxr14", "medqa", "medmcqa", "pubmedqa", "ehr"}


def _write_text(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_imaging_csv(tmp_path):
    manifest = _write_text(
        tmp_path / "imaging.csv",
        "case_id,patient_id,image_ref,report,label\n"
        "c1,p1,images/a.png,No acute findings.,No Finding\n"
        "c2,p2,images/b.png,Small left effusion.,Effusion\n",
    )
    cases = load_cases(manifest)
    assert len(cases) == 2
    assert all(isinstance(c, Case) for c in cases)
    assert all(c.modality is Modality.IMAGE for c in cases)
    # distinct images
    assert {c.image_ref for c in cases} == {"images/a.png", "images/b.png"}
    assert cases[0].patient_id == "p1"
    assert cases[1].label == "Effusion"


def test_imaging_round_trip(tmp_path):
    cases = [
        Case("c1", "p1", Modality.IMAGE, image_ref="a.png", report="clear", label="No Finding"),
        Case("c2", "p2", Modality.IMAGE, image_ref="b.png", report="effusion", label="Effusion"),
    ]
    out = write_manifest(cases, tmp_path / "out.csv")
    reloaded = load_cases(out)
    assert [c.case_id for c in reloaded] == ["c1", "c2"]
    assert [c.image_ref for c in reloaded] == ["a.png", "b.png"]
    assert reloaded[0].modality is Modality.IMAGE


def test_mcq_round_trip(tmp_path):
    cases = [
        Case(
            "q1",
            "q1",
            Modality.TEXT,
            question="Which vessel?",
            options=("Aorta", "Vena cava", "Pulmonary artery"),
            answer_index=2,
        ),
    ]
    out = write_manifest(cases, tmp_path / "mcq.csv")
    reloaded = load_cases(out)
    assert len(reloaded) == 1
    got = reloaded[0]
    assert got.modality is Modality.TEXT
    assert got.options == ("Aorta", "Vena cava", "Pulmonary artery")
    assert got.answer_index == 2


def test_load_mcq_csv_without_patient_id(tmp_path):
    manifest = _write_text(
        tmp_path / "mcq.csv",
        "case_id,question,options,answer_index\n"
        "q1,Best next step?,Aspirin|Warfarin|Heparin,1\n"
        "q2,Likely diagnosis?,MI|PE|Pneumonia,2\n",
    )
    cases = load_cases(manifest)
    assert len(cases) == 2
    assert all(c.modality is Modality.TEXT for c in cases)
    # patient_id defaults to case_id when the column is absent
    assert cases[0].patient_id == "q1"
    assert cases[0].options == ("Aspirin", "Warfarin", "Heparin")
    assert cases[1].answer_index == 2


def test_missing_image_ref_raises(tmp_path):
    manifest = _write_text(
        tmp_path / "bad.csv",
        "case_id,patient_id,image_ref,report\nc1,p1,,No findings.\n",
    )
    with pytest.raises(ValueError, match="cannot detect modality"):
        load_cases(manifest)


def test_answer_index_out_of_range_raises(tmp_path):
    manifest = _write_text(
        tmp_path / "bad_mcq.csv",
        "case_id,question,options,answer_index\nq1,Which?,A|B,5\n",
    )
    with pytest.raises(ValueError, match="out of range"):
        load_cases(manifest)


def test_registry_lists_all_datasets():
    assert set(registry.names()) == EXPECTED_DATASETS
    assert len(registry.REGISTRY) == len(EXPECTED_DATASETS)
    for name in EXPECTED_DATASETS:
        module = registry.get(name)
        assert module.SPEC.name == name
        assert isinstance(module.SPEC, base.DatasetSpec)


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        registry.get("nope")


def test_dataset_status_covers_registered_adapters():
    registered = set(registry.names())
    status_names = set(status.names())

    assert registered <= status_names
    for name in registered:
        entry = status.get(name)
        assert entry.lane
        assert entry.staged
        assert entry.adapter
        assert entry.solo
        assert entry.cascade
        assert entry.plausibility
        assert entry.referee


def test_dataset_status_coverage_values_are_explicit():
    allowed = {"done", "pending", "not applicable"}

    for name in status.names():
        entry = status.get(name)
        assert entry.solo in allowed
        assert entry.cascade in allowed
        assert entry.plausibility in allowed
        assert entry.referee in allowed


def test_dataset_status_get_unknown_raises():
    with pytest.raises(KeyError, match="Unknown dataset status"):
        status.get("nope")




def test_adapter_build_manifest_fails_loudly_on_missing_data(tmp_path):
    # mimic_cxr is implemented; pointed at an empty dir (no metadata csv) it must raise a
    # clear error rather than silently produce an empty or wrong manifest.
    module = registry.get("mimic_cxr")
    with pytest.raises((FileNotFoundError, ValueError)):
        module.build_manifest(tmp_path, tmp_path / "m.csv")


def test_all_adapters_fail_loudly_on_missing_data(tmp_path):
    # Every adapter's build_manifest must raise on an empty raw_root (no silent success).
    # The CSV-backed adapters raise FileNotFoundError; the ehr entry point raises
    # NotImplementedError pointing at load_resource_contexts.
    for name in EXPECTED_DATASETS:
        module = registry.get(name)
        with pytest.raises((FileNotFoundError, ValueError, NotImplementedError)):
            module.build_manifest(tmp_path, tmp_path / f"{name}.csv")


def test_finalize_writes_and_rejects_duplicates(tmp_path):
    cases = [
        Case("c1", "p1", Modality.IMAGE, image_ref="a.png", report="r"),
        Case("c2", "p2", Modality.IMAGE, image_ref="b.png", report="r"),
    ]
    out = base.finalize(cases, tmp_path / "final.csv")
    assert load_cases(out)[0].case_id == "c1"

    dupes = [
        Case("c1", "p1", Modality.IMAGE, image_ref="a.png", report="r"),
        Case("c1", "p2", Modality.IMAGE, image_ref="b.png", report="r"),
    ]
    with pytest.raises(ValueError, match="Duplicate case_id"):
        base.finalize(dupes, tmp_path / "dupe.csv")
