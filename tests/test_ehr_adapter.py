"""Tests for the EHR resource-context loader (stage 5 scrutiny context)."""
from __future__ import annotations
import pytest
from benchmaxxing.datasets import ehr
from benchmaxxing.datasets.base import DatasetSpec
from benchmaxxing.schema import Modality
def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path
def test_load_two_row_csv(tmp_path):
    csv_path = _write(
        tmp_path / "resources.csv",
        "scenario_id,icu_stay_count,budget_pressure,shift\n"
        "s1,1,0.25,night\n"
        "s2,3,0.9,day\n",
    )
    contexts = ehr.load_resource_contexts(csv_path)
    assert len(contexts) == 2
    first, second = contexts
    assert first.scenario_id == "s1"
    assert first.icu_stay_count == 1.0
    assert first.budget_pressure == 0.25
    assert isinstance(first.icu_stay_count, float)
    assert isinstance(first.budget_pressure, float)
    # extra columns are preserved verbatim in meta
    assert first.meta == {"shift": "night"}
    assert second.scenario_id == "s2"
    assert second.icu_stay_count == 3.0
    assert second.budget_pressure == 0.9
    assert second.meta == {"shift": "day"}
def test_no_extra_columns_gives_empty_meta(tmp_path):
    csv_path = _write(
        tmp_path / "resources.csv",
        "scenario_id,icu_stay_count,budget_pressure\n"
        "s1,2,0.5\n",
    )
    (context,) = ehr.load_resource_contexts(csv_path)
    assert context.meta == {}
    assert context.budget_pressure == 0.5
def test_missing_required_column_raises(tmp_path):
    csv_path = _write(
        tmp_path / "bad.csv",
        "scenario_id,icu_stay_count\n"
        "s1,1\n",
    )
    with pytest.raises(ValueError, match="budget_pressure"):
        ehr.load_resource_contexts(csv_path)
def test_non_numeric_field_raises(tmp_path):
    csv_path = _write(
        tmp_path / "bad.csv",
        "scenario_id,icu_stay_count,budget_pressure\n"
        "s1,many,0.5\n",
    )
    with pytest.raises(ValueError, match="must be numeric"):
        ehr.load_resource_contexts(csv_path)
def test_empty_numeric_field_raises(tmp_path):
    csv_path = _write(
        tmp_path / "bad.csv",
        "scenario_id,icu_stay_count,budget_pressure\n"
        "s1,,0.5\n",
    )
    with pytest.raises(ValueError, match="is empty"):
        ehr.load_resource_contexts(csv_path)
def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ehr.load_resource_contexts(tmp_path / "nope.csv")
def test_resource_context_is_frozen(tmp_path):
    csv_path = _write(
        tmp_path / "resources.csv",
        "scenario_id,icu_stay_count,budget_pressure\n"
        "s1,2,0.5\n",
    )
    (context,) = ehr.load_resource_contexts(csv_path)
    with pytest.raises((AttributeError, TypeError)):
        context.icu_stay_count = 9.0
def test_spec_is_text_modality_and_documented():
    assert isinstance(ehr.SPEC, DatasetSpec)
    assert ehr.SPEC.name == "ehr"
    assert ehr.SPEC.modality is Modality.TEXT
    assert "MIMIC-IV" in ehr.SPEC.raw_hint
    assert "load_resource_contexts" in ehr.SPEC.notes
def test_build_manifest_points_to_loader(tmp_path):
    with pytest.raises(NotImplementedError, match="load_resource_contexts"):
        ehr.build_manifest(tmp_path, tmp_path / "ehr.csv")
def test_extra_column_empty_string_preserved_in_meta(tmp_path):
    # Real MIMIC-IV-derived exports can carry a genuinely empty (not missing) value in a
    # non-required column, e.g. an unset insurance field. Must round-trip as an empty string
    # in meta, not be dropped or coerced to None.
    csv_path = _write(
        tmp_path / "resources.csv",
        "scenario_id,icu_stay_count,budget_pressure,insurance\n"
        "s1,1,4.25,\n",
    )
    (context,) = ehr.load_resource_contexts(csv_path)
    assert context.meta == {"insurance": ""}
    assert context.icu_stay_count == 1.0