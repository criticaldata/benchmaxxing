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
        "scenario_id,staffing,beds,budget_pressure,shift\n"
        "s1,0.8,12,0.25,night\n"
        "s2,1.5,4,0.9,day\n",
    )
    contexts = ehr.load_resource_contexts(csv_path)
    assert len(contexts) == 2

    first, second = contexts
    assert first.scenario_id == "s1"
    assert first.staffing == 0.8
    assert first.beds == 12.0
    assert first.budget_pressure == 0.25
    assert isinstance(first.staffing, float)
    assert isinstance(first.beds, float)
    # extra columns are preserved verbatim in meta
    assert first.meta == {"shift": "night"}

    assert second.scenario_id == "s2"
    assert second.staffing == 1.5
    assert second.beds == 4.0
    assert second.budget_pressure == 0.9
    assert second.meta == {"shift": "day"}


def test_no_extra_columns_gives_empty_meta(tmp_path):
    csv_path = _write(
        tmp_path / "resources.csv",
        "scenario_id,staffing,beds,budget_pressure\n"
        "s1,2.0,8,0.5\n",
    )
    (context,) = ehr.load_resource_contexts(csv_path)
    assert context.meta == {}
    assert context.budget_pressure == 0.5


def test_missing_required_column_raises(tmp_path):
    csv_path = _write(
        tmp_path / "bad.csv",
        "scenario_id,staffing,beds\n"
        "s1,0.8,12\n",
    )
    with pytest.raises(ValueError, match="budget_pressure"):
        ehr.load_resource_contexts(csv_path)


def test_non_numeric_field_raises(tmp_path):
    csv_path = _write(
        tmp_path / "bad.csv",
        "scenario_id,staffing,beds,budget_pressure\n"
        "s1,plenty,12,0.5\n",
    )
    with pytest.raises(ValueError, match="must be numeric"):
        ehr.load_resource_contexts(csv_path)


def test_empty_numeric_field_raises(tmp_path):
    csv_path = _write(
        tmp_path / "bad.csv",
        "scenario_id,staffing,beds,budget_pressure\n"
        "s1,,12,0.5\n",
    )
    with pytest.raises(ValueError, match="is empty"):
        ehr.load_resource_contexts(csv_path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ehr.load_resource_contexts(tmp_path / "nope.csv")


def test_resource_context_is_frozen(tmp_path):
    csv_path = _write(
        tmp_path / "resources.csv",
        "scenario_id,staffing,beds,budget_pressure\n"
        "s1,2.0,8,0.5\n",
    )
    (context,) = ehr.load_resource_contexts(csv_path)
    with pytest.raises((AttributeError, TypeError)):
        context.staffing = 9.0


def test_spec_is_text_modality_and_documented():
    assert isinstance(ehr.SPEC, DatasetSpec)
    assert ehr.SPEC.name == "ehr"
    assert ehr.SPEC.modality is Modality.TEXT
    assert "MIMIC-IV" in ehr.SPEC.raw_hint
    assert "load_resource_contexts" in ehr.SPEC.notes


def test_build_manifest_points_to_loader(tmp_path):
    with pytest.raises(NotImplementedError, match="load_resource_contexts"):
        ehr.build_manifest(tmp_path, tmp_path / "ehr.csv")
