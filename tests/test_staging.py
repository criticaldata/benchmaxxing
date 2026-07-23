"""Tests for dataset staging (benchmaxxing.datasets.staging + `benchmaxxing datasets stage`).

Uses a tiny synthetic ChestX-ray14 layout: the point is the staging contract (validated manifest
plus a provenance record), not the pixels.
"""

from __future__ import annotations

import json

import pytest

from benchmaxxing import cli
from benchmaxxing.data import load_cases
from benchmaxxing.datasets import registry, staging

HEADER = (
    "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,"
    "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]"
)


def _nih_release(root, n=3, with_images=True):
    """A minimal ChestX-ray14 release: the metadata csv plus (optionally) its png files."""
    raw = root / "nih_cxr14"
    raw.mkdir(parents=True)
    rows = [HEADER]
    for i in range(n):
        rows.append(f"cxr{i}.png,Cardiomegaly|Effusion,0,{i},58,M,PA,2048,2500,0.143,0.143")
        if with_images:
            (raw / f"cxr{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (raw / "Data_Entry_2017.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return raw


def test_every_registered_dataset_has_a_source_entry():
    # a dataset nobody can trace back to a source is not stageable
    assert set(registry.names()) <= set(staging.SOURCES)
    for name, source in staging.SOURCES.items():
        assert source.access in {"open", "registration", "credentialed"}, name
        assert source.url.startswith("http"), name
        assert source.license, name


def test_dataset_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(staging.DATASET_ROOT_ENV, str(tmp_path))
    assert staging.raw_dir("nih_cxr14") == tmp_path / "nih_cxr14"
    monkeypatch.delenv(staging.DATASET_ROOT_ENV)
    assert staging.dataset_root().name == staging.DEFAULT_DATASET_ROOT


def test_stage_writes_a_validated_manifest_and_provenance(tmp_path):
    raw = _nih_release(tmp_path)
    provenance = staging.stage_dataset("nih_cxr14", root=tmp_path, check_images=True)

    manifest = tmp_path / "nih_cxr14_manifest.csv"
    assert manifest.is_file()
    assert provenance["validation"]["clean"] is True
    assert provenance["validation"]["n_missing_images"] == 0
    assert provenance["counts"]["n_cases"] == 3
    assert provenance["counts"]["by_modality"] == {"image": 3}
    assert provenance["source"]["access"] == "open"
    assert provenance["raw_root"] == str(raw)
    assert len(provenance["manifest_sha256"]) == 64

    on_disk = json.loads((tmp_path / "nih_cxr14_provenance.json").read_text())
    assert on_disk == provenance
    stanza = (tmp_path / "nih_cxr14_SOURCE.txt").read_text()
    assert "access:    open" in stanza
    assert provenance["manifest_sha256"] in stanza
    assert "Raw data is not committed" in stanza


def test_per_case_meta_survives_staging(tmp_path):
    _nih_release(tmp_path)
    staging.stage_dataset("nih_cxr14", root=tmp_path)

    cases = load_cases(tmp_path / "nih_cxr14_manifest.csv")
    # the imaging meta (view, findings) is what the natural-cue arms read; it has to reach disk
    assert all(case.meta["view"] == "PA" for case in cases)
    assert cases[0].meta["findings"] == ["Cardiomegaly", "Effusion"]


def test_missing_images_fail_the_staging_when_checked(tmp_path):
    _nih_release(tmp_path, with_images=False)
    with pytest.raises(ValueError, match="did not validate"):
        staging.stage_dataset("nih_cxr14", root=tmp_path, check_images=True)


def test_missing_raw_release_names_the_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="nihcc.app.box.com"):
        staging.stage_dataset("nih_cxr14", root=tmp_path)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        staging.stage_dataset("not_a_dataset", raw_root=".")


def test_limit_is_recorded_as_a_subset(tmp_path):
    _nih_release(tmp_path, n=5)
    provenance = staging.stage_dataset("nih_cxr14", root=tmp_path, limit=2)
    assert provenance["counts"]["n_cases"] == 2
    assert provenance["limit"] == 2
    assert "a subset, not the full release" in (tmp_path / "nih_cxr14_SOURCE.txt").read_text()


def test_cli_stage_reports_where_everything_landed(tmp_path, monkeypatch, capsys):
    _nih_release(tmp_path)
    monkeypatch.setenv(staging.DATASET_ROOT_ENV, str(tmp_path))
    rc = cli.main(["datasets", "stage", "nih_cxr14", "--check-images"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "staged nih_cxr14: 3 cases" in out
    assert "sha256:" in out
    assert "access: open" in out


def test_cli_stage_missing_data_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(staging.DATASET_ROOT_ENV, str(tmp_path))
    rc = cli.main(["datasets", "stage", "nih_cxr14"])
    assert rc != 0
    assert "error" in capsys.readouterr().err.lower()
