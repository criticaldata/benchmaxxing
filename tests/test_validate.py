"""Tests for benchmaxxing.validate: manifest validation reports and content checksums.

Everything here runs offline against synthetic manifests written to ``tmp_path``; no backend, no
network, no real data.
"""

from __future__ import annotations

import hashlib

import pytest

from benchmaxxing.validate import (
    Problem,
    ValidationReport,
    manifest_checksum,
    validate_manifest,
)

# A clean two-row imaging manifest: unique ids, all required fields (including ground-truth label)
# populated, images referenced relatively.
_CLEAN_IMAGING = (
    "case_id,patient_id,image_ref,report,label\n"
    "c1,p1,images/a.png,No acute findings.,No Finding\n"
    "c2,p2,images/b.png,Small left effusion.,Effusion\n"
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_manifest_is_clean(tmp_path):
    manifest = _write(tmp_path / "clean.csv", _CLEAN_IMAGING)

    report = validate_manifest(manifest)

    assert isinstance(report, ValidationReport)
    assert report.is_clean
    assert report.problems == []
    assert report.n_cases == 2
    assert report.n_unique_case_ids == 2
    assert report.n_duplicate_case_ids == 0
    assert report.n_missing_fields == 0
    # images are not checked unless requested
    assert report.n_images_checked == 0


def test_valid_text_manifest_is_clean(tmp_path):
    manifest = _write(
        tmp_path / "mcq.csv",
        "case_id,question,options,answer_index\n"
        "q1,Best next step?,Aspirin|Warfarin|Heparin,0\n"  # answer_index 0 is valid, not "missing"
        "q2,Likely diagnosis?,MI|PE|Pneumonia,2\n",
    )

    report = validate_manifest(manifest)

    assert report.is_clean
    assert report.n_cases == 2


def test_duplicate_case_id_and_missing_field_are_reported(tmp_path):
    # c1 is duplicated; the second imaging row (c3) has no label (a required ground-truth field
    # that load_cases tolerates as blank), so both problems surface in a single pass.
    manifest = _write(
        tmp_path / "bad.csv",
        "case_id,patient_id,image_ref,report,label\n"
        "c1,p1,images/a.png,r,No Finding\n"
        "c1,p2,images/b.png,r,Effusion\n"
        "c3,p3,images/c.png,r,\n",
    )

    report = validate_manifest(manifest)

    assert not report.is_clean
    # all three rows still load, so the counts reflect the whole table
    assert report.n_cases == 3
    assert report.n_unique_case_ids == 2
    assert report.n_duplicate_case_ids == 1
    assert report.n_missing_fields == 1

    dupes = report.problems_of("duplicate_case_id")
    missing = report.problems_of("missing_field")
    assert len(dupes) == 1
    assert dupes[0].case_id == "c1"
    assert len(missing) == 1
    assert missing[0].case_id == "c3"
    assert "label" in missing[0].detail
    # both distinct kinds are collected together
    assert {p.kind for p in report.problems} == {"duplicate_case_id", "missing_field"}


def test_malformed_manifest_collected_not_raised(tmp_path):
    # A row that cannot be parsed at all (no image_ref and no question -> undetectable modality).
    manifest = _write(
        tmp_path / "unparseable.csv",
        "case_id,patient_id,report\nc1,p1,No findings.\n",
    )

    report = validate_manifest(manifest)

    assert not report.is_clean
    load_errors = report.problems_of("load_error")
    assert len(load_errors) == 1
    assert report.n_cases == 0


def test_missing_manifest_is_collected_not_raised(tmp_path):
    report = validate_manifest(tmp_path / "does_not_exist.csv")

    assert not report.is_clean
    assert report.problems_of("load_error")
    assert report.n_cases == 0


def test_check_images_flags_missing_and_passes_present(tmp_path):
    (tmp_path / "images").mkdir()
    # only a.png exists on disk; b.png is referenced but absent
    (tmp_path / "images" / "a.png").write_bytes(b"\x89PNG\r\n")
    manifest = _write(tmp_path / "imaging.csv", _CLEAN_IMAGING)

    # default root: the manifest's own directory
    report = validate_manifest(manifest, check_images=True)

    assert report.n_images_checked == 2
    assert report.n_missing_images == 1
    missing = report.problems_of("missing_image")
    assert len(missing) == 1
    assert missing[0].case_id == "c2"
    assert "b.png" in missing[0].detail


def test_check_images_all_present_is_clean(tmp_path):
    root = tmp_path / "assets"
    (root / "images").mkdir(parents=True)
    (root / "images" / "a.png").write_bytes(b"x")
    (root / "images" / "b.png").write_bytes(b"y")
    # manifest lives elsewhere; images resolved against an explicit root
    manifest = _write(tmp_path / "imaging.csv", _CLEAN_IMAGING)

    report = validate_manifest(manifest, check_images=True, root=root)

    assert report.is_clean
    assert report.n_images_checked == 2
    assert report.n_missing_images == 0


def test_checksum_is_stable_and_matches_hashlib(tmp_path):
    manifest = _write(tmp_path / "m.csv", _CLEAN_IMAGING)

    first = manifest_checksum(manifest)
    second = manifest_checksum(manifest)

    assert first == second  # stable across calls
    assert isinstance(first, str)
    assert len(first) == 64  # sha256 hex digest
    assert first == hashlib.sha256(_CLEAN_IMAGING.encode("utf-8")).hexdigest()


def test_checksum_differs_on_content_change(tmp_path):
    a = _write(tmp_path / "a.csv", _CLEAN_IMAGING)
    b = _write(tmp_path / "b.csv", _CLEAN_IMAGING + "c3,p3,images/c.png,r,Cardiomegaly\n")

    # identical bytes hash identically regardless of file name
    a_copy = _write(tmp_path / "a_copy.csv", _CLEAN_IMAGING)
    assert manifest_checksum(a) == manifest_checksum(a_copy)
    assert manifest_checksum(a) != manifest_checksum(b)


def test_missing_checksum_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        manifest_checksum(tmp_path / "nope.csv")


def test_problem_str_is_readable():
    p = Problem(kind="missing_field", detail="required field 'label' is empty", case_id="c3")
    text = str(p)
    assert "missing_field" in text and "c3" in text
