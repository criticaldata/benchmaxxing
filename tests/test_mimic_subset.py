"""Tests for the MIMIC-CXR subset selector: study-level nesting, expansion, and determinism.

Pure shape tests on synthetic Cases: no raw tree, no images, no network.
"""

from __future__ import annotations

import random

from benchmaxxing.schema import Case, Modality
from experiments.mimic_cxr_image.build_subset import _download_lines, _study_key, select_arms


def _case(study: int, dicom: int, label: str = "pneumothorax"):
    """One image in study ``study``. Two dicoms per study model MIMIC's frontal+lateral pair."""
    subject = 10_000 + study
    return Case(
        case_id=f"d{study}_{dicom}",
        patient_id=str(subject),
        modality=Modality.IMAGE,
        label=label,
        image_ref=f"files/p{str(subject)[:2]}/p{subject}/s{study}/d{study}_{dicom}.jpg",
        meta={"study_id": str(study)},
    )


def _pool(n_studies: int, images_per_study: int = 2, label: str = "pneumothorax"):
    return [_case(s, d, label) for s in range(n_studies) for d in range(images_per_study)]


def _study_set(cases):
    return {_study_key(c) for c in cases}


def test_smaller_arms_are_strict_subsets_of_larger_ones():
    # The whole point of one seed: blind < cascade < referee < solo must nest at study level.
    cases = _pool(50)
    sizes = {"solo": 30, "referee": 20, "cascade": 10}
    arms = select_arms(cases, seed=7, sizes=sizes)
    solo, referee, cascade = map(_study_set, (arms["solo"], arms["referee"], arms["cascade"]))
    assert cascade < referee < solo
    assert len(solo) == 30


def test_nih_match_is_sized_by_image_count_and_nests():
    # nih_match matches NIH's one-image-per-observation shape: sized by IMAGE count, not studies.
    # 2 images/study, so 5 studies would be 10 images; image-sizing must give exactly 5 images.
    cases = _pool(50, images_per_study=2)
    arms = select_arms(cases, seed=7, sizes={"cascade": 10, "nih_match": 5})
    nih = arms["nih_match"]
    assert len(nih) == 5                                  # exactly 5 images, not 5 studies (=10 images)
    assert len(_study_set(nih)) <= 3                      # 5 images span a prefix of ranked studies
    assert _study_set(nih) < _study_set(arms["cascade"])  # still nests inside the study-sized arm


def test_nih_match_is_deterministic_and_odd_image_counts_are_exact():
    # Exact n even when the walk would overshoot a multi-image study, and stable across runs.
    cases = _pool(40, images_per_study=3)
    a = select_arms(cases, seed=2, sizes={"nih_match": 7})["nih_match"]
    b = select_arms(cases, seed=2, sizes={"nih_match": 7})["nih_match"]
    assert len(a) == 7
    assert [c.case_id for c in a] == [c.case_id for c in b]


def test_all_images_of_a_selected_study_are_kept():
    # "All images per study": a picked study contributes every one of its images, not just one.
    cases = _pool(20, images_per_study=3)
    arms = select_arms(cases, seed=1, sizes={"solo": 5})
    picked = arms["solo"]
    assert len(_study_set(picked)) == 5
    assert len(picked) == 15  # 5 studies x 3 images
    for key in _study_set(picked):
        assert sum(1 for c in picked if _study_key(c) == key) == 3


def test_selection_is_invariant_to_pool_order():
    # Study ranking depends on content, not arrival order: a reshuffled pool selects the same studies.
    cases = _pool(40)
    shuffled = cases[:]
    random.Random(123).shuffle(shuffled)
    a = select_arms(cases, seed=3, sizes={"solo": 12})["solo"]
    b = select_arms(shuffled, seed=3, sizes={"solo": 12})["solo"]
    assert _study_set(a) == _study_set(b)


def test_no_finding_studies_are_dropped_by_default():
    findings = _pool(10, label="pneumothorax")
    negatives = [_case(s, d, label="no finding") for s in range(100, 110) for d in range(2)]
    arms = select_arms(findings + negatives, seed=0, sizes={"solo": 10})
    # Only the 10 finding studies are eligible, so the arm is exactly those.
    assert len(_study_set(arms["solo"])) == 10
    assert all(c.label == "pneumothorax" for c in arms["solo"])


def test_include_no_finding_keeps_negatives():
    findings = _pool(5, label="pneumothorax")
    negatives = [_case(s, d, label="no finding") for s in range(100, 105) for d in range(2)]
    arms = select_arms(findings + negatives, seed=0, sizes={"solo": 10}, require_finding=False)
    assert len(_study_set(arms["solo"])) == 10


def test_download_lines_are_unique_and_prefixed():
    cases = _pool(3, images_per_study=2)
    lines = _download_lines(cases, "https://physionet.org/files/mimic-cxr-jpg/2.1.0/")
    assert len(lines) == 6  # 3 studies x 2 images, all distinct
    assert len(set(lines)) == 6
    assert all(u.startswith("https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/") for u in lines)
