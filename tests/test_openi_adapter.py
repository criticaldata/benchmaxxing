"""Tests for the Open-i / Indiana University adapter (benchmaxxing.datasets.openi_cxr).

The fixture XML mirrors the shape of a real ecgen-radiology report (MeSH major terms, the
labelled AbstractText sections, one parentImage per view). The manifest CSV round-trip keeps
meta, so the label policy is asserted both on the parsed Cases and through the manifest.
"""

from __future__ import annotations

import pytest

from benchmaxxing import runner
from benchmaxxing.data import load_cases
from benchmaxxing.datasets import openi_cxr
from benchmaxxing.schema import Modality

TWO_VIEW_REPORT = """<eCitation>
  <MeSH>
    <major>Cardiomegaly/mild</major>
    <major>Pulmonary Congestion/mild</major>
    <minor>Lung/hyperdistention</minor>
    <automatic>heart</automatic>
  </MeSH>
  <MedlineCitation><Article><Abstract>
    <AbstractText Label="COMPARISON">None.</AbstractText>
    <AbstractText Label="INDICATION">Shortness of breath.</AbstractText>
    <AbstractText Label="FINDINGS">Mild cardiomegaly with pulmonary vascular congestion.</AbstractText>
    <AbstractText Label="IMPRESSION">Mild congestive changes.</AbstractText>
  </Abstract></Article></MedlineCitation>
  <parentImage id="CXR7_IM-2263-1001">
    <caption>PA</caption>
  </parentImage>
  <parentImage id="CXR7_IM-2263-2001">
    <caption>Lateral</caption>
  </parentImage>
</eCitation>
"""

NORMAL_REPORT = """<eCitation>
  <MeSH><major>normal</major></MeSH>
  <MedlineCitation><Article><Abstract>
    <AbstractText Label="FINDINGS">The lungs are clear.</AbstractText>
    <AbstractText Label="IMPRESSION">Normal chest.</AbstractText>
  </Abstract></Article></MedlineCitation>
  <parentImage id="CXR9_IM-2400-1001"><caption>PA</caption></parentImage>
</eCitation>
"""

IMAGELESS_REPORT = """<eCitation>
  <MeSH><major>normal</major></MeSH>
  <MedlineCitation><Article><Abstract>
    <AbstractText Label="IMPRESSION">Normal chest.</AbstractText>
  </Abstract></Article></MedlineCitation>
</eCitation>
"""


def _write_reports(directory, reports):
    reports_dir = directory / "ecgen-radiology"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, text in reports.items():
        (reports_dir / name).write_text(text, encoding="utf-8")
    return reports_dir


def test_spec_name_stable():
    assert openi_cxr.SPEC.name == "openi_cxr"
    assert openi_cxr.SPEC.modality is Modality.IMAGE


def test_two_views_share_the_patient_key(tmp_path):
    reports_dir = _write_reports(tmp_path, {"7.xml": TWO_VIEW_REPORT})
    cases = openi_cxr._cases_from_report(reports_dir / "7.xml")

    assert [c.case_id for c in cases] == ["CXR7_IM-2263-1001", "CXR7_IM-2263-2001"]
    # both views of the study carry the report id, which is what the same-patient swap needs
    assert {c.patient_id for c in cases} == {"7"}
    assert [c.image_ref for c in cases] == ["CXR7_IM-2263-1001.png", "CXR7_IM-2263-2001.png"]
    assert [c.meta["view"] for c in cases] == ["PA", "Lateral"]


def test_label_policy_drops_severity_and_joins_findings(tmp_path):
    reports_dir = _write_reports(tmp_path, {"7.xml": TWO_VIEW_REPORT})
    case = openi_cxr._cases_from_report(reports_dir / "7.xml")[0]

    assert case.label == "cardiomegaly|pulmonary congestion"
    # the raw terms survive in meta so a different policy needs no re-parse
    assert case.meta["mesh_major"] == ["Cardiomegaly/mild", "Pulmonary Congestion/mild"]
    assert case.meta["mesh_automatic"] == ["heart"]
    assert case.meta["indication"] == "Shortness of breath."


def test_anatomy_led_term_collapses_to_anatomy(tmp_path):
    # Documented wart: 'Lung/hypoinflation' has no finding head, so the label is 'lung' and
    # meta carries the verbatim term. Pinned so a future policy change is a deliberate one.
    report = NORMAL_REPORT.replace("<major>normal</major>", "<major>Lung/hypoinflation</major>")
    reports_dir = _write_reports(tmp_path, {"9.xml": report})
    case = openi_cxr._cases_from_report(reports_dir / "9.xml")[0]
    assert case.label == "lung"
    assert case.meta["mesh_major"] == ["Lung/hypoinflation"]


def test_report_is_findings_plus_impression(tmp_path):
    reports_dir = _write_reports(tmp_path, {"9.xml": NORMAL_REPORT})
    case = openi_cxr._cases_from_report(reports_dir / "9.xml")[0]
    assert case.report == "The lungs are clear.\nNormal chest."
    # the healthy MeSH class maps to the shared negative sentinel, not the literal "normal"
    assert case.label == "no finding"


def test_healthy_class_is_filtered_out_of_finding_runs(tmp_path):
    # regression: a MeSH `normal` study must be dropped by the same filter every imaging experiment
    # uses (runner._positive_findings), or the negatives get scored as pathology. Assert through
    # that filter, not a hardcoded sentinel tuple: emptying _NEGATIVE_LABELS -- the exact regression
    # -- then leaves a positive finding here and fails this test.
    reports_dir = _write_reports(tmp_path, {"9.xml": NORMAL_REPORT})
    case = openi_cxr._cases_from_report(reports_dir / "9.xml")[0]
    assert runner._positive_findings(case) == []
    assert case.meta["mesh_major"] == ["normal"]  # raw MeSH term preserved for a different policy


def test_admin_nonfinding_labels_are_filtered_out_of_finding_runs(tmp_path):
    # "No Indexing" (study never findings-indexed) and the image-quality flag are admin non-findings.
    # Without them in _NEGATIVE_LABELS the finding lane asks "Does this X-ray show no indexing?".
    for major in ("No Indexing", "technical quality of image unsatisfactory"):
        report = NORMAL_REPORT.replace("<major>normal</major>", f"<major>{major}</major>")
        reports_dir = _write_reports(tmp_path, {"9.xml": report})
        case = openi_cxr._cases_from_report(reports_dir / "9.xml")[0]
        assert runner._positive_findings(case) == [], major


def test_build_manifest_round_trip(tmp_path):
    _write_reports(tmp_path, {"7.xml": TWO_VIEW_REPORT, "9.xml": NORMAL_REPORT})
    out = tmp_path / "manifest.csv"

    assert openi_cxr.build_manifest(tmp_path, out) == out
    cases = load_cases(out)
    assert len(cases) == 3
    assert all(c.modality is Modality.IMAGE for c in cases)
    assert [c.patient_id for c in cases] == ["7", "7", "9"]
    assert cases[0].label == "cardiomegaly|pulmonary congestion"
    assert cases[-1].report.endswith("Normal chest.")


def test_raw_root_can_be_the_reports_dir(tmp_path):
    reports_dir = _write_reports(tmp_path, {"9.xml": NORMAL_REPORT})
    out = tmp_path / "manifest.csv"
    openi_cxr.build_manifest(reports_dir, out)
    assert len(load_cases(out)) == 1


def test_report_without_images_is_skipped(tmp_path):
    _write_reports(tmp_path, {"9.xml": NORMAL_REPORT, "11.xml": IMAGELESS_REPORT})
    out = tmp_path / "manifest.csv"
    openi_cxr.build_manifest(tmp_path, out)
    cases = load_cases(out)
    assert [c.case_id for c in cases] == ["CXR9_IM-2400-1001"]


def test_limit_cuts_at_a_study_boundary(tmp_path):
    # limit=1 still emits both views of study 7: a study's views are never split.
    _write_reports(tmp_path, {"7.xml": TWO_VIEW_REPORT, "9.xml": NORMAL_REPORT})
    out = tmp_path / "manifest.csv"
    openi_cxr.build_manifest(tmp_path, out, limit=1)
    cases = load_cases(out)
    assert len(cases) == 2
    assert {c.patient_id for c in cases} == {"7"}


def test_missing_reports_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="ecgen-radiology"):
        openi_cxr.build_manifest(tmp_path, tmp_path / "manifest.csv")
