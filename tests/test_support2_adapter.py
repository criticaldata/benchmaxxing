"""Tests for the SUPPORT2 tabular adapter: row parsing, leakage exclusion, and manifest I/O."""

from __future__ import annotations

import pytest

from benchmaxxing.data import load_cases
from benchmaxxing.datasets import registry, support2
from benchmaxxing.schema import Modality

HEADER = (
    "id,age,sex,race,dzgroup,dzclass,num.co,ca,diabetes,dementia,scoma,adls,hday,"
    "meanbp,hrt,resp,temp,wblc,pafi,alb,bili,crea,sod,ph,glucose,bun,urine,"
    "surv2m,surv6m,sps,aps,dnr,slos,d.time,death,hospdead\n"
)
ROW_DIES = (
    "1,72.3,male,white,ARF/MOSF w/Sepsis,ARF/MOSF,2,no,1,0,0,3,1,"
    "78,110,28,38.5,14.2,180,2.4,1.1,1.2,138,7.31,142,45,800,"
    "0.21,0.11,32.5,71,no,9,12,1,1\n"
)
ROW_SURVIVES = (
    "2,55,female,black,COPD,COPD/CHF/Cirrhosis,1,no,0,0,0,0,1,"
    "92,88,20,37,,180,3.8,0.6,0.9,140,7.4,,18,1500,"
    "0.88,0.79,15.0,33,no,5,600,0,0\n"
)
ROW_NO_OUTCOME = ROW_SURVIVES.replace(",0,0\n", ",,\n").replace("2,55,", "3,55,")


def _write_csv(path, *rows):
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def test_registered():
    assert "support2" in registry.names()
    assert registry.get("support2") is support2


def test_build_manifest_parses_rows(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_DIES, ROW_SURVIVES)
    out = support2.build_manifest(raw, tmp_path / "manifest.csv")
    cases = load_cases(out)

    assert [c.case_id for c in cases] == ["support2-1", "support2-2"]
    assert all(c.modality is Modality.TEXT for c in cases)
    # option 0 is always the good outcome, so answer_index is the raw hospdead flag.
    assert [c.answer_index for c in cases] == [1, 0]
    assert cases[0].options == support2.TARGETS["hospdead"][1]
    assert cases[0].label == "Dies in hospital"


def test_vignette_renders_features_and_hides_leakage(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_DIES)
    cases = load_cases(support2.build_manifest(raw, tmp_path / "m.csv"))
    question = cases[0].question

    assert "Patient record:" in question
    assert "- age: 72.3 years" in question
    assert "- mean arterial pressure: 78 mmHg" in question
    assert "- diabetes: yes" in question          # 0/1 flags render as no/yes
    assert "- dementia: no" in question
    assert question.rstrip().endswith("Will this patient survive to hospital discharge?")
    # The SUPPORT model's own survival estimates would hand the model the answer.
    for leaked in ("surv2m", "surv6m", "0.21", "0.11", "32.5", "APACHE"):
        assert leaked not in question


def test_meta_carries_the_record_for_the_cues(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_SURVIVES)
    cases = load_cases(support2.build_manifest(raw, tmp_path / "m.csv"))
    meta = cases[0].meta

    keys = [f["key"] for f in meta["fields"]]
    assert "meanbp" in keys and "temp" in keys
    # wblc and glucose are blank in this row, so they belong to 'absent', not 'fields'.
    assert "wblc" not in keys and "glucose" not in keys
    assert {a["key"] for a in meta["absent"]} == {"wblc", "glucose"}
    assert meta["stem"] == support2.TARGETS["hospdead"][0]
    assert meta["target"] == "hospdead"


def test_rows_without_an_outcome_are_skipped(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_DIES, ROW_NO_OUTCOME)
    cases = load_cases(support2.build_manifest(raw, tmp_path / "m.csv"))
    assert [c.case_id for c in cases] == ["support2-1"]


def test_death_target_and_limit(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_DIES, ROW_SURVIVES)
    out = support2.build_manifest(raw, tmp_path / "m.csv", limit=1, target="death")
    cases = load_cases(out)

    assert len(cases) == 1
    assert cases[0].options == support2.TARGETS["death"][1]
    assert cases[0].answer_index == 1


def test_directory_raw_root(tmp_path):
    _write_csv(tmp_path / "support2.csv", ROW_DIES)
    cases = load_cases(support2.build_manifest(tmp_path, tmp_path / "m.csv"))
    assert len(cases) == 1


def test_bad_inputs(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_DIES)
    with pytest.raises(ValueError, match="unknown target"):
        support2.build_manifest(raw, tmp_path / "m.csv", target="slos")
    with pytest.raises(FileNotFoundError):
        support2.build_manifest(tmp_path / "missing.csv", tmp_path / "m.csv")

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing the target column"):
        support2.build_manifest(wrong, tmp_path / "m.csv")


def test_unnamed_index_column_is_realigned(tmp_path):
    """The distributed support2.csv writes a row index its header does not name. Read naively the
    columns shift by one and 'hospdead' becomes 'sex', so the ground truth would be the patient's
    sex. The rows below are the real file's shape: 1 field more than the header."""
    shifted = tmp_path / "support2.csv"
    shifted.write_text(
        HEADER.replace("id,", "", 1) + "1," + ROW_DIES.split(",", 1)[1] +
        "2," + ROW_SURVIVES.split(",", 1)[1],
        encoding="utf-8",
    )
    cases = load_cases(support2.build_manifest(shifted, tmp_path / "m.csv"))

    assert [c.case_id for c in cases] == ["support2-1", "support2-2"]
    assert [c.answer_index for c in cases] == [1, 0]
    assert "- sex: male" in cases[0].question
    assert "- age: 72.3 years" in cases[0].question


def test_manifest_round_trips_meta(tmp_path):
    raw = _write_csv(tmp_path / "support2.csv", ROW_DIES)
    out = support2.build_manifest(raw, tmp_path / "m.csv")
    reloaded = load_cases(out)
    assert reloaded[0].meta["fields"][0]["key"] == "age"
    assert reloaded[0].question.count("\n") > 5   # the vignette survived CSV quoting
