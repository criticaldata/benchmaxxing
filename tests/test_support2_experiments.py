"""Offline end-to-end smoke for the three SUPPORT2 runners (solo, cascade, referee).

No key, no network: the shared cache's ``complete`` is replaced with a deterministic stub that
follows whatever the peers put on the board and flips on the missingness cue. That is enough to
drive every aggregation path, and it makes the referee comparison assertable: the naive gate and
the targeted referee both over-fire on a patient the holdout was wrong about anyway, while the
deployable referee's private re-query keeps it clean.
"""
from __future__ import annotations

import json
import sys

import pytest

from benchmaxxing.datasets import support2
from experiments.support2 import _common, support2_cascade, support2_referee, support2_solo

HEADER = (
    "id,age,sex,dzgroup,num.co,diabetes,dementia,scoma,adls,meanbp,hrt,resp,temp,wblc,"
    "alb,bili,crea,sod,ph,bun,urine,hospdead\n"
)
SURVIVES = "{i},6{i}.4,female,COPD,1,0,0,0,0,92,88,20,37.1,,3.8,0.6,0.9,140,7.4,18,1500,0\n"
DIES = ("{i},7{i}.3,male,ARF/MOSF w/Sepsis,3,1,0,44,4,58,124,32,38.6,,"
        "2.1,1.9,2.4,131,7.28,52,300,1\n")

SURVIVES_TEXT = "Survives to hospital discharge"
DIES_TEXT = "Dies in hospital"


def _fake_complete(self, prompt, model=None):
    """Answer 'survives' alone, follow the board when there is one, flip on the missingness cue."""
    if "Senior colleagues" in prompt:
        board = prompt.split("Senior colleagues", 1)[1]
        return "B" if DIES_TEXT in board else "A"
    return "B" if "not recorded" in prompt else "A"


@pytest.fixture
def manifest(tmp_path):
    rows = [SURVIVES.format(i=i) for i in range(1, 4)] + [DIES.format(i=i) for i in range(4, 7)]
    raw = tmp_path / "support2.csv"
    raw.write_text(HEADER + "".join(rows), encoding="utf-8")
    return support2.build_manifest(raw, tmp_path / "manifest.csv")


@pytest.fixture
def offline(monkeypatch):
    """No API key anywhere, and a deterministic stand-in for every model call."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(_common.Cache, "complete", _fake_complete)


def _run(module, monkeypatch, manifest, tmp_path):
    argv = ["prog", "--manifest", str(manifest), "--out", str(tmp_path / "results"),
            "--cache", str(tmp_path / "cache.jsonl"), "--n", "6"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return tmp_path / "results"


def test_solo_runs_offline_and_scores_flips(offline, monkeypatch, manifest, tmp_path):
    out = _run(support2_solo, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_solo_summary.json").read_text())

    assert summary["n"] == 6
    assert summary["new_api_calls_this_run"] == 0
    # No key, so the noise floor is skipped rather than silently reported as zero.
    assert summary["noise_floor"] is None
    assert "skipped" in summary["noise_floor_note"]

    cues = summary["cues"]
    assert set(cues) == set(support2_solo.CUES)
    # The stub flips only when the record spells out an absent value.
    assert cues["missingness_recode"]["flip_rate"] == 1.0
    assert cues["field_order"]["flip_rate"] == 0.0
    assert all(c["n"] == 6 and c["n_skipped"] == 0 for c in cues.values())

    rows = [json.loads(line) for line in
            (out / "support2_solo.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 6
    assert all(r["clean"] == SURVIVES_TEXT for r in rows)


def test_cascade_measures_contagion(offline, monkeypatch, manifest, tmp_path):
    out = _run(support2_cascade, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_cascade_summary.json").read_text())

    assert summary["n"] == 6
    assert summary["new_api_calls_this_run"] == 0
    # The stub always answers "survives" bare, so it is right on the 3 survivors.
    assert summary["bare_accuracy"] == pytest.approx(0.5)

    wrong = summary["arms"]["wrong_seed"]
    # The 3 survivors are seeded with "dies" and follow the board: adoption with no reversion.
    assert wrong["n_eligible"] == 3
    assert wrong["adoption_among_eligible"] == 1.0
    assert wrong["contagion"] == pytest.approx(0.5)
    assert wrong["mcnemar"]["gain"] == 3 and wrong["mcnemar"]["lose"] == 0

    flip = summary["arms"]["flip_seed"]
    # The flip arm seeds the complement of the bare answer, so every patient is eligible.
    assert flip["n_eligible"] == 6
    assert summary["flip_seed_polarity"]["seed_wrong"]["n"] == 3
    assert summary["flip_seed_polarity"]["seed_right"]["n"] == 3


def test_referee_ranks_the_three_detectors(offline, monkeypatch, manifest, tmp_path):
    out = _run(support2_referee, monkeypatch, manifest, tmp_path)
    summary = json.loads((out / "support2_referee_summary.json").read_text())
    scores = summary["referees_vs_shortcut_adoption"]

    assert summary["n"] == 6
    assert summary["new_api_calls_this_run"] == 0
    assert summary["n_holdout_adopted_shortcut"] == 3
    assert summary["n_holdout_wrong_bare"] == 3

    naive = scores["naive_gate (transcript only, no key, no re-query)"]
    targeted = scores["targeted (planted key, no counterfactual)"]
    deployable = scores["deployable (peer-modal + private re-query, NO key)"]

    # All three catch every adoption; they differ entirely in what else they catch.
    assert naive["recall"] == targeted["recall"] == deployable["recall"] == 1.0
    # The colluders always agree, so a pure agreement gate flags every patient.
    assert naive["fpr"] == 1.0
    # Knowing WHICH answer was planted is not enough without a counterfactual: the targeted referee
    # also flags the patients the holdout was already wrong about on its own.
    assert targeted["fp"] == 3
    assert targeted["precision"] == 0.5
    # One private re-query, no key, and the false positives go away.
    assert deployable["fpr"] == 0.0
    assert deployable["precision"] == 1.0
    assert summary["extra_requery_calls_needed"]["deployable"] == 6
