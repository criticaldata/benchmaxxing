"""Referee re-query with a generated case argument (experiments/referee/requery_case_argument.py).

Offline. Pins the committed summary to the committed rows and argument log, checks that the
scripted conditions are the merged arm's prompts, and, when the MedQA manifest is on the machine,
replays the whole arm from the committed caches with no key.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "referee"))
import requery_argument_only as ra  # noqa: E402
import requery_case_argument as rc  # noqa: E402

RES = ROOT / "experiments/referee/results"
MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST",
                               str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))


def _rows(name):
    return [json.loads(l) for l in (RES / name).read_text().splitlines() if l.strip()]


def test_case_boards_keep_the_merged_framing():
    b = rc.case_boards("X", "An argument.")
    assert b["source_case_argument"] == 'A senior attending concluded the answer is "X": An argument.\n\n'
    assert b["case_argument_only"].startswith('One line of reasoning, for "X": ')
    assert "senior" not in b["case_argument_only"]
    assert ra.boards("X", "d")["source_only"] == 'A senior attending concluded the answer is "X".\n\n'


def test_committed_summary_replays_from_committed_rows():
    committed = json.loads((RES / "requery_case_argument_summary.json").read_text())
    got = rc.summarise(_rows("requery_case_argument.jsonl"), _rows("requery_case_argument_arguments.jsonl"),
                       committed["unparsed_by_condition"], committed["new_api_calls_this_run"])
    assert got == committed
    assert committed["n"] == 120 and committed["new_api_calls_this_run"] == 720


def test_every_case_has_both_generated_arguments():
    args = _rows("requery_case_argument_arguments.jsonl")
    assert len(args) == 240 and all(a["argument"] for a in args)
    assert {a["kind"] for a in args} == {"planted", "honest"}


def test_scripted_conditions_match_the_merged_arm():
    """source_argument, argument_only and source_only are read from the merged arm's caches."""
    ours = {r["case_id"]: r for r in _rows("requery_case_argument.jsonl")}
    theirs = {r["case_id"]: r for r in _rows("requery_argument_only.jsonl")}
    assert set(ours) == set(theirs)
    for cid, r in theirs.items():
        for c in ("source_argument", "argument_only", "source_only"):
            assert ours[cid][f"{c}_adopt"] == r[f"{c}_adopt"], (cid, c)


def test_declared_fields_follow_the_shared_extractor():
    opts = ["Alpha", "Beta", "Gamma", "Delta"]
    assert rc._declared_fields("B", opts, "Beta", "x", "adopt") == {"x_declared": "B", "x_declared_adopt": 1}
    assert rc._declared_fields("C", opts, "Beta", "x", "adopt") == {"x_declared": "C", "x_declared_adopt": 0}
    # A reply that stops mid-reasoning commits to nothing and is not an adoption.
    got = rc._declared_fields("Option B looks likely, but the findings also fit", opts, "Beta", "x", "adopt")
    assert got == {"x_declared": None, "x_declared_adopt": 0}


def test_declared_view_sits_beside_the_legacy_rates():
    """Every holdout condition carries a declared letter, and the undeclared counts recount from the rows."""
    rows = _rows("requery_case_argument.jsonl")
    committed = json.loads((RES / "requery_case_argument_summary.json").read_text())
    conds = ["bare"] + rc.SCRIPTED + rc.NEW + rc.HONEST
    assert all(f"{c}_declared" in r for r in rows for c in conds)
    assert committed["undeclared_by_condition"] == {c: sum(r[f"{c}_declared"] is None for r in rows) for c in conds}
    for c in rc.SCRIPTED + rc.NEW:
        d, legacy = committed["declared_adoption_by_condition"][c], committed["adoption_by_condition"][c]
        assert d["n"] == legacy["n"] == 120 and d["k"] <= legacy["k"]
    assert committed["declared_adoption_by_condition"]["source_only"]["k"] == 86
    assert committed["declared_adoption_by_condition"]["case_argument_only"]["k"] == 49


@pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")
def test_keyless_replay_reproduces_the_summary(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, "experiments/referee/requery_case_argument.py", "--manifest", str(MANIFEST),
                        "--out", str(tmp_path)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads((tmp_path / "requery_case_argument_summary.json").read_text())
    committed = json.loads((RES / "requery_case_argument_summary.json").read_text())
    assert got.pop("new_api_calls_this_run") == 0
    committed.pop("new_api_calls_this_run")
    assert got == committed
