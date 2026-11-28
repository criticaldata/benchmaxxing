"""The NIH 500-film results: the patient-clustered summary rebuilds from the committed rows."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/imaging/results_n500"


def test_patient_clustered_summary_reproduces():
    committed = json.loads((OUT / "patient_clustered.json").read_text())
    fresh = json.loads(subprocess.run([sys.executable, str(ROOT / "experiments/imaging/scale_patient_clustered.py")],
                                      cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    assert fresh == committed
    assert committed["n_films"] == 500 and committed["n_patients"] == 132


def test_the_500_rows_cover_the_scaled_manifest():
    ids = {r["case_id"] for r in csv.DictReader(open(ROOT / "experiments/imaging/results/nih_manifest_scaled.csv"))}
    for f in ("imaging_cascade_scale.jsonl", "imaging_blind_metric.jsonl", "matched_baseline_reason.jsonl"):
        rows = {json.loads(line)["case_id"] for line in (OUT / f).read_text().splitlines() if line.strip()}
        assert rows == ids, f


QWEN = "Qwen/Qwen2.5-VL-72B-Instruct"
QOUT = OUT / "Qwen_Qwen2.5-VL-72B-Instruct"


def test_qwen_patient_clustered_summary_reproduces():
    p = QOUT / "patient_clustered.json"
    committed = p.read_text()
    try:
        fresh = json.loads(subprocess.run([sys.executable, str(ROOT / "experiments/imaging/scale_patient_clustered.py"),
                                           "--model", QWEN], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    finally:
        p.write_text(committed)
    assert fresh == json.loads(committed)
    assert fresh["n_films"] == 500 and fresh["n_patients"] == 132 and fresh["model"] == QWEN


def test_qwen_500_rows_cover_the_scaled_manifest():
    ids = {r["case_id"] for r in csv.DictReader(open(ROOT / "experiments/imaging/results/nih_manifest_scaled.csv"))}
    for f in ("imaging_cascade_scale.jsonl", "imaging_blind_metric.jsonl", "matched_baseline_reason.jsonl"):
        rows = {json.loads(line)["case_id"] for line in (QOUT / f).read_text().splitlines() if line.strip()}
        assert rows == ids, f
