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
