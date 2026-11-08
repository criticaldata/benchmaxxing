"""experiments/medqa/cross_lane_reconciliation.py rebuilds its committed summary from the committed summaries."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_cross_lane_summary_reproduces(tmp_path):
    committed = json.loads((ROOT / "experiments/medqa/results/cross_lane_reconciliation_summary.json").read_text())
    subprocess.run([sys.executable, "experiments/medqa/cross_lane_reconciliation.py", "--out", str(tmp_path)],
                   cwd=ROOT, check=True, capture_output=True)
    assert json.loads((tmp_path / "cross_lane_reconciliation_summary.json").read_text()) == committed
