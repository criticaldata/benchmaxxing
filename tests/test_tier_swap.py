"""experiments/blind_metric/tier_swap_summary.py rebuilds its committed summary from the committed rows."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tier_swap_summary_reproduces():
    committed = json.loads((ROOT / "experiments/blind_metric/results/tier_swap_summary.json").read_text())
    out = subprocess.run([sys.executable, str(ROOT / "experiments/blind_metric/tier_swap_summary.py")],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == committed


def test_swapped_arms_cover_the_committed_cases():
    for a, b in (("experiments/blind_metric/results/blind_metric.jsonl",
                  "experiments/blind_metric/results/gemini-2.5-flash/blind_metric.jsonl"),
                 ("experiments/imaging/results/imaging_blind_metric.jsonl",
                  "experiments/imaging/results/gemini-2.5-flash-lite/imaging_blind_metric.jsonl")):
        ids = lambda p: {json.loads(l)["case_id"] for l in (ROOT / p).read_text().splitlines() if l.strip()}
        assert ids(a) == ids(b), (a, b)
