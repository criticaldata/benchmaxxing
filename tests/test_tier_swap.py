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
                 ("experiments/medmcqa/results/blind_metric.jsonl",
                  "experiments/medmcqa/results/gemini-2.5-flash/blind_metric.jsonl"),
                 ("experiments/imaging/results/imaging_blind_metric.jsonl",
                  "experiments/imaging/results/gemini-2.5-flash-lite/imaging_blind_metric.jsonl")):
        ids = lambda p: {json.loads(l)["case_id"] for l in (ROOT / p).read_text().splitlines() if l.strip()}
        assert ids(a) == ids(b), (a, b)


def test_the_support2_pair_uses_the_cases_both_tiers_answer_correctly_at_baseline():
    rows = lambda p: {json.loads(l)["case_id"]: json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()}
    lite = rows("experiments/support2/results/blind_metric.jsonl")
    flash = rows("experiments/support2/results/gemini-2.5-flash/blind_metric.jsonl")
    shared = sorted(set(lite) & set(flash))
    s = json.loads((ROOT / "experiments/blind_metric/results/tier_swap_summary.json").read_text())
    cell = s["tier_within_modality"]["support2"]
    assert cell["n_paired"] == len(shared)
    assert cell["blind_flash_vs_lite"]["gain"] == sum(1 for i in shared if flash[i]["blind_is_decoy"] and not lite[i]["blind_is_decoy"])
    assert cell["blind_flash_vs_lite"]["lose"] == sum(1 for i in shared if lite[i]["blind_is_decoy"] and not flash[i]["blind_is_decoy"])
    # two options, so the rewarded option is the same letter in both tiers on a shared case
    assert all(lite[i]["decoy_letter"] == flash[i]["decoy_letter"] for i in shared)

