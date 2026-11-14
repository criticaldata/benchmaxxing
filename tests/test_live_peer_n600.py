"""The live-peer arm at 600 MedQA cases replays from its committed cache with no key, its summary
counts are its rows, and its first 120 cases are the committed 120-case arm."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/medqa/results/live_peer_n600"
MANIFEST = Path(os.environ.get("BENCHMAXXING_MEDQA_MANIFEST", str(Path.home() / "Documents/DATASET/medqa/manifest_test.csv")))
VOLATILE = {"read", "new_api_calls_this_run"}


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.mark.skipif(not MANIFEST.exists(), reason="the MedQA manifest is not on this machine")
def test_the_arm_replays_keyless():
    summary, rows = OUT / "live_peer_organic_summary.json", OUT / "live_peer_organic.jsonl"
    saved = {p: p.read_bytes() for p in (summary, rows)}
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    try:
        r = subprocess.run([sys.executable, "experiments/medqa/live_peer_organic.py", "--manifest", str(MANIFEST),
                            "--n", "600", "--out", str(OUT), "--cache", str(OUT / "live_peer_organic_cache.jsonl")],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
        after, before = json.loads(summary.read_text()), json.loads(saved[summary])
        assert after["new_api_calls_this_run"] == 0
        assert {k: v for k, v in after.items() if k not in VOLATILE} == {k: v for k, v in before.items() if k not in VOLATILE}
        # rows are written in completion order, so compare them as a set
        assert sorted(rows.read_text().splitlines()) == sorted(saved[rows].decode().splitlines())
    finally:
        for p, b in saved.items():
            p.write_bytes(b)


def test_the_summary_counts_are_the_rows():
    s = json.loads((OUT / "live_peer_organic_summary.json").read_text())
    rows = _rows(OUT / "live_peer_organic.jsonl")
    wrong = [r for r in rows if r["consensus_wrong"]]
    right = [r for r in rows if r["consensus_right"]]
    assert s["n"] == len(rows) == 600
    assert s["n_organic_wrong_consensus"] == len(wrong) and s["n_organic_right_consensus"] == len(right)
    assert s["follow_rate_on_wrong_consensus"] == round(sum(r["follows_consensus"] for r in wrong) / len(wrong), 4)
    assert s["follow_rate_on_right_consensus"] == round(sum(r["follows_consensus"] for r in right) / len(right), 4)
    assert s["net_harm_cases"] == sum(r["net_harm"] for r in rows)


def test_the_first_120_cases_are_the_committed_arm():
    old = {r["case_id"]: r for r in _rows(ROOT / "experiments/medqa/results/live_peer_organic.jsonl")}
    new = {r["case_id"]: r for r in _rows(OUT / "live_peer_organic.jsonl")}
    assert len(old) == 120 and set(old) <= set(new)
    for k, r in old.items():
        for field in ("board", "peer_answers", "consensus", "follows_consensus", "net_harm"):
            assert new[k][field] == r[field], (k, field)
